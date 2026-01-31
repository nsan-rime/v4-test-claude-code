import argparse
import datasets
import json
import os
import torch
import wandb

import numpy as np
import torch.distributed as dist

from pathlib import Path
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM
from transformers.trainer_utils import set_seed

# Make all imports below relative to root (where .rv-root is)
import rootutils
root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False, indicator=[".rv-root"])

# Model classes

import torch
import torch.nn as nn
import torchtune
from torchtune.models import llama3_2

def qwen2_1500M(device="cuda"):
    # https://github.com/linkedin/Liger-Kernel?tab=readme-ov-file#supercharge-your-model-with-liger-kernel
    # Modify AutoModelForCausalLM classes with fused kernels
    from liger_kernel.transformers import apply_liger_kernel_to_qwen2
    apply_liger_kernel_to_qwen2()

    # Qwen 2.5 1.5B Instruct with extra 2048+10 tokens added to vocab
    return AutoModelForCausalLM.from_pretrained(
        "/workspace/tmp/qwen-1.5b-instruct-expanded-2048",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device
    )

def llama3_2_100M() -> torchtune.modules.transformer.TransformerDecoder:
    return llama3_2.llama3_2(
        vocab_size=128_256,
        num_layers=4,
        num_heads=8,
        num_kv_heads=2,
        embed_dim=1024,
        max_seq_len=2048,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
    )

def _prepare_transformer(model):
    embed_dim = model.tok_embeddings.embedding_dim
    # No embedding lookup (pass in pre looked up embeddings)
    model.tok_embeddings = nn.Identity()
    # No lm_head
    model.output = nn.Identity()
    return model, embed_dim

# Code based on https://github.com/SesameAILabs/csm/blob/main/models.py
class CSMDepthDecoder(nn.Module):
    def __init__(
        self,
        csm_audio_vocab_size=2051,  # For some reason CSM vocab size is 2048+3
        csm_audio_num_codebooks=32,
        csm_backbone_dim=2048,
        # Rime depth decoder
        qwen_backbone_dim=1536,
        rime_num_codebooks=12
    ):
        super().__init__()

        self.csm_audio_vocab_size = csm_audio_vocab_size
        self.rime_num_codebooks = rime_num_codebooks

        # Just load all relevant weights in original shapes for now (can prune shapes later)
        self.decoder, decoder_dim = _prepare_transformer(llama3_2_100M())
        
        self.audio_embeddings = nn.Embedding(csm_audio_vocab_size * csm_audio_num_codebooks, csm_backbone_dim)
        
        self.projection = nn.Linear(csm_backbone_dim, decoder_dim, bias=False)
        self.codebook0_head = nn.Linear(csm_backbone_dim, csm_audio_vocab_size, bias=False)
        self.audio_head = nn.Parameter(torch.empty(csm_audio_num_codebooks - 1, decoder_dim, 2051))

        # Load relevant pre-trained weights
        cp = torch.load("/workspace/tmp/v4-crash-course/csm-depth-decoder-weights.pt", map_location='cpu')
        self.load_state_dict(cp)

        # Random init projection from Qwen hidden dim to CSM embedding dim
        self.qwenH_to_csmH = nn.Linear(qwen_backbone_dim, csm_backbone_dim, bias=False)

    def set_csm_trainable(self, trainable: bool):
        
        for module in [ self.decoder, self.audio_embeddings, self.projection, self.codebook0_head ]:
            for p in module.parameters():
                p.requires_grad = trainable

        # Parameter: audio_head
        self.audio_head.requires_grad = trainable
    
    def forward(self, backbone_hidden_states, mimi_targets):
        
        # Case in case backbone_hidden_states in bfloat16
        backbone_hidden_states = backbone_hidden_states.to(
            dtype=self.qwenH_to_csmH.weight.dtype # 
        )

        # Up-project Qwen to match CSM hidden states [B, 1, 1536] -> [B, 1, 2048]
        csmH = self.qwenH_to_csmH(backbone_hidden_states)

        # Input embeds: [B, C, 2048] from [B, 1, 2048] prefix onto [B, C-1, 2048] (all but last codebook)
        input_embeds = torch.cat(
            [ csmH ] + 
            [ self.audio_embeddings(mimi_targets[:, i-1] + ((i-1) * self.csm_audio_vocab_size)).unsqueeze(1) for i in range(1, self.rime_num_codebooks) ],
            dim=1
        )

        # Following CSM, down-project input_embeds [B, C, 2048] -> [B, C, 1024] right before input to decoder
        decoder_h = self.decoder(self.projection(input_embeds))

        # Codebook 0 targets are predicted directly from hidden state
        # Codebook 1-N targets are predicted from Llama hidden state
        logits_stacked = torch.stack(
            [ self.codebook0_head(csmH[:,0,:]) ] +
            [ torch.mm(decoder_h[:, i, :], self.audio_head[i - 1]) for i in range(1, self.rime_num_codebooks) ],
             dim=1
        )

        loss = torch.nn.functional.cross_entropy(
            logits_stacked.reshape(-1, self.csm_audio_vocab_size), # (B*C, V)
            mimi_targets.reshape(-1),                              # (B*C,)
            reduction="mean"
        )

        return loss

class DualDecoder(torch.nn.Module):

    def __init__(self, device="cuda"):
        super().__init__()
        self.device=device
        self.backbone = qwen2_1500M(device=device)
        self.decoder  = CSMDepthDecoder().to(device)

    def forward(self, batch):
        # Backbone forward
        backbone_batch = move_batch_to_device(batch['backbone_inputs'])
        backbone_outputs = self.backbone(**backbone_batch, output_hidden_states=True)
        backbone_loss = backbone_outputs[0]

        # Subset frames from hidden states (one long packed sequence)
        backbone_hidden_state = backbone_outputs.hidden_states[-1][:, batch['indices_for_backbone_output'], :].transpose(0, 1) # [B, 1, H]

        # Decoder forward 
        decoder_loss = self.decoder(
            backbone_hidden_state,
            batch['mimi_targets'].to(self.device)
        )

        # Should this be weighted?
        total_loss = backbone_loss + decoder_loss

        # For logging 
        loss_dict = {
            "loss/total" : total_loss.item(),
            "loss/backbone" : backbone_loss.item(),
            "loss/decoder" : decoder_loss.item()
        }

        return total_loss, loss_dict

# Data collator

def collate_for_dual_decoder(data):
    assert len(data) == 1, "Expecting packed sequence of batch size 1"

    packed_example = data[0]

    # Concat all seqs into single long seq (<= 8192 in len)
    input_ids = np.concatenate([ i for i in packed_example['input_ids'] ]).astype(np.int64)
    # Set all non-speech and non-EndOfSpeech tokens to -100
    labels = np.where((input_ids == 151_667) | (input_ids >= 151_675), input_ids, -100)
    # Position_ids for packed causal mask and RoPE
    position_ids = np.concatenate([ np.array(range(len(i))) for i in packed_example['input_ids'] ]).astype(np.int64)

    backbone_inputs = {
        "input_ids" : torch.LongTensor(input_ids).unsqueeze(0),
        "labels" : torch.LongTensor(labels).unsqueeze(0),
        "position_ids" : torch.LongTensor(position_ids).unsqueeze(0)
    }

    mimi_seq_lengths = [ d.shape[1] for d in packed_example['mimi_24khz_tokens'] ]
    mimi_frames_to_sample = [ np.random.randint(0, s, min(16, s)) for s in mimi_seq_lengths ]

    first_speech_token_indices = np.where(input_ids==151_666)[0] + 1

    indices_for_backbone_output = np.concatenate([ indices + offset for (offset, indices) in zip(first_speech_token_indices, mimi_frames_to_sample) ]).astype(np.int64)

    mimi_targets = torch.cat([
        torch.LongTensor(seq_all_frames[:, indices_to_sample].T)
        for (seq_all_frames, indices_to_sample) in
        zip(packed_example['mimi_24khz_tokens'], mimi_frames_to_sample)
    ])

    return {
        "_ids" : [ str(id) for id in data[0]['_ids'] ], # Return IDs for debugging/logging 
        "backbone_inputs" : backbone_inputs,
        "indices_for_backbone_output" : indices_for_backbone_output,
        "mimi_targets" : mimi_targets
    }

# Helpers

def save_checkpoint(model, global_step, checkpoint_dir, metadata=None):
    write_path = f"{checkpoint_dir}/checkpoint_{global_step}"
    Path(write_path).mkdir(parents=True, exist_ok=True)
    print(f"Saving model to {write_path}")
    torch.save(model.state_dict(), f"{write_path}/dual-decoder.pt")

def move_batch_to_device(batch, device="cuda"):
    # Move keys that don't start with '_' (e.g. '_ids')
    return { k:v.to(device) for (k,v) in batch.items() if not k.startswith('_') }

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('run_name')
    parser.add_argument('train_dataset')
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--csm-freeze-steps', type=int, default=500)
    parser.add_argument('--checkpoint-interval', type=int, default=4000)
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--grad-acc-steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-4)
    args = parser.parse_args()

    set_seed(0)

    ds = datasets.Dataset.load_from_disk(args.train_dataset).with_format("numpy")

    torch.set_float32_matmul_precision('high')

    GLOBAL_RANK = int(os.environ.get('RANK', 0))
    GLOBAL_WORLD_SIZE =int(os.environ.get('WORLD_SIZE', 1))
    LOCAL_RANK = int(os.environ.get('LOCAL_RANK', 0))
    device = f"cuda:{LOCAL_RANK}"
    torch.cuda.set_device(device)

    BATCH_SIZE=1 # Batch size always 1 with packed sequences
    EFFECTIVE_BATCH_SIZE = BATCH_SIZE * args.grad_acc_steps * GLOBAL_WORLD_SIZE
    NUM_UPDATE_STEPS = args.epochs * (len(ds) // EFFECTIVE_BATCH_SIZE)

    CHECKPOINT_DIR = f"/workspace/outputs/{args.run_name}"

    if GLOBAL_WORLD_SIZE > 1:
        dist.init_process_group(backend='nccl', init_method='env://')

    model = DualDecoder(device=device)
    model.decoder.set_csm_trainable(False)

    if GLOBAL_WORLD_SIZE > 1:
        model = DDP(model, device_ids=[ LOCAL_RANK ], find_unused_parameters=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=[0.9, 0.999],
        eps=1e-8,
        weight_decay=0.0,
        fused=True
    )

    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=NUM_UPDATE_STEPS
    )

    dataloader = DataLoader(
        ds,
        batch_size=1,
        collate_fn=collate_for_dual_decoder,
        # NOTE: this sampler will split dataset evenly across workers
        sampler=DistributedSampler(ds, shuffle=True, drop_last=True) if GLOBAL_WORLD_SIZE > 1 else None,
    )
    dliter = iter(dataloader)

    if GLOBAL_RANK == 0:
        wandb_run = wandb.init(
            project="nay-dev-mimi-12",
            name = args.run_name
        )

    for global_step in range(NUM_UPDATE_STEPS):

        if global_step == args.csm_freeze_steps:
            print("Unfreezing CSM weights...")
            model.module.decoder.set_csm_trainable(True) if GLOBAL_WORLD_SIZE > 1 else model.decoder.set_csm_trainable(True)

        optimizer.zero_grad()
        loss_accum = 0.0
        tokens_this_step = 0

        for micro_step in range(args.grad_acc_steps):

            try:
                batch = next(dliter)
            except StopIteration:
                dliter = iter(dataloader)
                batch = next(dliter)

            loss, loss_dict = model(batch)
        
            loss = loss / args.grad_acc_steps
            loss_accum += loss.detach()
            loss.backward()

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        # Get lr that was used by optimizer before scheduler gets advanced
        lr_at_update = scheduler.get_last_lr()[0]
        scheduler.step()

        metrics = {
            "train/loss" : loss_accum.item(),
            "train/grad_norm": norm.item(),
            "train/learning_rate" : lr_at_update,
            "train/epoch": global_step / NUM_UPDATE_STEPS,
            "train/global_step" : global_step
        } | loss_dict

        if GLOBAL_RANK == 0:

            if global_step % args.log_interval == 0:
                wandb.log(metrics)
                print(metrics)

            if global_step > 0 and (global_step % args.checkpoint_interval == 0 or global_step == NUM_UPDATE_STEPS - 1):
                # save model
                save_checkpoint(model.module if GLOBAL_WORLD_SIZE > 1 else model, global_step, CHECKPOINT_DIR, metadata={ "wandb_run_id":wandb_run.id })

    if GLOBAL_WORLD_SIZE > 1:
        dist.destroy_process_group()
