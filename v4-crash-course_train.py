import argparse
import datasets
import json
import os
import torch
import wandb

import numpy as np
import torch.distributed as dist

from pathlib import Path
from qwen_tts import Qwen3TTSTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
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
        rime_num_codebooks=16
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

        # Random init projection from Qwen hidden dim to CSM embedding dim
        self.qwenH_to_csmH = nn.Linear(qwen_backbone_dim, csm_backbone_dim, bias=False)

        # Seperate projection to transformer 1536-dim Qwen Codebook 0 embeddings to match Codebook 1+ embeddings
        self.qwenC0_to_csmC0 = nn.Linear(qwen_backbone_dim, csm_backbone_dim, bias=False)
    
    def forward(self, backbone_hidden_states, backbone_codebook0_embeddings, qwen16_targets):
        
        # Case in case backbone weights/outputs are in bfloat16

        # Up-project Qwen to match CSM hidden states [B, 1, 1536] -> [B, 1, 2048]
        backbone_hidden_states = backbone_hidden_states.to(dtype=self.qwenH_to_csmH.weight.dtype)
        csmH = self.qwenH_to_csmH(backbone_hidden_states)

        # Up-project Qwen Codebook 0 embeddings to match pre-trained CSM Codebook 1+ embeddings
        backbone_codebook0_embeddings = backbone_codebook0_embeddings.to(dtype=self.qwenC0_to_csmC0.weight.dtype)
        csmC0 = self.qwenC0_to_csmC0(backbone_codebook0_embeddings)

        # Input embeds: [ H ] + [ C0 ] + [ C1 ... Cn-1 ]
        input_embeds = torch.cat(
            [ csmH ] + 
            [ csmC0 ] +
            [ self.audio_embeddings(qwen16_targets[:, i-1] + ((i-1) * self.csm_audio_vocab_size)).unsqueeze(1) for i in range(2, self.rime_num_codebooks) ],
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
            qwen16_targets.reshape(-1),                              # (B*C,)
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

        # Look up bacbkone's codebook0 embeddings to pass to depth decoder
        backbone_codebook0_embeddings = self.backbone.get_input_embeddings()((batch['qwen16_targets'][:,0] + 151_675).cuda()).unsqueeze(1)

        # Decoder forward 
        decoder_loss = self.decoder(
            backbone_hidden_state,
            backbone_codebook0_embeddings,
            batch['qwen16_targets'].to(self.device)
        )

        return backbone_loss, decoder_loss

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        backbone_max_tokens=1000,
        backbone_temp=0.5,
        backbone_top_p=1.0,
        bacbkone_rep_pen=1.1,
        depth_topk=32,
        depth_temp=0.1
    ):
        
        backbone_outputs = self.backbone.generate(
            input_ids=input_ids.cuda(),
            attention_mask=torch.ones_like(input_ids).cuda(),
            max_new_tokens=backbone_max_tokens,
            do_sample=True,
            temperature=backbone_temp,
            top_p=backbone_top_p,
            repetition_penalty=bacbkone_rep_pen,
            num_return_sequences=1,
            eos_token_id=151_667,
            return_dict_in_generate=True,
            output_hidden_states=True
        )

        last_layer_hidden_states = [ all_layers_hidden_states[-1] for all_layers_hidden_states in backbone_outputs.hidden_states[:-1] ]

        generated_frames = []
        
        for t in range(len(last_layer_hidden_states)):
        
            backbone_hidden_states=last_layer_hidden_states[t][:, -1, :]
            
            backbone_hidden_states = backbone_hidden_states.to(
                dtype=self.decoder.qwenH_to_csmH.weight.dtype
            )
            
            with torch.no_grad():
                csmH = self.decoder.qwenH_to_csmH(backbone_hidden_states)
                
                c0_logits = self.decoder.codebook0_head(csmH)
                c0_sample = sample_topk(c0_logits, depth_topk, depth_temp)
                
                c0_qwen_embed = self.backbone.get_input_embeddings()(c0_sample + 151_675)
                c0_csm_embed  = self.decoder.qwenC0_to_csmC0(c0_qwen_embed.to(dtype=self.decoder.qwenC0_to_csmC0.weight.dtype))
            
                curr_h = torch.cat([ csmH.unsqueeze(1), c0_csm_embed ], dim=1)
                curr_sample = c0_sample.clone()
            
                for i in range(1, self.decoder.rime_num_codebooks):
                    decoder_h = self.decoder.decoder(self.decoder.projection(curr_h))
                    
                    ci_logits = torch.mm(decoder_h[:, -1, :], self.decoder.audio_head[i - 1])
                    ci_sample = sample_topk(ci_logits, depth_topk, depth_temp)
                    ci_embed = self.decoder.audio_embeddings(ci_sample + i*2051)
            
                    curr_h = torch.cat([curr_h, ci_embed], dim=1)
                    curr_sample = torch.cat([curr_sample, ci_sample], dim=1)
                    
            generated_frames.append(curr_sample)

        qwen16_codes = torch.stack(generated_frames, dim=-1).cpu().squeeze(0).T

        return qwen16_codes

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

    qwen16_seq_lengths = [ d.shape[0] for d in packed_example['qwen_24khz_tokens'] ]
    qwen16_frames_to_sample = [ np.random.choice(s, size=min(16, s), replace=False) for s in qwen16_seq_lengths ]

    first_speech_token_indices = np.where(input_ids==151_666)[0] + 1

    indices_for_backbone_output = np.concatenate([ indices + offset for (offset, indices) in zip(first_speech_token_indices, qwen16_frames_to_sample) ]).astype(np.int64)

    qwen16_targets = torch.cat([
        torch.LongTensor(seq_all_frames[indices_to_sample, :])
        for (seq_all_frames, indices_to_sample) in
        zip(packed_example['qwen_24khz_tokens'], qwen16_frames_to_sample)
    ])

    return {
        "_ids" : [ str(id) for id in data[0]['_ids'] ], # Return IDs for debugging/logging 
        "backbone_inputs" : backbone_inputs,
        "indices_for_backbone_output" : indices_for_backbone_output,
        "qwen16_targets" : qwen16_targets
    }

# Helpers

def get_loss_weights(global_step, total_steps):
    """
    Linearly go from:
       For first third of training: backbone weight 1.0, decoder weight 0.1
       For second third of training: backbone weight: 0.5, decoder weight: 1.0
       For final third of training: backbone weight: 0.1, decoder weight: 1.0
    """
    if global_step < total_steps // 3:
        return 1.0, 0.1
    elif global_step < 2 * total_steps // 3:
        return 0.5, 1.0
    else:
        return 0.1, 1.0

def save_checkpoint(model, global_step, checkpoint_dir, metadata=None):
    write_path = f"{checkpoint_dir}/checkpoint_{global_step}"
    Path(write_path).mkdir(parents=True, exist_ok=True)
    print(f"Saving model to {write_path}")
    torch.save(model.state_dict(), f"{write_path}/dual-decoder.pt")

def move_batch_to_device(batch, device="cuda"):
    # Move keys that don't start with '_' (e.g. '_ids')
    return { k:v.to(device) for (k,v) in batch.items() if not k.startswith('_') }

def prepare_input_ids(text):
    start_of_text      = 151644 # <|im_start|>
    end_of_text        = 151645 # <|im_end|>
    tokeniser_length   = 151665
    start_of_speech    = tokeniser_length + 1
    end_of_speech      = tokeniser_length + 2
    system_token_id    = 8948 # system
    assistant_token_id = 77091 # assistant
    newline_token_id   = 198

    text_tokens = qwen_text_tokenizer.encode(text, add_special_tokens=False)

    # Qwen chat format
    input_ids = torch.LongTensor([(
        [start_of_text] + [system_token_id, newline_token_id] + text_tokens + [end_of_text, newline_token_id] +
        [start_of_text] + [assistant_token_id, newline_token_id, start_of_speech]
    )])

    return input_ids

def _multinomial_sample_one_no_sync(probs):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)

def sample_topk(logits: torch.Tensor, topk: int, temperature: float):
    logits = logits / temperature

    filter_value: float = -float("Inf")
    indices_to_remove = logits < torch.topk(logits, topk)[0][..., -1, None]
    scores_processed = logits.masked_fill(indices_to_remove, filter_value)
    scores_processed = torch.nn.functional.log_softmax(scores_processed, dim=-1)
    probs = torch.nn.functional.softmax(scores_processed, dim=-1)

    sample_token = _multinomial_sample_one_no_sync(probs)
    return sample_token

def decode_audio(qwen_codes_cpu):
    audio_samples, audio_sr = qwen_audio_tokenizer.decode({ "audio_codes" : qwen_codes_cpu })
    return audio_samples[0]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('run_name')
    parser.add_argument('train_dataset')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--checkpoint-interval', type=int, default=10000)
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--grad-acc-steps', type=int, default=1)
    parser.add_argument('--audio-gen-interval', type=int, default=2500)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--lr-schedule-total-steps', type=int, default=None, help='Total steps for LR schedule (use full dataset value for consistent LR trajectory)')

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

    if GLOBAL_WORLD_SIZE > 1:
        model = DDP(model, device_ids=[ LOCAL_RANK ])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=[0.9, 0.999],
        eps=1e-8,
        weight_decay=0.0,
        fused=True
    )

    # Use reference steps for LR schedule if provided, otherwise use actual steps
    # Used for when using a smaller dev dataset but keeping LR schedule same as full run
    LR_SCHEDULE_TOTAL_STEPS = args.lr_schedule_total_steps or NUM_UPDATE_STEPS

    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=LR_SCHEDULE_TOTAL_STEPS
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

        # Only run infer on rank 0
        qwen_text_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
        qwen_audio_tokenizer = Qwen3TTSTokenizer.from_pretrained("Qwen/Qwen3-TTS-Tokenizer-12Hz", device_map="cuda")

    for global_step in range(NUM_UPDATE_STEPS):

        optimizer.zero_grad()
        loss_accum = 0.0
        tokens_this_step = 0

        for micro_step in range(args.grad_acc_steps):

            try:
                batch = next(dliter)
            except StopIteration:
                dliter = iter(dataloader)
                batch = next(dliter)

            backbone_loss, decoder_loss = model(batch)
            backbone_loss_weight, decoder_loss_weight = 1.0, 1.0 # Turn off loss weighting for now

            backbone_loss *= backbone_loss_weight
            decoder_loss  *= decoder_loss_weight

            loss = backbone_loss + decoder_loss

            loss_dict = {
                "loss/total": loss.item(),
                "loss/backbone": backbone_loss.item(),
                "loss/decoder": decoder_loss.item()
            }

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

            model_ref = model.module if GLOBAL_WORLD_SIZE > 1 else model

            if global_step % args.log_interval == 0:
                wandb.log(metrics)
                print(metrics)

            if global_step > 0 and (global_step % args.checkpoint_interval == 0 or global_step == NUM_UPDATE_STEPS - 1):
                # save model
                save_checkpoint(model_ref, global_step, CHECKPOINT_DIR, metadata={ "wandb_run_id":wandb_run.id })

            if global_step > 0 and args.audio_gen_interval > 0 and (global_step % args.audio_gen_interval == 0 or global_step == NUM_UPDATE_STEPS - 1):

                print("Generating audio...")

                model_ref.eval()

                prompts = [
                    '{ Grover Gardner }:  I am a speech generation model that can sound like a real person.'
                ]

                sample_audios_table = wandb.Table(columns=["Text", "Audio"])

                for p in prompts:
                    input_ids = prepare_input_ids(p)
                    qwen16_codes = model_ref.generate(input_ids)
                    reconstructed_audio = decode_audio(qwen16_codes)

                    sample_audios_table.add_data(p, wandb.Audio(reconstructed_audio, sample_rate=24_000))

                wandb.log({"sample_audios_table": sample_audios_table})

                model_ref.train()

    if GLOBAL_WORLD_SIZE > 1:
        dist.destroy_process_group()
