import numpy as np
import os
import pandas as pd

from datasets import Dataset, load_from_disk, concatenate_datasets
from tqdm import tqdm

import numpy as np

# -----------------------------
# Optimised evenly_place_np
# -----------------------------
def evenly_place_np(longer, shorter):
    """
    Evenly interleave elements of `shorter` inside `longer`.
    
    Parameters
    ----------
    longer : array-like of int
        The larger array (indices) into which elements of `shorter` will be interleaved.
    shorter : array-like of int
        The smaller array (indices) to be interleaved.
    
    Returns
    -------
    np.ndarray
        Interleaved array of indices.
    
    Example:
    evenly_place_np([10,11,12,13,14,15,16,17,18], [1,2,3])
    -> array([10, 11,  1, 12, 13,  2, 14, 15, 16,  3, 17, 18])
    """
    longer  = np.asarray(longer,  dtype=np.int64)
    shorter = np.asarray(shorter, dtype=np.int64)

    nL = len(longer)
    nS = len(shorter)

    # Edge cases
    if nS == 0:
        return longer.copy()
    if nL == 0:
        return shorter.copy()

    # Evenly spaced positions inside (0, nL)
    insert_pos = np.linspace(0, nL, nS + 2)[1:-1]
    insert_pos = np.round(insert_pos).astype(np.int64)

    # Allocate output array
    out = np.empty(nL + nS, dtype=np.int64)

    # Shifted insertion indices
    insert_idx = insert_pos + np.arange(nS, dtype=np.int64)

    # Boolean mask for shorter array
    mask = np.zeros(nL + nS, dtype=bool)
    mask[insert_idx] = True

    # Fill partitions
    out[mask] = shorter
    out[~mask] = longer

    return out

# -----------------------------
# Interleave multiple datasets
# -----------------------------
def interleave_multiple(datasets, seed=0):
    """
    Interleave multiple datasets by size, merging largest two first.
    Shuffles indices within each dataset first.
    Parameters
    ----------
    datasets : list of HF Datasets
        Datasets to merge.
    seed : int, optional
        Random seed for reproducibility.
    Returns
    -------
    np.ndarray of interleaved global indices
    """
    rng = np.random.default_rng(seed)

    # Compute lengths and global index ranges, shuffled within each dataset
    lengths = np.array([len(ds) for ds in datasets], dtype=np.int64)
    cumulative = np.cumsum(lengths)
    starts = np.concatenate(([0], cumulative[:-1]))

    # Shuffled index ranges
    ranges = []
    for i, length in enumerate(lengths):
        idx = np.arange(length, dtype=np.int64)
        rng.shuffle(idx)  # shuffle indices within dataset
        idx += starts[i]  # shift to global dataset
        ranges.append(idx)

    # Sort datasets by descending size
    order = np.argsort(-lengths)
    sorted_ranges = [ranges[i] for i in order]
    sorted_lengths = lengths[order]

    # Merge pairwise until one dataset remains
    while len(sorted_ranges) > 1:
        longer  = sorted_ranges[0]
        shorter = sorted_ranges[1]

        merged = evenly_place_np(longer, shorter)

        # Replace first two with merged dataset
        new_range = merged
        new_len = len(merged)
        sorted_ranges = [new_range] + sorted_ranges[2:]
        sorted_lengths = np.array([new_len] + list(sorted_lengths[2:]))

        # Resort by descending size
        order = np.argsort(-sorted_lengths)
        sorted_ranges = [sorted_ranges[i] for i in order]
        sorted_lengths = sorted_lengths[order]

    # Only one dataset remains
    return sorted_ranges[0]

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

def create_input_ids(example):
    start_of_text = 151644 # <|im_start|>
    end_of_text = 151645 # <|im_end|>
    tokeniser_length = 151665
    start_of_speech = tokeniser_length + 1
    end_of_speech = tokeniser_length + 2
    system_token_id = 8948 # system
    assistant_token_id = 77091 # assistant
    newline_token_id = 198

    Last, First = [ name.title() for name in example['_id'].split('/')[-1].split("_")[:2] ]
    example["text"] = "{ " + First + " " + Last + " }:  " + example["text"]

    example["text_tokens"] = tokenizer.encode(example["text"], add_special_tokens=False)

    # first list in example['mimi_24khz_tokens'] = coarse tokens
    tokeniser_length += 10
    all_codes = [ i + tokeniser_length for i in example['mimi_24khz_tokens'][0] ]

    example["input_ids"] = (
        [start_of_text] + [system_token_id, newline_token_id] + example["text_tokens"] + [end_of_text, newline_token_id] +
        [start_of_text] + [assistant_token_id, newline_token_id, start_of_speech] + all_codes + [end_of_speech, end_of_text]
    )

    return example

def make_ds():

    from pathlib import Path

    ds_paths  = [ str(p) for p in Path("/rime-training-data/arcana-stage1-audiobooks-mimi-12_v2026.01/eng/").glob("*") ]

    all_ds_list = [ load_from_disk(p) for p in ds_paths ]

    all_ds = concatenate_datasets(all_ds_list).map(create_input_ids, num_proc=os.cpu_count())

    interleaved_indices = interleave_multiple(all_ds_list)

    all_ds_interleaved = all_ds.select(interleaved_indices)

    return all_ds_interleaved

if __name__ == "__main__":

    def batch_gen():
        ds = make_ds()

        # Define accumulators
        current_batch_total_tokens = 0
        current_batch_ids = []
        current_batch_input_ids = []
        current_batch_speaker_ids = []
        current_batch_mimi_24khz_tokens = []
        
        for d in tqdm(ds, total=len(ds)):

            current_example_num_tokens=len(d['input_ids'])

            if current_example_num_tokens > 8_192:
                # Skip if single example longer than 8192
                continue

            if current_batch_total_tokens + current_example_num_tokens <= 8_192:
                # Add current example to accumulated batch
                current_batch_total_tokens += current_example_num_tokens
                current_batch_ids.append(d['_id'])
                current_batch_input_ids.append(d['input_ids'])
                # current_batch_speaker_ids.append(d['speaker_id'])
                current_batch_mimi_24khz_tokens.append(d['mimi_24khz_tokens'])
            else:
                # Yield what has been accumulated (but don't add current example)
                yield {
                    '_ids' : current_batch_ids,
                    'input_ids' : current_batch_input_ids,
                    # 'speaker_ids' : current_batch_speaker_ids,
                    'mimi_24khz_tokens' : current_batch_mimi_24khz_tokens
                }

                # Start new batch with current example
                current_batch_total_tokens = current_example_num_tokens
                current_batch_ids = [d['_id']]
                current_batch_input_ids = [d['input_ids']]
                # current_batch_speaker_ids = [d['speaker_id']]
                current_batch_mimi_24khz_tokens = [d['mimi_24khz_tokens']]

        # Yield any remaining batch
        if current_batch_ids:
            yield {
                '_ids': current_batch_ids,
                'input_ids': current_batch_input_ids,
                # 'speaker_ids': current_batch_speaker_ids,
                'mimi_24khz_tokens' : current_batch_mimi_24khz_tokens
            }

    ds_packed = Dataset.from_generator(batch_gen)

    ds_packed.save_to_disk("/workspace/tmp/v4-crash-course/packed___eng-all")
