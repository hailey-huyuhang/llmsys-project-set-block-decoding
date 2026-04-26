import numpy as np
import time
from typing import List, Optional, Tuple

from minitorch import tensor_from_numpy
from minitorch.modules_transfomer import DecoderLM
from minitorch.tensor_ops import TensorBackend

from inference.sbd_mask import build_sbd_inference_mask
from inference.eb_sampler import entropy_bounded_sample


def _to_numpy(storage) -> np.ndarray:
    if hasattr(storage, 'copy_to_host'):
        return storage.copy_to_host()
    return np.array(storage)


def generate_ntp(
    model: DecoderLM,
    prompt: List[int],
    max_new_tokens: int,
    backend: TensorBackend,
) -> Tuple[List[int], int]:
    """
    NTP decoding with KV-cache. Prefills the prompt once, then generates
    one token at a time using cached K/V from all previous positions.

    Returns the full token sequence and total NFE (== max_new_tokens).
    """
    tokens = list(prompt)
    vocab_size = model.n_vocab

    # Prefill: run the full prompt through the model, get KV-cache
    idx = tensor_from_numpy(
        np.array([tokens], dtype=np.float32), backend=backend
    )
    logits, past_kvs = model(idx, past_kvs=None, offset=0)

    # Sample first new token from last position of prefill
    raw = _to_numpy(logits._tensor._storage).reshape(1, len(tokens), vocab_size)
    next_token = int(np.argmax(raw[0, -1]))
    tokens.append(next_token)
    nfe = 1  # prefill counts as 1 NFE

    # Decode: one token at a time, reusing KV-cache
    for _ in range(max_new_tokens - 1):
        offset = len(tokens) - 1
        idx = tensor_from_numpy(
            np.array([[tokens[-1]]], dtype=np.float32), backend=backend
        )
        logits, past_kvs = model(idx, past_kvs=past_kvs, offset=offset)

        raw = _to_numpy(logits._tensor._storage).reshape(1, 1, vocab_size)
        next_token = int(np.argmax(raw[0, 0]))
        tokens.append(next_token)
        nfe += 1

    return tokens, nfe


def generate_sbd(
    model: DecoderLM,
    prompt: List[int],
    max_new_tokens: int,
    block_size: int,
    gamma: float,
    mask_token_id: int,
    backend: TensorBackend,
) -> Tuple[List[int], float]:
    """
    SBD block-level decoding with KV-cache, which is Algorithm 2 of the paper.

    The prompt is prefilled once. Each block is decoded by sample_block,
    which uses the KV-cache for the history and runs bidirectional attention
    only over the current block positions.

    Returns the full token sequence and average NFE per block.
    """
    tokens = list(prompt)
    vocab_size = model.n_vocab
    n_blocks = max_new_tokens // block_size
    total_nfe = 0

    # Prefill prompt
    idx = tensor_from_numpy(
        np.array([tokens], dtype=np.float32), backend=backend
    )
    _, past_kvs = model(idx, past_kvs=None, offset=0)

    for _ in range(n_blocks):
        tokens, past_kvs, block_nfe = sample_block(
            model, tokens, past_kvs, block_size, gamma, mask_token_id, backend
        )
        total_nfe += block_nfe

    avg_nfe = total_nfe / n_blocks if n_blocks > 0 else 0.0
    return tokens, avg_nfe


def sample_block(
    model: DecoderLM,
    tokens: List[int],
    past_kvs: list,
    block_size: int,
    gamma: float,
    mask_token_id: int,
    backend: TensorBackend,
) -> Tuple[List[int], list, int]:
    """
    Decode one block of block_size tokens, which is Algorithm 3 of the paper.

    Uses past_kvs (KV-cache of all history before this block) on the first
    forward call. Within the block, iterates until all masked positions are
    revealed. Returns updated tokens, updated KV-cache, and NFE count.
    """
    causal_point = len(tokens)
    seq_len = causal_point + block_size
    n_head = model.t_layer_1.attention.n_head
    vocab_size = model.n_vocab
    past_kvs_history = past_kvs

    block = [mask_token_id] * block_size
    masked_positions = list(range(block_size))
    nfe = 0

    full_mask = build_sbd_inference_mask(
                causal_point=causal_point,
                seq_len=seq_len,
                n_head=n_head,
                batch_size=1,
                backend=backend,
    )
    
    mask_np = _to_numpy(full_mask._tensor._storage).reshape(
        1, n_head, causal_point + block_size, causal_point + block_size
    )

    mask_block = tensor_from_numpy(mask_np[:, :, causal_point:, :], backend=backend)

    while masked_positions:
        block_idx = tensor_from_numpy(
            np.array([block], dtype=np.float32), backend=backend
        )
        
        logits, _ = model(
            block_idx, mask=mask_block, past_kvs=past_kvs_history, offset=causal_point
        )

        nfe += 1

        raw = _to_numpy(logits._tensor._storage).reshape(1, block_size, vocab_size)
        block_logits = raw[0]                          # (block_size, vocab)
        masked_logits  = block_logits[masked_positions]  # (n_masked, vocab)

        chosen_rel, chosen_tokens = entropy_bounded_sample(
            masked_logits, masked_positions, gamma
        )

        for rel_pos, token_id in zip(chosen_rel, chosen_tokens):
            block[rel_pos] = token_id

        masked_positions = [p for p in masked_positions if p not in chosen_rel]
    
        # One final forward to get the full history+block KV-cache for the next block
    block_idx = tensor_from_numpy(
        np.array([block], dtype=np.float32), backend=backend
    )
    _, final_kvs = model(
        block_idx, mask=mask_block, past_kvs=past_kvs_history, offset=causal_point
    )

    return tokens + block, final_kvs, nfe