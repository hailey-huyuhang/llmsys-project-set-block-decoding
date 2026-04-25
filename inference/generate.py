import numpy as np
from typing import List, Tuple

from minitorch import tensor_from_numpy
from minitorch.modules_transfomer import DecoderLM
from minitorch.tensor_ops import TensorBackend

from inference.sbd_mask import build_sbd_inference_mask
from inference.eb_sampler import entropy_bounded_sample


def _to_numpy(storage) -> np.ndarray:
    """Convert MiniTorch storage (numpy or CUDA array) to numpy."""
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
    Standard autoregressive NTP decoding, one token at a time.

    Returns the full token sequence and total NFE (== max_new_tokens).
    """
    tokens = list(prompt)
    nfe = 0

    for _ in range(max_new_tokens):
        idx = tensor_from_numpy(
            np.array([tokens], dtype=np.float32), backend=backend
        )
        logits = model(idx)

        seq_len = len(tokens)
        vocab_size = model.n_vocab
        raw = _to_numpy(logits._tensor._storage).reshape(1, seq_len, vocab_size)
        next_token = int(np.argmax(raw[0, -1]))

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
    SBD block-level decoding — Algorithm 2 of the paper.

    Generates tokens block by block. Within each block, sample_block is
    called until all positions are unmasked.

    Returns the full token sequence and average NFE per block.
    """
    tokens = list(prompt)
    total_nfe = 0
    n_blocks = max_new_tokens // block_size

    for _ in range(n_blocks):
        tokens, block_nfe = sample_block(
            model, tokens, block_size, gamma, mask_token_id, backend
        )
        total_nfe += block_nfe

    avg_nfe = total_nfe / n_blocks if n_blocks > 0 else 0.0
    return tokens, avg_nfe


def sample_block(
    model: DecoderLM,
    tokens: List[int],
    block_size: int,
    gamma: float,
    mask_token_id: int,
    backend: TensorBackend,
) -> Tuple[List[int], int]:
    """
    Decode one block of block_size tokens — Algorithm 3 of the paper.

    Initialises all block positions with [MASK], then calls the model and
    the EB-Sampler repeatedly until no masked positions remain.

    Returns context + decoded block and number of forward passes used.
    """
    causal_point = len(tokens)
    seq_len = causal_point + block_size
    n_head = model.t_layer_1.attention.n_head
    vocab_size = model.n_vocab

    block = [mask_token_id] * block_size
    masked_positions = list(range(block_size))
    nfe = 0

    while masked_positions:
        current_seq = tokens + block
        idx = tensor_from_numpy(
            np.array([current_seq], dtype=np.float32), backend=backend
        )
        mask = build_sbd_inference_mask(
            causal_point=causal_point,
            seq_len=seq_len,
            n_head=n_head,
            batch_size=1,
            backend=backend,
        )

        logits = model(idx, mask=mask)
        nfe += 1

        raw = _to_numpy(logits._tensor._storage).reshape(1, seq_len, vocab_size)
        block_logits = raw[0, causal_point:]          # (block_size, vocab)
        masked_logits = block_logits[masked_positions] # (n_masked, vocab)

        chosen_rel, chosen_tokens = entropy_bounded_sample(
            masked_logits, masked_positions, gamma
        )

        for rel_pos, token_id in zip(chosen_rel, chosen_tokens):
            block[rel_pos] = token_id

        masked_positions = [p for p in masked_positions if p not in chosen_rel]

    return tokens + block, nfe