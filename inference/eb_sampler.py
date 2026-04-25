import numpy as np
from typing import List, Tuple


def entropy_bounded_sample(
    logits: np.ndarray,
    masked_positions: List[int],
    gamma: float,
) -> Tuple[List[int], List[int]]:
    """
    EB-Sampler — Algorithm 3 / Equation 8 of the paper.

    Sorts masked positions by ascending entropy, then unmasks the largest
    prefix s such that cumulative entropy of positions 0..s-2 stays within
    gamma. At least one token is always revealed per call.

    Args:
        logits:           (n_masked, vocab_size) raw logits for masked positions
        masked_positions: token indices corresponding to each row in logits
        gamma:            entropy threshold

    Returns:
        chosen_positions: indices from masked_positions to unmask
        chosen_tokens:    greedy token id for each chosen position
    """
    probs = _softmax(logits)
    entropies = -np.sum(probs * np.log(probs + 1e-10), axis=-1)

    order = np.argsort(entropies)
    sorted_entropies = entropies[order]

    s = 1
    cumsum = 0.0
    for i in range(len(masked_positions) - 1):
        cumsum += sorted_entropies[i]
        if cumsum <= gamma:
            s = i + 2
        else:
            break

    chosen_order = order[:s]
    chosen_positions = [masked_positions[i] for i in chosen_order]
    chosen_tokens = [int(np.argmax(probs[i])) for i in chosen_order]

    return chosen_positions, chosen_tokens


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=-1, keepdims=True)