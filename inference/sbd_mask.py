import numpy as np
from minitorch import tensor_from_numpy
from minitorch.tensor_ops import TensorBackend


def build_sbd_inference_mask(
    causal_point: int,
    seq_len: int,
    n_head: int,
    batch_size: int,
    backend: TensorBackend,
):
    """
    4D additive attention mask for SBD block inference (Figure 10).

    Positions before causal_point use causal attention.
    Positions from causal_point onward attend to all positions.
    0.0 = attend, -inf = block.

    Returns MiniTorch tensor of shape (batch_size, n_head, seq_len, seq_len).
    """
    NEG_INF = -np.finfo(np.float32).max
    mask = np.zeros((batch_size, n_head, seq_len, seq_len), dtype=np.float32)

    for q in range(causal_point):
        for k in range(q + 1, seq_len):
            mask[:, :, q, k] = NEG_INF

    return tensor_from_numpy(mask, backend=backend)