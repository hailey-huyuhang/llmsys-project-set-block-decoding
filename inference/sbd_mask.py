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
    # causal region: q < causal_point, block future positions
    if causal_point > 0:
        q_idx = np.arange(causal_point)
        k_idx = np.arange(seq_len)
        future = k_idx[None, :] > q_idx[:, None]  # (causal_point, seq_len)
        mask[:, :, :causal_point, :] = np.where(future, NEG_INF, 0.0)
    return tensor_from_numpy(mask, backend=backend)