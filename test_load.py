import minitorch
from minitorch.cuda_kernel_ops import CudaKernelOps
from minitorch.modules_transfomer import DecoderLM
from load_weights import load_weights

backend = minitorch.TensorBackend(CudaKernelOps)

model = DecoderLM(
    n_vocab=50259, n_embd=768, n_head=12,
    n_positions=1024, n_layer=12,
    p_dropout=0.0, backend=backend
)

load_weights(model, "checkpoints/sbd_minitorch.npz")

# simple forward test
import numpy as np
from minitorch import tensor_from_numpy
idx = tensor_from_numpy(np.array([[1, 2, 3, 4, 5]], dtype=np.float32), backend=backend)
logits = model(idx)
print(f"logits shape: {logits.shape}")
print("Done.")