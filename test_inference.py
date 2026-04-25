import numpy as np
import minitorch
from minitorch.cuda_kernel_ops import CudaKernelOps
from minitorch.modules_transfomer import DecoderLM
from inference.generate import generate_ntp, generate_sbd

backend = minitorch.TensorBackend(CudaKernelOps)

config = {
    'n_vocab': 50258,
    'n_embd': 128,
    'n_head': 4,
    'n_positions': 64,
    'p_dropout': 0.0,
    'ln_eps': 1e-5,
    'backend': backend,
    'use_fused_kernel': False,
}

model = DecoderLM(**config)
prompt = [1, 2, 3, 4, 5]
mask_token_id = 50257

print("Testing generate_ntp...")
tokens, nfe = generate_ntp(model, prompt, max_new_tokens=4, backend=backend)
print(f"  output length: {len(tokens)}, NFE: {nfe}")

print("Testing generate_sbd...")
tokens, avg_nfe = generate_sbd(
    model, prompt, max_new_tokens=4,
    block_size=4, gamma=0.35,
    mask_token_id=mask_token_id, backend=backend
)
print(f"  output length: {len(tokens)}, avg NFE per block: {avg_nfe:.2f}")
print("Done.")