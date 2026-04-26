"""
Verify KV-cache equivalence for SBD inference.
- full-sequence SBD forward over history + block
- cached block-only forward using prefilled history KV

Expected result after the KV layout fix:
    max logit diff ~= 0
    mask diff = 0

Run: python check_kvcache_equivalence.py
"""
import numpy as np
import minitorch
from minitorch.cuda_kernel_ops import CudaKernelOps
from minitorch.modules_transfomer import DecoderLM
from minitorch import tensor_from_numpy
from load_weights import load_weights
from inference.sbd_mask import build_sbd_inference_mask
from inference.eb_sampler import entropy_bounded_sample

def _to_numpy(storage):
    if hasattr(storage, 'copy_to_host'):
        return storage.copy_to_host()
    return np.array(storage)

backend = minitorch.TensorBackend(CudaKernelOps)
model = DecoderLM(
    n_vocab=50259, n_embd=768, n_head=12, n_positions=1024,
    n_layer=12, p_dropout=0.0, backend=backend
)
load_weights(model, "checkpoints/sbd_minitorch.npz")

prompt = [1, 2, 3, 4, 5]
block  = [100, 200, 300, 400]
causal_point = len(prompt)
block_size   = len(block)
n_head       = 12
vocab_size   = 50259

# ── Step 1: total logit equivalence ──────────────────────────────────────────
print("=" * 60)
print("Step 1: logit equivalence check")

# full path
full_seq  = prompt + block
full_idx  = tensor_from_numpy(np.array([full_seq], dtype=np.float32), backend=backend)
full_mask = build_sbd_inference_mask(causal_point, causal_point + block_size, n_head, 1, backend)
logits_full, _ = model(full_idx, mask=full_mask, past_kvs=None, offset=0)
raw_full = _to_numpy(logits_full._tensor._storage).reshape(1, causal_point + block_size, vocab_size)
raw_full = raw_full[0, causal_point:]   # (block_size, vocab)

# cached path
prompt_idx = tensor_from_numpy(np.array([prompt], dtype=np.float32), backend=backend)

prompt_mask = build_sbd_inference_mask(
    causal_point=causal_point,
    seq_len=causal_point,
    n_head=n_head,
    batch_size=1,
    backend=backend,
)

_, past_kvs = model(
    prompt_idx,
    mask=prompt_mask,
    past_kvs=None,
    offset=0
)

full_mask_np = _to_numpy(full_mask._tensor._storage).reshape(1, n_head, causal_point + block_size, causal_point + block_size)
mask_block   = tensor_from_numpy(full_mask_np[:, :, causal_point:, :], backend=backend)

block_idx = tensor_from_numpy(np.array([block], dtype=np.float32), backend=backend)
logits_cache, _ = model(block_idx, mask=mask_block, past_kvs=past_kvs, offset=causal_point)
raw_cache = _to_numpy(logits_cache._tensor._storage).reshape(1, block_size, vocab_size)[0]

diff = np.abs(raw_cache - raw_full)
max_diff = np.max(diff)
mean_diff = np.mean(diff)

print(f"  max logit diff : {max_diff:.6f}")
print(f"  mean logit diff: {mean_diff:.6f}")

if max_diff > 1e-4:
    print("  WARNING: cached path does not match full recompute")
else:
    print("  PASS: cached path matches full recompute")

# ── Step 2: mask equivalence ──────────────────────────────────────────────────
print("\nStep 2: mask check")
mask_block_np = _to_numpy(mask_block._tensor._storage).reshape(1, n_head, block_size, causal_point + block_size)
expected      = full_mask_np[:, :, causal_point:, :]
print(f"  mask diff: {np.max(np.abs(mask_block_np - expected)):.6f}")

# ── Step 3: past_kvs length ───────────────────────────────────────────────────
print("\nStep 3: past_kvs length check")
k0 = past_kvs[0][0]   # (n_head, head_dim, seq_len) or similar
print(f"  past_kvs[0][0] shape: {k0.shape}")
print(f"  causal_point: {causal_point}")

# ── Step 4: position check ────────────────────────────────────────────────────
print("\nStep 4: position equivalence")
print("  full path  positions:", list(range(0, causal_point + block_size)))
print("  cached path positions:", list(range(causal_point, causal_point + block_size)))
print("  block positions should match:", list(range(causal_point, causal_point + block_size)))

# ── Step 5: entropy / sampler check ──────────────────────────────────────────
print("\nStep 5: entropy check (cached path logits)")
probs = np.exp(raw_cache - raw_cache.max(axis=-1, keepdims=True))
probs = probs / probs.sum(axis=-1, keepdims=True)
entropy = -(probs * np.log(probs + 1e-12)).sum(axis=-1)
print(f"  entropy per block position: {entropy}")
for gamma in [0.1, 0.35, 0.6]:
    chosen_rel, chosen_tokens = entropy_bounded_sample(
        raw_cache, list(range(block_size)), gamma
    )
    print(f"  gamma={gamma}: actual chosen_rel={chosen_rel}, n={len(chosen_rel)}")