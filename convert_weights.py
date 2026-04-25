"""
Convert HuggingFace GPT-2 SBD checkpoint to MiniTorch DecoderLM format.

Usage:
    python convert_weights.py \
        --ckpt checkpoints/sbd_5000steps.pt \
        --out  checkpoints/sbd_minitorch.npz \
        --n_layer 12
"""

import argparse
import numpy as np
import torch


def convert(ckpt_path: str, out_path: str, n_layer: int):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    out = {}

    # token and position embeddings
    out["token_embeddings.weights"] = ckpt["transformer.wte.weight"].numpy()
    # GPT-2 position embeddings: (1024, 768) -> take first n_positions rows at load time
    out["position_embeddings.weights"] = ckpt["transformer.wpe.weight"].numpy()

    for i in range(n_layer):
        hf = f"transformer.h.{i}"
        mt = f"t_layer_{i + 1}"

        # layer norm 1
        out[f"{mt}.ln_1.weights"] = ckpt[f"{hf}.ln_1.weight"].numpy()
        out[f"{mt}.ln_1.bias"]    = ckpt[f"{hf}.ln_1.bias"].numpy()

        # layer norm 2
        out[f"{mt}.ln_2.weights"] = ckpt[f"{hf}.ln_2.weight"].numpy()
        out[f"{mt}.ln_2.bias"]    = ckpt[f"{hf}.ln_2.bias"].numpy()

        # c_attn is (768, 2304) = Q, K, V concatenated along dim 1
        c_attn_w = ckpt[f"{hf}.attn.c_attn.weight"].numpy()  # (768, 2304)
        c_attn_b = ckpt[f"{hf}.attn.c_attn.bias"].numpy()    # (2304,)
        n_embd = c_attn_w.shape[0]

        out[f"{mt}.attention.q_projection.weights"] = c_attn_w[:, :n_embd]
        out[f"{mt}.attention.q_projection.bias"]    = c_attn_b[:n_embd]

        out[f"{mt}.attention.k_projection.weights"] = c_attn_w[:, n_embd:2 * n_embd]
        out[f"{mt}.attention.k_projection.bias"]    = c_attn_b[n_embd:2 * n_embd]

        out[f"{mt}.attention.v_projection.weights"] = c_attn_w[:, 2 * n_embd:]
        out[f"{mt}.attention.v_projection.bias"]    = c_attn_b[2 * n_embd:]

        # output projection
        out[f"{mt}.attention.out_projection.weights"] = ckpt[f"{hf}.attn.c_proj.weight"].numpy()
        out[f"{mt}.attention.out_projection.bias"]    = ckpt[f"{hf}.attn.c_proj.bias"].numpy()

        # feed forward
        out[f"{mt}.ff.linear_in.weights"]  = ckpt[f"{hf}.mlp.c_fc.weight"].numpy()
        out[f"{mt}.ff.linear_in.bias"]     = ckpt[f"{hf}.mlp.c_fc.bias"].numpy()
        out[f"{mt}.ff.linear_out.weights"] = ckpt[f"{hf}.mlp.c_proj.weight"].numpy()
        out[f"{mt}.ff.linear_out.bias"]    = ckpt[f"{hf}.mlp.c_proj.bias"].numpy()

    # final layer norm
    out["ln.weights"] = ckpt["transformer.ln_f.weight"].numpy()
    out["ln.bias"]    = ckpt["transformer.ln_f.bias"].numpy()

    # lm head (GPT-2 ties lm_head to wte, so use wte weight transposed)
    # MiniTorch Linear: x @ weights, so weights shape is (in, out) = (768, vocab)
    if "lm_head.weight" in ckpt:
        out["lm_head.weights"] = ckpt["lm_head.weight"].numpy().T
    else:
        out["lm_head.weights"] = ckpt["transformer.wte.weight"].numpy().T

    # lm_head has no bias in GPT-2
    np.savez(out_path, **out)
    print(f"Saved {len(out)} arrays to {out_path}")
    for k, v in out.items():
        print(f"  {k}: {v.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",    default="checkpoints/sbd_5000steps.pt")
    parser.add_argument("--out",     default="checkpoints/sbd_minitorch.npz")
    parser.add_argument("--n_layer", type=int, default=12)
    args = parser.parse_args()
    convert(args.ckpt, args.out, args.n_layer)