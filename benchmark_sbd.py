"""
Benchmark: MiniTorch NTP vs SBD decoding.

Measures NFE per token and wall-clock time for both decoding strategies
across gamma values of 0.1, 0.35, and 0.6.

Usage:
    python benchmark_sbd.py \
        --npz checkpoints/sbd_minitorch.npz \
        --max_new_tokens 32 \
        --block_size 4 \
        --n_prompts 5
"""

import argparse
import time
import numpy as np

import minitorch
from minitorch.cuda_kernel_ops import CudaKernelOps
from minitorch.modules_transfomer import DecoderLM
from transformers import GPT2Tokenizer

from load_weights import load_weights
from inference.generate import generate_ntp, generate_sbd

GAMMAS = [0.1, 0.35, 0.6]

PROMPTS = [
    "The quick brown fox",
    "In the beginning there was",
    "Scientists have discovered that",
    "The history of artificial intelligence",
    "Once upon a time in a land",
]


def build_model(npz_path: str, backend) -> DecoderLM:
    model = DecoderLM(
        n_vocab=50259,
        n_embd=768,
        n_head=12,
        n_positions=1024,
        n_layer=12,
        p_dropout=0.0,
        backend=backend,
    )
    load_weights(model, npz_path)
    model.eval()
    return model


def run_benchmark(args):
    backend = minitorch.TensorBackend(CudaKernelOps)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"pad_token": "[PAD]", "mask_token": "[MASK]"})
    mask_token_id = tokenizer.mask_token_id

    print("Loading model...")
    model = build_model(args.npz, backend)
    print("Model loaded.\n")

    prompts = PROMPTS[:args.n_prompts]
    prompt_ids = [tokenizer.encode(p) for p in prompts]

    results = {}

    # NTP baseline
    print("=" * 50)
    print("Running NTP baseline...")
    ntp_times = []
    for ids in prompt_ids:
        t0 = time.time()
        generate_ntp(model, ids, args.max_new_tokens, backend)
        ntp_times.append(time.time() - t0)

    ntp_avg_time = np.mean(ntp_times)
    ntp_nfe_per_token = 1.0
    ntp_tokens_per_sec = args.max_new_tokens / ntp_avg_time
    results["NTP"] = {
        "nfe_per_token": ntp_nfe_per_token,
        "avg_time_sec": ntp_avg_time,
        "tokens_per_sec": ntp_tokens_per_sec,
    }
    print(f"  avg time: {ntp_avg_time:.2f}s | tokens/sec: {ntp_tokens_per_sec:.2f}")

    # SBD at each gamma
    for gamma in GAMMAS:
        print(f"\nRunning SBD (gamma={gamma})...")
        sbd_times = []
        sbd_nfes = []

        for ids in prompt_ids:
            t0 = time.time()
            _, avg_nfe = generate_sbd(
                model, ids, args.max_new_tokens,
                args.block_size, gamma, mask_token_id, backend
            )
            sbd_times.append(time.time() - t0)
            sbd_nfes.append(avg_nfe)

        avg_time = np.mean(sbd_times)
        avg_nfe_per_block = np.mean(sbd_nfes)
        avg_nfe_per_token = avg_nfe_per_block / args.block_size
        tokens_per_sec = args.max_new_tokens / avg_time
        nfe_speedup = ntp_nfe_per_token / avg_nfe_per_token
        wall_speedup = ntp_avg_time / avg_time

        results[f"SBD_gamma={gamma}"] = {
            "nfe_per_token": avg_nfe_per_token,
            "nfe_per_block": avg_nfe_per_block,
            "nfe_speedup": nfe_speedup,
            "avg_time_sec": avg_time,
            "tokens_per_sec": tokens_per_sec,
            "wall_speedup": wall_speedup,
        }
        print(f"  avg NFE/block: {avg_nfe_per_block:.2f} / {args.block_size}  |  NFE speedup: {nfe_speedup:.2f}x")
        print(f"  avg time: {avg_time:.2f}s | tokens/sec: {tokens_per_sec:.2f} | wall speedup: {wall_speedup:.2f}x")

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'Method':<20} {'NFE/token':<12} {'NFE speedup':<14} {'Time (s)':<12} {'Wall speedup'}")
    print("-" * 70)
    for name, r in results.items():
        nfe_sp = r.get("nfe_speedup", 1.0)
        wall_sp = r.get("wall_speedup", 1.0)
        print(f"{name:<20} {r['nfe_per_token']:<12.3f} {nfe_sp:<14.2f} {r['avg_time_sec']:<12.2f} {wall_sp:.2f}x")

    # Also show generated text for first prompt to check quality
    print("\n" + "=" * 50)
    print("Generation quality check (first prompt):")
    print(f"Prompt: {prompts[0]!r}")

    ids = prompt_ids[0]
    ntp_out, _ = generate_ntp(model, ids, args.max_new_tokens, backend)
    print(f"NTP:    {tokenizer.decode(ntp_out)!r}")

    for gamma in GAMMAS:
        sbd_out, _ = generate_sbd(
            model, ids, args.max_new_tokens,
            args.block_size, gamma, mask_token_id, backend
        )
        print(f"SBD g={gamma}: {tokenizer.decode(sbd_out)!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz",            default="checkpoints/sbd_minitorch.npz")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--block_size",     type=int, default=4)
    parser.add_argument("--n_prompts",      type=int, default=5)
    args = parser.parse_args()
    run_benchmark(args)