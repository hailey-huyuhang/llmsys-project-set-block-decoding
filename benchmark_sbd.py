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
from minitorch import tensor_from_numpy
from transformers import GPT2Tokenizer

from load_weights import load_weights
from inference.generate import generate_ntp, generate_sbd

# GAMMAS = [0.1, 0.35, 0.6]
GAMMAS = [3.5, 6.0, 10.0, 16.0]

PROMPTS = [
    "The quick brown fox",
    "In the beginning there was",
    "Scientists have discovered that",
    "The history of artificial intelligence",
    "Once upon a time in a land",
]


def _to_numpy(storage) -> np.ndarray:
    if hasattr(storage, "copy_to_host"):
        return storage.copy_to_host()
    return np.array(storage)


def compute_self_ppl(model, tokens: list, prompt_len: int, backend) -> float:
    """Score tokens[prompt_len:] with the NTP model. Returns perplexity."""
    vocab_size = model.n_vocab
    n_gen = len(tokens) - prompt_len
    if n_gen <= 0:
        return float("inf")
    seq = tokens[:-1]
    idx = tensor_from_numpy(np.array([seq], dtype=np.float32), backend=backend)
    logits, _ = model(idx, past_kvs=None, offset=0)
    raw = _to_numpy(logits._tensor._storage).reshape(len(seq), vocab_size)
    total_nll = 0.0
    for i in range(n_gen):
        logit = raw[prompt_len - 1 + i].astype(np.float64)
        target = tokens[prompt_len + i]
        logit -= logit.max()
        total_nll += np.log(np.sum(np.exp(logit))) - logit[target]
    return float(np.exp(total_nll / n_gen))


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
    ntp_times, ntp_ppls, ntp_outs = [], [], []
    for ids in prompt_ids:
        t0 = time.time()
        out, _ = generate_ntp(model, ids, args.max_new_tokens, backend)
        ntp_times.append(time.time() - t0)
        ntp_outs.append(out)
        ntp_ppls.append(compute_self_ppl(model, out, len(ids), backend))

    ntp_avg_time = float(np.mean(ntp_times))
    ntp_avg_ppl  = float(np.mean(ntp_ppls))
    ntp_tokens_per_sec = args.max_new_tokens / ntp_avg_time
    results["NTP"] = {
        "block_size":    "-",
        "gamma":         "-",
        "nfe_per_token": 1.0,
        "nfe_speedup":   1.0,
        "wall_speedup":  1.0,
        "tokens_per_sec": ntp_tokens_per_sec,
        "avg_time_sec":  ntp_avg_time,
        "avg_ppl":       ntp_avg_ppl,
    }
    print(f"  avg time: {ntp_avg_time:.2f}s | tokens/sec: {ntp_tokens_per_sec:.2f} | PPL: {ntp_avg_ppl:.2f}")

    # SBD: sweep block_sizes x gammas
    for block_size in args.block_sizes:
        if args.max_new_tokens < block_size:
            print(f"\n[SKIP] block_size={block_size} > max_new_tokens={args.max_new_tokens}")
            continue
        for gamma in args.gammas:
            label = f"SBD bs={block_size} g={gamma}"
            print(f"\nRunning {label}...")
            sbd_times, sbd_nfes, sbd_ppls = [], [], []

            for ids in prompt_ids:
                t0 = time.time()
                out, avg_nfe = generate_sbd(
                    model, ids, args.max_new_tokens,
                    block_size, gamma, mask_token_id, backend,
                )
                sbd_times.append(time.time() - t0)
                sbd_nfes.append(avg_nfe)
                sbd_ppls.append(compute_self_ppl(model, out, len(ids), backend))

            avg_time      = float(np.mean(sbd_times))
            avg_nfe_block = float(np.mean(sbd_nfes))
            avg_nfe_token = avg_nfe_block / block_size
            avg_ppl       = float(np.mean(sbd_ppls))
            nfe_speedup   = 1.0 / avg_nfe_token if avg_nfe_token > 0 else float("inf")
            wall_speedup  = ntp_avg_time / avg_time

            results[label] = {
                "block_size":    block_size,
                "gamma":         gamma,
                "nfe_per_token": avg_nfe_token,
                "nfe_per_block": avg_nfe_block,
                "nfe_speedup":   nfe_speedup,
                "wall_speedup":  wall_speedup,
                "tokens_per_sec": args.max_new_tokens / avg_time,
                "avg_time_sec":  avg_time,
                "avg_ppl":       avg_ppl,
            }
            print(
                f"  NFE/block: {avg_nfe_block:.2f}/{block_size} | "
                f"NFE speedup: {nfe_speedup:.2f}x | "
                f"wall speedup: {wall_speedup:.2f}x | PPL: {avg_ppl:.2f}"
            )

    # Summary table
    print("\n" + "=" * 85)
    print(f"{'Method':<24} {'BS':>3} {'γ':>5} {'NFE/tok':>8} {'NFE↑':>7} {'Wall↑':>7} {'tok/s':>6} {'PPL':>8}")
    print("-" * 85)
    for name, r in results.items():
        print(
            f"{name:<24} {str(r['block_size']):>3} {str(r['gamma']):>5} "
            f"{r['nfe_per_token']:>8.3f} {r['nfe_speedup']:>7.2f}x "
            f"{r['wall_speedup']:>7.2f}x {r['tokens_per_sec']:>6.2f} "
            f"{r['avg_ppl']:>8.2f}"
        )

    # Spot-check: first prompt, each block_size at the middle gamma
    print("\n" + "=" * 60)
    print(f"Generation spot-check  prompt: {prompts[0]!r}")
    print(f"NTP: {tokenizer.decode(ntp_outs[0])!r}")
    mid_gamma = args.gammas[len(args.gammas) // 2]
    ids0 = prompt_ids[0]
    for block_size in args.block_sizes:
        if args.max_new_tokens < block_size:
            continue
        out, _ = generate_sbd(
            model, ids0, args.max_new_tokens,
            block_size, mid_gamma, mask_token_id, backend,
        )
        print(f"SBD bs={block_size} g={mid_gamma}: {tokenizer.decode(out)!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz",            default="checkpoints/sbd_34000steps_minitorch.npz")
    parser.add_argument("--max_new_tokens", type=int,   default=32)
    parser.add_argument("--block_sizes",    type=int,   nargs="+", default=[4, 8, 16])
    parser.add_argument("--gammas",         type=float, nargs="+", default=[3.5, 6.0, 10.0, 16.0])
    parser.add_argument("--n_prompts",      type=int,   default=1)
    args = parser.parse_args()
    run_benchmark(args)