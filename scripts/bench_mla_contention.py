#!/usr/bin/env python3
"""
Benchmark: MLA (DeepSeek-V2-Lite) vs GQA (Qwen3.5) bandwidth contention.

The hypothesis: MLA's 25x KV cache reduction means inference reads less from
DRAM, so training contention causes less degradation.

Test:
  1. Solo inference → baseline tok/s
  2. Inference + GPU contention → degraded tok/s
  3. Compare degradation % between MLA and GQA models

Usage:
    source .venv/bin/activate
    python scripts/bench_mla_contention.py
"""

import argparse
import multiprocessing
import os
import sys
import time


def gpu_contention_worker(stop_event):
    """Run continuous GPU matmuls to simulate training bandwidth contention."""
    import mlx.core as mx

    dim = 2048
    hidden = 8192
    batch = 32
    W1 = mx.random.normal((hidden, dim))
    W2 = mx.random.normal((dim, hidden))
    mx.eval(W1, W2)

    i = 0
    while not stop_event.is_set():
        x = mx.random.normal((batch, dim))
        h = mx.maximum(x @ W1.T, 0)
        out = h @ W2.T
        grad_out = mx.random.normal(out.shape)
        grad_h = grad_out @ W2
        grad_h = grad_h * (h > 0)
        grad_W1 = grad_h.T @ x
        grad_W2 = grad_out.T @ h
        W1 = W1 - 0.001 * grad_W1
        W2 = W2 - 0.001 * grad_W2
        mx.eval(W1, W2)
        i += 1


def measure_throughput(model, tokenizer, prompt, n_tokens=50):
    """Measure tok/s for generation."""
    from mlx_lm.server import stream_generate

    # Warmup
    for _ in stream_generate(model, tokenizer, prompt, max_tokens=3):
        pass

    t0 = time.time()
    count = 0
    for _ in stream_generate(model, tokenizer, prompt, max_tokens=n_tokens):
        count += 1
    elapsed = time.time() - t0
    return count / elapsed


def bench_model(model_dir, label, prompt, n_tokens=50):
    """Benchmark a model solo and under contention."""
    from mlx_lm import load

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  {model_dir}")
    print(f"{'='*60}")

    print("Loading model...")
    model, tokenizer = load(model_dir)

    # Solo
    print("\n--- Solo (no contention) ---")
    solo_tps = measure_throughput(model, tokenizer, prompt, n_tokens)
    print(f"  {solo_tps:.1f} tok/s")

    # With contention
    print("\n--- With GPU contention ---")
    stop = multiprocessing.Event()
    worker = multiprocessing.Process(target=gpu_contention_worker, args=(stop,))
    worker.start()
    time.sleep(3)  # let contention warm up

    contention_tps = measure_throughput(model, tokenizer, prompt, n_tokens)
    print(f"  {contention_tps:.1f} tok/s")

    stop.set()
    worker.join(timeout=5)
    if worker.is_alive():
        worker.terminate()

    degradation = (1 - contention_tps / solo_tps) * 100
    print(f"\n  Solo:        {solo_tps:.1f} tok/s")
    print(f"  Contention:  {contention_tps:.1f} tok/s")
    print(f"  Degradation: {degradation:.0f}%")

    return solo_tps, contention_tps, degradation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=50)
    args = parser.parse_args()

    home = os.path.expanduser("~")
    lm_models = os.path.join(home, ".cache/lm-studio/models")

    # MLA model (DeepSeek-V2-Lite)
    mla_dir = os.path.join(lm_models, "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx")
    # GQA model (Qwen3.5-35B or 4B for comparison)
    gqa_dirs = [
        os.path.join(lm_models, "NexVeridian/Qwen3.5-35B-A3B-3bit"),
        os.path.join(lm_models, "mlx-community/Qwen3.5-4B-MLX-4bit"),
    ]

    prompt = "Write a Python function that computes the Fibonacci sequence efficiently using dynamic programming."

    results = {}

    # Test MLA model
    if os.path.exists(mla_dir):
        # Check if download is complete
        parts = [f for f in os.listdir(mla_dir) if f.endswith('.part')]
        if parts:
            print(f"DeepSeek-V2-Lite still downloading ({len(parts)} parts remaining)")
        else:
            s, c, d = bench_model(mla_dir, "MLA: DeepSeek-Coder-V2-Lite (16B, 2.4B active)", prompt, args.tokens)
            results["MLA"] = (s, c, d)
    else:
        print(f"MLA model not found: {mla_dir}")

    # Test GQA model
    for gqa_dir in gqa_dirs:
        if os.path.exists(gqa_dir):
            name = os.path.basename(gqa_dir)
            s, c, d = bench_model(gqa_dir, f"GQA: {name}", prompt, args.tokens)
            results[f"GQA:{name}"] = (s, c, d)
            break  # only need one GQA reference

    # Summary
    if len(results) >= 2:
        print(f"\n{'='*60}")
        print(f"  COMPARISON")
        print(f"{'='*60}")
        print(f"  {'Model':<30} {'Solo':>8} {'Contend':>8} {'Degrad':>8}")
        print(f"  {'-'*54}")
        for name, (s, c, d) in results.items():
            print(f"  {name:<30} {s:>7.1f} {c:>7.1f} {d:>7.0f}%")
        print()
        print("  If MLA degrades less than GQA, the KV cache bandwidth")
        print("  thesis is confirmed: smaller cache = less bus contention.")


if __name__ == "__main__":
    main()
