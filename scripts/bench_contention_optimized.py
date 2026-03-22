#!/usr/bin/env python3
"""
Benchmark: optimized contention strategies.

Tests four bandwidth-reduction techniques individually and combined:
1. Lower precision training (bf16 vs fp32)
2. Gradient checkpointing (recompute vs cache activations)
3. Reduced training frequency (every Nth token)
4. All combined

Measures inference tok/s under each contention scenario.
"""

import multiprocessing
import os
import time
import sys


def training_worker(stop_event, config):
    """Simulate training with configurable precision and checkpointing."""
    import mlx.core as mx

    precision = config.get("precision", "float32")
    checkpoint = config.get("checkpoint", False)
    train_every_n = config.get("train_every_n", 1)
    dtype = mx.bfloat16 if precision == "bfloat16" else mx.float32

    dim = 2048
    hidden = 8192
    batch = 32

    W1 = mx.random.normal((hidden, dim)).astype(dtype)
    W2 = mx.random.normal((dim, hidden)).astype(dtype)
    mx.eval(W1, W2)

    i = 0
    while not stop_event.is_set():
        # Simulate training frequency: skip iterations
        if i % train_every_n != 0:
            i += 1
            time.sleep(0.001)  # yield CPU
            continue

        x = mx.random.normal((batch, dim)).astype(dtype)

        # Forward
        h = x @ W1.T
        h = mx.maximum(h, 0)

        if checkpoint:
            # Gradient checkpointing: don't cache intermediate h
            # Recompute in backward (simulated by del + recompute)
            del h
            # Recompute for backward
            h = mx.maximum(x @ W1.T, 0)

        out = h @ W2.T

        # Backward
        grad_out = mx.random.normal(out.shape).astype(dtype)
        grad_h = grad_out @ W2
        grad_h = grad_h * (h > 0)
        grad_W1 = grad_h.T @ x
        grad_W2 = grad_out.T @ h

        # Update
        W1 = W1 - 0.001 * grad_W1
        W2 = W2 - 0.001 * grad_W2
        mx.eval(W1, W2)
        i += 1


def measure_throughput(model, tokenizer, prompt, n_tokens=30):
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


def run_scenario(model, tokenizer, prompt, label, config, n_tokens=30):
    """Run inference under specific contention config."""
    print(f"\n  {label}")

    stop = multiprocessing.Event()
    worker = multiprocessing.Process(target=training_worker, args=(stop, config))
    worker.start()
    time.sleep(2)  # let training warm up

    tps = measure_throughput(model, tokenizer, prompt, n_tokens)

    stop.set()
    worker.join(timeout=5)
    if worker.is_alive():
        worker.terminate()

    print(f"    → {tps:.1f} tok/s")
    return tps


def main():
    from mlx_lm import load

    home = os.path.expanduser("~")
    model_dir = os.path.join(home, ".cache/lm-studio/models/NexVeridian/Qwen3.5-35B-A3B-3bit")
    if not os.path.exists(model_dir):
        # Fall back to 4-bit
        model_dir = os.path.join(home, ".cache/lm-studio/models/mlx-community/Qwen3.5-35B-A3B-4bit")
    if not os.path.exists(model_dir):
        print("No 35B model found")
        sys.exit(1)

    prompt = "Write a Python function that computes the Fibonacci sequence efficiently using dynamic programming."
    n_tokens = 30

    print(f"Loading {os.path.basename(model_dir)}...")
    model, tokenizer = load(model_dir)

    # Baseline: solo (no contention)
    print(f"\n{'='*60}")
    print(f"  CONTENTION OPTIMIZATION BENCHMARK")
    print(f"  Model: {os.path.basename(model_dir)}")
    print(f"{'='*60}")

    print(f"\n  Solo (no contention)")
    solo = measure_throughput(model, tokenizer, prompt, n_tokens)
    print(f"    → {solo:.1f} tok/s")

    results = {"solo": solo}

    # Scenario 1: Current (fp32 training, no checkpointing)
    results["fp32_baseline"] = run_scenario(
        model, tokenizer, prompt,
        "fp32 training (current baseline)",
        {"precision": "float32", "checkpoint": False})

    # Scenario 2: bf16 training
    results["bf16"] = run_scenario(
        model, tokenizer, prompt,
        "bf16 training (lower precision)",
        {"precision": "bfloat16", "checkpoint": False})

    # Scenario 3: fp32 + gradient checkpointing
    results["fp32_ckpt"] = run_scenario(
        model, tokenizer, prompt,
        "fp32 + gradient checkpointing",
        {"precision": "float32", "checkpoint": True})

    # Scenario 4: bf16 + gradient checkpointing
    results["bf16_ckpt"] = run_scenario(
        model, tokenizer, prompt,
        "bf16 + gradient checkpointing",
        {"precision": "bfloat16", "checkpoint": True})

    # Scenario 5: bf16 + checkpointing + reduced frequency (train every 4th step)
    results["bf16_ckpt_freq4"] = run_scenario(
        model, tokenizer, prompt,
        "bf16 + checkpointing + train every 4th step",
        {"precision": "bfloat16", "checkpoint": True, "train_every_n": 4})

    # Scenario 6: bf16 + checkpointing + reduced frequency (train every 8th step)
    results["bf16_ckpt_freq8"] = run_scenario(
        model, tokenizer, prompt,
        "bf16 + checkpointing + train every 8th step",
        {"precision": "bfloat16", "checkpoint": True, "train_every_n": 8})

    # Summary
    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  {'Scenario':<45} {'tok/s':>7} {'Degrad':>7} {'Recovery':>8}")
    print(f"  {'-'*67}")

    baseline_loss = solo - results["fp32_baseline"]

    for name, tps in results.items():
        degrad = (1 - tps / solo) * 100 if name != "solo" else 0
        recovery = ((tps - results["fp32_baseline"]) / baseline_loss * 100) if name != "solo" and baseline_loss > 0 else 0
        label = {
            "solo": "Solo (ceiling)",
            "fp32_baseline": "fp32 training (current)",
            "bf16": "bf16 training",
            "fp32_ckpt": "fp32 + checkpointing",
            "bf16_ckpt": "bf16 + checkpointing",
            "bf16_ckpt_freq4": "bf16 + ckpt + train@1/4 freq",
            "bf16_ckpt_freq8": "bf16 + ckpt + train@1/8 freq",
        }.get(name, name)
        if name == "solo":
            print(f"  {label:<45} {tps:>6.1f}       —        —")
        else:
            print(f"  {label:<45} {tps:>6.1f}   {degrad:>5.0f}%   {recovery:>6.0f}%")

    print()
    print(f"  Recovery% = how much of the contention loss is recovered")
    print(f"  Target: >50% recovery = meaningful improvement")


if __name__ == "__main__":
    main()
