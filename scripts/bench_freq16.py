#!/usr/bin/env python3
"""Quick benchmark: bf16 training at 1/16 frequency."""
import multiprocessing
import os
import time


def training_worker(stop_event):
    """Train every 16th step at bf16."""
    import mlx.core as mx
    dtype = mx.bfloat16
    dim, hidden, batch = 2048, 8192, 32
    W1 = mx.random.normal((hidden, dim)).astype(dtype)
    W2 = mx.random.normal((dim, hidden)).astype(dtype)
    mx.eval(W1, W2)
    i = 0
    while not stop_event.is_set():
        if i % 16 != 0:
            i += 1
            time.sleep(0.001)
            continue
        x = mx.random.normal((batch, dim)).astype(dtype)
        h = mx.maximum(x @ W1.T, 0)
        out = h @ W2.T
        grad_out = mx.random.normal(out.shape).astype(dtype)
        grad_h = grad_out @ W2
        grad_h = grad_h * (h > 0)
        grad_W1 = grad_h.T @ x
        grad_W2 = grad_out.T @ h
        W1 = W1 - 0.001 * grad_W1
        W2 = W2 - 0.001 * grad_W2
        mx.eval(W1, W2)
        i += 1


def main():
    from mlx_lm import load
    from mlx_lm.server import stream_generate

    home = os.path.expanduser("~")
    model_dir = os.path.join(home, ".cache/lm-studio/models/NexVeridian/Qwen3.5-35B-A3B-3bit")
    prompt = "Write a Python function that computes the Fibonacci sequence efficiently using dynamic programming."

    print("Loading 35B 3-bit...")
    model, tokenizer = load(model_dir)

    # Solo baseline
    for _ in stream_generate(model, tokenizer, prompt, max_tokens=3):
        pass
    t0 = time.time()
    c = sum(1 for _ in stream_generate(model, tokenizer, prompt, max_tokens=40))
    solo = c / (time.time() - t0)
    print(f"Solo: {solo:.1f} tok/s")

    # bf16 + train@1/16
    print("Starting bf16 training at 1/16 frequency...")
    stop = multiprocessing.Event()
    worker = multiprocessing.Process(target=training_worker, args=(stop,))
    worker.start()
    time.sleep(3)

    for _ in stream_generate(model, tokenizer, prompt, max_tokens=3):
        pass
    t0 = time.time()
    c = sum(1 for _ in stream_generate(model, tokenizer, prompt, max_tokens=40))
    tps = c / (time.time() - t0)

    stop.set()
    worker.join(timeout=5)
    if worker.is_alive():
        worker.terminate()

    degrad = (1 - tps / solo) * 100
    # Use 13.3 as the fp32 baseline contention from earlier benchmark
    fp32_baseline = 13.3
    recovery = (tps - fp32_baseline) / (solo - fp32_baseline) * 100

    print(f"\n{'='*50}")
    print(f"  Solo:             {solo:.1f} tok/s")
    print(f"  bf16 @1/16 freq:  {tps:.1f} tok/s")
    print(f"  Degradation:      {degrad:.0f}%")
    print(f"  Recovery:         {recovery:.0f}%")
    print(f"  (vs fp32 baseline 13.3 tok/s = 67% degradation)")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
