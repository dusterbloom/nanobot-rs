#!/usr/bin/env python3
"""
MoE routing hook — captures expert routing decisions during mlx-lm inference.

Wraps each MoE gate to log (layer, expert_indices, probs, x_norm) to a
binary ring buffer. The Rust RouterTrainer reads via drain_routing_targets_from_file().

Usage:
    source .venv/bin/activate
    python scripts/routing_hook.py                    # test with 30 tokens
    python scripts/routing_hook.py --tokens 200       # longer test
    python scripts/routing_hook.py --monitor           # watch buffer fill
"""

import argparse
import os
import struct
import sys
import time

import numpy as np

ROUTING_FILE = os.path.expanduser("~/.nanobot/routing_targets.bin")
HEADER_SIZE = 32


def record_size(dim, k):
    return 4 + k * 2 + k * 4 + dim * 4


def init_buffer(path, n_layers, dim, k, capacity=8192):
    rec_size = record_size(dim, k)
    total_size = HEADER_SIZE + capacity * rec_size
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<QQ III I", 0, 0, n_layers, dim, k, capacity))
        f.write(b"\x00" * (capacity * rec_size))
    return total_size, rec_size, capacity


class RoutingCollector:
    """Collects routing decisions in memory, flushes to disk periodically."""

    def __init__(self, path, dim, k, capacity, rec_size):
        self.path = path
        self.dim = dim
        self.k = k
        self.capacity = capacity
        self.rec_size = rec_size
        self.records = []
        self.write_pos = 0

    def record(self, layer_idx, expert_indices, expert_probs, x_norm):
        self.records.append((layer_idx, expert_indices, expert_probs, x_norm))

    def flush(self):
        if not self.records:
            return 0

        with open(self.path, "r+b") as f:
            # Read current write_pos
            f.seek(0)
            self.write_pos = struct.unpack("<Q", f.read(8))[0]

            for (layer_idx, indices, probs, x_norm) in self.records:
                offset = HEADER_SIZE + (self.write_pos % self.capacity) * self.rec_size
                k = self.k
                dim = self.dim

                rec = struct.pack("<HH", layer_idx, k)
                for idx in indices[:k]:
                    rec += struct.pack("<H", int(idx))
                for p in probs[:k]:
                    rec += struct.pack("<f", float(p))
                for d in range(dim):
                    val = float(x_norm[d]) if d < len(x_norm) else 0.0
                    rec += struct.pack("<f", val)

                f.seek(offset)
                f.write(rec)
                self.write_pos += 1

            # Update write_pos in header
            f.seek(0)
            f.write(struct.pack("<Q", self.write_pos))
            f.flush()

        flushed = len(self.records)
        self.records.clear()
        return flushed


class GateWrapper:
    """Wraps an MLX gate to capture routing decisions."""

    def __init__(self, original, layer_idx, collector, dim, k):
        self._original = original
        self._layer_idx = layer_idx
        self._collector = collector
        self._dim = dim
        self._k = k

    def __call__(self, x):
        import mlx.core as mx
        result = self._original(x)

        try:
            # Cast to float32 before numpy — MLX may return bfloat16
            gates_np = np.array(result.astype(mx.float32))
            x_np = np.array(x.astype(mx.float32))

            # Flatten to get the last token's gate logits
            # gates shape: (seq, num_experts) or (num_experts,)
            while gates_np.ndim > 1:
                gates_np = gates_np[-1]
            while x_np.ndim > 1:
                x_np = x_np[-1]

            n_experts = gates_np.shape[0]
            k = min(self._k, n_experts)
            if k < 1:
                return result

            top_idx = np.argpartition(gates_np, -k)[-k:]
            top_idx = top_idx[np.argsort(-gates_np[top_idx])]

            top_logits = gates_np[top_idx].astype(np.float64)
            top_logits -= top_logits.max()
            exp_l = np.exp(top_logits)
            probs = (exp_l / exp_l.sum()).astype(np.float32)

            x_norm = x_np[:self._dim].astype(np.float32)

            self._collector.record(self._layer_idx, top_idx, probs, x_norm)
        except Exception as e:
            if self._layer_idx == 0:
                import traceback
                traceback.print_exc()

        return result

    def __getattr__(self, name):
        return getattr(self._original, name)


def patch_model(model, collector, dim, k):
    layers = model.language_model.layers
    n_hooked = 0
    for i, layer in enumerate(layers):
        mlp = layer.mlp
        if hasattr(mlp, "gate") and hasattr(mlp, "switch_mlp"):
            mlp.gate = GateWrapper(mlp.gate, i, collector, dim, k)
            n_hooked += 1
    return n_hooked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--output", type=str, default=ROUTING_FILE)
    parser.add_argument("--monitor", action="store_true")
    parser.add_argument("--tokens", type=int, default=30)
    args = parser.parse_args()

    if args.monitor:
        print(f"Monitoring {args.output}...")
        last = 0
        while True:
            try:
                with open(args.output, "rb") as f:
                    d = f.read(HEADER_SIZE)
                    if len(d) >= HEADER_SIZE:
                        wp, rp = struct.unpack_from("<QQ", d, 0)
                        nl, dim, k, cap = struct.unpack_from("<IIII", d, 16)
                        if wp > last:
                            print(f"  w={wp} r={rp} pending={wp-rp} new={wp-last}")
                        last = wp
            except FileNotFoundError:
                pass
            time.sleep(2)
        return

    from mlx_lm import load, generate

    home = os.path.expanduser("~")
    if args.model_dir is None:
        for d in [
            f"{home}/.cache/lm-studio/models/NexVeridian/Qwen3.5-35B-A3B-3bit",
            f"{home}/.cache/lm-studio/models/mlx-community/Qwen3.5-35B-A3B-4bit",
        ]:
            if os.path.exists(d):
                args.model_dir = d
                break
    if not args.model_dir:
        print("No model found"); sys.exit(1)

    print(f"Loading {os.path.basename(args.model_dir)}...")
    model, tokenizer = load(args.model_dir)

    dim = model.language_model.args.hidden_size
    k = getattr(model.language_model.args, "num_experts_per_tok", 8)
    n_layers = len(model.language_model.layers)
    print(f"  dim={dim}, k={k}, layers={n_layers}")

    total_size, rec_size, capacity = init_buffer(args.output, n_layers, dim, k)
    collector = RoutingCollector(args.output, dim, k, capacity, rec_size)
    print(f"  Buffer: {args.output} ({total_size/1024:.0f} KB, {capacity} records)")

    n_hooked = patch_model(model, collector, dim, k)
    print(f"  Hooked {n_hooked}/{n_layers} MoE gates")

    prompt = "Explain the concept of quantum entanglement in simple terms."
    print(f"\nGenerating {args.tokens} tokens...")
    t0 = time.time()
    result = generate(model, tokenizer, prompt=prompt,
                      max_tokens=args.tokens, verbose=True)
    elapsed = time.time() - t0

    # Flush remaining records
    flushed = collector.flush()

    print(f"\n{'='*60}")
    print(f"  ROUTING HOOK RESULTS")
    print(f"{'='*60}")
    print(f"  Tokens:     ~{args.tokens}")
    print(f"  Records:    {collector.write_pos} ({flushed} flushed)")
    rpt = collector.write_pos / max(args.tokens, 1)
    print(f"  Rec/token:  {rpt:.1f} (expect {n_hooked})")
    print(f"  Speed:      {args.tokens/elapsed:.1f} tok/s")
    if collector.write_pos > 0:
        print(f"\n  SUCCESS — {collector.write_pos} routing decisions captured!")
    else:
        print(f"\n  FAILURE — no records")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
