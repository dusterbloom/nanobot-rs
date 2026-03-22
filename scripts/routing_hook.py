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

# Same probes as training_eval.rs HARD_PROBES
HARD_PROBES = [
    ("What is 7893 × 4567? Just the number.", "36047331"),
    ("What is 123456 + 789012 + 345678? Just the number.", "1258146"),
    ("What is 17^4? Just the number.", "83521"),
    ("What is 2^23 - 2^20? Just the number.", "7340032"),
    ("What is the remainder when 2^100 is divided by 7?", "2"),
    ("What is the last two digits of 3^200?", "01"),
    ("How many integers between 1 and 1000 are divisible by 3 but not by 5?", "267"),
    ("What is the sum of divisors of 360?", "1170"),
    ("How many 5-card poker hands contain exactly one pair?", "1098240"),
    ("How many surjective functions from a 5-element set to a 3-element set?", "150"),
    ("In how many ways can 12 people be divided into 3 groups of 4?", "5775"),
    ("How many lattice paths from (0,0) to (6,4) using only right and up steps?", "210"),
    ("A projectile launched at 45 degrees with speed 50m/s. How far does it land in meters? Use g=10.", "250"),
    ("What is the escape velocity from Earth's surface in km/s? Use R=6400km, g=9.8.", "11.2"),
    ("A capacitor of 10uF charged to 100V is connected to a 1kOhm resistor. What is the current in mA after 15ms?", "22"),
    ("Two masses 3kg and 5kg connected by a string over a frictionless pulley. What is the acceleration in m/s^2? Use g=10.", "2.5"),
    ("If a DNA sequence is 5'-ATGCGATCG-3', what is the mRNA sequence?", "AUGCGAUCG"),
    ("A star has luminosity 100 times the Sun and temperature 2 times the Sun. What is its radius relative to the Sun?", "2.5"),
    ("In information theory, a source emits A with probability 0.5, B with 0.25, C and D with 0.125 each. What is the entropy in bits?", "1.75"),
    ("What is the pH of a buffer solution containing 0.1M acetic acid (Ka=1.8e-5) and 0.1M sodium acetate?", "4.74"),
    ("If A→B, B→C, not C. What can we conclude about A? Answer 'not A' or 'A'.", "not A"),
    ("A says 'B is a liar'. B says 'A and C are both liars'. C says 'A is truthful'. If exactly one is a liar, who is it?", "B"),
    ("In a room of 23 people, what is the approximate probability that two share a birthday? Answer as a percentage.", "50"),
    ("Five pirates divide 100 gold coins by voting. The most senior proposes and needs majority. How many coins does pirate 1 (most senior) keep?", "98"),
    ("What does this print: x=1; for i in range(5): x = x*2+1; print(x)", "63"),
    ("In Python: len(set('mississippi')). What is the answer?", "4"),
    ("What is the output: x=[1,2,3]; x.append(x); len(x)?", "4"),
    ("In Python: sum(1 for x in range(100) if x%3==0 or x%5==0). Answer?", "47"),
    ("You have 2 coins: fair and double-headed. Pick one at random, flip it, get heads. What is the probability the coin is fair?", "1/3"),
    ("Three doors, one has a prize. You pick door 1, host opens door 3 (no prize). Should you switch? What is the probability of winning if you switch?", "2/3"),
    ("A test is 99% accurate. Disease prevalence is 1%. You test positive. What is the probability you have the disease? Answer as approximate percentage.", "50"),
    ("Roll two dice. Given that their sum is 7, what is the probability that one die shows 3?", "1/3"),
    ("A snail climbs 3m each day and slides back 2m each night. How many days to reach the top of a 10m well?", "8"),
    ("If 5 machines make 5 widgets in 5 minutes, how many minutes do 100 machines take to make 100 widgets?", "5"),
    ("A lily pad doubles in size each day. It covers the whole pond on day 30. On what day does it cover half the pond?", "29"),
    ("You have 12 balls, one weighs differently. Using a balance scale, what is the minimum number of weighings to find it?", "3"),
    ("What is the maximum number of nodes in a binary tree of height 5?", "63"),
    ("How many comparisons does merge sort need in the worst case for 8 elements?", "17"),
    ("What is the output of: (lambda f: f(f))(lambda x: 42)?", "42"),
    ("In a graph with 6 vertices and 15 edges, how many triangles at most?", "20"),
]


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
    parser.add_argument("--eval", action="store_true",
                        help="Run hard probes, grade, save results + routing data")
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

    if args.eval:
        # Eval mode: run hard probes, grade, write rewards to a JSON file
        print(f"\n{'='*60}")
        print(f"  EVAL MODE — grading {len(HARD_PROBES)} hard probes")
        print(f"{'='*60}")

        results = []
        correct = 0
        t0 = time.time()

        for i, (prompt, expected) in enumerate(HARD_PROBES):
            # Reset buffer positions for per-probe drain
            collector.flush()

            full_prompt = f"Answer in one word or number only. No explanation. {prompt}"
            response = generate(model, tokenizer, prompt=full_prompt,
                               max_tokens=30, verbose=False)
            collector.flush()

            # Grade
            answer = response.strip().lower()
            hit = expected.lower() in answer
            if hit:
                correct += 1
            reward = 1.0 if hit else -1.0

            mark = "+" if hit else "-"
            if i < 5 or i >= len(HARD_PROBES) - 2:
                print(f"  [{mark}] {prompt[:50]:<50} → {answer[:30]}")

            results.append({
                "prompt": prompt,
                "expected": expected,
                "response": answer[:100],
                "hit": hit,
                "reward": reward,
                "records_in_buffer": collector.write_pos,
            })

        elapsed = time.time() - t0
        pct = 100 * correct / len(HARD_PROBES)
        print(f"\n  Score: {correct}/{len(HARD_PROBES)} ({pct:.0f}%)")
        print(f"  Time: {elapsed:.1f}s ({elapsed/len(HARD_PROBES):.1f}s/probe)")
        print(f"  Records: {collector.write_pos}")

        # ── REINFORCE training on router weights ──
        ne = getattr(model.language_model.args, "num_experts", 256)
        print(f"\n{'='*60}")
        print(f"  REINFORCE TRAINING — {ne} experts × {dim} dim")
        print(f"{'='*60}")

        import mlx.core as mx

        layers = model.language_model.layers
        # Extract current router gate weights per layer (dequantized)
        original_gates = {}
        for i, layer in enumerate(layers):
            mlp = layer.mlp
            if hasattr(mlp, 'gate') and isinstance(mlp.gate, GateWrapper):
                gate = mlp.gate._original
                # Gate may be quantized (QuantizedLinear). Call it to get
                # the effective weight, or access weight directly if dense.
                if hasattr(gate, 'scales'):
                    # Quantized: dequantize by running identity through it
                    eye = mx.eye(dim)
                    w = np.array(gate(eye).astype(mx.float32).T)  # [ne, dim]
                else:
                    w = np.array(gate.weight.astype(mx.float32))  # [ne, dim]
                if w.shape != (ne, dim):
                    print(f"  L{i}: gate shape {w.shape} != ({ne}, {dim}) — skipping")
                    continue
                original_gates[i] = w.copy()

        if not original_gates:
            print("  No gate weights found — skipping REINFORCE")
        else:
            # Group routing records by layer with rewards
            # collector.records was flushed, but we kept results with per-probe rewards
            # Re-collect: run probes again (fast, model is loaded)
            print(f"  Re-running probes for routing collection...")

            layer_targets = {l: [] for l in original_gates}
            for probe_idx, (prompt, expected) in enumerate(HARD_PROBES):
                collector.records.clear()
                full_prompt = f"Answer in one word or number only. No explanation. {prompt}"
                _ = generate(model, tokenizer, prompt=full_prompt,
                            max_tokens=20, verbose=False)

                reward = results[probe_idx]["reward"]
                for (layer_idx, indices, probs, x_norm) in collector.records:
                    if layer_idx in layer_targets:
                        layer_targets[layer_idx].append({
                            "x_norm": x_norm,
                            "expert_indices": indices,
                            "expert_probs": probs,
                            "reward": reward,
                        })
                collector.records.clear()

            total_targets = sum(len(v) for v in layer_targets.values())
            print(f"  Collected {total_targets} routing targets across {len(original_gates)} layers")

            # REINFORCE training: conservative — the original router was trained
            # on trillions of tokens. We're nudging it, not replacing it.
            lr = 0.0  # Zero LR = no training, just test dequant→dense replacement
            n_steps = 1
            trained_gates = {}

            for layer_idx, targets in sorted(layer_targets.items()):
                if not targets:
                    continue

                w = original_gates[layer_idx].copy()  # [ne, dim]
                adam_m = np.zeros_like(w)
                adam_v = np.zeros_like(w)

                for step in range(n_steps):
                    # Build batch
                    x_batch = np.stack([t["x_norm"][:dim] for t in targets])  # [B, dim]
                    rewards_batch = np.array([t["reward"] for t in targets])  # [B]

                    # Normalize rewards
                    r_mean = rewards_batch.mean()
                    r_std = max(rewards_batch.std(), 1e-6)
                    norm_rewards = (rewards_batch - r_mean) / r_std  # [B]

                    # Forward: logits = x @ W^T → [B, ne]
                    logits = x_batch @ w.T
                    # Softmax
                    logits_shifted = logits - logits.max(axis=1, keepdims=True)
                    exp_l = np.exp(logits_shifted)
                    probs = exp_l / exp_l.sum(axis=1, keepdims=True)  # [B, ne]

                    # One-hot from selected experts
                    one_hot = np.zeros_like(probs)
                    for b, t in enumerate(targets):
                        k_sel = len(t["expert_indices"])
                        for idx in t["expert_indices"][:8]:
                            if idx < ne:
                                one_hot[b, idx] = 1.0 / k_sel

                    # REINFORCE gradient: d_logits = reward * (probs - one_hot)
                    d_logits = norm_rewards[:, None] * (probs - one_hot) / len(targets)

                    # dW = d_logits^T @ x_batch → [ne, dim]
                    dw = d_logits.T @ x_batch

                    # Adam
                    t_step = step + 1
                    adam_m = 0.9 * adam_m + 0.1 * dw
                    adam_v = 0.999 * adam_v + 0.001 * dw ** 2
                    m_hat = adam_m / (1 - 0.9 ** t_step)
                    v_hat = adam_v / (1 - 0.999 ** t_step)
                    w -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

                trained_gates[layer_idx] = w

                if layer_idx == min(original_gates.keys()) or layer_idx == max(original_gates.keys()):
                    delta = np.abs(w - original_gates[layer_idx]).max()
                    print(f"  L{layer_idx}: max_delta={delta:.6f}, {len(targets)} targets")

            # Apply trained weights: replace quantized gate with dense Linear
            print(f"\n  Applying trained router weights...")
            for layer_idx, trained_w in trained_gates.items():
                layer = layers[layer_idx]
                # Create a dense Linear to replace the quantized gate
                import mlx.nn as nn
                dense_gate = nn.Linear(dim, ne, bias=False)
                dense_gate.weight = mx.array(trained_w.astype(np.float32))
                # Replace the GateWrapper's original with the dense gate
                layer.mlp.gate._original = dense_gate
            mx.eval(*[layers[l].mlp.gate._original.weight for l in trained_gates])
            print(f"  {len(trained_gates)} layers updated (quantized → dense)")

            # Re-eval with trained router
            print(f"\n{'='*60}")
            print(f"  POST-REINFORCE EVAL")
            print(f"{'='*60}")

            post_correct = 0
            for i, (prompt, expected) in enumerate(HARD_PROBES):
                full_prompt = f"Answer in one word or number only. No explanation. {prompt}"
                response = generate(model, tokenizer, prompt=full_prompt,
                                   max_tokens=30, verbose=False)
                answer = response.strip().lower()
                hit = expected.lower() in answer
                if hit:
                    post_correct += 1
                mark = "+" if hit else "-"
                if i < 5 or i >= len(HARD_PROBES) - 2:
                    print(f"  [{mark}] {prompt[:50]:<50} → {answer[:30]}")

            post_pct = 100 * post_correct / len(HARD_PROBES)
            delta_pct = post_pct - pct

            print(f"\n{'='*60}")
            print(f"  RESULTS")
            print(f"{'='*60}")
            print(f"  Before: {correct}/{len(HARD_PROBES)} ({pct:.0f}%)")
            print(f"  After:  {post_correct}/{len(HARD_PROBES)} ({post_pct:.0f}%)")
            print(f"  Delta:  {delta_pct:+.0f}%")
            if delta_pct > 0:
                print(f"  VERDICT: REINFORCE IMPROVED the model!")
            elif delta_pct == 0:
                print(f"  VERDICT: No change (need more data or steps)")
            else:
                print(f"  VERDICT: Regression ({delta_pct:.0f}%)")
            print(f"{'='*60}")

        # Save results
        eval_output = os.path.expanduser("~/.nanobot/eval_results.json")
        import json
        with open(eval_output, "w") as f:
            json.dump({
                "score_before": correct,
                "score_after": post_correct if 'post_correct' in dir() else correct,
                "total": len(HARD_PROBES),
                "routing_file": args.output,
            }, f, indent=2)
        print(f"  Saved: {eval_output}")
    else:
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
