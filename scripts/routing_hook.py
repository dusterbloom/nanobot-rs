#!/usr/bin/env python3
"""
MoE routing hook for oMLX/mlx-lm inference.

Patches the MoE gate modules with forward hooks that log routing decisions
to a shared file. The Rust RouterTrainer reads from this file to train
the router from production inference traffic.

Usage:
    # Start oMLX with routing hooks enabled:
    python scripts/routing_hook.py --model-dir ~/.cache/lm-studio/models/NexVeridian/Qwen3.5-35B-A3B-3bit

    # Or inject into a running mlx-lm server (if model is already loaded):
    python scripts/routing_hook.py --attach-pid <mlx_lm_pid>

Output:
    ~/.nanobot/routing_targets.bin — binary ring buffer of routing decisions
    Format per record: [layer:u16, n_experts:u16, expert_indices:u16×8, expert_probs:f32×8, x_norm:f32×dim]
"""

import argparse
import mmap
import os
import struct
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np


# Ring buffer layout:
#   Header (32 bytes):
#     write_pos: u64       — next write offset (bytes, after header)
#     read_pos: u64        — next read offset (Rust consumer updates this)
#     n_layers: u32        — number of MoE layers in the model
#     dim: u32             — hidden dimension
#     k: u32               — experts per token
#     capacity: u32        — max records in buffer
#   Records (variable size each):
#     layer: u16
#     k: u16 (num experts this record)
#     expert_indices: u16 × k
#     expert_probs: f32 × k
#     x_norm: f32 × dim    (the input to the router, for training)

HEADER_SIZE = 32
ROUTING_FILE = os.path.expanduser("~/.nanobot/routing_targets.bin")


def record_size(dim, k):
    """Size of one routing record in bytes."""
    return 4 + k * 2 + k * 4 + dim * 4  # layer(2)+k(2) + indices + probs + x_norm


def init_buffer(path, n_layers, dim, k, capacity=4096):
    """Create or reset the shared ring buffer file."""
    rec_size = record_size(dim, k)
    total_size = HEADER_SIZE + capacity * rec_size

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        # Header
        f.write(struct.pack("<QQ III I",
                            0,          # write_pos
                            0,          # read_pos
                            n_layers,   # n_layers
                            dim,        # dim
                            k,          # experts_per_tok
                            capacity))  # capacity
        # Zero-fill records area
        f.write(b"\x00" * (capacity * rec_size))

    return total_size


def create_routing_hook(buf_path, layer_idx, dim, k):
    """Create a forward hook for one MoE gate layer."""
    rec_size = record_size(dim, k)

    def hook(module, args, output):
        """Called after gate forward: output = gate logits [batch, n_experts]."""
        # output shape: [1, n_experts] for single-token decode
        logits = output
        if isinstance(logits, tuple):
            logits = logits[0]

        # Convert to numpy for processing
        logits_np = np.array(logits.reshape(-1), dtype=np.float32)

        # Get input (x_norm) from args
        x = args[0] if args else None
        if x is None:
            return
        x_np = np.array(x.reshape(-1)[:dim], dtype=np.float32)

        # Top-k selection + softmax
        top_indices = np.argpartition(logits_np, -k)[-k:]
        top_indices = top_indices[np.argsort(-logits_np[top_indices])]
        top_logits = logits_np[top_indices]
        top_logits -= top_logits.max()
        exp_logits = np.exp(top_logits)
        probs = exp_logits / exp_logits.sum()

        # Write to ring buffer
        try:
            with open(buf_path, "r+b") as f:
                mm = mmap.mmap(f.fileno(), 0)

                # Read header
                write_pos = struct.unpack_from("<Q", mm, 0)[0]
                capacity = struct.unpack_from("<I", mm, 24)[0]

                # Compute write offset
                offset = HEADER_SIZE + (write_pos % capacity) * rec_size

                # Pack record
                record = struct.pack("<HH", layer_idx, k)
                record += struct.pack(f"<{k}H", *top_indices.astype(np.uint16))
                record += probs.astype(np.float32).tobytes()
                record += x_np.astype(np.float32).tobytes()

                mm[offset:offset + len(record)] = record

                # Update write_pos
                struct.pack_into("<Q", mm, 0, write_pos + 1)

                mm.close()
        except Exception as e:
            pass  # Don't crash inference on buffer write failure

    return hook


def patch_model(model, buf_path):
    """Find MoE gate modules and attach routing hooks."""
    dim = model.args.hidden_size
    k = getattr(model.args, "num_experts_per_tok", 8)
    n_layers = len(model.model.layers)

    # Find MoE layers
    moe_layers = []
    for i, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        # Check for gate attribute (MoE routing gate)
        gate = getattr(mlp, "gate", None)
        if gate is not None:
            moe_layers.append((i, gate))

    if not moe_layers:
        print("No MoE gate modules found — model may not be MoE")
        return 0

    # Init buffer
    total_size = init_buffer(buf_path, n_layers, dim, k)
    print(f"Routing buffer: {buf_path} ({total_size / 1024:.1f} KB)")
    print(f"  {len(moe_layers)} MoE layers, dim={dim}, k={k}")

    # Attach hooks
    for layer_idx, gate in moe_layers:
        hook = create_routing_hook(buf_path, layer_idx, dim, k)
        # MLX doesn't have register_forward_hook like PyTorch.
        # Instead, we monkey-patch the gate's __call__ method.
        original_call = gate.__call__

        def patched_call(self, x, _orig=original_call, _hook=hook, _layer=layer_idx):
            output = _orig(x)
            try:
                _hook(self, (x,), output)
            except Exception:
                pass
            return output

        import types
        gate.__call__ = types.MethodType(patched_call, gate)
        if layer_idx == moe_layers[0][0]:
            print(f"  Hooked layer {layer_idx} gate: {type(gate).__name__}")

    print(f"  {len(moe_layers)} routing hooks installed")
    return len(moe_layers)


def main():
    parser = argparse.ArgumentParser(description="MoE routing hook for oMLX")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Path to model directory")
    parser.add_argument("--output", type=str, default=ROUTING_FILE,
                        help="Path to routing buffer file")
    parser.add_argument("--monitor", action="store_true",
                        help="Monitor buffer and print stats")
    args = parser.parse_args()

    if args.monitor:
        # Monitor mode: just watch the buffer file
        print(f"Monitoring {args.output}...")
        last_write = 0
        while True:
            try:
                with open(args.output, "rb") as f:
                    data = f.read(HEADER_SIZE)
                    if len(data) < HEADER_SIZE:
                        time.sleep(1)
                        continue
                    write_pos, read_pos, n_layers, dim, k, capacity = struct.unpack(
                        "<QQ III I", data)
                    new = write_pos - last_write
                    if new > 0:
                        print(f"  write={write_pos} read={read_pos} "
                              f"pending={write_pos - read_pos} "
                              f"new={new} layers={n_layers} dim={dim} k={k}")
                    last_write = write_pos
            except FileNotFoundError:
                pass
            time.sleep(2)
    else:
        # Load model and patch
        from mlx_lm import load
        print(f"Loading {args.model_dir}...")
        model, tokenizer = load(args.model_dir)
        n_hooked = patch_model(model, args.output)
        if n_hooked == 0:
            print("ERROR: No MoE layers found")
            sys.exit(1)

        # Quick test: generate one token to verify hooks fire
        from mlx_lm import generate
        print("Verifying hooks with test generation...")
        generate(model, tokenizer, prompt="Hello", max_tokens=5, verbose=False)

        with open(args.output, "rb") as f:
            data = f.read(HEADER_SIZE)
            write_pos = struct.unpack_from("<Q", data, 0)[0]
        print(f"  Buffer has {write_pos} records after test — hooks working!")

        # Now start the server
        print("\nStarting mlx-lm server with routing hooks...")
        from mlx_lm.server import main as server_main
        server_main()


if __name__ == "__main__":
    main()
