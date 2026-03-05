#!/usr/bin/env python3
"""Generate reference tensors from Python mlx_lm GatedDeltaNet for numerical comparison.

Runs a single linear_attn layer (layer 0) on a fixed input and saves:
  - input tensor (random but seeded)
  - output tensor from full GatedDeltaNet.__call__
  - intermediate: q, k, v after conv+split+norm, g, beta, recurrence output

Usage:
    python3 tests/gdn_reference.py
    # Creates tests/gdn_reference_tensors.npz
"""

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from pathlib import Path

MODEL_PATH = Path.home() / ".cache/lm-studio/models/mlx-community/Qwen3.5-2B-MLX-8bit"


def main():
    from mlx_lm.utils import load_model

    # Load full model to get properly initialized weights
    # strict=False to skip vision tower weights not in text model
    model, tokenizer = load_model(MODEL_PATH, strict=False)

    # Get layer 0 linear_attn module
    layer0 = model.layers[0]
    gdn = layer0.linear_attn  # GatedDeltaNet instance

    # Fixed input: [1, 4, 2048] — batch=1, seq=4, hidden=2048
    mx.random.seed(42)
    x = mx.random.normal((1, 4, 2048)).astype(mx.bfloat16)

    # Run the full GatedDeltaNet forward (training mode = ops path, no kernel)
    gdn.train()  # Force ops path (no metal kernel)
    output = gdn(x, mask=None, cache=None)
    mx.eval(output)

    # Now extract intermediates by replaying the forward manually
    B, S, _ = x.shape

    qkv = gdn.in_proj_qkv(x)
    z = gdn.in_proj_z(x).reshape(B, S, gdn.num_v_heads, gdn.head_v_dim)
    b = gdn.in_proj_b(x)
    a = gdn.in_proj_a(x)

    # Conv1d (causal)
    conv_state = mx.zeros((B, gdn.conv_kernel_size - 1, gdn.conv_dim), dtype=x.dtype)
    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    conv_out = nn.silu(gdn.conv1d(conv_input))

    # Split
    q_raw, k_raw, v_raw = [
        t.reshape(B, S, h, d)
        for t, h, d in zip(
            mx.split(conv_out, [gdn.key_dim, 2 * gdn.key_dim], -1),
            [gdn.num_k_heads, gdn.num_k_heads, gdn.num_v_heads],
            [gdn.head_k_dim, gdn.head_k_dim, gdn.head_v_dim],
        )
    ]

    # QK norm
    inv_scale = gdn.head_k_dim ** -0.5
    q = (inv_scale**2) * mx.fast.rms_norm(q_raw, None, 1e-6)
    k = inv_scale * mx.fast.rms_norm(k_raw, None, 1e-6)

    # Gated delta update (ops path)
    from mlx_lm.models.gated_delta import gated_delta_ops, compute_g

    beta = mx.sigmoid(b)
    g = compute_g(gdn.A_log, a, gdn.dt_bias)
    state = mx.zeros((B, gdn.num_v_heads, gdn.head_v_dim, gdn.head_k_dim), dtype=q.dtype)
    recurrence_out, final_state = gated_delta_ops(q, k, v_raw, g, beta, state, None)

    # Norm + gate
    normed = gdn.norm(recurrence_out, z)
    final_out = gdn.out_proj(normed.reshape(B, S, -1))

    # Evaluate all
    mx.eval(x, output, q, k, v_raw, g, beta, recurrence_out, final_state, final_out, a, z)

    # Verify our manual replay matches
    diff = mx.abs(output - final_out).max().item()
    print(f"Manual replay vs __call__ max diff: {diff:.2e}")
    assert diff < 1e-3, f"Replay mismatch: {diff}"

    # Save as float32 numpy arrays (both npz and raw binary for Rust)
    out_path = Path(__file__).parent / "gdn_reference_tensors.npz"
    tensors = {
        "input": np.array(x.astype(mx.float32)),
        "output": np.array(output.astype(mx.float32)),
        "q_normed": np.array(q.astype(mx.float32)),
        "k_normed": np.array(k.astype(mx.float32)),
        "v": np.array(v_raw.astype(mx.float32)),
        "g": np.array(g.astype(mx.float32)),
        "beta": np.array(beta.astype(mx.float32)),
        "recurrence_out": np.array(recurrence_out.astype(mx.float32)),
        "final_state": np.array(final_state.astype(mx.float32)),
        "z_gate": np.array(z.astype(mx.float32)),
        "a_raw": np.array(a.astype(mx.float32)),
        "A_log": np.array(gdn.A_log.astype(mx.float32)),
        "dt_bias": np.array(gdn.dt_bias.astype(mx.float32)),
    }
    np.savez(out_path, **tensors)

    # Also save raw f32 binary for Rust tests (no numpy dependency needed)
    raw_dir = Path(__file__).parent / "gdn_reference_raw"
    raw_dir.mkdir(exist_ok=True)
    manifest = {}
    for name, arr in tensors.items():
        arr_f32 = arr.astype(np.float32)
        (raw_dir / f"{name}.bin").write_bytes(arr_f32.tobytes())
        manifest[name] = list(arr_f32.shape)
    import json
    (raw_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"Saved reference tensors to {out_path}")
    print(f"Saved raw f32 binaries to {raw_dir}/")
    for name, arr in tensors.items():
        print(f"  {name}: {arr.shape}")


if __name__ == "__main__":
    main()
