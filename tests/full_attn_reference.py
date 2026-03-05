#!/usr/bin/env python3
"""Generate reference tensors from Python mlx_lm full attention layer for numerical comparison.

Runs layers 0-2 (linear_attn) to get input to layer 3 (first full_attn),
then runs layer 3's self_attn step-by-step saving intermediates.

Usage:
    python3 tests/full_attn_reference.py
    # Creates tests/full_attn_reference_raw/
"""

import numpy as np
import mlx.core as mx
from pathlib import Path

MODEL_PATH = Path.home() / ".cache/lm-studio/models/mlx-community/Qwen3.5-2B-MLX-8bit"


def main():
    from mlx_lm.utils import load_model

    model, tokenizer = load_model(MODEL_PATH, strict=False)

    # Fixed input: [1, 4, 2048]
    mx.random.seed(42)
    x = mx.random.normal((1, 4, 2048)).astype(mx.bfloat16)

    # Run layers 0-2 to get residual stream input to layer 3
    h = x
    for i in range(3):
        h = model.layers[i](h, mask=None, cache=None)
    mx.eval(h)

    # Layer 3 is the first full_attn layer
    layer3 = model.layers[3]
    attn = layer3.self_attn

    # Apply input_layernorm to get self_attn input
    attn_input = layer3.input_layernorm(h)
    mx.eval(attn_input)

    # Run full self_attn forward for reference output (with causal mask, matching production)
    full_output = attn(attn_input, mask="causal", cache=None)
    mx.eval(full_output)

    # Now replay self_attn step-by-step to capture intermediates
    B, S, _ = attn_input.shape
    n_heads = attn.num_attention_heads
    n_kv_heads = attn.num_key_value_heads
    head_dim = attn.head_dim

    q_raw = attn.q_proj(attn_input)
    k_raw = attn.k_proj(attn_input)
    v_raw = attn.v_proj(attn_input)

    # Per-head reshape then split Q and gate (matches Python __call__ exactly)
    q_reshaped = q_raw.reshape(B, S, n_heads, -1)        # [B, S, 8, 512]
    queries, gate = mx.split(q_reshaped, 2, axis=-1)      # [B, S, 8, 256] each
    gate_flat = gate.reshape(B, S, -1)                     # [B, S, 2048]

    # QK norm
    q_normed = attn.q_norm(queries)                        # [B, S, 8, 256]
    k_4d = k_raw.reshape(B, S, n_kv_heads, -1)            # [B, S, 2, 256]
    k_normed = attn.k_norm(k_4d)                           # [B, S, 2, 256]

    # Transpose for RoPE: [B, H, S, D]
    q_t = q_normed.transpose(0, 2, 1, 3)
    k_t = k_normed.transpose(0, 2, 1, 3)
    v_t = v_raw.reshape(B, S, n_kv_heads, -1).transpose(0, 2, 1, 3)

    # RoPE
    q_roped = attn.rope(q_t)
    k_roped = attn.rope(k_t)

    # SDPA with causal masking (matching production path)
    sdpa_out = mx.fast.scaled_dot_product_attention(
        q_roped, k_roped, v_t, scale=attn.scale, mask="causal"
    )

    # Transpose back and reshape
    attn_out = sdpa_out.transpose(0, 2, 1, 3).reshape(B, S, -1)

    # Apply output gate + o_proj
    replay_output = attn.o_proj(attn_out * mx.sigmoid(gate_flat))

    mx.eval(q_raw, k_raw, v_raw, queries, gate_flat, q_normed, k_normed,
            q_roped, k_roped, sdpa_out, attn_out, replay_output)

    # Verify replay matches
    diff = mx.abs(full_output - replay_output).max().item()
    print(f"Manual replay vs __call__ max diff: {diff:.2e}")
    assert diff < 1e-3, f"Replay mismatch: {diff}"

    # Save tensors
    tensors = {
        "attn_input": np.array(attn_input.astype(mx.float32)),
        "attn_output": np.array(full_output.astype(mx.float32)),
        "q_after_split": np.array(queries.astype(mx.float32)),
        "gate": np.array(gate_flat.astype(mx.float32)),
        "q_after_norm": np.array(q_normed.astype(mx.float32)),
        "k_after_norm": np.array(k_normed.astype(mx.float32)),
        "q_after_rope": np.array(q_roped.astype(mx.float32)),
        "k_after_rope": np.array(k_roped.astype(mx.float32)),
        "sdpa_out": np.array(sdpa_out.astype(mx.float32)),
        "gated_attn": np.array((attn_out * mx.sigmoid(gate_flat)).astype(mx.float32)),
    }

    raw_dir = Path(__file__).parent / "full_attn_reference_raw"
    raw_dir.mkdir(exist_ok=True)
    manifest = {}
    for name, arr in tensors.items():
        arr_f32 = arr.astype(np.float32)
        (raw_dir / f"{name}.bin").write_bytes(arr_f32.tobytes())
        manifest[name] = list(arr_f32.shape)
    import json
    (raw_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"Saved raw f32 binaries to {raw_dir}/")
    for name, arr in tensors.items():
        print(f"  {name}: {arr.shape}")


if __name__ == "__main__":
    main()
