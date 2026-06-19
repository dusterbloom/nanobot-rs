# Fused Backward Attention GQA — Implementation Plan

## Goal
Replace 4 ANE dispatches (wot_bwd, sdpa_bwd1, sdpa_bwd2, qkv_bwd) + CPU ops
(gate backward, RoPE backward) with 1 fused MIL kernel per layer.
Saves ~96 dispatches/step (4 per layer × 24 layers for Qwen3.5-0.8B).

## Input Layout [1, in_ch, 1, seq] fp32
| Field      | Channels   | Description                    |
|------------|-----------|--------------------------------|
| dx2        | dim       | Incoming gradient              |
| Q_rot      | attn_dim  | Post-RoPE Q from forward       |
| K_rot      | kv_dim    | Post-RoPE K from forward       |
| V          | kv_dim    | V from forward                 |
| pre_gate   | attn_dim  | a_out before gate (if gate)    |
| gate_raw   | attn_dim  | Raw gate values (if gate)      |

Total: dim + ad + 2*kvd (no gate) or dim + 3*ad + 2*kvd (gate)

## Output: [1, dim, 1, seq] fp32 — dx_attn

## BLOBFILE Weights (SAME orientation as forward)
- Wq [1,1,dim,qpd], Wk [1,1,dim,kvd], Wv [1,1,dim,kvd], Wo [1,1,ad,dim]
- rope_cos [1,1,S,hd/2], rope_sin [1,1,S,hd/2], mask [1,1,S,S]
- Backward uses `matmul(transpose_y=True)` to get W^T effect

## MIL Phases (~95 ops total)
1. **Cast + slice** input channels
2. **Wo^T projection**: da = dx2 @ Wo^T → [1,1,S,ad]
3. **Gate backward** (if gate): sigmoid, d_attn = da*sig, d_gate = da*pre_gate*sig*(1-sig)
4. **SDPA backward**: recompute probs, dV=A^T@dO, dP=dO@V^T, softmax_bwd, dQ=scale*dS@K, dK=scale*dS^T@Q
   - GQA: [kvH,hpg,S,S] batch form, reduce_mean*hpg for K/V group aggregation
   - reduce_sum trick: reduce_mean(x,axis)*dim_size
5. **RoPE backward**: R^T = [[cos,sin],[-sin,cos]] (swap sin signs vs forward)
6. **Merge Q+gate**: concat(dq_pre, d_gate) per head if gate
7. **QKV^T projections**: dx_q = dq@Wq^T, dx_k = dk@Wk^T, dx_v = dv@Wv^T
8. **Sum + output**: dx_attn = dx_q + dx_k + dx_v → [1,dim,1,S] fp32

## ANE Workarounds
- `reduce_sum` → `reduce_mean * dim_size`
- `rsqrt` → `pow(x, -0.5)` (not needed here, only in QK-norm)

## Implementation Steps
1. `gen_fused_attn_gqa_bwd()` in ane_mil.rs + compile test
2. `pack_fused_attn_gqa_bwd_input()` in ane_weights.rs
3. `bwd_fused_attn_gqa_kernels` field + `prime_attn_bwd_kernels()` in PrePackedWeights
4. `mha_backward_fused_dx_attn()` fast path in backward loop
5. Numerical test: compare against multi-dispatch `mha_backward_ane_dx_attn`

## Reference Functions
- Forward: `gen_fused_attn_gqa_fwd` at ane_mil.rs:1839
- CPU backward: `mha_backward_cpu_dx_attn` at ane_backward.rs:1074
- ANE backward: `mha_backward_ane_dx_attn` at ane_backward.rs:1153
- Forward eval: `eval_fwd_fused_attn_gqa` at ane_weights.rs:2614
- Forward prime: `prime_attn_kernels` at ane_weights.rs:2529
