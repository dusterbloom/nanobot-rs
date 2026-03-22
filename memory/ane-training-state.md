# ANE Training State (2026-03-18, VERIFIED)

## Honest Assessment: CPU Training with ANE-Accelerated FFN

ANE utilization is ~1.7% (confirmed via mactop during 35B training).
CPU does ~95% of the work. ANE handles fused FFN matmuls only.

## Yield Guard: REMOVED (commit b39da4d)

Benchmark proof (35B-A3B-3bit, real experience.db traces, oMLX inference):
- Without guard: inference degrades 10% (24→27ms/tok)
- WITH guard: inference degrades 3,461% (24→858ms/tok)
- Guard was evict→sleep(200ms)→reprime cycle that destroyed inference 35x

## What's ON ANE (per layer)

### Forward (~2 dispatches/layer for Qwen3.5)
- FFN: W1+W3+SiLU+gate+W2 — **ANE** (1 fused dispatch, prepacked)
- SDPA core: Q@K^T→softmax→@V — **ANE** (1 dispatch, only when seq>=192)

### Backward (~2-8 dispatches/layer)
- FFN bwd W2^T — **ANE** (prepacked)
- FFN bwd W1^T+W3^T — **ANE** (prepacked)
- LoRA W2 weight grads — **ANE** (3 dispatches via AdapterWeightGradKernels)
- LoRA Wo weight grads — **ANE** (3 dispatches via AdapterWeightGradKernels)

## What's CPU (for Qwen3.5 GQA+gate)
- All RMSNorm (4 per layer), embedding, residuals
- QKV projections, Q/gate split, QK-norm, RoPE, sigmoid gate, O projection
- Entire attention backward
- SiLU backward, optimizer (AdamW), cross-entropy loss, dW accumulation

## Path to MAX ANE (priority order)

1. **Wire gen_fused_attn_gqa_fwd** (ane_mil.rs:1837) — moves QKV+RoPE+SDPA+gate+O
   from CPU to 1 ANE dispatch. Already written+tested. Biggest single win.
2. **Add RMSNorm into fused FFN** — pow(x,-0.5)+reduce_mean work on ANE (proven)
3. **Fused attention backward for GQA** — new kernel needed
4. **Target: 4 dispatches/layer** (attn_fwd, ffn_fwd, attn_bwd, ffn_bwd)
   Should push ANE from 1.7% to 20-40%

## Training Details
- Data: real user conversations from ~/.nanobot/experience.db (270 experiences)
- Method: LoRA (rank 32, alpha 32) on Wo + W2 adapters
- Optimizer: AdamW, lr=5e-4
- Loss: cross-entropy on assistant tokens only (user/system masked with IGNORE_LABEL)
- Goal: on-device personalization — learn user's tool-use patterns + response style
