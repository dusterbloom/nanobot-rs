# Higgs / Qwen3-Next 16K→32K Bottleneck Analysis (final)

Commit `a649e017a` (after `d278259de`), exe SHA `fbf8b3e8…697a2`. Base M4 32 GB, native Escha W2, AC, 1024-tok chunks, packed GEMM default, FP32 execution unchanged.
**Evidence class:** all component timings are host-side eval-barrier deltas sampled only through the first FA layer, extrapolated ×30 GDN + ×10 FA. They are **not** wall-time shares or utilization; **no GPU counters** exist in this bundle. 16K ran on AC; 32K on battery (LPM off, normal pressure, 17.17 GiB peak, zero swapouts) — **cross-run timing comparisons are power-confounded**; only same-run early/late chunk deltas cleanly isolate context growth.

## 1. Measured wall truth (not extrapolation)

| | 16K (AC) | 32K (battery) |
|---|---|---|
| prompt / completion tok | 16,008 / 128 | 32,010 / 128 |
| TTFT | 76.53 s | 184.59 s |
| prefill (to last full mark) | 75.17 s | 182.56 s |
| decode | 65.1 ms/tok (8.34 s) | 97.6 ms/tok (12.49 s) |
| total | 84.87 s | 197.08 s |

32K TTFT = **2.41×** 16K for 2.0× tokens — superlinear, FA-driven (ratio measured; mechanism hypothesis; power confound unresolved).

**Within-32K-run growth (valid regardless of power):** chunk wall median 4.51 s (chunks 3–8, ctx 2–7 K) → 7.25 s (last 8 full chunks, ctx 24–31 K) ≈ **+120 ms per 1 K context**. Sampled FA attn 72.0 → 348.9 ms/layer = **12.0 ms/K/FA-layer** (×10 ≈ 120 ms/K — matches the wall slope). Flat terms drift only +2–10% (gdn_attn 43.1→46.8, gdn_mlp 62.4→65.1, fa_mlp 64.9→71.4 ms), likely power/thermal drift, not context. The 16K AC run's slope (~70 ms/K) is **not** comparable to 32K's (~120 ms/K) — confounded.

Est-vs-wall consistency: 16K Σest ≈ 1.02× prefill wall but its context slope was 1.44× inflated; the 8-row canonical block est overestimates 4× (small-L barrier overhead). Decode est/wall: 1.15× (16K), 1.03× (32K).

## 2. Component attribution (extrapolated medians; NOT wall shares)

| share | 16K prefill | 32K early (3–8) | 32K late (8) | 32K decode | 16K decode |
|---|---|---|---|---|---|
| FA attn | 22.5% | 15.9% | **46.1%** | **47.9%** | 34.4% |
| GDN attn | 26.8% | 28.5% | 18.6% | 24.2% | 29.2% |
| MLP (GDN+FA) | 50.7% | **55.6%** | 35.3% | 27.9% | 36.3% |

MLP is the largest component early/overall (43.1% across all 32K chunks); FA attention is the only strongly growing term and dominates late chunks and decode.

## 3. What "expert MLP" timing includes

Inseparable inside `mlp.forward`: (1) MoE router — dense FP projection + top-k/argsort/any sort (none excerpted); (2) per-row `su`/`sv` scale gathers; (3) two blockwise Hadamards; (4) packed gather-QGEMM launches (gate_up, down) with **in-kernel** trellis decode (bit-unpack → `(h·mul+add)&mask ^ xor` → half2 sum → f32 stage → simdgroup FP32 MMA) — even a GPU-side "GEMM" number would include decode; (5) activation/shared expert; (6) the barrier sync. Routing vs Hadamard vs GEMM **cannot be split** from these timings.

## 4. Attention: append vs kernels, context scaling

- Decode TurboQuant per FA layer: append(quantize)/attn = 0.465/1.66 ms (16K) → 0.78/4.00 ms (32K). Append share fell 22%→16%; its 1.68× growth for 2× KV is **power-confounded** — O(KV) append unproven (MLX donation unproven too). Decode FA attn grew 1.86× for 2× KV — consistent with linear (also confounded).
- Append spikes (≤2.96 ms) co-occur with attn spikes **in the same token** → transient interference (allocator/command-buffer), not steady whole-KV copy.
- Prefill cache is dense fp16 (`cache.rs:1171`); quantization deferred to activation. The prefill→TTFT gap (16K 1.35 s, 32K 2.03 s) still has ~0.7–1.0 s unattributed: suspected one-time bulk TQ activation + host plumbing (hypothesis).
- GDN attn is flat within runs (state-based); 18.6–29% of attribution.

## 5. Shapes/dtypes supported by excerpts

- Attention: x `[B,L,hidden]`; q_proj doubled → Q `[B,L,H,D]` + gate; K/V `[B,L,H_kv,D]`→`[B,H,L,D]`; per-head RMSNorm; RoPE manual (prefill, length-independent) vs `mlx_fast_rope` (L=1); sigmoid gating; canonical rows **L∈[1,8]**; FA mask = `Causal` or materialized `Array` `create_causal_mask(T, kv_offset)`; GDN unmasked. Dense KV strictly `[B,H,T,D]`, fp16 during prefill; TQ restore = codes/norms/gammas (dtypes not excerpted); MLA latent `[1,1,T, kv_lora_rank+rope_dim]` (separate path).
- Escha: input `[rows,in]` cast **f32**; `su`/`sv` gathers; `had_blockwise` pre/post; QGEMM for `rows > GATHER_QMV_MAX_ROWS` (const not excerpted), QMV below. Kernel: `xh` f32 `[rows,TK·16]`; code u32 packed, `WORDS=8·K`, expert stride `TK·TN·16K`; `eids` u32, consecutive-run grouping (sorted input assumed); `cb` i32[5]; **K∈[1,8]**; output **f32** `[rows_pad,TN·16]`; `(NT,BM,XP)`=(128,32,40) default or `(256,64,72)` via `HIGGS_ESCHA_QGEMM_BM=64`; f32 accumulators. Non-expert: int8 `[out,in]`×f32 `[out]` scale; routers FP. Scratch f16 path = reference only.

## 6. Top three next experiments (whole-request gain vs risk)

**E1 — Split the MLP interior + run the deferred BM=64 real-router A/B.** Instrument `mlp.forward` at the already-sampled layers only: time router+topk(+sort); su/sv+Hadamard; gate_up QGEMM; down+sv+Hadamard (same barrier pattern, ~4 extra syncs/sample). Then `HIGGS_ESCHA_QGEMM_BM=64` at 32K (env-only, revert by unset) — the A/B `metal_kernel.rs:1198` explicitly defers. *Ceiling:* MLP = 43–56% of prefill extrapolation, 28% of decode; a 10% GEMM win ⇒ ~3–5% whole request; if routing dominates, later effort redirects. *Risk:* ~zero; quality gate = identical greedy output.

**E2 — Replace the materialized prefill causal mask with native SDPA causal if semantics allow.** Continuation chunks build a `[L,KV]` f32 mask per FA layer (~131 MB write+read per layer-chunk at 32K), and SDPA may lose its fused causal path given an explicit array. Verify MLX causal ≡ bottom-right-aligned causal for `kv_offset>0`, then A/B: chunk wall + greedy parity. *Ceiling:* FA attn is 46% of late-chunk extrapolation; a 10–30% cut ⇒ ~2–8% whole request at 32K, growing with context. *Risk:* low-medium — mask-form-only change, semantics must be proven identical.

**E3 — Fused single-pass TurboQuant decode attention (+append-spike tracing).** Add one tracing event at TQ growth/activation (cheaply tests the 0.78 ms append trend and the TTFT gap). Prototype `decode_scores → online softmax → decode_values` as one kernel: single KV pass, no materialized weights array. *Ceiling:* decode FA attn = 48% of a 32K token; halving ⇒ ~25% decode = ~1.5% of the 32K request — the dominant lever for long-generation workloads. *Risk:* medium — new Metal kernel, softmax accumulation order must be quality-gated; dense path untouched.

**Rejected/blocked:** output Hadamard fusion (~1%, worse decode); ANE offload (no evidence); f16 SDPA (blocked by unchanged-FP32 gate; arithmetic ceiling ≈ 10% of prefill if ever relaxed). **Guardrails:** quality A/Bs on identical prompt/harness; 45K capacity (E3 tracing watches TQ growth there); cancellation behavior untouched.