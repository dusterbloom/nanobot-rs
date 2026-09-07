# Dense attention prefill — 2026-09-07

Committed to Higgs nightly as `d77352ef8` (`perf: bound M4 dense prefill scratch`). Installed and running: verified binary SHA256 `a9eb4019bc05e03bb1f8d5ffb10ab58713a6ad2f811b8d6249e6857081d37b58`; Metal library SHA256 `d4ec42fe79abd9d24922c84a6a160b98e9bba7ebed68dbed844981b1190eafb5`. Live installed server returned READY in 0.684s, normal memory pressure, no rejections, prompt capacity 47,104. User Nanobot process was not restarted. No push performed.

## Result

The improvement is memory headroom, not an established end-to-end speed gain. Matched AC serving used 45,003 prompt tokens and 35 completion tokens:

| Metric | Existing path | Query-block path |
|---|---:|---:|
| Total seconds | 294.107 | 292.698 |
| Time to first token, seconds | 290.540 | 289.466 |
| Sampled peak process footprint, GiB | 18.546 | 16.169 |
| New swapouts | 0 | 0 |
| Cached follow-up tokens | 45,037 | 45,037 |
| Follow-up seconds | 1.183 | 1.223 |

Both arms retrieved all three planted facts correctly; both follow-ups returned Neri. Pressure remained normal. Peak process memory fell 2.38 GiB. One matched pair does not establish a small latency improvement.

Final release binary, automatic selection with no diagnostic override: 296.824s total, 293.456s TTFT, 16.192 GiB sampled peak, zero new swapouts, normal pressure; all facts correct. Follow-up correct in 1.243s with 45,037 cached tokens. Historical swap usage existed; zero new swapouts does not mean the machine had no swap allocated.

## Implementation

Pinned mlx-rs b46423d lacks fused full-attention support for this FP32 D256 shape. The fallback's Q1024/H16/K32768 score array alone occupies 2 GiB. Higgs now materializes shared Q/K/V once and evaluates 128-query attention blocks sequentially, bounding temporary scratch. It keeps the model prefill chunk at 1024, FP32 precision and complete KV history.

Automatic selection is limited to base M4, FP32 D256, Q16/KV2, query length >128 and KV length >=16384. Causal blocks use aligned KV prefixes; arbitrary 2D masks keep all keys; general broadcast masks use existing MLX. Decode and canonical scheduling are unchanged. No permanent experimental selector, custom kernel or ANE offload was added.

## Validation and rejected approaches

Release regression failed on old scratch allocation (563,283,808 bytes against a 384 MiB budget), then passed with the change. Two new tests cover bounded scratch, offset causality, tail blocks, noncausal future keys and broadcast masks. Existing canonical full-attention, canonical TurboQuant and base-M4 regressions passed; release build passed. Independent review found no actionable issues.

Graph impact before edits identified two direct indexed callers/test flows. Refreshed pre-commit detect_changes reported CRITICAL risk and 16 affected flows; warning communicated. Detect result was not partial/truncated. The underlying process index warns of capped global flow enumeration, so it is not proof of exhaustive coverage. Only qwen3_next.rs and docs/models.md were committed; unrelated README, database and ANE docs were preserved.

Three isolated FP32 fused-kernel variants passed numerical checks but failed speed tests; none shipped. Standalone query blocking won ~16–18%, but that did not establish serving speed improvement. V1 32K serving reduced memory but slowed TTFT ~1.4%; V2's positive 32K pair had control drift. The valid 45K AC comparison above is the deployment evidence. An earlier 45K battery run was invalidated by a power-source transition and excluded.

Raw synthetic measurement JSONs, final patch, release logs, graph analysis, review and installation receipt accompany this file. Larger speed gains remain unproven; no roofline or ANE performance claim is made.
