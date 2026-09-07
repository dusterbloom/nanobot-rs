# Reviewed GLM profiling findings — 2026-09-07

GLM-5.3-Flash analyzed curated Higgs source and 16K/32K timing artifacts through OpenCode. Its original report is preserved in GLM-RAW.md; the corrections below supersede conflicting claims there. No production optimization was implemented in this profiling follow-up.

## Measured result

The fresh 32,010-token prompt with 128 output tokens completed on battery, Low Power Mode off, as explicitly authorized: 197.08 seconds total, 184.59 seconds to first token, peak process footprint 17.17 GiB, normal memory pressure, zero new swapouts. The installed Higgs server was restored afterward. Prior 16K measurements ran on AC, so cross-run speed ratios do not isolate context effects.

| Sampled component, per layer | Early 32K prefill chunks | Late 32K prefill chunks |
| --- | ---: | ---: |
| GDN attention | 43.15 ms | 46.76 ms |
| GDN expert MLP | 62.42 ms | 65.13 ms |
| Full attention | 72.00 ms | 348.86 ms |
| Full-attention-layer expert MLP | 64.87 ms | 71.43 ms |

Decode medians per full-attention layer: cache append 0.780 ms, attention kernels 4.004 ms. Full attention grows sharply within the prefill run; MLP timing changes much less. Attention kernels exceed append time during decode. These host-side eval-barrier measurements sample leading layers; extrapolated percentages are diagnostic attribution, not actual wall-time shares or GPU utilization. Same-run comparisons reduce power-source confounding but cannot exclude thermal/time drift. No inference roofline saturation was established.

## Corrections from source review

1. `PROFILE-TQ` is a diagnostic label, not proof of TurboQuant execution. In `qwen3_next.rs:4652`, profiling is enabled for single-token decoding regardless of cache variant; `attend_one_query` dispatches dense or TurboQuant. Cache mode defaults Off, the default activation threshold is 100,000 tokens, and no relevant override was found in the loaded configuration. The measurements do not justify prioritizing TurboQuant fusion for this workload.
2. Dense cache allocation preserves `keys.dtype()` / `values.dtype()` (`cache.rs:1092–1103`). The comment mentioning fp16 does not establish the actual dtype. Reject the raw report's “strictly fp16” claim and its unsupported bulk-TurboQuant-activation explanation for the TTFT gap.
3. `create_causal_mask` returns a boolean comparison (`utils.rs:228–236`). The continuation mask is created outside the layer loop and shared. Reject the claimed 131 MB float-mask allocation per attention layer. A native causal-mask experiment may still be useful, but its gain is unmeasured and offset semantics must be proven.
4. BM=64 is a reversible diagnostic experiment, not zero risk. Require existing numerical oracle checks, matched behavior probes, and 45K memory/recovery validation. Identical output on a single greedy prompt is insufficient.
5. Unverified wall-slope correlations, projected optimization gains, and append-spike explanations in the raw report remain hypotheses.

## Next experiments, in order

1. Split sampled expert MLP timings into routing, transforms, gate/up GEMM, and down GEMM. Compare BM=32 and BM=64 with actual routed experts. Use diagnostic timers to locate cost, then uninstrumented matched requests to establish any speed gain.
2. Prove native causal masking matches offset continuation masking, including unequal query/key lengths, before comparing the two forms at long context. Retain recurrent/cache correctness gates.
3. Record the active cache variant and dtype explicitly, then profile the active dense SDPA decode kernels with Metal counters. Choose a kernel change only after identifying bandwidth, compute, or dispatch limitations. TurboQuant fusion is not supported as the next production change by this capture.

Preserve 45K usable context, zero new swapouts in matched runs, cancellation/recovery behavior, and numerical/behavior gates. Compare identical prompts and binaries under matched power conditions; report time-to-first-token and decode separately.

Committed on Higgs nightly: `d278259de` (capacity/watchdog recovery), `a649e017a` (measured M4 GEMM default and profiler corrections). Neither was pushed. Installed binary SHA256: `fbf8b3e8b9f3450c7f4caa1fb8f1f32baf39f307a69d4edb20ff8eb0e87697a2`.
