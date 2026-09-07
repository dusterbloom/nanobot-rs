# Verified adversarial GLM review

Requested provider/model verified from exported session: zai-coding-plan/glm-5.3-flash, ses_f849d45e6ffeGUz3sV7ikquCi1. User explicitly approved curated source/evidence sharing. Raw authored report is GLM-FUSED-RAW.md. OpenCode rejected its file write despite the intended allow rule; the complete report was recovered from the tool event. No production change or new GPU benchmark was made.

## Accepted findings

- Existing bracket timing drift is larger than the first fused candidate's ~6% observed deficit. This weakens the precision of the speed-loss estimate; it does not demonstrate a hidden win.
- The benchmark uses only three timed repetitions per arm after one discarded iteration. A stronger comparison must interleave repeated controls/candidates and include the actual shipped explicit-mask/query128 path in the same process and conditions.
- The review found no existing correctness defect or compiler-flag mismatch. Random-normal/tail tests remain narrower than full model validation.
- Larger key tiles are a plausible experiment to reduce online-softmax/barrier iterations, with resource tradeoffs to measure.

## Rejected/corrected claims

1. E1 is invalid as written: aliasing Q and KV storage while retaining upstream per-iteration loads of Q overwrites Q with K/V before those loads. Existing register-Q avoids this by loading all Q into registers first. Never implement E1's proposed combination.
2. E2 is already the long-context serving path. qwen3_next.rs:8861 builds an explicit offset causal mask after the first chunk; dense_prefill_attention:4453 slices its query rows while retaining full K/V. Calling the causal diagnostic the shipped baseline was incorrect. The curated patch did not include this upstream caller; insufficient context contributed to the mistake.
3. Whole-request ~5% ceiling is unsupported. The report multiplies per-attention timing by chunks but omits ten full-attention layers (40 layers, interval4); timings also change across chunks. A valid bound needs same-run sums across layers/chunks, not one late-context microbenchmark.
4. Register count, hardware bank-conflict, occupancy and spilling conclusions are hypotheses without compiler resource reports/GPU counters. Logical fragment storage is not a physical-register measurement. Short-shape correctness cannot prove long-shape slowdown is occupancy.
5. Expanding KV heads to16 does not prove eightfold DRAM traffic; cache reuse and allocation alter the experiment. Equal timing cannot rule out every GQA or split-K redesign.
6. Allocator-cache pressure plausibly disadvantages the fallback, but exact per-iteration physical reallocation and graph-node counts were not traced; those claims are not established.
7. Fixed historical timing gates and cross-session percentages are unsuitable acceptance criteria. Compare matched contemporary arms, report dispersion, then verify serving.

## Recommended next discriminator

First establish a same-run FP32 benchmark: old fallback, current explicit-mask query128, original16x8 fused, then a valid16x16 candidate. Materialize identical Q/K/V; warm each arm; interleave at least6 brackets at 8K/32K/45K plus offset/tail correctness. Track pressure/new swapouts, allocator peak and timing dispersion. A candidate must beat current query128 consistently before serving integration; no fixed historical millisecond threshold.

For16x16, retain Q/KV separately with a layout fitting32KiB (zero-padding arithmetic gives32768 bytes but bank behavior must be measured), or use the existing safe register-Q handoff at smaller BQ. These are untested alternatives, not claimed fixes. Use impact analysis before edits, capture resource/limiter evidence if available, and retain FP32 numerical gates. Follow any microbenchmark win with matched45K fact retrieval/cached follow-up and memory checks. Installed d77352ef8 remains the control.
