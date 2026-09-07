# GLM / HY4 profiling continuation

GLM completed via verified zai-coding-plan/glm-5.3-flash. HY4 was requested and diagnostics sharing explicitly approved, but OpenRouter returned402 insufficient credits before any model answer; its tooling task went to a local fast_scan sub-agent. A routine_worker implemented the isolated GLM diagnostic; primary reviewed, compiled and ran it. No new production change or commit.

## Results

All values milliseconds per32K attention call (Q1024, FP32 D256 Q16/KV2), medians of6round medians. Each round3timed reps, interleaved controls. Compare arms within their own run.

| Experiment | Installed query128 | Existing reg-causal | Diagnostic |
|---|---:|---:|---:|
| Unused mask template float→bool | 291.57 | 1127.14 | 1139.60 |
| GLM causal-tail loop split | 287.96 | 1092.46 | 1135.45 |

Neither diagnostic improves the target. In the peel run query128 varied286.72–290.66ms, existing causal1040.00–1155.43ms, peel1093.53–1183.88ms across rounds. No candidate is close to production promotion. Correctness gates passed normal and amplified Q/K, causal/offset-mask equivalence, and31/1055 tail shape. Same numerical gates as prior experiment; no post-result relaxation.

The unused mask template test changes only the no-array mask template selection while keeping has_mask=false and native causality. The peeled kernel separates the unmasked bulk from masked tail, with one shared body included at compile time. Primary source comparison verified body identity except the causal condition; safe Qtiles handoff, KV loader order and accumulation sequence retained. No production shader/source/library was overwritten.

GLM proposed the peel but overclaimed that success/failure would uniquely identify compiler scheduling or function-constant specialization. We reject that inference: the negative result only rules out this source transformation as a remedy. Compiler floating-point transformations also prevent claiming guaranteed bit identity solely from source equivalence. Actual register/spill/limiter evidence remains unavailable through current public tools.

Local tooling agent verified metal -S -emit-llvm can expose intermediate structure only; causal/explicit function-constant specialization occurs later. It cannot prove final physical register allocation, occupancy or bank conflicts. No installed metal-objdump/metal-nm/metal-dsymutil found. Public device counters remain timestamp-only; valid GPU captures from prior step remain available for Xcode inspection.

Every benchmark was isolated and serialized with idle Higgs, with sampled normal-pressure/no-new-swapout/stable-power guards. First masktype build failed due an edit placed in the wrong dispatcher; corrected to the intended full-attention function, then built and tested. Higgs restoration is recorded in run logs and final receipt. Installed d77352ef8 remains the banked memory-saving build.

Final restoration initially failed at model-load capacity boundary; OS pressure subsequently normal, manual retry succeeded. Live READY and installed hash verified in restore-receipt.json. This recurring startup-policy issue is separate from shader performance and remains unfixed by these experiments.
