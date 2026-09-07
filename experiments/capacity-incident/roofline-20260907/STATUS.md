# Higgs M4 performance validation — 2026-09-07

Installed and running locally from `~/.local/bin/higgs`; six reviewed performance/docs files integrated into the Higgs `nightly` working tree. Committed locally on nightly: `d278259de` (capacity/watchdog recovery) and `a649e017a` (M4 GEMM/profiling). No public release or push was made. Existing recovery changes and unrelated edits were preserved; Nanobot PID 53654 was not restarted.

Native Escha prefill now defaults to packed trellis GEMM on base M4, with 1024-token chunks. Other hardware retains scratch. Explicit `HIGGS_ESCHA_TRELLIS_GEMM=0` selects scratch. Execution and capacity identity share the same selector and cached hardware discovery.

| AC workload | Scratch, chunk 1024 | Packed GEMM, chunk 1024 |
|---|---:|---:|
| 16K prompt, 128 output, repeated candidate comparison | 89.83–93.94 s | 81.92–82.58 s |
| 45K retained session, matched candidate pair | 307.81 s | 286.66 s |
| Final release binary, 45K with no performance overrides | — | 293.46 s |
| Final release follow-up at 45K, 3 output | — | 1.208 s |
| Final release follow-up cached tokens | — | 45,037 |
| Final release sampled peak process footprint, 45K | — | 18.63 GiB |
| New swapouts during each valid 45K run | 0 | 0 |

Both matched arms and the final binary returned the correct launch key, recovery key, and late-corrected owner. One long matched pair supports a 6.87% time reduction; the final binary repeated at 293.46 s, about 4.7% below control. These are observations, not confidence intervals. Temperature 0, thinking disabled in requests, exact same 45K prompt; numeric session_id retained across follow-up.

Seven matched behavior cases produced byte-identical answers: both passed 6/7. Both returned inventory 96 instead of 103. Failed versus planned actions, late correction, tainted content, and two budget decisions passed. These direct-inference probes do not establish universal numerical equivalence or full Nanobot endurance. GEMM and scratch round differently.

Validation: 642 engine tests, 782 server tests, three runtime-identity integration tests, focused kernel/oracle and hardware-policy tests passed in release mode. The M4 default regression failed before the policy change and passed after. Existing oracle accuracy tolerance was retained and applied to both kernels; only the obsolete scratch-default assertion changed. GLM review found no blocking issue. Graph review identified expected performance/docs and pre-existing recovery files (high/critical hot-path impact); no unexpected files were integrated.

Installed executable SHA256: `fbf8b3e8b9f3450c7f4caa1fb8f1f32baf39f307a69d4edb20ff8eb0e87697a2`.
Metal library SHA256: `d4ec42fe79abd9d24922c84a6a160b98e9bba7ebed68dbed844981b1190eafb5`.
Startup selected `trellis_qgemm_simd` without a kernel override; normal pressure, 47,104 prompt tokens available. Installed API smoke returned READY in 0.737 s. Previous executable/library pair saved in `/private/tmp/higgs-roofline-evidence/installed-before-default`.

## Remaining performance evidence

Corrected serving-path profiler: sampled/extrapolated prefill components are approximately 51% expert MLP, 27% GDN attention/projections, 22% full attention/projections. At 16K decode, median per-full-attention-layer append was 0.465 ms versus attention kernels 1.660 ms. Eval barriers and sampling affect execution: these are diagnostic proportions, not uninstrumented wall-time shares or hardware saturation. Earlier inclusive attention timing cannot be compared directly.

AC Metal copy reference: 99.43 decimal GB/s, 256 MiB input plus 256 MiB output, 20 timed samples, no new swapping. Earlier battery/LPM reference was 98.61 GB/s. Neither is an inference roofline. GPU-active time does not establish bandwidth/FLOP saturation. No ANE benefit for this model on this Mac has been established.

Rejected output Hadamard fusion: 2× isolated operator speed, but only 1.12% full-request gain and 3.65% slower decode. Rejected larger chunks as default: roughly two extra seconds saved at 16K for about 2 GiB more memory demand. Preserve 45K headroom.

Next optimization targets: packed expert MLP prefill and long-context attention decode. Any ANE overlap must beat GPU execution including conversion/synchronization and pass recurrent-state quality checks. Retained-session streams currently expose an initial progress event but no per-chunk events; cancellation/pressure boundary coverage needs a separate audit. No retained-session watchdog failure was reproduced.

## GLM follow-up

User explicitly authorized the relevant source/evidence payload. GLM is analyzing a curated bundle in `/private/tmp/higgs-glm-profile-bundle` (source excerpts and sanitized timings only; external-directory and shell access denied). User then authorized running regardless of power. Fresh32K capture completed on battery with Low Power Mode off:197.08s total,184.59s TTFT,128 outputtokens,17.17GiB peak,normalpressure,no new swapouts. Installedserver restored. Same-run FA prefill median grew72.00ms/layer in earlychunks to348.86ms in latechunks; decode append0.780ms vsSDPA4.004ms perFA layer. Cross-run16K AC vs32Kbattery does not isolate context scaling. GLM final report pending.


GLM follow-up complete. Raw authored report preserved in GLM-RAW.md; PROFILE-REVIEW.md records verified measurements and source-based corrections. Reject unsupported TurboQuant-active, strictly-fp16 cache, and float-mask-per-layer claims. No additional production change from profiling. Next: MLP interior/BM A/B, offset-mask equivalence, active dense SDPA counters.
