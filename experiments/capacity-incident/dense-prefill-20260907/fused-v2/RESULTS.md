# Fused D256 follow-up results

Completed isolated FP32 base-M4 benchmark on battery with stable power source. No production code or installed artifact changes. Installed d77352ef8 restored; live READY response and binary hash verified in restore-receipt.json.

## Same-process causal/explicit comparison

Each cell is the median of six round medians, milliseconds per attention call. Each round has three timed repetitions per arm after separate warmup. Query shape1x16x1024x256, KV1x2xKx256; strided Q and padded KV backing, identical materialized arrays. Query128 uses explicit sliced offset mask/full KV, matching the installed long-context path. Candidate causal arms use equivalent native lower-right causal masks.

| Arm | K32768 | K45056 |
|---|---:|---:|
| Old fallback | 407.00 | 561.15 |
| Installed query128 | **299.74** | **467.23** |
| Original16x8, explicit | 955.08 | 1339.09 |
| Separate unpadded16x16, explicit | 837.29 | 1160.23 |
| Safe register-Q16x16, explicit | 476.00 | 673.89 |
| Original16x8, causal | 437.95 | 600.45 |
| Separate unpadded16x16, causal | 468.35 | 673.57 |
| Safe register-Q16x16, causal | 1206.64 | 1695.69 |

No fused candidate won against query128 in any of six rounds at either long shape. Best fused original-causal median paired ratio to query128:1.446 at32K,1.302 at45K. Absolute values still drift; raw ranges are in analysis.json. This is kernel-level evidence, not whole-request speedup or roofline utilization.

All numerical gates passed against full FP32 fallback for offset/tail31x1055 and long shapes. First explicit-only pass also checked8K. Normal Q/K max_abs gate1e-5 and relL2 gate1e-5; amplified Q/K (each multiplied by3) predefined max_abs gate1e-4 with unchanged relL2 gate1e-5. No tolerances were relaxed after results. Native-causal equivalence checked as well. Random synthetic checks do not establish arbitrary full-model behavior.

Two valid candidates: separate Q/KV with zero padding (32KiB shared-memory arithmetic), or safe existing register-Q handoff with16x16/WM2. Distinct library+pipeline cache keys prevent candidate collisions. No unsafe Q/KV aliasing with later shared-Q reads. Compiler math flags unchanged. Reg-Q's mask-dependent slowdown is measured; bank conflicts/spills/occupancy are not proven without compiler/counter data.

Supervisor polled power, normal OS pressure and no new swapouts during each benchmark subprocess. No polling guard tripped; this is sampled monitoring, not exhaustive transient capture. Post-first-pass model startup was rejected by capacity policy despite normal pressure on subsequent inspection; manual retry restored Higgs. Second pass included20s settling before successful automatic restoration. Historical swap counters changed outside measured subprocesses; do not claim the entire task produced zero swaps.

Decision: reject these candidates for serving integration. The GLM-suggested small tile change did not produce a win. A future redesign should begin with actual shader resource/limiter evidence, rather than another blind tile sweep. Banked2.38GiB memory saving remains installed.
