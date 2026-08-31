# Higgs Post-April Benchmark Summary (M4 Base)

## Key Findings from Git History Since April 3, 2026

### Escha QGEMM SIMD Kernel (July-August 2026)
- **e2e decode**: 92.5 tok/s @ 2K context (noise band 92-97 confirmed)
- At sorted-8192: decode ~1.07G element-decodes fills ~55ms of 77ms window vs ~22ms MMA — decode dominates 72/28
- Phase-0 baselines: e2e 94-97 tok/s at 2-16K; qgemm 1.5-1.6x scratch
- After fix: simd matches scalar agreement (rel 2.16e-4), edges it at scale (818 vs 736 GFLOP/s)
- Default kernel is BM=32 simd (BM=64 debug closed, needs GPU capture session)

### Bonsai-Q2 Microbench (July 18, 2026)
- gate_up (17408x5120): Q2 M=5 = **1.79x** vs Q1 TG-LUT4
- down (5120x17408): Q2 M=5 = **1.73x** vs Q1
- Projected end-to-end dSpark/AR ratio: ~**1.55x**, above 1.45x target

### dSpark Decode Speedup Floors (July 16, 2026)
- Reference speedup: **1.435x** (20.42 → 29.31 tok/s)
- Min acceptance rate: 87%
- Min decode speedup floor: ~1.392x (3% tolerance)

### Escha Native Path for 35B MoE (August 2026)
- Native path: holds 11.2 GB, loads in ~6 s
- Affine path: holds 21.7 GB, takes ~140 s, drives 32GB machine into swap → collapses to 2.6 tok/s at 8k context
- Native is now the default (HIGGS_ESCHA_NATIVE=0 selects affine with =0 flag inverted)

### Previous April Baseline (for comparison)
- Qwen3.5-35B-A3B-3bit MoE (~14GB): **~55-56.6 tok/s** decode on M4 Base
- Qwen3.5-27B-4bit (~16GB): **~6.8-6.9 tok/s** decode on M4 Base

### Implications for mlx-serve Comparison
The escha simd QGEMM improvements (92-97 tok/s range at 2-16K) suggest significant decode throughput gains over the April baseline. The native trellis path for 35B MoE avoids the swap collapse that would hurt performance. For a fair mlx-serve comparison on your M4 Base, we'd need to estimate what mlx-serve's M4 Max numbers would be after dividing by ~3.3x bandwidth ratio, then compare against these improved Higgs numbers.