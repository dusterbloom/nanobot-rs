# Higgs Escha W2 kernel-default audit

Date: 2026-09-05  
Source: `higgs` commit `327e5021ef957a7d6968f6780a7644f00b410a9e` plus the local candidate described below  
Model: `EschaLabs/Qwen3.6-35B-A3B-Escha-W2`  
Host: Apple M4, 32 GiB, Darwin 27, release profile

## Decision

Keep `HIGGS_ESCHA_TRELLIS_GEMM` off by default. The direct trellis QGEMM is
correct and wins isolated synthetic expert kernels, but it was 34.48% slower
than the scratch path in the matched real-model 1,024-token prefill runs below.
The release environment should leave this variable unset. `=1` remains a
diagnostic/performance-development override.

Keep native W2 and its decode QMV path. Leave the other established safe
defaults unchanged. None of the opt-in decode experiments has native Escha 35B
correctness plus end-to-end performance evidence sufficient for promotion.

## Real-model matched benchmark

The benchmark binary was built after temporarily changing the no-override
trellis selector to QGEMM. That candidate was reverted after the result. Every
measurement used a fresh process. The alternating order was scratch, candidate,
scratch, candidate to reduce load-order and thermal bias.

Benchmark binary SHA-256:
`b5b1a95bd225508d1e23a00a1f501d84cfb208e86b069f777c904b7e2179537c`.
The temporary candidate source patch was:

```diff
-    *MODE.get_or_init(|| std::env::var("HIGGS_ESCHA_TRELLIS_GEMM").is_ok_and(|v| v == "1"))
+    *MODE.get_or_init(|| !std::env::var("HIGGS_ESCHA_TRELLIS_GEMM").is_ok_and(|v| v == "0"))
```

On the restored source, `HIGGS_ESCHA_TRELLIS_GEMM=1` selects the same QGEMM
kernel as the candidate's unset environment. Rebuilding the final source and
using that explicit override is the reproducible comparison form.

Build:

```sh
cd /private/tmp/higgs-recovery
CARGO_TARGET_DIR=/Users/peppi/Dev/higgs/target \
  cargo build --release -p higgs-bench --bin bench_frontier
```

Scratch invocation:

```sh
cd /private/tmp/higgs-recovery
HIGGS_ESCHA_NATIVE=1 HIGGS_ESCHA_TRELLIS_GEMM=0 \
  /Users/peppi/Dev/higgs/target/release/bench_frontier \
  --model-dir /Users/peppi/.cache/lm-studio/models/EschaLabs/Qwen3.6-35B-A3B-Escha-W2 \
  --frontiers 1024 --probe-tokens 32 --runs 1 \
  --prefill-chunk-size 1024 --format json
```

Candidate invocation was identical except
`HIGGS_ESCHA_TRELLIS_GEMM` was unset, selecting the temporary default-on
QGEMM implementation.

| Order | Prefill route | Prefill ms | Prefill tok/s | Decode tok/s | Output digest | KV bytes |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | scratch | 8,289.806 | 123.525 | 17.121 | `12792644362472167692` | 107,806,720 |
| 2 | QGEMM candidate | 12,055.581 | 84.940 | 17.985 | `12792644362472167692` | 107,806,720 |
| 3 | scratch | 6,671.465 | 153.490 | 20.232 | `12792644362472167692` | 107,806,720 |
| 4 | QGEMM candidate | 10,604.151 | 96.566 | 17.991 | `12792644362472167692` | 107,806,720 |

Mean scratch prefill was 138.507 tok/s. Mean QGEMM prefill was 90.753
tok/s, a 34.48% regression; scratch was 1.526x as fast. Identical greedy-output
digests and KV bytes provide end-to-end output and cache-layout parity for this
probe. Decode samples are short and noisy; the route change only applies above
32 gathered rows, so they are a parity check rather than a decode speed claim.

Durable JSON copies:

- `kernel-bench/327e502__Qwen3.6-35B-A3B-Escha-W2__20260905T133410Z-749.json`
- `kernel-bench/327e502__Qwen3.6-35B-A3B-Escha-W2__20260905T133429Z-945.json`
- `kernel-bench/327e502__Qwen3.6-35B-A3B-Escha-W2__20260905T133453Z-265.json`
- `kernel-bench/327e502__Qwen3.6-35B-A3B-Escha-W2__20260905T133508Z-389.json`

Limitations: this is four total runs at one 1,024-token frontier on one host.
The benchmark reports KV allocation bytes but does not sample MLX active,
cached, or peak allocator memory. It is enough to reject this default promotion,
not to claim a global optimum across all prompts, context lengths, and devices.

## Correctness checks

Release tests used the same local source and shared release target:

```sh
cd /private/tmp/higgs-recovery
CARGO_TARGET_DIR=/Users/peppi/Dev/higgs/target \
  cargo test -p higgs-models --release \
  eschamoe::tests::escha_proj_gather_forward_matches_oracle -- --exact --nocapture
CARGO_TARGET_DIR=/Users/peppi/Dev/higgs/target \
  cargo test -p higgs-models --release \
  eschamoe::tests::eschamoe_gather_qgemm_matches_scratch_matmul -- --exact --nocapture
CARGO_TARGET_DIR=/Users/peppi/Dev/higgs/target \
  cargo test -p higgs-models --release \
  eschamoe::tests::eschamoe_gather_kernels_preserve_logical_row_bits -- --exact --nocapture
CARGO_TARGET_DIR=/Users/peppi/Dev/higgs/target \
  cargo test -p higgs-engine --release --test runtime_identity
```

All passed. Direct QGEMM versus scratch covered K=2 and K=3,
sorted/unsorted routing, forward composition, and a partial block. Maximum
reported relative error was `3.9956288e-4`; forward relative gaps were
`2.64677e-4` and `2.9513036e-4`. Logical-row bit preservation passed. The
default-route oracle test now also requires the greater-than-32-row production
path to match forced scratch bit-for-bit.

## Environment decision table

“Unset” means use the code default; it does not mean the feature is disabled in
every row.

| Variable/path | Code default | Release setting | Evidence and scope |
| --- | --- | --- | --- |
| `HIGGS_ESCHA_NATIVE` | on | unset | Required for the packed W2 artifact. Prior real-artifact validation loaded in 6.4 s, used about 11.16 GiB active/11.69 GiB peak, and completed a 129-token greedy fixture. Affine conversion exceeded practical memory. `=0` remains diagnostic. |
| `HIGGS_ESCHA_TRELLIS_GEMM` | off | unset | Real-model chunk-1024 result above rejects promotion. `=1` remains an explicit comparison override. |
| `HIGGS_ESCHA_QGEMM_SIMD` | on inside QGEMM | unset | SIMD was 5–11% faster than scalar at scale and neutral at 512 rows in isolated tests. It is dormant while trellis GEMM is off. `=0` selects the scalar diagnostic kernel after QGEMM is enabled. |
| `HIGGS_ESCHA_QGEMM_BM` | 32 | unset | BM64 was correct after its grid fix but mixed: one K=3 down projection improved while an even-run gate/up case lost to scratch. No full-router win supports promotion. |
| Native gather QMV (`rows <= 32`) | on, no flag | keep | Covers ordinary one-token decode. GPU-vs-CPU validation was bit-identical at decode level; QMV relative gaps were about `2.1e-7` synthetic and `3.7e-7` on real weights. |
| `HIGGS_COMPILED_GATING` | on | unset | Established general Qwen gating default. Historical gain was small (about 0.8% on an affine model), but there is no native regression evidence that warrants disabling it. |
| `HIGGS_ASYNC_LAYER_STATE_EVAL` | on | unset | Established S=1 attention/GDN state-evaluation path; does not replace native expert arithmetic. |
| `HIGGS_CACHE_GATED_DELTA_CONFIGS` | on | unset | Reuses stable GDN kernel configurations. Keep the established cache default. |
| `HIGGS_COMPILED_GDN_DECODE` | off | unset | Changes the S=1 recurrent primitive; DFlash validation explicitly rejects the mode and there is no native 35B parity/performance proof. |
| `HIGGS_ENABLE_SELECTED_DECODE_GEMV` | off | unset | Applies to selected affine dense `QLinear` projections, not the native packed expert QMV. No native 35B evidence supports enabling it. |
| `HIGGS_QGEMV_FFN_MODE` | `both`, dormant | unset | Only matters when selected decode GEMV is enabled. It does not justify enabling that experimental path. |
| `HIGGS_CACHE_QGEMV_CONFIGS` | off | unset | Dormant with selected decode GEMV off; no native expert benefit demonstrated. |
| `HIGGS_MOE_FFN_GATE_UP` | off | unset | Unsafe/inapplicable to native Escha: the normal global-sort path detects `self.escha` and uses its already-fused native gate/up projection; this flag routes through the affine fused path and placeholder tensors. Historical speedup was for affine 35B 3-bit. |
| `HIGGS_CROSSROW_QMV` | on where eligible | unset | Only applies to eligible affine Q4 g64 batches with M=2..9. Native experts use Escha QMV/QGEMM; ordinary autoregressive decode is M=1. |
| `HIGGS_SEPARATE_GDN_PROJ` | off | unset | Explicitly unsupported with EschaMoE and has no validated release path. |
| `HIGGS_DENSE_REQUANT_8BIT` | off | unset | Alters dense GDN weights, not native expert storage; no native behavioral/performance gate supports it. |
| `HIGGS_BONSAI_Q2_SIMD` and Bonsai kernel flags | model-specific | unset | Tuned for other packed model families; irrelevant to this Escha 35B W2 artifact. |
| TurboQuant activation threshold | 100,000 tokens | unset | Earlier activation imposed a severe decode tax; measured decode fell near 10 tok/s around 7.4K with the old path versus roughly 47 tok/s dense. The high threshold preserves dense KV for normal contexts. |
| MTP/dSpark/DFlash experiment flags | off/unselected | unset | The published 35B auxiliary head lacks the routed-expert weights required for a validated speculative release path. |
| Profiling, verification, finite/NaN flags | off | unset | Diagnostic only; keep them out of release binaries unless collecting a bounded trace. |

The evidence-backed release environment is therefore the empty override set for
these kernels. Defaults already select native packed W2, small-row QMV decode,
compiled gating, asynchronous state evaluation, and the GDN config cache while
retaining scratch prefill and excluding unvalidated arithmetic paths.
