# Longer-context FP16 attention validation

**Outcome:** Long-context exact greedy parity failed: FP16 and FP32 produced different 128-token digests at4K,8K and16K. Each mode repeated its own digest exactly in both fresh processes. This establishes repeatable precision-associated trajectory differences for this fixture, not their semantic quality. Candidate remains uninstalled.

Same binary SHA-256 `db8dfb78ba8382d16fc96b8a2430fc00f8967ff1cf21b1c5840504cc3e2db9c1`; fixed process order OFF/ON/ON/OFF; native Escha W2;4K/8K/16K frontiers;128-token probes;1024-token prefill chunks. No compiler, graph indexing or second model ran alongside the sweeps.

| Context | FP32 decode mean tok/s | FP16 decode mean tok/s | Change | FP32 incremental prefill ms | FP16 incremental prefill ms |
|---|---:|---:|---:|---:|---:|
| 4096 | 17.94 | 17.53 | -2.3% | 31176 | 25694 |
| 8192 | 16.12 | 16.43 | +1.9% | 32142 | 27371 |
| 16384 | 8.64 | 14.48 | +67.7% | 90668 | 67516 |

Only two observations per precision. At16K, individual FP32 decode rates were5.565 and11.711 tok/s, versus FP16 rates16.502 and12.464 tok/s. Both FP16 observations exceed both FP32 observations, but the wide variation limits the precision of the mean speedup. Process-order, thermal and host-memory effects remain possible. The4K/8K differences do not establish a meaningful decode win.

| Mode | KV at4K | KV at8K | KV at16K |
|---|---:|---:|---:|
| FP32 | 233,635,840 B | 401,408,000 B | 736,952,320 B |
| FP16 | 149,749,760 B | 233,635,840 B | 401,408,000 B |

Every row satisfies fixed GDN state65,863,680B + context length ×40,960B(FP32) or20,480B(FP16). Thus variable dense KV halves; total retained memory at16K falls from702.8125MiB to382.8125MiB. Mean whole-process peak physical footprint is25,609,881,068B versus25,423,341,108B, a0.728% reduction. Do not describe this as halving whole-model or process RAM.

Four processes exited0. Those successful exits mean the benchmark ran; they do not turn cross-precision digest mismatches into an accuracy pass. The benchmark uses fixed repeated ordinary prose and stores digests, not decoded text or per-step logits. It cannot locate the first changed token or score answer quality. Its cache flow appends context and rolls back decode probes; full HTTP retained-session save/reload, batching and DenseMTP remain outside this test.

Next precision gates: capture decoded outputs/per-step distributions on diverse long-context tasks; score retrieval, structured tool use and numerical stability; validate retained-cache restoration through the actual server path. Keep production FP32 until those gates support a change.

Raw commands, hashes, all rates/digests, process timing/footprint and analysis are preserved in `long-context/`. `run.py` preserves the exact invocation used and refuses to overwrite its original output directory.

## Source-supported kernel interpretation

The Rust bindings (`mlx-rs`/`mlx-sys`) are pinned to `b46423d2447d3db354c134a0ef25ff55dfdfe8b6`; the built MLX core source is `ce45c52505c8158ea48d2a54e8caae05efd86bfe` (`v0.31.1`). These are different repository identities. Audited core root: `/Users/peppi/Dev/higgs/target/release/build/mlx-sys-e955ea9cb0a9ee5d/out/build/_deps/mlx-src`.

Native Escha uses16 query heads,2 KV heads and head dimension256. For one-token decode, both FP32 and FP16 qualify for fused vector attention; at the tested lengths both select the two-pass vector route (`mlx/backend/metal/scaled_dot_product_attention.cpp:610–636,681–749`). Dtype chooses kernel specialization, not different fused eligibility. FP16 narrows K/V loads and the typed partial-output buffer; dot-product, softmax and output accumulation remain FP32 (`kernels/sdpa_vector.h:180–316`). This is a data-width mechanism consistent with a long-context benefit, not a measured attribution of the bottleneck.

Architecture-dependent block-count changes above8K or at16K apply equally to both dtypes. Runtime architecture suffix was not recorded, so those branches cannot explain this run conclusively. The1024-row prefill has head dimension256, which excludes the fused full-attention route for both dtypes. No evidence establishes a dtype-driven eligibility switch or host RAM exhaustion.
