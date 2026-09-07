# ANE prefill applicability for Escha 35B W2 on M4 base / 32 GiB

## Verdict

ANE is credible for **one fixed-shape dense projection benchmark**, but the evidence does not support routing the whole Escha prefill through ANE. The safest first target is a token-local projection (for example GDN `z`) or a dense MLP projection while recurrent q/k/v remains at checkpoint precision on GPU; MoE gather matmuls, the stateful GDN recurrence, and 45K attention are outside the demonstrated compiler/runtime envelope. Existing Higgs work reached isolated ANE projection wins but never demonstrated an end-to-end TTFT win, and several integrated attempts regressed.

The current production baseline is the datum to beat: `experiments/capacity-incident/recovery-validation-20260906/45k-measurements.json` records 44,992 uncached prompt tokens in 467.32 s (**96.2766 tok/s**) on Apple M4 base / 32 GiB, peak sampled footprint 17.96 GiB, normal pressure, and zero new swapouts. The short 507-token request has no terminal prefill event, so it does not provide a short-prompt prefill rate.

## Existing work

### Higgs

- Branch `feat/ane-prefill` contains the private-API prototype:
  - `89ab5fdc8`: ANE bridge and MIL generation.
  - `e7172aa6c`: `GdnPrefillEngine`, two kernels per sequence bucket (512/1024/2048), and GDN projection correctness. Its commit record says cosine 0.999991 on a real 35B-A3B **3-bit** checkpoint and Phase-1 ANE times 38–68 ms/layer versus GPU 3–17 ms/layer.
  - `cd78076c3`: zero-copy IOSurface input/output; its commit record says 7% faster at S128 and 35% faster at S512 for isolated projections. It did not wire the ANE route into `higgs-engine` or report end-to-end TTFT.
- An uncommitted worktree note, `.claude/worktrees/loving-matsumoto-d1b0fb/.planning/ANE-HANDOFF.md` at base `1dc083267` (2026-06-23), records later public-CoreML experiments. Treat these as investigator notes because raw benchmark artifacts are under `/tmp` and the worktree is dirty:
  - FP16 CoreML qkvz 2048→12288, S512: 35.6 ms, 0.72 TFLOP/s; note compares it with a ~7.5 ms GPU projection and warns it would regress prefill.
  - An int8 CoreML probe at the same shape: 4.86 ms (`CPU_AND_NE`) versus 12.5 ms GPU CoreML / 17.4 ms CPU CoreML; `MLComputePlan` selected Neural Engine. This establishes operator feasibility in that prototype, not current Escha end-to-end performance.
  - A concurrency probe reports ANE and GPU can overlap, but a channel-split projection still lost (9.95 ms split versus 7.5 ms GPU-only). The note identifies dependencies and I/O/synchronization as the remaining integration problem.
  - Earlier `HIGGS_TARGET_ANE_GDN=1` results cited by the note regressed prefill about 3×. The referenced `mc-ane` raw claims are not in the current repository, so their precise hardware/build provenance remains unverified here.
- The plan `.claude/worktrees/loving-matsumoto-d1b0fb/.planning/ane-prefill-plan.md` describes the old 4-bit route: dequantize GDN projection weights to FP16 BLOBFILE form, retain about 1.51 GB for qkvz alone, and use fixed buckets. This memory and format analysis does **not** automatically transfer to the current native Escha W2 build.
- Current Escha uses a different hot path. `crates/higgs-models/src/eschamoe.rs:1385` keeps expert projections in native trellis form (11.2 GB stated versus 21.7 GB for affine expansion on the 35B release). `EschaProj::gather_forward_mode` routes top-k experts through a native Metal QMV, scratch decode+matmul, or opt-in gathered trellis QGEMM. Commit `a2f8c169d` measured the corrected gathered kernel at 818 GFLOP/s in isolation but neutral full prefill (97.5/93.8 versus 94.1/96.8), showing that expert gather is a minority of total prefill.
- Commit `5e064e14d` tested a fused triangular-solve Metal GDN path at a 1024-token prefill. Per GDN layer, the existing serial kernel was 37.0 ms versus 42.2/43.0 ms for chunkwise variants; materialized intermediates, not dispatch count, were the limiting cost.
- The active Metal direction is documented in `docs/superpowers/specs/2026-09-02-indexless-metal-prefill-design.md`. Commit `8b4cdff3a` added a benchmark-only sparse-attention feasibility probe on `perf/metal-prefill-feasibility`; no results document was found. This targets the measured long-context position-dependent cost more directly than the old ANE projection branch.

### nanobot-rs

- `archive/research/ane-mlx-removed/` records the earlier ANE training/decode program. The durable constraints include roughly 16 BLOBFILE-weighted ops and ~32 MB BLOBFILE per program, ~119 loaded programs/process, no useful dynamic-weight path, IOSurface transfer costs, and high variance/contention. A tall-skinny ANE classifier was at best comparable to CPU and sometimes slower.
- Commit `1010e2697` records GDN ANE kernels at only 1.05× break-even and severe unthrottled contention; commit `c3ebd4f0a` reached 1.4 tok/s in a hybrid 35B decode but left scalar CPU MoE at ~650 ms/token. Commit `a6c34a0b4` expanded the prototype. Commits `bc1701d5a` and `e899d689f` later removed the broken in-process MLX and ANE product features rather than preserving parallel production paths.
- `bridge/ane/ane_bridge.{h,m}` remains as orphan research code using private APIs. It is no longer built by a live feature. `experiments/gauge-distill/qwen_gauge_distill.py:1068::export_for_ane` exports only a small Conv1d/GELU/add distilled core; commit `1010e2697` records only 2.7% top-1 agreement, so it is not a faithful Escha prefill substitute.

## What the M4 ANE compiler article proves

Primary source: [Inside the M4 ANE, Part 4b: Inside the Compiler](https://maderix.github.io/articles/inside-the-m4-ane-part-4b/) (Manjeet Singh, August 2026). It traces MIL → Zin IR → Zin MIR → scheduling/allocation → HWX on one M4, macOS 26.3, compiler 9.202.0, H16G.

Its research compiler coverage is deliberately narrow: selected FP16 Conv1x1 geometries; square matmuls N128/256/512 and tiled multiples of 128 through N4096; one four-layer C64/S64 W8A8 Conv1x1 chain; and fixed multi-op probes. S128/D128 FP16 attention needs three HWX programs and two IOSurface boundaries (369.458 μs research compiler, 127.208 μs Apple compiler). Chunked DeltaNet C128/D128 needs 58 programs and 22.728792 ms, with synchronous submissions dominating. Unsupported shapes, dtypes, axes, and dynamic cases are rejected. This is evidence that these primitives can execute on ANE, not evidence for a 45K transformer or W2 trellis execution.

| Escha prefill part | Applicability |
|---|---|
| Dense GDN / full-attention projections | **Plausible isolated target.** Fixed Conv1x1/int8 shapes fit ANE strengths. Must use the actual current checkpoint representation and include conversion, IOSurface, synchronization, and resident-copy costs. The article's W8A8 result does not establish native W2 support. |
| MoE experts | **Poor target.** Escha selects 8 of 256 experts per token and uses gathered native trellis kernels. The article has no sparse expert gather, top-k dispatch, or 2-bit trellis decoder. Expanding/routing static expert programs would add memory, program-load, and dispatch pressure. |
| GDN recurrence | **Poor target now.** It is stateful and sequential. The article's DeltaNet proof uses 58 programs and is submission-bound; Higgs's own chunkwise/materialized GPU variant already lost to the resident serial Metal kernel. |
| Attention | **Unproven for this workload.** The article covers fixed S128/D128 FP16 attention. Escha's prompt is chunked at 1024 while KV length grows toward 45K, with GQA, causal masking, and dynamic position. Tiling that into fixed ANE programs would add IOSurface/submission costs. Only every fourth layer is full attention, but the live trace's smooth slowdown makes attention/KV the more relevant long-context mechanism to measure. |

## Smallest discriminating experiment

Do not restore the whole ANE branch. Add no production path initially. In an isolated benchmark, use the **current production Escha checkpoint/build** and one real token-local projection at the actual 1024-token prefill chunk shape. Do not quantize the recurrent q/k/v path merely to make it ANE-compatible; current oMLX work ([PR 3133](https://github.com/jundot/omlx/pull/3133)) keeps recurrent qkv at checkpoint precision on GPU and applies INT8 only to token-local `z`, making quality preservation part of the operator boundary.

1. Measure the candidate token-local projection's current GPU latency and its fraction `f` of copy-inclusive full-prefill time at early and late prompt positions. Separately measure recurrent qkv, which remains on GPU.
2. Run only the candidate token-local projection through a public-CoreML int8 Conv1x1 kernel, using fixed IOSurface-backed buffers. Measure warm compute, activation input/output copies, synchronization, one-time conversion/load time, and incremental resident/peak physical bytes separately.
3. Require matching numerical output under an agreed tolerance and confirm ANE placement with `MLComputePlan`.
4. Stop unless ANE reduces **copy-and-sync-inclusive projection latency by at least 20%**, adds memory that the live capacity ledger can admit, and does not slow concurrent Metal work. A 20% projection-latency reduction yields only about `0.20 × f` whole-prefill latency reduction; it needs `f ≥ 25%` to offer even a 5% end-to-end gain. If “20% faster” means 1.2× throughput, the time reduction is 16.7% and `f` must be at least 30% for a 5% overall gain.

Only after that gate should an engine-level overlap experiment be considered. Transformer layer and chunk dependencies mean ANE/GPU concurrency alone does not guarantee schedulable independent work.

## GitNexus and evidence limits

GitNexus queries were run for both repositories. The Higgs index is 37 commits behind and could not resolve current `forward_scheduled`, `run_prefill`, or `forward_raw_hidden`; those results are **UNKNOWN**, not evidence of no callers. Text corroboration finds `Qwen3NextAttention::forward_scheduled` called by `forward` and `forward_canonical_rows`, and `SimpleEngine::run_prefill` called from generation/session paths in `crates/higgs-engine/src/simple.rs`.

The nanobot-rs index is 9 commits behind. Context resolves `export_for_ane` with textual/graph caller `main`. It resolves the header declaration `ane_bridge_compile` with no graph caller; text corroboration finds only the orphan bridge implementations and no current production Rust integration. No repository edits, builds, inference, installation, or process changes were made during this investigation.
