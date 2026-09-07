# ANE branch and current GitHub prefill work — 2026-09-06

This supersedes the earlier suggestion to start by approximating the whole GDN qkvz projection.

## Existing Higgs work

The newer committed implementation is **`feat/magic-canvas`**, tip `96d2ed9da`. Relevant commit **`89141aa6e`** adds public-CoreML INT8 layer-0 dense MLP prefill. Reusable pieces include `ane_mlmodel.rs`, `qwen3_next_ane.rs`, model loader finalization, and `benchmarks/ane_int8_mlpackage_probe/`. Its predecessor `92f91f59e` reports a 2.23x isolated gate-projection gain on Carnice-9B / M4 Max; this is not current Escha / base-M4 evidence. The branch has 119 unique commits and lacks 637 nightly commits at this inspection, so a whole-branch merge is inappropriate.

`feat/ane-prefill` at `2cb7808b5` (2026-04-11) is the older private-API GDN prototype, including `cd78076c3` zero-copy projection work. The newer dirty `loving-matsumoto` worktree has CoreML qkvz probes, but its engine changes wire the DFlash ANE drafter; qkvz prefill is not wired. Archived nanobot failures do not settle the current prefill question.

## Primary GitHub evidence

| Project | Hardware / model | Published finding | Interpretation |
|---|---|---|---|
| [Cider experimental](https://github.com/Mininglamp-AI/cider/blob/main/experimental/README.md) | M4, Qwen3-VL-2B, T1024 | 1348.6ms GPU vs1156.9ms split,1.17x; T5121.039x | Concurrent output-channel split. README explicitly says no end-to-end advantage yet because integration does not preserve MLX lazy evaluation. Useful mechanism, not a proven server speedup. |
| [oMLX issue3116](https://github.com/jundot/omlx/issues/3116) | M4Pro48GB, Qwen3.8-27B AWQ5bpw, singleANE | GPU129–132tok/s; manual nonfused split152–155; earlier tuner166 | User-reported real prefill benefit on M4-family hardware. Issue concerns erroneous tuner topology/verification; not a base-M4/W2 matched result. |
| [oMLX PR3133](https://github.com/jundot/omlx/pull/3133) | M3Ultra, Qwen3.6-27B oQ4e,32K | GPU410.1 vs recurrent-safe hybrid503.0tok/s,+22.7% | Most relevant long-context implementation. Approximate INT8 is limited to token-local z rows; recurrent qkv stays GPU checkpoint precision. Prior wider split caused32Kquality failures. This is approximate, not bit-exact inference. |
| [maderix/ane-prefill-bench](https://github.com/maderix/ane-prefill-bench) | BaseM4,24GB,Qwen3.6-27B | 19.1/24.3/27.8tok/s at256/512/1024 | Executable ANE-prefill reference with layerwise GGUF-to-FP16 conversion, ANE projections/FFN and CPU DeltaNet. No paired GPU speedup established by its table. |
| [AtomGradient batch-prefill](https://github.com/AtomGradient/hybird-batch-prefill-on-ane) | M2Ultra,0.8B/2B | 11.3x/7.3x vs sequential ANE dispatch | Not M4 and not a speedup against GPU; the shown GPU baseline is faster at74tokens. Useful dispatch batching technique. |

## Revised direction

Reuse `feat/magic-canvas` public CoreML prefill plumbing selectively, with the older branch's zero-copy lessons. Study oMLX's concurrent channel split, banked programs, fused GPU merge, and recurrent-safe z-only selection before adapting it. Keep Escha native trellis expert weights intact. Gate on same-checkpoint whole-prefill latency and32K/45Kquality, including partial-prefix restore; projection cosine alone is inadequate. Cider's no-E2E limitation is exactly the synchronization/lazy-evaluation pitfall to avoid. No public result found establishes a45KbaseM4nativeEschaW2 gain.

No code, model, runtime configuration, or processes changed during this follow-up.
