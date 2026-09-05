# Higgs allocation, retention and capacity audit — 2026-09-05

Studied the exact server source at327e5021ef957a7d6968f6780a7644f00b410a9e,
the native Escha model configuration, captured policy-run telemetry, design/plan
files, Git history, and original OpenCode review sessions. No runtime settings,
production source, model residency or server processes were changed. This report
identifies real inconsistencies; it does not establish physical RAM exhaustion.

## Findings

### 1. Confirmed: the KV cost slope is half the observed native cache slope

The capacity factory in `crates/higgs/src/state.rs:1975-1988` invokes
`EngineCostDescription::fp16_from_model_dir`. That function in
`crates/higgs-engine/src/mlx_tuning.rs:442-475` assumes two bytes per KV element.
For this model it computes10 full-attention layers × keys/values ×2 KV heads ×256
head dimension ×2 bytes =20,480 bytes/token.

The captured retained-cache values instead exactly match FOUR bytes per element:

`retained_bytes = 65,863,680 + round_up(tokens, 256) * 40,960`

The fixed term matches30 GDN layers, each with32×128×128 FP32 recurrent state and
three convolution-history rows across8192 FP32 channels. The cache allocation
code preserves the input K/V dtype rather than enforcing FP16
(`higgs-models/src/cache.rs:1054-1105`). Retained byte accounting sums actual
array `nbytes` (`higgs-models/src/lib.rs:532`, `:591`; target-only retained state
in `higgs-engine/src/cache/paired.rs:1769` counts one cache, not two copies).
The native expert projections explicitly return FP32
(`higgs-models/src/eschamoe.rs:615-653`, `:716`). A BF16 model-config label is
therefore insufficient evidence for the runtime KV dtype.

| Retained token observation | Allocated token slots | Recorded bytes | FP32 prediction |
|---|---:|---:|---:|
| 5,085 | 5,120 | 275,578,880 | 275,578,880 |
| 7,007 | 7,168 | 359,464,960 | 359,464,960 |
| 9,129 | 9,216 | 443,351,040 | 443,351,040 |
| 11,202 | 11,264 | 527,237,120 | 527,237,120 |

All10 distinct captured nonempty retention points reconcile exactly. See
[higgs-cache-accounting.json](higgs-cache-accounting.json), independently
recomputed from raw telemetry by [verify_higgs_cache_accounting.py](verify_higgs_cache_accounting.py).

This proves a cost-slope mismatch; it does not prove the whole request ledger is
underestimated by2×. The separate256MiB fixed-session allowance is larger than
the observed GDN fixed term and partially cushions the error at shorter contexts.
The other workspace/reserve charges also remain part of the total ledger.

### 2. Confirmed: the 512MiB byte ceiling explains the retention cutoff

The test config sets16,384 retained tokens but only536,870,912 retained bytes.
At11,264 allocated slots the observed layout needs527,237,120 bytes and fits.
At11,520 slots it needs537,722,880 bytes and no longer fits. Thus more than11,264
live tokens can lose retention despite being well below the16K token ceiling.

`stash_into_bounded_with_source` in `higgs-engine/src/simple.rs:1294-1311` rejects
state exceeding EITHER ceiling and removes the prior smaller state. It explicitly
does not count this as an eviction. Consequently `sessions_evicted=0` cannot
prove that retention was preserved. The run showed retained reuse through about
11.2K and then `retained_tokens=0`, full suffix13,041: this is consistent with the
exact byte-limit calculation, without requiring an OOM explanation.

The higher16K token value is a separate ceiling, not guaranteed retention up to
that length. Treating it as a guaranteed warm-context threshold is incorrect.
The older `SESSION_CONTEXT_GOVERNOR.md` already describes the general retained-
limit→cold-prefill cliff, although its old24,576-token configuration and causes
must not be substituted for this run's byte ceiling.

### 3. Confirmed: native Escha does not imply the direct-GEMM prefill path

The live server process79309 used `HIGGS_ESCHA_NATIVE=1` and had
`HIGGS_ESCHA_TRELLIS_GEMM` unset. It was the policy-decision server left running,
not a newly launched comparison. Only selected nonsecret flags were read.

`EschaProj::gather_forward_mode` chooses native matvec for ≤32 rows. Above32 rows,
it uses `scratch_matmul` unless the separate trellis-GEMM flag is enabled.
`scratch_matmul` (`eschamoe.rs:684-716`) groups expert rows, dequantizes selected
expert weights into temporary dense buffers, constructs matmuls, concatenates
outputs and returns FP32. This is still native trellis execution; it is not the
persistent full affine-model fallback.

The direct trellis GEMM path avoids those dense scratch weights and host expert-
ID readback. However, the August30 commit titled "simd QGEMM is the default kernel"
changed only the scalar-versus-SIMD selector INSIDE the opt-in GEMM path; it did
not enable `HIGGS_ESCHA_TRELLIS_GEMM` globally. Reading the commit title alone can
therefore produce a wrong belief about the active kernel.

The measured run used1024-token prefill chunks and clear_cache_after_prefill=true.
Model weights, growing live KV/GDN state, chunk activations, decoded expert
scratch, and temporary cache-copy/growth buffers can coexist. This explains why
peak allocation exceeds steady loaded weights. The existing measurements do NOT
attribute the exact14.69GiB published peak among those tensors. Switching kernels
has not been benchmarked here and would not itself correct the KV cost slope:
the direct native projection also returns FP32.

### 4. Confirmed: per-request learning exists but is not connected to production

The design at `nanobot-rs/docs/superpowers/specs/2026-09-02-adaptive-capacity-control-design.md:292-320`
requires content-free measurements from real requests and cautious upward learning
from three clean allocation-bearing observations spanning five minutes.

The primitives exist:
- `RequestMemorySampler` resets the process-global peak under the MLX gate and
  can capture bounded prefill/decode high water (`mlx_tuning.rs:173-237`).
- `CapacityController::observe(AllocationObservation)` records cost evidence and
  qualifies rises (`capacity.rs:1214-1290`).

But whole-crate source references show:
- RequestMemorySampler::start has no live inference caller; its uses are the
  implementation, doctest and injected-probe tests.
- AllocationObservation occurs only in capacity.rs, whose observe calls are tests.
- The registry has no request-completion observation delivery method.
- Reservation release removes accounting and restores cache policy; it does not
  submit an allocation observation (`capacity/registry.rs:734-753`).

Thus real requests do not train this controller in this server build. Pressure
reductions and lifecycle/reclamation recovery are active; measurement-driven cost
learning and its qualified rise path are not. All recorded policy-run snapshots
remain basis=conservative, consistent with this source finding. Existing profile
restore/persist support does not connect live measurements on its own.

### 5. Confirmed: exported peak telemetry is not a per-request allocation trace

`MlxMemorySnapshot` reads MLX's process-wide active and peak counters. The reset
operation is only in the unused request-sampler path. Registry diagnostics expose
the last published memory snapshot (`capacity/registry.rs:425-468`), while allocator
refresh occurs at admission/lifecycle/reclamation points rather than every metrics
poll or each prefill allocation. The pressure observer updates pressure and VM
deltas; it does not continuously refresh the allocator snapshot.

Therefore a flat mlxActiveBytes during a long request is not evidence that its
live allocation stayed flat. The14.69GiB figure is the highest published MLX peak
we captured, not a complete per-phase trace or total system footprint. These
records omit allocator cached bytes, process physical footprint, raw OS pressure
versus derived state, and full cumulative VM counters. This limits both peak
attribution and the earlier "did physical RAM run out?" diagnosis.

## Why the governor reduces its allowance

The20%/30% protected reserve, immediate75% warning /50% critical reductions,
critical rejection and sticky recent-swap rule were deliberate September2–3 design
choices, repeatedly reviewed to prevent unsafe recovery. The normal→constrained
transition can be generated by any new compression activity. A return to normal
rearms the episode; another constrained episode can downshift again. Normal does
not simply restore the old capacity. The policy decision run followed
51,200→37,888→27,648→20,480→10,240→0 tokens.

This asymmetric policy relies on cautious recovery evidence. The missing live
learning connection matters: working requests cannot provide its intended
counterbalance. This is distinct from whether the final critical signal was an
OS event or sticky swap; that initiating signal remains unresolvable from the
five-second saved metrics alone. Do not disable critical rejection as a diagnosis.

## Design and commit provenance

| Commit | Intent / consequence |
|---|---|
| 4666a0645 (Aug4) | Native trellis support; native projection result is FP32. |
| 2e04f3933 (Aug5) | Adds opt-in trellis GEMM prefill alongside scratch reference. |
| f304bd05a (Aug30), restored by1fa23866b | SIMD is default within the GEMM option; does not enable GEMM itself. |
| ddc927885 (Sep2) | Adds memory sampler and hardcoded FP16 cost derivation. Sampler primitive is not live wiring. |
| d442cf362 /602f03918 (Sep2–3) | Pure adaptive controller, learned evidence and byte policy. |
| c0a1f927d /58d2baccf /b9ad7203d (Sep3) | Live OS/counter observation and pressure-episode hardening. |
| 5d62633bc /22e3e545c (Sep3) | Strict measured/revision-bound recovery, preventing unjustified zero-capacity reopening. |
| 327e5021e (Sep4, tested build) | Native loader/headroom and minimum-request cache allocation fixes; does not wire the learning sampler. |

Session evidence read directly from local OpenCode storage:
- ses_f9838fadcffeUNOUZR06beDa3A: adaptive admission/controller implementation and
  review work; identifies guarded hot paths and incident-driven fixes.
- ses_f9b873986ffeA83BFwMxXxk70J: architecture review explicitly agrees to pressure
  downshifts and the qualified five-minute recovery design.
- ses_f940b0875ffeGiDHEQR2fM1rZM and ses_f940c1f22ffenRPfRp1Qz11IXU: final review
  packet reports a three-turn retained replay, cold-request controls, no observed
  swap growth during its isolated monitor, and a later two-byte optional-cache
  fixture specifically isolating contract/timing. These are useful bounded checks,
  not evidence that the current five-minute learning path receives observations.
  Reviewers stated they relied on attached hardware evidence rather than rerunning it.

Task8 continuation checkboxes are all marked complete, but that is not evidence
that every intended adaptive behavior is live. The actual source and observations
above take precedence. The work was substantial and protects real invariants;
the gaps lie between tested components, runtime selection and integration.

## Next changes justified by this audit

1. Obtain runtime-accurate cache byte/dtype facts; correct the estimator without
   changing numerical precision or weakening reserves. Pin the recorded FP32
   geometry and byte-ceiling cutoff as regression cases.
2. Connect phase measurements to the one existing controller through worker-owned
   completion, preserving the global MLX gate and reservation/cancellation ordering.
   Demonstrate actual qualification/rise on real requests before declaring learning live.
3. Expose pressure origin and synchronized cumulative VM/process/allocator metrics.
   Repeat the observed cliff to distinguish real system pressure from policy sensitivity.
4. Separately compare scratch versus direct-GEMM prefill with exact same model,
   outputs, retention budgets and cold/warm sequence. Measure correctness, peak
   active+cached memory and elapsed time. Do not silently change the kernel during
   the memory-governor comparison or assume a speed/memory win without measurement.

No production edits were made. The code-graph query was partial and bound to the
main Higgs index, not a clean index of this detached hardening revision; source,
commit and raw-session evidence were used for this read-only audit. Any actual
symbol change still requires fresh impact analysis and the release verification
required by the repository.
