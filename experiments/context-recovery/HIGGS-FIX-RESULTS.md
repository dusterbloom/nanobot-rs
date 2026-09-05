# Higgs allocation repair — verification record

Status (2026-09-05): Higgs core fixes `890039d7f` are installed and pushed. Local nightly additionally includes cache-label correction `78da18f13`; its public push awaits explicit destination approval after automatic review rejected it. Installed Higgs SHA-256 starts `e7a811f1`; installed nanobot includes both replay/pending repairs (`5d4a680`, SHA-256 starts `8503b965`). Full release validation passed. Experiment harness/evidence are committed as `6e6642d`.

The quick FP16 experiments passed bounded numerical checks. FP16 KV alone regressed speed because FP32 queries promote KV inside MLX attention. FP16 Q/K/V attention instead showed a warm-repeat decode improvement of 9.8–19.4%, with identical 129-token greedy continuation and a 1.6% lower process peak. This isolated candidate is not installed; long-context quality and HTTP retained-cache behavior remain unvalidated.

The final frozen-input, per-arm-profile-isolated endurance run finished: A 4/4 correct submitted updates, B 5/5; both then suspended with durable pending input. Neither completed 20 updates or three recovery boundaries. See `ENDURANCE-FINAL-ISOLATED.md`. Prior runs below are historical diagnostics, not a passing strategy comparison. Shared persisted learning plus pressure classification and upward hysteresis explained earlier startup variation; see `STARTUP-CAPACITY-DIAGNOSIS.md`.

## Historical verification record

Source: isolated `fix/capacity-runtime-integration`, based on `327e5021e`.
Local nightly has already fast-forwarded to the 30 existing hardening commits at that base.

## Changes

- Native Escha cache estimation charges FP32 (40,960 bytes/token) and 255 slots of maximum allocation slack. Numerical cache precision is unchanged.
- Retained-cache oversized drops, dropped sizes and effective retained byte ceiling are observable. The 16K endurance fixture now reserves 768 MiB; its measured layout requires 736,952,320 bytes.
- Simple inference records serialized prefill/decode peaks, actual cold/retained/radix execution and accepted retained bytes. One route helper preserves worker/cancellation/reservation order across the serving APIs.
- Completed, valid observations feed the existing controller. Qualification excludes incomplete measurements, stale pressure samples, non-normal admission pressure and in-request pressure activity. Changed decisions advance capacity generation.
- Learned absolute cold peaks are converted to an additional charge before adding the model baseline. A realistic 12 GiB resident /14 GiB observed-peak regression caught the prior double count.
- Diagnostics expose fresh allocator active/cached/peak counters, timestamps, raw versus effective pressure, cumulative swap/compression activity, observation counts and qualified cold bands. Startup logs resolved execution and executable/Metal-library identity.
- Best-effort retained requests reserve cold capacity because the worker may bootstrap; required-continuation requests can reserve only their known suffix.

## Kernel defaults

The proposed direct-QGEMM default was rejected by real-model measurements: 90.75 tok/s versus 138.51 tok/s for scratch at 1,024-token chunks. Four fresh-process runs had identical output digests. Keep the existing scratch prefill/native QMV decode defaults with no kernel overrides. See `HIGGS-KERNEL-DEFAULTS.md` and `kernel-bench/` for commands, numbers, scope and limitations.

## Verification so far

- Realistic absolute-peak regression failed before the ledger fix (16,535,624,090 versus 3,650,722,202 additional bytes).
- Final Higgs server unit/regression suite: 770 passed, 0 failed. The earlier three fixture failures were corrected for absolute peak units and required continuation.
- Final focused scratch-dispatch regression passed; QGEMM numerical checks and runtime identity checks also passed during candidate evaluation.
- Installed nanobot now matches the previously validated release artifact: `18d5c9bc123d686596b62bf5aa53d843477db9082a5753e8353ff016486c86bc`. Old executable backed up in `/private/tmp`.

## Deliberate limits

Live learning is supported by the Simple worker used for this Escha deployment. Batch execution uses a separate worker and reports unavailable observations rather than claiming valid measurements. Invalid/incomplete captures do not qualify learning. Phase peaks and actual retained bytes are recorded; undecomposed suffix workspace is not mislabeled as transient evidence. Global OS VM counters are captured separately from allocator counters; neither is a process physical-footprint measurement.

The legacy Rust field `compressor_growth_bytes` is documented as a compression-activity cleanliness signal, never a ledger byte charge. Exported metrics use accurate units. A graph-aware rename preview resolves all four references; the existing Rust field name is retained to avoid an unnecessary API rename.

## Final release validation and installation

Server 770 passed; engine 634 passed (5 ignored); Escha 45 passed; MLX gate 5 passed; runtime identity 3 passed. The separately invoked real-model retention test passed (2 live sessions after 4 conversations). Release build and diff whitespace checks passed. Complete structured graph review: 14 files, 189 symbols, 180 affected processes, critical hot-path scope.

Fix commit `890039d7f` is integrated into local nightly. Installed Higgs and experiment Higgs match the release artifact (SHA-256 `d93d8d3548e2358826b462720898b9594b3b9354c375187a92e30925de3bbabf`); matching Metal library SHA-256 `d4ec42fe79abd9d24922c84a6a160b98e9bba7ebed68dbed844981b1190eafb5`. Active PID 11719 was verified with lsof at installation. Startup without kernel/profile overrides resolves auto→throughput, scratch_matmul, native Escha, 1,024-token chunks and 40,960 cache bytes/token.

Initial live probe started before server readiness and failed connection setup; preserved as `endurance-fixed-allocation`. The actual probe began after confirmed readiness in `endurance-fixed-allocation-ready`.

Remote `fork/nightly` updated successfully to `890039d7f`. Process environment inspection confirms no kernel/profile overrides. First live sample succeeded but was excluded from clean qualification: 22 global compression events, zero new swap-outs, raw pressure normal and effective pressure briefly constrained. Second sample succeeded and incremented clean observations. Three samples with an initial dirty sample cannot alone prove three-clean-sample qualification.

## Live evidence

All three allocation requests returned READY; observations=3, clean=2. Installed nanobot CLI exited 0 and returned VERIFIED. The old 16K endurance prerequisite failed honestly: corrected startup safe total is 18,432 with published prompt maximum 14,336 (recommended output 4,096). A separate paired 12,288-ceiling run preserves the 20-update decision-policy fixture.

During arm B, global compression activity caused further governor downshifts with no new swap-outs. A direct `vmmap -summary` measurement at PID 14592 reported 11.6G physical footprint, 15.1G peak; raw artifact saved under the run. This does not establish host RAM exhaustion. The governor deliberately reacts to compression events even while raw pressure reports Normal.

Small (2,773-token, band-4,096) probe requests cannot restore a 13,312-token capacity: the recovery target is band-limited. Testing upward recovery requires larger admitted cold prompts and three clean samples spanning five minutes per upward step. No upward-recovery claim is made from this probe.

## Endurance outcome (not a passing comparison)

Notes/reset arm B at 12,288: 6/20 submitted, 4 correct, 322.30 seconds, 0 voluntary resets, 6 context inspections, 0 notes/history calls, 2 safety compactions, then CapacityUnavailable. First wrong revision 4 repeated revision 3 state. No forbidden actions, tool errors, wire audit errors, unexpected restarts or telemetry errors. Actual peak prompt 9,200; this run does not cover the former 11K retention cliff or three recovery boundaries.

Compaction arm A did not start: both fresh-server attempts failed the advertised prompt-headroom prerequisite. The separate retry advertised safe total 9,216 / prompt 5,120, with zero observations/downshifts and Normal raw/effective pressure. This is an infrastructure-limited comparison, not evidence that either strategy wins. Runs and exact artifacts: `endurance-fixed-supported/` and `endurance-fixed-supported-A/`.

A third A attempt started successfully on the fresh installed service after restoration (boot `e58e7296-e7fa-4945-89e0-8d4c7541ff44`, startup safe total 18,432 / prompt 14,336). It uses the same harness/fixture and 12,288 ceiling, with readiness checked against that fresh server rather than reloading it again. Wrapper and provenance are preserved in `endurance-fixed-ready-A/`. This demonstrates variable startup capacity; the cause has not yet been established, and earlier blocked attempts remain evidence.

Wire diagnosis establishes a nanobot production regression: SQLite retains current authoritative revision4/5 input, but first outbound requests after safety compaction omit it. Request IDs `3ce93f82` and `c68f0e92` contain old summary plus Continue (and for the latter an empty user message). The resulting stale revision3 submissions are not evidence that the model ignored intact current input. A production replay fix and regression are in progress.

## Pre-replay-fix comparison

| Observed metric | A: compaction | B: notes/reset available |
|---|---:|---:|
| Exact correct / submitted | 3 / 5 | 4 / 6 |
| Planned updates | 20 | 20 |
| First wrong revision | 3 | 4 |
| Elapsed seconds | 414.79 | 322.30 |
| Voluntary resets | 0 | 0 |
| Safety compactions | 2 | 2 |
| Final outcome | CapacityUnavailable | CapacityUnavailable |
| Three-boundary coverage | no | no |

Both are pre-fix failures; B never exercised notes/reset. Timing is not a sound strategy ranking: host pressure varied and neither completed the planned workload. A used the fresh installed service after two blocked startup attempts. Both observed one boot per arm and no wire-schema errors, but the semantic current-update loss escaped the earlier wire audit.

## Nanobot follow-on repair evidence

Root cause: `src/agent/token_budget.rs::keep_recent_within_budget` skipped an oversized newest user message while retaining older history. Initial regression failed (expected authoritative revision4 input; got previous answer), then passed after tail protection. Review exposed the same loss after a tool exchange; the strengthened regression failed again with current `_turn:4` input absent. The refined fix protects the current-turn suffix, maintaining user input and tool protocol together; release verification is in progress. Logs: `/private/tmp/current-user-red.log`, `current-user-green.log`, `current-turn-red.log`, `current-turn-green.log`.

Nanobot refined fix passes all 19 token-budget tests and full release library suite: 2,951 passed, 0 failed, 25 ignored. Fresh graph analysis is complete: 21 changed symbols across 5 currently dirty files / 6 affected processes, HIGH enclosing trimming path; only `token_budget.rs` is the new production change. Release executable build and post-fix live validation follow.

A second production nanobot gap explains pending_capacity_turns=0: repeated typed CapacityExceeded returns TurnPending but its callers only emit saved-as-pending text. The actual record_turn_suspended/record_pending_capacity_turn writes occur only in the separate CapacityUnavailable path. This is not a harness bypass. A persistence repair and regression are being added before final recovery claims.

Pending-413 regression reproduced the gap (exit101, no suspension event). The shared hot path now records suspension and durable pending input before returning TurnPending, reusing the same mechanism as CapacityUnavailable and preserving the existing retry-delay policy. Focused release validation is underway (`/private/tmp/pending-413-red.log`, `pending-413-green.log`). Higgs was stopped during the final compilation to separate our compiler memory load from the next live trials.

Both nanobot repairs committed as `5d4a680` after final full release library suite (2,951 passed, 25 ignored) and release build passed. Installed binary matches `target/release/nanobot`, SHA-256 `8503b96521535dc1749a816fce3b3f7f3e50495d78a9dea3f7918692affd6cdb`. Production commit excludes pre-existing experiment hooks/user files. Post-fix turn check and paired endurance run are launched in `nanobot-postfix-live` with compilation complete.

## First post-fix live result

B submitted 4/4 correct snapshots, then stopped on capacity; 1 durable pending row contains the full revision3 user input (6,665 UTF-8 bytes). No stale snapshot, reset, tool error, wire-schema error, forbidden action or restart occurred. A startup was blocked again. This is correct preservation/suspension evidence, not a 20-update endurance pass.

Reproducibility review found appendix filler read live `src/agent/token_budget.rs`, so production fixes changed appendix text (expected snapshots unchanged). This first post-fix run is not an exact pre/post workload match. Freezing the original source blob before the final A/B run; preserve all prior artifacts.

Turn-benchmark post-fix all three requests succeeded with identical prompt/output/cache token counts to baseline. Wall times before: 18.577/1.975/1.979 seconds; after:29.966/5.661/3.042. Most difference is reported TTFT; no speed improvement or strict speed non-regression claim is made from these host-sensitive runs.

The frozen final run again failed the 12K startup prerequisite (fresh safe total9,216 / prompt5,120). Earlier attribution of startup variation to host headroom was an inference, not established evidence. Investigating persisted learning/cross-arm state before final interpretation.

## Current baseline and FP16 experiment

Native cache metadata correction committed locally to nightly as `78da18f13` after 770 server tests and release build passed. Installed/worktree baseline SHA-256 `e7a811f1dec325741d4a5d69fa91fcdd93a9f9b0d4040a325f7056d72feec7c7`. This describes observed FP32 storage; it does not establish that Escha inherently requires FP32. Public push was rejected twice by automatic approval review despite verifying public fork/admin access; explicit destination approval requested. Remote remains890039d7f until approved.

User requested a fast FP16 feasibility experiment. An isolated candidate will store only dense KV in FP16 and restore views to activation dtype before attention; weights and GDN state remain unchanged. No candidate precision change is installed. Shared build/GPU window handed to kernel_defaults; frozen per-arm-isolated endurance waits for this experiment.

FP16 feasibility interim evidence: direct mixed-dtype SDPA is accepted, but MLX internally promotes K/V to FP32 for FP32 queries. At 1,024/2,048 tokens, actual retained bytes fall by20,971,520/41,943,040 exactly; both greedy digests match. Peak process footprint is essentially unchanged. Initial apparent speedup reversed against a later warmed FP32 control; a final ON repeat and tiny logit check are pending before timing conclusions. This remains an isolated experiment, not an installed precision change.

## Completed FP16 feasibility probe

Both variants passed real-model short-fixture checks with exact 129-token greedy agreement. KV-only FP16 retained 20,480 rather than 40,960 variable bytes/token, but averaged 15.5–18.6% slower decode because MLX promoted K/V for FP32 queries. True FP16 Q/K/V attention restored the output to the original activation dtype and avoided that promotion: the repeat ON run decoded 19.4% faster at 1K and 9.8% faster at 2K than its same-binary OFF control. Prefill was 0.2%/9.6% faster in that repeat; the first ON run had a substantial 1K first-use penalty. Peak process footprint fell from 19.514 to 19.197 GB (1.62%), not by half. Maximum short-fixture logit drift was 0.003558.

These are bounded feasibility results from one ON/OFF/ON sequence, not a production speed guarantee. Packed W2 weights, GDN/recurrent state and the surrounding residual stream were unchanged. DenseMTP, batching, long-context quality and full HTTP cache restoration remain unvalidated. Installed binaries are unchanged. Full commands, hashes, numerical evidence and candidate patches: [fp16-probe/README.md](fp16-probe/README.md).

## Restored service

After all experiments, tmux `recovery-higgs:0.0` runs installed `/Users/peppi/.local/bin/higgs` (PID41674, boot `29551c1f-4b1c-4166-8b85-40dd60433b99`). `lsof` confirms installed executable and adjacent `mlx.metallib`; all recorded hashes still match. Process environment contains no Higgs kernel/precision overrides. Startup resolves auto→throughput, scratch prefill, native Escha, 1,024-token chunks and 40,960 baseline KV bytes/token. The metrics endpoint is live.
