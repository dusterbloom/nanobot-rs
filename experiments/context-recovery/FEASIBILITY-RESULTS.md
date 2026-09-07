# Scheduled reset feasibility — 2026-09-06

The same 20-update workload completed on this machine with prescribed checkpoint/reset after every submission. Exact recovery correctness failed: 13/20 snapshots matched. Execution feasibility and correct endurance are separate gates.

| Measure | Result |
|---|---:|
| Submitted updates | 20/20 |
| Exact correct snapshots | 13/20 |
| Scheduled resets / next-window handoffs | 20 / 19 |
| Notes writes / reads | 20 / 19 |
| Time | 985.3 seconds (16.4 minutes) |
| Capacity rejections / pending turns | 0 / 0 |
| LCM boundaries / downshifts | 0 / 0 |
| Duplicate submissions / external actions | 0 / 0 |
| Requests / output tokens | 79 / 10809 |
| Largest actual prompt | 5445 tokens |
| Safe total / published prompt capacity | 18432 / 14336 tokens throughout |
| Actual response reservation | 2048 tokens |
| Peak sampled Higgs physical footprint | 17.33 GiB |
| Peak sampled MLX active allocation | 13.69 GiB |
| Peak system compressor-resident memory | 2.25 GiB |
| Memory samples / sampling errors | 197 / 0 |
| Server restarts / watchdog termination | 0 / 0 |

The fixture JSON is identical to the prior autonomous 20-update run. Same 12288 context ceiling, native FP32 model and corrected overflow-order implementation. This diagnostic policy prescribes each boundary and removes the mandatory post-submit context-status step. It therefore establishes a workable execution schedule, not that the earlier autonomous policy was feasible or that resets caused all of the difference. The model still authors notes and executes tools; no expected-state oracle supplies checkpoints or corrects submissions. Missing prescribed boundaries fail the control.

## First failure and subsequent drift

Zero-based revision 8 (the ninth update) required carrying checksum `7aF07-bC9x-00Q` forward. The preceding revision-7 checkpoint contained that value. The model read the checkpoint; the actual next provider request contained the correct value and did not contain the later invented `8bR29-mD7y-01P`. The model nevertheless submitted the invented value and then wrote it into notes. No compaction, admission failure or missing notes read explains this first error.

Wrong revisions: 8, 9, 10, 14, 16, 17, 18. All seven contain checksum errors; revision 10 additionally added an extra `result` JSON nesting level. Removing that wrapper alone would not make revision 10 correct. The scorer records these unchanged, provides no correctness feedback, and never repairs them. A later authoritative checksum restores correctness temporarily. Valid tool-call syntax and durable writes do not guarantee semantic preservation.

## Memory interpretation

The physical-footprint measurements use macOS `proc_pid_rusage` v2, not RSS as a proxy. RSS, raw `vm_stat` counters, swap usage, MLX counters and capacity envelopes are also retained. Five-second samples do not capture every instantaneous peak. The different maxima are not simultaneous and must not be added together.

The system already had about 2429.62 MiB swap occupied at startup; sampled swap-out deltas remained zero. Free-page minimum was about 76.5 MiB, but the run continued successfully: free pages alone are not total reclaimable/allocatable RAM. No OOM, allocation failure or pressure-driven capacity reduction occurred during the control. This does not prove the machine was never memory-constrained during earlier runs.

## Decision and reproducibility

Execution feasibility: **PASS for this schedule/run**. Exact recovery feasibility: **FAIL**. Autonomous comparison was not rerun because the correctness prerequisite failed. Next work should isolate checksum copying and checkpoint representation, including strict output-shape validation, before using this control as a reliability baseline. Do not compare 16.4-minute completed work against the earlier truncated runs as if completion coverage were matched.

Run-time HEAD was `0be0e65` with uncommitted harness changes subsequently committed as `fdb4ebf`; recorded hashes of `endurance_eval.rs` and `endurance.py` match that commit. Production nanobot remains corrected `7b8b24a`, Higgs `dd6730133`. Release harness validation: 3004 passed, 0 failed, 31 ignored. Existing optional/decision prompt strings verified unchanged. Installed Higgs defaults restored; binary/library mappings and hashes verified; READY smoke passed. Runtime evidence: `BINARY-PROVENANCE-AFTER.json`.

Run from the repository in tmux, using a new output directory:

```sh
ENDURANCE_LONG_FORM_MIN_TOKENS=2048 ENDURANCE_RESET_HANDOFF=1 python3 experiments/context-recovery/endurance.py experiments/context-recovery/endurance-feasibility-new --updates 20 --ceiling 12288 --minutes 30 --arms B --policy scheduled
```

Counter limitation: `delivered_update_bytes` and `delivered_update_estimated_window_equivalents` count `next_task` tool deliveries, while this driver supplies updates directly. Their zero values do not mean zero delivered updates and must not be used as workload measurements.

Compact evidence: `feasibility-evidence.json`, `feasibility-first-error.json`. Raw immutable replay, fixtures, receipts and telemetry: `endurance-feasibility-scheduled/`. Driver log: `/private/tmp/feasibility-control.log`; tmux run exited 0 with valid measurement and explicit failed correctness score.
