# Frozen-input, isolated-profile endurance result

Both arms reached correct snapshots and then capacity suspension. Neither completed the planned 20 updates or three recovery boundaries. Notes/reset availability did not cause voluntary reset use; this run does not establish a strategy winner.

| Metric | A: compaction | B: notes/reset available |
|---|---:|---:|
| Correct updates | 4 | 5 |
| Submitted updates | 4 | 5 |
| Seconds | 167.21 | 236.71 |
| Voluntary resets | 0 | 0 |
| Completed safety compactions | 1 | 1 |
| Context inspections | 4 | 5 |
| Durable pending rows | 1 | 1 |
| Peak prompt tokens | 8849 | 9200 |
| Minimum advertised prompt capacity | 5120 | 5120 |
| Governor downshifts | 1 | 2 |

Each arm observed one server boot, no wire-schema errors, no forbidden actions, no tool errors and no telemetry errors. Both test processes exited 0 because the harness records outcomes; the workload pass flag is **false** for both. They were not restarted within an arm.

Input appendix is frozen to Git blob `4d1c0e7dcb769386349fc1bee4d66b3e9d0230a9`; both arms used identical configuration bytes in separate `server-A/` and `server-B/` directories, with independent durable profiles. Compiler and indexing work were excluded from the live run. This eliminates the earlier mutable-fixture and shared-profile defects, while pressure and run-order effects remain.

Pending input verification:
- A: 6706 UTF-8 bytes, SHA-256 `f7c05f4ecd54a0ff1230dae3806ee421117b760728d615fc875191c4b34292c0`.
- B: 6690 UTF-8 bytes, SHA-256 `3cbd8f6cb9a0cd9a2a4261b56b7024aac58ec27bdc26adfe583b49e78817a777`.

Raw local artifacts: `endurance-final-isolated/` (SQLite replay histories, per-arm summaries, telemetry, wire payloads, configs and provenance). Compact terminal summaries and provenance are preserved alongside this report in `endurance-final-evidence.json`.

The current test demonstrates correct preservation/suspension, not resumed execution to completion. A lower-capacity endurance regime and a separate suspension/resume test would be needed to complete the remaining recovery coverage.
