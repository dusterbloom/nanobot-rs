# Live validation — September 12, 2026

All live checks ran on battery after the user explicitly said “do it anyway.” The isolated server was stopped afterward. No user server configuration was changed.

## Verified results

- **Inference:** exact `RESTART-7319` response for a 1,479-token prompt. Cold request: 13.856 s.
- **Hybrid disk restart:** stopped and restarted the actual Higgs process, then repeated the identical request. Same exact answer; **1,472 cached tokens**; 1.873 s. Disk file remained **96,011,860 bytes**, below its 1 GiB ceiling. This proves measured restart reuse for this hybrid model, not universal bitwise attention-state equality.
- **Checkpoint/reset recovery:** recorded announcement was persisted, notes and reset occurred, exact case-sensitive identifiers survived fresh recovery, and the result was submitted once. Artifact `pass: true`.
- **Durable LCM recovery:** created one persisted summary node, rebuilt the agent, and recovered checksum `7af09cB2-e41D-009x` exactly. One submission, zero forbidden actions, artifact `pass: true`; 50.18 s total.
- **Fixed capacity:** advertised 32,768 total tokens and `basis: configured`. Actual sampled OS pressure was normal during this run. Warning/critical-pressure behavior is covered by regression tests; this live run does not claim those pressure conditions occurred.
- **Three-turn smoke:** all three new-binary answers correct. Wall times: 20.099 s cold, 1.249 s and 1.142 s warm. Provider metrics show 1,992 and 2,033 reused prompt tokens on the warm turns.

The saved old Nanobot binary returned capacity-unavailable messages on all three benchmark turns because it cannot consume the new configured capacity basis. Its zero exit status was not a successful inference result. A matched speedup cannot be claimed. The benchmark script also looks for timing traces on stderr, while this build writes them to stdout; wall times and provider metrics above were inspected directly.

Release validation preceding these checks: 3,036 Nanobot tests and 796 Higgs tests passed; both release builds passed. The two ignored live recovery tests then passed as well.

Structured evidence: [LIVE-RESULTS.json](LIVE-RESULTS.json). Full logs and isolated cache/database fixtures: `/tmp/kv-review-20260912/live/`.
