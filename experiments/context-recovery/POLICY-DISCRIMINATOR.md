# Optional versus explicit context-management decision

Approved follow-up to ENDURANCE-RESULTS.md. Keep the B interface (existing
LCM/retrieval plus notes/history/reset) in both arms. Change only the initial
system policy: optional use versus required post-submission headroom inspection
and an operational continue/retrieve/checkpoint decision. No reset threshold or
preferred action is prescribed. No per-update reminder or oracle correction.

Run 20 updates, 16K context, 45-minute limit, continuous Higgs per arm. Fresh
boot only between arms. Use the same release binary/config/fixture. Inspect
instruction compliance separately from exact snapshots and sustained progress.
Capacity suspension ends the attempt; saved work is not proof of resumed work.
Three observed boundaries remain the minimum endurance-coverage gate.

- [x] Add policy selection and record it in requests/results/provenance.
- [x] Validate release regressions and wire policy/tool parity.
- [x] Run optional and decision-policy arms in tmux, preserve all artifacts.
- [x] Report choices, correctness, interruptions and coverage separately.

Parallel Astra investigations own LOCAL-PORTAL-ASSESSMENT.md and
SHRINK-EXACT-ASSESSMENT.md. They must not alter runtime/test processes.

Results: POLICY-RESULTS.md. Both bounded attempts ended on capacity; neither
achieved repeated-boundary endurance. No production behavior changes.
