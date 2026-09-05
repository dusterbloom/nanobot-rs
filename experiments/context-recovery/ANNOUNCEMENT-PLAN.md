# Context announcement recovery

Approved: explicit reset announcement triggers a bounded constrained tool retry;
only successful durable checkpoint execution permits reset. Model chooses timing.

- [x] Inspect actual failure and existing recovery path.
- [x] Impact analysis: shared retry LOW; test tool UNKNOWN dynamic dispatch, test runner HIGH (three cfg(test) callers).
- [x] Add narrow exact decision marker recovery, restricted notes/reset catalog, and regression coverage. Preserve ordinary prose, continue decisions, cloud behavior, real calls, and blocked-call handling.
- [x] Verify/repair Higgs required-tool decoding: current request ignores tool_choice; existing JSON/regex FSM can support the existing tool-call envelope.
- [x] Require a successful current-turn durable checkpoint before fixture reset; cover failures and duplicates.
- [x] Run release checks and real Escha recovery receipt test through tmux.
- [x] Continue frozen handoff OFF/ON replay and record results separately from autonomous choice/endurance.
- [x] Update tracker and restore verified service binaries/defaults.

No reset is inferred from a quoted example or ordinary capability prose. No automatic
threshold decision is introduced. Constrained syntax does not establish note accuracy.

Evidence so far: announcement RED exit101; forced-recovery14/14 and fixture2/2 GREEN. Persisted-scaffold and parent-directory fsync changes passed final release checks. Higgs required-tool RED exited101 for missing constructor/resolver before implementation.

Review found and resolved two integration defects before live validation:
- Failed constrained retries must retract a streamed announcement and clear the
  streamed flag so the truthful failure replacement is delivered.
- Required retained-cache continuation conflicts with constrained decoding;
  the forced request now strips session-id, lease, and cache-policy controls
  from its wire copy, preserving durable context/session state.

The pending announcement and runtime instruction are persisted using the existing
scaffold mechanism, so a successful notes call cannot erase the reset obligation.
Independent source review found no remaining concrete blocker. Final release suites and live grammar, announcement, fresh-session recovery,
frozen handoff and matched endurance checks are recorded in ANNOUNCEMENT-RESULTS.md.

## Endurance-discovered overflow ordering defect

- [x] Confirm actual replay array reversal and analyze keep_recent_within_budget impact (LOW; direct caller trim_to_fit_with_age, 20 upstream hits).
- [x] Reproduce chronological current-turn failure, fix reverse walk, run release validation.
- [x] Install corrected binary and update provenance; preserve pre-fix endurance scores.
