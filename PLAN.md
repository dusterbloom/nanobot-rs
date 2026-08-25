# Current Plan

- [x] Squash-land refactoring branch onto main (d0601fc; 63 commits, one
  commit; branch ref kept as history; branch tip repaired e32fd13).
- [x] Pre-squash: port 5 superseded main fixes (3cdd294), drain Aug audit
  deletions -687 LOC (b754d6b), supersession checklist + refreshed audit in
  .planning/.
- [ ] Post-squash backlog (in order):
  1. Error-protocol Phases 2-4: migrate 18 remaining tool files to
     execute_typed, flip trait off String returns, finish_reason enum
     (research: docs/research/2026-08-06-error-conventions-and-host-bridge.md).
  2. Turn bench vs higgs local server (informational cloud run impossible:
     config is local-default, higgs down). scripts/turn_bench.sh once higgs
     serves again.
  3. God-fn/file splits needing their own plans: step_call_llm, shared.rs,
     session/db.rs split, cmd_agent (see .planning/TECH_DEBT_AUDIT.md rows
     13-15).
- Note: origin/main is behind local main; push when ready.
