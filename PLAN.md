# Current Plan — landing v0.4 (set 2026-08-25)

Owner decisions, in order. All pending.

- [ ] **1. Squash-merge `refactoring/maximum-speed-with-less-code` onto main.**
      Not a fast-forward: branch is 60 ahead / 22 behind (main advanced to `c6103a2`,
      2026-08-02; merge-base `d7a801c`). Reconcile the 22 main-side commits (delegated
      tool invariants, lease accounting, replay verification). Gate TBD — at minimum
      full test suite + release build green before squash. Owed artifacts to fold in
      or explicitly waive: `09-03`/`09-04` SUMMARYs, 3-way smoke transcripts
      (Cloud + Higgs + cluster-remote).
- [ ] **2. Error-protocol Phases 2–4 — after the squash.**
      Phase 0–1 done. Phase 2 remainder: migrate 18 tool files still on
      `execute -> String`; delete legacy `execute` from the 4 already-typed tools
      (message, recall, remember, spawn); flip trait `execute`/`execute_with_context`
      in `src/agent/tools/base.rs` to `ToolResult`. Then Phases 3–4 per protocol plan.
- [ ] **3. Re-run tech-debt audit.** `.planning/TECH_DEBT_AUDIT.md` is stale
      (pre-Phase-09, pre-TUI-move). Fresh run + `quality-sentinel.sh` before new work.
- [ ] **4. Turn bench.** Never ran (local inference server down). Bench against a
      reachable cloud provider first; restore the local server only if a regression
      shows.

Constraints (unchanged):

- Production path stays: channel → agent_loop → provider → tools → reply.
- No feature flags, no parallel pipelines; one way to do each thing.
- Replay history (Exact Turn Replay plan, completed 2026-08) stays implemented as shipped.
