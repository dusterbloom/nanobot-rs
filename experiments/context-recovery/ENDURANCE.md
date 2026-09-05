# Autonomous context management and endurance experiment

Goal: measure self-chosen context resets and sustained progress without restarting
Higgs to hide capacity degradation. Extend the existing test-only harness.

Design:
- A uses existing LCM and retrieval. B additionally exposes notes/reset controls.
  Both receive accurate context telemetry, the same task instructions, explicit
  result schema, and the same ordered stream of updates.
- Updates arrive as real user turns in one continuing session; the agent submits
  the current project snapshot after each. No checkpoint-only turn or reset reminder is sent. Task state must
  persist across many updates; corrections supersede old state and the original
  checksum/receipt remain needed later. Diagnostic appendices grow the context.
- B chooses whether/when to inspect usage, write notes, and reset. The successful
  durable batch boundary hook is reused. History is refreshed from SQLite after
  each batch across all generated sessions, rather than remaining a static seed.
- Output status fields have enums. Submission is scored at EVERY update, without
  revealing oracle answers in tool results. Duplicate/skipped submissions and
  prohibited actions are recorded; finishing early fails progress coverage.
- Keep one Higgs boot throughout each arm, including all B resets. Fresh boot
  between arms is allowed for a matched starting state; never during an arm.
- Retain production preflight/compaction. Count B fallback compactions explicitly:
  a correct final artifact alone cannot prove autonomous context management.
- Fast live canary: 3 updates at 16K. Main bounded endurance run: 20 updates at 16K,
  45-minute deadline per arm. Initial incoming evidence is shared exactly. Report
  actual source tokens and boundaries: fewer than 3 boundaries is insufficient
  repeated-boundary coverage, even if all snapshots are correct. A longer 16K,
  60-update run is supported by the same driver, with a 90-minute per-arm limit.
- Primary outcomes: completed/correct snapshots, first failure, duplicate actions,
  early termination, voluntary resets, fallback compactions, capacity suspension,
  and whether work remains durably recoverable. Also measure latency, calls,
  output/prompt tokens, context at reset, and server memory/capacity over time.
- Count resets below 25% usage descriptively; do not label them irrational without
  considering the recorded reason. No threshold is prescribed to the model.
- A stall ends the scored attempt and preserves pending state; no hidden retries
  or service restarts. Watchdog cancellation is recorded, not converted to pass.

Implementation plan:
- [x] Add a deterministic update stream and exact schema; test corrections,
  stable identifiers, stream bounds and duplicate submission rejection.
- [x] Extend test tools, dynamic context telemetry and SQLite history refresh;
  retain the original forced-recovery test path and original artifacts.
- [x] Add autonomous runner with bounded deadline and durable per-update scores.
- [x] Add tmux orchestration with per-arm continuous Higgs and sampled metrics.
- [x] Release build, regression checks, wire/schema canary, then main paired attempt (both stopped on capacity).
- [x] Report capability, endurance coverage and interruptions separately in ENDURANCE-RESULTS.md.

No production feature, commit, cloud call or actual external action is introduced.

Regression evidence: the new stream check first failed on the empty stream stub,
then passed with the implementation. All 268 agent-loop regressions pass (12
opt-in tests ignored). Context status reports its previous-batch observation
time and includes both estimated content and the last actual provider count.

Canary-driven corrections (invalid artifacts retained, never scored):
- Tool metadata is immutable and independent of the telemetry lock; querying
  schemas while updating telemetry cannot reacquire that lock.
- Include the existing inspect_tool_result reader. History hydrates old tool
  bodies from SQLite, preserving access across session-scoped handle boundaries.
- Single-turn next_task({}) polling was incompatible with duplicate-call replay.
  Use actual new user turns instead, matching the production conversation path;
  do not weaken the duplicate-call guard or add a nonce workaround.
- Each update includes a required deterministic diagnostic code in its appendix.
  The agent must combine fresh evidence with retained project facts. All fields
  and status enums are specified before inference.
- Normal background compaction starts at turn end and the next foreground turn
  may preempt it, as in production. Deferred work for an explicitly retired reset
  window has no consumer; hard preflight and fallback remain active.

The 8K multi-turn canary was rejected before inference: adaptive output budgeting
requested 6144 output tokens, leaving max_prompt_tokens=2048 for a 4032-token
rendered prompt. Preserve this budget interaction as a separate negative finding.
Use the original 16K configuration for scored autonomy/endurance, keeping adaptive
output budgeting active; record actual output reservations from wire requests.

The 16K retry exposed a second admission mismatch: adaptive output=6144 versus
Higgs configured output ceiling=4096. Endurance uses higgs-endurance.toml with
8192 output ceiling for BOTH arms; the original forced-trial config is preserved.

The first main B arm was stopped as confounded: fallback summaries advertised
lcm_expand, but B lacked that tool. Both arms now retain recall/lcm_expand; B
adds notes/history/reset. The wire audit requires both fallback retrieval tools.
Do not attribute the invalid arm's post-compaction mistakes to model capability.
