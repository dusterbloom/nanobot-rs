# Exact Turn Replay Implementation

- [x] Add append-only session events and content-addressed replay artifacts.
- [x] Add replay folding, availability classification, and corruption checks.
- [x] Record exact foreground model requests and terminal responses fail-closed.
- [x] Refactor tool execution into pre-execute, execute, and post-execute phases.
- [x] Add whole-turn replay and snapshot regressions.
- [x] Run formatting, build, full tests, turn benchmark, and GitNexus change detection.
  The benchmark harness was invoked but could not run without a local inference server.
- [x] Review fixes (journal-failure degradation, TDD red→green):
  - Durable compaction and persisted replies survive a failed `turn_finished`
    journal write (warn + replay degrades to Incomplete).
  - Router/specialist lanes journal fail-soft (`journal_aux_request` /
    `journal_aux_terminal`); the main provider boundary stays fail-closed.
  - `chat_stream` closes its journal on terminal-persist failure and on
    consumer drop (`StreamCancelGuard`), mirroring `shared.rs` wording.
  - `store_replay_artifact` verifies bytes only when the insert was ignored
    (1 query per fresh store instead of 2; ~782µs per ~60KB artifact,
    measured release, 500 stores).
  - Delegated tools report real implementation-exit timings
    (`ToolRunOutcome.duration_ms`) instead of elapsed/count.
  - Turn benchmark still blocked on local inference server being down.

Constraints:

- Exact means canonical provider-call inputs after protocol and control injection.
- API credentials and HTTP-only metadata are never recorded.
- Existing sessions remain readable and become replayable only from new events onward.
- Replay never calls a provider or executes a tool.
- No feature flag or parallel execution pipeline.
