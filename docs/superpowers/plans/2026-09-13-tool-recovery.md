# Tool Recovery Implementation Plan

> **For agentic workers:** Use subagent-driven-development to implement these approved tasks. Track test evidence and review findings here.

**Goal:** Prevent predictable tool-use errors and bound recovery while preserving truthful results, execution safety, and durable replay.

**Architecture:** Repair the existing route/execute/persist path. Keep concise operational contracts in delivered schemas and scope completion markers precisely. Add bounded recovery to existing turn control; no second pipeline or model-specific flags.

**Tech Stack:** Rust 2021, Tokio, SQLite, existing release tests.

## Global constraints

- One production hot path; no new organizing modules or dependencies.
- Preserve exact requested command bytes, raw result bytes/status, permission/workspace/taint checks, and protocol pairing.
- Cache replay must identify its original durable result and must never imply a new execution.
- No heuristic semantic deduplication or automatic rewriting/retrying of shell commands or side effects.
- Existing running nanobot session and installation remain untouched.
- GitNexus upstream impact before symbol edits; report HIGH/CRITICAL risks; detect changes before completion/commit.
- Release builds/tests only. Regression tests must exercise behavior; guidance serialization tests protect the delivered contract, not source prose.

## Task 1: Delivered tool contracts (Luna)

Files: src/agent/tools/registry.rs, shell.rs, read_skill.rs, stash_search.rs; directly associated tests.

- [x] Add failing boundary tests for loss of important tool guidance in the local delivered schema, useful bounded skill discovery, and stored-result completion that does not imply file completion.
- [x] Preserve concise operational descriptions intentionally instead of sentence truncation for the affected tools; keep ordinary descriptions bounded.
- [x] Explain pipefail and producer-native limits before use. Add conditional SIGPIPE diagnostic after nonzero exit without changing status, raw stdout/stderr, or command execution.
- [x] Make discovery usable without reading an XML opener; distinguish listing unknown skills from loading known names.
- [x] Scope pagination wrapper markers to the stored result and preserve exact artifact bytes/cursors.
- [x] Run focused release tests; record before/after results.

## Task 2: Replay, persistence, and bounded correction (Sol)

Files: src/agent/tool_guard.rs, router.rs, tool_engine.rs, agent_loop/shared.rs, agent_loop/tests.rs, lease.rs, src/session/db.rs; existing supporting types only as required.

- [x] Protect three identical successful explicit-cwd calls through the real agent hot path and reproduce omitted-cwd lookup/storage mismatch. The running binary’s explicit-cwd discrepancy remains unexplained; do not claim a proven live root cause.
- [x] Materialize execution defaults once before routing. Preserve requested argument provenance and exact command bytes. Verify different cwd remains distinct.
- [x] Persist a distinct cached-replay disposition with original call ID/result digest; canonicalize semantic argument digests without modifying raw command strings. Keep durable success ordering and cache invalidation.
- [x] Fix capacity-interruption persistence failure handling so it never claims unsaved output was saved; test failure injection and no provider retry.
- [x] Add admitted-attempt accounting including failed calls, preserve batch reservations and success lease, and retain durable admission events for replay/audit. Compaction within a turn must not reset accounting. Do not introduce automatic logical-turn resumption across new user turns.
- [x] Use repeated read-only evidence for a bounded recovery nudge, never to skip executions. Add a separate neutral advisory for independently executed exec calls returning identical nonempty output; do not infer command equivalence. Preserve advancing cursors/different queries/changed state. Remove unsupported claims that sufficient evidence already exists.
- [x] Add focused end-to-end tests for repeated variants, failure attempts, compaction, cross-turn replay rejection, final honest partial outcome, exactly-once side effects, and protocol pairs.

## Task 3: Integration and evaluation (controller/reviewer)

- [x] Review each task's diff and test evidence, resolve findings, and verify combined protocol/replay suites.
- [x] Run cargo build --release and cargo test --release; isolate environment-dependent failures before attributing them to changes.
- [x] Assess speed-validation isolation and document the limitation: scripts/turn_bench.sh uses the active user configuration/database. Run only isolated next-action inference probes without executing generated calls. No matched speed benchmark or measured improvement is claimed; see the validation report.
- [x] Run GitNexus change analysis and final independent review; preserve user changes and provide branch/build/test evidence.

## Execution record

Base: f32f5be. Branch: codex/tool-recovery. Worktree: /private/tmp/nanobot-tool-recovery.

Design adjustment: automatic restoration of prior results and pending-call blocking on every new turn was rejected by automatic approval review because it can reuse stale evidence or block a session indefinitely. That implementation was not applied. Recovery remains in the current logical turn; interrupted turns retain durable evidence and terminate honestly. CLI session resume remains distinct from automatic tool execution resumption.

Baseline release library tests: 2996 passed, 27 ignored; local mock HTTP tests require socket access. Initial DB regressions failed for noncanonical argument artifacts and missing CachedReplay type; both passed after fixes. Interruption persistence fault regression failed before error propagation was added.

Independent review (Sol) completed across the combined diff: two findings assigned to recovery_core in one fix wave. P1: near-limit cached receipts must remain stable across history reload and reference the original execution. P2: compact skill discovery must preserve unmet-requirement status. Other reviewed replay, argument, failure-budget, and pagination invariants were coherent. Task1 source edits are complete; final suite must include the 80-skill fixture. DB regression fixture for different-turn rejection was corrected to use an actual different request ID; rerun in final suite.

Current ownership: recovery_core owns all remaining review fixes and integration test adjustments. Parent owns final verification/documentation/commit. recovery_review will perform a scoped re-review once the fix wave completes. No automatic cross-turn restoration exists in the current diff.

Focused release regressions passed for explicit and omitted cwd replay, distinct cwd, new user turns, empty directory reads, repeated read/exec evidence, near-limit cached receipt reload, and unavailable skills. Independent scoped re-review accepted receipt stability, skill status, and advisory-only exec handling after a test/advisory wording mismatch was aligned. Final GitNexus change analysis reports the expected 12 source/test files, 108 symbols, 42 flows, critical aggregate risk; full suite validation follows.

Final acceptance: cargo test --release passed 3061 tests, 31 ignored, zero failures; cargo build --release passed on final source. Formatting and diff checks passed. Three full-suite fixture failures were corrected without weakening production invariants: recovery wording, materialized-cwd guard seed, and an undersized conflict fixture whose background compaction consumed the scripted provider queue. See validation report for evidence and limitations.
