# Durable Subagent Lifecycle and Guaranteed Final Delivery Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task by task, and `test-driven-development` for each behavior change.

**Goal:** Make subagent work durable, progress-aware, non-empty on success, and recoverable after cancellation or process restart without blindly repeating side effects.

**Architecture:** Keep the existing `SubagentManager` as the only execution path. Give every child its own ordinary SQLite session plus one `subagent_tasks` lifecycle row linked to the concrete parent session. Journal child model and tool boundaries through the existing session replay store, checkpoint partial text and iteration progress in the task row, and make SQLite—not `events.jsonl` or an in-memory broadcast—the source of truth for list/check/wait. A 15-iteration soft limit may extend in five-iteration blocks only while measurable progress is occurring, up to a 45-iteration hard limit; every exit path then runs one tools-disabled synthesis call and maps to a typed terminal outcome. `completed` with blank output is forbidden both in Rust and by a SQLite `CHECK` constraint.

**Tech Stack:** Rust 2021, Tokio, rusqlite/SQLite, serde JSON, existing OpenAI-compatible `LLMProvider`, existing replay artifacts/session events, release-only Cargo validation.

---

## Non-negotiable invariants

1. There remains one child execution loop in `src/agent/subagent.rs`; do not add a second durable runner or feature-flagged pipeline.
2. `completed` always contains non-whitespace final text. Provider failure, failed synthesis, cancellation, and restart recovery are `failed` or `interrupted`, never successful sentinel strings.
3. Each provider request is journaled before the call. Its response or failure is journaled before that result can cause a tool execution or another request.
4. Each tool boundary is persisted in order: pre-execute decision, raw outcome, then exact model-visible result.
5. Task state and child transcript are committed before announcement/display. Broadcast channels may wake waiters but never own the result.
6. Iterations extend only for new evidence: a new non-blank partial response or a newly completed successful tool call. Repeated/failed calls and mere elapsed time are not progress.
7. Recovery never automatically re-executes a tool whose durable outcome is unknown. An interrupted task may resume only from its last fully committed boundary.
8. Do not retain `events.jsonl` as a fallback after the SQLite read path ships.

## Budget and outcome contract

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
enum SubagentOutcome {
    Completed { content: String },
    Failed { reason: String, partial: Option<String> },
    Interrupted { reason: String, partial: Option<String> },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct IterationBudget {
    soft_limit: u32,       // default 15
    extension: u32,        // default 5
    hard_limit: u32,       // default 45
    granted_limit: u32,
}
```

At the current granted limit, extend by `extension` only when progress occurred since the previous grant. Otherwise stop research and synthesize. The hard limit bounds tool-enabled work; one tools-disabled synthesis call is always permitted afterward. Synthesis has a default ceiling of 4096 output tokens, clamped only when the provider's actual context window requires it.

## Durable task row

Add this table to the existing idempotent `SCHEMA` in `src/session/db.rs`:

```sql
CREATE TABLE IF NOT EXISTS subagent_tasks (
    task_id             TEXT PRIMARY KEY,
    parent_session_id   TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    child_session_id    TEXT NOT NULL UNIQUE REFERENCES sessions(id) ON DELETE CASCADE,
    label               TEXT NOT NULL,
    task                TEXT NOT NULL,
    model               TEXT NOT NULL,
    status              TEXT NOT NULL
                        CHECK(status IN ('running', 'completed', 'failed', 'interrupted')),
    soft_limit          INTEGER NOT NULL CHECK(soft_limit > 0),
    hard_limit          INTEGER NOT NULL CHECK(hard_limit >= soft_limit),
    granted_limit       INTEGER NOT NULL CHECK(granted_limit >= soft_limit),
    iteration           INTEGER NOT NULL DEFAULT 0 CHECK(iteration >= 0),
    successful_tools    INTEGER NOT NULL DEFAULT 0 CHECK(successful_tools >= 0),
    latest_partial      TEXT,
    result              TEXT,
    error               TEXT,
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL,
    terminal_at         TEXT,
    announced_at        TEXT,
    CHECK(status != 'completed' OR (result IS NOT NULL AND length(trim(result)) > 0)),
    CHECK(status = 'running' OR terminal_at IS NOT NULL)
);
CREATE INDEX IF NOT EXISTS idx_subagent_tasks_parent
    ON subagent_tasks(parent_session_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_subagent_tasks_status
    ON subagent_tasks(status, updated_at DESC);
```

The child session key is `subagent:<task_id>`. Store the original system/user messages in `messages`; store exact provider/tool bytes in the already-existing replay artifacts and `session_events` tables.

---

### Task 1: Prove the failure contract with red tests

**Files:**

- Modify: `src/agent/subagent.rs` (existing test module)
- Modify: `src/session/db.rs` (existing test module)
- Modify: `src/config/schema.rs` (existing test module)

- [ ] **Step 1: Add a scripted provider fixture that returns tool calls through the soft limit, content alongside at least one tool call, and a final tools-disabled response.**

  Keep it in the existing `subagent.rs` test module. Record every received `tools` argument and response-token ceiling so the tests can prove synthesis is tools-disabled and uses 4096 tokens.

- [ ] **Step 2: Add failing lifecycle tests.**

  Cover:

  - tool calls on all 15 ordinary iterations do not become `Completed` with the old sentinel;
  - non-empty content accompanying a tool call becomes `latest_partial`;
  - no progress at iteration 15 stops extension and forces synthesis;
  - progress at iteration 15 grants exactly five more iterations;
  - repeated progress cannot exceed iteration 45;
  - blank synthesis maps to `Failed`, never `Completed`;
  - provider failure retains the last non-blank partial;
  - cancellation maps to `Interrupted` and retains its partial.

- [ ] **Step 3: Add failing DB constraint and transition tests.**

  Prove SQLite rejects `status='completed'` with `NULL`, empty, or whitespace-only `result`. Prove legal transitions are `running -> completed|failed|interrupted`, and a second terminal transition is rejected or is an idempotent replay of the exact same terminal state.

- [ ] **Step 4: Add failing config compatibility tests.**

  Assert defaults and camelCase decoding:

  ```json
  {
    "maxIterations": 15,
    "progressExtensionIterations": 5,
    "hardMaxIterations": 45,
    "finalSynthesisMaxTokens": 4096
  }
  ```

  Existing configs containing only `maxIterations` must continue to deserialize.

- [ ] **Step 5: Run only the new tests and confirm the intended failures.**

  Run: `cargo test --release subagent -- --nocapture`

  Expected: new behavior tests fail for the old sentinel/iteration loop; existing tests continue compiling after fixtures are introduced.

---

### Task 2: Add SQLite lifecycle state and legal transitions

**Files:**

- Modify: `src/session/db.rs`

- [ ] **Step 1: Before editing, run GitNexus impact analysis.**

  Run upstream impact for `SessionDb::open`, `SessionDb::create_session`, and every existing helper whose signature will change. If any result is HIGH or CRITICAL, stop and report the affected processes before editing.

- [ ] **Step 2: Add `subagent_tasks` to `SCHEMA` and define narrow public records.**

  Add `SubagentTaskStatus`, `SubagentTaskRecord`, and `NewSubagentTask`. Keep SQL conversion helpers private to `db.rs`; do not create a lifecycle module.

- [ ] **Step 3: Add one transactional create operation.**

  `create_subagent_task(parent_session_id, task_id, label, task, model, limits)` must create the `subagent:<task_id>` session and running task row in one SQLite transaction, returning both IDs. No Tokio task may start before this commits.

- [ ] **Step 4: Add checkpoint and terminal compare-and-set operations.**

  Implement:

  ```rust
  checkpoint_subagent(task_id, iteration, granted_limit, successful_tools, latest_partial)
  finish_subagent(task_id, SubagentOutcome)
  get_subagent_task(task_id_or_unique_prefix)
  list_subagent_tasks(parent_session_id, limit)
  interrupt_running_subagents(reason)
  mark_subagent_announced(task_id)
  ```

  `finish_subagent` uses `UPDATE ... WHERE status='running'`; validate affected row count and trim completed content before writing it. Prefix lookup must reject ambiguous prefixes.

- [ ] **Step 5: Make restart recovery explicit and durable.**

  `interrupt_running_subagents("process restarted before child completion")` changes every stale running row to interrupted in one transaction, preserving `latest_partial`. It must not alter child messages or replay events.

- [ ] **Step 6: Run the DB tests.**

  Run: `cargo test --release session::db -- --nocapture`

  Expected: task constraints, transitions, lookup ambiguity, and restart recovery pass.

---

### Task 3: Bind every spawn to its concrete parent and child sessions

**Files:**

- Modify: `src/agent/prepare_context.rs`
- Modify: `src/agent/tool_wiring.rs`
- Modify: `src/agent/agent_loop/mod.rs`
- Modify: `src/agent/subagent.rs`

- [ ] **Step 1: Run GitNexus impact analysis before each symbol edit.**

  Analyze upstream impact for `AgentLoopShared::build_tools`, `AgentLoop::new`, `AgentHost::spawn`, and `SubagentManager::new`/`spawn`. Warn before proceeding on HIGH or CRITICAL results.

- [ ] **Step 2: Resolve the foreground SQLite session before constructing tools.**

  In `prepare_context.rs`, move the existing `get_or_resume_with_idle` call before `build_tools`, then pass `&session_meta.id` into `build_tools`. Preserve the current idle-policy, compaction-admission, and tool-registration order otherwise.

- [ ] **Step 3: Carry the concrete parent ID through the existing host path.**

  Add `parent_session_id: String` to `AgentHost`; do not put it into model-supplied `SpawnRequest`. Pass it as a trusted argument to `SubagentManager::spawn`.

- [ ] **Step 4: Inject the existing `Arc<SessionDb>` into `SubagentManager`.**

  Source it from `SwappableCore.sessions` in `AgentLoop::new`. Remove any construction of a second DB handle from the subagent path.

- [ ] **Step 5: Persist before executing.**

  In `spawn`, resolve provider/model/budgets first, call `create_subagent_task`, build the child session's initial messages, and only then invoke `tokio::spawn`. If persistence fails, return a typed spawn error and do not create an in-memory running entry.

- [ ] **Step 6: Add startup recovery at the async loop boundary.**

  At the beginning of `AgentLoop::run`, call the manager recovery method once before consuming inbound messages. Announce previously unannounced interrupted tasks only after their terminal row is committed.

- [ ] **Step 7: Run focused wiring tests.**

  Run: `cargo test --release agent::tool_wiring agent::prepare_context agent::subagent -- --nocapture`

  Expected: spawned rows reference the exact foreground session and a unique child session; a DB error prevents launch.

---

### Task 4: Journal every child model and tool boundary

**Files:**

- Modify: `src/agent/subagent.rs`
- Modify: `src/agent/tool_runner/mod.rs`
- Modify: `src/session/db.rs` only if a small replay helper is missing

- [ ] **Step 1: Run GitNexus impact analysis.**

  Analyze `_run_subagent` and `process_tool_response` upstream. Because `process_tool_response` is shared by pipeline execution, explicitly report its callers and preserve its existing no-journal behavior for pipeline calls.

- [ ] **Step 2: Pass a narrow optional execution observer through the existing common tool-response function.**

  Define the observer beside `process_tool_response` in `tool_runner/mod.rs`. It receives pre-execute, raw result, and model-visible result callbacks; `None` retains current behavior. Do not add a second copy of the tool execution loop.

- [ ] **Step 3: Record model calls using existing replay primitives.**

  For each ordinary and synthesis call:

  1. render the exact wire messages;
  2. `record_model_request(..., ModelCallPurpose::Specialist, ...)`;
  3. call the provider;
  4. record `model_response` or `model_failure` before inspecting tool calls;
  5. append the committed assistant/tool messages to the child session.

  Use a stable child `turn_request_id` derived from `task_id` and the current iteration as `turn_tag`.

- [ ] **Step 4: Record tools through the observer.**

  Persist the exact tool arguments before execution, raw `ToolExecutionResult` immediately after execution, and exact truncated/model-visible text after it is appended. A journal failure aborts further execution and produces `Failed` with the existing partial; it must never continue unrecorded.

- [ ] **Step 5: Checkpoint after each committed iteration.**

  Save iteration number, granted limit, successful tool count, and latest partial only after all model/tool events for that iteration are durable.

- [ ] **Step 6: Add replay tests.**

  Load the child session with the existing replay API and assert request/response/tool order and exact artifact bytes. Inject journal failures at model request, model response, tool pre, tool result, and checkpoint boundaries; assert no later side effect occurs.

- [ ] **Step 7: Run focused replay tests.**

  Run: `cargo test --release agent::subagent session::db agent::tool_runner -- --nocapture`

  Expected: every successful child run replays deterministically; every injected journal fault stops at the last durable boundary.

---

### Task 5: Replace the fixed loop and sentinel with adaptive execution plus mandatory synthesis

**Files:**

- Modify: `src/config/schema.rs`
- Modify: `src/agent/subagent.rs`

- [ ] **Step 1: Run GitNexus impact analysis.**

  Analyze `SubagentTuning`, `resolve_spawn_settings`, and `_run_subagent` upstream before editing.

- [ ] **Step 2: Add validated tuning fields.**

  Add `progress_extension_iterations`, `hard_max_iterations`, and `final_synthesis_max_tokens` with defaults 5, 45, and 4096. Resolve depth-adjusted limits once at spawn. Reject/normalize zero extension and `hard < soft` at the config boundary; do not sprinkle fallback rules inside the loop.

- [ ] **Step 3: Capture partial content on every provider response.**

  Sanitize `response.content` even when tool calls are present. If the result is non-whitespace and differs from the prior partial, persist it and mark progress. Never overwrite a useful partial with blank content.

- [ ] **Step 4: Count only meaningful progress.**

  Track successful committed tool-call IDs plus the partial-content digest. At a grant boundary, extend by five only if either set changed since the previous boundary. Cap the new grant at 45.

- [ ] **Step 5: Always run one tools-disabled synthesis call on loop exhaustion.**

  Append a user instruction that states the task, current partial, tool findings already present in history, and the reason research stopped (`no_progress`, `soft_limit`, or `hard_limit`). Call the same resolved provider/model with `tools=None` and `final_synthesis_max_tokens`. This call is journaled as `ModelCallPurpose::Specialist` and may not invoke tools.

- [ ] **Step 6: Return a typed outcome.**

  - non-blank ordinary final response -> `Completed`;
  - non-blank synthesis -> `Completed`;
  - provider/journal/synthesis error -> `Failed { reason, partial }`;
  - cancellation/restart -> `Interrupted { reason, partial }`.

  Delete `"Subagent completed but produced no final text."`. Treat a provider that returns blank synthesis as `Failed { reason: "final synthesis produced no text", ... }`.

- [ ] **Step 7: Run adaptive-budget tests.**

  Run: `cargo test --release agent::subagent config::schema -- --nocapture`

  Expected: 15/5/45 behavior, 4096-token tools-disabled synthesis, partial preservation, and typed outcomes pass.

---

### Task 6: Make list/check/wait/cancel SQLite-backed

**Files:**

- Modify: `src/agent/subagent.rs`
- Modify: `src/agent/tool_wiring.rs`

- [ ] **Step 1: Run GitNexus impact analysis.**

  Analyze `SubagentManager::list_running`, `wait_for`, `cancel`, `AgentHost::list_subagents`, and `AgentHost::check` upstream.

- [ ] **Step 2: Persist terminal state before any notification.**

  On normal completion/failure, call `finish_subagent`; then announce/display; then `mark_subagent_announced`; finally wake broadcast subscribers and remove the in-memory handle. If any notification fails, the result remains queryable.

- [ ] **Step 3: Make query methods read SQLite.**

  - `list`: query recent task rows for the concrete parent session;
  - `check`: resolve the durable row and render status/result/error/partial;
  - `wait`: check SQLite before subscribing, subscribe only as an optimization, and re-read SQLite after wakeup or timeout;
  - `cancel`: persist `Interrupted` with latest partial before aborting the handle.

  An empty terminal result is an error response, never success.

- [ ] **Step 4: Remove JSONL ownership.**

  Delete `append_event`, `read_event_result`, `read_recent_completed`, rotation code, and all `events.jsonl` messages. Do not keep a compatibility fallback that can disagree with SQLite.

- [ ] **Step 5: Add race tests.**

  Cover completion immediately before subscription, completion during wait, wait timeout, notification receiver dropped, cancellation concurrent with completion, ambiguous ID prefix, and manager reconstruction after completion.

- [ ] **Step 6: Run host and subagent tests.**

  Run: `cargo test --release agent::tool_wiring agent::subagent -- --nocapture`

  Expected: all results survive manager reconstruction and notification loss.

---

### Task 7: Add explicit, side-effect-safe resume

**Files:**

- Modify: `src/agent/host_bridge.rs`
- Modify: `src/agent/tools/spawn.rs`
- Modify: `src/agent/tool_wiring.rs`
- Modify: `src/agent/subagent.rs`
- Modify: `src/session/db.rs`

- [ ] **Step 1: Run GitNexus impact analysis.**

  Analyze the spawn action parser/request types, dispatcher, and `SubagentManager::spawn` before editing.

- [ ] **Step 2: Add `action='resume'` with required `task_id`.**

  Allow only `interrupted` or `failed` tasks. Reject running/completed tasks. Resume the same durable task and child session via a compare-and-set transition back to running; do not create a parallel child record.

- [ ] **Step 3: Reconstruct only the last fully committed model-visible transcript.**

  Use child `messages` and replay validation. If replay ends after `tool_pre_execute` but before `tool_execute`, append a synthetic tool result saying the prior process stopped before a durable outcome was recorded and the model must decide whether retrying is safe. Never call the tool automatically.

- [ ] **Step 4: Resume budgets from checkpoints.**

  Preserve `iteration`, progress marks, partial, and the 45-iteration lifetime cap. A restart does not reset budget. The mandatory final synthesis remains available once.

- [ ] **Step 5: Add resume/restart tests.**

  Cover a clean model boundary, a completed read-only tool, an unknown mutating-tool outcome, exhausted hard limit, and a second process racing to resume the same task.

- [ ] **Step 6: Run focused tests.**

  Run: `cargo test --release agent::subagent agent::tools::spawn agent::tool_wiring session::db -- --nocapture`

  Expected: one runner wins the resume CAS; no uncertain tool is automatically repeated.

---

### Task 8: End-to-end replay and release verification

**Files:**

- Modify only if failures expose an implementation defect; do not weaken assertions.

- [ ] **Step 1: Run GitNexus change detection before final verification.**

  Run: `node .gitnexus/run.cjs detect-changes --repo nanobot-rs`

  Then compare with main using the available GitNexus compare scope. Confirm only subagent lifecycle, session replay, tool-host wiring, and configuration flows changed. Investigate any unrelated process.

- [ ] **Step 2: Run formatting and static diff checks.**

  Run:

  ```bash
  cargo fmt --all -- --check
  git diff --check
  ```

- [ ] **Step 3: Run the complete release test suite.**

  Run: `cargo test --release`

  Expected: all tests pass with no ignored new lifecycle test.

- [ ] **Step 4: Run the required release build.**

  Run: `cargo build --release`

  Expected: successful release binary.

- [ ] **Step 5: Run the agent-loop matched benchmark.**

  Run: `scripts/turn_bench.sh`

  Expected: no material foreground-turn regression; include before/after numbers in the handoff.

- [ ] **Step 6: Exercise a real end-to-end child.**

  Start the release binary, spawn a child that needs more than 15 tool steps, and capture:

  - parent session ID and child task/session IDs;
  - a progress-gated extension;
  - a non-empty 4096-token-cap synthesis;
  - DB-backed check after the in-memory manager is restarted;
  - explicit resume after interruption;
  - replay validation of the exact child session.

  Kill/restart only the Nanobot process, not SQLite. Verify the child becomes durably `interrupted`, its partial is readable, and no tool is automatically repeated.

- [ ] **Step 7: Confirm removal of the old failure path.**

  Run:

  ```bash
  rg -n "Subagent completed but produced no final text|events\.jsonl|read_event_result|append_event" src
  ```

  Expected: no matches in the production path.

- [ ] **Step 8: Update this plan with evidence.**

  Check off completed steps and append the release binary hash, test summary, benchmark result, and real child task/session IDs. Do not claim shipped until the SQLite rows and replay artifacts have been inspected directly.

## Acceptance criteria

- A child that uses tools through iteration 15 either earns bounded progress extensions or transitions to tools-disabled synthesis.
- The maximum tool-enabled lifetime is 45 iterations; final synthesis gets up to 4096 tokens.
- Text emitted alongside tool calls is queryable while the task is still running.
- `completed` plus blank output is impossible through both API and direct SQL.
- Model/tool history, checkpoints, partial output, and terminal state survive manager and process restart.
- Check/wait/list return SQLite truth even when broadcasts or terminal display fail.
- Cancellation and restart preserve partial output as `interrupted`.
- Explicit resume never automatically repeats a tool with an unknown durable outcome.
- The old `events.jsonl` result path and empty-success sentinel are gone.
- `cargo test --release`, `cargo build --release`, replay validation, and `scripts/turn_bench.sh` all pass.
