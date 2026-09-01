# Session Replay Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make failed tools, persistence failures, convergence limits, and turn outcomes truthful and bounded, then prove the repaired release through deterministic and live end-to-end replay before installing it.

**Architecture:** Keep the existing channel → agent loop → provider → tools → reply path. Classify each tool execution once, carry typed persistence and terminal outcomes through the current hot path, replace ineffective synthetic convergence prompts with one recorded terminal `tool_choice: none` request, and validate the compound behavior inside the existing agent-loop test harness. Exact SQLite replay remains the diagnostic source; no production replay pipeline is added.

**Tech Stack:** Rust 2021, Tokio, async-trait, rusqlite/SQLite, serde JSON, existing OpenAI-compatible provider, existing agent-loop and CLI smoke-test harnesses, tmux, GitNexus.

## Global Constraints

- Production remains one path: channel → agent loop → provider → tools → reply.
- Do not add a production replay module, alternate agent loop, protocol mode flag, or incompatible/destructive SQL migration.
- Tool definitions must remain byte-identical during a retained local session; terminal recovery sends the same definitions with `tool_choice: none`.
- No provider call or tool side effect may occur before its required SQLite protocol prefix is durable.
- A rejected tool call must still receive a matching protocol-valid result receipt.
- Lease renewal remains present with `DEFAULT_MAX_LEASES_PER_TURN = 3`; malformed-renewal correction remains capped by `MAX_LEASE_RENEWAL_REJECTIONS = 2`. `curl` consumes the normal lease and receives no read-only auto-renewal.
- Reuse `ModelCallPurpose::Continuation` for terminal recovery; distinguish it by recorded `tool_choice: "none"` so the rollback binary can deserialize replay events.
- Restore `tool_results.ok` only through a nullable additive column/migration ignored safely by the prior binary; do not backfill historical rows.
- All implementation follows RED → GREEN TDD. Run Rust builds and tests only with `--release`.
- Before editing any symbol, run GitNexus upstream impact and warn on HIGH/CRITICAL risk. Before every commit, run `npx gitnexus detect-changes --repo nanobot-rs` and `git diff --check`.
- Do not overwrite unrelated user work. The existing stash query-range change is retained; the dirty prompt-dump diagnostic is excluded from the release.
- Run `scripts/turn_bench.sh` in matched 20-turn before/after runs because the agent loop/provider path changes.

---

## Execution Setup (before Task 1)

- [ ] Record the baseline commit, current installed binary hash, current candidate hash, and `git status --short` in the SDD ledger. After the baseline release build, copy its binary to `/tmp/nanobot-session-replay-baseline` and record that hash too.
- [ ] Preserve the live database with SQLite's online backup command/API so WAL state is included; record the backup path and integrity-check result in the ledger.
- [ ] Preserve the two current dirty files in a named git stash. Do not drop that stash during implementation.
- [ ] Create ignored worktree `.worktrees/session-replay-reliability` on branch `session-replay-reliability` from the plan commit.
- [ ] Run `cargo build --release` followed by `cargo test --release` in tmux. A pre-existing failure blocks implementation and is recorded verbatim.
- [ ] If the configured local backend is available, run the matched 20-turn baseline now. If it is unavailable, record that fact and capture the baseline immediately after the backend is restored but before installing the candidate.

### Task 1: Make shell/API failures truthful and bound `curl`

**Files:**
- Modify: `src/agent/tools/shell.rs`
- Modify: `src/agent/tool_engine.rs`
- Test: existing `#[cfg(test)]` modules in both files

**Interfaces:**
- Consumes: `ToolExecutionResult::{ok,data,failure}` and `ExecTool::execute`.
- Produces: `detect_api_error_body(ToolExecutionResult) -> ToolExecutionResult`; pipeline-aware shell exit status; `is_read_only_exec_command("curl ...") == false`.

- [ ] **Step 1: Run GitNexus impact before tests or production edits**

Run upstream impact for `ExecTool::execute`, `detect_api_error_body`, and `is_read_only_exec_command`. Record direct callers, affected processes, and risk in the task report before editing.

- [ ] **Step 2: Write failing shell pipeline tests**

Add these cases beside `test_nonzero_exit_is_structured_failure`:

```rust
#[tokio::test]
async fn upstream_pipeline_failure_is_structured_failure() {
    let tool = make_exec_tool(false);
    let mut params = HashMap::new();
    params.insert(
        "command".to_string(),
        serde_json::Value::String("printf nope >&2; exit 7 | cat".to_string()),
    );
    let result = ToolExecutionResult::from(
        tool.execute(params, &ToolContext::sandbox()).await,
    );
    assert!(!result.ok());
    assert!(result.data().contains("Exit code: 7"), "{}", result.data());
}

#[tokio::test]
async fn successful_pipeline_remains_successful() {
    let tool = make_exec_tool(false);
    let mut params = HashMap::new();
    params.insert(
        "command".to_string(),
        serde_json::Value::String("printf ok | cat".to_string()),
    );
    let result = ToolExecutionResult::from(
        tool.execute(params, &ToolContext::sandbox()).await,
    );
    assert!(result.ok());
    assert_eq!(result.data(), "ok");
}
```

- [ ] **Step 3: Write failing API-envelope and `curl` tests**

Add pure classifier cases for these exact bodies:

```rust
for body in [
    r#"{"message":"API rate limit exceeded for 203.0.113.1.","documentation_url":"https://docs.github.com/rest/using-the-rest-api/rate-limits-for-the-rest-api"}"#,
    r#"{"message":"You have exceeded a secondary rate limit.","documentation_url":"https://docs.github.com/rest/using-the-rest-api/rate-limits-for-the-rest-api"}"#,
    r#"{"message":"Bad credentials","documentation_url":"https://docs.github.com/rest"}"#,
] {
    let classified = detect_api_error_body(ToolExecutionResult::success(body));
    assert!(!classified.ok(), "{body}");
}
let ordinary = detect_api_error_body(ToolExecutionResult::success(
    r#"{"message":"release created","documentation_url":"https://example.invalid"}"#,
));
assert!(ordinary.ok());
assert!(!is_read_only_exec_command("curl -sS https://api.github.com/rate_limit"));
```

Update `read_only_exec_classification` so `curl` is in the negative cases.

- [ ] **Step 4: Verify RED**

Run:

```bash
cargo test --release upstream_pipeline_failure_is_structured_failure -- --nocapture
cargo test --release api_error_body -- --nocapture
cargo test --release read_only_exec_classification -- --nocapture
```

Expected: the upstream pipeline, GitHub envelopes, and `curl` assertion fail for the current implementation; the successful pipeline and conservative JSON control pass.

- [ ] **Step 5: Implement minimal failure classification**

Invoke the shell with pipeline failure enabled while retaining the existing command bytes and output formatter:

```rust
Command::new("sh")
    .arg("-o")
    .arg("pipefail")
    .arg("-c")
    .arg(command)
```

Extend `detect_api_error_body` only for exact known GitHub/authentication error messages. Parse JSON first and require a recognized lowercased message (`api rate limit exceeded`, `secondary rate limit`, `bad credentials`, or `requires authentication`); do not classify an arbitrary top-level `message` as failure. Remove `curl` from the `is_read_only_exec_command` matcher. Do not change the shell deny-pattern checks.

- [ ] **Step 6: Verify GREEN and adjacent regressions**

Run:

```bash
cargo test --release upstream_pipeline_failure_is_structured_failure -- --nocapture
cargo test --release successful_pipeline_remains_successful -- --nocapture
cargo test --release api_error_body -- --nocapture
cargo test --release read_only_exec_classification -- --nocapture
cargo test --release read_only_classification_bounds_the_post_exhaustion_auto_renewal -- --nocapture
```

Expected: all pass with no new warning/error output.

- [ ] **Step 7: Scope check and commit**

Run `git diff --check` and GitNexus `detect-changes`; commit only Task 1 files:

```bash
git add src/agent/tools/shell.rs src/agent/tool_engine.rs
git commit -m "fix(tools): report pipeline and API failures"
```

### Task 2: Carry one immutable tool status through every representation

**Files:**
- Modify: `src/agent/tool_runner/mod.rs`
- Modify: `src/agent/tool_runner/tests.rs`
- Modify: `src/agent/tool_engine.rs`
- Modify: `src/session/db.rs`
- Modify: `src/session/filters.rs`
- Test: `src/agent/turn.rs`, `src/agent/protocol.rs`, and existing modules above

**Interfaces:**
- Consumes: Task 1's classified `ToolExecutionResult`.
- Produces: `ToolRunOutcome { tool_call_id, tool_name, data, ok, duration_ms }`; `store_tool_result_immutable_with_status(..., ok: bool)`; `load_tool_result_with_status(...) -> Option<(String, Option<bool>)>`; filtered messages preserving internal `ok`.

- [ ] **Step 1: Run GitNexus impact**

Run upstream impact for `ToolRunOutcome`, `store_tool_result_immutable`, `store_then_render_tool_result`, `filter_history`, and `Turn::from_messages`. Warn before continuing if any result is HIGH/CRITICAL.

- [ ] **Step 2: Write failing status persistence tests**

Restore focused tests from the prior status-aware implementation with these exact assertions:

```rust
assert!(matches!(
    db.store_tool_result_immutable_with_status(
        &session.id, "call_status", "exec", "same exact body", false
    ).await,
    StoredResult::Stored { .. }
));
assert!(matches!(
    db.store_tool_result_immutable_with_status(
        &session.id, "call_status", "exec", "same exact body", true
    ).await,
    StoredResult::Conflict { .. }
));
assert_eq!(
    db.load_tool_result_with_status(&session.id, "call_status").await,
    Some(("same exact body".to_string(), Some(false)))
);
```

Add a reopen test proving `Some(false)` survives `SessionDb::new` on the same file. Extend filtered-history/turn replay tests so a message with metadata `"ok": false` remains false after `get_history` and becomes a failed `Turn::ToolResult`. Keep `test_build_chat_request_strips_internal_tool_result_status` unchanged: `ok` is internal and must not enter native OpenAI tool JSON.

- [ ] **Step 3: Write failing delegated parity test**

Construct a delegated `ToolRunOutcome` whose body does not begin with `Error:` but whose source `ToolExecutionResult` is failed. Assert the provider-facing receipt, `ToolEvent::CallEnd`, and `SessionEventPayload::ToolExecute` all report `ok=false`. The test must fail because `ToolRunOutcome` currently drops the boolean and delegated execution re-infers it from text.

- [ ] **Step 4: Verify RED**

Run:

```bash
cargo test --release immutable_store_compares_tool_name_body_and_status -- --nocapture
cargo test --release tool_result_status_survives_database_reopen -- --nocapture
cargo test --release delegated_tool_status -- --nocapture
cargo test --release legacy_tool_result_preserves_explicit_failure_status -- --nocapture
```

Expected: status-aware store methods/column and delegated `ok` field are missing, and filtered replay drops the explicit failure.

- [ ] **Step 5: Restore the additive status schema and immutable API**

Fresh schema:

```sql
CREATE TABLE IF NOT EXISTS tool_results (
    session_id TEXT NOT NULL,
    tool_call_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    content TEXT NOT NULL,
    ok INTEGER,
    created_at TEXT NOT NULL,
    PRIMARY KEY (session_id, tool_call_id)
);
```

On open, execute the idempotent compatibility migration and ignore only SQLite's duplicate-column result:

```sql
ALTER TABLE tool_results ADD COLUMN ok INTEGER
```

Implement `load_tool_result_with_status` and `store_tool_result_immutable_with_status`. Treat `(tool_name, content, ok)` as immutable identity. Retain `store_tool_result_immutable` as the narrow success-default wrapper for callers that genuinely lack legacy status.

- [ ] **Step 6: Carry status instead of re-inferring it**

Add `pub ok: bool` to `ToolRunOutcome` and set it from the original `ToolExecutionResult::ok()` at every constructor. In delegated post-processing, remove text-based `raw_ok`/`Error:` inference and use `outcome.ok`. Pass the same boolean to `store_then_render_tool_result`, `CallEnd`, audit, replay, and the turn entry. Preserve `ok` in `src/session/filters.rs` when present.

- [ ] **Step 7: Verify GREEN**

Run:

```bash
cargo test --release immutable_store_compares_tool_name_body_and_status -- --nocapture
cargo test --release tool_result_status_survives_database_reopen -- --nocapture
cargo test --release delegated_tool_status -- --nocapture
cargo test --release replay_validates_tool_lifecycle_transitions -- --nocapture
cargo test --release local_textual_replay_formats_failed_tool_result_as_failure -- --nocapture
cargo test --release test_build_chat_request_strips_internal_tool_result_status -- --nocapture
```

Expected: all pass; a single failed execution has the same false status everywhere while native provider JSON still strips the internal field.

- [ ] **Step 8: Scope check and commit**

Run `git diff --check` and GitNexus `detect-changes`; commit:

```bash
git add src/agent/tool_runner/mod.rs src/agent/tool_runner/tests.rs src/agent/tool_engine.rs src/session/db.rs src/session/filters.rs src/agent/turn.rs src/agent/protocol.rs
git commit -m "fix(replay): persist truthful tool status"
```

### Task 3: Fail closed on protocol persistence and preserve terminal outcomes

**Files:**
- Modify: `src/agent/agent_loop/shared.rs`
- Modify: `src/agent/agent_loop/response.rs`
- Modify: `src/agent/prepare_context.rs`
- Modify: `src/agent/finalize_response.rs`
- Modify: `src/agent/tool_engine.rs`
- Modify: `src/agent/router.rs`
- Modify: `src/session/db.rs`
- Test: `src/agent/agent_loop/tests.rs` and existing module tests

**Interfaces:**
- Produces: `TurnOutcome::{Finished,Error,Cancelled,Empty,LimitExhausted}` with `wire_str()`; `TurnContext::persist_pending_protocol_messages(&mut self) -> anyhow::Result<()>`; `journal_tool_call_carrier(...) -> anyhow::Result<()>`; auxiliary request/terminal journaling returning `Result`.

- [ ] **Step 1: Run required HIGH/CRITICAL impact analysis**

Run upstream impact for `run_agent_loop`, `persist_pending_protocol_messages`, `journal_tool_call_carrier`, `journal_aux_request`, `journal_aux_terminal`, and finalization. Record the known CRITICAL blast radius before editing.

- [ ] **Step 2: Write failing typed-outcome tests**

Extend existing replay tests to assert the persisted `turn_finished.outcome` for:

```rust
assert_eq!(turn_outcome(&replay), "error");          // provider failure with rendered text
assert_eq!(turn_outcome(&replay), "cancelled");      // cancellation
assert_eq!(turn_outcome(&replay), "empty");          // empty SSE/content
assert_eq!(turn_outcome(&replay), "finished");       // ordinary final prose
```

Keep `turn_finish_journal_failure_still_returns_the_reply`: failure to write the final journal after a persisted reply still delivers that reply and leaves replay incomplete.

- [ ] **Step 3: Write failing fail-closed tests**

Use existing SQLite triggers/test fault counters and provider/tool invocation atomics. Cover inbound user batch, router auxiliary request, assistant tool carrier, pre-execute, and raw/post-result persistence. Every precondition case asserts:

```rust
assert_eq!(provider_calls.load(Ordering::SeqCst), expected_provider_calls_before_fault);
assert_eq!(tool_calls.load(Ordering::SeqCst), 0);
assert_eq!(turn_outcome(&replay), "error");
```

Replace `router_journal_failure_degrades_instead_of_failing_routing` with a test asserting the auxiliary provider is not called when its request artifact cannot be recorded.

- [ ] **Step 4: Verify RED**

Run:

```bash
cargo test --release persistence_failure_prevents -- --nocapture
cargo test --release router_journal_failure -- --nocapture
cargo test --release failed_local_call -- --nocapture
cargo test --release turn_finish_journal_failure_still_returns_the_reply -- --nocapture
```

Expected: current void persistence APIs continue into at least one forbidden call, and provider error text is persisted as `finished`.

- [ ] **Step 5: Add typed outcome without changing SQL outcome schema**

Add beside the existing loop outcomes:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TurnOutcome {
    Finished,
    Error,
    Cancelled,
    Empty,
    LimitExhausted,
}

impl TurnOutcome {
    pub(crate) const fn wire_str(self) -> &'static str {
        match self {
            Self::Finished => "finished",
            Self::Error => "error",
            Self::Cancelled => "cancelled",
            Self::Empty => "empty",
            Self::LimitExhausted => "limit_exhausted",
        }
    }
}
```

Store it in `TurnContext` and set it at each loop exit. Finalization records `ctx.turn_outcome.wire_str()` and never infers success from non-empty `final_content`.

- [ ] **Step 6: Make persistence APIs return errors and short-circuit**

Change the persistence helper to:

```rust
pub(crate) async fn persist_pending_protocol_messages(&mut self) -> anyhow::Result<()> {
    // collect the same pending group
    let row_ids = self.core.sessions
        .add_messages_checked(&self.session_id, &pending_messages)
        .await?;
    // attach row ids only after the complete transaction succeeds
    Ok(())
}
```

Propagate this result from inbound-message handling, route/tool carrier journaling, delegated/inline receipt injection, and final assistant persistence. Introduce `RouteResult::Error(String)` only where routing must distinguish infrastructure failure from a model-authored break. Make router auxiliary request and response journal functions return `Result` and refuse to consume an unrecorded provider result.

- [ ] **Step 7: Preserve provider error evidence**

In `handle_provider_error`, retain `[LLM Error] {exact provider detail}`. If one local health probe fails, append only:

```text
The local backend health endpoint was unavailable during the follow-up probe.
```

Do not emit “server crashed” unless process-exit evidence exists.

- [ ] **Step 8: Verify GREEN**

Run all tests from Steps 2–4 plus:

```bash
cargo test --release test_tool_call_carrier_persists_before_tool_result -- --nocapture
cargo test --release test_tool_round_is_durable_before_next_provider_call_completes -- --nocapture
cargo test --release checked_message_batch_rolls_back_when_middle_insert_fails -- --nocapture
```

Expected: no external call occurs after a failed prerequisite write; all five outcomes persist truthfully; final-journal failure semantics remain unchanged.

- [ ] **Step 9: Scope check and commit**

Run `git diff --check` and GitNexus `detect-changes`; commit:

```bash
git add src/agent/agent_loop/shared.rs src/agent/agent_loop/response.rs src/agent/prepare_context.rs src/agent/finalize_response.rs src/agent/tool_engine.rs src/agent/router.rs src/session/db.rs src/agent/agent_loop/tests.rs
git commit -m "fix(agent): fail closed and preserve turn outcomes"
```

### Task 4: Replace ineffective convergence scaffolds with a bounded terminal call

**Files:**
- Modify: `src/agent/agent_loop/shared.rs`
- Modify: `src/agent/prepare_context.rs`
- Modify: `src/agent/tool_engine.rs`
- Modify: `src/agent/router.rs`
- Modify: `src/providers/openai_compat.rs`
- Test: `src/agent/agent_loop/heuristics.rs`, `src/agent/agent_loop/tests.rs`, provider tests

**Interfaces:**
- Consumes: Task 3 `TurnOutcome` and existing `ToolChoice::None`.
- Produces: `ProviderCallMode::{Normal,TerminalNoTools}`; at most one terminal call recorded as `Continuation` with `tool_choice:"none"`; no response-boundary or repeat-result scaffold.

- [ ] **Step 1: Run impact analysis and report blast radius**

Run upstream impact for `should_arm_boundary`, `normalize_call_key`, `evaluate_repeated_tool_round`, `step_call_llm`, and `run_agent_loop`. Do not edit until HIGH/CRITICAL results are recorded.

- [ ] **Step 2: Write failing provider-body test**

Build a blocking OpenAI-compatible request with `ToolChoice::None` and assert:

```rust
assert_eq!(body["tool_choice"], serde_json::json!("none"));
assert_eq!(body["tools"], original_tools);
assert_eq!(body["stream"], serde_json::json!(false));
```

This pins the stable tool array rather than removing tools.

- [ ] **Step 3: Write failing agent-loop terminal tests**

Add a dedicated scripted provider that records `chat_with_tool_choice` arguments. Cover three responses: prose, a tool call despite `None`, and provider error/empty. Assert:

```rust
assert_eq!(terminal_calls, 1);
assert_eq!(recorded_choice, ToolChoice::None);
assert_eq!(normal_tools, terminal_tools);
assert_eq!(executed_tools, 0); // ignored-none case
assert_eq!(recorded_request.purpose, ModelCallPurpose::Continuation);
assert_eq!(recorded_request.tool_choice, "none");
```

Prose ends `finished`. Ignored `None`, provider error after the hard limit, or empty terminal output ends `limit_exhausted` and never retries.

- [ ] **Step 4: Write failing scaffold-removal regressions**

Rewrite boundary/repeat tests around behavioral outcomes. Run a side-effect tool followed by another legitimate tool and a repeated-call loop; query persisted messages and assert no content contains:

```text
Report what the previous tool results showed
Your tool results are already in the conversation above
You called the same tool(s) with the same arguments again
```

Assert tool-definition hashes remain byte-identical and the repeated loop reaches exactly one terminal `none` call.

- [ ] **Step 5: Verify RED**

Run:

```bash
cargo test --release terminal_no_tools -- --nocapture
cargo test --release wire_prefix_stable_across_turn_after_side_effect -- --nocapture
cargo test --release cached_duplicate_tool_receipts_trip_loop_circuit_breaker -- --nocapture
cargo test --release convergence_loop_terminates_without_mutating_tool_catalog -- --nocapture
```

Expected: the current loop injects scaffolds/static breaks and never makes the typed terminal `None` call.

- [ ] **Step 6: Implement one terminal-call mode**

Use an enum, not a boolean behavior selector:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProviderCallMode {
    Normal,
    TerminalNoTools,
}
```

Normal mode keeps the streaming path. Terminal mode performs the existing blocking `chat_with_tool_choice` call with the same `tool_defs`, `ToolChoice::None`, no thinking, no API retry, no forced-tool recovery, no validation retry, and no continuation. Record it under existing `ModelCallPurpose::Continuation` with `streaming:false`. Add one per-turn attempted state so every convergence trigger shares the same at-most-once authority.

If terminal prose is returned, emit it once to the delta channel and finish. If tool calls are returned, persist their carrier and matching rejected receipts but execute none. Provider/empty/ignored-none failure after the hard limit records `LimitExhausted`.

- [ ] **Step 7: Remove failed scaffold paths together**

Delete `ResponseBoundary`, `advance_response_boundary`, its `FlowControl` field/initialization, pre-call nudge injection, arming, and inline/delegated boundary rejection. Remove `RepeatBreakerAction::Nudge`, `repeat_nudged`, and duplicate-result scaffold pushes. Keep normalized repeat/no-progress counters as terminal triggers. Keep lease-renewal scaffolds, `DEFAULT_MAX_LEASES_PER_TURN = 3`, and `MAX_LEASE_RENEWAL_REJECTIONS = 2` unchanged.

- [ ] **Step 8: Verify GREEN**

Run all Step 2–5 tests plus lease renewal tests. Expected: no boundary/repeat scaffold persists, one terminal call is bounded, ignored tool calls never execute, and tool arrays stay identical.

- [ ] **Step 9: Scope check and commit**

Run `git diff --check` and GitNexus `detect-changes`; commit:

```bash
git add src/agent/agent_loop/shared.rs src/agent/prepare_context.rs src/agent/tool_engine.rs src/agent/router.rs src/providers/openai_compat.rs src/agent/agent_loop/heuristics.rs src/agent/agent_loop/tests.rs
git commit -m "fix(agent): replace convergence prompts with typed limit"
```

### Task 5: Preserve compacted/retrieved evidence and remove prompt dumping

**Files:**
- Modify: `src/agent/lcm.rs`
- Modify: `src/agent/tools/stash_search.rs`
- Test: existing modules in `lcm.rs` and `stash_search.rs`

**Interfaces:**
- Produces: `mechanical_headlines` preserving a canonical one-line handle; query miss falling back to a supplied range while query hits remain query results; no `NANOBOT_DUMP_PROMPT` release path.

- [ ] **Step 1: Run impact analysis**

Run upstream impact for `mechanical_headlines`, `SearchToolResultTool::execute`, `parse_range`, and `build_chat_request` before edits.

- [ ] **Step 2: Write failing LCM handle regression**

Use `store_then_render_tool_result` with a temporary `SessionDb` to create the real canonical handle. Feed the single-line tool message into `mechanical_headlines` and assert the output contains `TOOL_RESULT_HANDLE`, call id, and bounded excerpt. Do not handcraft a fake handle.

- [ ] **Step 3: Recreate the stash tests before applying the preserved change**

Add the pipe-alternation miss and query-miss-range tests from the preserved dirty patch, then add this hit control:

```rust
let output = execute_inspect(json!({
    "handle": handle,
    "query": "needle",
    "start_line": 1,
    "end_line": 2
})).await;
assert!(output.contains("needle"));
assert!(!output.contains("Lines 1-2"));
```

- [ ] **Step 4: Verify RED**

Run:

```bash
cargo test --release mechanical_headlines -- --nocapture
cargo test --release pipe_query_miss_reports_per_token_counts -- --nocapture
cargo test --release query_miss_falls_back_to_requested_line_range -- --nocapture
cargo test --release query_hit_does_not_fall_back_to_range -- --nocapture
```

Expected: the one-line handle is lost and committed stash logic ignores the supplied range on a miss.

- [ ] **Step 5: Implement the two narrow evidence fixes**

For tool headlines, retain the bounded canonical handle line when there is no separate body line. For stash inspection, preserve literal query search first; on miss, render the explicitly requested range, and when a pipe-separated literal misses include per-token match counts. A hit must keep existing query-result rendering.

The worktree starts from committed source without the dirty prompt-dump diagnostic. Run `rg -n "NANOBOT_DUMP_PROMPT" src/providers/openai_compat.rs`; expected exit is 1/no matches. Exact request artifacts in SQLite are the single prompt diagnostic.

- [ ] **Step 6: Verify GREEN and commit**

Run all Step 4 tests, `git diff --check`, and GitNexus `detect-changes`; commit:

```bash
git add src/agent/lcm.rs src/agent/tools/stash_search.rs
git commit -m "fix(replay): preserve compacted and ranged evidence"
```

### Task 6: Add the compound deterministic replay gate

**Files:**
- Modify: `src/agent/agent_loop/tests.rs`

**Interfaces:**
- Consumes: Tasks 1–5 production behavior and existing `ResponseSequenceProvider`, `build_local_inline_harness_with_iters`, `SessionDb::load_session_replay`.
- Produces: one production-loop replay proving the compound failure cannot recur without a new production module.

- [ ] **Step 1: Run impact analysis for any helper changed**

Tests may add a dedicated provider without impact analysis. If an existing production helper must change, run upstream impact before editing it.

- [ ] **Step 2: Write the compound replay test**

In the existing agent-loop test module, script this sequence:

1. successful evidence-producing result;
2. upstream pipeline failure;
3. zero-exit GitHub primary/secondary rate-limit JSON;
4. varied `curl` calls until the normal lease rejects them;
5. repeated blocked/distinct calls reaching terminal recovery;
6. terminal `ToolChoice::None` prose;
7. a separate empty-stream turn.

Use temporary SQLite/workspace and loopback HTTP only. Assert:

```rust
assert_eq!(terminal_none_calls, 1);
assert!(executed_network_calls
    <= DEFAULT_TOOLS_PER_LEASE * (1 + DEFAULT_MAX_LEASES_PER_TURN));
assert_eq!(turn_outcome(&successful_replay), "finished");
assert_eq!(turn_outcome(&empty_stream_replay), "empty");
assert!(all_failed_tool_events_and_rows_are_false(&replay));
assert!(assistant_tool_pairs_are_complete(&replay));
assert!(!persisted_text.contains("Report what the previous tool results showed"));
```

Decode the recorded terminal request and assert `purpose == Continuation`, `tool_choice == "none"`, `streaming == false`, and its `tools` bytes equal the prior main request.

- [ ] **Step 3: Prove the compound test is load-bearing**

The component tests in Tasks 1–5 already proved each red state. After the compound test first passes, use `apply_patch` to temporarily remove the `secondary rate limit` recognized-message arm added in Task 1. Run the compound test and require failure at the truthful-rate-limit assertion. Restore that exact arm with `apply_patch`, rerun, and require GREEN. Do not commit the mutation.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
cargo test --release compound_session_failure_replay -- --nocapture --test-threads=1
cargo test --release exact_turn_replay_survives_workspace_prompt_changes -- --nocapture
cargo test --release protocol_invariants -- --nocapture
cargo test --release protocol_tests -- --nocapture
cargo test --release lcm_e2e_tests -- --nocapture
```

Expected: all pass; the compound test exercises the real `AgentLoop`, tool engine, SQLite journal, and finalizer.

- [ ] **Step 5: Scope check and commit**

Run `git diff --check` and GitNexus `detect-changes`; commit:

```bash
git add src/agent/agent_loop/tests.rs
git commit -m "test(agent): replay compound session failure"
```

### Task 7: Whole-branch verification, live replay, and deployment

**Files:**
- Modify: no production source unless a failing gate exposes a regression; any fix restarts TDD and receives its own commit/review.
- Record: SDD reports/ledger and deployment evidence outside tracked source.

**Interfaces:**
- Produces: verified release binary, before/after performance evidence, live semantic replay, compatible rollback artifact, installed hash.

- [ ] **Step 1: Final source and scope review**

Run a broad whole-branch reviewer against the plan-base…HEAD diff. Resolve every Critical/Important finding through one final TDD fix round and scoped re-review. Run GitNexus compare detection against `main` and confirm only planned symbols/flows changed.

- [ ] **Step 2: Fast-forward the reviewed branch into `main`**

Confirm the main checkout is clean because the two original dirty files are still in the named backup stash. From the main checkout, run:

```bash
git merge --ff-only session-replay-reliability
```

Do not pop or drop the stash: its stash-search behavior is now committed and its prompt-dump behavior is intentionally excluded. Record the stash identifier as a recovery artifact.

- [ ] **Step 3: Full release verification on `main` in tmux**

Run in order:

```bash
cargo fmt --all -- --check
cargo build --release
cargo test --release
git diff --check
```

All commands must exit zero. Record exact test counts and warnings; do not claim completion from targeted tests.

- [ ] **Step 4: Matched performance/cache comparison**

With the same machine, provider/model, power state, and background load:

```bash
BIN=/tmp/nanobot-session-replay-baseline OUT=/tmp/nanobot-before scripts/turn_bench.sh 20 bench:session-replay-before
BIN=target/release/nanobot OUT=/tmp/nanobot-after scripts/turn_bench.sh 20 bench:session-replay-after
```

Normalize each run's wall/TTFT/cache rows. There must be zero additional failures, median wall and TTFT regression no greater than 10%, byte-identical tool-definition hashes, and cache-read efficiency no more than five percentage points below baseline. A first >10% result is rerun twice; persistent median regression is no-go.

- [ ] **Step 5: Fresh local-model semantic replay**

Run the latest failed task in a new isolated session against the candidate release and bounded local fixtures, not public GitHub. Query SQLite directly and assert a tool-free final response, complete recorded 25-item result, no unsupported claims, bounded requests/tools, no synthetic boundary rows, and `finished` outcome. Give baseline/candidate transcripts in blinded order to an independent reviewer; its judgment is secondary to the hard oracle.

- [ ] **Step 6: Rollback compatibility**

Back up the candidate DB online. Use the saved prior binary to run read-only `sessions list` and export the candidate smoke session. Confirm it tolerates the additive `ok` column and new outcome strings. Structural replay compatibility is proved by reusing `Continuation`, not by the CLI export alone.

- [ ] **Step 7: Atomic installation and smoke**

Record SHA-256 of `target/release/nanobot`. Copy the installed binary to a hash-addressed sibling. Copy the candidate to a temporary sibling, verify its hash, atomically rename it over the installed path, and verify the installed hash again. Restart through the existing process mechanism, run health checks and one bounded smoke turn, then query `messages`, `session_events`, and `tool_results` for truthful status/outcome.

- [ ] **Step 8: Roll back on any failed deployment gate**

Routine failure restores only the prior binary atomically and restarts it. Restore the database backup only for demonstrated corruption or incompatibility after stopping affected processes. Report exactly which gate failed; do not mark the plan complete.

- [ ] **Step 9: Mark completion**

Only after every gate passes, update the SDD ledger, run final `git status --short`, preserve the live DB backup and rollback binary paths, and report the installed/candidate hashes and final verification evidence.
