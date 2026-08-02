# Stable Tool Prefix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep Nanobot's tool schema byte-stable inside each Higgs retained-session epoch while making lease exhaustion protocol-valid and bounded to one rejected over-budget batch.

**Architecture:** `Lease` admits complete assistant tool-call batches atomically and owns the one result-annotation format used by inline and delegated execution. The agent loop never removes tools for lease enforcement: a final allowed batch tells the model to answer or renew, and the first over-budget batch is persisted with matching rejection receipts before deterministic termination. Legitimate tool-topology changes rotate the Higgs epoch before the changed request is sent.

**Tech Stack:** Rust 2021, Tokio, serde/serde_json, existing `LLMProvider`, SQLite session fixtures, Higgs retained-session extension, Ratatui TUI.

## Global Constraints

- Preserve the production path: channel → agent loop → provider → tools → reply.
- Within one local retained-session epoch, the serialized tool array must remain byte-identical.
- Behavioral flow control may append messages or reject execution; it may not mutate tool definitions.
- Lease admission is atomic per assistant tool-call batch: all calls execute or none execute.
- Every rejected call must have a matching assistant `tool_calls` carrier and tool-result receipt before persistence.
- Do not add a configuration flag, provider API, protocol mode, fallback pipeline, or new module.
- Keep `NO_PROGRESS_HARD_STOP` for non-lease zero-progress paths.
- Do not change lease size, renewal count, duplicate-call caching, tool-result stashing, Higgs decoding, or forced-tool-recovery timeout behavior.
- Preserve existing user changes in `AGENTS.md`, `CLAUDE.md`, `PLAN.md`, and unrelated untracked specs/plans.
- Run GitNexus impact analysis before editing every existing production symbol. Stop and warn before CRITICAL edits; the known CRITICAL surfaces are listed below.
- Run `node .gitnexus/run.cjs detect-changes` before every commit and verify only the intended symbols/flows are affected.
- Run `node .gitnexus/run.cjs analyze` after every commit so subsequent impact checks use a fresh index.

## Known Impact Baseline

| Symbol | Risk | Direct caller | Affected flows |
|---|---:|---|---:|
| `Lease::record_tool_call` | LOW | tests + `step_execute_tools` | 1 |
| `Lease::progress_signal` | LOW | delegated tool path + tests | 1 |
| `inject_tool_result` | LOW | `execute_tools_inline` | 2 |
| `execute_tools_delegated` | CRITICAL | `step_execute_tools` | 8 |
| `step_pre_call` | CRITICAL | `run_iteration` | 11 |
| `step_execute_tools` | CRITICAL | `run_iteration` | 11 |
| `step_process_response` | CRITICAL | `run_iteration` | 11 |
| `step_call_llm` | CRITICAL | `run_iteration` | 11 |
| `CacheResetReason` | LOW | exhaustive matches | 0 graph edges |
| `cache_status_label` | CRITICAL | TUI cell rendering | 20 |

---

### Task 1: Atomic Lease Admission and Shared Result Annotation

**Files:**
- Modify: `src/agent/lease.rs:44-229`
- Test: `src/agent/lease.rs:232-487`

**Interfaces:**
- Produces: `BatchAdmission::{Admitted, Rejected { remaining: u32 }}`.
- Produces: `Lease::admit_batch(&mut self, count: u32) -> BatchAdmission`.
- Produces: `Lease::annotate_result(&self, body: &str) -> String`.
- Preserves temporarily: `ToolCallResult`, `Lease::record_tool_call`, and
  `Lease::progress_signal`, so this commit compiles before Task 2 migrates their callers.

- [x] **Step 1: Refresh the graph and run impact analysis**

Run:

```bash
node .gitnexus/run.cjs status
node .gitnexus/run.cjs impact record_tool_call --direction upstream --repo nanobot-rs
node .gitnexus/run.cjs impact progress_signal --direction upstream --repo nanobot-rs
```

Expected: fresh index; both symbols remain LOW risk. Record the d=1 callers so none survive the final migration.

- [x] **Step 2: Write failing atomic-admission tests**

Replace the single-call budget test with these tests in `lease.rs`:

```rust
#[test]
fn batch_admission_is_atomic_at_remaining_boundary() {
    let mut lease = Lease::new(3, 1);
    assert_eq!(lease.admit_batch(2), BatchAdmission::Admitted);
    assert_eq!(
        lease.admit_batch(2),
        BatchAdmission::Rejected { remaining: 1 }
    );
    assert_eq!(lease.iterations_used, 2, "rejection must consume nothing");
    assert_eq!(lease.admit_batch(1), BatchAdmission::Admitted);
    assert!(lease.is_exhausted());
}

#[test]
fn admitted_multi_call_batch_consumes_every_call() {
    let mut lease = Lease::new(5, 2);
    assert_eq!(lease.admit_batch(3), BatchAdmission::Admitted);
    assert_eq!(lease.iterations_used, 3);
    assert_eq!(lease.admit_batch(2), BatchAdmission::Admitted);
    assert_eq!(lease.iterations_used, 5);
}
```

- [x] **Step 3: Write failing annotation tests**

Add exact contract tests:

```rust
#[test]
fn result_annotation_reports_post_batch_usage() {
    let mut lease = Lease::new(5, 2);
    assert_eq!(lease.admit_batch(2), BatchAdmission::Admitted);
    assert_eq!(
        lease.annotate_result("payload"),
        "[Lease usage after this batch: 2 of 5 calls — 2 renewals remaining.]\npayload"
    );
}

#[test]
fn final_batch_annotation_requires_answer_or_renewal() {
    let mut lease = Lease::new(2, 3);
    assert_eq!(lease.admit_batch(2), BatchAdmission::Admitted);
    let annotated = lease.annotate_result("payload");
    assert!(annotated.contains("Lease usage after this batch: 2 of 2 calls"));
    assert!(annotated.contains("Lease exhausted"));
    assert!(annotated.contains("findings:/next:/will:"));
    assert!(annotated.contains("Do not request another tool before renewal"));
    assert!(annotated.ends_with("\npayload"));
}
```

- [x] **Step 4: Run the tests to verify RED**

Run:

```bash
cargo test --lib agent::lease::tests -- --nocapture
```

Expected: compile failures for missing `BatchAdmission`, `admit_batch`, and `annotate_result`.

- [x] **Step 5: Implement the typed atomic API**

Add the typed batch API beside the existing single-call compatibility methods:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchAdmission {
    Admitted,
    Rejected { remaining: u32 },
}

impl Lease {
    pub fn admit_batch(&mut self, count: u32) -> BatchAdmission {
        let remaining = self.lease_size.saturating_sub(self.iterations_used);
        if count > remaining {
            return BatchAdmission::Rejected { remaining };
        }
        self.iterations_used = self.iterations_used.saturating_add(count);
        BatchAdmission::Admitted
    }

    pub fn annotate_result(&self, body: &str) -> String {
        let renewals_remaining = self.max_renewals.saturating_sub(self.renewals_used);
        let mut signal = format!(
            "[Lease usage after this batch: {} of {} calls — {} renewals remaining.",
            self.iterations_used, self.lease_size, renewals_remaining
        );
        if self.is_exhausted() {
            signal.push_str(
                " Lease exhausted: your next response must be either a final answer or a \
                 renewal checkpoint containing findings:/next:/will:. Do not request \
                 another tool before renewal."
            );
        }
        signal.push(']');
        format!("{signal}\n{body}")
    }
}
```

Migrate the lease-test helper in the same change:

```rust
fn tick(lease: &mut Lease) -> bool {
    lease.admit_batch(1) == BatchAdmission::Admitted
}
```

Remove the old `record_tool_call_allows_up_to_lease_size_with_no_family_cap`
and `lease_progress_signal_format` tests because Steps 2-3 supersede them.
Keep the compatibility methods themselves until Task 2 migrates their two
production callers; this keeps Task 1's commit buildable. Remove stale comments
saying callers strip tool definitions.

- [x] **Step 6: Run focused tests to verify GREEN**

Run:

```bash
cargo test --lib agent::lease::tests -- --nocapture
```

Expected: all lease tests pass, including exact annotation text and atomic rejection.

- [x] **Step 7: Check scope and commit**

Run:

```bash
node .gitnexus/run.cjs detect-changes
git diff --check
git add src/agent/lease.rs
git commit -m "refactor(lease): admit tool batches atomically"
node .gitnexus/run.cjs analyze
```

Expected GitNexus scope: `Lease` and its unit tests only. Do not stage unrelated files.

---

### Task 2: Stable Lease Enforcement in the Agent Hot Path

**Files:**
- Modify: `src/agent/tool_engine.rs:630-744,1135-1265`
- Modify: `src/agent/agent_loop/shared.rs:302-406,1212-1475,2663-2810`
- Modify: `src/agent/agent_loop/response.rs:471-500`
- Modify: `src/agent/prepare_context.rs:620-638`
- Modify/Test: `src/agent/agent_loop/tests.rs:2775-2840,3440-3575,6015-6240`

**Interfaces:**
- Consumes: `Lease::admit_batch(count) -> BatchAdmission` from Task 1.
- Consumes: `Lease::annotate_result(body) -> String` from Task 1.
- Produces: one deterministic terminal response constant:
  `LEASE_OVER_BUDGET_FINAL: &str`.
- Removes: `LEASE_BLOCKS_BEFORE_STRIP`, `FlowControl::consecutive_lease_blocks`, and all lease-driven `tool_defs.clear()` behavior.

- [x] **Step 1: Refresh the graph, run impact analysis, and report CRITICAL risk before editing**

Run:

```bash
node .gitnexus/run.cjs impact execute_tools_delegated --direction upstream --repo nanobot-rs
node .gitnexus/run.cjs impact inject_tool_result --direction upstream --repo nanobot-rs
node .gitnexus/run.cjs impact step_pre_call --direction upstream --repo nanobot-rs
node .gitnexus/run.cjs impact step_execute_tools --direction upstream --repo nanobot-rs
node .gitnexus/run.cjs impact step_process_response --direction upstream --repo nanobot-rs
```

Expected: `execute_tools_delegated`, `step_pre_call`, `step_execute_tools`, and
`step_process_response` are CRITICAL. Warn before edits and name their direct
caller (`run_iteration`) plus the agent-loop/tool-result flows requiring
regression coverage.

- [x] **Step 2: Extend the wire-recording fixture to capture complete tool arrays**

Change `WireRecordingProvider` in `agent_loop/tests.rs`:

```rust
struct WireRecordingProvider {
    name: String,
    responses: std::sync::Mutex<std::collections::VecDeque<LLMResponse>>,
    calls: std::sync::Mutex<Vec<Vec<Value>>>,
    tool_snapshots: std::sync::Mutex<Vec<Vec<Value>>>,
    higgs_session_cache: bool,
}

fn call_count(&self) -> usize {
    self.calls.lock().unwrap().len()
}

fn tool_call(id: &str, path: &str) -> crate::providers::base::ToolCallRequest {
    let mut arguments = std::collections::HashMap::new();
    arguments.insert("path".to_string(), json!(path));
    crate::providers::base::ToolCallRequest {
        id: id.to_string(),
        name: "list_dir".to_string(),
        arguments,
    }
}

fn tool_response(id: &str, path: &str) -> crate::providers::base::LLMResponse {
    crate::providers::base::LLMResponse {
        content: Some(String::new()),
        tool_calls: vec![Self::tool_call(id, path)],
        finish_reason: "tool_calls".to_string(),
        usage: std::collections::HashMap::new(),
    }
}

fn tool_snapshots(&self) -> Vec<Vec<Value>> {
    self.tool_snapshots.lock().unwrap().clone()
}
```

Initialize `tool_snapshots` and `higgs_session_cache: false` in `new`. In `chat`, record `tools.unwrap_or(&[]).to_vec()` before dequeuing the response. Override `supports_higgs_session_cache` to return the fixture field; Task 3 will add a builder that enables it.

- [x] **Step 3: Replace the sticky-strip convergence assertion with the stable-schema contract**

Rename the current sticky-strip test and make these assertions:

```rust
#[tokio::test]
async fn convergence_stops_first_over_budget_batch_with_stable_tools() {
    let provider = Arc::new(LoopingProvider::new("local-main"));
    let (agent_loop, workspace) =
        build_local_inline_harness_with_iters(provider.clone() as Arc<dyn LLMProvider>, 20);
    let session_key = format!("conv-stable-tools-{}", uuid::Uuid::new_v4());
    let response = agent_loop
        .process_direct("list files forever", &session_key, "test", "offline")
        .await;

    assert_eq!(response, LEASE_OVER_BUDGET_FINAL);
    assert_eq!(
        provider.call_count(),
        crate::agent::lease::DEFAULT_TOOLS_PER_LEASE + 1,
        "one rejected call after the allowance must terminate without another inference"
    );
    let snapshots = provider.tool_snapshots();
    assert!(snapshots.iter().all(|tools| !tools.is_empty()));
    assert!(
        snapshots.windows(2).all(|w| serde_json::to_vec(&w[0]).unwrap()
            == serde_json::to_vec(&w[1]).unwrap()),
        "tool definitions must remain byte-identical across lease exhaustion"
    );
    let _ = std::fs::remove_dir_all(&workspace);
}
```

Update `LoopingProvider` to record complete arrays instead of the old boolean:

```rust
struct LoopingProvider {
    name: String,
    call_count: std::sync::atomic::AtomicU32,
    tool_snapshots: std::sync::Mutex<Vec<Vec<Value>>>,
}

fn tool_snapshots(&self) -> Vec<Vec<Value>> {
    self.tool_snapshots.lock().unwrap().clone()
}
```

Initialize the new field with
`tool_snapshots: std::sync::Mutex::new(Vec::new())`. At the start of `chat`,
before incrementing `call_count`, add:

```rust
self.tool_snapshots
    .lock()
    .unwrap()
    .push(tools.unwrap_or(&[]).to_vec());
```

Delete `saw_tools_absent` and its accessor.

- [x] **Step 4: Add an inline annotation regression to the existing prefix test**

Extend `test_local_wire_prompt_tool_result_appends_only` after `assert_wire_prefix`:

```rust
let result = calls[1]
    .iter()
    .find(|m| m.get("tool_call_id").and_then(Value::as_str) == Some("tc_prefix"))
    .expect("tool result must be present on the continuation call");
let content = result.get("content").and_then(Value::as_str).unwrap_or("");
assert!(
    content.starts_with("[Lease usage after this batch: 1 of 12 calls"),
    "inline results must carry the same lease contract as delegated results: {content}"
);
```

- [x] **Step 5: Add the crossing-boundary protocol test**

Use `WireRecordingProvider` for eleven single `list_dir` calls followed by one
two-call response. The complete test setup and assertions are:

```rust
#[tokio::test]
async fn lease_rejects_crossing_batch_atomically_with_complete_protocol_pairing() {
    let mut responses: Vec<_> = (0..11)
        .map(|n| {
            WireRecordingProvider::tool_response(
                &format!("tc_allowed_{n}"),
                &format!("dir{n}"),
            )
        })
        .collect();
    responses.push(crate::providers::base::LLMResponse {
        content: Some(String::new()),
        tool_calls: vec![
            WireRecordingProvider::tool_call("tc_over_a", "over-a"),
            WireRecordingProvider::tool_call("tc_over_b", "over-b"),
        ],
        finish_reason: "tool_calls".to_string(),
        usage: std::collections::HashMap::new(),
    });
    let provider = Arc::new(WireRecordingProvider::new("local-main", responses));
    let (agent_loop, workspace) = build_local_inline_harness_with_iters(
        provider.clone() as Arc<dyn LLMProvider>,
        20,
    );
    let session_key = format!("lease-crossing-{}", uuid::Uuid::new_v4());
    let response = tokio::time::timeout(
        std::time::Duration::from_secs(30),
        agent_loop.process_direct("keep listing", &session_key, "test", "offline"),
    )
    .await
    .expect("over-budget batch must terminate without another inference");

    let core = agent_loop.shared.core_handle.swappable();
    let meta = core.sessions.get_latest_session(&session_key).await.unwrap();
    let raw = core.sessions.get_all_messages(&meta.id).await;
    let carrier = raw
        .iter()
        .find(|message| {
            message
                .get("tool_calls")
                .and_then(Value::as_array)
                .is_some_and(|calls| {
                    calls.iter().any(|call| {
                        call.get("id").and_then(Value::as_str) == Some("tc_over_a")
                    })
                })
        })
        .expect("over-budget assistant carrier must be persisted");
    let call_ids: Vec<_> = carrier["tool_calls"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|call| call.get("id").and_then(Value::as_str))
        .collect();
    assert_eq!(call_ids, vec!["tc_over_a", "tc_over_b"]);
    for id in ["tc_over_a", "tc_over_b"] {
        let receipt = raw
            .iter()
            .find(|message| {
                message.get("tool_call_id").and_then(Value::as_str) == Some(id)
            })
            .expect("every rejected call must have a result receipt");
        assert!(receipt["content"].as_str().unwrap().contains("lease"));
        assert_eq!(receipt["ok"], json!(false));
    }
    assert_eq!(response, LEASE_OVER_BUDGET_FINAL);
    assert_eq!(provider.call_count(), 12, "must not make a thirteenth inference");
    let _ = std::fs::remove_dir_all(&workspace);
}
```

- [x] **Step 6: Add a valid-renewal continuation test**

This guards the cooperative path: the final admitted result advertises
exhaustion, a valid checkpoint renews without changing the schema, and another
tool call executes normally.

```rust
#[tokio::test]
async fn exhausted_lease_renews_without_changing_tool_schema() {
    let mut responses: Vec<_> = (0..crate::agent::lease::DEFAULT_TOOLS_PER_LEASE)
        .map(|n| {
            WireRecordingProvider::tool_response(
                &format!("tc_before_renewal_{n}"),
                &format!("before-renewal-{n}"),
            )
        })
        .collect();
    responses.push(WireRecordingProvider::text_response(
        "Findings: initial scan complete.\nNext: inspect one final directory.\nWill: call list_dir once.",
    ));
    responses.push(WireRecordingProvider::tool_response(
        "tc_after_renewal",
        "after-renewal",
    ));
    responses.push(WireRecordingProvider::text_response("done after renewal"));

    let provider = Arc::new(WireRecordingProvider::new("local-main", responses));
    let (agent_loop, workspace) = build_local_inline_harness_with_iters(
        provider.clone() as Arc<dyn LLMProvider>,
        20,
    );
    let session_key = format!("lease-renewal-{}", uuid::Uuid::new_v4());
    let response = agent_loop
        .process_direct("inspect until complete", &session_key, "test", "offline")
        .await;

    assert!(response.contains("done after renewal"));
    assert_eq!(
        provider.call_count(),
        crate::agent::lease::DEFAULT_TOOLS_PER_LEASE as usize + 3
    );
    let snapshots = provider.tool_snapshots();
    assert!(snapshots.iter().all(|tools| !tools.is_empty()));
    assert!(snapshots.windows(2).all(|pair| {
        serde_json::to_vec(&pair[0]).unwrap() == serde_json::to_vec(&pair[1]).unwrap()
    }));

    let core = agent_loop.shared.core_handle.swappable();
    let meta = core.sessions.get_latest_session(&session_key).await.unwrap();
    let raw = core.sessions.get_all_messages(&meta.id).await;
    let post_renewal = raw
        .iter()
        .find(|message| {
            message.get("tool_call_id").and_then(Value::as_str)
                == Some("tc_after_renewal")
        })
        .expect("post-renewal tool must execute and persist its result");
    assert_eq!(post_renewal["ok"], json!(true));
    let _ = std::fs::remove_dir_all(&workspace);
}
```

- [x] **Step 7: Run the new tests to verify RED**

Run:

```bash
cargo test --lib agent::agent_loop::tests::test_local_wire_prompt_tool_result_appends_only -- --exact --nocapture
cargo test --lib agent::agent_loop::tests::convergence_stops_first_over_budget_batch_with_stable_tools -- --exact --nocapture
cargo test --lib agent::agent_loop::tests::lease_rejects_crossing_batch_atomically_with_complete_protocol_pairing -- --exact --nocapture
cargo test --lib agent::agent_loop::tests::exhausted_lease_renews_without_changing_tool_schema -- --exact --nocapture
```

Expected failures: inline result lacks annotation; tools become absent after
repeated blocks; over-boundary batch is partially admitted or lacks a complete
assistant/results group. The renewal test is a cooperative-path control and may
already pass before the fix.

- [x] **Step 8: Use the shared annotation in both result paths**

In delegated execution, replace manual `progress_signal` formatting with:

```rust
injected = ctx.flow.lease.annotate_result(&injected);
```

In inline `inject_tool_result`, keep the existing shaping expression intact,
then shadow its result immediately before `ContextBuilder::add_tool_result_*`:

```rust
let data = ctx.flow.lease.annotate_result(&data);
```

Do not change raw stash bytes, tool audit bytes, or `ToolEvent::CallEnd` bytes; only the prompt/session result receives the lease annotation.

- [x] **Step 9: Make lease admission atomic and terminal in `step_execute_tools`**

After routing, batch deduplication, and `working_dir` injection—but before any tool executes—replace the per-call admission loop with:

```rust
let batch_count = u32::try_from(routed_tool_calls.len()).unwrap_or(u32::MAX);
if let BatchAdmission::Rejected { remaining } = ctx.flow.lease.admit_batch(batch_count) {
    let tc_json: Vec<Value> = routed_tool_calls.iter().map(|tc| tc.to_openai_json()).collect();
    ContextBuilder::add_assistant_message(
        &mut ctx.messages,
        response.content.as_deref(),
        Some(&tc_json),
    );
    for tc in &routed_tool_calls {
        let receipt = format!(
            "lease exhausted: {} was not executed — this batch requested {} calls \
             with {} remaining. Write a renewal checkpoint before requesting \
             another tool in a new turn.",
            tc.name, batch_count, remaining
        );
        ContextBuilder::add_tool_result_immutable_with_status(
            &mut ctx.messages,
            &tc.id,
            &tc.name,
            &receipt,
            false,
        );
    }
    ctx.persist_pending_protocol_messages().await;
    ctx.emit_pending_request_metrics(0);
    return StepResult::Done(IterationOutcome::Finished(
        LEASE_OVER_BUDGET_FINAL.to_string(),
    ));
}
```

Define beside the other loop constants:

```rust
pub(crate) const LEASE_OVER_BUDGET_FINAL: &str =
    "I stopped this turn because the model requested another tool batch after \
     exhausting its tool lease. Ask me to continue if you want a fresh lease.";
```

An admitted batch proceeds unchanged. Do not set `round_executed_no_tools` for the terminal lease path.

- [x] **Step 10: Remove the schema mutation and dead strip state**

Delete:

- `LEASE_BLOCKS_BEFORE_STRIP` and its ordering relationship;
- `FlowControl::consecutive_lease_blocks` and its initialization;
- `lease_forced_text_only` and the lease-driven `tool_defs.clear()` block;
- the `!lease_forced_text_only` router-passthrough exception;
- blocked-round increment/reset code that only fed sticky stripping.
- `ToolCallResult`, `Lease::record_tool_call`, and `Lease::progress_signal` after
  confirming `rg -n "record_tool_call|progress_signal|ToolCallResult" src` has no
  production callers.

In `step_process_response`, keep renewal behavior unchanged but delete the
`consecutive_lease_blocks = 0` assignment and its sticky-strip explanation.

Keep `tool_defs` mutable because trio passthrough still restores `saved_tool_defs`. Update comments in `FlowControl`, `step_pre_call`, `step_execute_tools`, and the phantom-claim lease gate so they describe stable execution-time enforcement rather than tools-absent generation.

- [x] **Step 11: Run focused and neighboring regressions to verify GREEN**

Run:

```bash
cargo test --lib agent::lease::tests -- --nocapture
cargo test --lib agent::agent_loop::tests::test_local_wire_prompt_tool_result_appends_only -- --exact --nocapture
cargo test --lib agent::agent_loop::tests::convergence_ -- --nocapture
cargo test --lib agent::agent_loop::tests::lease_rejects_crossing_batch_atomically_with_complete_protocol_pairing -- --exact --nocapture
cargo test --lib agent::agent_loop::tests::exhausted_lease_renews_without_changing_tool_schema -- --exact --nocapture
cargo test --lib agent::tool_engine::tests -- --nocapture
```

Expected: stable serialized tool arrays, immediate termination on the first over-budget batch, complete assistant/result pairing, inline/delegated annotation parity, and unchanged non-lease convergence.

- [x] **Step 12: Check scope and commit**

Run:

```bash
node .gitnexus/run.cjs detect-changes
git diff --check
git add src/agent/lease.rs src/agent/tool_engine.rs src/agent/agent_loop/shared.rs src/agent/agent_loop/response.rs src/agent/prepare_context.rs src/agent/agent_loop/tests.rs
git commit -m "fix(agent): keep tools stable through lease exhaustion"
node .gitnexus/run.cjs analyze
```

Expected scope: tool execution, agent-loop convergence, and prompt-prefix flows. Verify the sole d=1 caller `run_iteration` remains covered by the targeted tests.

---

### Task 3: Rotate Higgs Epochs on Legitimate Tool-Topology Changes

**Files:**
- Modify: `src/turn_stream.rs:389-423,525-546,621-649`
- Modify: `src/tui_app/app.rs:2709-2739,4997-5087`
- Modify: `src/agent/agent_loop/shared.rs:2022-2290`
- Modify/Test: `src/agent/agent_loop/tests.rs:2775-2840,3310-3490`

**Interfaces:**
- Produces: `CacheResetReason::ToolTopology` encoded as `tool_topology`.
- Extends: `WireRecordingProvider::with_higgs_session_cache() -> Self`.
- Enforces: a changed tool hash rotates the retained-session epoch before `stable_higgs_session_id` is evaluated.

- [x] **Step 1: Refresh the graph, run impact analysis, and report CRITICAL risk before editing**

Run:

```bash
node .gitnexus/run.cjs impact step_call_llm --direction upstream --repo nanobot-rs
node .gitnexus/run.cjs impact CacheResetReason --direction upstream --repo nanobot-rs
node .gitnexus/run.cjs impact cache_status_label --direction upstream --repo nanobot-rs
```

Expected: `step_call_llm` and `cache_status_label` are CRITICAL. Name `run_iteration` and `cell_lines_with_reply_mark` as d=1 callers and retain full exhaustive-match coverage.

- [x] **Step 2: Write failing marker round-trip and TUI tests**

Add `ToolTopology` to the `ControlMarker` round-trip fixture and add:

```rust
#[test]
fn tool_topology_reset_is_explicit() {
    let mut app = App::new();
    app.begin_turn("q");
    app.on_delta("\u{0}cache:reset:tool_topology");
    app.on_delta("real");
    app.finish_turn(String::new());
    let rendered: String = cell_lines(app.transcript.last().unwrap(), Mode::Calm, 1.0)
        .iter().flatten().map(|(text, _)| text.clone()).collect();
    assert!(rendered.contains("cache reset · tool topology"));
}
```

- [x] **Step 3: Write failing Higgs epoch-transition tests**

Extend `WireRecordingProvider`:

```rust
fn with_higgs_session_cache(mut self) -> Self {
    self.higgs_session_cache = true;
    self
}

fn supports_higgs_session_cache(&self) -> bool {
    self.higgs_session_cache
}

fn get_api_base(&self) -> Option<&str> {
    self.higgs_session_cache
        .then_some("http://127.0.0.1:9000/v1")
}

fn higgs_requests(&self) -> Vec<Vec<Value>> {
    self.calls()
        .into_iter()
        .filter(|messages| {
            messages.first().is_some_and(|message| {
                message
                    .get(crate::providers::openai_compat::NANOBOT_HIGGS_SESSION_ID_FIELD)
                    .is_some()
            })
        })
        .collect()
}
```

Add three tests:

```rust
#[tokio::test]
async fn unchanged_tool_topology_preserves_higgs_epoch() {
    let provider = Arc::new(
        WireRecordingProvider::new(
            "local-higgs-test",
            vec![
                WireRecordingProvider::text_response("first"),
                WireRecordingProvider::text_response("second"),
            ],
        )
        .with_higgs_session_cache(),
    );
    let (agent_loop, workspace) =
        build_local_inline_harness(provider.clone() as Arc<dyn LLMProvider>);
    let session_key = format!("higgs-stable-tools-{}", uuid::Uuid::new_v4());

    agent_loop
        .process_direct("first turn", &session_key, "test", "offline")
        .await;
    agent_loop
        .process_direct("second turn", &session_key, "test", "offline")
        .await;

    assert_eq!(
        agent_loop
            .shared
            .core_handle
            .counters
            .session_prompt_epoch(&session_key),
        0
    );
    let requests = provider.higgs_requests();
    assert_eq!(requests.len(), 2);
    let first_id = requests[0][0]
        [crate::providers::openai_compat::NANOBOT_HIGGS_SESSION_ID_FIELD]
        .as_u64()
        .unwrap();
    let second_id = requests[1][0]
        [crate::providers::openai_compat::NANOBOT_HIGGS_SESSION_ID_FIELD]
        .as_u64()
        .unwrap();
    assert_eq!(first_id, second_id);
    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn changed_tool_topology_rotates_before_request() {
    let provider = Arc::new(
        WireRecordingProvider::new(
            "local-higgs-test",
            vec![
                WireRecordingProvider::text_response("first"),
                WireRecordingProvider::text_response("second"),
            ],
        )
        .with_higgs_session_cache(),
    );
    let (agent_loop, workspace) =
        build_local_inline_harness(provider.clone() as Arc<dyn LLMProvider>);
    let session_key = format!("higgs-tool-change-{}", uuid::Uuid::new_v4());

    agent_loop
        .process_direct("first turn", &session_key, "test", "offline")
        .await;
    let first_requests = provider.higgs_requests();
    assert_eq!(first_requests.len(), 1);
    let first_id = first_requests[0][0]
        [crate::providers::openai_compat::NANOBOT_HIGGS_SESSION_ID_FIELD]
        .as_u64()
        .unwrap();
    let counters = &agent_loop.shared.core_handle.counters;
    let installed_hash = *counters
        .prompt_tool_hashes
        .lock()
        .get(&session_key)
        .expect("first request must install the tool hash");
    counters
        .prompt_tool_hashes
        .lock()
        .insert(session_key.clone(), installed_hash ^ 1);

    agent_loop
        .process_direct("second turn", &session_key, "test", "offline")
        .await;

    assert_eq!(counters.session_prompt_epoch(&session_key), 1);
    let requests = provider.higgs_requests();
    assert_eq!(requests.len(), 2);
    let second_head = &requests[1][0];
    let second_id = second_head
        [crate::providers::openai_compat::NANOBOT_HIGGS_SESSION_ID_FIELD]
        .as_u64()
        .unwrap();
    assert_ne!(second_id, first_id);
    assert_eq!(
        second_head[crate::providers::openai_compat::NANOBOT_HIGGS_DROP_SESSION_ID_FIELD],
        json!(first_id)
    );
    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn lease_exhaustion_keeps_one_higgs_epoch() {
    let responses: Vec<_> = (0..=crate::agent::lease::DEFAULT_TOOLS_PER_LEASE)
        .map(|n| {
            WireRecordingProvider::tool_response(
                &format!("tc_higgs_lease_{n}"),
                &format!("higgs-lease-{n}"),
            )
        })
        .collect();
    let provider = Arc::new(
        WireRecordingProvider::new("local-higgs-test", responses)
            .with_higgs_session_cache(),
    );
    let (agent_loop, workspace) = build_local_inline_harness_with_iters(
        provider.clone() as Arc<dyn LLMProvider>,
        20,
    );
    let session_key = format!("higgs-lease-stable-{}", uuid::Uuid::new_v4());

    let response = agent_loop
        .process_direct("keep listing", &session_key, "test", "offline")
        .await;

    assert_eq!(response, LEASE_OVER_BUDGET_FINAL);
    assert_eq!(
        agent_loop
            .shared
            .core_handle
            .counters
            .session_prompt_epoch(&session_key),
        0
    );
    let requests = provider.higgs_requests();
    assert_eq!(
        requests.len(),
        crate::agent::lease::DEFAULT_TOOLS_PER_LEASE as usize + 1
    );
    let ids: Vec<_> = requests
        .iter()
        .map(|request| {
            request[0][crate::providers::openai_compat::NANOBOT_HIGGS_SESSION_ID_FIELD]
                .as_u64()
                .unwrap()
        })
        .collect();
    assert!(ids.windows(2).all(|pair| pair[0] == pair[1]));
    assert!(requests.iter().all(|request| {
        request[0]
            .get(crate::providers::openai_compat::NANOBOT_HIGGS_DROP_SESSION_ID_FIELD)
            .is_none()
            && request[0]
                .get(crate::providers::openai_compat::NANOBOT_HIGGS_DROP_SESSION_IDS_FIELD)
                .is_none()
    }));
    let _ = std::fs::remove_dir_all(&workspace);
}
```

Use the existing singular drop constant because one active epoch is retired in
this test. Do not infer rotation from logs alone.

- [x] **Step 4: Run the new tests to verify RED**

Run:

```bash
cargo test --lib turn_stream::tests -- --nocapture
cargo test --lib tui_app::app::tests::tool_topology_reset_is_explicit -- --exact --nocapture
cargo test --lib agent::agent_loop::tests::unchanged_tool_topology_preserves_higgs_epoch -- --exact --nocapture
cargo test --lib agent::agent_loop::tests::changed_tool_topology_rotates_before_request -- --exact --nocapture
cargo test --lib agent::agent_loop::tests::lease_exhaustion_keeps_one_higgs_epoch -- --exact --nocapture
```

Expected: marker tests fail to compile and the changed-topology test observes no
epoch rotation. The unchanged-topology test is a control and should already pass.

- [x] **Step 5: Add the explicit reset reason**

Add:

```rust
CacheResetReason::ToolTopology => "tool_topology"
```

to encoding and parsing, include it in the round-trip fixture, and render it in `cache_status_label` as:

```rust
CacheResetReason::ToolTopology => {
    ("cache reset · tool topology".to_string(), WARN_COLOR)
}
```

Do not change cache-status ranking.

- [x] **Step 6: Enforce the epoch boundary in the existing tool-hash diagnostic block**

In `step_call_llm`, compute and install the hash without holding the lock across invalidation:

```rust
let tool_count = tool_defs_opt.map_or(0, |t| t.len());
let new_tool_hash = prompt_fingerprint::hash_tools(tool_defs_opt.unwrap_or(&[]));
let previous_tool_hash = ctx
    .counters
    .prompt_tool_hashes
    .lock()
    .insert(ctx.session_key.to_string(), new_tool_hash);

if let Some(previous_tool_hash) = previous_tool_hash.filter(|old| *old != new_tool_hash) {
    let rotated = if ctx.core.mode().is_local()
        && ctx.core.provider.supports_higgs_session_cache()
    {
        invalidate_prompt_cache_for_rewrite(ctx, CacheResetReason::ToolTopology)
    } else {
        false
    };
    ctx.counters
        .prompt_tool_hashes
        .lock()
        .insert(ctx.session_key.to_string(), new_tool_hash);
    warn!(
        session = %ctx.session_key,
        tool_count,
        prev_hash = previous_tool_hash,
        new_hash = new_tool_hash,
        rotated,
        "tool_block_changed — rotated retained session before changed prompt head"
    );
}
```

Place this before deriving `provider_session_id`. Non-Higgs providers keep the
existing warning-only behavior; they do not emit a cache-reset marker. The new
hash is still the current diagnostic baseline. The lock scope shown above is
mandatory: the Higgs invalidation path clears `prompt_tool_hashes` and would
deadlock if called while the map is locked.

- [x] **Step 7: Run focused and neighboring tests to verify GREEN**

Run:

```bash
cargo test --lib turn_stream::tests -- --nocapture
cargo test --lib tui_app::app::tests::tool_topology_reset_is_explicit -- --exact --nocapture
cargo test --lib agent::agent_loop::tests::unchanged_tool_topology_preserves_higgs_epoch -- --exact --nocapture
cargo test --lib agent::agent_loop::tests::changed_tool_topology_rotates_before_request -- --exact --nocapture
cargo test --lib agent::agent_loop::tests::lease_exhaustion_keeps_one_higgs_epoch -- --exact --nocapture
cargo test --lib agent::agent_core::tests::test_invalidate_prompt_cache_rotates_when_higgs_capable_clears_otherwise -- --exact --nocapture
cargo test --lib providers::openai_compat::tests::test_higgs_session_cache_capability_independent_of_port -- --exact --nocapture
```

Expected: explicit marker round trip, correct TUI label, stable epoch for equal tools, and pre-request rotation/drop marker for changed tools.

- [x] **Step 8: Check scope and commit**

Run:

```bash
node .gitnexus/run.cjs detect-changes
git diff --check
git add src/turn_stream.rs src/tui_app/app.rs src/agent/agent_loop/shared.rs src/agent/agent_loop/tests.rs
git commit -m "fix(cache): rotate Higgs session on tool changes"
node .gitnexus/run.cjs analyze
```

Expected scope: retained-session cache boundary plus TUI reset rendering. No provider request API changes.

---

### Task 4: Full Regression, Performance, and Change-Scope Verification

**Files:**
- Verify only; modify production files only if a preceding test exposes a defect in this plan's scope.

**Interfaces:**
- Consumes the completed Tasks 1-3.
- Produces verification evidence for correctness, cache stability, performance, and GitNexus scope.

- [x] **Step 1: Format and run focused regression suites**

Run:

```bash
cargo fmt --all -- --check
cargo test --lib agent::lease::tests -- --nocapture
cargo test --lib agent::agent_loop::tests::convergence_ -- --nocapture
cargo test --lib agent::tool_engine::tests -- --nocapture
cargo test --lib turn_stream::tests -- --nocapture
cargo test --lib tui_app::app::tests -- --nocapture
```

Expected: all pass with no ignored failure. If formatting fails, run `cargo fmt --all`, inspect the diff, and repeat the check.

Result: every focused suite passed. Repository-wide `cargo fmt --check`
still reports pre-existing formatting debt outside this range; the feature range
passes `git diff --check` and no unrelated formatting was rewritten.

- [x] **Step 2: Run full correctness validation**

Run:

```bash
cargo test
cargo build
```

Expected: both exit 0. Do not claim completion from targeted tests alone.

Result: `cargo test --quiet` passed 2,669 tests with 28 intentional ignores and
zero failures; `cargo build` exited 0.

- [ ] **Step 3: Run the matched turn benchmark**

Run:

```bash
scripts/turn_bench.sh
```

Expected: the benchmark completes and shows no unexplained regression in matched agent-loop/provider/context timing. Save the before/after figures in the handoff; do not weaken thresholds to obtain a pass.

Not run: no release `nanobot` binary or live local backend was available.

- [x] **Step 4: Verify final GitNexus scope and all direct dependents**

Run:

```bash
node .gitnexus/run.cjs detect-changes
git diff --check
git status --short
```

Expected affected scope:

- `Lease` admission/annotation;
- inline and delegated result injection;
- `step_pre_call`, `step_execute_tools`, and `step_call_llm` agent-loop flows;
- retained Higgs session rotation;
- cache-reset marker parsing/rendering;
- focused tests.

Confirm no unrelated user-owned file is staged or committed and every d=1 dependent from the impact baseline has test coverage.

- [ ] **Step 5: Reproduce the production trace when Higgs is available**

Run the existing local agent/Higgs setup until one lease exhausts. Confirm logs show all of:

```text
no tool_lease_stripping_after_blocks event
no lease-driven tool_block_changed event
stable common_prefix_tokens across the final admitted batch
one rejected over-budget batch
no provider request after that rejection
```

If no live Higgs server is available, report this check as not run; do not substitute mocked evidence for the runtime trace.

Not run: no Nanobot/Higgs process, tmux server, or listener on port 9000 was
available during final verification.

- [x] **Step 6: Commit any formatting-only delta, otherwise leave the verified commits unchanged**

Only when `cargo fmt --all` changed files:

```bash
node .gitnexus/run.cjs detect-changes
git add src/agent/lease.rs src/agent/tool_engine.rs src/agent/agent_loop/shared.rs src/agent/agent_loop/response.rs src/agent/prepare_context.rs src/agent/agent_loop/tests.rs src/turn_stream.rs src/tui_app/app.rs
git commit -m "style: format stable tool prefix changes"
node .gitnexus/run.cjs analyze
```

Do not create an empty verification commit.

---

### Task 5: Ponytail Review Remediation

- [x] Persist each pending protocol group with a checked SQLite transaction;
      roll back every row when any carrier or receipt insert fails.
- [x] Surface rejected-batch durability failure as an infrastructure error and
      never report normal lease exhaustion after a partial/failed commit.
- [x] Share one per-session lock across gateway, CLI, REPL, and TUI entrypoints
      so prompt topology comparison and Higgs epoch rotation cannot race.
- [x] Exercise a real channel-specific tool-registry transition and assert the
      serialized tool schemas differ before retained-session rotation.
- [x] Make the crossing-batch test attempt two file writes and prove neither
      side effect happens.
- [x] Run focused/full verification, GitNexus change detection, and the final
      ponytail review on the remediation diff.
