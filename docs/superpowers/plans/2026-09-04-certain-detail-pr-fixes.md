# Certain Detail PR Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the 18 approved, behavior-changing Detail fixes onto `feature/adaptive-capacity-nanobot` while preserving adaptive capacity and native Escha execution.

**Architecture:** Reproduce each defect with one focused release-mode regression test, then implement the smallest fix in the existing production path. Each behavior lands in its own commit so it can be reviewed or reverted independently; conflicts are resolved against adaptive-capacity interfaces rather than by accepting a PR side wholesale.

**Tech Stack:** Rust 2021, Tokio, serde_json, rusqlite/FTS5, reqwest SSE, existing protocol and tool abstractions, Cargo release tests, GitNexus.

## Global Constraints

- Work in `/private/tmp/nac` on `feature/adaptive-capacity-nanobot`.
- Preserve existing unstaged `AGENTS.md` and `CLAUDE.md`; never stage them.
- Use only release-mode Cargo commands.
- Run GitNexus upstream impact before modifying every production symbol and report HIGH/CRITICAL risk before editing.
- Add no new modules, dependencies, protocol flags, or alternate pipelines.
- Keep Escha on the native trellis kernel for all 40 expert layers; introduce no affine fallback.
- Follow red-green-refactor literally: test first, observe the expected failure, then edit production code.
- Before each commit run `npx gitnexus detect-changes --repo nanobot-rs`, `git diff --check`, and stage only task-owned files.

---

### Task 1: Establish the adaptive-capacity baseline

**Files:**
- Verify only: repository and release artifacts

**Interfaces:**
- Consumes: `feature/adaptive-capacity-nanobot` at design commit `752ebf4`
- Produces: a known-good baseline and recorded pre-existing failures

- [x] **Step 1: Confirm branch and preserve user edits**

Run: `git status --short --branch`

Expected: branch is `feature/adaptive-capacity-nanobot`; only `AGENTS.md` and `CLAUDE.md` are unstaged before this plan is added.

- [x] **Step 2: Build the baseline**

Run: `cargo build --release`

Expected: exit 0.

- [x] **Step 3: Run the baseline suite**

Run: `cargo test --release`

Expected: exit 0. If it fails, record the exact failing tests and stop before production edits unless the failures are already proven baseline failures.

- [x] **Step 4: Record baseline state**

Run: `git status --short`

Expected: Cargo did not modify tracked source files.

### Task 2: Block idle-write `..` traversal (#32)

**Files:**
- Modify: `src/agent/tools/filesystem/mod.rs`
- Modify: `src/agent/tools/filesystem/write.rs`
- Modify: `src/agent/tools/apply_patch.rs`

**Interfaces:**
- Consumes: `expand_path`, `expand_write_path`, `idle_write_allowed`
- Produces: `normalize_lexical(&Path) -> PathBuf`; identical checked and executed idle paths

- [x] **Step 1: Analyze impact**

Run:

```bash
npx gitnexus impact idle_write_allowed --direction upstream
npx gitnexus impact WriteFileTool::execute_write --direction upstream
npx gitnexus impact EditFileTool::execute --direction upstream
npx gitnexus impact ApplyPatchTool::execute --direction upstream
```

- [x] **Step 2: Add one traversal regression**

Add an inline filesystem test using two temporary directories:

```rust
#[test]
fn idle_allowlist_rejects_parent_traversal() {
    let workspace = Path::new("/workspace");
    let target = workspace.join("skills/../../outside.txt");
    assert!(!idle_write_allowed(&["skills/**".into()], &target, workspace));
}
```

- [x] **Step 3: Verify RED**

Run: `cargo test --release idle_allowlist_rejects_parent_traversal -- --nocapture`

Expected: FAIL because the unnormalized target passes the subtree prefix check.

- [x] **Step 4: Implement the shared normalization**

Add a component-wise lexical normalizer in `filesystem/mod.rs`; normalize inside `idle_write_allowed` and reuse the normalized path for idle `write_file`, `edit_file`, and `apply_patch` filesystem operations. Preserve leading relative `..`; clamp absolute paths at root.

- [x] **Step 5: Verify GREEN and neighbors**

Run:

```bash
cargo test --release idle_allowlist_rejects_parent_traversal -- --nocapture
cargo test --release agent::tools::filesystem -- --nocapture
cargo test --release agent::tools::apply_patch -- --nocapture
```

Expected: all exit 0.

- [x] **Step 6: Commit**

Commit message: `fix(tools): contain idle write traversal`

### Task 3: Preserve patch records beginning with `---` or `+++` (#33)

**Files:**
- Modify: `src/agent/tools/apply_patch.rs`

**Interfaces:**
- Consumes: `parse_unified_patch`
- Produces: file-header recognition only before an active hunk

- [x] **Step 1: Analyze impact**

Run: `npx gitnexus impact parse_unified_patch --direction upstream`

- [x] **Step 2: Add the parser regression**

```rust
#[test]
fn patch_removes_double_dash_comment() {
    let content = "-- comment\nkeep\n";
    let patch = "@@ -1,2 +1,1 @@\n--- comment\n keep\n";
    let (updated, _) = apply_unified_patch_to_content(content, patch).unwrap();
    assert_eq!(updated, "keep\n");
}
```

- [x] **Step 3: Verify RED**

Run: `cargo test --release patch_removes_double_dash_comment -- --nocapture`

Expected: FAIL because `--- comment` is discarded as a header.

- [x] **Step 4: Implement one guard**

Skip `--- `, `+++ `, and `diff ` only when `current.is_none()`.

- [x] **Step 5: Verify and commit**

Run: `cargo test --release agent::tools::apply_patch -- --nocapture`

Expected: exit 0.

Commit message: `fix(apply_patch): preserve header-like hunk lines`

### Task 4: Bound pipeline and loop cost (#9)

**Files:**
- Modify: `src/agent/tools/spawn.rs`
- Modify: `src/agent/tool_wiring.rs`
- Modify: `src/agent/pipeline.rs`

**Interfaces:**
- Consumes: `SpawnAction::parse`, `AgentHost::run_pipeline`, `vote_on_step`
- Produces: `MAX_AHEAD_BY_K = 3`, `MAX_LOOP_ROUNDS = 10`, bounded schemas and overflow-safe arithmetic

- [x] **Step 1: Analyze impact**

Run:

```bash
npx gitnexus impact SpawnAction::parse --direction upstream
npx gitnexus impact AgentHost::run_pipeline --direction upstream
npx gitnexus impact vote_on_step --direction upstream
```

- [x] **Step 2: Add bound regressions**

Add parser tests asserting an excessive `ahead_by_k` cannot exceed 3 and excessive `max_rounds` cannot exceed 10, plus schema assertions:

```rust
assert_eq!(properties["ahead_by_k"]["maximum"], json!(3));
assert_eq!(properties["max_rounds"]["maximum"], json!(10));
```

- [x] **Step 3: Verify RED**

Run: `cargo test --release spawn_cost_bounds -- --nocapture`

Expected: FAIL because both values are currently unbounded and the schema has no maximum.

- [x] **Step 4: Implement minimal caps**

Clamp parsed values to the constants, expose the same maxima in the JSON schema, use `saturating_mul(2).saturating_add(1)` for voter count, and `saturating_add` for vote convergence.

- [x] **Step 5: Verify and commit**

Run:

```bash
cargo test --release spawn_cost_bounds -- --nocapture
cargo test --release agent::pipeline -- --nocapture
cargo test --release agent::tools::spawn -- --nocapture
```

Expected: all exit 0.

Commit message: `fix(spawn): bound pipeline and loop cost`

### Task 5: Preserve assistant/tool pairing during anti-drift collapse (#18)

**Files:**
- Modify: `src/agent/anti_drift.rs`

**Interfaces:**
- Consumes: `collapse_repetitive_attempts`
- Produces: collapsed text with original `tool_calls` retained

- [x] **Step 1: Analyze impact**

Run: `npx gitnexus impact collapse_repetitive_attempts --direction upstream`

- [x] **Step 2: Add the pairing regression**

Build three identical assistant tool-call messages interleaved with their `role=tool` results, run the collapse, and assert every tool result ID remains announced by an assistant `tool_calls` array.

- [x] **Step 3: Verify RED**

Run: `cargo test --release anti_drift_collapse_preserves_tool_pairing -- --nocapture`

Expected: FAIL with orphan IDs from the first two collapsed attempts.

- [x] **Step 4: Implement deletion-only fix**

Remove the code that deletes `tool_calls`; continue replacing only assistant preamble content.

- [x] **Step 5: Verify and commit**

Run: `cargo test --release agent::anti_drift -- --nocapture`

Expected: exit 0.

Commit message: `fix(anti_drift): preserve tool result pairing`

### Task 6: Select pipeline protocol from the provider endpoint (#12)

**Files:**
- Modify: `src/agent/pipeline.rs`

**Interfaces:**
- Consumes: `LLMProvider::get_api_base`, `is_local_api_base`, `execute_step_with_tools`
- Produces: `pipeline_targets_local(provider, model) -> bool`

- [x] **Step 1: Analyze impact**

Run: `npx gitnexus impact execute_step_with_tools --direction upstream`

- [x] **Step 2: Add endpoint classification regressions**

Using a provider exposing a configurable API base, assert:

```rust
assert!(pipeline_targets_local(&lan_provider, "qwen/model"));
assert!(!pipeline_targets_local(&cloud_provider, "qwen/model"));
assert!(!pipeline_targets_local(&local_provider, "mlx:model"));
```

- [x] **Step 3: Verify RED**

Run: `cargo test --release pipeline_targets_local -- --nocapture`

Expected: compilation/test failure because pipeline selection is still model-prefix based.

- [x] **Step 4: Implement endpoint-based selection**

Return true only when `get_api_base()` is classified local and the model does not begin with `mlx:`. Replace `policy::is_local_model(model)` in `execute_step_with_tools` with this helper.

- [x] **Step 5: Verify and commit**

Run: `cargo test --release agent::pipeline -- --nocapture`

Expected: exit 0.

Commit message: `fix(pipeline): select protocol from endpoint`

### Task 7: Preserve explicit stream completion without `[DONE]` (#20)

**Files:**
- Modify: `src/providers/openai_compat.rs`

**Interfaces:**
- Consumes: `parse_sse_stream`
- Produces: a separate `finish_reason_seen` state bit

- [x] **Step 1: Analyze impact**

Run: `npx gitnexus impact parse_sse_stream --direction upstream`

- [x] **Step 2: Add the missing-sentinel regression**

Feed SSE chunks containing text and an explicit `finish_reason: "stop"`, omit `[DONE]`, and assert the final response is `FinishReason::Stop`.

- [x] **Step 3: Verify RED**

Run: `cargo test --release sse_stream_no_done_keeps_explicit_stop -- --nocapture`

Expected: FAIL with `Length`.

- [x] **Step 4: Implement state separation**

Set `finish_reason_seen = true` whenever a finish reason is parsed; convert default `Stop` to `Length` at abnormal EOF only when no finish reason was observed.

- [x] **Step 5: Verify and commit**

Run: `cargo test --release providers::openai_compat -- --nocapture`

Expected: exit 0.

Commit message: `fix(provider): preserve explicit stream completion`

### Task 8: Inject the blocked-tool recovery scaffold (#27)

**Files:**
- Modify: `src/agent/router.rs`

**Interfaces:**
- Consumes: `route_tool_calls`, `MessageLog::push_draft`, `scaffold_user`
- Produces: a visible recovery instruction after two all-blocked rounds

- [x] **Step 1: Analyze impact**

Run: `npx gitnexus impact route_tool_calls --direction upstream`

- [x] **Step 2: Add the message-log regression**

Drive two all-blocked, uncached tool rounds and assert `ctx.messages` contains `Your last several tool calls were duplicates or blocked`.

- [x] **Step 3: Verify RED**

Run: `cargo test --release circuit_breaker_scaffold_is_injected -- --nocapture`

Expected: FAIL because the constructed scaffold is discarded.

- [x] **Step 4: Implement the missing push**

Wrap the existing `scaffold_user(...)` value in `ctx.messages.push_draft(...)`; change no thresholds.

- [x] **Step 5: Verify and commit**

Run: `cargo test --release agent::router -- --nocapture`

Expected: exit 0.

Commit message: `fix(router): retain blocked-tool scaffold`

### Task 9: Keep system announcements out of coalescing (#28)

**Files:**
- Modify: `src/agent/agent_loop/mod.rs`
- Modify: `src/agent/agent_loop/tests.rs`

**Interfaces:**
- Consumes: `AgentLoop::run`, `InboundMessage.metadata["is_system"]`
- Produces: one shared `is_system_message` predicate used before and after batching

- [x] **Step 1: Analyze impact**

Run: `npx gitnexus impact AgentLoop::run --direction upstream`

- [x] **Step 2: Add the user/system coalescing regression**

Send a user message and same-session system announcement within the 400 ms window; assert the provider receives the user turn and the announcement is emitted separately.

- [x] **Step 3: Verify RED**

Run: `cargo test --release system_announcement_after_user_message_does_not_coalesce -- --nocapture`

Expected: FAIL because later metadata marks the merged message as system.

- [x] **Step 4: Implement both guards**

Extract `is_system_message(&InboundMessage)`. Exclude system, slash-command, and idle messages both when opening a coalescing batch and when accepting same-session followers.

- [x] **Step 5: Verify and commit**

Run:

```bash
cargo test --release system_announcement -- --nocapture
cargo test --release rapid_same_session_user_messages_still_coalesce -- --nocapture
```

Expected: all exit 0.

Commit message: `fix(agent-loop): isolate system announcements`

### Task 10: Use the effective response budget during overflow recovery (#13)

**Files:**
- Modify: `src/agent/agent_loop/budget.rs`
- Modify: `src/agent/agent_loop/shared.rs`

**Interfaces:**
- Consumes: adaptive `EffectiveTokenBudget`, per-call `max_tokens`, `attempt_overflow_recovery`
- Produces: `overflow_recovery_fallback_budget(window, effective_max_tokens)`

- [x] **Step 1: Analyze impact**

Run:

```bash
npx gitnexus impact attempt_overflow_recovery --direction upstream
npx gitnexus impact overflow_trim_threshold --direction upstream
```

Warn before editing if this critical-path analysis is HIGH or CRITICAL.

- [x] **Step 2: Add the pure budget regression**

```rust
#[test]
fn overflow_fallback_reserves_effective_response_budget() {
    assert_eq!(overflow_recovery_fallback_budget(32_768, 12_288), 16_384);
}
```

- [x] **Step 3: Verify RED**

Run: `cargo test --release overflow_fallback_reserves_effective_response_budget -- --nocapture`

Expected: compilation failure because the helper does not exist.

- [x] **Step 4: Implement against adaptive capacity**

Compute `(window.saturating_sub(effective_max_tokens as usize) as f64 * 0.80) as usize`. Pass the actual per-call `max_tokens` through all streaming and non-streaming overflow recovery call sites; retain server-supplied count handling unchanged.

- [x] **Step 5: Verify and commit**

Run:

```bash
cargo test --release overflow_fallback -- --nocapture
cargo test --release capacity_exceeded -- --nocapture
cargo test --release overflow_recovery -- --nocapture
```

Expected: all exit 0.

Commit message: `fix(agent-loop): reserve effective overflow budget`

### Task 11: Preserve lease-renewal checkpoints (#16)

**Files:**
- Modify: `src/agent/agent_loop/response.rs`

**Interfaces:**
- Consumes: `step_process_response`, lease renewal validation, `MessageLog`
- Produces: checkpoint assistant message preceding the renewal scaffold

- [x] **Step 1: Analyze impact**

Run: `npx gitnexus impact step_process_response --direction upstream`

- [x] **Step 2: Add the post-renewal wire regression**

Script 12 distinct read tool calls, then a valid `findings:/next:/will:` checkpoint and a final answer. Assert the first post-renewal provider call contains the checkpoint as `role=assistant` before the renewal scaffold.

- [x] **Step 3: Verify RED**

Run: `cargo test --release lease_renewal_persists_assistant_checkpoint -- --nocapture`

Expected: FAIL because only the scaffold is present.

- [x] **Step 4: Implement one message append**

After successful renewal and before `scaffold_user`, push `json!({"role":"assistant","content":content})` into the draft message log.

- [x] **Step 5: Verify and commit**

Run: `cargo test --release lease_renewal -- --nocapture`

Expected: exit 0.

Commit message: `fix(lease): retain renewal checkpoint`

### Task 12: Make failed plan steps terminal (#31)

**Files:**
- Modify: `src/agent/reasoning.rs`
- Modify: `src/agent/agent_loop/shared.rs`
- Modify: `src/agent/agent_loop/tests.rs`

**Interfaces:**
- Consumes: `ReasoningEngine::mark_current_failed`, `mark_current_completed`, per-step budget branch
- Produces: terminal failed status and a user-visible stopped-turn result when no checkpoint exists

- [x] **Step 1: Analyze impact**

Run:

```bash
npx gitnexus impact ReasoningEngine::mark_current_failed --direction upstream
npx gitnexus impact ReasoningEngine::mark_current_completed --direction upstream
```

- [x] **Step 2: Add engine regressions**

```rust
engine.mark_current_failed("iteration budget exhausted");
assert!(engine.step_instruction().is_none());
engine.mark_current_completed(Some("late answer".into()));
assert!(!engine.is_complete());
```

Add one plan-guided loop test asserting a read-only step stops near its configured step budget rather than running to `max_iterations`.

- [x] **Step 3: Verify RED**

Run: `cargo test --release mark_current_failed_clears_current_step -- --nocapture`

Expected: FAIL because the failed step remains current.

- [x] **Step 4: Implement terminal failure**

Clear `current_step` after marking failure; refuse to overwrite a `Failed` step in `mark_current_completed`; when budget exhaustion has no checkpoint, set an explicit failure response and break the turn.

- [x] **Step 5: Verify and commit**

Run:

```bash
cargo test --release failed_plan_step -- --nocapture
cargo test --release plan_guided -- --nocapture
```

Expected: all exit 0.

Commit message: `fix(reasoning): terminate failed plan steps`

### Task 13: Make FTS migration atomic and self-healing (#14)

**Files:**
- Modify: `src/agent/knowledge_store.rs`

**Interfaces:**
- Consumes: `KnowledgeStore::open`, `migrate_fts_tokenizer`
- Produces: one SQLite transaction and an empty-index detector based on `chunks_fts_docsize`

- [x] **Step 1: Analyze impact**

Run: `npx gitnexus impact KnowledgeStore::migrate_fts_tokenizer --direction upstream`

- [x] **Step 2: Add the stranded-index regression**

Create a populated store, replace `chunks_fts` with an empty porter external-content table, reopen it, and assert `search("daemon", 10)` returns the original chunk.

- [x] **Step 3: Verify RED**

Run: `cargo test --release heals_empty_porter_index_on_open -- --nocapture`

Expected: FAIL with zero search hits.

- [x] **Step 4: Implement atomic recovery**

Make the connection mutable during `open`, run DROP/CREATE/rebuild inside `conn.transaction()`, and rebuild an already-porter table when `chunks` is non-empty but `chunks_fts_docsize` is empty.

- [x] **Step 5: Verify and commit**

Run: `cargo test --release agent::knowledge_store -- --nocapture`

Expected: exit 0.

Commit message: `fix(memory): make FTS migration recoverable`

### Task 14: Parse spaced LCM ID ranges (#22)

**Files:**
- Modify: `src/agent/lcm.rs`

**Interfaces:**
- Consumes: `parse_id_runs`
- Produces: recognition of ASCII whitespace around a digit-flanked dash without merging plain space-separated IDs

- [x] **Step 1: Analyze impact**

Run: `npx gitnexus impact parse_id_runs --direction upstream`

- [x] **Step 2: Add range regressions**

```rust
assert_eq!(parse_message_ids(&json!("5 - 8")), vec![5, 6, 7, 8]);
assert_eq!(parse_message_ids(&json!("5 6 7 8")), vec![5, 6, 7, 8]);
assert!(parse_message_ids(&json!("0 - 999999")).is_empty());
```

- [x] **Step 3: Verify RED**

Run: `cargo test --release parse_message_ids -- --nocapture`

Expected: FAIL because the spaced range produces endpoints only.

- [x] **Step 4: Implement minimal normalization**

Use one lazily compiled existing `regex` dependency to replace `(<digits>)\s*-\s*(<digits>)` with `$1-$2` before the existing splitter. Keep the 10,000-ID expansion cap and existing invalid-range behavior.

- [x] **Step 5: Verify and commit**

Run: `cargo test --release parse_message_ids -- --nocapture`

Expected: exit 0.

Commit message: `fix(lcm): parse spaced message ranges`

### Task 15: Merge overlapping redaction spans (#17)

**Files:**
- Modify: `src/agent/provenance.rs`

**Interfaces:**
- Consumes: `redact_fabrications`
- Produces: sorted, disjoint unions of claimed byte spans

- [x] **Step 1: Analyze impact**

Run: `npx gitnexus impact redact_fabrications --direction upstream`

- [x] **Step 2: Add leak and disjoint regressions**

Create one sentence-wide claimed span containing a shorter claimed file span and assert the result is exactly one placeholder. Retain a second assertion that two disjoint claims yield two placeholders.

- [x] **Step 3: Verify RED**

Run: `cargo test --release overlapping_claims_do_not_leak -- --nocapture`

Expected: FAIL with a surviving fabricated tail or incorrect count.

- [x] **Step 4: Implement span union**

Collect claimed spans, sort ascending, merge overlapping or adjacent spans, then replace merged spans in reverse order. Count merged redactions, not source annotations.

- [x] **Step 5: Verify and commit**

Run: `cargo test --release agent::provenance -- --nocapture`

Expected: exit 0.

Commit message: `fix(provenance): merge overlapping redactions`

### Task 16: Skip stale automatic restarts (#10)

**Files:**
- Modify: `src/repl/commands/mod.rs`
- Modify: `src/agent/agent_loop/tests.rs`

**Interfaces:**
- Consumes: `ReplContext::handle_restart_requests`, `server::check_local_health`
- Produces: a health preflight immediately before automatic `cmd_restart`

- [x] **Step 1: Analyze impact and process safety**

Run: `npx gitnexus impact ReplContext::handle_restart_requests --direction upstream`

Confirm the regression uses only a mock health listener and cannot signal a real Higgs process. If the pre-fix path would reach real process control, add a test-only restart callback before running RED.

- [x] **Step 2: Add the recovered-server regression**

Queue one automatic main restart request against a mock `/health` endpoint returning 200 and assert `handle_restart_requests()` returns false with no restart display message.

- [x] **Step 3: Verify RED safely**

Run: `cargo test --release handle_restart_requests_skips_stale_restart -- --nocapture`

Expected: FAIL without sending signals or starting a real server.

- [x] **Step 4: Implement the preflight**

Before `cmd_restart()` for an automatic main request, call `check_local_health(&self.srv.local_port).await` and continue when healthy. Manual `/restart` remains unchanged.

- [x] **Step 5: Verify and commit**

Run: `cargo test --release restart_requests -- --nocapture`

Expected: exit 0.

Commit message: `fix(repl): skip stale automatic restarts`

### Task 17: Match proxy exclusions by exact tool name (#24)

**Files:**
- Modify: `src/agent/tools/registry.rs`

**Interfaces:**
- Consumes: `ToolRegistry::get_proxy_definition_excluding`, `Tool::name`
- Produces: exact exclusion before hint rendering

- [x] **Step 1: Analyze impact**

Run: `npx gitnexus impact ToolRegistry::get_proxy_definition_excluding --direction upstream`

- [x] **Step 2: Add the prefix-collision regression**

Register `exec` and `execute_code`, exclude `exec`, and assert the proxy description omits `exec(` but contains `execute_code(`.

- [x] **Step 3: Verify RED**

Run: `cargo test --release proxy_exclusion_matches_exact_tool_name -- --nocapture`

Expected: FAIL because `starts_with("exec")` removes both tools.

- [x] **Step 4: Implement exact filtering**

Filter tool objects by `t.name() == excluded_name` and rarely-advertised exact names before mapping them to rendered hints.

- [x] **Step 5: Verify and commit**

Run: `cargo test --release proxy_definition -- --nocapture`

Expected: exit 0.

Commit message: `fix(tools): match catalog exclusions exactly`

### Task 18: Make voice log previews UTF-8 safe (#29)

**Files:**
- Modify: `src/channels/telegram.rs`
- Modify: `src/channels/whatsapp.rs`
- Modify: `src/voice_pipeline.rs`
- Modify: `src/utils/helpers.rs`

**Interfaces:**
- Consumes: `floor_char_boundary`
- Produces: `utf8_prefix(&str, max_bytes) -> &str`; safe preview slices at 60, 80, and 100 bytes

- [x] **Step 1: Analyze impact**

Run:

```bash
npx gitnexus impact TelegramChannel --direction upstream
npx gitnexus impact WhatsAppChannel --direction upstream
npx gitnexus impact VoicePipeline::transcribe_file --direction upstream
```

- [x] **Step 2: Add the boundary regression**

Add a helper regression that calls the intended API:

```rust
let text = format!("a{}", "中".repeat(20));
assert!(!text.is_char_boundary(60));
assert_eq!(utf8_prefix(&text, 60), &text[..58]);
```

- [x] **Step 3: Verify RED against the unsafe expression**

Run: `cargo test --release voice_preview_is_utf8_safe -- --nocapture`

Expected: compilation failure because `utf8_prefix` does not exist.

- [x] **Step 4: Add one shared safe-slice helper and use it at all four sites**

Implement `utf8_prefix` in terms of `floor_char_boundary`, then replace each byte-indexed log slice in Telegram, WhatsApp, and the voice pipeline. This extracts the repeated operation once and leaves message content unchanged.

- [x] **Step 5: Verify and commit**

Run:

```bash
cargo test --release voice_preview_is_utf8_safe -- --nocapture
cargo test --release --features voice voice_pipeline -- --nocapture
cargo build --release --features voice
```

Expected: all exit 0, or a documented pre-existing patched-dependency failure identical to the baseline feature build.

Commit message: `fix(channels): truncate voice previews safely`

### Task 19: Resolve default subagent aliases in local mode (#30)

**Files:**
- Modify: `src/agent/subagent.rs`

**Interfaces:**
- Consumes: `resolve_spawn_settings`, `agent_profiles::resolve_model_for_env`, `SubagentManager::run_loop`
- Produces: uniform alias resolution for explicit, profile, default, and loop model choices

- [x] **Step 1: Analyze impact**

Run:

```bash
npx gitnexus impact resolve_spawn_settings --direction upstream
npx gitnexus impact SubagentManager::run_loop --direction upstream
```

- [x] **Step 2: Add local alias regressions**

```rust
let settings = resolve_spawn_settings(None, None, Some("haiku"), "served-local", true, 20);
assert_eq!(settings.model, "served-local");
assert_eq!(resolve_loop_model(Some("haiku"), None, "served-local", true), "served-local");
```

- [x] **Step 3: Verify RED**

Run: `cargo test --release default_subagent_model_alias_resolves_in_local_mode -- --nocapture`

Expected: FAIL because the alias is sent verbatim or the loop helper is absent.

- [x] **Step 4: Implement uniform resolution**

Pass `default_subagent_model` through the existing `resolve` closure. Add one small `resolve_loop_model` helper and use it in `run_loop`; retain provider-prefixed and full model IDs unchanged.

- [x] **Step 5: Verify and commit**

Run: `cargo test --release agent::subagent -- --nocapture`

Expected: exit 0.

Commit message: `fix(subagent): resolve local default aliases`

### Task 20: Full-stack verification and independent review

**Files:**
- Verify: all changed source and test files
- Update: this plan's checkboxes only

**Interfaces:**
- Consumes: all 18 isolated fix commits
- Produces: release evidence, speed comparison, GitNexus scope report, native-kernel confirmation

- [x] **Step 1: Check formatting and patch hygiene**

Run:

```bash
cargo fmt --all -- --check
git diff --check
```

Expected: both exit 0.

- [x] **Step 2: Run release validation**

Run:

```bash
cargo build --release
cargo test --release
```

Expected: both exit 0, with any baseline exception explicitly matched to Task 1 evidence.

- [ ] **Step 3: Run the matched speed track**

Run: `scripts/turn_bench.sh`

Expected: no unexplained regression in matched turn metrics. Record provider/model, machine, and before/after numbers.

- [x] **Step 4: Confirm adaptive capacity and native Escha invariants**

Run:

```bash
rg -n "affine|trellis|native" src/higgs.rs src/agent src/providers
git diff 752ebf4..HEAD -- src/higgs.rs
```

Expected: no integration commit changes `src/higgs.rs`; no affine expert fallback is introduced; the existing native-trellis contract remains intact for all 40 Escha expert layers.

- [x] **Step 5: Run GitNexus scope detection**

Run: `npx gitnexus detect-changes --repo nanobot-rs`

Expected: only the planned tool, protocol, agent-loop, memory, channel, and subagent flows are affected; no unrelated execution flow appears.

- [x] **Step 6: Request OpenCode/Muse review**

Run a read-only OpenCode review with `opencode/muse-spark-1.3-contributor-free` over `752ebf4..HEAD`, asking for correctness defects, regressions, adaptive-capacity conflicts, and any route from Escha native trellis to affine execution. Apply no suggestion without reproducing it.

- [x] **Step 7: Final status**

Run: `git status --short --branch`

Expected: only the pre-existing unstaged `AGENTS.md` and `CLAUDE.md` remain outside committed work.
