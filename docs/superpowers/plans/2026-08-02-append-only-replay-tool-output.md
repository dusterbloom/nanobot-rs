# Append-Only Replay and Durable Tool Outputs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep ordinary Higgs requests byte-for-byte append-only while storing every full tool output outside the prompt and bounding newly appended tool evidence.

**Architecture:** SQLite keeps two immutable representations: the exact full body in `tool_results` and the exact bounded provider-facing result in `messages`. Retained Higgs sessions disable routine history windows and reuse one epoch while prompts grow append-only; existing LCM/reset operations keep their explicit rewrite path, and unexpected divergence rotates only as emergency recovery.

**Tech Stack:** Rust 2021, Tokio, rusqlite, serde_json, existing `SessionDb`, `PromptFingerprint`, Higgs retained-session metadata, and the existing LCM engine.

## Global Constraints

- Store every completed tool output, including failures, immutably before publishing its prompt representation.
- Never rewrite a previously provider-visible message during routine replay.
- Apply output budgets only while appending new results; do not implement a sliding retroactive window.
- Leave LCM thresholds and compaction policy unchanged.
- Keep the advertised tool schema byte-stable.
- Local mode gets one atomic 12-call lease with zero renewals.
- Use no new feature flag, protocol mode, dependency, or standalone hygiene/guard module.
- Use strict red-green-refactor: no production edit before its failing regression test is observed.
- Before editing any Rust symbol, run `npx gitnexus impact <symbol> --direction upstream --repo nanobot-rs`; warn before HIGH or CRITICAL edits.
- Before every commit, run `npx gitnexus detect-changes --repo nanobot-rs` and inspect `git diff --check`.

---

## File Map

- `src/session/filters.rs`: stop transforming tool bodies already shaped at ingestion.
- `src/agent/prepare_context.rs`: retained-Higgs history loading and per-turn lease/output-budget initialization.
- `src/agent/agent_loop/shared.rs`: prompt-delta recovery and per-turn counters.
- `src/turn_stream.rs`: explicit emergency-divergence reset reason.
- `src/agent/tool_engine.rs`: unconditional durable storage and deterministic preview allocation for inline/delegated execution.
- `src/agent/lease.rs`: zero-renewal annotation semantics.
- `src/agent/agent_loop/response.rs`: keep cloud renewal while local zero-renewal leases finish.
- `src/agent/agent_loop/tests.rs`: provider-wire, retained-epoch, output-budget, and lease regressions.
- `src/session/db.rs`: durable-store regression coverage; no schema change.

---

### Task 1: Make Retained Replay Exactly Append-Only

**Files:**
- Modify: `src/session/filters.rs:159-365,1185-1360`
- Modify: `src/agent/prepare_context.rs:356-385`
- Modify: `src/agent/agent_loop/shared.rs:2035-2260`
- Modify: `src/turn_stream.rs:410-427,536-552,632-655`
- Test: `src/session/filters.rs`
- Test: `src/agent/agent_loop/tests.rs:3660-3865`

**Interfaces:**
- Consumes: `prompt_fingerprint::compare`, `invalidate_prompt_cache_for_rewrite`, `LLMProvider::supports_higgs_session_cache`.
- Produces: `CacheResetReason::UnexpectedReplayDivergence`; exact retained loading; emergency rotation before provider I/O.

- [ ] **Step 1: Run blast-radius checks before editing**

~~~bash
npx gitnexus impact filter_history --direction upstream --repo nanobot-rs
npx gitnexus impact prepare_context --direction upstream --repo nanobot-rs
npx gitnexus impact step_call_llm --direction upstream --repo nanobot-rs
npx gitnexus impact CacheResetReason --direction upstream --repo nanobot-rs
~~~

Record depth-1 callers. Stop and warn before editing if any result is HIGH or CRITICAL.

- [ ] **Step 2: Write replay tests requiring byte-identical stored content**

Add tests with these assertions:

~~~rust
#[test]
fn tool_result_content_is_byte_identical_during_replay() {
    let stored = "x".repeat(TOOL_RESULT_REPLAY_MAX_BYTES + 5_000);
    let messages = vec![
        user("inspect"),
        tool_call_assistant("call_1"),
        json!({
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "exec",
            "content": stored,
        }),
    ];

    let replay = filter_history(&messages, 0, 0);
    assert_eq!(replay[2]["content"], messages[2]["content"]);
}

#[test]
fn recalled_result_content_is_byte_identical_during_replay() {
    let messages = recall_result_fixture("bounded exact recalled bytes");
    let replay = filter_history(&messages, 0, 0);
    assert_eq!(replay.last().unwrap()["content"], "bounded exact recalled bytes");
}
~~~

- [ ] **Step 3: Write retained-Higgs integration regressions**

Drive enough persisted turns to exceed the normal history window:

~~~rust
let requests = provider.higgs_requests();
for pair in requests.windows(2) {
    assert_wire_prefix(&pair[0], &pair[1]);
    assert_eq!(higgs_id(&pair[0]), higgs_id(&pair[1]));
}
assert_eq!(counters.session_prompt_epoch(&session_key), 0);
~~~

Add a second test that seeds a prior fingerprint, mutates a provider-visible stored row, and asserts the next request uses a new Higgs ID and sends the old ID in `_nanobot_higgs_drop_session_id`.

- [ ] **Step 4: Run focused tests and verify RED**

~~~bash
cargo test --lib session::filters::tests::tool_result_content_is_byte_identical_during_replay -- --exact --nocapture
cargo test --lib session::filters::tests::recalled_result_content_is_byte_identical_during_replay -- --exact --nocapture
cargo test --lib agent::agent_loop::tests::retained_higgs_history_never_applies_routine_windowing -- --exact --nocapture
cargo test --lib agent::agent_loop::tests::unexpected_replay_divergence_rotates_before_provider_io -- --exact --nocapture
~~~

Expected: replay transformations, retained windowing/fingerprint clearing, and same-ID divergence make the tests fail for the intended reasons.

- [ ] **Step 5: Preserve tool content during replay**

In `filter_history`, use the stored content directly:

~~~rust
let content = m
    .get("content")
    .and_then(Value::as_str)
    .unwrap_or("")
    .to_string();
~~~

Remove replay-only capping and recalled-body replacement helpers/imports. Keep clear markers, non-provider synthetic filtering, structured content restoration, and assistant/tool protocol pairing.

- [ ] **Step 6: Disable routine windows for retained Higgs only**

~~~rust
let retained_higgs = core.mode().is_local()
    && core.provider.supports_higgs_session_cache();
let (max_messages, max_turns) = if retained_higgs {
    (0, 0)
} else {
    (
        crate::agent::agent_core::history_limit_lcm(core.token_budget.max_context()),
        core.max_history_turns,
    )
};
let history = core
    .sessions
    .get_history(&session_id, max_messages, max_turns)
    .await;
~~~

Delete unconditional removal of `prompt_fingerprints` and `prompt_cache_watermark`. Explicit reset/compaction paths remain their sole invalidators.

- [ ] **Step 7: Rotate only on unexpected real divergence**

Add `CacheResetReason::UnexpectedReplayDivergence` with wire label `unexpected_replay_divergence`. Copy `PromptDelta` out of the fingerprint lock, emit current diagnostics, then recover before deriving the Higgs marker:

~~~rust
let unexpected_higgs_divergence =
    matches!(prompt_delta, PromptDelta::Diverged { .. })
        && ctx.core.mode().is_local()
        && ctx.core.provider.supports_higgs_session_cache();
if unexpected_higgs_divergence {
    invalidate_prompt_cache_for_rewrite(
        ctx,
        CacheResetReason::UnexpectedReplayDivergence,
    );
}
~~~

The later request-marker construction must observe the incremented epoch and queued drop ID.

- [ ] **Step 8: Verify GREEN and commit**

Run the four Step 4 tests, then:

~~~bash
cargo test --lib session::filters::tests -- --nocapture
cargo test --lib agent::agent_loop::tests::test_local_wire_prompt_prefix_stable_across_turns -- --exact --nocapture
npx gitnexus detect-changes --repo nanobot-rs
git diff --check
git add src/session/filters.rs src/agent/prepare_context.rs src/agent/agent_loop/shared.rs src/turn_stream.rs src/agent/agent_loop/tests.rs
git commit -m "fix(cache): keep retained replay append-only"
~~~

Expected: all focused tests pass and GitNexus reports only replay/cache flows.

---

### Task 2: Store Every Tool Body and Bound New Prompt Detail

**Files:**
- Modify: `src/agent/tool_engine.rs:45-125,620-780,1088-1275`
- Modify: `src/agent/agent_loop/shared.rs:330-410`
- Modify: `src/agent/prepare_context.rs:615-648`
- Test: `src/agent/tool_engine.rs:1680-2560`
- Test: `src/session/db.rs:2615-2725`
- Test: `src/agent/agent_loop/tests.rs`

**Interfaces:**
- Consumes: `SessionDb::store_tool_result_immutable`, `render_tool_result_handle`, `build_tool_result_preview`.
- Produces: `TOOL_PREVIEW_BUDGET_CHARS: usize = 16_384`; `FlowControl::tool_preview_chars_remaining`; unconditional storage before message persistence.

- [ ] **Step 1: Run blast-radius checks**

~~~bash
npx gitnexus impact stash_tool_result_for_prompt_shaping --direction upstream --repo nanobot-rs
npx gitnexus impact inject_tool_result --direction upstream --repo nanobot-rs
npx gitnexus impact execute_tools_delegated --direction upstream --repo nanobot-rs
npx gitnexus impact FlowControl --direction upstream --repo nanobot-rs
~~~

Report HIGH/CRITICAL findings before editing and update every depth-1 construction site.

- [ ] **Step 2: Write failing durable-storage coverage**

Use the real SQLite-backed tool-engine harness:

~~~rust
for (id, body, ok) in [
    ("small_ok", "ok".to_string(), true),
    ("small_err", "Error: nope".to_string(), false),
    ("medium", "m".repeat(7_000), true),
    ("large", "l".repeat(40_000), true),
] {
    inject_fixture_result(&mut ctx, id, &body, ok).await;
    assert_eq!(
        db.load_tool_result(&session_id, id).await.as_deref(),
        Some(body.as_str()),
    );
}
~~~

The current small-result early return must make this RED.

- [ ] **Step 3: Write failing preview-budget coverage**

Add pure and integration assertions:

~~~rust
assert_eq!(turn_preview_cap(10_000, 4, 16_384), 4_096);
assert_eq!(turn_preview_cap(10_000, 1, 0), 0);
assert!(
    messages
        .iter()
        .filter(is_detailed_tool_result)
        .map(content_chars)
        .sum::<usize>()
        <= TOOL_PREVIEW_BUDGET_CHARS
);
assert!(messages.iter().any(is_tool_handle));
~~~

Also assert every handle ID loads from SQLite and exact persisted `messages.content` survives `get_history(..., 0, 0)`.

- [ ] **Step 4: Run focused tests and verify RED**

~~~bash
cargo test --lib agent::tool_engine::tests::every_tool_result_is_stored_before_prompt_shaping -- --exact --nocapture
cargo test --lib agent::tool_engine::tests::turn_preview_budget_bounds_multiple_medium_results -- --exact --nocapture
cargo test --lib agent::agent_loop::tests::stored_tool_preview_reloads_byte_identically -- --exact --nocapture
~~~

Expected: small/error bodies are absent and sequential medium results exceed the shared budget.

- [ ] **Step 5: Store exact bodies unconditionally**

Keep the helper name to avoid a broad rename, but store before returning:

~~~rust
let needs_shaping = force
    || data.chars().count() > cap
    || data.len() > TOOL_RESULT_REPLAY_MAX_BYTES;
match sessions
    .store_tool_result_immutable(session_id, tool_call_id, tool_name, data)
    .await
{
    StoredResult::Stored { .. } | StoredResult::Identical { .. } => {
        Ok(needs_shaping)
    }
    sr @ (StoredResult::Conflict { .. } | StoredResult::Failed) => Err(sr),
}
~~~

Use the exact pre-summary/pre-gate body for inline and delegated calls, including failed results. Keep fail-closed conflict handling.

- [ ] **Step 6: Add one append-only per-turn detail budget**

Add to `FlowControl`:

~~~rust
pub(crate) tool_preview_chars_remaining: usize,
~~~

Initialize from:

~~~rust
pub(crate) const TOOL_PREVIEW_BUDGET_CHARS: usize = 16 * 1024;
~~~

Allocate a batch share with:

~~~rust
fn turn_preview_cap(
    per_result_cap: usize,
    batch_len: usize,
    remaining: usize,
) -> usize {
    if remaining < MIN_BATCH_TOOL_RESULT_CAP_CHARS {
        return 0;
    }
    per_result_cap
        .min(remaining / batch_len.max(1))
        .max(MIN_BATCH_TOOL_RESULT_CAP_CHARS)
        .min(remaining)
}
~~~

For inline and delegated results: zero cap means deterministic handle; positive cap means raw when it fits or deterministic head/tail preview otherwise. Debit the detailed characters appended; handles do not consume detail budget. Never rewrite an older message to reclaim budget.

- [ ] **Step 7: Keep retrieval results on the same path**

Route `recall_tool_result`, `slice_tool_result`, and `search_tool_result` through the same append budget. Their implementations remain bounded, but an exhausted turn publishes a handle under the retrieval call's own ID. Add no replay-time special case.

- [ ] **Step 8: Verify GREEN and commit**

~~~bash
cargo test --lib agent::tool_engine::tests::every_tool_result_is_stored_before_prompt_shaping -- --exact --nocapture
cargo test --lib agent::tool_engine::tests::turn_preview_budget_bounds_multiple_medium_results -- --exact --nocapture
cargo test --lib agent::agent_loop::tests::stored_tool_preview_reloads_byte_identically -- --exact --nocapture
cargo test --lib agent::tool_engine::tests -- --nocapture
cargo test --lib session::db::tests::oversized_tool_results_survive_database_reopen -- --exact --nocapture
npx gitnexus detect-changes --repo nanobot-rs
git diff --check
git add src/agent/tool_engine.rs src/agent/agent_loop/shared.rs src/agent/prepare_context.rs src/agent/agent_loop/tests.rs src/session/db.rs
git commit -m "fix(tools): persist bodies and bound prompt detail"
~~~

Expected: every exact body is durable and appended detailed content stays within 16,384 characters.

---

### Task 3: Give Local Turns One Non-renewable 12-call Lease

**Files:**
- Modify: `src/agent/lease.rs:17-205,395-445`
- Modify: `src/agent/prepare_context.rs:625-635`
- Modify: `src/agent/agent_loop/response.rs:460-545`
- Test: `src/agent/lease.rs`
- Test: `src/agent/agent_loop/tests.rs:3810-4150`

**Interfaces:**
- Consumes: `RuntimeMode::is_local`, `Lease::new`.
- Produces: `LOCAL_MAX_LEASE_RENEWALS: u32 = 0`; exhaustion annotation that requires a final answer and offers no checkpoint renewal.

- [ ] **Step 1: Run blast-radius checks**

~~~bash
npx gitnexus impact Lease::new --direction upstream --repo nanobot-rs
npx gitnexus impact Lease::annotate_result --direction upstream --repo nanobot-rs
npx gitnexus impact step_process_response --direction upstream --repo nanobot-rs
~~~

If lookup is ambiguous, rerun with `--file src/agent/lease.rs` or the UID GitNexus prints.

- [ ] **Step 2: Write failing zero-renewal tests**

~~~rust
#[test]
fn zero_renewal_lease_requires_final_answer_at_exhaustion() {
    let mut lease = Lease::new(2, 0);
    assert_eq!(lease.admit_batch(2), BatchAdmission::Admitted);
    let note = lease.annotate_result("evidence");
    assert!(note.contains("final answer"));
    assert!(!note.contains("findings:/next:/will:"));
    assert_eq!(
        lease
            .try_renew("Findings: x\nNext: y\nWill: z")
            .missing_field(),
        "out_of_leases",
    );
}
~~~

Add a local loop regression supplying 12 successful tools followed by a valid renewal checkpoint; assert no 13th tool executes and no renewal nudge appears.

- [ ] **Step 3: Run tests and verify RED**

~~~bash
cargo test --lib agent::lease::tests::zero_renewal_lease_requires_final_answer_at_exhaustion -- --exact --nocapture
cargo test --lib agent::agent_loop::tests::local_lease_does_not_renew_after_twelve_calls -- --exact --nocapture
~~~

Expected: annotation advertises renewal and local preparation grants three renewals.

- [ ] **Step 4: Configure lease by runtime mode**

Define:

~~~rust
pub const LOCAL_MAX_LEASE_RENEWALS: u32 = 0;
~~~

Initialize:

~~~rust
let max_renewals = if core.mode().is_local() {
    crate::agent::lease::LOCAL_MAX_LEASE_RENEWALS
} else {
    crate::agent::lease::DEFAULT_MAX_LEASES_PER_TURN
};
let lease = Lease::new(DEFAULT_TOOLS_PER_LEASE, max_renewals);
~~~

When an exhausted lease has zero renewals, `annotate_result` must require the next response to be final; only leases with remaining renewals advertise `findings:/next:/will:`. Keep the existing cloud renewal path.

- [ ] **Step 5: Verify GREEN and preserve cloud coverage**

~~~bash
cargo test --lib agent::lease::tests::zero_renewal_lease_requires_final_answer_at_exhaustion -- --exact --nocapture
cargo test --lib agent::agent_loop::tests::local_lease_does_not_renew_after_twelve_calls -- --exact --nocapture
cargo test --lib agent::lease::tests -- --nocapture
cargo test --lib agent::agent_loop::tests::lease_exhaustion_keeps_one_higgs_epoch -- --exact --nocapture
~~~

Move the existing valid-renewal integration setup to a non-local/cloud harness, or retain cloud renewal coverage through the real `Lease::new(..., 3)` state-machine tests if the integration harness would require unrelated changes.

- [ ] **Step 6: Scope-check and commit**

~~~bash
npx gitnexus detect-changes --repo nanobot-rs
git diff --check
git add src/agent/lease.rs src/agent/prepare_context.rs src/agent/agent_loop/response.rs src/agent/agent_loop/tests.rs
git commit -m "fix(lease): stop local turns after twelve tools"
~~~

---

### Task 4: Cross-path Regression and Performance Verification

**Files:**
- Modify only if a new failing regression requires it: files already listed in Tasks 1-3.
- Update: `docs/superpowers/plans/2026-08-02-append-only-replay-tool-output.md` checkboxes.

**Interfaces:**
- Consumes: Tasks 1-3.
- Produces: fresh correctness, build, formatting, scope, and speed evidence.

- [ ] **Step 1: Run all targeted suites**

~~~bash
cargo test --lib session::filters::tests -- --nocapture
cargo test --lib agent::tool_engine::tests -- --nocapture
cargo test --lib agent::lease::tests -- --nocapture
cargo test --lib agent::agent_loop::tests -- --nocapture
~~~

Expected: zero failures. If one fails, write the smallest new failing regression before changing production code.

- [ ] **Step 2: Run full correctness and build tracks**

~~~bash
cargo test
cargo build
~~~

Expected: both exit 0; record fresh passed/ignored counts.

- [ ] **Step 3: Run formatting and whitespace checks**

~~~bash
cargo fmt --all -- --check
git diff --check
~~~

Expected: no formatting or whitespace errors.

- [ ] **Step 4: Run the hot-path speed track**

~~~bash
scripts/turn_bench.sh
~~~

Expected: no matched speed regression outside the script's accepted variance. Record append-only prompt deltas and suffix-size evidence when emitted.

- [ ] **Step 5: Verify graph scope**

~~~bash
npx gitnexus detect-changes --scope all --repo nanobot-rs
~~~

Expected: only session replay, prompt-cache, tool persistence/shaping, and lease flows. Investigate any unrelated process.

- [ ] **Step 6: Audit the branch against the design**

~~~bash
git diff be33003...HEAD -- src/agent src/session src/turn_stream.rs docs/superpowers/specs docs/superpowers/plans
~~~

Confirm:

- ordinary retained turns do not clear fingerprints/watermarks;
- replay never changes stored tool content;
- every exact body is stored before its prompt row;
- output detail is bounded only while appending;
- LCM thresholds are untouched;
- tool definitions remain stable through exhaustion;
- local renewal count is zero.

- [ ] **Step 7: Commit verification bookkeeping only if changed**

Run `npx gitnexus detect-changes --repo nanobot-rs` first, then:

~~~bash
git add docs/superpowers/plans/2026-08-02-append-only-replay-tool-output.md
git commit -m "docs(agent): record append-only replay verification"
~~~

Skip this commit if no tracked plan content changed.
