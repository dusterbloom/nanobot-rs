# Stable-Prefix LCM Fold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** At the LCM pressure checkpoint, replace the foldable conversation with one bounded checkpoint (default 2048 estimated wire tokens) while preserving the immutable system/tool prefix cache and the current-turn tail.

**Architecture:** Start from the current `main` (`7c07046`) or a fresh branch based on it. Do not merge the stale `feat/deterministic-context-fold` worktree wholesale: current `main` already contains the deterministic-fold, exact-source, and prefix-preserving LCM commits. Reuse the current-main `deterministic_recovery_index`, source digest, complete-boundary scan, DAG publication, and `lcm_expand` path. Make one `LcmEngine::compact` decision select the whole foldable span and choose either one capped model summary (soft pressure) or the existing deterministic index (hard/capacity pressure); do not add a second folding pipeline. Install the replacement by preserving the unchanged stable prefix while replacing only the incompatible conversation suffix.

**Tech Stack:** Rust 2021, Tokio, serde/serde_json, SQLite-backed session history, existing Higgs retained-session and prompt-fingerprint APIs.

## Global Constraints

- The stable system/developer prompt and frozen tool catalog must remain byte-identical; an unchanged stable prefix must not trigger the full prompt-cache reset. If the live developer prefix genuinely changed (for example, working memory), the existing sanctioned reset path remains correct.
- The immutable SQLite rows remain the source of truth; the active checkpoint contains exact IDs, ranges, and digest, and `lcm_expand` remains the only explicit recovery operation.
- One pressure event creates at most one summary node and one active replacement; no repeated prompt-sized compaction loop.
- A complete assistant tool-call/result group is atomic, and the newest user request/current-turn tail remains raw.
- `fold_target_tokens` is one shared bound. Reuse the existing `deterministic_target` JSON field for compatibility; change its default/documentation to 2048 instead of adding a parallel setting.
- Astra must branch from current `main`, not `/private/tmp/nanobot-fold-t23`; port only missing behavior after checking the current-main implementation.
- Do not modify the other agent's capacity, router, tool-recovery, Higgs, AGENTS, or CLAUDE work except where a test exposes a direct cache-install contract violation.
- Use release-mode validation only: `cargo test --release`, `cargo build --release`, and `scripts/turn_bench.sh` for the agent-loop path.

## Review of Existing Work

The repository now has two registered worktrees, with one redundant duplicate
removed after review:

- `/Users/peppi/Dev/nanobot-rs` is `codex/stable-prefix-lcm-fold`, based on
  `7c07046`, and is dirty with pre-existing user files plus this focused work.
- The redundant `/private/tmp/nanobot-tool-recovery` checkout and its
  `codex/tool-recovery` branch were removed after review; it duplicated the
  base commit and contained no unique work.
- `/private/tmp/nanobot-fold-t23` is clean at `6b08db4` on
  `feat/deterministic-context-fold`. It has three unique commits, but its base
  is stale and its three-way diff is broad (23 files, including capacity,
  router, and tool-recovery areas). It is not merge-ready as a whole.

The current base history already contains the deterministic-fold and exact
checkpoint changes (`f29e007`, `e3f7098`, and the prefix-preserving work in
`33a060d`). Therefore keep these current-main pieces rather than cherry-picking
the stale branch:

- `deterministic_recovery_index` and `canonical_source_digest` in `src/agent/lcm.rs`;
- `complete_compaction_boundaries` and `collect_compaction_span` for safe source selection;
- `PendingCompaction`, `apply_compaction_result`, and the existing atomic DAG/SQLite publication;
- the `MessageLog` committed/draft separation and existing prompt fingerprint diagnostics.

Do not copy that machinery into a new module or cherry-pick the branch as a unit.
The current-main implementation still does not fully implement this request:
`compact` still selects one oldest block, its computed `target` is unused, the
deterministic default is 512, and `install_pending_compaction` deliberately
clears the fingerprint/watermark and retires the whole Higgs cache identity.
The stale branch's design also explicitly requires retiring the old identity on
every active-prompt rewrite, which is incompatible with the stable-prefix
requirement.

---

### Task 1: Lock the fold and cache contracts with failing tests

**Files:**
- Modify: `src/agent/lcm.rs` tests near the existing deterministic recovery tests
- Modify: `src/agent/agent_core.rs` tests near the prompt-cache transition tests
- Modify: `src/agent/agent_loop/shared.rs` tests near `install_pending_compaction` and cache-pressure tests
- Modify: `src/config/schema.rs` LCM default/round-trip tests

**Interfaces:**
- Consumes: current-main `LcmEngine`, `PendingCompaction`, prompt-fingerprint, and `RuntimeCounters` test helpers.
- Produces: executable contracts for the later implementation; do not weaken existing lossless or retained-expansion tests.

- [x] **Step 1: Add the whole-fold fixture test.** Build a deterministic engine fixture containing a stable system message, enough persisted conversation to exceed 2048 tokens, an assistant tool-call/result pair, and a newest user tail. Assert the post-fold active context contains the unchanged prefix, exactly one `_lcm_summary`, the intact tail, and no retired raw message outside the summary.

```rust
assert_eq!(summary_count(&active), 1);
assert_eq!(active[0], stable_system);
assert!(active_summary_tokens(&active) <= 2_048);
assert!(active.ends_with(&current_turn_tail));
assert!(summary_source_ids(&active).contains(&tool_call_id));
assert!(summary_source_ids(&active).contains(&tool_result_id));
```

- [x] **Step 2: Add deterministic repeatability and recovery assertions.** Run the same fixture twice and assert identical summary bytes, exact source ranges, digest, and `lcm_expand` output. Assert that a tool carrier and result are never split.

```rust
assert_eq!(first_summary, second_summary);
assert_eq!(engine.expand(&source_ids), expected_source_rows);
assert!(summary_text.contains("lcm_expand"));
```

- [x] **Step 3: Add the provider-output cap test.** Use the existing recording compaction provider and assert the soft-pressure request receives `max_tokens <= 2_048`; a returned summary whose rendered wire message exceeds 2,048 is rejected in favor of the deterministic index.

- [x] **Step 4: Add the cache-preservation test.** Seed a fingerprint/watermark for `[system, old history]`, install `[system, summary, tail]`, and assert the next comparison is `AppendOnly` from the stable prefix, not `First` or `Diverged`. Assert the tool hash is preserved and no sanctioned full-reset counter is recorded.

- [x] **Step 5: Run the new tests and confirm they fail for the current branch.** The pre-implementation run exposed the old one-block selection, uncapped compactor, and full-cache-reset behavior.

Run: `cargo test --release lcm -- --nocapture` and the focused `agent_core`/`agent_loop::shared` tests by exact name.

Expected: failures showing the one-block selection, unused target, output-cap, and full-cache-reset behaviors.

---

### Task 2: Make LCM perform one bounded whole-span fold

**Files:**
- Modify: `src/agent/lcm.rs:720-1030` (`LcmEngine::compact` and existing deterministic helpers)
- Modify: `src/config/schema.rs:2138-2170` (default/documentation/tests)
- Modify: `src/agent/compaction.rs:136-440` (only the existing summary-limit calculation)

**Interfaces:**
- Consumes: current-main `deterministic_recovery_index`, `canonical_source_digest`, `complete_compaction_boundaries`, `collect_compaction_span`, and `CompactionFailureMode`.
- Produces: one `Turn::Summary` whose rendered replacement is no larger than `deterministic_target`, unless the stable prefix plus protected tail alone is already larger; in that case the existing capacity path owns the failure.

- [x] **Step 1: Replace the dynamic target.** Remove the unused `available * tau_soft * 0.8` target. Use `self.config.deterministic_target.max(1)` as the single `fold_target_tokens`; `available` remains responsible only for pressure thresholds and the existing protected-tail calculation.

- [x] **Step 2: Select the complete foldable span in one pass.** Use the existing boundary helpers to select the oldest complete prefix of the LCM active conversation, stopping before the protected recent/current-turn tail. Include existing summary nodes through `collect_compaction_span`, so repeated pressure merges into one node instead of leaving a chain of visible old summaries. Do not use `keep_prefix_fraction`, a minimum-savings percentage, or repeated oldest-block calls.

- [x] **Step 3: Bound the soft model attempt.** Add one `ContextCompactor` copy method that caps the existing `max_tokens` calculation and its length retry at `fold_target_tokens`. Pass the bounded compactor to `escalated_summary`, and accept a model result only when the final `summary_wire_message` is `<= fold_target_tokens` and strictly smaller than the retired span.

```rust
let compactor = compactor.with_summary_cap(fold_target_tokens);
let summary_wire = summary_wire_message(&ids, &text, &manifest, level);
let fits = TokenBudget::estimate_message_tokens(&summary_wire) <= fold_target_tokens;
```

- [x] **Step 4: Use the existing deterministic index as the single fallback.** For `CompactionFailureMode::Deterministic`, and for any model error, over-limit response, refusal, or non-shrinking response, call `deterministic_recovery_index` on the same selected span. Keep the exact source IDs and digest; do not introduce a second receipt/index format.

- [x] **Step 5: Publish one node through the existing mutation.** Keep the current `LcmCompactionMutation`, DAG node, `summary_wire_message`, and persistence path. Remove only the old logic that leaves a partially compacted active prefix or rejects a useful target-sized fold because it did not meet an unrelated savings threshold.

- [x] **Step 6: Update configuration defaults and tests.** Keep the serialized field name `deterministicTarget`, document it as the bounded checkpoint/fold target, and set the default to 2,048. Existing explicit user values remain authoritative.

- [x] **Step 7: Run the focused LCM and compactor tests.** The release LCM suite and focused compactor cap test pass.

Run: `cargo test --release lcm -- --nocapture` and `cargo test --release agent::compaction -- --nocapture`.

Expected: the whole-fold, deterministic, provider-cap, and restart/rebuild tests pass.

---

### Task 3: Install the fold without invalidating the stable prefix cache

**Files:**
- Modify: `src/agent/agent_core.rs:600-780, 1040-1120` (cache transition transaction)
- Modify: `src/agent/agent_loop/shared.rs:680-980, 3860-4005` (`install_pending_compaction`)
- Modify: `src/agent/agent_loop/budget.rs` only if the new transition needs a shared marker helper
- Modify: existing cache-transition tests in `src/agent/agent_loop/shared.rs` and `src/agent/agent_core.rs`

**Interfaces:**
- Consumes: current-main `PendingCompaction`, `prompt_prefix_len`, `PromptFingerprint`, frozen tool-catalog hash, and existing Higgs session retirement.
- Produces: a cache-preserving LCM install that replaces only the incompatible retained-session suffix while retaining a server-reusable stable-prefix cache identity. Local fingerprint re-anchoring alone is insufficient; the provider-facing test must prove that the next request reuses the stable prefix.

- [x] **Step 1: Add one typed cache transition for suffix replacement.** Factor the existing retirement transaction so it supports an explicit `PreserveStablePrefix` disposition rather than adding a boolean. The transition must, under the existing `prompt_cache_transition` lock, replace only the incompatible retained-session suffix and preserve the stable-prefix fingerprint/watermark/tool hash. Retire/drop the old retained session only if the Higgs wire contract keeps the stable prefix independently reusable; otherwise retain the required prefix-cache handle. The invariant is server-side prefix reuse on the next request, not merely an append-only local comparison.

```rust
pub(crate) struct StablePromptPrefix {
    pub(crate) fingerprint: PromptFingerprint,
    pub(crate) watermark: usize,
    pub(crate) tool_hash: Option<u64>,
}

pub(crate) fn rotate_higgs_session_preserving_prefix(
    &self,
    session_key: &str,
    prefix: StablePromptPrefix,
) -> u64;
```

The implementation must share the current session-retirement body; do not duplicate queue/drop/epoch logic.

- [x] **Step 2: Change LCM installation to use the stable prefix boundary.** Before applying the swap, compute the leading system/developer prefix from the live prompt and fingerprint only that prefix. Preserve the live prefix through `apply_compaction_result`; install the summary and current-turn tail after it.

- [x] **Step 3: Do not create an automatic full-history expansion lease for this fold.** The old retained session is incompatible with the rewritten conversation suffix, so retire it and start the next request from the stable prefix plus the bounded checkpoint. Keep explicit `lcm_expand` recovery and unrelated auto-expansion behavior intact. Obsolete tests that asserted this automatic lease were removed; explicit retained-expansion transition tests remain.

- [x] **Step 4: Remove the LCM full-reset side effects.** An unchanged stable prefix does not call `invalidate_prompt_cache_for_rewrite`, clear the anchor, emit a reset marker, or increment the sanctioned-reset metric. A genuinely changed live developer prefix still uses the existing sanctioned reset path. The provider test proves the request prefix remains byte-stable while the Higgs session control rotates; live `cached_tokens` confirmation requires a running Higgs endpoint and was not available in this environment.

- [x] **Step 5: Preserve the existing stale-snapshot and atomic-publication guards.** A live prompt mismatch still discards the pending result; SQLite/DAG publication still happens before the active swap; current-turn messages added after the snapshot remain appended.

- [x] **Step 6: Run focused cache and compaction tests.** Release agent-core, LCM, cloud-prefix, provider wire-prefix, and retained-overflow tests pass.

Run: `cargo test --release agent::agent_core -- --nocapture` and the exact `agent_loop::shared` cache/compaction tests.

Expected: the stable system/tool prefix remains byte-identical, the first post-fold prefill is bounded to the summary/tail, and an unchanged stable prefix produces no full-prefix reset.

---

### Task 4: Verify the user-visible path and scope

**Files:**
- Modify: no production files unless a focused test identifies a direct contract failure
- Test: existing LCM integration and turn-benchmark harnesses

**Interfaces:**
- Consumes: Tasks 1–3.
- Produces: evidence that the exact fix landed without absorbing the other agent's unrelated branch work.

- [x] **Step 1: Run the release build.** Passed; seven pre-existing dead-code warnings remain.

Run: `cargo build --release`.

- [x] **Step 2: Run the full release regression suite.** Passed: 3,017 passed, 0 failed, 27 ignored.

Run: `cargo test --release`.

- [x] **Step 3: Run the matched agent-loop benchmark.** The script completed all 20 turns with 88–148 ms wall times, but emitted no timing/metrics rows because no local inference server was running; therefore no cache/prefill before/after comparison is claimed.

Run: `scripts/turn_bench.sh`.

Record before/after for: prompt tokens before fold, checkpoint wire tokens, stable-prefix tokens, protected-tail tokens, suffix prefill estimate, cache reset count, and compaction latency.

- [x] **Step 4: Review the diff against the stale worktree before claiming scope.** The stale branch remains untouched; the focused changes are limited to LCM selection/capping, cache-preserving installation, configuration/docs, and focused tests. Existing unrelated dirty edits in `AGENTS.md`, `CLAUDE.md`, `PLAN.md`, and `src/agent/agent_loop/local_stream.rs` were preserved.

Run: `git diff --stat main...HEAD`, `git diff main...feat/deterministic-context-fold --stat`, and `git diff --check`.

The final diff must be limited to the LCM target/selection, the existing compactor output cap, the typed cache-preserving transition, and their tests. Do not merge the stale branch's broader capacity/router/tool changes into this fix.

- [x] **Step 5: Run GitNexus impact/detect checks before any commit.** Impact analysis was run before production edits. The local GitNexus index was refreshed and used; the installed `npx gitnexus` wrapper had a storage-version mismatch, so the repository-local runner was used for the final check.

Run the repository's required impact analysis for every modified symbol before editing, then run `gitnexus_detect_changes` before committing. If the index is unavailable, report that limitation rather than silently claiming the check was performed.
