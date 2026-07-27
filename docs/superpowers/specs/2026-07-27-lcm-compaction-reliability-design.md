# LCM Compaction Reliability — TDD Red→Green

Date: 2026-07-27
Owner: nanobot-rs (LCM + compaction hot path)

## Problem (source-grounded)

Live session `20260727_094539_eeab48` showed three failures between 12:09–13:14.
All three are reproducible from code; none require Higgs.

### Bug #1 — `length` finish_reason at Level 1
`src/agent/compaction.rs:153-164` computes `summary_token_limit` with a single
`SUMMARY_COMPRESSION_RATIO = 8` for both escalation levels. For a 11506-token
block (the live failure), `max_tokens ≈ 1439`. A Level-1 `preserve_details`
handoff + manifest does not fit → `finish_reason="length"` → bail at
`src/agent/compaction.rs:312-317`. 5 of 6 live failures were this pattern.

### Bug #2 — Auto_expand undoes compaction
`src/agent/lcm.rs:939-1065` runs every turn. Two structural defects:

1. `auto_expand`'s headroom check (`lcm.rs:970-975`) uses
   `self.active_tokens()`, but reinjected originals go into the *wire*
   (`agent_loop/shared.rs:1830`), not `self.active`. Headroom is over-counted.
2. `auto_expanded: HashSet<usize>` (`lcm.rs:984`) only blocks re-expansion of
   the *same* node. Each new compaction creates a new node, which is a fresh
   expansion candidate the very next turn.

Live proof: 12:12:42 compaction succeeded (12463→1398 tokens). At 12:13:06
auto_expand reinjected +12463 tokens for the 24-second-old summary.

### Bug #3 — Manifest sources schema mismatch
Prompt at `src/agent/compaction.rs:85-88` says `"sources": [id, ...]` without
specifying integer type. Transcript labels (`compaction.rs:519-527`) render as
`[message_id: 55531]`. The model emits `"msg 55531"` (string) and serde rejects
the whole manifest: `invalid type: string "msg 55531", expected usize`.

## Acceptance Criteria

1. **Compaction succeeds on the first attempt** for blocks similar to the live
   failure (11506 tokens). The Level-1 summarizer must receive enough
   `max_tokens` to produce a complete `preserve_details` handoff.
2. **Auto_expand feedback loop is structurally impossible.** The exact live
   sequence (compact → next-turn reinject of the entire fresh summary's
   sources) cannot occur.
3. **Higgs-nightly caching contract preserved.** Per
   `~/Dev/higgs-nightly/SESSION_CONTEXT_GOVERNOR.md` the failure mode is
   `ColdPromptTooLarge` ExactBootstrap when the wire crosses the 24576-token
   retained-session cap. Our fix must keep the wire bounded; no higgs-nightly
   code changes.

## Design

### Fix 1 — Per-mode compression ratio (criterion #1)

`summary_token_limit` becomes mode-aware:
- `preserve_details` → ratio 4 (Level-1 faithful handoff + manifest fits)
- `bullet_points` → ratio 8 (unchanged, aggressive)

For the 11506-token block, Level-1 returns `max_tokens ≈ 2876` instead of
1439. The existing `MAX_SUMMARY_TOKENS = 4_096` cap and `summary_max_tokens =
512` floor are unchanged.

**Files**: `src/agent/compaction.rs` only. No wire/format change → no caching
impact.

### Fix 2 — Auto_expand cannot undo compaction (criteria #2 + #3)

Two structural changes, both required to make the loop impossible:

1. **Fresh-summary cooldown (1 turn).** `SummaryNode` gains
   `created_at_turn: u64`. `auto_expand` skips nodes where
   `current_turn - node.created_at_turn < 1`. The summary created by the most
   recent compaction is ineligible until the next turn. This kills the
   24-second-later reinject pattern structurally.
2. **Wire-aware budget.** `auto_expand` signature gains a `wire_tokens: usize`
   parameter (the actual rendered prompt size including prior expansions, which
   `agent_loop/shared.rs` already computes as `prompt_total_estimate` at line
   1977-1979). Headroom = `hard_limit.saturating_sub(wire_tokens)`. Reinjecting
   originals now consumes the *real* remaining budget; the wire cannot cross
   tau_hard via reinjection.

**Files**: `src/agent/lcm.rs` (SummaryNode, auto_expand, compact stamps turn),
`src/agent/agent_loop/shared.rs` (pass `wire_tokens`).

**Caching impact**: strictly positive. The reinjection that pushed the wire
toward the 24576 Higgs retained cap cannot occur; the wire stays under tau_hard
of model context (well below Higgs cap). No higgs-nightly changes (per Governor
Non-Goals: "Do not implement arbitrary KV eviction without prompt rewriting").

### Fix 3 — Lenient manifest sources (criterion #1 supporting)

Tighten the prompt *and* make deserialization lenient:
- Custom deserializer for `ManifestItem::sources` accepts integers, numeric
  strings (`"123"`), and `"msg 55531"`-style strings. Extracts the first
  integer; non-numeric values are skipped (not fatal).
- Prompt at `src/agent/compaction.rs:85-88` becomes explicit:
  `"sources": [<integer id>, ...]` with a concrete example.

**Files**: `src/agent/lcm.rs` (deserializer), `src/agent/compaction.rs`
(prompt).

## What I am NOT touching

- The 80-message deterministic-truncation path
- `MAX_SUMMARY_TOKENS = 4_096` cap
- `summary_max_tokens = 512` floor
- Higgs-nightly code (criterion #3 is satisfied on the nanobot side)
- The `auto_expanded` HashSet (stays as session-wide idempotency guard)

## TDD test list (RED first, watch fail, then GREEN)

All tests live in existing test modules; use existing mocks
(EchoSummarizerMock, RecordingProvider, FinishReasonProvider).

### Fix 1
- `compaction_level1_max_tokens_accommodates_faithful_handoff`
  For 11506-token input, `summarize_for_lcm(.., "preserve_details")` calls the
  provider with `max_tokens >= 2876`. RED: today returns 1439.

### Fix 2
- `auto_expand_skips_freshly_created_summary`
  Compact → immediately call auto_expand with relevance 1.0 user message.
  Must return empty. RED: today reinjects.
- `auto_expand_budget_uses_wire_tokens_not_active`
  Build an engine with small internal active but pass `wire_tokens` already at
  `hard_limit`. Must return empty. RED: today uses `active_tokens()`.
- `auto_expand_cannot_reinject_more_than_just_compacted`
  End-to-end: ingest 12463 tokens of conversation, compact (succeeds), call
  auto_expand next turn with same user message → returns empty.

### Fix 3
- `manifest_sources_accepts_string_with_msg_prefix`
  JSON `{"text":"x","sources":["msg 55531", 123, "456"]}` deserializes to
  `sources: vec![55531, 123, 456]`. RED: serde rejects mixed types.
- `manifest_sources_skips_non_numeric_strings`
  JSON `{"text":"x","sources":["hello", 789]}` deserializes to
  `sources: vec![789]`. RED: serde fails the whole manifest.

## Out of scope (deliberately)

- Governor workstream N1 (Higgs pressure model). Not needed for these three
  criteria — the wire-aware budget in Fix 2 covers criterion #3.
- Removing auto_expand entirely. Bigger behavior change; not required.
