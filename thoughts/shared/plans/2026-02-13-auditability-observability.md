# Plan: Auditability & Observability for Agent Loop

**Date**: 2026-02-13
**Status**: DRAFT
**Prerequisite**: Memory architecture overhaul (cc3626e) — landed

## Problem Statement

With 1M context (Opus 4.6), the token budget rarely triggers trim_to_fit or compaction. Old messages accumulate indefinitely, causing:

1. **Context rot**: The LLM confuses past turns with current state. Stale tool results from 50 turns ago pollute reasoning about the current question.
2. **Phantom tool calls**: The LLM claims "I read the file" or "I ran the command" without any actual tool call. Provenance verification catches some fabrications post-hoc, but the user has no real-time visibility into what's happening.
3. **Invisible agent state**: No way to inspect what the LLM is actually seeing (message count, token usage, working memory contents, tool call history) without enabling RUST_LOG=debug and parsing stderr.

The provenance system (dc34208) adds verification but it's a safety net, not observability. The user needs to *see* the agent working, not just trust it.

## Design Principles

- Observability should be visible in the REPL by default, not behind debug flags
- Audit data should be structured and queryable, not buried in logs
- Context hygiene should be proactive (age-based eviction), not reactive (size-based only)
- Changes should be non-invasive to the hot path (agent loop latency stays the same)

---

## Phase 1: REPL Observability

### Change 1: Per-turn status line

Show a compact status line after each assistant response:

```
ctx: 12.4K/1M | msgs: 34 | tools: 3 (read_file, exec, recall) | wm: 820tok
```

**Files**:
- `src/repl.rs` — add status line after response rendering
- `src/agent/agent_loop.rs` — return turn metadata alongside response

**Implementation**:
- Add a `TurnMetadata` struct returned from `process_message()`:
  ```rust
  struct TurnMetadata {
      context_tokens: usize,
      max_context: usize,
      message_count: usize,
      tools_called: Vec<String>,
      working_memory_tokens: usize,
      oldest_message_age_turns: usize,
  }
  ```
- Render as a dim grey line below the response in the REPL
- Gate behind a config flag (default: on for CLI, off for gateway)

### Change 2: Tool call indicators in REPL

Currently tool events are emitted but only visible via `tool_event_tx` in streaming mode. Surface them as inline indicators:

```
  [read_file] /tmp/test.txt → 1.2K chars (ok, 45ms)
  [exec] ls -la → 340 chars (ok, 120ms)
```

**Files**:
- `src/repl.rs` — render tool events inline during streaming
- Already have `ToolEvent::CallStart` / `CallEnd` — just need REPL rendering

### Change 3: `/context` REPL command

Dump the current context state on demand:

```
/context
  System prompt: 2,840 tokens
  History: 31 messages (oldest: 47 turns ago)
  Working memory: 820 tokens (2 compaction summaries)
  Learning context: 3 tool patterns
  Tool definitions: 1,200 tokens
  Total: 12,400 / 1,000,000 tokens (1.2%)
```

**Files**:
- `src/repl.rs` — add `/context` command handler
- `src/agent/agent_loop.rs` — expose context introspection method

### Change 4: `/memory` REPL command

Show working memory contents for the current session:

```
/memory
  ## Compaction Summary (2026-02-13T14:30:00Z)
  User asked about Rust ownership. Discussed borrow checker...

  ## Tool Patterns
  - exec: 8/10 succeeded recently
  - Recent errors: exec failed: command not found
```

**Files**:
- `src/repl.rs` — add `/memory` command
- Already have `WorkingMemoryStore::get_context()` — just expose it

---

## Phase 2: Audit Trail

### Change 5: Structured turn audit log

Write a per-session audit log (JSONL) with one entry per turn:

```json
{
  "turn": 7,
  "timestamp": "2026-02-13T14:30:00Z",
  "user_message": "Read /tmp/test.txt",
  "context_tokens": 12400,
  "message_count": 34,
  "tools_called": [{"name": "read_file", "id": "tc_1", "ok": true, "duration_ms": 45, "result_chars": 1200}],
  "response_tokens": 150,
  "provenance": {"verified": 3, "fabricated": 0, "redacted": 0},
  "working_memory_tokens": 820
}
```

**Files**:
- `src/agent/audit.rs` — extend existing AuditLog with turn-level summary
- `src/agent/agent_loop.rs` — collect and write turn summary after each turn
- Store at `~/.nanobot/sessions/{key}.audit.jsonl`

### Change 6: Session replay (`/replay` command)

Dump the exact messages array that was sent to the LLM on the last turn:

```
/replay
  [0] system: "# nanobot\n\nYou are nanobot..." (2840 tokens)
  [1] user: "Hello" (2 tokens)
  [2] assistant: "Hi! How can I help?" (8 tokens)
  ...
  [33] user: "Read /tmp/test.txt" (6 tokens)
```

With `/replay full` to dump actual content, or `/replay N` to show message N.

**Files**:
- `src/repl.rs` — add `/replay` command
- `src/agent/agent_loop.rs` — stash last turn's messages array for inspection

---

## Phase 3: Context Hygiene

### Change 7: Age-based message eviction

The current system only evicts based on token count. For large context windows, this means messages from 100+ turns ago persist. Add age-based eviction:

- Track turn number on each message (add `_turn: N` metadata)
- In trim_to_fit Stage 2, prefer dropping messages older than `max_message_age_turns` (default: 50)
- Configurable via `agents.defaults.maxMessageAgeTurns`

**Files**:
- `src/agent/agent_loop.rs` — tag messages with turn number
- `src/agent/token_budget.rs` — age-aware eviction in Stage 2

### Change 8: Configurable compaction threshold

Currently hardcoded at 66.6% of context. For 1M context that's 660K tokens before compaction fires — way too late. Make it configurable:

```json
{
  "memory": {
    "compactionThresholdPercent": 20,
    "compactionThresholdTokens": 50000
  }
}
```

Use whichever threshold is hit first. Default: 66.6% or 100K tokens, whichever is smaller.

**Files**:
- `src/config/schema.rs` — add threshold config
- `src/agent/compaction.rs` — use configurable threshold

### Change 9: Phantom tool call detection

Beyond provenance verification (which checks claims against audit log), add real-time detection of phantom patterns:

- If the LLM's response mentions tool names but the turn had zero tool calls, flag it
- If the LLM says "I found" / "The result shows" but no tool was called, inject a system reminder: "You did not call any tools this turn. Do not claim tool results."

**Files**:
- `src/agent/agent_loop.rs` — add phantom detection after final_content
- `src/agent/provenance.rs` — add `detect_phantom_claims()` function

---

## Execution Order

Recommended by user impact:

1. Change 1 (status line) — immediate visibility, low risk
2. Change 2 (tool indicators) — already have events, just rendering
3. Change 3 (/context) — debug power tool
4. Change 4 (/memory) — debug power tool
5. Change 8 (compaction threshold) — fixes the root cause for large context
6. Change 7 (age eviction) — prevents stale accumulation
7. Change 5 (audit log) — structured data for analysis
8. Change 9 (phantom detection) — catches fabrication in real time
9. Change 6 (/replay) — deep debugging

---

## Open Questions

- UNCONFIRMED: Should the status line be colorized differently when context usage exceeds a threshold (e.g. yellow at 50%, red at 80%)?
- UNCONFIRMED: Should `/replay` write to a file or stdout? Large message arrays could flood the terminal.
- UNCONFIRMED: For age-based eviction, should tool result messages age faster than user/assistant messages?
- UNCONFIRMED: Should phantom detection be a hard block (inject system message) or soft (just log)?
