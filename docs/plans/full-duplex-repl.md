# Full-Duplex REPL: Instant Cancel + Priority Message Injection

**Date:** 2026-02-15  
**Status:** Design  
**Problem:** During agent streaming/tool execution, the user is completely locked out.

## Problems

1. **No quick abort** — In Claude Code, ESC+ESC instantly stops the model. In nanobot, Ctrl+C is slow (SIGINT → async task → CancellationToken propagation) and output keeps flowing while waiting.

2. **Conversation is blocked** — While the agent runs a long chain of tool calls, the user can't type anything. Input channel is locked until the agent is "done."

3. **No way to inject context mid-task** — If the user sees the agent going down the wrong path (e.g., tool runner burning 150 iterations on irrelevant files), there's no way to say "hey stop, try this instead" without killing the whole thing.

**Root cause:** The conversation is half-duplex when it should be full-duplex.

## Architecture

```
┌─────────────────────────────────────────────┐
│              stream_and_render()            │
│                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ Stream   │  │ Key      │  │ Tool     │  │
│  │ printer  │  │ watcher  │  │ events   │  │
│  │ (deltas) │  │ (raw)    │  │ display  │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  │
│       │              │              │        │
│       │         ESC+ESC → cancel    │        │
│       │         ` → inject prompt   │        │
│       │         Ctrl+C → cancel     │        │
│       │              │              │        │
│       │         ┌────▼─────┐        │        │
│       │         │ priority │        │        │
│       │         │ channel  │        │        │
│       │         └────┬─────┘        │        │
│       │              │              │        │
│  ┌────▼──────────────▼──────────────▼────┐  │
│  │           agent_loop                   │  │
│  │  (checks priority_rx between tools)   │  │
│  └───────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

## Current State (what exists)

- **Input:** `readline_async()` blocks on a background thread, only active between turns
- **During streaming:** `stream_and_render()` takes full control — readline is NOT active
- **Ctrl+C:** Spawns `tokio::signal::ctrl_c()` task → cancels `CancellationToken` → propagates to agent loop (slow, multi-hop)
- **Voice mode already has a key watcher:** `spawn_interrupt_watcher()` in `tui.rs` uses crossterm raw mode to detect Enter/Ctrl+Space — this is the pattern to reuse

## Implementation

### 1. `repl/mod.rs` — Add `spawn_input_watcher()`

Reuse the pattern from `spawn_interrupt_watcher` (voice mode), but with three actions:

```rust
/// Watches for keypresses during agent streaming.
/// - ESC+ESC (within 500ms): instant cancel
/// - Ctrl+C: cancel (backup)
/// - Backtick (`): open injection prompt
pub(crate) fn spawn_input_watcher(
    cancel_token: tokio_util::sync::CancellationToken,
    inject_tx: mpsc::UnboundedSender<String>,
    done: Arc<AtomicBool>,
) -> std::thread::JoinHandle<()>
```

Uses crossterm raw mode to read keys:
- **On backtick:** temporarily disable raw mode, print `\ninject> ` prompt, read a line with basic stdin (not rustyline — keep it simple), send through `inject_tx`, re-enable raw mode
- **On ESC:** record timestamp. On second ESC within 500ms: `cancel_token.cancel()`
- **On Ctrl+C:** `cancel_token.cancel()` (backup)

~60 lines of code.

### 2. `agent/agent_loop.rs` — Add `priority_rx` parameter

At line ~1255 (the cancellation check between iterations), also check for injected messages:

```rust
// Check for priority user messages
if let Ok(msg) = priority_rx.try_recv() {
    messages.push(Message {
        role: "user".into(),
        content: Some(format!("[PRIORITY USER MESSAGE]: {}", msg)),
        ..Default::default()
    });
    // Don't break — let the LLM see this and adjust
}

// Check cancellation
if cancellation_token.as_ref().map_or(false, |t| t.is_cancelled()) {
    break;
}
```

~10 lines of code.

### 3. `repl/mod.rs` `stream_and_render()` — Wire it up

Replace the current SIGINT-only approach:

```rust
let cancel_token = CancellationToken::new();
let (inject_tx, inject_rx) = mpsc::unbounded_channel();
let done = Arc::new(AtomicBool::new(false));

// Spawn key watcher (handles ESC+ESC, backtick, Ctrl+C)
let watcher = spawn_input_watcher(cancel_token.clone(), inject_tx, done.clone());

// Pass inject_rx to agent loop
let response = agent_loop
    .process_direct_streaming(
        input, session_id, channel, "direct", lang,
        delta_tx, tool_event_tx,
        Some(cancel_token.clone()), Some(inject_rx),
    )
    .await;

done.store(true, Ordering::Relaxed);
watcher.join().ok();
```

~5 lines changed.

### 4. System prompt addition

```
If you see a [PRIORITY USER MESSAGE], the user typed this while you were working.
Acknowledge it briefly and adjust your approach accordingly. This takes precedence
over your current plan.
```

## UX Example

```
> analyze all the rust files in src/

  ▶ exec(find src/ -name "*.rs")  ✓ 230ms
  ▶ read_file(src/main.rs)  ✓ 12ms
  ▶ read_file(src/config.rs)  ...

  [user presses `]
  inject> skip config, focus on agent_loop.rs

  ▶ read_file(src/agent/agent_loop.rs)  ✓ 15ms
  [PRIORITY USER MESSAGE acknowledged, focusing on agent_loop.rs]
  ...
```

Cancel example:
```
> do something expensive

  ▶ exec(find / -name "*.rs")  ...
  [user presses ESC ESC]

  Cancelled.
>
```

## Why This Design

1. **Minimal code** — ~75 lines total across 4 files
2. **Reuses existing patterns** — `spawn_interrupt_watcher` for voice already does raw-mode key watching
3. **Non-breaking** — `inject_rx` is `Option<UnboundedReceiver>`, existing callers pass `None`
4. **Works at the right level** — injection happens between tool iterations, which is exactly when the LLM can actually process new info
5. **ESC+ESC is instant** — no SIGINT, no async propagation, direct token cancel
6. **Backtick is ergonomic** — easy to reach, rarely conflicts with normal typing (user isn't typing during streaming)

## Files to Modify

| File | Change |
|------|--------|
| `src/repl/mod.rs` | Add `spawn_input_watcher()`, update `stream_and_render()` |
| `src/agent/agent_loop.rs` | Add `priority_rx` param, check between iterations |
| `src/agent/agent_loop.rs` | Update `process_direct_streaming()` signature |
| System prompt (AGENTS.md or inline) | Add priority message handling instruction |
