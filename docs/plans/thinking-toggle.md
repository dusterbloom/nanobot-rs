# Thinking Toggle — Implementation Plan

**Status:** TODO
**Priority:** High
**Created:** 2026-02-16
**Context:** Thinking/reasoning is currently always discarded. Should be a first-class
toggle the user controls (Ctrl+T, /think) and the agent can request when needed.
Connects to Layer 2 (status bar / input area improvements).

## Overview

Add a thinking mode toggle that:
1. User can flip with **Ctrl+T** (during input) or **`/think`** slash command
2. Agent can request via system prompt hint ("think harder about this")
3. Prompt line shows current state visually
4. Thinking output is displayed (dimmed) instead of discarded
5. Works across all providers: Anthropic (extended thinking), OpenAI-compat (reasoning_content), ZhiPu (/nothink toggle)

## Architecture

```
┌──────────────────────────────────────────────┐
│  ReplContext                                  │
│  ├── thinking_on: bool  (NEW)                │
│  └── build_prompt() → shows 🧠 when on       │
├──────────────────────────────────────────────┤
│  Input Layer                                  │
│  ├── Ctrl+T during readline → toggle          │
│  ├── /think command → toggle + status msg     │
│  └── Prompt: "🧠>" (thinking) vs ">" (normal) │
├──────────────────────────────────────────────┤
│  Agent Loop                                   │
│  ├── Passes thinking_on to provider           │
│  └── System prompt hint when thinking is on   │
├──────────────────────────────────────────────┤
│  Provider Layer                               │
│  ├── Anthropic: extended_thinking block       │
│  ├── OpenAI-compat: model-specific params     │
│  └── ZhiPu: remove /nothink suffix           │
├──────────────────────────────────────────────┤
│  Display Layer                                │
│  ├── Show thinking in dimmed/collapsed block  │
│  ├── Status bar shows "thinking" indicator    │
│  └── Voice: skip thinking in TTS (keep strip) │
└──────────────────────────────────────────────┘
```

## Implementation Steps

### Step 1: State & Toggle (ReplContext)

**File:** `src/repl/commands.rs`

```rust
// Add to ReplContext struct:
pub thinking_on: bool,

// Add to impl ReplContext:
pub fn toggle_thinking(&mut self) -> bool {
    self.thinking_on = !self.thinking_on;
    self.thinking_on
}
```

**File:** `src/repl/mod.rs` — update `build_prompt()`:

```rust
pub(crate) fn build_prompt(is_local: bool, voice_on: bool, thinking_on: bool) -> String {
    let think_indicator = if thinking_on { "🧠" } else { "" };
    if voice_on {
        format!("{}{}{}~>{} ", think_indicator, BOLD, MAGENTA, RESET)
    } else if is_local {
        format!("{}{}{}L>{} ", think_indicator, BOLD, YELLOW, RESET)
    } else {
        format!("{}{}{}>{} ", think_indicator, BOLD, GREEN, RESET)
    }
}
```

### Step 2: Slash Command `/think`

**File:** `src/repl/commands.rs` — add to dispatch:

```rust
"/think" | "/t" => {
    let on = self.toggle_thinking();
    println!("  {} Thinking mode {}",
        if on { "🧠" } else { "💤" },
        if on { "ON — model will reason deeply" } else { "OFF — fast mode" }
    );
    true
}
```

Update `/help` to include `/think`.
Update `/status` to show thinking state.

### Step 3: Input Area with Bottom Bar (Claude Code Style)

**Goal:** Render a persistent context bar below the input prompt, with toggle hints.

```
> your message here
──────────────────────────────────────────────────
  ~/Dev/nanobot · opus-4-6 · 🧠 thinking
  ⏵⏵ /t toggle thinking · /l toggle local · 3 agents · ctx 12K/1M
```

**Implementation approach — "render below, cursor above":**

1. Before calling `rl.readline(&prompt)`, print the bottom bar (2-3 lines)
2. Use ANSI `\x1b[{n}A` to move cursor back UP to the prompt line
3. Rustyline takes over on the prompt line — user types normally
4. On submit (Enter), the bar scrolls away naturally with the output

**File:** `src/tui.rs` — new function:

```rust
/// Render the Claude Code-style input context bar.
/// Returns the number of lines printed (for cursor repositioning).
pub fn render_input_bar(cwd: &str, model: &str, thinking: bool, agents: usize, ctx_used: usize, ctx_max: usize) -> usize {
    let width = terminal_width();
    let separator = "─".repeat(width.min(80));
    
    // Line 1: separator
    println!("{}{}{}", DIM, separator, RESET);
    
    // Line 2: context
    let think_str = if thinking { " · 🧠 thinking" } else { "" };
    println!("{}  {} · {}{}{}", DIM, cwd, model, think_str, RESET);
    
    // Line 3: hints
    let mut hints = vec!["⏵⏵ /t toggle thinking".to_string()];
    if agents > 0 {
        hints.push(format!("{} agents", agents));
    }
    hints.push(format!("ctx {}/{}", format_tokens(ctx_used), format_tokens(ctx_max)));
    println!("{}  {}{}", DIM, hints.join(" · "), RESET);
    
    3 // lines printed
}
```

**File:** `src/repl/mod.rs` — before readline:

```rust
// Render bottom bar
let bar_lines = tui::render_input_bar(&cwd, &model_name, ctx.thinking_on, agent_count, ctx_used, ctx_max);
// Move cursor back up to prompt line
print!("\x1b[{}A", bar_lines);
io::stdout().flush().ok();

// Now readline takes over on the prompt line
match ctx.rl.readline(&prompt) { ... }
```

**Why not Alt+Tab / Shift+Tab for toggling?**
- Rustyline intercepts most key combos — Alt+Tab is captured by the OS/window manager
- Shift+Tab could work via rustyline's `KeyEvent` binding, but requires a custom `EventHandler`
- `/t` is simpler, works today, and is shown in the hints bar
- **Future:** Can add Shift+Tab via rustyline `ConditionalEventHandler` as a polish item

**Why not Ctrl+T?**
- Ctrl+T is traditionally "transpose characters" in readline/emacs
- Overriding it breaks muscle memory for some users
- `/t` is two keystrokes (same as Ctrl+T) and more discoverable

### Step 4: Provider — Anthropic Extended Thinking

**File:** `src/providers/anthropic.rs`

In both `chat()` and `chat_stream()`, when thinking is enabled:

```rust
// After building the base body json...
if thinking_enabled {
    // Anthropic extended thinking requires:
    // 1. temperature must be 1.0 (Anthropic constraint)
    // 2. Add thinking block with budget_tokens
    body["temperature"] = json!(1);
    body["thinking"] = json!({
        "type": "enabled",
        "budget_tokens": 10000  // configurable, start with 10K
    });
    // max_tokens must be > budget_tokens
    if max_tokens <= 10000 {
        body["max_tokens"] = json!(16000);
    }
}
```

The `interleaved-thinking-2025-05-14` beta flag is already in OAUTH_BETA.

**Streaming:** The stream will emit `content_block_start` with `type: "thinking"`
blocks. Currently these are likely ignored or cause issues. Need to:
1. Detect thinking blocks in the stream parser
2. Forward them to display as a separate channel (not mixed with content)

### Step 5: Provider — OpenAI-Compat (ZhiPu, DeepSeek, etc.)

**File:** `src/providers/openai_compat.rs`

For ZhiPu models:
- When thinking OFF: append `/nothink` to system prompt (current behavior)
- When thinking ON: remove `/nothink`, let model reason naturally
- `reasoning_content` in response: forward to display instead of discarding

For other OpenAI-compat providers (DeepSeek, etc.):
- Some support `reasoning_effort` parameter
- Others use `reasoning_content` in response
- Keep it simple: just stop discarding `reasoning_content` when thinking is ON

### Step 6: Agent Loop — Pass Thinking State

**File:** `src/agent/agent_loop.rs`

The agent loop needs to know if thinking is on. Options:

**Option A: Add to AgentLoop struct**
```rust
pub struct AgentLoop {
    // ... existing fields
    pub thinking_on: bool,
}
```
Pass it through to provider calls. Simple, direct.

**Option B: Add to SharedCoreHandle counters/config**
More global, accessible from anywhere. Better for agent self-toggle.

**Recommendation:** Option B — put it in SharedCoreHandle so the agent can
toggle it via a tool, and providers can read it directly.

```rust
// In SharedCoreHandle or its counters:
pub thinking_on: AtomicBool,
```

### Step 7: Display Thinking Output

**File:** `src/repl/mod.rs` (print_task) and `src/tui.rs`

When thinking content arrives:
1. **During streaming:** Show in dim text with a prefix:
   ```
   💭 Let me analyze the error message...
   💭 The issue is in the borrow checker because...
   💭 Two possible fixes: either clone or use a reference...
   ```
2. **In final re-render:** Show thinking in a collapsed block:
   ```
   ▸ 🧠 Thinking (142 tokens) — click to expand
   ```
   (Terminal can't really "click" — show dimmed, or behind a `/thinking` command)

3. **Voice mode:** Continue to strip thinking from TTS (already works).

**Simpler v1:** Just show thinking tokens dimmed inline during streaming,
then strip them from the final re-render. User sees the thinking happen
in real-time but the final clean response is thinking-free.

### Step 8: System Prompt Hint

When thinking is ON, append to the system prompt:

```
## Thinking Mode Active
You have extended thinking enabled. Use it for:
- Complex reasoning, debugging, architecture decisions
- Multi-step analysis where you need to work through the logic
- When the user explicitly asked you to "think harder"

Your thinking is visible to the user (dimmed). Be genuine in your reasoning.
```

When thinking is OFF (default), the existing narration rules apply —
the model should be fast and action-oriented.

### Step 9: Agent Self-Toggle (Future)

The agent could request thinking mode via a tool:
```json
{"name": "set_thinking", "arguments": {"enabled": true}}
```

Or simpler: detect phrases like "let me think about this more carefully"
and auto-enable for that turn.

**Defer this** — manual toggle is enough for v1.

## Provider-Specific Details

### Anthropic Extended Thinking
- Requires `interleaved-thinking-2025-05-14` beta flag ✅ (already present)
- Temperature must be exactly 1.0 when thinking is enabled
- `budget_tokens` controls how much the model can think (1K-100K)
- Stream emits `thinking` content blocks interleaved with `text` blocks
- Tool use works with thinking (interleaved thinking)

### ZhiPu
- `/nothink` suffix disables chain-of-thought (currently always appended)
- Remove suffix when thinking ON → model reasons in `reasoning_content` field
- Only useful for glm-4.6+ (reasoning models); glm-4.5-air doesn't reason regardless

### OpenAI / DeepSeek
- `reasoning_effort`: "low" | "medium" | "high" parameter
- `reasoning_content` in response delta
- Some models (o1, o3) always reason; others (gpt-4o) don't

### Local Models
- Most local models use `<thinking>...</thinking>` tags
- Already stripped by `strip_thinking_blocks()` in tui.rs
- When thinking ON: stop stripping, display dimmed instead
- Nanbeige4.1-3B: unclear if it has a thinking mode — test needed

## Status Bar Integration (Layer 2 Connection)

The thinking toggle naturally extends the status bar:

**Current status bar:**
```
ctx 12.4K/1M | msgs:8 | tools:3 (read_file, exec, edit_file) | wm:450tok | turn:5
```

**With thinking indicator:**
```
🧠 ctx 12.4K/1M | msgs:8 | tools:3 (read_file, exec, edit_file) | wm:450tok | turn:5
```

**Future Layer 2 input area:**
```
  ~/Dev/nanobot · 🧠 thinking · 3 tools running · 42w · 15s
  ⏵⏵ /think to toggle · ESC to cancel
```

## File Change Summary

| File | Changes |
|------|---------|
| `src/repl/commands.rs` | Add `thinking_on` to ReplContext, `/think` command, update `/status` and `/help` |
| `src/repl/mod.rs` | Update `build_prompt()` signature, pass thinking state to agent loop |
| `src/providers/anthropic.rs` | Add extended thinking params to request body, parse thinking stream blocks |
| `src/providers/openai_compat.rs` | Conditional `/nothink`, stop discarding `reasoning_content` when thinking ON |
| `src/agent/agent_loop.rs` | Read thinking state, pass to providers, optional system prompt hint |
| `src/config/schema.rs` | Optional: `thinking_budget_tokens` config field |
| `src/tui.rs` | Conditional `strip_thinking_blocks()`, dimmed thinking display, status bar update |
| `src/voice.rs` | No change — keep stripping thinking from TTS regardless |

## Implementation Order

1. **ReplContext + /think command** — get the toggle working (30 min)
2. **build_prompt update** — visual indicator (10 min)
3. **SharedCoreHandle thinking_on** — plumb the state (20 min)
4. **Anthropic provider** — extended thinking params + stream parsing (1-2 hours)
5. **Display layer** — dimmed thinking output during streaming (1 hour)
6. **OpenAI-compat** — conditional /nothink + reasoning_content display (30 min)
7. **System prompt hint** — append thinking instructions (10 min)
8. **Status bar update** — thinking indicator (10 min)
9. **Testing** — all providers, on/off states (1 hour)

**Total estimate:** ~4-5 hours

## Open Questions

1. **Budget tokens default?** 10K seems reasonable. Make configurable via `/think 20K`?
2. **Per-turn vs persistent?** Current plan: persistent toggle. Alternative: `/think` enables for next turn only, then auto-disables. Persistent feels more natural.
3. **Thinking in session logs?** Should thinking content be saved in session JSONL? Probably yes — useful for debugging and recall.
4. **Cost warning?** Thinking tokens cost money. Show a warning on first enable? "🧠 Thinking mode ON — this uses more tokens and may be slower."
