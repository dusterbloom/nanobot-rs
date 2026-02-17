# Nanobot Backlog

## Commit 2 Review Action Items (2026-02-17)

All resolved:

1. ~~**Fix stale comment in `ensure_compaction_model`**~~ — Done. Updated to Qwen3-1.7B.
2. ~~**Raise tool result truncation threshold**~~ — Done. 1200 char threshold, first 600 + last 400.
3. ~~**Document multi-session CONTEXT.md race**~~ — Done. Implemented per-channel files (`CONTEXT-cli.md`, `CONTEXT-telegram.md`). Writer extracts channel from session key, reader falls back to legacy `CONTEXT.md`.

## UX Bugs

All resolved:

4. ~~**Input box and status bar disappear during streaming**~~ — Done. Replaced `\x1b[J` (Erase in Display) with line-by-line `\x1b[2K` at 3 locations in `repl/mod.rs`.
5. ~~**Agent interruption is too slow**~~ — Done. Wrapped streaming loop in `tokio::select!` racing cancellation token in `agent_loop.rs`.

## Broken Windows (tech debt)

6. **132 compiler warnings** — Accumulated `unused_imports`, `dead_code`, `unused_variables` across the codebase. Makes real errors invisible. One `cargo fix` pass would clear most of them.
7. **2 pre-existing test failures** — `test_web_search_no_api_key` fails when BRAVE_API_KEY is set (test assumes it isn't). `test_normalize_alias_all_aliases` has stale alias mapping (`/prov` vs `/provenance`). Both should be quick fixes.

## Ideas from Spacebot (2026-02-17)

Source: https://github.com/spacedriveapp/spacebot — Rust agentic system by Spacedrive. FSL-licensed, don't copy code, only ideas.

### Priority 1: Non-blocking compaction
Spacebot's compactor is a programmatic monitor that spawns a compaction worker via `tokio::spawn`. The channel keeps talking while compaction runs in the background. Three tiers: background (80%), aggressive (85%), emergency truncation (95%). Currently nanobot's compaction blocks the agent loop. Spawn it as a background task and swap the result in when done.

### Priority 2: Status injection
Every turn, Spacebot injects a live status block into the channel's context: active workers, recently completed work, branch states. Workers set their own status via `set_status` tool. Short branches are invisible (only appear if running >3s). Nanobot's spawn list/check is the manual version — auto-injecting a status block would make the agent naturally aware of background work without explicit tool calls.

### Priority 3: Message coalescing
When messages arrive rapidly in a channel, Spacebot batches them into a single LLM turn with timing context. The LLM "reads the room" and picks the most interesting thing to engage with, or stays quiet. Configurable debounce timing, automatic DM bypass. Needed for nanobot's Telegram/Discord channels where multiple messages can arrive before the agent responds.

### Priority 4: Branch concept (context-fork subagents)
Branches are forks of the channel's context that go off to think independently. They inherit full conversation history (just `channel_history.clone()`), operate independently, and return a conclusion that gets injected back. Cheaper than a subagent because they already have context. Multiple branches can run concurrently per channel. Nanobot's subagents are closer to Spacebot's "workers" — stateless, no context. Adding branches would fill the gap.

### Priority 5: Prompt complexity routing
Four-level model routing: process-type defaults, task-type overrides, prompt complexity scoring (light/standard/heavy via keyword scorer, <1ms, no external calls), and fallback chains with rate-limit tracking. Nanobot's model hierarchy is manual (AGENTS.md instructions). Auto-downgrading simple messages to cheaper models would save cost.

### Priority 6: Memory bulletin (Cortex)
A periodic LLM job (default 60 min) queries memory across multiple dimensions and synthesizes a ~500 word briefing. Cached in `ArcSwap` so every channel reads it on every turn at zero cost. Nanobot has MEMORY.md which is similar but static/manual. A periodic auto-generated briefing would keep it current.

### Not worth stealing
- Rig framework dependency (locked into rig v0.30.0; nanobot owns its agent loop)
- Browser tool (web_fetch covers most cases)
- OpenCode integration (specific to their coding use case)
- Typed memory graph (interesting but heavy lift; flat markdown works for now)

---

## Context Gate: Intelligent Content Management

**Design doc:** [plans/context-gate.md](plans/context-gate.md)

Replace dumb char-limit truncation with a `ContentGate` that makes context-aware decisions:
- **Pass raw** when content fits the model's context budget
- **Structural briefing** (via compactor) when it doesn't, with full content cached to disk
- **Drill-down** via `read_file` with line ranges to navigate cached content

Components: `ContextBudget` (token accounting), `OutputCache` (disk cache), `ContentGate` (decision gate), extended `Compactor` (briefing mode).

Key property: **zero agent-facing API changes**. Same tools, same interface, every model from 3B to 200K context. Infrastructure is invisible.
