---
date: "2026-02-14T16:00:00Z"
session_name: "model-router"
researcher: claude-opus
branch: main
repository: nanobot
topic: "Dynamic Model Router — One Agent, Many Models"
tags: [planning, model-router, budget, free-tier, routing, subagent]
status: ready-to-implement
last_updated: "2026-02-14"
last_updated_by: claude-opus
type: implementation_handoff
---

# Handoff: Dynamic Model Router

## Vision

The model is a resource, not an identity. Nanobot uses whatever model is optimal for the exact task — balancing cost, speed, capability, and availability. Quality degrades gracefully, never breaks.

## Plan Source

`~/.nanobot/workspace/plans/dynamic-model-router.md` (full plan with diagrams at `dynamic-model-router-diagram.txt`)

## Current State: 30% Ready

The architecture already supports separate models for different roles. What's missing is the config wiring, registry, router, and budget tracking.

## What EXISTS

| Component | Location | Status |
|-----------|----------|--------|
| SubagentManager accepts separate model | `src/agent/subagent.rs:63-77` | Architecture ready |
| Tool delegation with separate model | `ToolRunnerConfig.provider/model` in `tool_runner.rs` | Fully wired |
| 8 providers configured | `src/config/schema.rs:272-291` | anthropic, openai, openrouter, deepseek, groq, zhipu, vllm, gemini |
| Provider priority chain | `Config::get_api_key()` at `schema.rs:690-707` | OpenRouter > DeepSeek > Anthropic > OpenAI > Gemini > Zhipu > Groq > vLLM |
| Gemini free tier URL | `schema.rs:733` | `https://generativelanguage.googleapis.com/v1beta/openai` |
| OpenRouter URL | `schema.rs:720` | `https://openrouter.ai/api/v1` |
| LLMResponse.usage field | `providers/base.rs:23` | `HashMap<String, i64>` — captured but NEVER read |
| LearningStore | `src/agent/learning.rs` | Can log tool outcomes, extensible for route decisions |
| `/local` and `/model` hot-swap | `SwappableCore` + `AgentHandle` in `agent_loop.rs` | Core swap without losing counters |
| is_local flag | `SwappableCore.is_local` | Distinguishes local vs cloud mode |

## What's MISSING (by phase)

### Phase 1: Subagent Model Config (~30 lines) — START HERE

**Goal**: Different models for main chat, subagents, and delegation.

**Changes needed**:
1. `src/config/schema.rs` — Add fields to `AgentDefaults`:
   ```rust
   pub subagent_model: Option<String>,    // falls back to model
   pub delegation_model: Option<String>,  // falls back to model
   ```
2. `src/agent/agent_loop.rs` — Wire `subagent_model` into `SubagentManager::new()`
3. `src/cli.rs` — Wire `delegation_model` into `build_core_handle` / `rebuild_core`

**Config JSON**:
```json
{
  "agents": {
    "defaults": {
      "model": "claude-opus-4-6",
      "subagentModel": "claude-sonnet-4-20250514",
      "delegationModel": "claude-haiku-3-20250414"
    }
  }
}
```

**Why first**: Quick win, immediately useful, unblocks Phase 3 testing.

### Phase 2: Free Tier Providers (~50 lines)

**Goal**: $0/month operation using free-tier APIs.

**What works already**: Gemini and OpenRouter URLs are configured. Just needs:
1. Test with Gemini free tier API key + `gemini-2.0-flash` model name
2. Test with OpenRouter free tier (rate-limited free models)
3. Document which free models work: `google/gemini-2.0-flash-exp`, `meta-llama/llama-4-maverick:free`
4. Add default rate limit values for free providers

### Phase 3: Model Registry + Router (~400 lines)

**Goal**: Task-aware model selection.

**New files**:
- `src/agent/model_registry.rs` — `ModelEntry`, `CapabilityTier`, `ModelRegistry`
- Task classifier can live in same file (initially rule-based)

**Key structs**:
```rust
enum CapabilityTier { Reasoning, Generation, Routine, Simple }

struct ModelEntry {
    provider: String,
    model: String,
    tier: CapabilityTier,
    cost_per_mtok_in: f64,
    cost_per_mtok_out: f64,
    speed: Speed,
    is_local: bool,
    rate_limit: Option<RateLimit>,
    context_window: usize,
    available: bool,
}
```

**Router logic**: `route(tier) -> ModelEntry` picks cheapest available model for the required tier.

**Fallback chain** (hardcoded, always terminates):
```
Opus → Sonnet → Haiku → Gemini Flash (free) → OpenRouter free → Local 8B → Local 3B
```

**Integration point**: Replace `config.agents.defaults.model` in `agent_loop.rs::process_message()` with `router.route(classify(task))`.

### Phase 4: Budget Controller (~400 lines)

**Goal**: Track spend, enforce limits, auto-fallback.

**New file**: `src/agent/budget.rs`

**Key design**:
- Extract `usage` from every `LLMResponse` (currently discarded)
- `BudgetController` tracks daily/monthly spend per model
- Persistence: `~/.nanobot/budget.json` (daily reset, monthly rollover)
- On budget exhaustion: transparent fallback to next tier
- CLI: `nanobot budget` shows spend breakdown

**Hardest part**: Concurrent access — budget must be checked before every LLM call and updated after. Use `Arc<Mutex<BudgetController>>` in `SwappableCore`.

### Phase 5: Learned Routing (future, ~300 lines)

**Goal**: Router learns from outcomes.

- Log route decisions + task outcomes to LearningStore
- Local model confidence estimation
- Threshold routing: high confidence → local, low → cloud
- LoRA specialist loading

**Defer until**: Phases 1-4 are stable and battle-tested.

## Deployment Profiles (from plan)

| Profile | Orchestrator | Workers | Delegation | Cost |
|---------|-------------|---------|------------|------|
| Zero Budget | Gemini Flash (free) | OpenRouter free | OpenRouter free | $0 |
| Hybrid Hacker | Claude Opus | Sonnet / local | Local 3B | ~$5-12/mo |
| Team (5 devs) | Opus | Sonnet | Haiku / local | ~$30-50/mo |
| Offline | Best local | Local | Local | $0 |

## Implementation Order

```
Phase 1 (30 LOC, 1 session)
  ↓
Phase 2 (50 LOC, 1 session — can be done in parallel with 1)
  ↓
Phase 3 (400 LOC, 2-3 sessions)
  ↓
Phase 4 (400 LOC, 2-3 sessions — can be done in parallel with 3)
  ↓
Phase 5 (future)
```

**Total estimated**: ~880 lines of Rust across 4-6 sessions.

## Key Files to Touch

| Phase | New Files | Modified Files |
|-------|-----------|----------------|
| 1 | — | `schema.rs`, `agent_loop.rs`, `cli.rs` |
| 2 | — | Integration tests, docs |
| 3 | `model_registry.rs` | `agent_loop.rs`, `mod.rs` |
| 4 | `budget.rs` | `agent_loop.rs`, `tool_runner.rs`, `cli.rs`, `repl/commands.rs` |

## Design Principles (from plan)

1. **Never break.** Fallback chain always terminates.
2. **Transparent.** User sees which model handled what.
3. **Override-able.** `@opus think about this` forces a model.
4. **Zero config works.** Free tier users get a working system.
5. **Cost is first-class.** Every call tracked, budget always visible.
6. **Local first.** If a local model can handle it, prefer local.

## Open Questions

- UNCONFIRMED: Does Gemini free tier support function calling via the OpenAI-compat endpoint?
- UNCONFIRMED: What's the current OpenRouter free tier rate limit? (was 20 req/min in 2025)
- Should the classifier be a separate LLM call or pure heuristic? (plan says rule-based v1)
- Where should model metadata live — hardcoded registry, config JSON, or auto-detected?

## Also Completed This Session

Before this handoff, the following was done:
1. **Post-session qmd indexing**: `index_sessions_background()` added to `src/repl/mod.rs` — runs `qmd update` on REPL exit and single-message mode exit
2. **Borrow fix**: Fixed double mutable borrow of `ctx` in voice recording path (`repl/mod.rs:800`)
3. **Removed unused import**: `BufRead` from `repl/mod.rs`
4. **Normalized dedup**: Fixed second `seen_calls` site in `tool_runner.rs:351` to use `normalize_call_key()`
5. **Plan audit**: Verified RLM completion (plan 2) and tool runner improvements (plan 3) are fully implemented — only the dedup normalization was missing

## Uncommitted Changes Summary

All on `main` branch, not yet committed:
- `src/agent/agent_loop.rs` — SharedCore → SwappableCore + RuntimeCounters split
- `src/agent/subagent.rs` — ExecTool max_output_chars param
- `src/agent/tools/shell.rs` — Configurable max_output_chars
- `src/cli.rs` — AgentHandle, swap_core(), counter persistence
- `src/tui.rs` — Read counters directly from RuntimeCounters
- `src/repl.rs` → `src/repl/` — Module split + qmd indexing + borrow fix
- `src/agent/tool_runner.rs` — normalize_call_key at second site
- `Cargo.lock` — Dependency updates
