# Dynamic Model Router

## Vision

The model is a resource, not an identity. Nanobot uses whatever model is optimal for the exact task at the exact moment — balancing cost, speed, capability, and availability. The same architecture serves everyone from zero-budget laptops to GPU-rich power users.

## Principle

One agent, many models. The user talks to **nanobot**, not to Claude or Gemini or Qwen. The router decides which model handles each piece of work. Quality degrades gracefully, never breaks.

---

## Architecture

### Components

```
User Request
    │
    ▼
┌──────────────┐
│ Task Classifier │  ← "what kind of work is this?"
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Router     │  ← "cheapest model that can handle it"
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Model Registry│  ← providers, models, costs, capabilities, limits
└──────┬───────┘
       │
       ├──→ Cloud (Opus, Sonnet, Haiku, Gemini, OpenRouter free)
       ├──→ Local (llama.cpp, vLLM, ExLlamaV2)
       └──→ Hybrid (cloud orchestrator + local workers)
```

### 1. Model Registry

Each available model is tagged with metadata:

```rust
struct ModelEntry {
    provider: String,        // "anthropic", "openrouter", "local"
    model: String,           // "claude-opus-4-6", "nanbeige-3b"
    tier: CapabilityTier,    // Reasoning, Generation, Routine, Simple
    cost_per_mtok_in: f64,   // 0.0 for free/local
    cost_per_mtok_out: f64,
    speed: Speed,            // Fast, Medium, Slow
    is_local: bool,
    rate_limit: Option<RateLimit>,
    context_window: usize,
    available: bool,         // runtime: is this model reachable right now?
}

enum CapabilityTier {
    Reasoning,   // Architecture, debugging, novel problems (Opus, DeepSeek R1)
    Generation,  // Code gen, research, multi-step tasks (Sonnet, Gemini Flash)
    Routine,     // Boilerplate, tests, docs, summaries (Haiku, Qwen 8B)
    Simple,      // File ops, formatting, grep-and-report (Nanbeige 3B, tool delegation)
}
```

### 2. Task Classifier

Determines what capability tier a task needs. Starts rule-based, evolves to learned.

**Rule-based v1:**
- Tool delegation (read_file, exec, list_dir) → `Simple`
- Subagent tasks with clear instructions → `Routine` or `Generation`
- User conversation (main loop) → `Generation` (default) or `Reasoning` (if escalated)
- Explicit user escalation ("think harder", complex debugging) → `Reasoning`

**Learned v2 (future):**
- Confidence estimation from local model
- If confidence > 0.85 → handle locally
- If confidence > 0.6 → handle locally + cloud verification
- If confidence < 0.6 → escalate to cloud

### 3. Router

Given a task's required tier, picks the cheapest available model that meets it.

```
route(task) -> (provider, model):
    tier = classify(task)
    candidates = registry.filter(tier >= required, available == true)
    sort by: cost ASC, speed ASC
    if budget_remaining(tier) > 0:
        return candidates[0]
    else:
        return fallback_chain.next()
```

**Fallback chain** (always defined, never fails):
```
Opus → Sonnet → Haiku → Gemini Flash (free) → OpenRouter free → Local 8B → Local 3B → Queue for later
```

### 4. Budget Controller

```rust
struct BudgetController {
    daily_limit_usd: f64,
    monthly_limit_usd: f64,
    spent_today: f64,
    spent_this_month: f64,
    per_tier_limits: HashMap<CapabilityTier, f64>,  // optional
}
```

- Tracks spend per tier per day/month
- When a tier's budget is exhausted, transparently falls back to next tier
- User-visible: `nanobot budget` shows current spend and remaining
- Alert at 80% of budget

---

## Deployment Profiles

### Profile: Zero Budget (no GPU, no API key)

| Role | Model | Cost |
|------|-------|------|
| Orchestrator | Gemini 2.0 Flash (free) | $0 |
| Subagents | Llama 4 Maverick / OpenRouter free | $0 |
| Tool delegation | Qwen3-8B / OpenRouter free | $0 |
| **Total** | | **$0/month** |

Constraint: rate limits only. Queue + backoff handles this.

### Profile: Hybrid Hacker (GPU + small budget)

| Role | Model | Cost |
|------|-------|------|
| Reasoning | Claude Opus (cloud) | ~$5-10/month |
| Main (local) | Qwen3-30B-A3B-Instruct-2507 Q4 (~17GB) | electricity |
| Subagents | Sonnet or Gemini Flash free | ~$2/month or $0 |
| RLM / Tool delegation | Nanbeige4.1-3B Q8 (~3.5GB) | electricity |
| Memory | Qwen3-0.6B Q4 (~0.4GB) | electricity |
| Voice | Local Whisper + Kokoro | electricity |
| **Total** | | **~$5-12/month** |

80% of tasks stay local. Cloud only for genuine reasoning.
See `plans/local-model-matrix.md` for hardware tier details.

### Profile: Team (5 devs, shared instance)

| Role | Model | Cost |
|------|-------|------|
| Architecture / review | Opus | per-user budget |
| Daily coding | Sonnet | shared pool |
| Boilerplate / docs | Haiku or local | minimal |
| Tool delegation | Local or Haiku | minimal |
| **Total** | | **~$30-50/month** |

Budget controller allocates per user. Escalation on request.

### Profile: Offline

| Role | Model | Cost |
|------|-------|------|
| Main | Qwen3-30B-A3B-Instruct-2507 Q4 (if 24GB+) or Nanbeige4.1-3B (if 8GB+) | electricity |
| RLM | Nanbeige4.1-3B Q8 (or shared with main on 8GB) | electricity |
| Memory | Qwen3-0.6B Q4 (or shared with main on 8GB) | electricity |
| **Total** | | **$0** |

Seamless fallback when connectivity drops. Sessions sync on reconnect. Optional post-hoc cloud review.
See `plans/local-model-matrix.md` for full hardware tier matrix.

### Profile: Kernel (future)

| Role | Model | Cost |
|------|-------|------|
| 95% of tasks | Personal LoRA-tuned local model | electricity |
| Novel problems | Cloud Opus (on demand) | minimal |
| Specialists | Task-specific LoRA adapters | electricity |
| **Total** | | **~$1-2/month** |

The model learns when to ask for help. Routing is learned, not hardcoded.

---

## Implementation Plan

### Phase 1: Subagent Model Config (now)

Minimal change. Add to config:

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

Wire into `SubagentManager::new()`. ~30 lines changed.

### Phase 2: Free Tier Providers

- Wire up OpenRouter free tier (already in config, needs API key + model list)
- Wire up Gemini free tier
- Test: full agent loop on $0 budget

### Phase 3: Model Registry + Simple Router

- Define `ModelEntry` struct with cost/tier/availability
- Auto-detect local models (poll llama.cpp `/v1/models`)
- Auto-detect cloud availability (API key present + valid)
- Simple rule-based routing: task type → tier → cheapest available model

### Phase 4: Budget Controller

- Track token usage per model per day
- Config: daily/monthly limits
- CLI: `nanobot budget` dashboard
- Automatic fallback when budget exhausted

### Phase 5: Learned Routing (with Zero/Kernel)

- Local model confidence estimation
- Route based on confidence threshold
- LoRA specialists for common task types
- The router itself becomes a learned component

---

## What Already Exists

The codebase is 80% ready:

- ✅ Multiple providers configured (`config.json` has anthropic, openai, openrouter, groq, gemini, vllm, deepseek, zhipu)
- ✅ Tool delegation with separate model (`tool_runner_provider`, `tool_runner_model`)
- ✅ Subagent spawning (`SubagentManager`)
- ✅ Local/cloud mode (`is_local` flag)
- ✅ Provider abstraction (`LLMProvider` trait)

What's missing:
- ❌ Subagent model config (Phase 1)
- ❌ Model registry with metadata
- ❌ Router logic
- ❌ Budget tracking
- ❌ Free tier provider setup

---

## Design Principles

1. **Never break.** Fallback chain always terminates. Worst case = queued for later.
2. **Transparent.** User can see which model handled what. `nanobot status` shows routing decisions.
3. **Override-able.** User can force a model: `@opus think about this` or config per-channel.
4. **Zero config works.** Defaults are sensible. Free tier users get a working system without touching config.
5. **Cost is a first-class metric.** Every model call is tracked. Budget is always visible.
6. **Local first.** If a local model can handle it, prefer local. Faster, cheaper, private.
