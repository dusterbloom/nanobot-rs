# Multi-Cloud/Local/Hybrid Provider Refactor

**Date:** 2026-02-14
**Status:** Planning
**Goal:** Make nanobot work seamlessly with multiple providers, free tiers, local models, and hybrid configurations — with zero-config onboarding.

---

## Current State (Problems)

### Architecture: 35,036 lines Rust

```
src/
├── agent/
│   ├── agent_loop.rs    (2025)  # Main loop, SwappableCore, AgentHandle
│   ├── agent_profiles.rs (316)  # Agent profiles (YAML frontmatter .md files)
│   ├── tool_runner.rs   (2542)  # RLM agent loop (autonomous tool execution)
│   ├── subagent.rs       (771)  # Spawn background agents
│   ├── worker_tools.rs   (804)  # Recursive delegation
│   ├── context.rs        (818)  # System prompt / context builder
│   ├── compaction.rs     (723)  # Context compaction
│   ├── provenance.rs    (1005)  # Audit trail
│   ├── skills.rs         (696)  # Persistent skills
│   ├── token_budget.rs   (602)  # Token counting
│   ├── working_memory.rs (541)  # Session working memory
│   ├── learning.rs       (523)  # Learning store
│   ├── tools/
│   │   ├── shell.rs     (1213)
│   │   ├── web.rs        (813)
│   │   ├── filesystem.rs (629)
│   │   ├── registry.rs   (522)
│   │   ├── spawn.rs      (410)
│   │   └── ...
│   └── ...
├── providers/
│   ├── base.rs                  # LLMProvider trait
│   └── openai_compat.rs  (790)  # Single provider impl
├── config/
│   ├── schema.rs        (1089)  # All config structs
│   └── loader.rs                # Config loading
├── channels/                    # telegram, email, whatsapp
├── cli.rs               (1186)  # CLI entry + create_provider()
├── server.rs            (1540)  # HTTP gateway
├── repl/                        # Interactive REPL
└── voice.rs             (1053)  # Voice pipeline
```

### Provider Architecture (Single Provider, Single Model)

1. **`create_provider()`** in `cli.rs` — picks FIRST API key found (OpenRouter > DeepSeek > Anthropic > OpenAI > Gemini > Zhipu > Groq > vLLM), creates ONE `OpenAICompatProvider`
2. **`ProvidersConfig`** — hardcoded struct with named fields (`anthropic`, `openai`, `openrouter`, etc.) — not extensible without code changes
3. **`get_api_key()` / `get_api_base()`** — waterfall priority, returns FIRST match — cannot use multiple providers simultaneously
4. **Model routing** — model string sent to whatever single provider was created, no model→provider mapping
5. **Subsystem providers** — compaction, RLM, memory each have their own `Option<ProviderConfig>`, bolted on separately

### Anti-Patterns

| Issue | Where | Description |
|-------|-------|-------------|
| God function | `build_swappable_core()` | 17 parameters |
| Hardcoded providers | `ProvidersConfig` | Named fields, not a map |
| Waterfall priority | `get_api_key()`/`get_api_base()` | First-match-wins, can't use multiple |
| No model→provider routing | `create_provider()` | Model string disconnected from provider |
| Provider config duplication | `ToolDelegationConfig`, `MemoryConfig` | Each subsystem has `Option<ProviderConfig>` |
| God struct | `SwappableCore` | 30+ fields mixing provider, memory, tools, config |
| No free tier awareness | nowhere | No concept of free models, rate limits, fallback |

### SOLID Violations

- **SRP**: `SwappableCore` does everything. `cli.rs` mixes provider creation with CLI parsing. `agent_loop.rs` mixes config building, provider resolution, and agent execution.
- **OCP**: `ProvidersConfig` is closed — adding a new provider requires struct change.
- **DIP**: `create_provider()` directly constructs `OpenAICompatProvider`. Agent loop knows provider internals.
- **DRY**: "if has provider config, use it; else fallback to main" pattern repeated 3+ times for RLM, compaction, memory.

---

## Target State

### Vision
- OpenRouter free tier as seamless onboarding (no API key needed for basic use)
- Nanbeige4.1-3B as default local model
- Hybrid: expensive model for reasoning, capable model for RLM, cheap/free for memory
- Zero-config: detect local llama-server, use free OpenRouter, or both
- Accessible to people without GPUs or API budgets

### The 3-Model Architecture

Nanobot uses **3 distinct model roles**, each with fundamentally different requirements:

| Role | Name | What it actually does | Requirements |
|------|------|----------------------|-------------|
| **Main** | Reasoning Model | Primary conversation, planning, decision-making | Smartest available, large context |
| **RLM** | Runner Language Model | Full autonomous agent loop: reasons about tasks, plans multi-step tool chains, executes tools, handles errors, iterates, and delivers structured results back to Main | Capable of tool use, reasoning, and multi-turn execution. This is a real agent, not a summarizer. |
| **Memory** | Memory Model | Compaction (summarize old context), Reflection (distill observations → MEMORY.md facts) | Good at summarization, can be small/cheap |

**Key distinction:** RLM is NOT a summarizer. It runs its own agent loop with full tool access — it reads files, executes commands, searches the web, and chains multiple operations together. It's the workhorse that does the actual work while Main focuses on reasoning and conversation. Memory model is the lightweight one — it just needs to compress text and extract facts.

**In local mode (2 models, 3 roles):**
```
Big model (GPU)   ──▶  Main (reasoning + conversation)
Small model (CPU) ──▶  RLM (autonomous tool execution agent)
                  ──▶  Memory (compaction + reflection)
```

RLM and Memory can share the same small model, but RLM could benefit from a more capable one. In cloud/hybrid mode, all 3 can be different:

```
Claude Opus       ──▶  Main
Claude Haiku      ──▶  RLM (cheap but capable tool-use agent)
Gemma 3 :free     ──▶  Memory (just summarization, free tier is fine)
```

### Config Example (Target)

```toml
# Providers are a map — extensible without code changes
[providers.openrouter]
api_key = "sk-or-..."

[providers.local]
api_base = "http://localhost:8080/v1"

[providers.anthropic]
api_key = "sk-ant-..."

# Model aliases — roles map to specific models
[models]
main = "anthropic/claude-opus-4-20250514"            # Smartest: reasoning + conversation
rlm = "anthropic/claude-haiku"                       # Capable: autonomous tool execution
memory = "openrouter/google/gemma-3-27b-it:free"     # Cheap: summarization + fact extraction
local_main = "local/qwen3-30b"                       # Local big model
local_worker = "local/nanbeige4.1-3b"                # Local small model

# Role assignments — which model serves each purpose
[roles]
reasoning = "main"          # Primary conversation
rlm = "rlm"                # Runner Language Model (tool execution agent)
compaction = "memory"       # Context compression
reflection = "memory"       # Observation → facts distillation

# Fallback chains — what to try when primary fails
[fallback]
main = ["main", "rlm", "local_main"]
rlm = ["rlm", "memory", "local_worker"]
memory = ["memory", "local_worker", "rlm"]
```

---

## Implementation Plan

### Phase 1: Provider Registry & Model Router (Foundation)

**Goal:** Replace hardcoded provider fields with extensible registry. Route model strings to correct provider.

**Files to change:**
- `src/config/schema.rs` — Replace `ProvidersConfig` struct with `HashMap<String, ProviderEntry>`
- `src/providers/mod.rs` — Add `ProviderRegistry` and `ModelRouter`
- `src/providers/base.rs` — Add provider metadata (name, supports_tools, supports_streaming, etc.)
- `src/cli.rs` — Replace `create_provider()` with registry construction

**New structs:**

```rust
/// A configured provider endpoint.
pub struct ProviderEntry {
    pub api_key: String,
    pub api_base: Option<String>,
    /// Provider type hint (openai, anthropic, local, etc.)
    /// Most are openai-compatible; this allows future native impls.
    pub kind: ProviderKind,
    /// Optional: models this provider serves (for auto-routing)
    pub models: Vec<String>,
}

/// Registry of all configured providers.
pub struct ProviderRegistry {
    providers: HashMap<String, Arc<dyn LLMProvider>>,
    model_aliases: HashMap<String, String>,  // "fast" -> "openrouter/gemma-3:free"
    role_assignments: HashMap<String, String>, // "rlm" -> "haiku"
    fallback_chains: HashMap<String, Vec<String>>,
}

impl ProviderRegistry {
    /// Resolve a model string to (provider, model_name).
    /// Handles: aliases, role names, provider/model syntax, fallbacks.
    pub fn resolve(&self, model_or_role: &str) -> Result<(Arc<dyn LLMProvider>, String)>;

    /// Get provider for a specific role with fallback.
    pub fn for_role(&self, role: &str) -> Result<(Arc<dyn LLMProvider>, String)>;
}
```

**Model resolution order:**
1. Check if it's a role name → resolve to model alias
2. Check if it's a model alias → resolve to full model string
3. Parse `provider/model` syntax → look up provider by name
4. Bare model name → try to match against provider model lists
5. Fallback chain if primary fails

**Migration path:**
- Old `ProvidersConfig` still loads but converts to `HashMap<String, ProviderEntry>` internally
- `get_api_key()` / `get_api_base()` become `registry.default_provider()`
- Existing configs keep working, new format adds capabilities

### Phase 2: Refactor SwappableCore (SOLID Cleanup)

**Goal:** Break the god struct into composed parts. Eliminate the 17-param function.

**Extract from SwappableCore:**

```rust
/// All provider references for different roles.
pub struct ProviderSet {
    pub registry: Arc<ProviderRegistry>,
    // Cached resolved providers for hot path:
    pub main: (Arc<dyn LLMProvider>, String),
    pub rlm: Option<(Arc<dyn LLMProvider>, String)>,
    pub compaction: (Arc<dyn LLMProvider>, String),
    pub memory: (Arc<dyn LLMProvider>, String),
}

/// Agent behavior configuration (no provider stuff).
pub struct AgentConfig {
    pub max_iterations: u32,
    pub max_tokens: u32,
    pub temperature: f64,
    pub max_context_tokens: usize,
    pub exec_timeout: u64,
    pub restrict_to_workspace: bool,
    pub max_tool_result_chars: usize,
    pub is_local: bool,
}

/// Memory subsystem configuration + state.
pub struct MemorySubsystem {
    pub compactor: ContextCompactor,
    pub learning: LearningStore,
    pub working_memory: WorkingMemoryStore,
    pub enabled: bool,
    pub reflection_threshold: usize,
    pub working_memory_budget: usize,
    pub session_complete_after_secs: u64,
    pub max_message_age_turns: usize,
    pub max_history_turns: usize,
}

/// Simplified SwappableCore — thin container.
pub struct SwappableCore {
    pub providers: ProviderSet,
    pub agent_config: AgentConfig,
    pub memory: MemorySubsystem,
    pub context: ContextBuilder,
    pub sessions: Arc<SessionManager>,
    pub token_budget: TokenBudget,
    pub tool_delegation_config: ToolDelegationConfig,
    pub provenance_config: ProvenanceConfig,
    pub workspace: PathBuf,
}
```

**Builder pattern:**

```rust
impl SwappableCore {
    pub fn builder(registry: Arc<ProviderRegistry>, workspace: PathBuf) -> SwappableCoreBuilder {
        SwappableCoreBuilder::new(registry, workspace)
    }
}
```

**Files to change:**
- `src/agent/agent_loop.rs` — Extract structs, replace `build_swappable_core()` with builder
- `src/agent/tool_runner.rs` — Accept `ProviderSet` instead of individual provider refs
- `src/agent/compaction.rs` — Use `ProviderSet` 
- `src/cli.rs` — Use builder pattern

### Phase 3: Fallback Chains & Hybrid Mode

**Goal:** Automatic fallback, free tier support, zero-config onboarding.

**Features:**
- **Circuit breaker per provider** — already have `circuit_breaker.rs`, wire it into `ProviderRegistry`
- **Rate limit tracking** — track 429s per provider, back off automatically
- **Free tier detection** — OpenRouter `:free` suffix models, auto-discover available free models
- **Auto-detect local** — probe `localhost:8080/v1/models` on startup
- **Zero-config mode:**
  1. No config file? → Check for local llama-server → Check for `OPENROUTER_API_KEY` env var → Use free tier
  2. First run wizard: "No API key found. Options: (1) Enter OpenRouter key (free tier available) (2) Start local model (3) Skip for now"

**Config additions:**

```toml
[providers.openrouter]
api_key = "sk-or-..."
# Rate limit tracking (auto-populated)
rate_limit_rpm = 20        # requests per minute for free tier
rate_limit_remaining = 15  # tracked at runtime

[fallback]
# When main model fails or is rate-limited
main = ["main", "fast", "local"]
# Delegation can use cheapest available
rlm = ["rlm", "local_worker", "main"]
```

**Files to change/add:**
- `src/providers/mod.rs` — Add fallback logic to `ProviderRegistry::resolve()`
- `src/providers/rate_limit.rs` — NEW: Rate limit tracker (parse `x-ratelimit-*` headers)
- `src/providers/discovery.rs` — NEW: Auto-detect local servers, probe endpoints
- `src/agent/circuit_breaker.rs` — Wire into registry (currently standalone)

---

## Dependency Order

```
Phase 1 (Provider Registry)
    └── Phase 2 (SwappableCore refactor) 
         └── Phase 3 (Fallback + Hybrid)
```

Phase 1 is the foundation — everything else builds on having a proper provider registry.

## Risk Mitigation

- **Backward compatibility**: Old config format auto-converts to new internally
- **Incremental migration**: Each phase is independently shippable
- **Test coverage**: Provider registry gets unit tests for resolution logic; integration tests for fallback chains
- **No breaking changes**: Existing single-provider setups keep working unchanged

## Success Criteria

1. `nanobot` starts with zero config and finds something to use (local or free tier)
2. Config can specify multiple providers and route models to them
3. Delegation uses cheap/free model while main uses expensive model — by default
4. Provider failure triggers automatic fallback with user notification
5. `SwappableCore` has ≤10 direct fields (composed structs don't count)
6. `build_swappable_core()` replaced with builder taking ≤3 params
7. No duplicated provider resolution logic
