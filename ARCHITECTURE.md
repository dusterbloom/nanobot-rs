# nanobot Architecture Map

> Complete reference for LLM onboarding. Last verified: 2026-02-21

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLI (main.rs)                                   │
│  Commands: onboard, agent, gateway, telegram, whatsapp, email, cron, eval   │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          │                           │                           │
          ▼                           ▼                           ▼
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│   REPL Mode     │         │  Gateway Mode   │         │   Eval Mode     │
│  (interactive)  │         │  (multi-chat)   │         │  (benchmarks)   │
└────────┬────────┘         └────────┬────────┘         └─────────────────┘
         │                           │
         │    ┌──────────────────────┴──────────────────────┐
         │    │                                             │
         ▼    ▼                                             ▼
┌─────────────────────────────────────┐    ┌─────────────────────────────────┐
│           AgentLoop                 │    │        ChannelManager           │
│  (core message processing)          │    │  (Telegram, WhatsApp, Email,    │
│                                     │    │   Feishu adapters)              │
└─────────────────────────────────────┘    └─────────────────────────────────┘
         │                                             │
         │              InboundMessage                │
         │◄────────────────────────────────────────────┘
         │
         │              OutboundMessage
         └────────────────────────────────────────────►
```

## Core Components

### 1. Entry Points (`src/main.rs`, `src/cli.rs`, `src/sessions_cmd.rs`)

**main.rs** - CLI parsing and command routing:
- `Commands` enum defines all CLI commands
- Routes to handler functions in `cli.rs` and `sessions_cmd.rs`
- Panic hook restores terminal state
- Global `LOCAL_MODE` atomic flag for cloud/local switching

**cli.rs** - Agent/config command implementations:
- `cmd_onboard()` - Initialize config and workspace
- `cmd_gateway()` - Start multi-channel gateway
- `build_core_handle()` - Create swappable core with provider
- `rebuild_core()` - Hot-swap provider on `/local` toggle
- `create_agent_loop()` - Wire agent loop with channels

**sessions_cmd.rs** - Session management CLI:
- `cmd_sessions_list()` - List sessions with date, size, message count
- `cmd_sessions_export()` - Export session as markdown or JSONL
- `cmd_sessions_nuke()` - Wipe all sessions, logs, and metrics
- `cmd_sessions_purge()` - Remove files older than a duration
- `cmd_sessions_archive()` - Show disk usage summary
- `make_session_key()` - Generate session key from optional name
- Also available in REPL via `/sessions` (`/ss` alias) with list/export/purge/archive/index subcommands

### 2. Agent Loop (`src/agent/agent_loop.rs`)

**The heart of message processing.** Key concepts:

```
AgentHandle (cloneable)
├── core: Arc<RwLock<Arc<SwappableCore>>>  -- Hot-swappable on /local
└── counters: Arc<RuntimeCounters>          -- Persists across swaps
```

**SwappableCore** contains (38 fields total, key ones shown):
- `provider: Arc<dyn LLMProvider>` - Main LLM
- `memory_provider: Arc<dyn LLMProvider>` - For compaction/reflection
- `router_provider: Option<Arc<dyn LLMProvider>>` - For trio routing
- `specialist_provider: Option<Arc<dyn LLMProvider>>` - For trio execution
- `tool_runner_provider: Option<Arc<dyn LLMProvider>>` - For delegated tool execution
- `context: ContextBuilder` - System prompt assembly
- `sessions: SessionManager` - Conversation persistence
- `token_budget: TokenBudget` - Context window management
- `compactor: ContextCompactor` - Background context compression
- `learning: LearningStore` - Experience database
- `working_memory: WorkingMemoryStore` - Per-session state
- Plus: `workspace`, `model`, `max_iterations`, `max_tokens`, `temperature`, `is_local`, `brave_api_key`, `exec_timeout`, `restrict_to_workspace`, `memory_enabled`, delegation/provenance configs, and per-provider model overrides

**RuntimeCounters** (survives core swaps, 15 fields: 13 atomic, 1 `Mutex<Vec<String>>`, 1 `Arc<AtomicBool>`):
- `delegation_healthy`, `thinking_budget`, `long_mode_turns`, `inference_active`, `last_tools_called`, etc.

**Processing Phases:**

1. **prepare_context()** - Build TurnContext:
   - Snapshot swappable core
   - Build per-turn tool registry
   - Load session history
   - Initialize tracking state

2. **run_agent_loop()** - Main loop:
   - Router preflight (trio mode)
   - LLM streaming call
   - Tool execution (delegated or inline)
   - Context compaction triggers
   - Response finalization

3. **finalize_response()** - Save and emit:
   - Persist session to JSONL
   - Update working memory
   - Queue OutboundMessage

### 2a. Agent Subsystem Modules (`src/agent/`)

Modules extracted from or supporting `agent_loop.rs`:

| Module | Purpose |
|--------|---------|
| `agent_core.rs` | SwappableCore, RuntimeCounters, AgentHandle structs; core build helpers |
| `tool_engine.rs` | Tool execution engine (delegated + inline paths) |
| `tool_guard.rs` | Tool dedup/blocking guard — prevents repeated identical tool calls (B9) |
| `tool_wiring.rs` | Dynamic per-phase tool registry assembly |
| `toolplan.rs` | `ToolPlan` / `ToolPlanAction` types for router output |
| `router.rs` | Trio router preflight, dispatch, and specialist coordination |
| `router_fallback.rs` | Deterministic fallback patterns (9 patterns + default) |
| `anti_drift.rs` | Context hygiene hooks: pollution scoring, turn eviction, babble collapse, format anchors (I6) |
| `circuit_breaker.rs` | Tool loop circuit breaker — forces text response after consecutive all-blocked rounds (B8) |
| `metrics.rs` | Per-request JSONL metrics with 10MB rotation |
| `pipeline.rs` | Multi-step tool pipelines for router (I0) |
| `thread_repair.rs` | Message protocol repair for local models (role alternation, orphan tools) |
| `policy.rs` | `SessionPolicy` — per-session flags (e.g. `local_only`) |
| `role_policy.rs` | Role-based policy enforcement |
| `context_gate.rs` | ContentGate — pass raw / structural briefing / drill-down (I3) |
| `confidence_gate.rs` | Confidence-based gating for router decisions |
| `budget_calibrator.rs` | Per-task-type budget calibration |
| `eval/` | Evaluation framework: `hanoi.rs`, `haystack.rs`, `learning.rs`, `sprint.rs`, `runner.rs` |

### 3. Provider System (`src/providers/`)

```
LLMProvider (trait)
├── chat()         - Single completion
├── chat_stream()  - Streaming completion (default: buffered fallback to chat())
├── get_default_model()
└── get_api_base() - For health checks (default: None)

OpenAICompatProvider (main implementation)
├── api_key, api_base, default_model
├── jit_gate: Option<Arc<JitGate>>  - Serializes requests to LM Studio
└── supports_cache_control()        - Anthropic prompt caching

AnthropicProvider (native Anthropic Messages API)
├── Used exclusively for OAuth/Claude Max flows
└── Speaks native Anthropic API, not OpenAI-compat
```

**Provider Selection Priority:**
```
OpenRouter > DeepSeek > Anthropic > OpenAI > Gemini > Zhipu > ZhipuCoding > Groq > vLLM
```

**Multi-Provider Routing:**
- Model names with prefixes (`groq/llama-3.3-70b`) resolve to specific providers
- `PROVIDER_PREFIXES` constant maps 9 prefixes to API bases: `groq/`, `gemini/`, `openai/`, `anthropic/`, `deepseek/`, `huggingface/`, `zhipu-coding/`, `zhipu/`, `openrouter/`
- Subagents can use different providers than main agent

**Error Classification (`src/errors.rs`):**
```rust
ProviderError {
    HttpError(String),       // Network/transport failures
    ResponseReadError,       // Body read failures
    JsonParseError,          // Malformed JSON
    RateLimited { status, retry_after_ms },  // 429 with server hint
    AuthError { status, message },           // 401/403
    ServerError { status, message },         // 5xx
    Cancelled,
}
```
`is_retryable()` classifies: `RateLimited` and `ServerError` always retry; `HttpError` retries on connection errors and JIT loading strings (but NOT "model not found" — prevents 3x retry on config typos).

**Retry System (`src/providers/retry.rs`):**
- Uses `backon` crate for exponential backoff with jitter
- `provider_backoff()`: 1s → 2s → 4s, max 30s, 3 retries (cloud APIs)
- `jit_backoff()`: 2s → 4s → 8s, 3 retries (local JIT servers)
- `adjust_for_rate_limit()`: respects `Retry-After` from 429 responses — `max(backoff, server_hint)`
- Applied to both `chat()` and `chat_stream()` via `.retry(backoff).when(|e| e.is_retryable())`

**JIT Gate Timing:**
- JIT permit acquisition wait measured separately as `jit_wait_ms`
- API call `elapsed_ms` starts *after* permit acquired
- Both fields logged in `llm_call_complete` / `llm_stream_started` events

### 4. Tool System (`src/agent/tools/`)

**Tool Trait (`base.rs`):**
```rust
trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters(&self) -> serde_json::Value;
    fn execute(&self, params: HashMap<String, Value>) -> String;
    fn execute_with_result(&self, params) -> ToolExecutionResult;          // default impl
    fn execute_with_context(&self, params, ctx) -> String;                 // default impl
    fn execute_with_result_and_context(&self, params, ctx) -> ToolExecutionResult; // default impl
}
```

**ToolRegistry (`registry.rs`):**
- Dynamic registration and execution
- Tool name normalization (aliases → canonical)
- Keyword-triggered tool scoping
- Phase-aware tool selection (FileEditing, WebResearch, etc.)

**Standard Tools:**
- `read_file`, `write_file`, `edit_file`, `list_dir`
- `exec` - Shell commands with safety guards
- `web_search`, `web_fetch` - Brave API integration
- `spawn` - Subagent spawning
- `message` - Channel messaging
- `recall` - Memory search
- `read_skill` - Lazy skill loading
- `cron` - Scheduled job management
- `check_inbox`, `send_email` - Email tools

### 5. Context Builder (`src/agent/context.rs`)

**Assembles the system prompt:**

```
# nanobot (identity)
## Context (time, workspace, model)
## Verification Protocol (if provenance enabled)
## AGENTS.md / SOUL.md / USER.md / TOOLS.md / IDENTITY.md (bootstrap files)
## Session Context (CONTEXT-{channel}.md, fallback to CONTEXT.md)
# Memory
## Long-term Memory (MEMORY.md only, tail-truncated)
## (daily notes and learnings are NOT in the system prompt)
# Skills (XML summary, lazy loading)
# Active Skills (eager-loaded, always:true)
# Subagent Profiles
# Requested Skills (on-demand)
```

**Budget Management (two modes):**

Local (`set_lite_mode`): bootstrap 2%, memory 1%, skills 2%, profiles 1%, cap 30%
Cloud (`scale_budgets`): bootstrap 2%, memory 1%, skills 4%, profiles 2%, cap 40%

- `bootstrap_budget` - Max tokens for instruction files
- `long_term_memory_budget` - Max tokens for MEMORY.md
- `skills_budget` - Max tokens for skills section
- `profiles_budget` - Max tokens for subagent profiles
- `system_prompt_cap` - Hard limit (0 = disabled for cloud defaults)
- Automatic scaling for large context windows (1M tokens)

### 6. Session Management (`src/session/manager.rs`)

**JSONL Format:**
```
{"_type":"metadata","created_at":"...","updated_at":"...","metadata":{}}
{"role":"user","content":"hello","timestamp":"..."}
{"role":"assistant","content":"hi","timestamp":"..."}
{"role":"assistant","tool_calls":[...]}  # Preserves tool context
{"role":"tool","tool_call_id":"...","name":"read_file","content":"..."}
```

**Rotation Policy:**
- Daily rotation: `{session_key}_{YYYY-MM-DD}.jsonl` (cross-day starts fresh)
- Size rotation: When >1MB (same-day only, carries last 10 messages)

### 7. Channel System (`src/channels/`)

**Channel Trait (`base.rs`):**
```rust
trait Channel: Send + Sync {
    fn name(&self) -> &str;
    async fn start(&mut self) -> Result<()>;
    async fn stop(&mut self) -> Result<()>;
    async fn send(&self, msg: &OutboundMessage) -> Result<()>;
    fn is_allowed(&self, sender_id: &str, allow_list: &[String]) -> bool; // default impl
    fn is_running(&self) -> bool;
}
```

**ChannelManager (`manager.rs`):**
- Initializes enabled channels from config
- Spawns each channel as background task
- Dispatches OutboundMessage to correct channel
- Handles channel lifecycle (start/stop)

**Supported Channels:**
- Telegram - Bot API via long polling
- WhatsApp - Via Node.js bridge WebSocket
- Feishu/Lark - HTTP API, **send-only** (no receive; Lark SDK WebSocket has no Rust equivalent)
- Email - IMAP polling + SMTP sending

### 8. Memory System

**MemoryStore (`memory.rs`):**
- Long-term memory: `memory/MEMORY.md`
- Daily notes: `memory/YYYY-MM-DD.md`
- Atomic writes (temp + rename)

**WorkingMemoryStore (`working_memory.rs`):**
- Per-session state in flat Markdown files (`SESSION_{hash}.md` with YAML frontmatter)
- Stored at `{workspace}/memory/sessions/`
- Auto-completes after inactivity
- Injected into system prompt

**SessionIndexer (`session_indexer.rs`):**
- Bridges raw JSONL sessions → searchable `SESSION_{hash}.md` files
- `extract_session_content()` — pure function: extracts user+assistant messages from JSONL lines
- `index_sessions()` — scans `~/.nanobot/sessions/`, creates `.md` for orphaned JSONL files
- Indexed files have `status: indexed` (distinct from `active`/`completed`/`archived`)
- Run via `/sessions index` (REPL) or `nanobot sessions index` (CLI)

**LearningStore (`learning.rs`):**
- Experience database in JSONL (`{workspace}/memory/learnings.jsonl`)
- Append-only with file-level locking
- Used for reflection and improvement

### 9. Skills System (`src/agent/skills.rs`)

**Directory Structure:**
```
workspace/
├── skills/
│   └── my-skill/
│       └── SKILL.md
└── builtin_skills/
    └── core-skill/
        └── SKILL.md
```

**SKILL.md Format:**
```markdown
---
description: What this skill does
always: true
metadata: {"nanobot": {"requires": {"bins": ["python3"]}}}
---
# Instructions
...
```

**Lazy Loading (RLM Mode):**
- Skills appear as name+description only
- Agent uses `read_skill` tool to fetch full content
- Saves context tokens

### 10. Configuration (`src/config/schema.rs`)

**Root Structure:**
```rust
Config {
    agents: AgentsConfig {
        defaults: AgentDefaults {
            workspace, model, local_model, local_api_base,
            max_tokens, temperature, max_tool_iterations,
            max_context_tokens, max_concurrent_chats
        }
    },
    providers: ProvidersConfig {
        anthropic, openai, openrouter, deepseek, groq,
        zhipu, zhipu_coding, vllm, gemini, huggingface
    },
    channels: ChannelsConfig {
        telegram, whatsapp, feishu, email
    },
    tools: ToolsConfig { web, exec_ },
    memory: MemoryConfig,
    tool_delegation: ToolDelegationConfig,
    provenance: ProvenanceConfig,
    proprioception: ProprioceptionConfig,
    trio: TrioConfig,
    worker: WorkerConfig
}
```

**TrioConfig (SLM Trio, 16 fields):**
- `enabled` — Enable trio workflow
- `main_no_think` — Suppress `<think>` for main model
- `router_model`, `router_port`, `router_ctx_tokens`, `router_temperature`, `router_top_p`, `router_no_think` — Router config
- `specialist_model`, `specialist_port`, `specialist_ctx_tokens`, `specialist_temperature` — Specialist config
- `router_endpoint`, `specialist_endpoint` — Optional explicit `ModelEndpoint` overrides (take priority over port+model)
- `vram_cap_gb` — VRAM budget cap; context sizes auto-computed to fit
- `anti_drift: AntiDriftConfig` — Nested anti-drift hooks for SLM context stabilization

**AntiDriftConfig (nested in TrioConfig, 5 fields):**
- `enabled`, `anchor_interval`, `pollution_threshold`, `babble_max_tokens`, `repetition_min_count`

**WorkerConfig (5 fields):**
- `enabled`, `max_depth` (delegation depth, default 3), `python` (enable python_eval), `delegate` (enable recursive workers), `budget_multiplier` (0.0-1.0, default 0.5)

**ProprioceptionConfig (8 fields):**
- `enabled`, `dynamic_tool_scoping`, `audience_aware_compaction`, `grounding_interval`, `gradient_memory`, `raw_window`, `light_window`, `aha_channel`

**ToolDelegationConfig (17 fields):**
- `mode: DelegationMode` — enum: `Inline` (no delegation), `Delegated` (default, tool runner model), `Trio` (strict role separation)
- `apply_mode()` — Applies mode to flags: Inline disables all, Delegated enables tool runner, Trio enables `strict_no_tools_main` + `strict_router_schema` + `role_scoped_context_packs`
- Key fields: `enabled`, `model`, `provider`, `max_iterations`, `max_tokens`, `slim_results`, `max_result_preview_chars`, `auto_local`, `cost_budget`, `default_subagent_model`, `strict_no_tools_main`, `strict_router_schema`, `role_scoped_context_packs`, `strict_local_only`, `strict_toolplan_validation`, `deterministic_router_fallback`, `max_same_tool_call_per_turn`

### 11. Event Bus (`src/bus/events.rs`)

**InboundMessage:**
```rust
{
    channel: String,      // "telegram", "cli", "voice"
    sender_id: String,
    chat_id: String,
    content: String,
    timestamp: DateTime,
    media: Vec<String>,   // Attachment URLs
    metadata: HashMap<String, Value>
}
```

**OutboundMessage:**
```rust
{
    channel: String,
    chat_id: String,
    content: String,
    reply_to: Option<String>,
    media: Vec<String>,
    metadata: HashMap<String, Value>
}
```

**Message Coalescing:**
- Combines rapid-fire messages from same session
- Adds timing annotations `[+200ms]`
- CLI/voice channels bypass coalescing

### 12. Subagent System (`src/agent/subagent.rs`)

**SubagentManager:**
- Spawns background tasks with their own agent loops
- Profiles from `workspace/profiles/*.yaml`
- Max spawn depth: 3 (prevents infinite recursion)
- Results stored in `workspace/events.jsonl`

**Spawn Actions:**
- `spawn` - Start new subagent
- `check` - Get result of completed task
- `wait` - Block until completion
- `cancel` - Abort running task
- `list` - Show running subagents
- `pipeline` - Run multi-step voting pipeline
- `loop` - Autonomous refinement loop

### 13. Observability

**Structured Tracing (both providers):**
- `#[instrument]` spans on all 4 chat methods (`chat`/`chat_stream` x2 providers)
- `skip(self, messages, tools)` prevents logging full message arrays
- Structured fields: `model`, `api_base`, `status`, `elapsed_ms`, `jit_wait_ms`
- `body_snippet` (500 chars) on error responses prevents log explosion
- SSE parse errors logged at `warn!` level (not `debug!`) in both providers

**Per-Request Metrics (`src/agent/metrics.rs`):**
```
~/.nanobot/metrics.jsonl  (one JSON object per LLM call)
```
- Fields: `timestamp`, `request_id`, `role`, `model`, `provider_base`, `elapsed_ms`, `prompt_tokens`, `completion_tokens`, `status`, `tool_calls_requested/executed`
- Optional fields: `error_detail`, `anti_drift_score`, `anti_drift_signals`, `validation_result`
- 10MB size-based rotation: current file renamed to `.jsonl.1`, fresh file started
- Best-effort writes — failures never crash the agent loop

**Log Rotation:**
- `tracing-appender` daily rotation to `~/.nanobot/logs/`
- `nanobot sessions purge --older-than 7d` cleans old logs and metrics

**Request ID Tracking:**
- UUID per agent turn, propagated through pipeline
- Correlates: router decision → LLM call → tool execution → metrics

## Data Flow

### Single Message Processing

```
User Message (Telegram)
    │
    ▼
TelegramChannel receives update
    │
    ▼
InboundMessage created
    │
    ▼
Sent to bus_inbound_tx
    │
    ▼
AgentLoop receives from inbound_rx
    │
    ▼
Coalesced with recent messages
    │
    ▼
TurnContext prepared
    │
    ├── Session history loaded
    ├── System prompt built
    └── Tools registered
    │
    ▼
Router preflight (if trio mode)
    │
    ▼
LLM streaming call
    │
    ├── Text deltas → text_delta_tx → REPL/voice
    └── Tool calls → Tool execution
    │
    ▼
Tool results added to messages
    │
    ▼
Loop until no tool calls or max iterations
    │
    ▼
Final response assembled
    │
    ├── Session saved to JSONL
    ├── Working memory updated
    └── OutboundMessage sent
    │
    ▼
ChannelManager dispatches to TelegramChannel
    │
    ▼
Telegram API sends response
```

### Context Compaction Flow

```
Token usage exceeds threshold
    │
    ▼
Background compaction task spawned
    │
    ├── Snapshot current messages
    └── Call memory_provider to summarize
    │
    ▼
CompactionResult received
    │
    ├── Compact summary of old messages
    └── Recent messages preserved verbatim
    │
    ▼
Applied to live conversation
    │
    ├── Fresh system message
    ├── Compaction summary
    └── Messages after watermark
```

## Key Design Patterns

### 1. Swappable Core Pattern
- `AgentHandle` wraps `Arc<RwLock<Arc<SwappableCore>>>`
- `/local` toggle rebuilds core without losing counters
- Cheap snapshot via `handle.swappable()`

### 2. Fan-Out Concurrency
- Multiple sessions processed in parallel (up to `max_concurrent_chats`)
- Same session serialized via async mutex
- Background compaction doesn't block processing

### 3. Channel-Based Communication
- `mpsc::unbounded_channel` for message passing
- `broadcast` channel for subagent cancellation
- `CancellationToken` for graceful shutdown

### 4. Trait-Based Abstraction
- `LLMProvider` trait for multi-provider support
- `Tool` trait for extensibility
- `Channel` trait for chat adapters

### 5. Context Budget Management
- Proportional scaling for large context windows
- File-granularity inclusion (all-or-nothing, no mid-content truncation)
- Tail truncation for memory (newest facts survive)

## File Locations

```
~/.nanobot/
├── config.json           # Main configuration
├── metrics.jsonl          # Per-request LLM call metrics (10MB rotation)
├── metrics.jsonl.1        # Previous metrics backup
├── logs/                  # Daily rotated structured logs
│   └── nanobot.2026-02-21.log
├── sessions/             # Conversation history
│   └── cli_default_2026-02-21.jsonl
└── workspace/
    ├── AGENTS.md         # Agent instructions
    ├── SOUL.md           # Personality
    ├── USER.md           # User preferences
    ├── CONTEXT.md        # Session context
    ├── CONTEXT-cli.md    # Per-channel context
    ├── memory/
    │   ├── MEMORY.md     # Long-term facts
    │   ├── 2026-02-20.md # Daily notes
    │   └── sessions/     # Working memory + indexed sessions
    │       ├── SESSION_{hash}.md  # Compaction summaries + indexed JSONL extracts
    │       └── archived/          # Completed sessions
    ├── skills/           # Custom skills
    ├── profiles/         # Subagent profiles
    ├── events.jsonl      # Subagent results
    ├── observations/     # LLM-generated summaries
    └── audit.jsonl       # Tool call log (provenance)
```

## Testing

- All tests inline in `#[cfg(test)] mod tests`
- `cargo test` runs all tests
- `cargo test test_name` for specific test
- `-- --nocapture` to see test output
- ~1390 tests in codebase (run `cargo test -- --list` for current count)

## Feature Flags

- `default` - Core functionality
- `voice` - Voice mode (requires jack-voice, crossterm, lingua)

---

# Architecture Review: Flaws & Improvements

## Critical Flaws

### 1. **agent_loop.rs size** *(substantially addressed)*
**Previously:** 4200+ lines handling routing, streaming, tool execution, compaction, provenance, and trio mode.

**Progress (2544 lines remaining):** Major extractions completed:
- `agent_core.rs` — SwappableCore, RuntimeCounters, AgentHandle (extracted from agent_loop)
- `tool_engine.rs` — Tool execution engine (delegated + inline paths)
- `router.rs` — Trio router preflight and dispatch
- `router_fallback.rs` — Deterministic fallback patterns (9 patterns)
- `tool_guard.rs` — Tool dedup/blocking guard
- `tool_wiring.rs` — Dynamic per-phase tool registry assembly

**Remaining recommendation:**
- Extract `ConversationManager` - handles message assembly, history, compaction
- Keep `agent_loop.rs` as thin coordinator

### 2. **Implicit State Machine in TurnContext**
**Problem:** `TurnContext` has 26 fields tracking turn state, with flow control delegated to `FlowControl` (8 fields: 4 booleans, `ToolGuard`, 2x `u32`, `Option<Instant>`). Easy to get into invalid states.

**Recommendation:**
- Model turn lifecycle as explicit state machine:
  ```
  Preparing → Routing → Streaming → ToolExecution → Compacting → Finalizing
  ```
- Use enum-based states with state-specific data
- Impossible to have invalid state combinations

### 3. **Error Handling Inconsistency** *(partially addressed)*
**Problem:** Mix of `anyhow::Result`, `String` errors, `ToolExecutionResult`, and `LLMResponse` with `finish_reason: "error"`. Hard to track error origins.

**Progress:** `ProviderError` (7 variants: HttpError, ResponseReadError, JsonParseError, RateLimited, AuthError, ServerError, Cancelled) and `ToolErrorKind` (6 variants: Timeout, PermissionDenied, NotFound, InvalidArgs, ToolNotFound, ExecutionFailed) defined in `src/errors.rs` with `thiserror`. Provider errors have `is_retryable()` classification. Tool errors have `classify_tool_error()` from string matching.

**Remaining:**
- Define top-level `AgentError` enum wrapping all domain errors
- Eliminate `String` errors from tool execution path
- Never wrap errors in success types

### 4. **Global Mutable State (LOCAL_MODE)**
**Problem:** `static LOCAL_MODE: AtomicBool` is global mutable state. Hard to test, causes surprising behavior.

**Recommendation:**
- Move to `RuntimeCounters` or `SessionPolicy`
- Pass explicitly through context
- Makes state visible and testable

## Major Design Issues

### 5. **Provider Selection Logic Scattered**
**Problem:** Provider resolution in `cli.rs`, `config/schema.rs`, `subagent.rs`, and `agent_loop.rs`. Each has slightly different logic.

**Recommendation:**
- Create `ProviderRegistry` type that centralizes:
  - Provider creation from config
  - Model prefix resolution
  - Fallback chain management
- Single source of truth for provider selection

### 6. **Tight Coupling Between Core and Tools**
**Problem:** Tools like `SpawnTool` receive 7+ callbacks for different operations. Adding new tool features requires changes in multiple places.

**Recommendation:**
- Define `ToolContext` struct with all context:
  ```rust
  struct ToolContext {
      workspace: PathBuf,
      outbound_tx: UnboundedSender<OutboundMessage>,
      subagents: Arc<SubagentManager>,
      audit_tx: Option<UnboundedSender<ToolEvent>>,
      // ...
  }
  ```
- Tools receive context, not callbacks

### 7. **Context Budget Magic Numbers**
**Problem:** Budget calculations scattered with unexplained constants:
```rust
self.bootstrap_budget = (max_context_tokens / 50).clamp(300, 2_000); // 2%
```

**Recommendation:**
- Define named constants with documentation
- Create `BudgetConfig` struct with validation
- Add observability: log budget decisions

### 8. **Subagent Depth as Global Constant**
**Problem:** `MAX_SPAWN_DEPTH = 3` hardcoded. No way to configure for different workloads.

**Recommendation:**
- Move to `WorkerConfig` in schema
- Add runtime configuration option

## Moderate Issues

### 9. **Inconsistent Async Patterns**
**Problem:** Mix of `tokio::sync::Mutex` and `std::sync::Mutex`. Some `Arc<Mutex<T>>`, some `Arc<ArcSwap<T>>`.

**Recommendation:**
- Document when to use each sync primitive
- Standardize: `Arc<RwLock>` for read-heavy, `Arc<Mutex>` for write-heavy
- Consider `parking_lot` for better performance

### 10. **Session Rotation Triggers**
**Problem:** Rotation on 24h OR 1MB. No consideration for conversation complexity or token count.

**Recommendation:**
- Add token-based rotation threshold
- Consider conversation turn count
- Add configuration for rotation policy

### 11. **Tool Result Truncation Silent**
**Problem:** Tool results truncated without indication to LLM. May lose critical information.

**Recommendation:**
- Add `[truncated, X chars omitted]` marker
- Consider compression instead of truncation
- Allow tool to signal importance of output

### 12. ~~**No Rate Limiting on Provider Calls**~~ *(addressed)*
~~**Problem:** Can exceed API rate limits during tool loops. No backoff or retry logic at agent level.~~

**Resolved:** `backon` crate provides exponential backoff with jitter on both `chat()` and `chat_stream()`. `adjust_for_rate_limit()` respects `Retry-After` headers from 429 responses. See Provider System § Retry System above.

## Minor Improvements

### 13. **Logging Verbosity** *(substantially addressed)*
- ~~Add structured logging with spans~~ Done: `#[instrument]` on all chat methods, structured fields
- ~~Include session_id, turn_number in all logs~~ Partially: `request_id` tracked per turn
- ~~Add timing metrics for performance analysis~~ Done: `metrics.jsonl` with `elapsed_ms`, `jit_wait_ms`, token counts

### 14. **Configuration Validation**
- Add schema validation on load
- Detect conflicting settings early
- Warn on deprecated options

### 15. **Test Coverage Gaps**
- Add integration tests for multi-turn conversations
- Test tool execution with edge cases (timeouts, panics)
- Add property-based tests for budget calculations

### 16. **Documentation**
- Add rustdoc to all public APIs
- Document invariants (e.g., "must hold lock before calling")
- Add architecture decision records (ADRs)

## Recommended Refactoring Priority

1. **High Impact, Medium Effort:**
   - Extract `ToolExecutionEngine` from agent_loop.rs
   - Create `ProviderRegistry` to centralize provider logic
   - ~~Define domain error types~~ Done: `ProviderError`, `ToolErrorKind` (top-level `AgentError` remains)

2. **High Impact, High Effort:**
   - Model turn lifecycle as state machine
   - Eliminate global LOCAL_MODE

3. **Medium Impact, Low Effort:**
   - ~~Add rate limiting to providers~~ Done: `backon` retry with backoff
   - Improve tool result truncation signaling
   - ~~Add structured logging~~ Done: `#[instrument]` spans + `metrics.jsonl`

4. **Ongoing:**
   - Increase test coverage
   - Add rustdoc documentation
   - Refactor toward smaller modules

## Strengths (Keep These)

1. **SwappableCore Pattern** - Excellent for hot-reloading configuration
2. **Channel Abstraction** - Easy to add new chat platforms
3. **Tool Trait** - Clean extensibility for capabilities
4. **Context Budget System** - Prevents context overflow
5. **Session Rotation** - Prevents unbounded file growth
6. **JSONL Storage** - Human-readable, easy debugging
7. **Async-First Design** - Scales well for concurrent users
