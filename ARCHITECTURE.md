# nanobot Architecture Map

> Complete reference for LLM onboarding. Last updated: 2026-02-20

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

### 1. Entry Points (`src/main.rs`, `src/cli.rs`)

**main.rs** - CLI parsing and command routing:
- `Commands` enum defines all CLI commands
- Routes to handler functions in `cli.rs`
- Panic hook restores terminal state
- Global `LOCAL_MODE` atomic flag for cloud/local switching

**cli.rs** - Command implementations:
- `cmd_onboard()` - Initialize config and workspace
- `cmd_gateway()` - Start multi-channel gateway
- `build_core_handle()` - Create swappable core with provider
- `rebuild_core()` - Hot-swap provider on `/local` toggle
- `create_agent_loop()` - Wire agent loop with channels

### 2. Agent Loop (`src/agent/agent_loop.rs`)

**The heart of message processing.** Key concepts:

```
AgentHandle (cloneable)
├── core: Arc<RwLock<Arc<SwappableCore>>>  -- Hot-swappable on /local
└── counters: Arc<RuntimeCounters>          -- Persists across swaps
```

**SwappableCore** contains:
- `provider: Arc<dyn LLMProvider>` - Main LLM
- `memory_provider: Arc<dyn LLMProvider>` - For compaction/reflection
- `router_provider: Option<Arc<dyn LLMProvider>>` - For trio routing
- `specialist_provider: Option<Arc<dyn LLMProvider>>` - For trio execution
- `context: ContextBuilder` - System prompt assembly
- `sessions: SessionManager` - Conversation persistence
- `token_budget: TokenBudget` - Context window management
- `compactor: ContextCompactor` - Background context compression
- `learning: LearningStore` - Experience database
- `working_memory: WorkingMemoryStore` - Per-session state

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

### 3. Provider System (`src/providers/`)

```
LLMProvider (trait)
├── chat()         - Single completion
├── chat_stream()  - Streaming completion
├── get_default_model()
└── get_api_base() - For health checks

OpenAICompatProvider (main implementation)
├── api_key, api_base, default_model
├── jit_gate: Option<Arc<JitGate>>  - Serializes requests to LM Studio
└── supports_cache_control()        - Anthropic prompt caching
```

**Provider Selection Priority:**
```
OpenRouter > DeepSeek > Anthropic > OpenAI > Gemini > Zhipu > Groq > vLLM
```

**Multi-Provider Routing:**
- Model names with prefixes (`groq/llama-3.3-70b`) resolve to specific providers
- `PROVIDER_PREFIXES` constant maps prefixes to API bases
- Subagents can use different providers than main agent

### 4. Tool System (`src/agent/tools/`)

**Tool Trait (`base.rs`):**
```rust
trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters(&self) -> serde_json::Value;
    fn execute(&self, params: HashMap<String, Value>) -> String;
    fn execute_with_result(&self, params) -> ToolExecutionResult;
    fn execute_with_context(&self, params, ctx) -> String;
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

### 5. Context Builder (`src/agent/context.rs`)

**Assembles the system prompt:**

```
# nanobot (identity)
## Context (time, workspace, model)
## Verification Protocol (if provenance enabled)
## AGENTS.md / SOUL.md / USER.md (bootstrap files)
## Session Context (CONTEXT-{channel}.md)
# Memory
## Long-term Memory (MEMORY.md, tail-truncated)
# Skills (XML summary, lazy loading)
# Active Skills (eager-loaded)
# Subagent Profiles
## Current Session (channel, chat_id)
## Voice Mode (if applicable)
```

**Budget Management:**
- `bootstrap_budget` - Max tokens for instruction files
- `long_term_memory_budget` - Max tokens for MEMORY.md
- `skills_budget` - Max tokens for skills section
- `system_prompt_cap` - Hard limit (0 = disabled for cloud)
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
- Daily rotation: `{session_key}_{YYYY-MM-DD}.jsonl`
- Size rotation: When >1MB
- Carries last 10 messages to new session

### 7. Channel System (`src/channels/`)

**Channel Trait (`base.rs`):**
```rust
trait Channel: Send + Sync {
    fn name(&self) -> &str;
    async fn start(&mut self) -> Result<()>;
    async fn stop(&mut self) -> Result<()>;
    async fn send(&self, msg: &OutboundMessage) -> Result<()>;
    fn is_allowed(&self, sender_id: &str, allow_list: &[String]) -> bool;
}
```

**ChannelManager (`manager.rs`):**
- Initializes enabled channels from config
- Spawns each channel as background task
- Dispatches OutboundMessage to correct channel
- Handles channel lifecycle (start/stop)

**Supported Channels:**
- Telegram - Bot API via long polling
- WhatsApp - Via bridge WebSocket
- Feishu/Lark - WebSocket long connection
- Email - IMAP polling + SMTP sending

### 8. Memory System

**MemoryStore (`memory.rs`):**
- Long-term memory: `memory/MEMORY.md`
- Daily notes: `memory/YYYY-MM-DD.md`
- Atomic writes (temp + rename)

**WorkingMemoryStore (`working_memory.rs`):**
- Per-session state in SQLite
- Auto-completes after inactivity
- Injected into system prompt

**LearningStore (`learning.rs`):**
- Experience database (SQLite)
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

**TrioConfig (SLM Trio):**
- Router model for dispatch decisions
- Specialist model for execution
- Context size auto-computed from VRAM cap

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
├── nanobot.log           # REPL logs
├── sessions/             # Conversation history
│   └── cli_default_2026-02-20.jsonl
└── workspace/
    ├── AGENTS.md         # Agent instructions
    ├── SOUL.md           # Personality
    ├── USER.md           # User preferences
    ├── CONTEXT.md        # Session context
    ├── CONTEXT-cli.md    # Per-channel context
    ├── memory/
    │   ├── MEMORY.md     # Long-term facts
    │   └── 2026-02-20.md # Daily notes
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
- 1248 tests in codebase

## Feature Flags

- `default` - Core functionality
- `voice` - Voice mode (requires jack-voice, crossterm, lingua)

---

# Architecture Review: Flaws & Improvements

## Critical Flaws

### 1. **Monolithic agent_loop.rs (1700+ lines)**
**Problem:** Single file handles routing, streaming, tool execution, compaction, provenance, and trio mode. Hard to navigate and test in isolation.

**Recommendation:**
- Extract `ToolExecutionEngine` - handles tool running, delegation, routing
- Extract `ConversationManager` - handles message assembly, history, compaction
- Extract `TrioOrchestrator` - handles router/specialist coordination
- Keep `agent_loop.rs` as thin coordinator

### 2. **Implicit State Machine in TurnContext**
**Problem:** `TurnContext` has 30+ fields tracking turn state. Flow control via boolean flags (`force_response`, `router_preflight_done`, `forced_finalize_attempted`). Easy to get into invalid states.

**Recommendation:**
- Model turn lifecycle as explicit state machine:
  ```
  Preparing → Routing → Streaming → ToolExecution → Compacting → Finalizing
  ```
- Use enum-based states with state-specific data
- Impossible to have invalid state combinations

### 3. **Error Handling Inconsistency**
**Problem:** Mix of `anyhow::Result`, `String` errors, `ToolExecutionResult`, and `LLMResponse` with `finish_reason: "error"`. Hard to track error origins.

**Recommendation:**
- Define domain error types:
  ```rust
  enum AgentError {
      Provider(ProviderError),
      Tool(ToolError),
      Session(SessionError),
      Context(ContextError),
  }
  ```
- Use `thiserror` for structured errors
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

### 12. **No Rate Limiting on Provider Calls**
**Problem:** Can exceed API rate limits during tool loops. No backoff or retry logic at agent level.

**Recommendation:**
- Add `RateLimiter` to provider wrapper
- Implement exponential backoff
- Respect `Retry-After` headers

## Minor Improvements

### 13. **Logging Verbosity**
- Add structured logging with spans
- Include session_id, turn_number in all logs
- Add timing metrics for performance analysis

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
   - Define domain error types

2. **High Impact, High Effort:**
   - Model turn lifecycle as state machine
   - Eliminate global LOCAL_MODE

3. **Medium Impact, Low Effort:**
   - Add rate limiting to providers
   - Improve tool result truncation signaling
   - Add structured logging

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
