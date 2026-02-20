---
date: 2026-02-19
researcher: Claude (Sonnet 4.6)
repository: nanobot
branch: vibe-1771523298
git_commit_at_research: 0550f7d2c7be47656c4a342db0fdc6319a54daac
type: repo_research
tags: [architecture, local-llm, trio, tools, providers, channels, voice]
---

# Nanobot Repository Research Handoff

## Executive Summary

Nanobot is a Rust-native personal AI assistant framework, ported from a Python original (HKUDS/nanobot, MIT licensed). It is built around a single `OpenAICompatProvider` that speaks to all LLM backends, an `AgentLoop` at its core, and a modular tool system. The project is in active development on a vibe branch with heavy focus on local LLM reliability (llama.cpp + LM Studio + mistral.rs backends) and a "trio" multi-model architecture.

Current test suite: **1213 tests, 0 failures, 5 ignored.**

---

## Architecture and Structure

### Repository Layout

```
/home/peppi/Dev/nanobot/
├── src/
│   ├── main.rs                   Entry point, CLI parsing (clap)
│   ├── cli.rs                    Command implementations
│   ├── repl/                     Interactive REPL (agent subcommand)
│   │   ├── mod.rs                REPL loop, streaming renderer, voice pipeline wiring
│   │   └── commands.rs           Slash-command dispatch, ReplContext struct
│   ├── agent/                    Core agent subsystem (40+ modules)
│   │   ├── agent_loop.rs         AgentLoop + SwappableCore + AgentHandle (4168 lines)
│   │   ├── context.rs            ContextBuilder (prompt assembly)
│   │   ├── tool_runner.rs        Delegated tool execution loop
│   │   ├── subagent.rs           Background subagent manager
│   │   ├── thread_repair.rs      OpenAI protocol violation repair
│   │   ├── provenance.rs         Claim verification against audit log
│   │   ├── audit.rs              SHA-256 hash-chained audit log (JSONL)
│   │   ├── context_gate.rs       Budget-aware content sizing
│   │   ├── tools/                Tool implementations
│   │   │   ├── base.rs           Tool trait, ToolExecutionResult, ToolExecutionContext
│   │   │   ├── registry.rs       ToolRegistry, alias normalization, validation
│   │   │   ├── filesystem.rs     read_file, write_file, list_dir, edit_file
│   │   │   ├── shell.rs          exec (shell command execution)
│   │   │   ├── web.rs            web_fetch, web_search (Brave API)
│   │   │   ├── spawn.rs          SpawnTool (subagent spawning)
│   │   │   ├── message.rs        MessageTool (inter-channel messaging)
│   │   │   └── cron_tool.rs      CronScheduleTool
│   ├── providers/
│   │   ├── base.rs               LLMProvider trait, LLMResponse, StreamChunk
│   │   ├── openai_compat.rs      Single universal provider (all backends)
│   │   ├── jit_gate.rs           Single-permit semaphore for LM Studio JIT loading
│   │   ├── anthropic.rs          Anthropic-native (prompt caching)
│   │   └── transcription.rs      Whisper STT
│   ├── config/
│   │   ├── schema.rs             All config structs (1589 lines)
│   │   └── loader.rs             Config file I/O
│   ├── channels/
│   │   ├── telegram.rs           Telegram polling
│   │   ├── whatsapp.rs           WhatsApp WebSocket bridge
│   │   ├── feishu.rs             Feishu/Lark WebSocket
│   │   └── email.rs              IMAP + SMTP
│   ├── bus/events.rs             InboundMessage / OutboundMessage types
│   ├── session/manager.rs        JSONL session persistence
│   ├── cron/                     Scheduled job system
│   ├── heartbeat/                Periodic heartbeat service
│   ├── server.rs                 Local llama-server lifecycle management
│   ├── lms.rs                    LM Studio `lms` CLI wrapper
│   ├── mistralrs.rs              mistral.rs binary wrapper
│   ├── tui.rs                    Terminal raw mode management
│   ├── syntax.rs                 Syntax highlighting (syntect)
│   ├── voice.rs                  Voice session (feature-gated)
│   └── voice_pipeline.rs         TTS pipeline (feature-gated)
├── plans/                        Design documents, strategy plans
├── thoughts/                     Continuity ledgers, handoffs
│   ├── ledgers/                  Per-topic continuity files
│   └── shared/handoffs/          Session handoff documents
├── skills/                       Built-in skill markdown files
├── experiments/                  Experimental code
├── scripts/                      Helper shell scripts
├── docs/                         Documentation
├── Cargo.toml                    Dependencies
├── CLAUDE.md                     Claude Code project instructions
└── AGENTS.md                     Agent-facing workspace bootstrap
```

### Message Flow

```
User (CLI/Channel)
    |
    v
InboundMessage (bus/events.rs)
    |
    v
AgentLoop.run() [gateway] or process_direct() [CLI]
    |
    +--- ContextBuilder: assembles system prompt (bootstrap files + memory + skills)
    |
    v
OpenAICompatProvider.chat_stream()
    |
    v
LLM Response (streaming via SSE)
    |
    +--- TextDelta -> REPL incremental renderer
    +--- ThinkingDelta -> REPL (dimmed, optionally suppressed for TTS)
    |
    v (if tool calls)
ToolRegistry.execute() [inline] or ToolRunnerConfig [delegated]
    |
    v
Tool results -> inject back into message history
    |
    v
Loop up to max_tool_iterations (default: 20)
    |
    v
OutboundMessage (bus/events.rs)
```

---

## Key Design Decisions

### 1. Single Provider Architecture

All LLM backends go through `OpenAICompatProvider` in `src/providers/openai_compat.rs`. Provider selection happens at construction time via API key prefix detection:

- `sk-or-` prefix -> OpenRouter
- `sk-ant-` prefix -> Anthropic direct
- `gsk_` prefix -> Groq
- `sk-` with non-routed model -> OpenAI direct
- Model contains `deepseek` -> DeepSeek
- Explicit `api_base` provided -> custom endpoint
- Default fallback -> OpenRouter

Model name normalization is built in: `"opus"` becomes `"claude-opus-4-6"`, `"sonnet"` becomes `"claude-sonnet-4-5-20250929"`, etc.

### 2. SwappableCore / AgentHandle Pattern

The agent core is split into two parts to support live hot-swapping via `/local` and `/model` commands:

- **`SwappableCore`**: Provider, model, context builder, sessions, budgets — everything that changes on mode switch. Wrapped in `Arc<RwLock<Arc<SwappableCore>>>`.
- **`RuntimeCounters`**: Atomic counters (context usage, token counts, inference state, delegation health) that survive core swaps.
- **`AgentHandle`**: Combines both; cheap to clone (two Arc bumps).

Files: `src/agent/agent_loop.rs` lines 67-202.

### 3. Tool System

The `Tool` trait in `src/agent/tools/base.rs` requires four methods:
- `name() -> &str`
- `description() -> &str`
- `parameters() -> serde_json::Value` (JSON Schema for the tool's arguments)
- `execute(params: HashMap<String, Value>) -> String`

Two additional optional overrides:
- `execute_with_result()` — returns `ToolExecutionResult { ok, data, error }`, maps `"Error:"` prefixed strings to failures automatically
- `execute_with_context()` — adds cancellation token + progress event channel for long-running tools

Tools are exposed to LLMs via `to_schema()` which emits OpenAI function-calling JSON format.

Tools needing runtime state (MessageTool, SpawnTool, CronScheduleTool) use callback closures (`Arc<dyn Fn(...) -> Pin<Box<...>>>`) injected at loop construction time.

Registry alias normalization (`wait/check/list/cancel` -> `spawn` actions; `q` -> `query`, `link` -> `url`, etc.) lives in `src/agent/tools/registry.rs`.

### 4. Local LLM Protocol Constraints (Critical)

Local models (llama-server, mistral.rs) impose strict message protocol requirements that cloud APIs do not:

- Conversations **must** end with `role: "user"`
- Assistant message prefill is **not** supported
- `role: "tool"` messages cannot be the last message
- Mid-conversation `role: "system"` messages may be rejected

After any tool result injection, add a user continuation:

```rust
messages.push(json!({
    "role": "user",
    "content": "Based on the tool results above, continue with the task."
}));
```

The `is_local` flag on `SwappableCore` gates this behavior. The `thread_repair.rs` module (`repair_messages()`) handles wider protocol repair:
1. Dedup tool results by tool_call_id
2. Remove positionally orphaned tool results
3. Fix orphaned tool_calls (no matching result)
4. Merge consecutive user messages
5. Ensure first non-system message is user role

### 5. Trio Architecture (Active Development)

The "trio" mode (`DelegationMode::Trio` in config) routes through three specialized models:
- **Main (orchestrator)**: conversation, task decomposition (NanBeige 3B or Nemotron-Nano)
- **Router/factotum**: tool dispatch only, JSON-only output (Qwen3-1.7B or FunctionGemma)
- **Specialist**: summary, coding, extraction (Ministral-3-8B)

Config fields controlling trio behavior (in `~/.nanobot/config.json`):
```json
{
  "toolDelegation": {
    "mode": "trio",
    "strictNoToolsMain": true,
    "strictRouterSchema": true,
    "roleScopedContextPacks": true
  },
  "trio": {
    "enabled": true,
    "routerModel": "qwen3-1.7b",
    "routerPort": 8094,
    "specialistModel": "ministral-3-8b-instruct-2512",
    "specialistPort": 8095,
    "mainNoThink": true
  }
}
```

The `apply_mode()` function in `src/config/schema.rs` lines 920-943 sets strict flags automatically based on `DelegationMode`. Setting `mode: "trio"` without using `apply_mode()` will not activate the strict flags — this was a prior bug.

### 6. Context Budget System

Multiple overlapping systems manage context:
- **TokenBudget**: Estimates token counts using tiktoken (cl100k_base)
- **ContextGate**: Routes large tool outputs to disk cache, returns briefing summaries instead
- **ContextCompactor**: Summarizes conversation history when it exceeds compaction threshold
- **ContextBuilder**: Assembles system prompt with separate budgets for bootstrap, memory, skills, profiles

Bootstrap files loaded from workspace root: `AGENTS.md`, `SOUL.md`, `USER.md`, `TOOLS.md`, `IDENTITY.md`.

### 7. LM Studio JIT Gate

When `local_api_base` points to a remote LM Studio instance, the `JitGate` (`src/providers/jit_gate.rs`) serializes all three trio model providers through a single-permit semaphore. This prevents concurrent model switching that crashes LM Studio's JIT loader.

### 8. Inference Engine Selection

Config field `agents.defaults.inferenceEngine: "auto" | "llama" | "lms"`.

- `"auto"`: tries LM Studio -> llama-server in order
- `"lms"`: uses LM Studio via `lms` CLI (WSL2 path-aware: `/mnt/c/Users/*/...`)
- `"llama"`: uses llama-server (legacy)

> **Note**: mistral.rs was evaluated and rejected (2026-02-20) — could not achieve stable operation after full-day testing.

Model name pipeline for local models:
1. Config: `"nanbeige4.1-3b-q8_0.gguf"` (GGUF filename)
2. `strip_gguf_suffix()` in `cli.rs`: `"nanbeige4.1-3b"` (LM Studio model ID)
3. Internal routing: `"local:nanbeige4.1-3b"` (prefix for local routing decisions)
4. Provider API call: `"nanbeige4.1-3b"` (`local:` stripped in `openai_compat.rs`)

### 9. Provenance and Audit System

Every tool call is recorded in an immutable append-only JSONL audit log with SHA-256 hash chain (stored at `{workspace}/memory/audit/{session_key}.jsonl`).

`ClaimVerifier` in `src/agent/provenance.rs` performs regex-based claim extraction from agent responses and verifies them against the audit log. No LLM involved in verification — purely mechanical. Claims are classified as: `Observed`, `Derived`, `Claimed`, `Recalled`.

### 10. Memory Architecture

Three layers:
- **Working Memory**: Per-session state in `WorkingMemoryStore`, budget-limited injection into system prompt
- **Daily Memory Notes**: Written to `{workspace}/memory/YYYY-MM-DD.md`
- **Long-term Memory**: `{workspace}/MEMORY.md`, condensed by `Reflector` when token threshold hit

Background `Reflector` task condenses observations into `MEMORY.md` at `reflection_threshold` tokens (default: 20,000).

---

## Provider Configuration

### Priority Order (Config::get_api_key)

OpenRouter > DeepSeek > Anthropic > OpenAI > Gemini > Zhipu > Groq > vLLM

The first provider with a non-empty, non-`"none"` API key wins.

### Model Prefix Routing

`ProvidersConfig::resolve_model_prefix()` in `src/config/schema.rs` handles prefixed model names like `groq/llama-3.3-70b`, `anthropic/claude-opus-4-6`, etc. The `PROVIDER_PREFIXES` constant is the single source of truth for prefix-to-endpoint mapping.

### Supported Providers

| Prefix | Default Base URL |
|--------|-----------------|
| `anthropic/` | `https://api.anthropic.com/v1` |
| `openai/` | `https://api.openai.com/v1` |
| `openrouter/` | `https://openrouter.ai/api/v1` |
| `groq/` | `https://api.groq.com/openai/v1` |
| `gemini/` | `https://generativelanguage.googleapis.com/v1beta/openai` |
| `deepseek/` | `https://api.deepseek.com` |
| `huggingface/` | `https://router.huggingface.co/v1` |
| `zhipu/` | `https://api.z.ai/api/paas/v4` |
| `zhipu-coding/` | `https://api.z.ai/api/coding/paas/v4` |

---

## CLI Commands

Binary: `nanobot`

| Command | Description |
|---------|-------------|
| `nanobot onboard` | Initialize config and workspace |
| `nanobot agent [-m "message"] [-s session] [--local]` | Interactive REPL or one-shot message |
| `nanobot gateway [--port 18790]` | Start all channels + agent loop |
| `nanobot status` | Show channel/model status |
| `nanobot tune --input bench.json` | Select best local profile from benchmark |
| `nanobot channels status` | Show channel status |
| `nanobot cron list/add/remove/enable` | Manage scheduled tasks |
| `nanobot whatsapp` | Quick-start WhatsApp channel |
| `nanobot telegram [--token TOKEN]` | Quick-start Telegram |
| `nanobot email [--imap-host H --smtp-host H]` | Quick-start email channel |
| `nanobot ingest files...` | Ingest documents into knowledge store |
| `nanobot search query` | Search knowledge store |
| `nanobot eval hanoi/haystack/learn/sprint` | Run evaluation benchmarks |

### REPL Slash Commands (partial list)

`/local`, `/model`, `/think [budget]`, `/long`, `/nothink`, `/status`, `/restart`, `/voice`, `/prov`, `/provenance`, `/m`, `/session`, `/clear`, `/compact`, `/forget`, `/mem`, `/cron`, `/spawn`, `/check`, `/cancel`, `/wait`

---

## Configuration Schema

Config file: `~/.nanobot/config.json` (camelCase keys throughout).

Key config struct tree (from `src/config/schema.rs`):

```
Config
├── agents: AgentsConfig
│   └── defaults: AgentDefaults
│       ├── workspace: "~/.nanobot/workspace"
│       ├── model: "anthropic/claude-opus-4-5"
│       ├── localModel: ""              # GGUF filename
│       ├── localApiBase: ""            # Remote LM Studio URL
│       ├── localMaxContextTokens: 32768
│       ├── maxTokens: 2048
│       ├── temperature: 0.7
│       ├── maxToolIterations: 20
│       ├── maxContextTokens: 128000
│       ├── maxConcurrentChats: 4
│       ├── maxToolResultChars: 10000
│       ├── inferenceEngine: "auto"     # auto|mistralrs|lms|llama
│       └── lmsPort: 1234
├── providers: ProvidersConfig
│   ├── anthropic: ProviderConfig { apiKey, apiBase }
│   ├── openai, openrouter, deepseek, groq, zhipu, vllm, gemini, huggingface
├── channels: ChannelsConfig
│   ├── whatsapp, telegram, feishu, email
├── gateway: GatewayConfig { host, port }
├── tools: ToolsConfig
│   ├── web.search: WebSearchConfig { apiKey, maxResults }
│   └── exec: ExecToolConfig { timeout, restrictToWorkspace }
├── memory: MemoryConfig
│   ├── enabled: true
│   ├── model: ""
│   ├── workingMemoryBudget: 1500
│   ├── reflectionThreshold: 20000
│   └── lazySkills: true
├── provenance: ProvenanceConfig
│   ├── enabled, auditLog, showToolCalls, verifyClaims
│   ├── strictMode, systemPromptRules, responseBoundary
├── toolDelegation: ToolDelegationConfig
│   ├── mode: "inline" | "delegated" | "trio"
│   ├── enabled, model, maxIterations, maxTokens
│   ├── slimResults, defaultSubagentModel
│   ├── strictNoToolsMain, strictRouterSchema, roleScopedContextPacks
│   └── strictToolplanValidation: true
└── trio: TrioConfig
    ├── enabled, mainNoThink
    ├── routerModel, routerPort: 8094, routerCtxTokens: 4096
    ├── routerTemperature: 0.6, routerTopP: 0.95, routerNoThink: true
    ├── specialistModel, specialistPort: 8095
    ├── specialistCtxTokens: 8192, specialistTemperature: 0.7
```

---

## Channel Adapters

All channels push to the same `InboundMessage` bus and consume from `OutboundMessage`.

| Channel | Transport | File |
|---------|-----------|------|
| Telegram | HTTP long polling | `src/channels/telegram.rs` |
| WhatsApp | WebSocket bridge (separate Node.js bridge process) | `src/channels/whatsapp.rs` |
| Feishu/Lark | WebSocket long connection | `src/channels/feishu.rs` |
| Email | IMAP polling + SMTP sending | `src/channels/email.rs` |

Config controls `allow_from` allowlists per channel.

---

## Voice Mode

Feature-gated (`cargo build --features voice`). Requires `jack-voice` crate at `../jack-voice/jack-voice`.

Components:
- STT: Whisper (via jack-voice)
- TTS: Supertonic (ONNX diffusion model), sentence-by-sentence pipeline
- Audio: PulseAudio (`paplay` subprocess), WSL2 uses `/mnt/wslg/PulseServer`
- Language detection: `lingua` crate for TTS routing (8 languages configured)

Known upstream TTS issues: supertonic drops words on multi-sentence synthesis (non-deterministic diffusion). VAD is listed as future work.

Remaining voice work: `/voice` command activates `VoiceSession`, streaming TTS at ~300-500ms time-to-first-audio.

---

## Current Branch State (vibe-1771523298)

### Modified files (uncommitted)

- `src/agent/agent_loop.rs`
- `src/cli.rs`
- `src/config/schema.rs`
- `src/main.rs`
- `src/providers/mod.rs`
- `src/providers/openai_compat.rs`
- `src/repl/commands.rs`
- `src/repl/mod.rs`
- `src/server.rs`
- `src/tui.rs`
- `thoughts/ledgers/CONTINUITY_CLAUDE-voice-mode.md`

### Untracked new files

- `src/lms.rs` — LM Studio CLI wrapper
- ~~`src/mistralrs.rs`~~ — removed (mistral.rs rejected 2026-02-20)
- `src/providers/jit_gate.rs` — JIT semaphore gate
- `plans/local-model-reliability-tdd.md` — TDD plan for local reliability
- `plans/local-trio-strategy-2026-02-18.md` — Trio architecture strategy
- `plans/streaming-rewrite.md` — Incremental markdown renderer plan
- `skills/slm-agentic-reliability/` — SLM reliability skill files
- `thoughts/ledgers/CONTINUITY_CLAUDE-strategic-roadmap.md`
- `thoughts/shared/handoffs/channels-expansion/2026-02-19_17-58-31_lmstudio-trio-integration.md`

### Recent commits

```
0550f7d fix: remove oneOf from SpawnTool schema for local model compatibility
0fed6e6 fix: prevent false health watchdog restarts during LLM inference
3fd1c80 fix: eliminate SIGWINCH crash from double DefaultEditor signal handler
5f007e5 refactor: Sprint 3 "Make it right" — architecture improvements
32a0e80 refactor: Sprint 2 "Make it clean" — eliminate duplication across agent core
b757db9 feat: Sprint 1 "Make it fit" — context budget planner + DRY/safety fixes
```

---

## Active Work and Next Steps

### From Strategic Roadmap Ledger

Phase 0 (stabilization) is in progress. Phase 1 next:
1. **Phase 1.1**: Install and test mistral.rs as inference engine
2. **Phase 1.2**: Empirical trio testing — 3 VRAM combos vs reliability gates
3. **Phase 1.3**: Context diet (system prompt budget, tool result sanitization)
4. **Phase 1.4**: Evaluate UTCP and ZeptoClaw patterns

Hardware targets:
- RTX 3090 24GB (WSL2 on this machine)
- M4 32GB (macOS)

Candidate models (NanBeige 3B beats 32B on BFCL-V4 tool calling benchmark at 56.50 score).

### From LM Studio Handoff

The trio flow (NanBeige -> Qwen3 router -> Ministral specialist) has **never been tested end-to-end**. All individual pieces are code-complete but the integration is unverified.

### Outstanding Test Issues

1. `EXDEV` cross-device rename errors in observer/working_memory/reflector (need `copy+remove` fallback)
2. `test_web_search_no_api_key` fails when `BRAVE_API_KEY` set in env (not hermetic)
3. `/prov` alias not mapped (only `/p` is)
4. Socket-bind tests fail in restricted runtime (need capability probe)
5. Provider selection mismatch for `"none"` sentinel in `get_api_base()`

---

## Code Conventions

### Testing

- All tests are `#[cfg(test)] mod tests` inline within source files
- No integration tests, no CI/linter config
- Test commands: `cargo test`, `cargo test -- test_name`, `cargo test module::tests`
- Pure function extraction pattern: IO-coupled functions call `pick_X(data)` pure variants; tests hit the pure variant with synthetic data

### Naming

- Config structs: PascalCase with `#[serde(rename_all = "camelCase")]`
- Default functions follow pattern: `fn default_field_name() -> Type`
- Tool callbacks: `SpawnCallback`, `ListCallback`, `CancelCallback` etc. (type aliases in `spawn.rs`)
- Provider methods: `chat()` (blocking) and `chat_stream()` (SSE streaming)

### Error Handling

- `anyhow::Result` for propagation
- `thiserror` for custom error types
- Tool errors: return string prefixed with `"Error:"` — automatically mapped to `ToolExecutionResult::failure()` by `execute_with_result()`

### Async Runtime

Tokio with `features = ["full"]`. All async code uses `async-trait` for object-safe async trait methods.

### Module Organization

`src/agent/mod.rs` re-exports all agent submodules publicly. `src/agent/tools/mod.rs` re-exports tool types. Channel and provider modules follow the same pattern.

---

## Critical Gotchas

1. **`local:` prefix stripping**: Must happen in BOTH `chat()` and `chat_stream()` in `openai_compat.rs`. Fixing only one path is a common bug pattern.

2. **`apply_mode()` is a silent override**: Setting `trio.enabled=true` without `toolDelegation.mode="trio"` does nothing. `DelegationMode::Delegated` force-resets strict flags to false even if manually set.

3. **`/model` command must load fresh config before saving**: In-memory config save clobbers externally-edited fields. Fix is in `commands.rs` — load from disk, update only the model field, save back.

4. **Watchdog has 4 independent code paths**: Startup, `restart_watchdog()`, `handle_restart_requests()`, and the health check loop itself. All must be gated behind `has_remote_local` check or risk watchdog spam.

5. **Local message protocol**: After any `role: "tool"` message, always append a `role: "user"` continuation when `is_local=true`. Checked at build time via `SwappableCore.is_local`.

6. **oneOf removal from SpawnTool schema**: Local models reject `oneOf` JSON Schema constructs (commit `0550f7d`). Avoid complex schema combinators in tool definitions for local compatibility.

7. **`DelegationMode::Inline` vs `Delegated`**: `Inline` disables delegation entirely (main model calls tools directly). `Delegated` is the default (tools handed to runner model). `Trio` is strict separation with no tool calls from main.
