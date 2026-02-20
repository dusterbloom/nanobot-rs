# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**nanobot** - A lightweight personal AI assistant framework in Rust, ported from [nanobot](https://github.com/HKUDS/nanobot) (Python, MIT). Binary name: `nanobot`.

## Build & Test

```bash
cargo build --release        # Release build
cargo build                  # Debug build
cargo test                   # Run all tests (unit tests are inline in each module)
cargo test -- test_name      # Run a single test by name
cargo test module::tests     # Run tests for a specific module
RUST_LOG=debug cargo run -- agent -m "Hello"  # Run with debug logging
```

No CI, no linter config, no integration tests. All tests are `#[cfg(test)] mod tests` inside their source files.

## Architecture

### Message Flow

```
Channels (Telegram/WhatsApp/Feishu)
        │
        ▼
   InboundMessage ──► AgentLoop ──► LLM Provider (OpenAI-compat API)
   (bus/events.rs)    │    ▲            │
                      │    │            ▼
                      │    └──── tool calls ──► ToolRegistry
                      │                            │
                      ▼                            ▼
               OutboundMessage          Tool implementations
               (bus/events.rs)          (agent/tools/*.rs)
```

The `AgentLoop` (`src/agent/agent_loop.rs`) is the core: it receives messages via `mpsc` channels, builds context (system prompt + history), calls the LLM, executes tool calls in a loop (up to `max_tool_iterations`), and emits responses. Two entry points: `run()` for gateway mode (consuming from bus), `process_direct()` for CLI mode.

### Key Modules

- **`agent/`** - Agent core: loop, context builder, memory, skills, subagents, tools
- **`agent/tools/`** - Tool trait (`base.rs`) + registry (`registry.rs`) + implementations (filesystem, shell, web, message, spawn, cron)
- **`providers/`** - Single `OpenAICompatProvider` that talks to all providers via OpenAI-compatible chat completions API
- **`config/`** - JSON config schema (`schema.rs`) + loader. Config lives at `~/.nanobot/config.json`
- **`channels/`** - Chat channel adapters (Telegram polling, WhatsApp WebSocket bridge, Feishu WebSocket)
- **`bus/`** - `InboundMessage`/`OutboundMessage` event types, message queue
- **`session/`** - JSONL-based session persistence in `~/.nanobot/sessions/`
- **`cron/`** - Scheduled job system with interval and cron expression support
- **`heartbeat/`** - Periodic heartbeat service

### Config & Provider Selection

`Config::get_api_key()` and `Config::get_api_base()` use the same priority order to select the active provider: OpenRouter > DeepSeek > Anthropic > OpenAI > Gemini > Zhipu > ZhipuCoding > Groq > vLLM. The first non-empty API key wins.

Most providers are accessed through `OpenAICompatProvider` which sends requests to the provider-specific base URL. `AnthropicProvider` speaks the native Anthropic Messages API for OAuth/Claude Max flows. Config JSON uses camelCase keys (`#[serde(rename_all = "camelCase")]`).

### Tool System

Tools implement the `Tool` trait (`agent/tools/base.rs`): `name()`, `description()`, `parameters()` (JSON Schema), `execute()`. They're registered in `ToolRegistry` and exposed to the LLM as OpenAI function-calling schema. Tools needing runtime state (MessageTool, SpawnTool, CronScheduleTool) are wrapped in `Arc` with proxy structs in `agent_loop.rs`.

### Context & Memory

`ContextBuilder` assembles the system prompt from: identity text, bootstrap files (`AGENTS.md`, `SOUL.md`, `USER.md`, `TOOLS.md`, `IDENTITY.md` in workspace), memory (`MEMORY.md` only — daily notes and learnings are excluded from the system prompt), and skills. The workspace defaults to `~/.nanobot/workspace/`.

### Skills

Markdown files at `{workspace}/skills/{name}/SKILL.md` with optional YAML frontmatter (description, requires, always). Workspace skills shadow built-in skills. Skills with `always: true` are loaded into every prompt; others appear as summaries the agent can read on demand.

### Local LLM Protocol Constraints

Local models (via LM Studio) have stricter message protocol than cloud APIs:

- Conversations **MUST** end with a `role: "user"` message
- Assistant message prefill is **NOT** supported
- Tool result messages (`role: "tool"`) cannot be the last message

After adding tool results to any message array, always append a user continuation before calling the LLM:

```rust
messages.push(json!({
    "role": "user",
    "content": "Based on the tool results above, continue with the task."
}));
```

**Affected code paths:** `tool_runner.rs`, `subagent.rs`, `agent_loop.rs` (inline path, conditional on `is_local`). Any new code path that builds message arrays with tool results must follow this pattern.
