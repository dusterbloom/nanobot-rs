# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quality Gates (MANDATORY — check before writing ANY function)

Before writing or modifying any function, run through these gates. If a gate triggers, resolve it BEFORE writing the implementation. These gates exist because AI assistants (including you) systematically optimize for "make it work in this session" over "keep the codebase healthy across sessions." These gates counteract that bias.

**G1 — BOOL → ENUM:** Adding a `bool` parameter, field, or `let mut flag` in a loop?
- Ask: "Can this represent more than 2 downstream behaviors?" → Define an enum.
- Ask: "Does this flag change meaning during execution?" → It's a state machine. Name the states as enum variants.
- *Why:* A bool throws away information. Every downstream if/else is the code trying to recover what the bool discarded.

**G2 — SECOND USE → EXTRACT:** About to write logic that already exists elsewhere?
- The **second** occurrence triggers extraction, not the third. Search with `ast-grep` before writing.
- *Why:* Duplicated logic drifts. Two copies today become three divergent copies next month.

**G3 — NEST → GUARD:** Is the happy path more than 2 indentation levels deep?
- Invert conditions, use early returns: `let Some(x) = y else { return; };`
- The real work stays at the top level. Validation and error cases exit early.
- *Why:* Nesting hides logic. If the "real work" is at indent level 4, the reader has to hold 4 conditions in their head.

**G4 — GROW → SPLIT:** Is a function doing more than one nameable thing, or exceeding ~40 lines?
- If a block of code could have a name, make it a function with that name.
- *Why:* Long functions accrete because each session adds "just one more check." Split proactively.

**G5 — BRANCH → TYPE:** Adding an if/else that selects between two strategies or code paths?
- That branch should be a trait impl or enum variant dispatched via `match`, not an inline conditional.
- Especially true when the **same condition is tested in multiple places** across the codebase.
- *Why:* Inline branches distribute a single decision across many locations. A type centralizes it.

**Decision tree when in doubt:** Define the states → Name the transitions → Let `match` enforce exhaustive handling.

**Enforcement (two layers):**
- `cargo clippy` — handles G1 (bool params), G3 (cognitive complexity), G4 (function length) via `clippy.toml`
- `./quality-sentinel.sh` — catches what clippy misses: mutable bool flags (G1), else-if chains 3+ (G5)
- `./quality-sentinel.sh --full` — runs both together
- See `.planning/phases/06-state-driven-refactor/AUDIT.md` for the inventory of existing violations.

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

# ANE tests (require Apple Silicon, use serial execution to avoid hardware contention)
cargo build --features ane
cargo test --features ane --lib -- "ane_" --test-threads=1
cargo test --features ane --release --lib -- "bench_" --nocapture --test-threads=1  # Benchmarks
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

### Local Model Function Calling

For local models to use native function calling (tool_calls JSON), LM Studio must have `--jinja` enabled. This requires llama.cpp b8148 or newer. Without `--jinja`, models receive the `tools` parameter but can't generate proper `tool_calls` responses.

**Protocol modes:**
- `NativeToolCalls` — Model generates `tool_calls` JSON (requires LM Studio `--jinja`)
- `TextualReplay` — Tool calls rendered as `[I called: tool_name({...})]` in text; nanobot parses them back

Protocol is auto-selected by model capabilities. Override via:
```bash
NANOBOT_LOCAL_PROTOCOL_MODE=native nanobot agent -m "message"
```

**Model capability overrides** (`~/.nanobot/config.json`):
```json
{
  "modelCapabilities": {
    "qwen3.5-35b": {
      "sizeClass": "medium",
      "toolCalling": true,
      "maxReliableOutput": 4096,
      "scratchPadRounds": 8
    }
  }
}
```

Keys match as case-insensitive substrings against the model name. Override `sizeClass` to `"medium"` for MoE models whose names contain small-model markers (e.g., `"a3b"` in `qwen3.5-35b-a3b`).
