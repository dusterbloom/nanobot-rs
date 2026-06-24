# Agent Notes

`nanobot` is a personal AI assistant framework. It receives messages on channels
(Telegram/WhatsApp/Feishu/Email), runs an agent loop against an OpenAI-compatible
LLM, executes tools, and replies. The goal is a small, readable, fast Rust
codebase with one well-tuned hot path per concern.

## Goals

- Keep the production path as: channel → agent_loop → provider → tools → reply.
- One way to do each thing. No protocol-mode flags, no router-fallback fallbacks,
  no parallel "experimental" pipelines living next to the real one.
- Local (LM Studio) and cloud (Anthropic/OpenAI/etc) are two branches of one
  `RuntimeMode` enum, not two parallel codebases.
- Long sessions practical via JSONL session files + on-disk `MEMORY.md`.

## Quality Rules

- Comment important agent code where the LLM-protocol contract, tool-call bytes,
  message-array invariants, or session-replay rules are not obvious.
- Prefer comments beside the implementation over separate design documents.
- Keep public APIs narrow. `src/agent/mod.rs` should re-export ~20 things, not 70.
- Do not add permanent semantic variants behind flags. Diagnostic switches are
  fine when they validate the one release path.
- The second occurrence of a piece of logic triggers extraction, not the third.
- A `bool` parameter that selects between two downstream behaviors is an enum.
- Failure modes are solved inside the hot path with comments, not extracted into
  `_gate` / `_guard` / `_hygiene` / `anti_*` modules.
- Do not introduce a new module to "organize" code that already fits in 500 lines.

## Safety

- Shell and code-execution tools must check deny-patterns before running.
- File tools must validate paths and honor workspace restrictions when configured.
- Web-fetched content must be marked tainted before reaching exec tools.

## Layout

- `src/main.rs`, `src/cli/`, `src/repl/`: command-line and REPL entrypoints.
- `src/agent/agent_loop.rs`: the message-processing loop. Builds context, calls
  the LLM, runs tools, emits reply. The hot path.
- `src/agent/tools/`: one file per tool. Each implements the `Tool` trait.
- `src/agent/context.rs`: system prompt assembly. Identity + bootstrap files +
  memory + skills.
- `src/agent/skills.rs`, `src/agent/memory.rs`: workspace skills and `MEMORY.md`.
- `src/providers/`: OpenAI-compatible HTTP client (covers 9 providers) plus
  Anthropic-native client.
- `src/channels/`: chat adapters (one file per channel).
- `src/bus/`, `src/session/`: message types, JSONL session persistence.
- `src/config/`: JSON config schema and loader.

## Testing

Use `cargo build` for build validation. Use `cargo test` for unit/regression
tests. Use `cargo run --release --bin nanobot-bench` for speed regressions when
changing the agent loop, provider client, or context builder. See
`CONTRIBUTING.md` for the correctness and speed regression tracks.

## Code Style

Rust 2021. `snake_case` fns, `PascalCase` types, `SCREAMING_SNAKE` consts.
`serde` `camelCase` for JSON config. `anyhow::Result` at app layer; tools return
`String` (prefix errors with `"Error: "`). Async via `async_trait` + `tokio`.
Shared state via `Arc<Mutex<_>>`.

```rust
use std::collections::HashMap;
use std::path::PathBuf;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::agent::tools::base::Tool;
```

## Provider Selection

Config chooses provider in this order (first non-empty API key wins):
OpenRouter > DeepSeek > Anthropic > OpenAI > Gemini > Zhipu > Groq > vLLM.

All providers use OpenAI-compatible chat completions API via
`OpenAICompatProvider`, except Anthropic (native Messages API).

## Configuration

- Config: `~/.nanobot/config.json`
- Sessions: `~/.nanobot/sessions/`
- Workspace (skills, memory): `~/.nanobot/workspace/`

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **nanobot-rs** (11158 symbols, 30909 relationships, 300 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> Index stale? Run `node .gitnexus/run.cjs analyze` from the project root — it auto-selects an available runner. No `.gitnexus/run.cjs` yet? `npx gitnexus analyze` (npm 11 crash → `npm i -g gitnexus`; #1939).

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows. For regression review, compare against the default branch: `detect_changes({scope: "compare", base_ref: "main"})`.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `context({name: "symbolName"})`.

## Never Do

- NEVER edit a function, class, or method without first running `impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `rename` which understands the call graph.
- NEVER commit changes without running `detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/nanobot-rs/context` | Codebase overview, check index freshness |
| `gitnexus://repo/nanobot-rs/clusters` | All functional areas |
| `gitnexus://repo/nanobot-rs/processes` | All execution flows |
| `gitnexus://repo/nanobot-rs/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
