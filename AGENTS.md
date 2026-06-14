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

This project is indexed by GitNexus as **nanobot-rs** (8041 symbols, 21040 relationships, 300 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## When Debugging

1. `gitnexus_query({query: "<error or symptom>"})` — find execution flows related to the issue
2. `gitnexus_context({name: "<suspect function>"})` — see all callers, callees, and process participation
3. `READ gitnexus://repo/nanobot-rs/process/{processName}` — trace the full execution flow step by step
4. For regressions: `gitnexus_detect_changes({scope: "compare", base_ref: "main"})` — see what your branch changed

## When Refactoring

- **Renaming**: MUST use `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` first. Review the preview — graph edits are safe, text_search edits need manual review. Then run with `dry_run: false`.
- **Extracting/Splitting**: MUST run `gitnexus_context({name: "target"})` to see all incoming/outgoing refs, then `gitnexus_impact({target: "target", direction: "upstream"})` to find all external callers before moving code.
- After any refactor: run `gitnexus_detect_changes({scope: "all"})` to verify only expected files changed.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Tools Quick Reference

| Tool | When to use | Command |
|------|-------------|---------|
| `query` | Find code by concept | `gitnexus_query({query: "auth validation"})` |
| `context` | 360-degree view of one symbol | `gitnexus_context({name: "validateUser"})` |
| `impact` | Blast radius before editing | `gitnexus_impact({target: "X", direction: "upstream"})` |
| `detect_changes` | Pre-commit scope check | `gitnexus_detect_changes({scope: "staged"})` |
| `rename` | Safe multi-file rename | `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` |
| `cypher` | Custom graph queries | `gitnexus_cypher({query: "MATCH ..."})` |

## Impact Risk Levels

| Depth | Meaning | Action |
|-------|---------|--------|
| d=1 | WILL BREAK — direct callers/importers | MUST update these |
| d=2 | LIKELY AFFECTED — indirect deps | Should test |
| d=3 | MAY NEED TESTING — transitive | Test if critical path |

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/nanobot-rs/context` | Codebase overview, check index freshness |
| `gitnexus://repo/nanobot-rs/clusters` | All functional areas |
| `gitnexus://repo/nanobot-rs/processes` | All execution flows |
| `gitnexus://repo/nanobot-rs/process/{name}` | Step-by-step execution trace |

## Self-Check Before Finishing

Before completing any code modification task, verify:
1. `gitnexus_impact` was run for all modified symbols
2. No HIGH/CRITICAL risk warnings were ignored
3. `gitnexus_detect_changes()` confirms changes match expected scope
4. All d=1 (WILL BREAK) dependents were updated

## Keeping the Index Fresh

After committing code changes, the GitNexus index becomes stale. Re-run analyze to update it:

```bash
npx gitnexus analyze
```

If the index previously included embeddings, preserve them by adding `--embeddings`:

```bash
npx gitnexus analyze --embeddings
```

To check whether embeddings exist, inspect `.gitnexus/meta.json` — the `stats.embeddings` field shows the count (0 means no embeddings). **Running analyze without `--embeddings` will delete any previously generated embeddings.**

> Claude Code users: A PostToolUse hook handles this automatically after `git commit` and `git merge`.

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
