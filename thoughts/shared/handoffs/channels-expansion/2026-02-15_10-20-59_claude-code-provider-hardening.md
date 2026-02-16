---
date: 2026-02-15T10:20:59+0100
session_name: channels-expansion
researcher: peppi
git_commit: 6d04749
branch: main
repository: nanobot
topic: "ClaudeCodeProvider Hardening & Multi-Provider Routing Fixes"
tags: [claude-code-provider, multi-provider, streaming, tool-calling, subagent-routing]
status: complete
last_updated: 2026-02-15
last_updated_by: peppi
type: implementation_strategy
root_span_id: ""
turn_span_id: ""
---

# Handoff: ClaudeCodeProvider band-aids applied, root cause investigation needed

## Task(s)

1. **Debug multi-provider subagent routing failures** (COMPLETED)
   - Diagnosed why `xai/`, `zhipu/`, `openrouter/` prefixed models failed with "claude CLI exited with 1"
   - Root cause: missing routing entries in `resolve_provider_for_model` in both `subagent.rs` and `schema.rs`
   - Added `zhipu/` and `openrouter/` routing entries (user declined adding `xai/` — no config field exists)

2. **ClaudeCodeProvider output quality improvements** (COMPLETED — but band-aids, not root cause fixes)
   - Added robust JSON parsing (`try_parse_json`) for `<tool_call>` blocks with raw newlines
   - Added CLI artifact stripping (`strip_cli_artifacts`) for "I ran out of tool iterations" etc.
   - Added `ToolCallFilter` for streaming path — suppresses `<tool_call>` blocks AND CLI artifact strings from live streaming output
   - Tightened tool instruction prompt to reduce model fluff

3. **Root cause investigation of ClaudeCodeProvider architecture** (DISCUSSED — NOT STARTED)
   - User correctly identified that all above changes are band-aids
   - The fundamental architecture (`claude -p --max-turns 1 --tools ""`) is the wrong approach
   - **This is the main action item for the next session**

## Critical References

- `src/providers/claude_code.rs` — the entire ClaudeCodeProvider, all changes are here
- `src/agent/subagent.rs:362-398` — `resolve_provider_for_model` routing table
- `src/config/schema.rs:777-800` — `Config::resolve_provider_for_model` routing table (duplicate)

## Recent changes

All changes are uncommitted on `main`:

- `src/providers/claude_code.rs` — Major changes:
  - `try_parse_json()` (new, ~line 175): fallback JSON parser that escapes raw control chars inside string values
  - `strip_cli_artifacts()` (refactored, ~line 230): uses shared `CLI_ARTIFACT_PATTERNS` constant
  - `CLI_ARTIFACT_PATTERNS` (new, ~line 137): shared list of CLI boilerplate strings
  - `ToolCallFilter` (new, ~line 148): stream filter that suppresses `<tool_call>` blocks AND CLI artifact strings with partial-prefix holdback
  - `format_tool_instructions()` (updated, ~line 140): tighter prompt — "valid JSON on single line", "escape newlines", "minimal surrounding text"
  - `parse_tool_calls()` (updated): uses `try_parse_json` instead of raw `serde_json::from_str`, calls `strip_cli_artifacts` on clean text
  - Streaming path in `chat_stream()`: wired through `ToolCallFilter`
  - 42 tests total (12 new)

- `src/agent/subagent.rs:369-370` — Added `zhipu/` and `openrouter/` to subagent routing table
- `src/config/schema.rs:785-786` — Added `zhipu/` and `openrouter/` to config routing table

## Learnings

### The fundamental architecture problem with ClaudeCodeProvider

The `ClaudeCodeProvider` uses `claude -p --max-turns 1 --tools ""` to turn the Claude CLI into a raw LLM backend. This has three structural problems that no amount of filtering can fix:

1. **Prompt-engineered tool calling** — Tool schemas are injected into the system prompt and tool calls are parsed from `<tool_call>` XML blocks in text. This is fundamentally worse than native function calling: the model mixes prose with tool calls, can't return `content: null`, generates explanatory fluff, and complex JSON breaks parsing.

2. **CLI boilerplate injection** — Claude CLI adds its own messages ("I ran out of tool iterations") when `--max-turns 1` terminates with pending work. These are CLI-level messages, not model output. We strip them but they still flash during streaming.

3. **Cold process spawn per turn** — Each agent loop iteration spawns a fresh `claude` process, serializes the entire conversation as flat text (`User: ... Assistant: ... Tool Result [...]: ...`), and pipes it via stdin. No conversation persistence, no semantic structure.

### Provider routing gap pattern

Both `SubagentManager::resolve_provider_for_model` (subagent.rs) and `Config::resolve_provider_for_model` (schema.rs) maintain DUPLICATE routing tables. When a new provider is added to `ProvidersConfig`, both tables must be updated. Currently missing from both: `vllm/` (uses api_base not api_key). Missing from ProvidersConfig entirely: `xai/` (Grok).

### ToolCallFilter holdback trade-off

The stream filter holds back text matching prefixes of suppressed patterns (e.g., holding back "I ran out of tool" until next delta confirms it's an artifact). The longest artifact is ~70 chars. Since "I" is a prefix, single-character holdbacks are frequent but released on next delta (~100ms delay). Acceptable trade-off.

## Post-Mortem (Required for Artifact Index)

### What Worked
- The `try_parse_json` fallback parser is solid — char-by-char state machine that correctly handles `\"`, `\\`, raw newlines, tabs
- `ToolCallFilter` with partial-prefix holdback is architecturally clean and handles split-across-deltas correctly
- Test-driven approach: 12 new tests caught edge cases (heredoc f-strings, split deltas, partial tag false positives)

### What Failed
- Tried: Prompt engineering ("keep surrounding text minimal") to reduce fluff → Marginally effective at best, model still narrates tool calls
- Tried: Filtering only `<tool_call>` blocks from stream without CLI artifacts → User still saw "I ran out of tool iterations"
- Overall approach: Band-aids on an architecture mismatch. The user correctly called this out.

### Key Decisions
- Decision: Added zhipu/openrouter routing but NOT xai
  - Reason: xai has no `ProvidersConfig` field — would need schema change. User only asked for zhipu.
- Decision: Used shared `CLI_ARTIFACT_PATTERNS` constant between `strip_cli_artifacts` and `ToolCallFilter`
  - Reason: Single source of truth, both paths suppress the same strings
- Decision: Did NOT try to strip all model content when tool calls are present
  - Reason: Would lose legitimate content like "I'll check the file for errors"

## Artifacts

- `src/providers/claude_code.rs` — All provider changes (try_parse_json, ToolCallFilter, strip_cli_artifacts, 42 tests)
- `src/agent/subagent.rs:369-370` — zhipu/openrouter routing
- `src/config/schema.rs:785-786` — zhipu/openrouter routing

## Action Items & Next Steps

### Priority 1: Root Cause Investigation (NEXT SESSION)

Investigate whether the `ClaudeCodeProvider` architecture can be fundamentally improved:

1. **Can we use Claude CLI's native tool calling?** Instead of `--tools ""` (disable all tools), can we define custom tools that map to nanobot's tool registry? This would give us native function calling through the CLI.

2. **Can we use a persistent `claude` session?** Instead of spawning a fresh process per turn with `-p`, can we keep a session alive and pipe messages in/out? Check `claude --help` for session/conversation modes.

3. **Does Max plan expose an API endpoint?** The Claude CLI must call some API internally. Can we use that same endpoint directly with `OpenAICompatProvider`? Check if Max plan includes API access.

4. **Should we use Claude CLI's MCP (Model Context Protocol)?** Claude Code supports MCP servers. Could nanobot's tools be exposed as an MCP server that Claude CLI connects to natively?

### Priority 2: Dedup the routing tables

`SubagentManager::resolve_provider_for_model` (subagent.rs:362) and `Config::resolve_provider_for_model` (schema.rs:777) are duplicate code. Refactor to a single source of truth.

### Priority 3: Add `xai` provider support

Add `xai` field to `ProvidersConfig` in schema.rs, then add `xai/` routing entry to both routing tables.

## Other Notes

- The changes are **uncommitted** on main. Use `/commit` to save them.
- All 810 tests pass. Release build is clean.
- The `ToolCallFilter` is only used in the streaming path (`chat_stream`). The non-streaming path (`chat`) uses `parse_tool_calls` → `strip_cli_artifacts` which handles the same cleanup for the final result.
- The duplicate display issue the user saw (message appearing twice) was likely from running the old binary — needs verification after restart.
