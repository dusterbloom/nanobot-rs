# Continuity Ledger: Claude Max Provider

## Goal
Zero-cost LLM inference via Claude Max subscription ($200/mo) with full fidelity — native function calling, streaming, no per-token charges. Replace the CLI-subprocess `ClaudeCodeProvider` with direct API access using OAuth tokens.

## Constraints
- OAuth tokens from `~/.claude/.credentials.json` (Claude CLI stores them)
- Anthropic Messages API requires specific identity headers for OAuth (discovered from OpenClaw source)
- Must work transparently with existing `LLMProvider` trait
- Subagents must also route through OAuth when no API key is configured

## Key Decisions
- **Direct API over proxy**: Tried ClaudeCodeProvider (CLI subprocess, too slow), npm proxy (requires Node), Rust proxy (still CLI under the hood). Direct API with OAuth identity headers is the right approach. [2026-02-15]
- **OAuth identity headers**: `anthropic-beta: claude-code-20250219,oauth-2025-04-20,...`, `user-agent: claude-cli/2.1.2`, `x-app: cli`, system prompt must include Claude Code identity string. Discovered by reading OpenClaw source at `@mariozechner/pi-ai/dist/providers/anthropic.js`. [2026-02-15]
- **AnthropicProvider**: Native Messages API translation layer (OpenAI format ↔ Anthropic format) rather than using OpenAI-compat endpoint (which may not accept OAuth). [2026-02-15]
- **Token refresh at startup only**: Option A from plan — refresh on provider creation, don't wrap in auto-refresh proxy. Sessions rarely last 8h. [2026-02-15]
- **RAW display prefix**: Subagent result blocks bypass termimad markdown rendering via `\x1b[RAW]` prefix to avoid gray background artifacts. [2026-02-15]

## State
- Done:
  - [x] Phase 1: OAuth token manager (`src/providers/oauth.rs`)
  - [x] Phase 2: Config + provider wiring in `cli.rs` (`claude-max` prefix + auto-detect)
  - [x] Phase 3: AnthropicProvider with OAuth identity headers (`src/providers/anthropic.rs`)
  - [x] Phase 5: Routing dedup (`PROVIDER_PREFIXES` in `schema.rs`, shared by `subagent.rs`)
  - [x] Fix: Model ID normalization — sonnet→4.5, haiku date→20251001 (all 3 normalizers)
  - [x] Fix: UTF-8 safe string truncation — `floor_char_boundary()` helper, fixed 8 panic sites
  - [x] Fix: Subagent display styling — bypass markdown rendering for ANSI-formatted blocks
  - [x] Feat: Spawn tool list/cancel actions for subagent management
- Now: [DONE] — All core phases complete
- Remaining:
  - [x] Phase 4: Deleted `ClaudeCodeProvider` (`src/providers/claude_code.rs`) — zero external refs [2026-02-16]
  - [ ] Token auto-refresh mid-session (Option B from plan — needed if sessions exceed 8h)
  - [ ] Investigate delegation model "returned no summary" warning
  - [ ] End-to-end test: tool calling via OAuth path
  - [ ] End-to-end test: streaming fidelity

## Open Questions
- UNCONFIRMED: Does OAuth token work with extended thinking / 1M context beta headers?
- UNCONFIRMED: Rate limits on Max plan — `rateLimitTier: "default_claude_max_20x"` suggests generous but unknown thresholds
- UNCONFIRMED: Will identity headers / OAuth beta string need updating as Claude CLI versions bump?
- Delegation model warning: "returned no summary — marking provider unhealthy" — may be subagent prompt issue, not provider

## Working Set
- Files: `src/providers/{anthropic,oauth,openai_compat}.rs`, `src/cli.rs`, `src/agent/{subagent,agent_loop,agent_profiles}.rs`, `src/repl/commands.rs`, `src/utils/helpers.rs`
- Branch: main
- Commit: `a26acd7`
- Build: `cargo build --release`
- Test: `cargo test` (844 pass, 0 fail)
- Plan: `~/.claude/plans/velvety-cuddling-galaxy.md`
