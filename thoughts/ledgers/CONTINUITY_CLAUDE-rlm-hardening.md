# Continuity Ledger: RLM Hardening & Delegation Safety

## Goal
Make the local delegation model (RLM) robust, safe, and performant for both local (Ministral-3B) and cloud (Claude Opus) modes.

## Constraints
- Local LLM protocol: conversations MUST end with `role: "user"` for llama-server
- Delegation model has 4K context — all inputs must fit
- Short-circuit threshold: 200 chars (skip LLM for trivial outputs)
- Tool filtering: delegation model can ONLY use tools from the initial request
- ExecTool deny patterns must block all known destructive commands

## Key Decisions
- **Ministral-3 first** over Ministral-8B: smaller = faster TTFT, fits GPU better [2026-02-14]
- **Dynamic VRAM GPU allocation**: compute layers from available VRAM at spawn, don't hardcode [2026-02-14]
- **dom_smoothie over scraper**: Mozilla Readability port extracts content better [2026-02-14]
- **Short-circuit at 200 chars**: trivial outputs skip delegation LLM entirely [2026-02-14]
- **Tool filtering**: delegation only sees tools from initial request (prompt injection defense) [2026-02-14]
- **Iteration scaling**: cloud models (1M ctx) get up to 50 iters, local capped at 15 [2026-02-14]

## State
- Done:
  - [x] Phase 1: Truncate tool results for delegation model (max_tool_result_chars, 6K cap)
  - [x] Phase 2: Replace scraper with dom_smoothie for HTML extraction
  - [x] Phase 3: Antifragile delegation (health tracking, auto-fallback, re-probe, /restart)
  - [x] Phase 4: Dynamic VRAM-based GPU allocation for delegation server
  - [x] Phase 5: Model preferences (smallest first) + loop detection (seen_calls HashSet)
  - [x] Phase 6: Short-circuit for trivial outputs + main model instructions passthrough
  - [x] Phase 7: Tool filtering (delegation model restricted to initial tools only)
  - [x] Phase 8: Stronger deny patterns (rm variants, find -delete, shred, sudo)
  - [x] Phase 9: Iteration scaling (cloud gets more iters based on context size)
- Now: Testing & validation
- Next: User testing in real scenarios

## Open Questions
- UNCONFIRMED: Does the delegation model handle multi-turn web scraping well?
- Hallucination on repetitive content (yes flood) — need better detection?
- Over-exploration (find test hitting max_iter) — summarize-first prompt tuning?

## Working Set
- Files: `src/agent/tool_runner.rs`, `src/agent/agent_loop.rs`, `src/agent/tools/shell.rs`, `src/cli.rs`, `src/config/schema.rs`, `src/server.rs`, `src/agent/tools/web.rs`, `src/main.rs`, `src/repl.rs`
- Branch: main
- Test: `cargo test`
- Config: `~/.nanobot/config.json` (toolDelegation section)
