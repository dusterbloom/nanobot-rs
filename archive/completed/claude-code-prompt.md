# Nanobot Upgrade Sprint - Claude Code Prompt

## Context

You are upgrading nanobot, a Rust-based AI assistant at `/home/peppi/Dev/nanobot/`.
The codebase is ~22,400 lines of Rust. Config is at `~/.nanobot/config.json`.

The goal: remove all limitations that prevent nanobot from being an effective research assistant for deep, multi-session analytical work across a 34,000+ file research corpus.

## Current Limitations (ordered by impact)

### L1: No Web Search (CRITICAL)
- Brave API key is empty in config
- **Fix**: Add a fallback search provider that doesn't require an API key
- Options: DuckDuckGo HTML scraping, SearXNG local instance, or Google scraping
- File: `src/agent/tools/web.rs`
- **Test**: `cargo test` + manual search query returns results

### L2: No Semantic Memory / Retrieval (CRITICAL)
- The agent has no vector/embedding-based retrieval
- It can only read files it knows about by path
- For research work across 34K files, it needs semantic search
- **Fix**: Add a local embedding + vector search tool
- Options: 
  - Use `fastembed-rs` crate for local embeddings (no API needed)
  - Store vectors in a simple HNSW index (use `instant-distance` or `hnsw` crate)
  - Index workspace + configurable external paths
  - New tool: `semantic_search(query, scope, top_k)` 
- Files: new `src/agent/tools/semantic.rs`, update `src/agent/tools/mod.rs`
- **Test**: Index a directory, query it, get relevant file paths + snippets

### L3: maxToolIterations too low (HIGH)
- Currently 20, too few for deep research sessions
- **Fix**: Make it configurable per-session, default to 50
- Add a `/iterations N` command in the REPL
- File: `src/repl.rs`, `src/config/mod.rs`
- **Test**: Verify iteration limit respects runtime override

### L4: maxContextTokens not maximized (HIGH)
- Set to 128K, but Opus 4.6 may support up to 1M
- **Fix**: Add model-specific defaults in provider config
- If model is `claude-opus-4-6`, default to 1000000
- File: `src/config/mod.rs`, `src/providers/`
- **Test**: Verify correct context size selected per model

### L5: Temperature not task-adaptive (MEDIUM)
- Fixed at 0.7 for everything
- **Fix**: Add `/temperature N` REPL command for runtime override
- Also add a `temperature` field to subagent spawning so analytical subagents use 0.2
- Files: `src/repl.rs`, `src/agent/subagent.rs`
- **Test**: Verify temperature override propagates to API calls

### L6: Subagent iteration limit too low (MEDIUM)
- Hardcoded to 15 in `src/agent/subagent.rs` (line: `const MAX_SUBAGENT_ITERATIONS: u32 = 15`)
- **Fix**: Make configurable, default to 30
- File: `src/agent/subagent.rs`
- **Test**: Verify subagent respects configured limit

### L7: No research workspace indexing (MEDIUM)
- Agent can't efficiently navigate large file trees
- **Fix**: Add a `tree` tool that shows directory structure with file sizes and types
- Smarter than `list_dir` - recursive, filterable, with token-budget-aware truncation
- File: new tool or enhance `src/agent/tools/filesystem.rs`
- **Test**: `tree("/path", depth=3, filter="*.md")` returns structured output

### L8: MEMORY.md is a single flat file (LOW)
- All long-term memory in one file, no structure
- **Fix**: Support sectioned memory with headers that the reflector maintains
- Categories: User, Projects, Preferences, Research, Decisions
- File: `src/agent/reflector.rs`, `src/agent/memory.rs`
- **Test**: Reflector produces categorized output

### L9: No exec timeout configurability at runtime (LOW)
- Fixed at 60s in config
- **Fix**: Add per-command timeout via tool parameter
- File: `src/agent/tools/shell.rs`
- **Test**: Long-running command respects custom timeout

## Architecture Rules

1. **TDD Red-Green-Refactor**: Write failing test first, then implement, then clean up
2. **No breaking changes**: All existing tests must pass
3. **Integration tests**: Each feature gets an integration test in `tests/`
4. **Parallel work**: L1-L2 are independent. L3-L6 are independent. Batch accordingly.
5. **Rust idioms**: Use `anyhow::Result`, `tracing` for logs, `async_trait` where needed
6. **No new external services**: Everything must work offline/locally (except Brave if key provided)

## Execution Plan

### Phase 1: Quick Wins (parallel)
- [ ] L3: Bump maxToolIterations, add REPL command
- [ ] L5: Add temperature REPL command  
- [ ] L6: Make subagent iterations configurable
- [ ] L9: Per-command exec timeout

### Phase 2: Core Upgrades (parallel)
- [ ] L1: Fallback web search (DuckDuckGo scraping)
- [ ] L7: Enhanced tree/filesystem tool
- [ ] L4: Model-specific context size defaults

### Phase 3: Semantic Memory (sequential, depends on Phase 1)
- [ ] L2: Local embeddings + vector search
  - Step 1: Add fastembed-rs dependency, test embedding generation
  - Step 2: Build HNSW index, test insert/query
  - Step 3: Wire as tool, test end-to-end
  - Step 4: Background indexing of configured paths

### Phase 4: Memory Refinement
- [ ] L8: Structured MEMORY.md with reflector categories

## Validation

After all phases:
```bash
cargo test
cargo build --release
# Manual: start nanobot, verify:
# 1. /iterations 50 works
# 2. /temperature 0.3 works  
# 3. web_search returns results without Brave key
# 4. semantic_search finds relevant files
# 5. tree tool works on large directories
# 6. subagent runs with higher iteration limit
```

## Important Files

```
src/
  agent/
    agent_loop.rs    # Main agent loop
    compaction.rs     # Context compaction
    context.rs        # Prompt assembly  
    learning.rs       # Tool outcome tracking
    memory.rs         # MEMORY.md read/write
    observer.rs       # Cross-session observations
    reflector.rs      # Background memory consolidation
    skills.rs         # Skill loading
    subagent.rs       # Background task agents
    token_budget.rs   # Token counting/trimming
    tools/
      base.rs         # Tool trait
      filesystem.rs   # read/write/list/edit
      shell.rs        # exec
      web.rs          # web_search, web_fetch
      spawn.rs        # spawn subagent
      mod.rs          # Tool registry
  config/             # Config loading
  repl.rs             # Interactive REPL
  cli.rs              # CLI entry point
```

Config: `~/.nanobot/config.json`
Tests: `tests/` directory + inline `#[cfg(test)]` modules
