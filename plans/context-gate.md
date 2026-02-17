# Context Gate: Intelligent Content Management for LLM Agents

**Status:** Proposed
**Date:** 2026-02-17
**Author:** Peppi + Nanobot

## Problem

Tool output truncation is currently a dumb char limit applied uniformly regardless of:
- The model's context window (4K local vs 200K cloud)
- How much context is already consumed
- Whether the agent actually needs the full content

This causes:
- Cloud models (opus, sonnet) unable to read files they have plenty of room for
- Local models getting truncated output with no way to access the rest
- Agents wasting inference tokens on meta-reasoning about truncation
- Information loss that breaks multi-step tasks

## Design Principle

**The infrastructure is invisible. The agent thinks in tasks, not tokens.**

The agent should never have to:
- Use special markers like `[VERBATIM]` or `[FULL]`
- Reason about whether content will fit
- Know that summarization happened (unless it needs to drill deeper)
- Use different APIs based on model size

Every agent -- from a 3B local model to opus -- uses the same tool interface and always gets useful results. The quality scales with the context budget, but the interface never changes.

## Architecture

### Components

```
ContextBudget          -- invisible resource manager
  |
  v
ContentGate            -- single entry point for ALL content entering context
  |                       (tool outputs, memory injection, file reads)
  |-- fits? --> pass raw
  |-- tight? --> pass raw but flag for compaction soon
  |-- doesn't fit? --> Compactor produces a "briefing"
  |                    OutputCache stores full content
  |                    Briefing includes natural navigation hints
  v
Agent sees natural content, never infrastructure
```

### 1. ContextBudget

Single responsibility: track token budget, answer "does this fit?"

```rust
pub struct ContextBudget {
    /// Model's max context window in tokens
    max_tokens: usize,
    /// Tokens currently consumed (system prompt + messages + tool results)
    used_tokens: usize,
    /// Fraction reserved for model output generation (default: 0.20)
    output_reserve: f32,
}

impl ContextBudget {
    /// Tokens available for new content
    pub fn available(&self) -> usize {
        let ceiling = (self.max_tokens as f32 * (1.0 - self.output_reserve)) as usize;
        ceiling.saturating_sub(self.used_tokens)
    }

    /// Record tokens consumed
    pub fn consume(&mut self, tokens: usize) {
        self.used_tokens += tokens;
    }

    /// Reset (e.g., after compaction)
    pub fn reset(&mut self, used: usize) {
        self.used_tokens = used;
    }
}
```

No I/O. No side effects. Pure accounting.

### 2. OutputCache

Single responsibility: persist full tool outputs to disk, serve them back by reference or range.

```rust
pub struct OutputCache {
    cache_dir: PathBuf,  // e.g., ~/.nanobot/cache/tool_outputs/
}

impl OutputCache {
    /// Store content, return a stable reference ID
    pub fn store(&self, content: &str) -> CacheRef;

    /// Retrieve full content by reference
    pub fn get(&self, ref_id: &CacheRef) -> Option<String>;

    /// Retrieve a line range (1-indexed, inclusive)
    pub fn get_lines(&self, ref_id: &CacheRef, start: usize, end: usize) -> Option<String>;

    /// Cleanup old entries (e.g., older than 24h)
    pub fn gc(&self, max_age: Duration);
}
```

### 3. ContentGate

Single responsibility: decide how content enters the agent's context. This is the ONLY place where content sizing decisions happen.

```rust
pub enum GateResult {
    /// Content fits, pass through unchanged
    Raw(String),
    /// Content was summarized; full version cached
    Briefing {
        summary: String,
        cache_ref: CacheRef,
        original_size: usize,
    },
}

pub struct ContentGate {
    budget: ContextBudget,
    cache: OutputCache,
    // Uses the existing Compactor for summarization
}

impl ContentGate {
    /// Gate any content entering the agent's context
    pub fn admit(&mut self, content: &str, compactor: &Compactor) -> GateResult {
        let tokens = estimate_tokens(content);
        if tokens <= self.budget.available() {
            self.budget.consume(tokens);
            GateResult::Raw(content.into())
        } else {
            let cache_ref = self.cache.store(content);
            let target_tokens = self.budget.available() / 2;
            let summary = compactor.summarize_for_briefing(content, target_tokens);
            let summary_tokens = estimate_tokens(&summary);
            self.budget.consume(summary_tokens);
            GateResult::Briefing {
                summary,
                cache_ref,
                original_size: content.len(),
            }
        }
    }
}
```

### 4. Compactor (existing, extended)

The existing compactor already summarizes conversations. We extend it with one new method:

```rust
impl Compactor {
    /// Produce a structural briefing of content (not a conversation summary)
    /// Output format: structure map + key definitions + navigation hints
    pub fn summarize_for_briefing(&self, content: &str, target_tokens: usize) -> String;
}
```

The briefing prompt instructs the compaction model to produce a **structural map**, not a lossy summary.

## Briefing Format

When content is too large for the context, the agent receives a briefing like this:

```
# big_file.rs (847 lines, 24KB)

## Structure
- Lines 1-45: Imports and type definitions (Config, ParseError, Runtime)
- Lines 46-200: Parser implementation
  - parse_config() at line 52
  - parse_args() at line 98
  - validate() at line 156
- Lines 201-400: Runtime engine
  - Engine::new() at line 205
  - Engine::run() at line 280
- Lines 401-847: Tests (12 test functions)

## Key Signatures
- pub fn parse_config(path: &Path) -> Result<Config, ParseError>
- pub struct Config { model: String, context_size: usize, ... }
- pub struct Engine { config: Config, state: State }

To inspect a section, use: read_file("big_file.rs", lines="46:200")
```

This is:
- **Navigable** -- the agent knows what's where and can drill in
- **Natural** -- reads like documentation, not infrastructure noise
- **Actionable** -- includes the exact tool call to get more detail
- **Model-agnostic** -- useful for both 3B and 200K-context models

## Integration Points

### Tool Runner

Currently applies hardcoded truncation. Change to:

```rust
// Before (current):
let output = tool.execute(input)?;
let truncated = truncate(output, MAX_CHARS);
return truncated;

// After:
let output = tool.execute(input)?;
return content_gate.admit(&output, &compactor);
```

One line change in the tool runner. All intelligence lives in ContentGate.

### Agent Loop

Currently assembles messages with no budget awareness. Change to:

```rust
// Before:
messages.push(tool_result);

// After:
let gated = content_gate.admit(&tool_result, &compactor);
match gated {
    GateResult::Raw(content) => messages.push(content),
    GateResult::Briefing { summary, .. } => messages.push(summary),
}
```

### Memory Injection

System prompt, working memory, and skill injection all go through the same gate:

```rust
let system = content_gate.admit(&system_prompt, &compactor);
let memory = content_gate.admit(&working_memory, &compactor);
```

This prevents the scenario where memory injection eats the entire budget before the agent even starts working.

### read_file Tool Enhancement

Add optional `lines` parameter for range reads:

```rust
// Agent can request specific line ranges
read_file(path, lines="200:300")  // returns lines 200-300 only
```

This is how agents navigate after receiving a briefing. No new tool -- just a parameter on the existing one.

## Token Estimation

Simple and fast. No need for exact tokenizer:

```rust
fn estimate_tokens(text: &str) -> usize {
    // ~4 chars per token for English/code, conservative
    text.len() / 3
}
```

Erring on the high side (div by 3 instead of 4) is safer -- better to summarize slightly early than to blow the context.

## Model Context Windows

Known at config time from the model registry:

| Model | Context | Available (80%) |
|-------|---------|-----------------|
| Nanbeige4.1-3B | 4,096 | 3,277 |
| Qwen3-30B-A3B | 32,768 | 26,214 |
| Haiku | 200,000 | 160,000 |
| Sonnet | 200,000 | 160,000 |
| Opus | 200,000 | 160,000 |

## Graceful Degradation Examples

### Scenario: Agent reads a 10K line file

**On opus (200K context):**
- ContentGate: fits easily -> pass raw
- Agent sees full file, works normally

**On Qwen3-30B (32K context):**
- ContentGate: might fit depending on conversation length
- Early in conversation: pass raw
- Late in conversation: briefing + cache

**On Nanbeige (4K context):**
- ContentGate: doesn't fit -> briefing
- Agent gets structural map, drills into specific sections
- Takes 3-4 turns instead of 1, but completes the task

Same code path. Same tools. Different budget, different experience, both work.

## What This Replaces

- Hardcoded char truncation in tool runner
- `[VERBATIM]` marker in agent instructions
- Manual chunking strategies
- Per-tool output limits
- Any model-specific truncation config

## What This Reuses

- Existing Compactor (extended with briefing mode)
- Existing tool interface (tools unchanged)
- Existing agent loop structure (minimal changes)
- Existing model config (already has context window info)

## Implementation Order

1. **ContextBudget** -- pure struct, unit testable, no dependencies
2. **OutputCache** -- file I/O only, unit testable
3. **ContentGate** -- composes Budget + Cache + Compactor
4. **Briefing prompt** -- the compactor prompt that produces structural maps
5. **Tool runner integration** -- replace truncation with gate.admit()
6. **Agent loop integration** -- gate memory injection and tool results
7. **read_file lines parameter** -- enable range reads for drill-down
8. **Remove old truncation code** -- clean up

## Success Criteria

- No agent ever sees "[TRUNCATED at N chars]" or loses information
- Cloud models read files raw 95%+ of the time
- Local models can complete file-reading tasks via briefing + drill-down
- Zero new agent-facing API surface (no special markers or commands)
- Single code path for all models
