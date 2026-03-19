# AGENTS.md

# nanobot - Agent Instructions
A lightweight personal AI assistant framework in Rust. Binary name: `nanobot`.

## Build Commands
```bash
cargo build --release        # Release build (optimized)
cargo build                  # Debug build
cargo test                   # Run all tests
cargo test test_name         # Run a single test by name (partial match)
cargo test module::tests     # Run tests for a specific module
RUST_LOG=debug cargo run -- agent -m "Hello"  # Run with debug logging
```

**Testing Notes**: 
- All tests are inline in modules under `#[cfg(test)] mod tests { ... }`
- Use `cargo test -- --nocapture` to see test output

## Project Structure
```
src/
├── main.rs              # CLI entry point, command routing
├── agent/
│   ├── agent_loop.rs    # Core message processing loop
│   ├── context.rs       # System prompt building
│   ├── tools/           # Tool implementations
│   ├── memory.rs        # Long-term memory management
│   └── skills.rs        # Skill loading and execution
├── providers/           # LLM provider clients (OpenAI-compatible)
├── config/              # JSON config schema and loader
├── channels/            # Chat adapters (Telegram, WhatsApp, Feishu, Email)
├── bus/                 # InboundMessage/OutboundMessage event types
└── session/             # JSONL-based session persistence
```

## Code Style Guidelines
### Imports
Group imports: std → external crates → internal modules
```rust
use std::collections::HashMap;
use std::path::PathBuf;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::agent::tools::base::Tool;
```

### Naming Conventions
- **Types/Structs/Enums**: `PascalCase` (e.g., `AgentLoop`, `ToolRegistry`)
- **Functions/Methods**: `snake_case` (e.g., `process_direct`, `build_context`)
- **Variables**: `snake_case` (e.g., `session_id`, `max_iterations`)
- **Constants**: `SCREAMING_SNAKE_CASE` (e.g., `MAX_TOKENS`)

### Error Handling
```rust
// Application-level: use anyhow
use anyhow::{Context, Result};

fn load_config() -> Result<Config> {
    let content = std::fs::read_to_string(&path)
        .context("Failed to read config file")?;
}

// Tool execute() returns String, not Result
// - Success: return output directly
// - Error: prefix with "Error: "
async fn execute(&self, params: HashMap<String, Value>) -> String {
    if missing_param {
        return "Error: 'command' parameter is required".to_string();
    }
}
```

### Async Patterns
```rust
use async_trait::async_trait;

#[async_trait]
pub trait Tool: Send + Sync {
    async fn execute(&self, params: HashMap<String, Value>) -> String;
}

// Use Arc for shared state
use std::sync::Arc;
use tokio::sync::Mutex;
let shared = Arc::new(Mutex::new(State::new()));
```

### Struct and Config Patterns
```rust
// Use serde with camelCase for JSON config
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Config {
    pub max_iterations: u32,     // Rust: snake_case
    pub api_key: String,         // JSON: "apiKey"
}
```

### Tool Development
All tools implement the `Tool` trait from `src/agent/tools/base.rs`:
```rust
#[async_trait]
impl Tool for MyTool {
    fn name(&self) -> &str { "my_tool" }
    fn description(&self) -> &str { "Brief description" };

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "param_name": {"type": "string", "description": "..."}
            },
            "required": ["param_name"]
        })
    }

    async fn execute(&self, params: HashMap<String, serde_json::Value>) -> String {
        let value = params.get("param_name")
            .and_then(|v| v.as_str())
            .unwrap_or("default");
        format!("Result: {}",$value)
    }
}
```

### Safety Guidelines
- **Shell commands**: Use safety guards with deny patterns (see `src/agent/tools/shell.rs`)
- **File operations**: Validate paths and restrict to workspace when configured
- **User input**: Always validate and sanitize before use

### Testing Patterns
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_name() {
        let tool = MyTool::new();
        let result = tool.method();
        assert_eq!(result, expected);
    }

    #[tokio::test]
    async fn test_async_operation() {
        let result = async_function().await;
        assert!(result.is_ok());
    }
}
```

## Configuration
- Config location: `~/.nanobot/config.json`
- Session storage: `~/.nanobot/sessions/`
- Workspace (skills, memory): `~/.nanobot/workspace/`

## Provider Selection
Config chooses provider in this order (first non-empty API key wins):
OpenRouter > DeepSeek > Anthropic > OpenAI > Gemini > Zhipu > Groq > vLLM

All providers use OpenAI-compatible chat completions API via `OpenAICompatProvider`.

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **nanobot-rs** (8336 symbols, 25125 relationships, 300 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

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

## CLI

- Re-index: `npx gitnexus analyze`
- Check freshness: `npx gitnexus status`
- Generate docs: `npx gitnexus wiki`

<!-- gitnexus:end -->
