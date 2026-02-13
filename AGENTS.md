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
    fn description(&self) -> &str { "Brief description" }
    
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
        format!("Result: {}", value)
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
