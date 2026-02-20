# Nanobot Codebase Architecture Review

## Executive Summary

The nanobot codebase is generally well-structured with clear module boundaries and good separation of concerns. However, there are several architectural issues that would benefit from refactoring. This review identifies specific patterns that need attention.

---

## 1. Inconsistent Patterns

### 1.1 Error Handling Inconsistency

**Files affected:** Multiple across the codebase

**Issue:** The codebase mixes multiple error handling approaches:

1. **`anyhow::Result`** - Used in most places (good)
2. **String-based errors** - Tool execution returns `String` with "Error:" prefix
3. **Silent failures** - Some functions use `.ok()` to swallow errors

**Examples:**

```rust
// src/agent/tools/base.rs - String-based error detection
async fn execute_with_result(&self, params: ...) -> ToolExecutionResult {
    let out = self.execute(params).await;
    if let Some(err) = out.strip_prefix("Error:").map(|s| s.trim().to_string()) {
        // Parse error from string prefix
    }
}

// src/main.rs - Silent error swallowing
std::fs::write(&file_path, content).ok();
println!("  Created {}", filename);
```

**Recommendation:** 
- Migrate tool execution to return `Result<String, ToolError>` instead of string parsing
- Replace `.ok()` with explicit error handling or logging

### 1.2 Inconsistent Async Patterns

**Files affected:** `src/session/manager.rs`, `src/channels/`

**Issue:** Some async functions take `&self` with internal `Mutex`, others take `&mut self`.

```rust
// src/session/manager.rs - Internal mutex, takes &self
pub async fn get_history(&self, key: &str, max_messages: usize) -> Vec<Value> {
    let mut cache = self.cache.lock().await;
    // ...
}

// src/channels/base.rs - Takes &mut self
async fn start(&mut self) -> Result<()>;
async fn stop(&mut self) -> Result<()>;
```

**Recommendation:** Standardize on `&self` with internal synchronization for all async methods to enable easier concurrent usage.

### 1.3 Configuration Path Duplication

**Files affected:** `src/config/schema.rs`, `src/utils/helpers.rs`

**Issue:** `expand_tilde()` is defined in both files with identical implementations.

```rust
// src/config/schema.rs
fn expand_tilde(path: &str) -> PathBuf { ... }

// src/utils/helpers.rs  
fn expand_tilde(path: &str) -> PathBuf { ... }
```

**Recommendation:** Export from `utils/helpers.rs` and import in `config/schema.rs`.

---

## 2. Circular Dependencies

**Status:** No circular module dependencies detected.

The module hierarchy is clean:
- `main.rs` → `agent`, `channels`, `config`, `providers`, etc.
- `agent` → `providers`, `config`, `session`, `bus`
- `channels` → `config`, `bus`, `providers` (for voice)

The design uses message passing (`InboundMessage`/`OutboundMessage`) through channels rather than direct module coupling, which is a good architectural choice.

---

## 3. God Objects / Large Modules

### 3.1 `main.rs` - 2700+ lines

**File:** `src/main.rs`

**Issue:** This file handles:
- CLI parsing and dispatch
- REPL loop with all commands
- Local LLM server management
- Voice mode handling
- Channel quick-start commands
- GGUF metadata parsing
- Memory detection
- TUI helpers

**Specific concerns:**
- `cmd_agent()` function is ~600 lines
- Inline TUI module with ANSI codes
- Local model management mixed with CLI logic

**Recommendation:** Extract into separate modules:
```
src/
  cli/
    mod.rs          # Cli struct, command dispatch
    agent.rs        # cmd_agent, REPL loop
    channels.rs     # cmd_whatsapp, cmd_telegram, cmd_email
    cron.rs         # cmd_cron_*
    status.rs       # cmd_status
  local/
    mod.rs          # Local LLM server management
    gguf.rs         # GGUF metadata parsing
    memory.rs       # VRAM/RAM detection
  tui/
    mod.rs          # ANSI helpers, print_logo, loading_animation
```

### 3.2 `SharedCore` - Too Many Responsibilities

**File:** `src/agent/agent_loop.rs`

**Issue:** `SharedCore` struct holds 20+ fields and manages:
- LLM provider configuration
- Token budgeting
- Context compaction
- Session management
- Learning store
- Memory configuration
- Execution settings

```rust
pub struct SharedCore {
    pub provider: Arc<dyn LLMProvider>,
    pub workspace: PathBuf,
    pub model: String,
    pub max_iterations: u32,
    pub max_tokens: u32,
    pub temperature: f64,
    pub context: ContextBuilder,
    pub sessions: SessionManager,
    pub token_budget: TokenBudget,
    pub compactor: ContextCompactor,
    pub learning: LearningStore,
    pub brave_api_key: Option<String>,
    pub exec_timeout: u64,
    pub restrict_to_workspace: bool,
    pub memory_enabled: bool,
    pub memory_provider: Arc<dyn LLMProvider>,
    pub memory_model: String,
    pub reflection_threshold: usize,
    pub learning_turn_counter: AtomicU64,
    pub last_context_used: AtomicU64,
    pub last_context_max: AtomicU64,
    pub is_local: bool,
}
```

**Recommendation:** Split into focused components:
- `AgentConfig` - model, temperature, max_tokens, max_iterations
- `ToolConfig` - exec_timeout, restrict_to_workspace, brave_api_key
- `MemoryConfig` - already exists, but memory_provider/model should move there
- `AgentState` - last_context_used, last_context_max, learning_turn_counter

### 3.3 `AgentLoop::process_message` - Complex Method

**File:** `src/agent/agent_loop.rs`

**Issue:** Both `process_message()` and `process_message_streaming()` are ~150 lines each with significant code duplication.

**Recommendation:** Extract common logic into shared helper methods:
- `prepare_context()` - build messages, extract media
- `run_tool_loop()` - the iteration loop with tool execution
- `finalize_response()` - session updates, metadata propagation

---

## 4. Leaky Abstractions

### 4.1 Tool Proxy Wrappers

**File:** `src/agent/agent_loop.rs`

**Issue:** Three nearly identical proxy structs exist just to wrap `Arc<T>`:

```rust
struct MessageToolProxy(Arc<MessageTool>);
struct SpawnToolProxy(Arc<SpawnTool>);
struct CronToolProxy(Arc<CronScheduleTool>);
```

Each implements the same delegation pattern.

**Recommendation:** Either:
1. Implement `Tool` for `Arc<T> where T: Tool` via blanket impl
2. Create a generic `ToolProxy<T: Tool>` wrapper
3. Use `Box<dyn Tool>` directly without Arc indirection

### 4.2 Provider Detection Logic Exposed

**File:** `src/providers/openai_compat.rs`

**Issue:** `OpenAICompatProvider::new()` has complex detection logic for API base URLs based on key prefixes:

```rust
} else if api_key.starts_with("sk-or-") {
    "https://openrouter.ai/api/v1".to_string()
} else if api_key.starts_with("sk-ant-") {
    "https://api.anthropic.com/v1".to_string()
```

This mixes concerns (provider detection should be separate from provider implementation).

**Recommendation:** Extract provider detection into a separate `ProviderResolver` that returns a `ProviderConfig` struct.

### 4.3 Channel Config Leaking Through Manager

**File:** `src/channels/manager.rs`

**Issue:** `ChannelManager::new()` directly accesses nested config fields and has feature-flag conditionals scattered throughout:

```rust
if config.channels.telegram.enabled {
    let groq_key = config.providers.groq.api_key.clone();
    let ch = TelegramChannel::new(
        config.channels.telegram.clone(),
        bus_inbound_tx.clone(),
        groq_key,
        #[cfg(feature = "voice")]
        voice_pipeline.clone(),
    );
```

**Recommendation:** 
- Have each channel type implement a `Channel::from_config()` factory method
- Move feature-flag logic into the channel implementations

---

## 5. Dead Code / Unused Items

### 5.1 Unused Constants and Variables

**File:** `src/main.rs`

```rust
const VERSION: &str = "0.1.0";  // Used
const LOGO: &str = "*";          // Used only in print statements, could be inline

// In cmd_agent():
let mut compaction_process: Option<std::process::Child> = None;  // Used
let mut compaction_port: Option<String> = None;                   // Used
```

### 5.2 Commented-Out Code

**Status:** No significant commented-out code blocks found.

### 5.3 Potentially Unused Functions

**File:** `src/channels/manager.rs`

```rust
pub fn get_status(&self) -> HashMap<String, Value>  // Not called anywhere in codebase
```

**File:** `src/session/manager.rs`

```rust
pub fn save(&self, session: &Session)      // Only save_session() internal version used
pub async fn save_cached(&self, key: &str) // Not called anywhere
pub async fn delete(&self, key: &str)      // Not called anywhere
pub fn list_sessions(&self)                // Not called anywhere
```

**Recommendation:** Run `cargo +nightly udeps` or review with `cargo clippy` to identify unused code.

### 5.4 Unreachable Branches

**File:** `src/agent/agent_loop.rs`

```rust
// In process_message():
if final_content.is_empty() && messages.len() > 2 {
    final_content = "I ran out of tool iterations...";
}
// This branch is only reachable if max_iterations is exhausted AND no content
// produced, which is a valid edge case, not dead code.
```

No significant unreachable branches found.

---

## 6. Additional Observations

### 6.1 Test Coverage

The codebase has good unit test coverage for:
- Configuration parsing
- Tool registry
- Session management
- Telegram markdown conversion
- Provider response parsing

Missing test coverage:
- Integration tests for the agent loop
- Channel start/stop lifecycle tests
- End-to-end message flow tests

### 6.2 Documentation

Well-documented modules with `//!` doc comments. Most public APIs have doc comments. The codebase follows Rust documentation conventions.

### 6.3 Feature Flag Usage

The `#[cfg(feature = "voice")]` flag is used extensively but creates code duplication in several places (Telegram, WhatsApp channels). Consider extracting voice handling into a separate trait or helper module.

---

## Priority Recommendations

### High Priority
1. **Extract `main.rs` into modules** - Improves maintainability and testability
2. **Unify error handling** - Replace string-based tool errors with proper Result types
3. **Remove code duplication** - `expand_tilde()`, process_message vs process_message_streaming

### Medium Priority
4. **Split `SharedCore`** - Reduce coupling and improve testability
5. **Clean up unused public APIs** - `SessionManager` methods, `ChannelManager::get_status()`
6. **Standardize async patterns** - Consistent `&self` with internal sync

### Low Priority
7. **Generic tool proxy** - Reduce boilerplate
8. **Extract provider detection** - Cleaner separation of concerns
9. **Voice feature consolidation** - Reduce cfg flag duplication

---

## Conclusion

The nanobot codebase demonstrates solid Rust practices with clear module boundaries and good use of async/await patterns. The main architectural debt is concentrated in `main.rs` (god object) and some inconsistent patterns that evolved organically. The recommended refactorings would improve maintainability without requiring fundamental architectural changes.
