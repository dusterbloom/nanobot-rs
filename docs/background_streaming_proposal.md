# Background Streaming Capability - Enhancement Proposal

## Summary

Add native background process support to the `exec` tool so agents can run long-lived processes (like audio streams) without blocking terminal interaction.

---

## Problem Statement

Currently, when an agent executes commands like:
```bash
ffplay -nodisp http://stream.example.com/jazz
```

The command blocks until completion, preventing other operations. Workarounds require manual `nohup &` patterns which agents can't reliably use due to safety guards blocking background syntax (`&`, `nohup`).

---

## Solution Overview

Add a new optional parameter `background: bool` to the `ExecTool` that:
1. Spawns the process detached from parent terminal
2. Redirects stdout/stderr to log files in `/tmp/nanobot/`
3. Returns immediately with PID and status
4. Provides tools to monitor/manage background processes

---

## Architecture

### New Tool Parameters (extended `exec`)

```json
{
  "type": "object",
  "properties": {
    "command": {
      "type": "string"
    },
    "background": {
      "type": "boolean",
      "description": "If true, run detached with stdout/stderr redirected to /tmp/nanobot/{pid}.log. Returns immediately with PID."
    }
  },
  "required": ["command"]
}
```

### New Struct: `BackgroundProcessManager`

```rust
pub struct BackgroundProcessManager {
    log_dir: PathBuf,
    processes: Arc<Mutex<HashMap<Pid, BackgroundProcessInfo>>>,
}

pub struct BackgroundProcessInfo {
    pub pid: Pid,
    pub command: String,
    pub start_time: Instant,
    pub log_file: PathBuf,
    pub status: ProcessStatus,
}

pub enum ProcessStatus {
    Running,
    Terminated,
    Failed,
}
```

### New Tool Methods

1. **`execute_with_background(params)`** - Extended `exec` with background mode
2. **`list_processes()`** - List all managed background processes
3. **`get_logs(pid)`** - Fetch logs for a specific process
4. **`stop_process(pid)`** - Gracefully terminate a background process

---

## Code References

### Base Tool Pattern (src/agent/tools/base.rs)

```rust
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters(&self) -> serde_json::Value;
    async fn execute(&self, params: HashMap<String, serde_json::Value>) -> String;
}
```

### ExecTool Structure (src/agent/tools/shell.rs)

Already has:
- `timeout` configuration
- Safety guards (`guard_command`)
- Streaming progress events via `execute_with_context`
- Cancellation support

**Reuse these patterns for background mode.**

---

## Implementation Plan

### Phase 1: Core Background Execution

Add to `ExecTool`:

```rust
pub struct ExecTool {
    // Existing fields...
    pub log_dir: Option<String>,
}

impl ExecTool {
    pub fn new(..., log_dir: Option<String>) -> Self { ... }
    
    async fn execute_with_background(
        &self,
        params: HashMap<String, serde_json::Value>
    ) -> String {
        // 1. Parse background flag
        let should_bg = params.get("background")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        
        if !should_bg {
            return self.execute(params).await;
        }
        
        // 2. Safety checks (same as foreground)
        // Reuse guard_command()
        
        // 3. Spawn detached process
        let log_dir = Path::new(&self.log_dir.unwrap_or_else(|| {
            std::env::temp_dir().join("nanobot").to_string_lossy().to_string()
        }));
        
        tokio::fs::create_dir_all(log_dir).await?;
        let log_file = log_dir.join(format!("{}.log", pid));
        
        // Use Command with detached behavior:
        let mut child = Command::new("sh")
            .arg("-c")
            .arg(command)
            .stdin(Stdio::null())
            .stdout(Stdio::File::create(&log_file)?);
            .stderr(Stdio::File::create(log_file.clone())?);
        // Note: On Unix, use .process_group(false) to detach from terminal
        
        let pid = child.id();
        
        // 4. Return immediately with metadata
        Ok(format!("Background process started.\nPID: {}\nLog file: {}", pid, log_file.display()))
    }
}
```

### Phase 2: Process Management Tools

Add new tools in `src/agent/tools/background.rs`:

```rust
pub struct ListProcessesTool;
impl Tool for ListProcessesTool {
    fn name(&self) -> &str { "list_processes" }
    
    async fn execute(...) {
        // Return JSON list of all running background processes
    }
}

pub struct GetLogsTool;
impl Tool for GetLogsTool {
    fn name(&self) -> &str { "get_background_logs" }
    
    async fn execute(...) {
        let pid = params["pid"];
        // Read and return log file contents
    }
}

pub struct StopProcessTool;
impl Tool for StopProcessTool {
    fn name(&self) -> &str { "stop_background_process" }
    
    async fn execute(...) {
        let pid = params["pid"];
        // Kill process gracefully (SIGTERM), then forcefully if needed
    }
}
```

---

## TDD Test Plan

### Test 1: Background execution returns PID immediately
**Red:** Write test, expect timeout failure  
**Green:** Implement basic background spawning  

```rust
#[tokio::test]
async fn test_background_mode_returns_pid() {
    let tool = ExecTool::new(10, None, None, None, false, 30000);
    let mut params = HashMap::new();
    params.insert("command".to_string(), "sleep 5".into());
    params.insert("background".to_string(), true.into());
    
    let result = tool.execute(params).await;
    assert!(result.contains("PID: "), "Expected PID in output");
}
```

### Test 2: Process actually runs in background
**Red:** Write test, expect process not found  
**Green:** Implement proper detachment  

```rust
#[tokio::test]
async fn test_background_process_continues_running() {
    let tool = ExecTool::new(10, None, None, None, false, 30000);
    let mut params = HashMap::new();
    params.insert("command".to_string(), "sleep 10".into());
    params.insert("background".to_string(), true.into());
    
    // Start background process
    let result = tool.execute(params).await;
    assert!(result.contains("PID: "));
    
    // Extract PID from result (simplified)
    let pid_str = result.split(": ").nth(1).unwrap();
    let pid: u32 = pid_str.parse().unwrap();
    
    // Wait 1 second, process should still be running
    tokio::time::sleep(Duration::from_secs(1)).await;
    
    // Check if process exists via /proc or ps command
    let status_result = ExecTool::execute_command(&format!("ps -p {} > /dev/null", pid));
    assert!(status_result.contains("Exit code: 0"), "Process should still be running");
}
```

### Test 3: Logs are written to temp directory
**Red:** Write test, expect file not found  
**Green:** Implement log file creation  

```rust
#[tokio::test]
async fn test_background_logs_written_to_file() {
    let tool = ExecTool::new(10, None, None, None, false, 30000);
    let mut params = HashMap::new();
    params.insert("command".to_string(), "echo 'hello world'".into());
    params.insert("background".to_string(), true.into());
    
    let result = tool.execute(params).await;
    assert!(result.contains("Log file: "));
    
    // Extract log path and verify content after process completes
    tokio::time::sleep(Duration::from_secs(2)).await;
    
    let log_path = result.split(": ").nth(1).unwrap().trim();
    let logs = fs::read_to_string(log_path).await.unwrap();
    assert!(logs.contains("hello world"));
}
```

### Test 4: Background execution is safe (blocked patterns)
**Red:** Write test, expect bypass  
**Green:** Reuse guard_command() for background mode  

```rust
#[tokio::test]
async fn test_background_blocked_patterns() {
    let tool = ExecTool::new(10, None, None, None, true, 30000); // restrict=true
    let mut params = HashMap::new();
    params.insert("command".to_string(), "rm -rf /".into());
    params.insert("background".to_string(), true.into());
    
    let result = tool.execute(params).await;
    assert!(result.contains("blocked"), "Dangerous commands should still be blocked in background mode");
}
```

### Test 5: list_processes tool works
**Red:** Write test for new tool  
**Green:** Implement process registry  

```rust
#[tokio::test]
async fn test_list_processes_tool() {
    let list_tool = ListProcessesTool;
    let params = HashMap::new(); // No params needed
    
    let result = list_tool.execute(params).await;
    
    // Should return valid JSON array
    let json: Value = serde_json::from_str(&result).unwrap();
    assert!(json.is_array());
}
```

### Test 6: stop_process tool terminates background job
**Red:** Write test, expect no termination  
**Green:** Implement SIGTERM/SIGKILL logic  

```rust
#[tokio::test]
async fn test_stop_background_process() {
    let start_tool = ExecTool::new(10, None, None, None, false, 30000);
    let stop_tool = StopProcessTool;
    
    // Start long-running background process
    let mut params = HashMap::new();
    params.insert("command".to_string(), "sleep 60".into());
    params.insert("background".to_string(), true.into());
    
    let start_result = start_tool.execute(params.clone()).await;
    assert!(start_result.contains("PID: "));
    let pid_str = start_result.split(": ").nth(1).unwrap();
    let pid: u32 = pid_str.parse().unwrap();
    
    // Stop it
    let mut stop_params = HashMap::new();
    stop_params.insert("pid".to_string(), pid.into());
    
    let stop_result = stop_tool.execute(stop_params).await;
    assert!(stop_result.contains("stopped") || stop_result.contains("terminated"));
    
    // Verify process is gone
    tokio::time::sleep(Duration::from_millis(500)).await;
    let status = ExecTool::execute_command(&format!("ps -p {} > /dev/null", pid));
    assert!(!status.contains("Exit code: 0"), "Process should be terminated");
}
```

### Test 7: Concurrent background processes don't interfere
**Red:** Write test with 2 simultaneous processes  
**Green:** Implement thread-safe process registry  

```rust
#[tokio::test]
async fn test_concurrent_background_processes() {
    let tool = ExecTool::new(10, None, None, None, false, 30000);
    
    // Start two background processes simultaneously
    let mut params1 = HashMap::new();
    params1.insert("command".to_string(), "sleep 5 && echo 'process1'".into());
    params1.insert("background".to_string(), true.into());
    
    let mut params2 = HashMap::new();
    params2.insert("command".to_string(), "sleep 5 && echo 'process2'".into());
    params2.insert("background".to_string(), true.into());
    
    let (result1, result2) = tokio::join!(
        tool.execute(params1),
        tool.execute(params2)
    );
    
    assert!(result1.contains("PID: "));
    assert!(result2.contains("PID: "));
    assert_ne!(result1, result2); // Different PIDs
    
    // Wait for completion and check logs separately
    tokio::time::sleep(Duration::from_secs(7)).await;
    
    let log1 = fs::read_to_string(result1.split(": ").nth(1).unwrap().trim()).await.unwrap();
    let log2 = fs::read_to_string(result2.split(": ").nth(1).unwrap().trim()).await.unwrap();
    
    assert!(log1.contains("process1"));
    assert!(log2.contains("process2"));
}
```

---

## Safety Considerations

### Reuse existing guards:
- `deny_patterns` still apply (no rm, sudo, etc.)
- `allow_patterns` still enforce allowlists
- Workspace restrictions still checked
- **New**: Background mode doesn't bypass any security checks

### Log directory isolation:
```rust
let log_dir = Path::new(&self.log_dir.unwrap_or_else(|| {
    std::env::temp_dir().join("nanobot").to_string_lossy().to_string()
}));
// Ensure temp_dir is world-writable but agent-controlled only
```

### Process lifecycle management:
- Agent can't spawn infinite processes (track count in memory)
- Max concurrent processes configurable (default: 10)
- Auto-cleanup of orphaned PIDs on restart (scan /proc or use process registry)

---

## Integration Points

### Existing patterns to reuse:
1. **Progress events** - `execute_with_context()` already supports streaming
2. **Cancellation tokens** - Background processes can be stopped via cancellation
3. **Tool schema generation** - Add `"background"` to existing parameters JSON schema
4. **Error classification** - Reuse `classify_tool_error()` for background failures

### File locations:
- New tool: `/src/agent/tools/background.rs` (similar pattern to `shell.rs`)
- Register in: `/src/agent/tools/mod.rs` and `/registry.rs`
- Config path: Add `background_log_dir` to config struct if needed

---

## Example Usage Flow

```python
# Agent wants to play radio without blocking
params = {
    "command": "/opt/homebrew/bin/ffplay -nodisp -autoexit 'http://stream.radioparadise.com/aac-320'",
    "background": True
}
result = exec.execute(params)
# Output: 
# Background process started.
# PID: 36705
# Log file: /tmp/nanobot/36705.log

# Agent can now do other things!
# Later check status:
params = {}
status = list_processes.execute(params)
# Output: [{"pid": 36705, "command": "...", "status": "running"}]

# Or stop it:
params = {"pid": 36705}
result = stop_process.execute(params)
# Output: "Process 36705 stopped successfully"
```

---

## Success Metrics

- ✅ TDD all tests pass (red → green cycle verified)
- ✅ No security regressions (same guards as foreground mode)
- ✅ Works across platforms (Unix `nohup`/`setsid`, Windows job objects)
- ✅ Clear error messages when background mode fails
- ✅ Logs accessible via new tools

---

## Next Steps After Implementation

1. Update webradio skill to use `background: true` parameter automatically for audio streams
2. Add CLI command `nanobot bg list` / `nanobot bg stop <pid>` for terminal users
3. Consider adding process monitoring dashboard in TUI mode
