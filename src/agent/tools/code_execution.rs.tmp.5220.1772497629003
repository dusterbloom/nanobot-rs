//! Execute-Code RPC tool.
//!
//! Lets the LLM write a Python script that calls nanobot tools via a
//! Unix-domain-socket (UDS) RPC server.  Multi-step tool chains collapse to a
//! single LLM turn because the script drives all tool calls internally.
//!
//! # Execution flow
//!
//! 1. Write a Python stub (RPC helpers + user code) to a temp file.
//! 2. Open a `UnixListener` on `{tmp}/rpc.sock`.
//! 3. Spawn `python3 {script}` with `NANOBOT_RPC_SOCKET` set.
//! 4. Accept connections, read JSON-line requests, dispatch via registry,
//!    write JSON-line responses.
//! 5. On child exit or timeout, collect stdout and return it.
//!
//! # Anti-recursion
//!
//! `execute_code` is never included in the available-tools list exposed to the
//! Python stub, so scripts cannot nest `execute_code` calls.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixListener;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::task;

use super::base::Tool;
use super::registry::{ToolConfig, ToolRegistry};

// ---------------------------------------------------------------------------
// Python stub generation
// ---------------------------------------------------------------------------

/// Generate the Python preamble that wires RPC helpers for each tool.
///
/// The stub is prepended to the user code so every tool appears as a
/// plain Python function.  `execute_code` is excluded to prevent recursion.
pub fn generate_python_stub(available_tools: &[String]) -> String {
    let mut lines = Vec::new();

    lines.push(
        r#"import socket, json, os, sys

_SOCK_PATH = os.environ["NANOBOT_RPC_SOCKET"]

def _rpc_call(tool_name, **kwargs):
    """Call a nanobot tool via RPC."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(_SOCK_PATH)
    request = json.dumps({"tool": tool_name, "params": kwargs})
    sock.sendall((request + "\n").encode())
    response = b""
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            break
        response += chunk
    sock.close()
    result = json.loads(response.decode())
    if "error" in result:
        raise RuntimeError(result["error"])
    return result.get("result", "")
"#
        .to_string(),
    );

    // One convenience wrapper per tool (excluding execute_code).
    for tool_name in available_tools {
        if tool_name == "execute_code" {
            continue;
        }
        lines.push(format!(
            "\ndef {name}(**kwargs):\n    return _rpc_call(\"{name}\", **kwargs)\n",
            name = tool_name
        ));
    }

    lines.push("\n# --- User code below ---\n".to_string());
    lines.join("\n")
}

// ---------------------------------------------------------------------------
// RPC dispatcher (runs in a blocking thread)
// ---------------------------------------------------------------------------

/// Accepts connections on `listener`, dispatches tool calls through a fresh
/// registry built from `tool_config`, until the child exits or `max_tool_calls`
/// is exhausted.
fn run_rpc_server(
    listener: UnixListener,
    registry: Arc<ToolRegistry>,
    max_tool_calls: usize,
    call_count: Arc<Mutex<usize>>,
) {
    listener.set_nonblocking(false).ok();

    // We process one connection at a time (each tool call = one connection).
    for stream in listener.incoming() {
        match stream {
            Ok(mut stream) => {
                // Read one JSON line.
                let request_line = {
                    let mut reader = BufReader::new(&stream);
                    let mut line = String::new();
                    if reader.read_line(&mut line).is_err() || line.is_empty() {
                        continue;
                    }
                    line
                };

                let response: Value = match serde_json::from_str::<Value>(request_line.trim()) {
                    Ok(req) => {
                        let tool_name = req
                            .get("tool")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();

                        let params: HashMap<String, Value> = req
                            .get("params")
                            .and_then(|v| v.as_object())
                            .map(|obj| obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                            .unwrap_or_default();

                        // Enforce max_tool_calls.
                        let count = {
                            let mut c = call_count.lock().unwrap();
                            *c += 1;
                            *c
                        };
                        if count > max_tool_calls {
                            json!({"error": format!("max_tool_calls ({}) exceeded", max_tool_calls)})
                        } else {
                            // We are already in a spawn_blocking context, so we
                            // need a way to run async code.  Use block_in_place
                            // if we can reach the tokio runtime, otherwise build
                            // a mini runtime.
                            let result =
                                match tokio::runtime::Handle::try_current() {
                                    Ok(handle) => tokio::task::block_in_place(|| {
                                        handle.block_on(registry.execute(&tool_name, params))
                                    }),
                                    Err(_) => tokio::runtime::Runtime::new()
                                        .expect("tokio runtime")
                                        .block_on(registry.execute(&tool_name, params)),
                                };

                            if result.ok {
                                json!({"result": result.data})
                            } else {
                                json!({"error": result.error.unwrap_or(result.data)})
                            }
                        }
                    }
                    Err(e) => {
                        json!({"error": format!("invalid JSON request: {}", e)})
                    }
                };

                let mut response_bytes = serde_json::to_vec(&response).unwrap_or_default();
                response_bytes.push(b'\n');
                stream.write_all(&response_bytes).ok();
            }
            Err(_) => break,
        }
    }
}

// ---------------------------------------------------------------------------
// Tool struct
// ---------------------------------------------------------------------------

/// Execute-Code tool.
///
/// `tool_config` is used to build a fresh `ToolRegistry` for each script
/// execution.  This avoids circular `Arc` references and keeps the tool
/// stateless across executions.
pub struct CodeExecutionTool {
    pub enabled: bool,
    pub timeout_secs: u64,
    pub max_tool_calls: usize,
    /// Tool names that will appear as Python functions in the stub.
    /// `execute_code` is excluded automatically.
    pub available_tools: Vec<String>,
    /// Config snapshot used to build the per-execution registry.
    /// `None` means no tools are available to scripts (tests / sandboxed mode).
    pub tool_config: Option<ToolConfig>,
}

impl CodeExecutionTool {
    pub fn new(
        enabled: bool,
        timeout_secs: u64,
        max_tool_calls: usize,
        available_tools: Vec<String>,
        tool_config: Option<ToolConfig>,
    ) -> Self {
        Self {
            enabled,
            timeout_secs,
            max_tool_calls,
            available_tools,
            tool_config,
        }
    }
}

#[async_trait]
impl Tool for CodeExecutionTool {
    fn name(&self) -> &str {
        "execute_code"
    }

    fn description(&self) -> &str {
        "Execute a Python script that can call nanobot tools via RPC. \
        The script has access to all registered tools as plain Python functions. \
        Only stdout from the script is returned. \
        Use this to run multi-step tool chains in a single LLM turn."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "language": {
                    "type": "string",
                    "enum": ["python"],
                    "description": "Script language (only 'python' is supported)"
                },
                "code": {
                    "type": "string",
                    "description": "Python script source code to execute"
                }
            },
            "required": ["code"]
        })
    }

    fn is_available(&self) -> bool {
        self.enabled
    }

    async fn execute(&self, params: HashMap<String, Value>) -> String {
        if !self.enabled {
            return "Error: execute_code tool is not enabled. \
                Set tools.codeExecution.enabled = true in config."
                .to_string();
        }

        let code = match params.get("code").and_then(|v| v.as_str()) {
            Some(c) if !c.trim().is_empty() => c.to_string(),
            _ => return "Error: 'code' parameter is required and must not be empty".to_string(),
        };

        // Build the complete script: stub + user code.
        let stub = generate_python_stub(&self.available_tools);
        let full_script = format!("{}\n{}", stub, code);

        // Create a temp directory for this execution.
        let tmp_dir = match tempfile::tempdir() {
            Ok(d) => d,
            Err(e) => return format!("Error: failed to create temp dir: {}", e),
        };

        let script_path = tmp_dir.path().join("script.py");
        let socket_path = tmp_dir.path().join("rpc.sock");

        if let Err(e) = std::fs::write(&script_path, &full_script) {
            return format!("Error: failed to write script: {}", e);
        }

        // Open UDS listener before spawning the child so the socket exists.
        let listener = match UnixListener::bind(&socket_path) {
            Ok(l) => l,
            Err(e) => return format!("Error: failed to bind RPC socket: {}", e),
        };

        // Build a fresh registry for this execution.
        let registry = Arc::new(match &self.tool_config {
            Some(cfg) => ToolRegistry::with_standard_tools(cfg),
            None => ToolRegistry::new(),
        });

        let max_tool_calls = self.max_tool_calls;
        let call_count = Arc::new(Mutex::new(0usize));
        let call_count_clone = Arc::clone(&call_count);

        // Spawn RPC server in a dedicated blocking thread.
        let _server_handle = task::spawn_blocking(move || {
            run_rpc_server(listener, registry, max_tool_calls, call_count_clone);
        });

        // Spawn child python3 process.
        let child_result = tokio::process::Command::new("python3")
            .arg(&script_path)
            .env("NANOBOT_RPC_SOCKET", &socket_path)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true) // Automatically kill on drop (timeout path).
            .spawn();

        let child = match child_result {
            Ok(c) => c,
            Err(e) => return format!("Error: failed to spawn python3: {}", e),
        };

        // Wait for child with timeout.
        // `wait_with_output` takes ownership, so we handle timeout via kill_on_drop.
        let timeout = Duration::from_secs(self.timeout_secs);
        let output = tokio::time::timeout(timeout, child.wait_with_output()).await;

        match output {
            Ok(Ok(out)) => {
                let stdout = String::from_utf8_lossy(&out.stdout).to_string();
                let stderr = String::from_utf8_lossy(&out.stderr).to_string();

                if !out.status.success() && stdout.trim().is_empty() {
                    if stderr.trim().is_empty() {
                        format!(
                            "Script exited with code {}",
                            out.status.code().unwrap_or(-1)
                        )
                    } else {
                        format!(
                            "Script error (exit {}): {}",
                            out.status.code().unwrap_or(-1),
                            stderr.trim()
                        )
                    }
                } else {
                    stdout
                }
            }
            Ok(Err(e)) => format!("Error: waiting for child process failed: {}", e),
            Err(_) => {
                // kill_on_drop(true) ensures the child is killed when dropped.
                format!(
                    "Error: script timed out after {} seconds",
                    self.timeout_secs
                )
            }
        }
        // tmp_dir drops here, cleaning up script and socket.
        // child was already moved into wait_with_output — kill_on_drop handles cleanup.
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn disabled_tool() -> CodeExecutionTool {
        CodeExecutionTool {
            enabled: false,
            timeout_secs: 30,
            max_tool_calls: 20,
            available_tools: vec![],
            tool_config: None,
        }
    }

    fn enabled_tool() -> CodeExecutionTool {
        CodeExecutionTool {
            enabled: true,
            timeout_secs: 10,
            max_tool_calls: 5,
            available_tools: vec![],
            tool_config: None,
        }
    }

    // ------------------------------------------------------------------
    // Unit tests (no child process)
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn test_code_execution_disabled_returns_error() {
        let tool = disabled_tool();
        let mut params = HashMap::new();
        params.insert("code".to_string(), json!("print('hello')"));
        let result = tool.execute(params).await;
        assert!(
            result.contains("not enabled") || result.contains("disabled"),
            "expected disabled message, got: {}",
            result
        );
    }

    #[tokio::test]
    async fn test_code_execution_missing_code_returns_error() {
        let tool = enabled_tool();
        let result = tool.execute(HashMap::new()).await;
        assert!(
            result.starts_with("Error:"),
            "expected Error, got: {}",
            result
        );
    }

    #[tokio::test]
    async fn test_code_execution_empty_code_returns_error() {
        let tool = enabled_tool();
        let mut params = HashMap::new();
        params.insert("code".to_string(), json!("   "));
        let result = tool.execute(params).await;
        assert!(
            result.starts_with("Error:"),
            "expected Error, got: {}",
            result
        );
    }

    #[test]
    fn test_stub_contains_tool_functions() {
        let tools = vec!["read_file".to_string(), "web_search".to_string()];
        let stub = generate_python_stub(&tools);
        assert!(stub.contains("def read_file("), "missing read_file");
        assert!(stub.contains("def web_search("), "missing web_search");
    }

    #[test]
    fn test_stub_excludes_execute_code() {
        let tools = vec![
            "read_file".to_string(),
            "execute_code".to_string(),
            "web_search".to_string(),
        ];
        let stub = generate_python_stub(&tools);
        // Anti-recursion: execute_code must not appear as a defined function.
        assert!(
            !stub.contains("def execute_code("),
            "execute_code must not appear in stub"
        );
        assert!(stub.contains("def read_file("));
        assert!(stub.contains("def web_search("));
    }

    #[test]
    fn test_stub_contains_rpc_helper() {
        let stub = generate_python_stub(&[]);
        assert!(stub.contains("def _rpc_call("));
        assert!(stub.contains("NANOBOT_RPC_SOCKET"));
    }

    #[test]
    fn test_stub_has_user_code_marker() {
        let stub = generate_python_stub(&[]);
        assert!(
            stub.contains("# --- User code below ---"),
            "stub should delimit user code"
        );
    }

    #[test]
    fn test_rpc_request_format() {
        let req = json!({"tool": "read_file", "params": {"path": "/tmp/test.txt"}});
        let s = serde_json::to_string(&req).unwrap();
        assert!(s.contains("read_file"));
        assert!(s.contains("/tmp/test.txt"));
    }

    #[test]
    fn test_rpc_response_success_format() {
        let resp = json!({"result": "file contents here"});
        let s = serde_json::to_string(&resp).unwrap();
        assert!(s.contains("result"));
        assert!(s.contains("file contents here"));
    }

    #[test]
    fn test_rpc_response_error_format() {
        let resp = json!({"error": "tool not found"});
        let s = serde_json::to_string(&resp).unwrap();
        assert!(s.contains("error"));
        assert!(s.contains("tool not found"));
    }

    #[test]
    fn test_tool_name() {
        assert_eq!(disabled_tool().name(), "execute_code");
    }

    #[test]
    fn test_tool_schema_requires_code() {
        let params = disabled_tool().parameters();
        assert_eq!(params["type"], "object");
        let required = params["required"].as_array().unwrap();
        assert!(required.iter().any(|v| v == "code"), "code must be required");
    }

    #[test]
    fn test_disabled_tool_is_not_available() {
        assert!(!disabled_tool().is_available());
    }

    #[test]
    fn test_enabled_tool_is_available() {
        assert!(enabled_tool().is_available());
    }

    // ------------------------------------------------------------------
    // Integration tests (require python3)
    // ------------------------------------------------------------------

    fn python3_available() -> bool {
        std::process::Command::new("python3")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    #[tokio::test]
    async fn test_simple_print() {
        if !python3_available() {
            return;
        }
        let tool = enabled_tool();
        let mut params = HashMap::new();
        params.insert("code".to_string(), json!("print('hello from rpc')"));
        let result = tool.execute(params).await;
        assert!(
            result.contains("hello from rpc"),
            "expected script stdout, got: {}",
            result
        );
    }

    #[tokio::test]
    async fn test_syntax_error_does_not_panic() {
        if !python3_available() {
            return;
        }
        let tool = enabled_tool();
        let mut params = HashMap::new();
        params.insert("code".to_string(), json!("def (broken syntax"));
        let result = tool.execute(params).await;
        // Should return an error, not panic.
        assert!(
            !result.is_empty(),
            "expected non-empty output for syntax error"
        );
    }

    #[tokio::test]
    async fn test_timeout() {
        if !python3_available() {
            return;
        }
        let tool = CodeExecutionTool {
            enabled: true,
            timeout_secs: 1,
            max_tool_calls: 5,
            available_tools: vec![],
            tool_config: None,
        };
        let mut params = HashMap::new();
        params.insert("code".to_string(), json!("import time; time.sleep(60)"));
        let result = tool.execute(params).await;
        assert!(
            result.contains("timed out"),
            "expected timeout message, got: {}",
            result
        );
    }

    #[tokio::test]
    async fn test_multiline_output() {
        if !python3_available() {
            return;
        }
        let tool = enabled_tool();
        let mut params = HashMap::new();
        params.insert(
            "code".to_string(),
            json!("for i in range(3):\n    print(f'line {i}')"),
        );
        let result = tool.execute(params).await;
        assert!(result.contains("line 0"), "got: {}", result);
        assert!(result.contains("line 1"), "got: {}", result);
        assert!(result.contains("line 2"), "got: {}", result);
    }
}
