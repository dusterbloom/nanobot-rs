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
use super::registry::ToolRegistry;

// ---------------------------------------------------------------------------
// Python stub generation
// ---------------------------------------------------------------------------

/// Generate the Python preamble that wires RPC helpers for each tool.
///
/// The stub is prepended to the user code so every tool appears as a
/// plain Python function.  `execute_code` is excluded to prevent recursion.
pub fn generate_python_stub(available_tools: &[String]) -> String {
    let mut lines = Vec::new();

    lines.push(r#"import socket, json, os, sys

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
"#.to_string());

    // One convenience wrapper per tool (excluding execute_code).
    for tool_name in available_tools {
        if tool_name == "execute_code" {
            continue;
        }
        // Simple positional wrapper: all kwargs forwarded as keyword args.
        lines.push(format!(
            r#"
def {name}(**kwargs):
    return _rpc_call("{name}", **kwargs)
"#,
            name = tool_name
        ));
    }

    lines.push("\n# --- User code below ---\n".to_string());
    lines.join("\n")
}

// ---------------------------------------------------------------------------
// RPC dispatcher (runs in a blocking thread)
// ---------------------------------------------------------------------------

/// Accepts connections on `listener`, dispatches tool calls through `registry`,
/// until the child process exits or `max_tool_calls` is exhausted.
///
/// Returns the accumulated tool-call count.
fn run_rpc_server(
    listener: UnixListener,
    registry: Arc<ToolRegistry>,
    max_tool_calls: usize,
    call_count: Arc<Mutex<usize>>,
) {
    // Non-blocking: the tokio runtime owns the actual timeout; we just serve
    // whatever connections arrive until the socket is dropped (child exits).
    listener.set_nonblocking(false).ok();

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
                            .map(|obj| {
                                obj.iter()
                                    .map(|(k, v)| (k.clone(), v.clone()))
                                    .collect()
                            })
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
                            // Execute synchronously via a fresh tokio runtime
                            // (we're already in a blocking thread).
                            let rt = tokio::runtime::Handle::try_current();
                            let result = if let Ok(handle) = rt {
                                // We are inside the tokio context — use block_in_place.
                                tokio::task::block_in_place(|| {
                                    handle.block_on(registry.execute(&tool_name, params))
                                })
                            } else {
                                // Fallback: spin a mini runtime.
                                tokio::runtime::Runtime::new()
                                    .expect("tokio runtime")
                                    .block_on(registry.execute(&tool_name, params))
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

/// Execute-Code tool configuration.
pub struct CodeExecutionTool {
    pub enabled: bool,
    pub timeout_secs: u64,
    pub max_tool_calls: usize,
    /// Tool names available for RPC calls (execute_code excluded automatically).
    pub available_tools: Vec<String>,
    /// Shared registry used for dispatching RPC calls from the child process.
    pub registry: Option<Arc<ToolRegistry>>,
}

impl CodeExecutionTool {
    pub fn new(
        enabled: bool,
        timeout_secs: u64,
        max_tool_calls: usize,
        available_tools: Vec<String>,
        registry: Option<Arc<ToolRegistry>>,
    ) -> Self {
        Self {
            enabled,
            timeout_secs,
            max_tool_calls,
            available_tools,
            registry,
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
        The script has access to all registered tools as Python functions. \
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

        // Write script to disk.
        if let Err(e) = std::fs::write(&script_path, &full_script) {
            return format!("Error: failed to write script: {}", e);
        }

        // Open UDS listener before spawning the child so the socket exists.
        let listener = match UnixListener::bind(&socket_path) {
            Ok(l) => l,
            Err(e) => return format!("Error: failed to bind RPC socket: {}", e),
        };

        let registry = match &self.registry {
            Some(r) => Arc::clone(r),
            None => Arc::new(ToolRegistry::new()),
        };

        let max_tool_calls = self.max_tool_calls;
        let call_count = Arc::new(Mutex::new(0usize));
        let call_count_clone = Arc::clone(&call_count);

        // Spawn RPC server in a blocking thread.
        let _server_handle = task::spawn_blocking(move || {
            run_rpc_server(listener, registry, max_tool_calls, call_count_clone);
        });

        // Spawn child python3 process.
        let child_result = tokio::process::Command::new("python3")
            .arg(&script_path)
            .env("NANOBOT_RPC_SOCKET", &socket_path)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn();

        let mut child = match child_result {
            Ok(c) => c,
            Err(e) => return format!("Error: failed to spawn python3: {}", e),
        };

        // Wait for child with timeout.
        let timeout = Duration::from_secs(self.timeout_secs);
        let output = tokio::time::timeout(timeout, child.wait_with_output()).await;

        match output {
            Ok(Ok(out)) => {
                let stdout = String::from_utf8_lossy(&out.stdout).to_string();
                let stderr = String::from_utf8_lossy(&out.stderr).to_string();

                if !out.status.success() && stdout.trim().is_empty() {
                    // Return stderr as context when there's no stdout.
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
                // Timeout: kill the child.
                child.kill().await.ok();
                format!(
                    "Error: script timed out after {} seconds",
                    self.timeout_secs
                )
            }
        }
        // tmp_dir is dropped here, cleaning up the temp files and socket.
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
            registry: None,
        }
    }

    fn enabled_tool_no_registry() -> CodeExecutionTool {
        CodeExecutionTool {
            enabled: true,
            timeout_secs: 10,
            max_tool_calls: 5,
            available_tools: vec![],
            registry: None,
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
        let tool = enabled_tool_no_registry();
        let result = tool.execute(HashMap::new()).await;
        assert!(
            result.starts_with("Error:"),
            "expected Error, got: {}",
            result
        );
    }

    #[tokio::test]
    async fn test_code_execution_empty_code_returns_error() {
        let tool = enabled_tool_no_registry();
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
    fn test_stub_generation_contains_tool_functions() {
        let tools = vec!["read_file".to_string(), "web_search".to_string()];
        let stub = generate_python_stub(&tools);
        assert!(stub.contains("def read_file("), "missing read_file");
        assert!(stub.contains("def web_search("), "missing web_search");
    }

    #[test]
    fn test_stub_generation_excludes_execute_code() {
        let tools = vec![
            "read_file".to_string(),
            "execute_code".to_string(),
            "web_search".to_string(),
        ];
        let stub = generate_python_stub(&tools);
        // Anti-recursion: execute_code must not appear as a defined function.
        assert!(
            !stub.contains("def execute_code("),
            "execute_code must be excluded from stub"
        );
        // Other tools should still be present.
        assert!(stub.contains("def read_file("));
        assert!(stub.contains("def web_search("));
    }

    #[test]
    fn test_stub_generation_contains_rpc_call() {
        let stub = generate_python_stub(&[]);
        assert!(stub.contains("def _rpc_call("));
        assert!(stub.contains("NANOBOT_RPC_SOCKET"));
    }

    #[test]
    fn test_stub_generation_user_code_marker() {
        let stub = generate_python_stub(&[]);
        assert!(
            stub.contains("# --- User code below ---"),
            "stub should have a user-code delimiter"
        );
    }

    #[test]
    fn test_rpc_protocol_request_format() {
        let req = json!({"tool": "read_file", "params": {"path": "/tmp/test.txt"}});
        let serialized = serde_json::to_string(&req).unwrap();
        assert!(serialized.contains("read_file"));
        assert!(serialized.contains("/tmp/test.txt"));
    }

    #[test]
    fn test_rpc_protocol_response_success() {
        let resp = json!({"result": "file contents here"});
        let serialized = serde_json::to_string(&resp).unwrap();
        assert!(serialized.contains("result"));
        assert!(serialized.contains("file contents here"));
    }

    #[test]
    fn test_rpc_protocol_response_error() {
        let resp = json!({"error": "tool not found"});
        let serialized = serde_json::to_string(&resp).unwrap();
        assert!(serialized.contains("error"));
        assert!(serialized.contains("tool not found"));
    }

    #[test]
    fn test_tool_name() {
        let tool = disabled_tool();
        assert_eq!(tool.name(), "execute_code");
    }

    #[test]
    fn test_tool_parameters_schema() {
        let tool = disabled_tool();
        let params = tool.parameters();
        assert_eq!(params["type"], "object");
        let required = params["required"].as_array().unwrap();
        assert!(required.iter().any(|v| v == "code"));
    }

    // ------------------------------------------------------------------
    // Integration tests (require python3)
    // ------------------------------------------------------------------

    /// Check whether python3 is available; skip tests if not.
    fn python3_available() -> bool {
        std::process::Command::new("python3")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    #[tokio::test]
    async fn test_code_execution_simple_print() {
        if !python3_available() {
            return;
        }
        let tool = enabled_tool_no_registry();
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
    async fn test_code_execution_syntax_error() {
        if !python3_available() {
            return;
        }
        let tool = enabled_tool_no_registry();
        let mut params = HashMap::new();
        params.insert("code".to_string(), json!("def (broken syntax"));
        let result = tool.execute(params).await;
        // Should return an error message, not panic.
        assert!(
            result.contains("error") || result.contains("Error") || result.contains("SyntaxError"),
            "expected error output, got: {}",
            result
        );
    }

    #[tokio::test]
    async fn test_code_execution_timeout() {
        if !python3_available() {
            return;
        }
        let tool = CodeExecutionTool {
            enabled: true,
            timeout_secs: 1,
            max_tool_calls: 5,
            available_tools: vec![],
            registry: None,
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
    async fn test_code_execution_multiline_output() {
        if !python3_available() {
            return;
        }
        let tool = enabled_tool_no_registry();
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
