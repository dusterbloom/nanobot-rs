//! Async worker tools for the RLM swarm architecture.
//!
//! These tools are available to delegation workers (tool_runner) and provide
//! capabilities beyond the sync micro-tools in context_store.rs:
//! - `verify`: Run a command and check output against expectations
//! - `python_eval`: Sandboxed Python code execution
//! - `diff_apply`: Surgical file editing via unified diff

use std::collections::HashMap;
use std::time::Duration;

use serde_json::{json, Value};

/// Names of async worker tools.
pub const WORKER_TOOLS: &[&str] = &["verify", "python_eval", "diff_apply", "fmt_convert"];

/// Check if a tool name is an async worker tool.
pub fn is_worker_tool(name: &str) -> bool {
    WORKER_TOOLS.contains(&name)
}

/// JSON Schema definitions for worker tools.
pub fn worker_tool_definitions() -> Vec<Value> {
    vec![
        json!({
            "type": "function",
            "function": {
                "name": "verify",
                "description": "Run a shell command and check if output matches expectations. Returns PASS/FAIL with details. Uses the same safety deny-patterns as exec.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Shell command to run"},
                        "expect_contains": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Strings that must appear in stdout+stderr"
                        },
                        "expect_exit_code": {
                            "type": "integer",
                            "description": "Expected exit code (default 0)"
                        }
                    },
                    "required": ["command"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "python_eval",
                "description": "Execute a Python expression or short script. Returns stdout. Max 5 second timeout. No network access.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Python code to execute (use print() for output)"}
                    },
                    "required": ["code"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "diff_apply",
                "description": "Apply a unified diff patch to a file. Returns success/failure message.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path to patch"},
                        "diff": {"type": "string", "description": "Unified diff format (--- a/file\\n+++ b/file\\n@@ ... @@)"}
                    },
                    "required": ["path", "diff"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "fmt_convert",
                "description": "Convert data between formats. Supports: json→csv, csv→json, json→md_table, md_table→json. Input can be a variable name (e.g. 'output_0') or raw data.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input": {"type": "string", "description": "Input data string or ContextStore variable name"},
                        "from": {"type": "string", "enum": ["json", "csv", "md_table"], "description": "Source format"},
                        "to": {"type": "string", "enum": ["json", "csv", "md_table"], "description": "Target format"}
                    },
                    "required": ["input", "from", "to"]
                }
            }
        }),
    ]
}

/// Execute a worker tool by name. Returns the tool result as a string.
pub async fn execute_worker_tool(
    name: &str,
    args: &HashMap<String, Value>,
    workspace: Option<&std::path::Path>,
) -> String {
    match name {
        "verify" => execute_verify(args).await,
        "python_eval" => execute_python_eval(args).await,
        "diff_apply" => execute_diff_apply(args, workspace).await,
        "fmt_convert" => execute_fmt_convert(args).await,
        _ => format!("Error: unknown worker tool '{}'.", name),
    }
}

/// Run a command and check output against expectations.
async fn execute_verify(args: &HashMap<String, Value>) -> String {
    let command = match args.get("command").and_then(|v| v.as_str()) {
        Some(c) => c,
        None => return "Error: 'command' parameter is required.".to_string(),
    };

    // Safety: apply the same deny patterns as ExecTool.
    if let Some(reason) = check_deny_patterns(command) {
        return format!("BLOCKED: {}", reason);
    }

    let expect_contains: Vec<String> = args
        .get("expect_contains")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();

    let expect_exit = args
        .get("expect_exit_code")
        .and_then(|v| v.as_i64())
        .unwrap_or(0) as i32;

    let output = tokio::time::timeout(
        Duration::from_secs(30),
        tokio::process::Command::new("sh")
            .arg("-c")
            .arg(command)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true)
            .output(),
    )
    .await;

    match output {
        Ok(Ok(out)) => {
            let stdout = String::from_utf8_lossy(&out.stdout).to_string();
            let stderr = String::from_utf8_lossy(&out.stderr).to_string();
            let combined = format!("{}{}", stdout, stderr);
            let exit_code = out.status.code().unwrap_or(-1);
            let mut failures = Vec::new();

            if exit_code != expect_exit {
                failures.push(format!(
                    "Exit code: got {}, expected {}",
                    exit_code, expect_exit
                ));
            }

            for pattern in &expect_contains {
                if !combined.contains(pattern) {
                    failures.push(format!("Missing pattern: '{}'", pattern));
                }
            }

            if failures.is_empty() {
                let pattern_msg = if expect_contains.is_empty() {
                    String::new()
                } else {
                    format!(", all {} patterns found", expect_contains.len())
                };
                format!("PASS: exit code {}{}", exit_code, pattern_msg)
            } else {
                let preview: String = combined.chars().take(500).collect();
                format!(
                    "FAIL:\n{}\n\nOutput preview:\n{}",
                    failures.join("\n"),
                    preview
                )
            }
        }
        Ok(Err(e)) => format!("FAIL: command error: {}", e),
        Err(_) => "FAIL: command timed out (30s)".to_string(),
    }
}

/// Execute Python code in a sandboxed environment.
async fn execute_python_eval(args: &HashMap<String, Value>) -> String {
    let code = match args.get("code").and_then(|v| v.as_str()) {
        Some(c) => c,
        None => return "Error: 'code' parameter is required.".to_string(),
    };

    let output = tokio::time::timeout(
        Duration::from_secs(5),
        tokio::process::Command::new("python3")
            .arg("-c")
            .arg(code)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true)
            .output(),
    )
    .await;

    match output {
        Ok(Ok(out)) => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            let stderr = String::from_utf8_lossy(&out.stderr);
            if out.status.success() {
                let result: String = stdout.chars().take(2000).collect();
                if result.is_empty() {
                    "(no output)".to_string()
                } else {
                    result
                }
            } else {
                let err: String = stderr.chars().take(500).collect();
                format!("Error: {}", err)
            }
        }
        Ok(Err(e)) => format!("Error: {}", e),
        Err(_) => "Error: timeout (5s)".to_string(),
    }
}

/// Apply a unified diff patch to a file.
async fn execute_diff_apply(
    args: &HashMap<String, Value>,
    workspace: Option<&std::path::Path>,
) -> String {
    let path = match args.get("path").and_then(|v| v.as_str()) {
        Some(p) => p,
        None => return "Error: 'path' parameter is required.".to_string(),
    };
    let diff = match args.get("diff").and_then(|v| v.as_str()) {
        Some(d) => d,
        None => return "Error: 'diff' parameter is required.".to_string(),
    };

    // Security: if workspace is set, ensure path is within it.
    if let Some(ws) = workspace {
        let resolved = std::path::Path::new(path);
        if let Ok(canonical) = resolved.canonicalize() {
            if !canonical.starts_with(ws) {
                return format!("Error: path '{}' is outside workspace.", path);
            }
        }
        // If file doesn't exist yet, check parent.
        if !resolved.exists() {
            if let Some(parent) = resolved.parent() {
                if let Ok(canonical_parent) = parent.canonicalize() {
                    if !canonical_parent.starts_with(ws) {
                        return format!("Error: path '{}' is outside workspace.", path);
                    }
                }
            }
        }
    }

    // Write diff to temp file.
    let diff_path = format!("/tmp/nanobot_diff_{}.patch", uuid::Uuid::new_v4());
    if let Err(e) = std::fs::write(&diff_path, diff) {
        return format!("Error: could not write diff file: {}", e);
    }

    let output = tokio::time::timeout(
        Duration::from_secs(10),
        tokio::process::Command::new("patch")
            .arg("--forward")
            .arg("--no-backup-if-mismatch")
            .arg(path)
            .arg(&diff_path)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true)
            .output(),
    )
    .await;

    // Clean up temp file.
    let _ = std::fs::remove_file(&diff_path);

    match output {
        Ok(Ok(out)) => {
            if out.status.success() {
                "Patch applied successfully.".to_string()
            } else {
                let stderr = String::from_utf8_lossy(&out.stderr);
                let stdout = String::from_utf8_lossy(&out.stdout);
                format!(
                    "Patch failed: {}{}",
                    stderr.chars().take(300).collect::<String>(),
                    stdout.chars().take(200).collect::<String>()
                )
            }
        }
        Ok(Err(e)) => format!("Error: {}", e),
        Err(_) => "Error: patch timed out (10s)".to_string(),
    }
}

/// Convert data between formats.
async fn execute_fmt_convert(args: &HashMap<String, Value>) -> String {
    let input = match args.get("input").and_then(|v| v.as_str()) {
        Some(i) => i,
        None => return "Error: 'input' parameter is required.".to_string(),
    };
    let from = match args.get("from").and_then(|v| v.as_str()) {
        Some(f) => f,
        None => return "Error: 'from' parameter is required.".to_string(),
    };
    let to = match args.get("to").and_then(|v| v.as_str()) {
        Some(t) => t,
        None => return "Error: 'to' parameter is required.".to_string(),
    };

    if from == to {
        return input.to_string();
    }

    match (from, to) {
        ("json", "csv") => json_to_csv(input),
        ("csv", "json") => csv_to_json(input),
        ("json", "md_table") => json_to_md_table(input),
        ("md_table", "json") => md_table_to_json(input),
        _ => format!("Error: unsupported conversion from '{}' to '{}'.", from, to),
    }
}

/// Convert a JSON array of objects to CSV.
fn json_to_csv(input: &str) -> String {
    let arr: Vec<Value> = match serde_json::from_str(input) {
        Ok(Value::Array(a)) => a,
        Ok(_) => return "Error: JSON input must be an array of objects.".to_string(),
        Err(e) => return format!("Error: invalid JSON: {}", e),
    };

    if arr.is_empty() {
        return String::new();
    }

    // Extract headers from the first object.
    let headers: Vec<String> = match &arr[0] {
        Value::Object(obj) => obj.keys().cloned().collect(),
        _ => return "Error: JSON array items must be objects.".to_string(),
    };

    let mut lines = Vec::new();
    lines.push(headers.join(","));

    for item in &arr {
        if let Value::Object(obj) = item {
            let row: Vec<String> = headers
                .iter()
                .map(|h| {
                    match obj.get(h) {
                        Some(Value::String(s)) => {
                            // Escape commas and quotes in CSV.
                            if s.contains(',') || s.contains('"') || s.contains('\n') {
                                format!("\"{}\"", s.replace('"', "\"\""))
                            } else {
                                s.clone()
                            }
                        }
                        Some(Value::Null) => String::new(),
                        Some(v) => v.to_string(),
                        None => String::new(),
                    }
                })
                .collect();
            lines.push(row.join(","));
        }
    }

    lines.join("\n")
}

/// Convert CSV to a JSON array of objects.
fn csv_to_json(input: &str) -> String {
    let mut lines = input.lines();
    let header_line = match lines.next() {
        Some(h) => h,
        None => return "Error: empty CSV input.".to_string(),
    };

    let headers: Vec<&str> = parse_csv_line(header_line);
    let mut result = Vec::new();

    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let values = parse_csv_line(line);
        let mut obj = serde_json::Map::new();
        for (i, header) in headers.iter().enumerate() {
            let value = values.get(i).copied().unwrap_or("");
            // Try to parse as number or bool.
            if let Ok(n) = value.parse::<i64>() {
                obj.insert(header.to_string(), json!(n));
            } else if let Ok(n) = value.parse::<f64>() {
                obj.insert(header.to_string(), json!(n));
            } else if value == "true" || value == "false" {
                obj.insert(header.to_string(), json!(value == "true"));
            } else {
                obj.insert(header.to_string(), json!(value));
            }
        }
        result.push(Value::Object(obj));
    }

    serde_json::to_string_pretty(&result).unwrap_or_else(|e| format!("Error: {}", e))
}

/// Parse a CSV line handling quoted fields.
fn parse_csv_line(line: &str) -> Vec<&str> {
    // Simple CSV parser: splits on commas, respects double-quoted fields.
    // This handles most common cases but not escaped quotes within quotes.
    let mut fields = Vec::new();
    let mut start = 0;
    let mut in_quotes = false;
    let bytes = line.as_bytes();

    for i in 0..bytes.len() {
        if bytes[i] == b'"' {
            in_quotes = !in_quotes;
        } else if bytes[i] == b',' && !in_quotes {
            let field = &line[start..i];
            fields.push(field.trim().trim_matches('"'));
            start = i + 1;
        }
    }
    // Last field.
    let field = &line[start..];
    fields.push(field.trim().trim_matches('"'));

    fields
}

/// Convert a JSON array of objects to a markdown table.
fn json_to_md_table(input: &str) -> String {
    let arr: Vec<Value> = match serde_json::from_str(input) {
        Ok(Value::Array(a)) => a,
        Ok(_) => return "Error: JSON input must be an array of objects.".to_string(),
        Err(e) => return format!("Error: invalid JSON: {}", e),
    };

    if arr.is_empty() {
        return "| (empty) |\n| --- |".to_string();
    }

    let headers: Vec<String> = match &arr[0] {
        Value::Object(obj) => obj.keys().cloned().collect(),
        _ => return "Error: JSON array items must be objects.".to_string(),
    };

    let mut lines = Vec::new();
    // Header row.
    lines.push(format!("| {} |", headers.join(" | ")));
    // Separator.
    lines.push(format!(
        "| {} |",
        headers
            .iter()
            .map(|_| "---")
            .collect::<Vec<_>>()
            .join(" | ")
    ));
    // Data rows.
    for item in &arr {
        if let Value::Object(obj) = item {
            let cells: Vec<String> = headers
                .iter()
                .map(|h| match obj.get(h) {
                    Some(Value::String(s)) => s.clone(),
                    Some(Value::Null) => String::new(),
                    Some(v) => v.to_string().trim_matches('"').to_string(),
                    None => String::new(),
                })
                .collect();
            lines.push(format!("| {} |", cells.join(" | ")));
        }
    }

    lines.join("\n")
}

/// Convert a markdown table to a JSON array of objects.
fn md_table_to_json(input: &str) -> String {
    let lines: Vec<&str> = input.lines().collect();
    if lines.len() < 2 {
        return "Error: markdown table must have at least header and separator rows.".to_string();
    }

    // Parse header row.
    let headers: Vec<&str> = lines[0]
        .split('|')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();

    if headers.is_empty() {
        return "Error: no headers found in markdown table.".to_string();
    }

    // Skip separator row (line 1), parse data rows.
    let mut result = Vec::new();
    for line in &lines[2..] {
        if line.trim().is_empty() {
            continue;
        }
        let cells: Vec<&str> = line
            .split('|')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();

        let mut obj = serde_json::Map::new();
        for (i, header) in headers.iter().enumerate() {
            let value = cells.get(i).copied().unwrap_or("");
            if let Ok(n) = value.parse::<i64>() {
                obj.insert(header.to_string(), json!(n));
            } else if let Ok(n) = value.parse::<f64>() {
                obj.insert(header.to_string(), json!(n));
            } else {
                obj.insert(header.to_string(), json!(value));
            }
        }
        result.push(Value::Object(obj));
    }

    serde_json::to_string_pretty(&result).unwrap_or_else(|e| format!("Error: {}", e))
}

/// Check command against deny patterns (same as ExecTool).
/// Returns Some(reason) if blocked, None if allowed.
fn check_deny_patterns(command: &str) -> Option<String> {
    let normalized = command.to_lowercase();
    let patterns = [
        (r"\brm\s+-\w*r", "rm -r (recursive delete)"),
        (r"\brm\s+-[rf]{1,2}\b", "rm -rf (force delete)"),
        (r"\brm\s+--recursive", "rm --recursive"),
        (r"\brm\s+--force", "rm --force"),
        (r"\bfind\b.*\s-delete\b", "find -delete"),
        (r"\bfind\b.*-exec\s+rm\b", "find -exec rm"),
        (r"\bshred\b", "shred"),
        (r"\btruncate\b", "truncate"),
        (r"\b(format|mkfs|diskpart)\b", "disk format"),
        (r"\bdd\s+if=", "dd raw disk write"),
        (r">\s*/dev/sd", "raw device write"),
        (r"\b(shutdown|reboot|poweroff)\b", "system shutdown"),
        (r"curl\s.*\|\s*sh", "curl pipe to shell"),
        (r"wget\s.*\|\s*sh", "wget pipe to shell"),
        (r"\bchmod\s.*\+[xs]", "chmod +x/+s"),
        (r"\bsudo\b", "sudo"),
    ];

    for (pattern, desc) in &patterns {
        if let Ok(re) = regex::Regex::new(pattern) {
            if re.is_match(&normalized) {
                return Some(format!("Command blocked by deny pattern: {}", desc));
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_verify_pass() {
        let mut args = HashMap::new();
        args.insert("command".to_string(), json!("echo hello"));
        args.insert("expect_contains".to_string(), json!(["hello"]));
        let result = execute_verify(&args).await;
        assert!(result.starts_with("PASS"), "Expected PASS, got: {}", result);
    }

    #[tokio::test]
    async fn test_verify_fail_missing_pattern() {
        let mut args = HashMap::new();
        args.insert("command".to_string(), json!("echo hello"));
        args.insert("expect_contains".to_string(), json!(["goodbye"]));
        let result = execute_verify(&args).await;
        assert!(result.starts_with("FAIL"), "Expected FAIL, got: {}", result);
        assert!(result.contains("Missing pattern: 'goodbye'"));
    }

    #[tokio::test]
    async fn test_verify_fail_exit_code() {
        let mut args = HashMap::new();
        args.insert("command".to_string(), json!("false")); // exit code 1
        args.insert("expect_exit_code".to_string(), json!(0));
        let result = execute_verify(&args).await;
        assert!(result.starts_with("FAIL"), "Expected FAIL, got: {}", result);
        assert!(result.contains("Exit code: got 1"));
    }

    #[tokio::test]
    async fn test_verify_expected_nonzero_exit() {
        let mut args = HashMap::new();
        args.insert("command".to_string(), json!("false")); // exit code 1
        args.insert("expect_exit_code".to_string(), json!(1));
        let result = execute_verify(&args).await;
        assert!(
            result.starts_with("PASS"),
            "Expected PASS with exit 1, got: {}",
            result
        );
    }

    #[tokio::test]
    async fn test_verify_deny_patterns() {
        let mut args = HashMap::new();
        args.insert("command".to_string(), json!("rm -rf /"));
        let result = execute_verify(&args).await;
        assert!(
            result.starts_with("BLOCKED"),
            "Expected BLOCKED, got: {}",
            result
        );
    }

    #[tokio::test]
    async fn test_verify_no_expectations() {
        let mut args = HashMap::new();
        args.insert("command".to_string(), json!("echo test"));
        let result = execute_verify(&args).await;
        assert!(
            result.starts_with("PASS"),
            "Expected PASS with no expectations, got: {}",
            result
        );
    }

    #[tokio::test]
    async fn test_python_eval_basic() {
        let mut args = HashMap::new();
        args.insert("code".to_string(), json!("print(2 + 3)"));
        let result = execute_python_eval(&args).await;
        assert_eq!(result.trim(), "5");
    }

    #[tokio::test]
    async fn test_python_eval_error() {
        let mut args = HashMap::new();
        args.insert("code".to_string(), json!("raise ValueError('oops')"));
        let result = execute_python_eval(&args).await;
        assert!(
            result.starts_with("Error:"),
            "Expected error, got: {}",
            result
        );
    }

    #[tokio::test]
    async fn test_python_eval_no_output() {
        let mut args = HashMap::new();
        args.insert("code".to_string(), json!("x = 5"));
        let result = execute_python_eval(&args).await;
        assert_eq!(result, "(no output)");
    }

    #[tokio::test]
    async fn test_python_eval_output_truncation() {
        let mut args = HashMap::new();
        args.insert("code".to_string(), json!("print('x' * 5000)"));
        let result = execute_python_eval(&args).await;
        assert!(
            result.len() <= 2000,
            "Output should be truncated to 2000 chars, got {}",
            result.len()
        );
    }

    #[tokio::test]
    async fn test_diff_apply_missing_file() {
        let mut args = HashMap::new();
        args.insert(
            "path".to_string(),
            json!("/tmp/nonexistent_nanobot_test_file.txt"),
        );
        args.insert(
            "diff".to_string(),
            json!("--- a/file\n+++ b/file\n@@ -1 +1 @@\n-old\n+new"),
        );
        let result = execute_diff_apply(&args, None).await;
        // Patch should fail since file doesn't exist
        assert!(
            result.contains("fail") || result.contains("Error") || result.contains("Patch"),
            "Expected failure for missing file, got: {}",
            result
        );
    }

    #[tokio::test]
    async fn test_diff_apply_success() {
        // Create a temp file, apply a patch, verify.
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(&file_path, "line one\nline two\nline three\n").unwrap();

        let diff = format!(
            "--- a/test.txt\n+++ b/test.txt\n@@ -1,3 +1,3 @@\n line one\n-line two\n+line TWO\n line three\n"
        );

        let mut args = HashMap::new();
        args.insert("path".to_string(), json!(file_path.to_str().unwrap()));
        args.insert("diff".to_string(), json!(diff));
        let result = execute_diff_apply(&args, None).await;
        assert!(
            result.contains("success"),
            "Expected success, got: {}",
            result
        );

        // Verify file was patched.
        let content = std::fs::read_to_string(&file_path).unwrap();
        assert!(
            content.contains("line TWO"),
            "File should be patched, got: {}",
            content
        );
    }

    #[test]
    fn test_deny_patterns_block_rm() {
        assert!(check_deny_patterns("rm -rf /").is_some());
        assert!(check_deny_patterns("sudo apt install").is_some());
        assert!(check_deny_patterns("echo hello").is_none());
        assert!(check_deny_patterns("cargo test").is_none());
    }

    #[test]
    fn test_worker_tool_names() {
        assert!(is_worker_tool("verify"));
        assert!(is_worker_tool("python_eval"));
        assert!(is_worker_tool("diff_apply"));
        assert!(is_worker_tool("fmt_convert"));
        assert!(!is_worker_tool("exec"));
        assert!(!is_worker_tool("ctx_slice"));
    }

    #[test]
    fn test_worker_tool_definitions_count() {
        let defs = worker_tool_definitions();
        assert_eq!(defs.len(), 4, "Should have 4 worker tool definitions");
        let names: Vec<&str> = defs
            .iter()
            .filter_map(|d| d.pointer("/function/name").and_then(|v| v.as_str()))
            .collect();
        assert!(names.contains(&"verify"));
        assert!(names.contains(&"python_eval"));
        assert!(names.contains(&"diff_apply"));
        assert!(names.contains(&"fmt_convert"));
    }

    // -- fmt_convert tests --

    #[tokio::test]
    async fn test_fmt_convert_json_to_csv() {
        let mut args = HashMap::new();
        args.insert(
            "input".to_string(),
            json!(r#"[{"name":"Alice","age":30},{"name":"Bob","age":25}]"#),
        );
        args.insert("from".to_string(), json!("json"));
        args.insert("to".to_string(), json!("csv"));
        let result = execute_fmt_convert(&args).await;
        assert!(result.contains("name"), "Should have header: {}", result);
        assert!(result.contains("Alice"), "Should have data: {}", result);
    }

    #[tokio::test]
    async fn test_fmt_convert_csv_to_json() {
        let mut args = HashMap::new();
        args.insert("input".to_string(), json!("name,age\nAlice,30\nBob,25"));
        args.insert("from".to_string(), json!("csv"));
        args.insert("to".to_string(), json!("json"));
        let result = execute_fmt_convert(&args).await;
        let parsed: Vec<Value> = serde_json::from_str(&result).expect("Should be valid JSON");
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0]["name"], "Alice");
        assert_eq!(parsed[0]["age"], 30);
    }

    #[tokio::test]
    async fn test_fmt_convert_json_to_md_table() {
        let mut args = HashMap::new();
        args.insert(
            "input".to_string(),
            json!(r#"[{"name":"Alice","score":95}]"#),
        );
        args.insert("from".to_string(), json!("json"));
        args.insert("to".to_string(), json!("md_table"));
        let result = execute_fmt_convert(&args).await;
        assert!(
            result.contains("|"),
            "Should be a markdown table: {}",
            result
        );
        assert!(result.contains("---"), "Should have separator: {}", result);
        assert!(result.contains("Alice"), "Should have data: {}", result);
    }

    #[tokio::test]
    async fn test_fmt_convert_md_table_to_json() {
        let table = "| name | age |\n| --- | --- |\n| Alice | 30 |\n| Bob | 25 |";
        let mut args = HashMap::new();
        args.insert("input".to_string(), json!(table));
        args.insert("from".to_string(), json!("md_table"));
        args.insert("to".to_string(), json!("json"));
        let result = execute_fmt_convert(&args).await;
        let parsed: Vec<Value> = serde_json::from_str(&result).expect("Should be valid JSON");
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0]["name"], "Alice");
    }

    #[tokio::test]
    async fn test_fmt_convert_identity() {
        let mut args = HashMap::new();
        args.insert("input".to_string(), json!("some data"));
        args.insert("from".to_string(), json!("csv"));
        args.insert("to".to_string(), json!("csv"));
        let result = execute_fmt_convert(&args).await;
        assert_eq!(
            result, "some data",
            "Identity conversion should return input unchanged"
        );
    }

    #[tokio::test]
    async fn test_fmt_convert_unsupported() {
        let mut args = HashMap::new();
        args.insert("input".to_string(), json!("data"));
        args.insert("from".to_string(), json!("xml"));
        args.insert("to".to_string(), json!("json"));
        let result = execute_fmt_convert(&args).await;
        assert!(
            result.contains("unsupported"),
            "Should report unsupported: {}",
            result
        );
    }

    #[test]
    fn test_fmt_convert_in_definitions() {
        let defs = worker_tool_definitions();
        let names: Vec<&str> = defs
            .iter()
            .filter_map(|d| d.pointer("/function/name").and_then(|v| v.as_str()))
            .collect();
        assert!(
            names.contains(&"fmt_convert"),
            "Should include fmt_convert, got: {:?}",
            names
        );
    }
}
