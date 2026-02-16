//! Claude Code CLI provider — uses the `claude` CLI as an LLM backend.
//!
//! This lets nanobot use a Claude Max subscription (flat-rate, unlimited) instead
//! of per-token API billing.  The provider shells out to `claude -p` with
//! `--tools ""` (no built-in tools) so it acts as a raw LLM interface.
//!
//! Tool calling is handled via prompt engineering: tool schemas are injected into
//! the system prompt and tool call requests are parsed from `<tool_call>` blocks.

use std::collections::HashMap;
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tracing::{debug, warn};

/// Global counter for tool call IDs — ensures uniqueness across the entire
/// session. Without this, `parse_tool_calls` would reset to `cc_0` every
/// response, causing `dedup_tool_results` to delete real results from
/// subsequent turns. The prefix includes the PID so restored sessions
/// from a previous process can't collide with new tool calls.
static TOOL_CALL_COUNTER: AtomicU64 = AtomicU64::new(0);
static TOOL_CALL_PID: std::sync::OnceLock<u32> = std::sync::OnceLock::new();

fn next_tool_call_id() -> String {
    let pid = *TOOL_CALL_PID.get_or_init(std::process::id);
    let seq = TOOL_CALL_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("cc_{pid}_{seq}")
}

use super::base::{LLMProvider, LLMResponse, StreamChunk, StreamHandle, ToolCallRequest};

// ---------------------------------------------------------------------------
// Provider struct
// ---------------------------------------------------------------------------

/// LLM provider that delegates to the `claude` CLI (Claude Code).
///
/// Requires `claude` to be installed and authenticated (Max plan).
pub struct ClaudeCodeProvider {
    /// Model alias passed to `--model` (e.g. "opus", "sonnet", "haiku").
    model: String,
}

impl ClaudeCodeProvider {
    pub fn new(model: Option<&str>) -> Self {
        Self {
            model: model.unwrap_or("opus").to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Message serialization
// ---------------------------------------------------------------------------

/// Extract the system prompt and serialize remaining messages to a
/// human-readable conversation transcript that claude -p can consume.
fn serialize_messages(messages: &[Value]) -> (String, String) {
    let mut system = String::new();
    let mut conversation = String::new();

    for msg in messages {
        let role = msg["role"].as_str().unwrap_or("");
        // Content can be a string or an array of content blocks.
        let content = if let Some(s) = msg["content"].as_str() {
            s.to_string()
        } else if let Some(arr) = msg["content"].as_array() {
            // Concatenate text blocks.
            arr.iter()
                .filter_map(|block| block["text"].as_str())
                .collect::<Vec<_>>()
                .join("\n")
        } else {
            String::new()
        };

        match role {
            "system" => {
                if !system.is_empty() {
                    system.push('\n');
                }
                system.push_str(&content);
            }
            "user" => {
                conversation.push_str(&format!("User: {}\n\n", content));
            }
            "assistant" => {
                // Reconstruct tool calls if present.
                let mut text = content.clone();
                if let Some(tcs) = msg["tool_calls"].as_array() {
                    for tc in tcs {
                        let name = tc["function"]["name"]
                            .as_str()
                            .unwrap_or("unknown");
                        let args = &tc["function"]["arguments"];
                        // arguments may be a JSON string or an object.
                        let args_val: Value = if let Some(s) = args.as_str() {
                            serde_json::from_str(s).unwrap_or(Value::Object(Default::default()))
                        } else {
                            args.clone()
                        };
                        text.push_str(&format!(
                            "\n<tool_call>{{\"name\":\"{}\",\"arguments\":{}}}</tool_call>",
                            name,
                            serde_json::to_string(&args_val).unwrap_or_default()
                        ));
                    }
                }
                conversation.push_str(&format!("Assistant: {}\n\n", text));
            }
            "tool" => {
                let tool_call_id = msg["tool_call_id"]
                    .as_str()
                    .unwrap_or("unknown");
                conversation.push_str(&format!(
                    "Tool Result [{}]: {}\n\n",
                    tool_call_id, content
                ));
            }
            _ => {
                // Unknown role — include as-is.
                conversation.push_str(&format!("{}: {}\n\n", role, content));
            }
        }
    }

    (system, conversation)
}

// ---------------------------------------------------------------------------
// Tool schema formatting
// ---------------------------------------------------------------------------

/// Format OpenAI-style tool definitions into a system prompt section that
/// instructs Claude to output `<tool_call>` blocks.
fn format_tool_instructions(tools: &[Value]) -> String {
    let mut out = String::from(
        "\n\n## Tool Calling\n\n\
         Call tools by wrapping a single-line JSON object in <tool_call></tool_call> tags:\n\n\
         <tool_call>{\"name\": \"tool_name\", \"arguments\": {\"arg\": \"value\"}}</tool_call>\n\n\
         Rules:\n\
         - Each <tool_call> MUST contain valid JSON. Escape newlines as \\n in string values.\n\
         - Output multiple <tool_call> blocks for parallel calls.\n\
         - When calling tools, keep surrounding text minimal.\n\n\
         ### Available Tools\n\n",
    );

    for tool in tools {
        let func = &tool["function"];
        let name = func["name"].as_str().unwrap_or("unknown");
        let desc = func["description"].as_str().unwrap_or("");
        let params = &func["parameters"];
        out.push_str(&format!(
            "**{}**: {}\nParameters: `{}`\n\n",
            name,
            desc,
            serde_json::to_string(params).unwrap_or_else(|_| "{}".into())
        ));
    }

    out
}

// ---------------------------------------------------------------------------
// Response parsing
// ---------------------------------------------------------------------------

/// Try to parse a JSON string, with a fallback that escapes raw newlines
/// inside JSON string values. Models sometimes emit literal newlines in
/// `<tool_call>` arguments (e.g. heredoc scripts) which breaks strict JSON.
fn try_parse_json(json_str: &str) -> Option<Value> {
    // Fast path: already valid JSON.
    if let Ok(v) = serde_json::from_str::<Value>(json_str) {
        return Some(v);
    }
    // Fallback: walk char-by-char, escaping raw control characters inside
    // JSON string values while leaving structure whitespace untouched.
    let mut out = String::with_capacity(json_str.len() + 64);
    let mut in_str = false;
    let mut escape_next = false;
    for ch in json_str.chars() {
        if escape_next {
            out.push(ch);
            escape_next = false;
            continue;
        }
        if in_str {
            match ch {
                '\\' => {
                    out.push(ch);
                    escape_next = true;
                }
                '"' => {
                    out.push(ch);
                    in_str = false;
                }
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                _ => out.push(ch),
            }
        } else {
            if ch == '"' {
                in_str = true;
            }
            out.push(ch);
        }
    }
    serde_json::from_str::<Value>(&out).ok()
}

/// Strip known Claude CLI boilerplate from response text.
///
/// The CLI appends messages like "I ran out of tool iterations" when
/// `--max-turns 1` terminates a turn with pending tool calls. These are
/// CLI artifacts, not model output — stripping them makes behavior match
/// the native API.
fn strip_cli_artifacts(text: &str) -> String {
    let mut cleaned = text.to_string();
    for artifact in CLI_ARTIFACT_PATTERNS {
        cleaned = cleaned.replace(artifact, "");
    }
    // Collapse triple-newlines left by stripping.
    while cleaned.contains("\n\n\n") {
        cleaned = cleaned.replace("\n\n\n", "\n\n");
    }
    cleaned.trim().to_string()
}

/// Parse `<tool_call>` blocks from Claude's text response.
///
/// Returns (clean_text, tool_calls) where clean_text has the blocks removed
/// and CLI artifacts stripped.
fn parse_tool_calls(text: &str) -> (String, Vec<ToolCallRequest>) {
    let mut clean = String::new();
    let mut calls = Vec::new();
    let mut remaining = text;

    while let Some(start_idx) = remaining.find("<tool_call>") {
        // Text before the tag.
        clean.push_str(&remaining[..start_idx]);
        let after_tag = &remaining[start_idx + "<tool_call>".len()..];

        if let Some(end_idx) = after_tag.find("</tool_call>") {
            let json_str = after_tag[..end_idx].trim();
            if let Some(parsed) = try_parse_json(json_str) {
                let name = parsed["name"]
                    .as_str()
                    .unwrap_or("unknown")
                    .to_string();
                let arguments: HashMap<String, Value> = parsed["arguments"]
                    .as_object()
                    .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                    .unwrap_or_default();

                calls.push(ToolCallRequest {
                    id: next_tool_call_id(),
                    name,
                    arguments,
                });
            } else {
                warn!("Failed to parse tool_call JSON: {}", json_str);
                // Keep the malformed block in the text so the user sees it.
                clean.push_str(&remaining[start_idx..start_idx + "<tool_call>".len() + end_idx + "</tool_call>".len()]);
            }
            remaining = &after_tag[end_idx + "</tool_call>".len()..];
        } else {
            // Unclosed tag — keep everything.
            clean.push_str(&remaining[start_idx..]);
            remaining = "";
            break;
        }
    }
    clean.push_str(remaining);

    (strip_cli_artifacts(&clean), calls)
}

// ---------------------------------------------------------------------------
// Stream filtering
// ---------------------------------------------------------------------------

/// Known CLI artifact strings to suppress from streaming output.
/// These are injected by `claude -p --max-turns 1` when the CLI exits with
/// pending work — they are not model output.
const CLI_ARTIFACT_PATTERNS: &[&str] = &[
    "I ran out of tool iterations before producing a final answer.",
    "The actions above may be incomplete.",
    "I need more turns to complete this task.",
];

/// Filters `<tool_call>...</tool_call>` blocks AND known CLI artifact strings
/// from streaming text so the user only sees clean prose — matching the
/// behavior of native API streaming where tool calls arrive as structured
/// data, not inline text.
struct ToolCallFilter {
    buffer: String,
    /// true while inside a `<tool_call>...</tool_call>` block (suppress all).
    in_tool_block: bool,
}

impl ToolCallFilter {
    fn new() -> Self {
        Self {
            buffer: String::new(),
            in_tool_block: false,
        }
    }

    /// Feed a text delta and return the portion safe to emit.
    fn feed(&mut self, text: &str) -> String {
        self.buffer.push_str(text);
        let mut emit = String::new();

        loop {
            // ── Inside a <tool_call> block: suppress until closing tag ──
            if self.in_tool_block {
                if let Some(end) = self.buffer.find("</tool_call>") {
                    self.buffer = self.buffer[end + "</tool_call>".len()..].to_string();
                    self.in_tool_block = false;
                    continue;
                }
                break; // Still inside — suppress everything.
            }

            // ── Check for <tool_call> start ─────────────────────────────
            if let Some(start) = self.buffer.find("<tool_call>") {
                emit.push_str(&self.buffer[..start]);
                self.buffer = self.buffer[start..].to_string();
                self.in_tool_block = true;
                continue;
            }

            // ── Check for complete CLI artifact strings ─────────────────
            let mut found_artifact = false;
            for artifact in CLI_ARTIFACT_PATTERNS {
                if let Some(pos) = self.buffer.find(artifact) {
                    emit.push_str(&self.buffer[..pos]);
                    self.buffer = self.buffer[pos + artifact.len()..].to_string();
                    found_artifact = true;
                    break;
                }
            }
            if found_artifact {
                continue;
            }

            // ── Hold back partial prefix matches at the end ─────────────
            // This catches both partial `<tool_call` tags and partial CLI
            // artifact prefixes (e.g. "I ran out of tool iter" at buffer end).
            let holdback = Self::max_partial_prefix(&self.buffer);
            if holdback > 0 {
                let safe = self.buffer.len() - holdback;
                emit.push_str(&self.buffer[..safe]);
                self.buffer = self.buffer[safe..].to_string();
            } else {
                emit.push_str(&self.buffer);
                self.buffer.clear();
            }
            break;
        }

        emit
    }

    /// How many trailing bytes of `s` match a prefix of any suppressed pattern?
    ///
    /// This ensures we hold back text like "I ran out of tool" at the end of a
    /// streaming delta until we can confirm it's part of an artifact (suppress)
    /// or not (emit).
    fn max_partial_prefix(s: &str) -> usize {
        let mut max = 0;

        // Check <tool_call> tag prefix.
        let tag = "<tool_call>";
        for len in (1..=tag.len().min(s.len())).rev() {
            if s.ends_with(&tag[..len]) {
                max = max.max(len);
                break;
            }
        }

        // Check CLI artifact prefixes.
        for artifact in CLI_ARTIFACT_PATTERNS {
            for len in (1..=artifact.len().min(s.len())).rev() {
                if s.ends_with(&artifact[..len]) {
                    max = max.max(len);
                    break;
                }
            }
        }

        max
    }
}

// ---------------------------------------------------------------------------
// JSON response parsing
// ---------------------------------------------------------------------------

/// Parse the JSON envelope returned by `claude -p --output-format json`.
fn parse_json_response(stdout: &str) -> Result<(String, bool)> {
    let json: Value =
        serde_json::from_str(stdout).context("Failed to parse claude CLI JSON output")?;

    let is_error = json["is_error"].as_bool().unwrap_or(false);
    let result = json["result"].as_str().unwrap_or("").to_string();

    if is_error {
        let subtype = json["subtype"].as_str().unwrap_or("unknown");
        warn!("Claude CLI returned error (subtype={}): {}", subtype, result);
    }

    Ok((result, is_error))
}

// ---------------------------------------------------------------------------
// CLI invocation helpers
// ---------------------------------------------------------------------------

/// Strip the "claude-code/" prefix from model names so the CLI gets a bare
/// model alias like "opus" instead of the nanobot routing prefix.
fn strip_routing_prefix(model: &str) -> &str {
    model
        .strip_prefix("claude-code/")
        .or_else(|| model.strip_prefix("claude-code"))
        .filter(|s| !s.is_empty())
        .unwrap_or(model)
}

/// Build the base `claude` command with common flags.
fn build_command(
    system_prompt: &str,
    model: &str,
    output_format: &str,
) -> tokio::process::Command {
    let mut cmd = tokio::process::Command::new("claude");
    cmd.arg("-p")
        .arg("Continue this conversation. Respond as the assistant.")
        .arg("--output-format")
        .arg(output_format)
        .arg("--system-prompt")
        .arg(system_prompt)
        .arg("--tools")
        .arg("") // Disable all built-in tools — we handle tool calling via prompt.
        .arg("--max-turns")
        .arg("1")
        .arg("--no-session-persistence")
        .arg("--model")
        .arg(model)
        // Unset CLAUDECODE to avoid "cannot launch inside another session" error.
        .env_remove("CLAUDECODE")
        .env_remove("CLAUDE_CODE_ENTRYPOINT")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    cmd
}

// ---------------------------------------------------------------------------
// LLMProvider implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl LLMProvider for ClaudeCodeProvider {
    async fn chat(
        &self,
        messages: &[Value],
        tools: Option<&[Value]>,
        model: Option<&str>,
        _max_tokens: u32,
        _temperature: f64,
        _thinking_budget: Option<u32>,
    ) -> Result<LLMResponse> {
        let (system, conversation) = serialize_messages(messages);
        let mut full_system = system;
        if let Some(tool_defs) = tools {
            if !tool_defs.is_empty() {
                full_system.push_str(&format_tool_instructions(tool_defs));
            }
        }

        // Strip "claude-code/" routing prefix before passing to CLI.
        let raw_model = model.unwrap_or(&self.model);
        let effective_model = strip_routing_prefix(raw_model);
        let mut cmd = build_command(&full_system, effective_model, "json");

        debug!(
            "ClaudeCodeProvider::chat model={} conversation_len={}",
            effective_model,
            conversation.len()
        );

        let mut child = cmd.spawn().context("Failed to spawn claude CLI")?;

        // Write conversation to stdin then close it.
        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(conversation.as_bytes())
                .await
                .context("Failed to write to claude stdin")?;
        }

        let output = tokio::time::timeout(
            std::time::Duration::from_secs(300),
            child.wait_with_output(),
        )
        .await
        .context("claude CLI timed out (300s)")?
        .context("claude CLI process error")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!(
                "claude CLI exited with {}: {}",
                output.status.code().unwrap_or(-1),
                stderr.chars().take(500).collect::<String>()
            );
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let (result_text, _is_error) = parse_json_response(&stdout)?;
        let (clean_text, tool_calls) = parse_tool_calls(&result_text);

        let finish_reason = if !tool_calls.is_empty() {
            "tool_calls"
        } else {
            "stop"
        };

        Ok(LLMResponse {
            content: if clean_text.is_empty() {
                None
            } else {
                Some(clean_text)
            },
            tool_calls,
            finish_reason: finish_reason.to_string(),
            usage: HashMap::new(),
        })
    }

    async fn chat_stream(
        &self,
        messages: &[Value],
        tools: Option<&[Value]>,
        model: Option<&str>,
        _max_tokens: u32,
        _temperature: f64,
        _thinking_budget: Option<u32>,
    ) -> Result<StreamHandle> {
        let (system, conversation) = serialize_messages(messages);
        let mut full_system = system;
        if let Some(tool_defs) = tools {
            if !tool_defs.is_empty() {
                full_system.push_str(&format_tool_instructions(tool_defs));
            }
        }

        // Strip "claude-code/" routing prefix before passing to CLI.
        let raw_model = model.unwrap_or(&self.model);
        let effective_model = strip_routing_prefix(raw_model);
        let mut cmd = build_command(&full_system, effective_model, "stream-json");
        cmd.arg("--verbose").arg("--include-partial-messages");

        debug!(
            "ClaudeCodeProvider::chat_stream model={} conversation_len={}",
            effective_model,
            conversation.len()
        );

        let mut child = cmd.spawn().context("Failed to spawn claude CLI")?;

        // Write conversation to stdin then close it.
        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(conversation.as_bytes())
                .await
                .context("Failed to write to claude stdin")?;
        }

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow::anyhow!("No stdout from claude CLI"))?;

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

        tokio::spawn(async move {
            let reader = BufReader::new(stdout);
            let mut lines = reader.lines();
            let mut full_text = String::new();
            let mut filter = ToolCallFilter::new();

            while let Ok(Some(line)) = lines.next_line().await {
                if line.is_empty() {
                    continue;
                }
                let json: Value = match serde_json::from_str(&line) {
                    Ok(v) => v,
                    Err(_) => continue,
                };

                let msg_type = json["type"].as_str().unwrap_or("");

                // Text delta from streaming.
                if msg_type == "stream_event" {
                    if let Some(delta_type) = json["event"]["delta"]["type"].as_str() {
                        if delta_type == "text_delta" {
                            if let Some(text) = json["event"]["delta"]["text"].as_str() {
                                full_text.push_str(text);
                                // Filter out <tool_call> blocks so the user
                                // only sees clean prose during streaming.
                                let clean = filter.feed(text);
                                if !clean.is_empty() {
                                    let _ = tx.send(StreamChunk::TextDelta(clean));
                                }
                            }
                        }
                    }
                }
                // Assistant message (non-streaming, full message).
                else if msg_type == "assistant" || msg_type == "message" {
                    if let Some(content) = json["message"]["content"].as_array() {
                        for block in content {
                            if block["type"].as_str() == Some("text") {
                                if let Some(t) = block["text"].as_str() {
                                    if full_text.is_empty() {
                                        full_text.push_str(t);
                                        let clean = filter.feed(t);
                                        if !clean.is_empty() {
                                            let _ =
                                                tx.send(StreamChunk::TextDelta(clean));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                // Final result.
                else if msg_type == "result" {
                    let result_text = json["result"]
                        .as_str()
                        .map(|s| s.to_string())
                        .unwrap_or(full_text.clone());

                    let (clean_text, tool_calls) = parse_tool_calls(&result_text);
                    let finish = if !tool_calls.is_empty() {
                        "tool_calls"
                    } else {
                        "stop"
                    };

                    let _ = tx.send(StreamChunk::Done(LLMResponse {
                        content: if clean_text.is_empty() {
                            None
                        } else {
                            Some(clean_text)
                        },
                        tool_calls,
                        finish_reason: finish.to_string(),
                        usage: HashMap::new(),
                    }));
                    break;
                }
            }

            // If we never got a "result" message, emit Done with whatever we collected.
            if !full_text.is_empty() {
                let (clean_text, tool_calls) = parse_tool_calls(&full_text);
                let finish = if !tool_calls.is_empty() {
                    "tool_calls"
                } else {
                    "stop"
                };
                let _ = tx.send(StreamChunk::Done(LLMResponse {
                    content: if clean_text.is_empty() {
                        None
                    } else {
                        Some(clean_text)
                    },
                    tool_calls,
                    finish_reason: finish.to_string(),
                    usage: HashMap::new(),
                }));
            }

            // Clean up child process.
            let _ = child.wait().await;
        });

        Ok(StreamHandle { rx })
    }

    fn get_default_model(&self) -> &str {
        &self.model
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ── Message serialization ──────────────────────────────────────

    #[test]
    fn test_serialize_system_message() {
        let messages = vec![json!({"role": "system", "content": "You are helpful."})];
        let (system, conv) = serialize_messages(&messages);
        assert_eq!(system, "You are helpful.");
        assert!(conv.is_empty());
    }

    #[test]
    fn test_serialize_conversation() {
        let messages = vec![
            json!({"role": "system", "content": "Be concise."}),
            json!({"role": "user", "content": "Hello"}),
            json!({"role": "assistant", "content": "Hi there!"}),
            json!({"role": "user", "content": "How are you?"}),
        ];
        let (system, conv) = serialize_messages(&messages);
        assert_eq!(system, "Be concise.");
        assert!(conv.contains("User: Hello"));
        assert!(conv.contains("Assistant: Hi there!"));
        assert!(conv.contains("User: How are you?"));
    }

    #[test]
    fn test_serialize_tool_result() {
        let messages = vec![json!({
            "role": "tool",
            "tool_call_id": "call_0",
            "content": "file contents here"
        })];
        let (_, conv) = serialize_messages(&messages);
        assert!(conv.contains("Tool Result [call_0]: file contents here"));
    }

    #[test]
    fn test_serialize_assistant_with_tool_calls() {
        let messages = vec![json!({
            "role": "assistant",
            "content": "I'll read that file.",
            "tool_calls": [{
                "id": "tc_1",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": "{\"path\": \"/tmp/test\"}"
                }
            }]
        })];
        let (_, conv) = serialize_messages(&messages);
        assert!(conv.contains("I'll read that file."));
        assert!(conv.contains("<tool_call>"));
        assert!(conv.contains("read_file"));
    }

    #[test]
    fn test_serialize_content_array() {
        let messages = vec![json!({
            "role": "user",
            "content": [
                {"type": "text", "text": "Part 1"},
                {"type": "text", "text": "Part 2"}
            ]
        })];
        let (_, conv) = serialize_messages(&messages);
        assert!(conv.contains("Part 1\nPart 2"));
    }

    // ── Tool instruction formatting ────────────────────────────────

    #[test]
    fn test_format_tool_instructions() {
        let tools = vec![json!({
            "type": "function",
            "function": {
                "name": "shell",
                "description": "Execute a shell command",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"}
                    },
                    "required": ["command"]
                }
            }
        })];
        let result = format_tool_instructions(&tools);
        assert!(result.contains("## Tool Calling"));
        assert!(result.contains("<tool_call>"));
        assert!(result.contains("**shell**"));
        assert!(result.contains("Execute a shell command"));
    }

    #[test]
    fn test_format_multiple_tools() {
        let tools = vec![
            json!({"function": {"name": "read", "description": "Read a file", "parameters": {}}}),
            json!({"function": {"name": "write", "description": "Write a file", "parameters": {}}}),
        ];
        let result = format_tool_instructions(&tools);
        assert!(result.contains("**read**"));
        assert!(result.contains("**write**"));
    }

    // ── Tool call parsing ──────────────────────────────────────────

    #[test]
    fn test_parse_no_tool_calls() {
        let (text, calls) = parse_tool_calls("Just a normal response.");
        assert_eq!(text, "Just a normal response.");
        assert!(calls.is_empty());
    }

    #[test]
    fn test_parse_single_tool_call() {
        let input = "I'll check that.\n<tool_call>{\"name\":\"shell\",\"arguments\":{\"command\":\"ls\"}}</tool_call>";
        let (text, calls) = parse_tool_calls(input);
        assert_eq!(text, "I'll check that.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
        assert_eq!(calls[0].arguments["command"], "ls");
        assert!(calls[0].id.starts_with("cc_"), "ID should have cc_ prefix");
    }

    #[test]
    fn test_parse_multiple_tool_calls() {
        let input = "Running both:\n\
            <tool_call>{\"name\":\"shell\",\"arguments\":{\"command\":\"ls\"}}</tool_call>\n\
            <tool_call>{\"name\":\"read_file\",\"arguments\":{\"path\":\"/tmp/x\"}}</tool_call>";
        let (text, calls) = parse_tool_calls(input);
        assert_eq!(text, "Running both:");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "shell");
        assert_eq!(calls[1].name, "read_file");
        assert_ne!(calls[0].id, calls[1].id, "IDs must be unique");
    }

    #[test]
    fn test_parse_tool_call_with_text_after() {
        let input = "Before\n<tool_call>{\"name\":\"foo\",\"arguments\":{}}</tool_call>\nAfter";
        let (text, calls) = parse_tool_calls(input);
        assert!(text.contains("Before"));
        assert!(text.contains("After"));
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "foo");
    }

    #[test]
    fn test_parse_malformed_json_preserved() {
        let input = "text <tool_call>not json at all</tool_call> more";
        let (text, calls) = parse_tool_calls(input);
        // Malformed block should be preserved in the text.
        assert!(text.contains("not json") || text.contains("<tool_call>"));
        assert!(calls.is_empty());
    }

    #[test]
    fn test_parse_unclosed_tag() {
        let input = "text <tool_call>{\"name\":\"foo\"";
        let (text, calls) = parse_tool_calls(input);
        // Unclosed tag — entire remainder kept.
        assert!(text.contains("<tool_call>"));
        assert!(calls.is_empty());
    }

    #[test]
    fn test_tool_call_ids_unique_across_responses() {
        // This is the exact bug: two parse_tool_calls invocations must
        // produce globally unique IDs, not restart from cc_0.
        let input1 = "<tool_call>{\"name\":\"exec\",\"arguments\":{\"command\":\"ls\"}}</tool_call>";
        let input2 = "<tool_call>{\"name\":\"exec\",\"arguments\":{\"command\":\"pwd\"}}</tool_call>";

        let (_, calls1) = parse_tool_calls(input1);
        let (_, calls2) = parse_tool_calls(input2);

        assert_eq!(calls1.len(), 1);
        assert_eq!(calls2.len(), 1);
        assert_ne!(
            calls1[0].id, calls2[0].id,
            "Tool call IDs must be unique across separate parse_tool_calls invocations"
        );
    }

    // ── JSON response parsing ──────────────────────────────────────

    #[test]
    fn test_parse_json_response_success() {
        let json = r#"{"type":"result","subtype":"success","result":"Hello!","session_id":"abc","is_error":false}"#;
        let (text, is_err) = parse_json_response(json).unwrap();
        assert_eq!(text, "Hello!");
        assert!(!is_err);
    }

    #[test]
    fn test_parse_json_response_error() {
        let json = r#"{"type":"result","subtype":"error","result":"Something failed","is_error":true}"#;
        let (text, is_err) = parse_json_response(json).unwrap();
        assert_eq!(text, "Something failed");
        assert!(is_err);
    }

    #[test]
    fn test_parse_json_response_invalid() {
        let result = parse_json_response("not json");
        assert!(result.is_err());
    }

    // ── Routing prefix stripping ─────────────────────────────────

    #[test]
    fn test_strip_prefix_full() {
        assert_eq!(strip_routing_prefix("claude-code/opus"), "opus");
        assert_eq!(strip_routing_prefix("claude-code/sonnet"), "sonnet");
    }

    #[test]
    fn test_strip_prefix_bare() {
        // "claude-code" alone without sub-model → keep as-is (provider default kicks in)
        assert_eq!(strip_routing_prefix("claude-code"), "claude-code");
    }

    #[test]
    fn test_strip_prefix_not_present() {
        assert_eq!(strip_routing_prefix("opus"), "opus");
        assert_eq!(strip_routing_prefix("claude-opus-4-6"), "claude-opus-4-6");
    }

    // ── Robust JSON parsing ──────────────────────────────────────

    #[test]
    fn test_try_parse_json_valid() {
        let v = try_parse_json(r#"{"name":"exec","arguments":{"cmd":"ls"}}"#);
        assert!(v.is_some());
        assert_eq!(v.unwrap()["name"], "exec");
    }

    #[test]
    fn test_try_parse_json_raw_newlines_in_string() {
        // Model emits literal newlines inside a JSON string value.
        let input = "{\"name\":\"exec\",\"arguments\":{\"command\":\"echo hello\necho world\"}}";
        assert!(
            serde_json::from_str::<Value>(input).is_err(),
            "Raw newlines should fail strict JSON"
        );
        let v = try_parse_json(input);
        assert!(v.is_some(), "Fallback parser should handle raw newlines");
        let parsed = v.unwrap();
        let cmd = parsed["arguments"]["command"].as_str().unwrap();
        assert!(cmd.contains("\\n") || cmd.contains('\n'));
    }

    #[test]
    fn test_try_parse_json_escaped_quotes_preserved() {
        // Already-escaped sequences must not be double-escaped.
        let input = r#"{"name":"exec","arguments":{"cmd":"echo \"hi\""}}"#;
        let v = try_parse_json(input);
        assert!(v.is_some());
        assert!(v.unwrap()["arguments"]["cmd"].as_str().unwrap().contains("\"hi\""));
    }

    #[test]
    fn test_try_parse_json_heredoc_with_fstring() {
        // Reproduces the exact failure from the debug session: Python f-string
        // with embedded braces and escaped quotes, plus literal newlines.
        let input = "{\"name\":\"exec\",\"arguments\":{\"command\":\"python3 << 'EOF'\nimport json\nwith open('test.json') as f:\n    c = json.load(f)\nprint(f'  {\\\"YES\\\" if c.get(\\\"key\\\") else \\\"NO\\\"}')\\nEOF\"}}";
        let v = try_parse_json(input);
        assert!(v.is_some(), "Should parse heredoc with f-string: {:?}", v);
        assert_eq!(v.unwrap()["name"], "exec");
    }

    #[test]
    fn test_try_parse_json_totally_invalid() {
        assert!(try_parse_json("not json").is_none());
        assert!(try_parse_json("").is_none());
    }

    // ── CLI artifact stripping ────────────────────────────────────

    #[test]
    fn test_strip_cli_artifacts_clean() {
        assert_eq!(strip_cli_artifacts("Hello world"), "Hello world");
    }

    #[test]
    fn test_strip_cli_artifacts_iteration_message() {
        let input = "Here's what I found.\n\nI ran out of tool iterations before producing a final answer. The actions above may be incomplete.";
        assert_eq!(strip_cli_artifacts(input), "Here's what I found.");
    }

    #[test]
    fn test_strip_cli_artifacts_only_boilerplate() {
        let input = "I ran out of tool iterations before producing a final answer.";
        assert_eq!(strip_cli_artifacts(input), "");
    }

    #[test]
    fn test_parse_tool_calls_strips_artifacts() {
        let input = "Let me check.\n<tool_call>{\"name\":\"exec\",\"arguments\":{\"cmd\":\"ls\"}}</tool_call>\nI ran out of tool iterations before producing a final answer.";
        let (text, calls) = parse_tool_calls(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(text, "Let me check.");
        assert!(!text.contains("ran out"));
    }

    #[test]
    fn test_parse_tool_call_with_raw_newlines() {
        // Tool call where the model put literal newlines in the command string.
        let input = "<tool_call>{\"name\":\"exec\",\"arguments\":{\"command\":\"echo hello\necho world\"}}</tool_call>";
        let (text, calls) = parse_tool_calls(input);
        assert_eq!(calls.len(), 1, "Should parse despite raw newlines");
        assert_eq!(calls[0].name, "exec");
        assert!(text.is_empty());
    }

    // ── Stream filter ─────────────────────────────────────────────

    #[test]
    fn test_filter_no_tool_calls() {
        let mut f = ToolCallFilter::new();
        assert_eq!(f.feed("Hello world"), "Hello world");
    }

    #[test]
    fn test_filter_strips_tool_call_block() {
        let mut f = ToolCallFilter::new();
        let result = f.feed("Before <tool_call>{\"name\":\"exec\"}</tool_call> After");
        assert_eq!(result, "Before  After");
    }

    #[test]
    fn test_filter_multiple_blocks() {
        let mut f = ToolCallFilter::new();
        let result = f.feed(
            "A <tool_call>{\"name\":\"a\"}</tool_call> B <tool_call>{\"name\":\"b\"}</tool_call> C",
        );
        assert_eq!(result, "A  B  C");
    }

    #[test]
    fn test_filter_split_across_deltas() {
        let mut f = ToolCallFilter::new();
        let r1 = f.feed("Hello <tool_");
        let r2 = f.feed("call>{\"name\":\"x\"}</tool_call> done");
        assert_eq!(format!("{}{}", r1, r2), "Hello  done");
    }

    #[test]
    fn test_filter_partial_tag_not_real() {
        let mut f = ToolCallFilter::new();
        let r1 = f.feed("Use <to");
        let r2 = f.feed("ols wisely");
        assert_eq!(format!("{}{}", r1, r2), "Use <tools wisely");
    }

    #[test]
    fn test_filter_only_tool_calls() {
        let mut f = ToolCallFilter::new();
        let result =
            f.feed("<tool_call>{\"name\":\"exec\",\"arguments\":{}}</tool_call>");
        assert_eq!(result, "");
    }

    #[test]
    fn test_filter_suppresses_cli_artifact_complete() {
        let mut f = ToolCallFilter::new();
        let result = f.feed(
            "Here's the result.\n\nI ran out of tool iterations before producing a final answer.",
        );
        assert_eq!(result, "Here's the result.\n\n");
    }

    #[test]
    fn test_filter_suppresses_artifact_split_across_deltas() {
        let mut f = ToolCallFilter::new();
        // Artifact arrives in two pieces.
        let r1 = f.feed("Done.\n\nI ran out of tool iter");
        let r2 = f.feed("ations before producing a final answer.");
        let combined = format!("{}{}", r1, r2);
        assert!(
            !combined.contains("ran out"),
            "Artifact should be suppressed even when split: {:?}",
            combined
        );
    }

    #[test]
    fn test_filter_emits_non_artifact_starting_with_I() {
        let mut f = ToolCallFilter::new();
        // "I'll check" starts with "I" which is a prefix of "I ran out..."
        // but should be emitted once the non-match is confirmed.
        let r1 = f.feed("I");
        let r2 = f.feed("'ll check that.");
        assert_eq!(format!("{}{}", r1, r2), "I'll check that.");
    }

    #[test]
    fn test_filter_artifact_plus_tool_call() {
        let mut f = ToolCallFilter::new();
        let result = f.feed(
            "Result\n<tool_call>{\"name\":\"x\"}</tool_call>\nI ran out of tool iterations before producing a final answer.",
        );
        assert_eq!(result, "Result\n\n");
    }

    // ── Provider creation ──────────────────────────────────────────

    #[test]
    fn test_provider_default_model() {
        let p = ClaudeCodeProvider::new(None);
        assert_eq!(p.get_default_model(), "opus");
    }

    #[test]
    fn test_provider_custom_model() {
        let p = ClaudeCodeProvider::new(Some("sonnet"));
        assert_eq!(p.get_default_model(), "sonnet");
    }
}
