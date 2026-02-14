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

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tracing::{debug, warn};

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
         You have access to the following tools. To call a tool, output a JSON \
         object wrapped in <tool_call></tool_call> tags:\n\n\
         <tool_call>{\"name\": \"tool_name\", \"arguments\": {\"arg\": \"value\"}}</tool_call>\n\n\
         You may include text before, between, or after tool calls. \
         Include multiple <tool_call> blocks for parallel calls.\n\n\
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

/// Parse `<tool_call>` blocks from Claude's text response.
///
/// Returns (clean_text, tool_calls) where clean_text has the blocks removed.
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
            if let Ok(parsed) = serde_json::from_str::<Value>(json_str) {
                let name = parsed["name"]
                    .as_str()
                    .unwrap_or("unknown")
                    .to_string();
                let arguments: HashMap<String, Value> = parsed["arguments"]
                    .as_object()
                    .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                    .unwrap_or_default();

                calls.push(ToolCallRequest {
                    id: format!("cc_{}", calls.len()),
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

    (clean.trim().to_string(), calls)
}

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
                                let _ = tx.send(StreamChunk::TextDelta(text.to_string()));
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
                                        let _ =
                                            tx.send(StreamChunk::TextDelta(t.to_string()));
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
        assert_eq!(calls[0].id, "cc_0");
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
        assert_eq!(calls[1].id, "cc_1");
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
        let input = "text <tool_call>not json</tool_call> more";
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
