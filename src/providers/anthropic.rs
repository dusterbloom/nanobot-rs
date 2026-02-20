//! Native Anthropic Messages API provider.
//!
//! Speaks the Anthropic Messages API (`POST /v1/messages`) directly, translating
//! between OpenAI-style tool/message formats used internally by nanobot and the
//! Anthropic-native format. Used with OAuth tokens from Claude Max subscriptions
//! where the OpenAI-compat endpoint doesn't work.

use std::collections::HashMap;

use anyhow::{Context, Result};
use async_trait::async_trait;
use futures_util::StreamExt;
use reqwest::Client;
use serde_json::{json, Value};
use tracing::{debug, warn};

use super::base::{LLMProvider, LLMResponse, StreamChunk, StreamHandle, ToolCallRequest};

const ANTHROPIC_API_BASE: &str = "https://api.anthropic.com";
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Beta flags required for OAuth tokens to work on the Messages API.
const OAUTH_BETA: &str = "claude-code-20250219,oauth-2025-04-20,fine-grained-tool-streaming-2025-05-14,interleaved-thinking-2025-05-14";
/// User-Agent that identifies as Claude Code CLI (required for OAuth).
const CLAUDE_CODE_UA: &str = "claude-cli/2.1.2 (external, cli)";
/// System prompt prefix required for OAuth (Claude Code identity).
const CLAUDE_CODE_IDENTITY: &str = "You are Claude Code, Anthropic's official CLI for Claude.";

/// Returns true if the key is an OAuth token (from Claude CLI credentials).
/// OAuth tokens start with `sk-ant-oat` and must use `Authorization: Bearer`,
/// not the `x-api-key` header, plus additional identity headers.
fn is_oauth_token(key: &str) -> bool {
    key.starts_with("sk-ant-oat")
}

/// Provider that talks to the Anthropic Messages API natively.
pub struct AnthropicProvider {
    api_key: String,
    default_model: String,
    client: Client,
}

impl AnthropicProvider {
    pub fn new(api_key: &str, default_model: Option<&str>) -> Self {
        let model = default_model.unwrap_or("claude-opus-4-6");
        // Strip provider prefix if present (e.g. "anthropic/claude-opus-4-6" → "claude-opus-4-6").
        let model = model.split('/').last().unwrap_or(model);
        // Normalize short names to canonical Anthropic model IDs.
        let model = normalize_claude_model(model);
        Self {
            api_key: api_key.to_string(),
            default_model: model,
            client: Client::new(),
        }
    }
}

/// Normalize Claude model short-names to canonical Anthropic API model IDs.
///
/// - `"opus"` → `"claude-opus-4-6"`
/// - `"opus-4-6"` → `"claude-opus-4-6"`
/// - `"sonnet"` → `"claude-sonnet-4-5-20250929"`
/// - Already-qualified names pass through unchanged.
fn normalize_claude_model(name: &str) -> String {
    let lower = name.to_lowercase();

    // Already qualified.
    if lower.starts_with("claude-") {
        return name.to_string();
    }

    // Bare aliases.
    match lower.as_str() {
        "opus" => return "claude-opus-4-6".to_string(),
        "sonnet" => return "claude-sonnet-4-5-20250929".to_string(),
        "haiku" => return "claude-haiku-4-5-20251001".to_string(),
        _ => {}
    }

    // Partial names without "claude-" prefix (e.g. "opus-4-6", "sonnet-4-5-20250929").
    if lower.starts_with("opus") || lower.starts_with("sonnet") || lower.starts_with("haiku") {
        return format!("claude-{}", name);
    }

    name.to_string()
}

// ---------------------------------------------------------------------------
// Request translation: OpenAI format → Anthropic format
// ---------------------------------------------------------------------------

/// Extract the system prompt from messages and convert the rest to Anthropic format.
///
/// OpenAI format has `role: "system"` messages inline.
/// Anthropic format has a top-level `system` field and different content block structures.
fn translate_messages(messages: &[Value]) -> (Option<String>, Vec<Value>) {
    let mut system_parts: Vec<String> = Vec::new();
    let mut anthropic_messages: Vec<Value> = Vec::new();

    for msg in messages {
        let role = msg["role"].as_str().unwrap_or("");

        match role {
            "system" => {
                // Collect system messages into the top-level system field.
                if let Some(text) = extract_text_content(msg) {
                    system_parts.push(text);
                }
            }
            "user" => {
                let content = translate_user_content(msg);
                anthropic_messages.push(json!({ "role": "user", "content": content }));
            }
            "assistant" => {
                let content = translate_assistant_content(msg);
                anthropic_messages.push(json!({ "role": "assistant", "content": content }));
            }
            "tool" => {
                // OpenAI tool results become user messages with tool_result content blocks.
                let tool_call_id = msg["tool_call_id"].as_str().unwrap_or("");
                let text = extract_text_content(msg).unwrap_or_default();

                let block = json!({
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": text,
                });

                // Merge consecutive tool results into the same user message.
                if let Some(last) = anthropic_messages.last_mut() {
                    if last["role"].as_str() == Some("user") {
                        if let Some(arr) = last["content"].as_array_mut() {
                            arr.push(block);
                            continue;
                        }
                    }
                }
                anthropic_messages.push(json!({ "role": "user", "content": [block] }));
            }
            _ => {
                // Unknown role — treat as user.
                let text = extract_text_content(msg).unwrap_or_default();
                anthropic_messages.push(json!({ "role": "user", "content": text }));
            }
        }
    }

    let system = if system_parts.is_empty() {
        None
    } else {
        Some(system_parts.join("\n"))
    };

    (system, anthropic_messages)
}

/// Extract text content from a message (handles both string and array content).
fn extract_text_content(msg: &Value) -> Option<String> {
    if let Some(s) = msg["content"].as_str() {
        return Some(s.to_string());
    }
    if let Some(arr) = msg["content"].as_array() {
        let texts: Vec<&str> = arr
            .iter()
            .filter_map(|block| {
                if block["type"].as_str() == Some("text") {
                    block["text"].as_str()
                } else {
                    block["text"].as_str().or_else(|| block.as_str())
                }
            })
            .collect();
        if !texts.is_empty() {
            return Some(texts.join("\n"));
        }
    }
    None
}

/// Translate user message content to Anthropic format.
fn translate_user_content(msg: &Value) -> Value {
    // Simple string content passes through.
    if msg["content"].is_string() {
        return msg["content"].clone();
    }
    // Array content — pass through (Anthropic accepts text blocks).
    if let Some(arr) = msg["content"].as_array() {
        let blocks: Vec<Value> = arr
            .iter()
            .map(|block| {
                if block["type"].as_str() == Some("text") {
                    block.clone()
                } else if let Some(s) = block.as_str() {
                    json!({ "type": "text", "text": s })
                } else {
                    block.clone()
                }
            })
            .collect();
        return Value::Array(blocks);
    }
    json!("")
}

/// Translate assistant message content to Anthropic format.
///
/// Handles tool_calls: OpenAI `tool_calls` array → Anthropic `tool_use` content blocks.
fn translate_assistant_content(msg: &Value) -> Value {
    let mut blocks: Vec<Value> = Vec::new();

    // Add text content if present.
    if let Some(text) = extract_text_content(msg) {
        if !text.is_empty() {
            blocks.push(json!({ "type": "text", "text": text }));
        }
    }

    // Convert OpenAI tool_calls to Anthropic tool_use blocks.
    if let Some(tool_calls) = msg["tool_calls"].as_array() {
        for tc in tool_calls {
            let id = tc["id"].as_str().unwrap_or("");
            let name = tc["function"]["name"].as_str().unwrap_or("");
            let args = &tc["function"]["arguments"];

            // Arguments may be a JSON string or an object.
            let input: Value = if let Some(s) = args.as_str() {
                serde_json::from_str(s).unwrap_or(json!({}))
            } else if args.is_object() {
                args.clone()
            } else {
                json!({})
            };

            blocks.push(json!({
                "type": "tool_use",
                "id": id,
                "name": name,
                "input": input,
            }));
        }
    }

    if blocks.is_empty() {
        json!("")
    } else {
        Value::Array(blocks)
    }
}

/// Convert OpenAI-style tool definitions to Anthropic format.
///
/// OpenAI: `{ type: "function", function: { name, description, parameters } }`
/// Anthropic: `{ name, description, input_schema }`
fn translate_tools(tools: &[Value]) -> Vec<Value> {
    tools
        .iter()
        .filter_map(|tool| {
            let func = &tool["function"];
            let name = func["name"].as_str()?;
            let description = func["description"].as_str().unwrap_or("");
            let parameters = &func["parameters"];

            Some(json!({
                "name": name,
                "description": description,
                "input_schema": parameters,
            }))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Response translation: Anthropic format → internal LLMResponse
// ---------------------------------------------------------------------------

/// Parse an Anthropic Messages API response into our internal format.
fn parse_anthropic_response(data: &Value) -> Result<LLMResponse> {
    let mut content_text = String::new();
    let mut tool_calls: Vec<ToolCallRequest> = Vec::new();

    if let Some(content) = data["content"].as_array() {
        for block in content {
            match block["type"].as_str() {
                Some("text") => {
                    if let Some(text) = block["text"].as_str() {
                        if !content_text.is_empty() {
                            content_text.push('\n');
                        }
                        content_text.push_str(text);
                    }
                }
                Some("tool_use") => {
                    let id = block["id"].as_str().unwrap_or("").to_string();
                    let name = block["name"].as_str().unwrap_or("").to_string();
                    let input = &block["input"];
                    let arguments: HashMap<String, Value> = input
                        .as_object()
                        .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                        .unwrap_or_default();

                    tool_calls.push(ToolCallRequest {
                        id,
                        name,
                        arguments,
                    });
                }
                _ => {}
            }
        }
    }

    // Map Anthropic stop_reason to OpenAI finish_reason.
    let stop_reason = data["stop_reason"].as_str().unwrap_or("end_turn");
    let finish_reason = match stop_reason {
        "tool_use" => "tool_calls",
        "end_turn" | "stop_sequence" => "stop",
        "max_tokens" => "length",
        other => other,
    };

    let mut usage = HashMap::new();
    if let Some(u) = data["usage"].as_object() {
        if let Some(n) = u.get("input_tokens").and_then(|v| v.as_i64()) {
            usage.insert("prompt_tokens".to_string(), n);
        }
        if let Some(n) = u.get("output_tokens").and_then(|v| v.as_i64()) {
            usage.insert("completion_tokens".to_string(), n);
        }
    }

    Ok(LLMResponse {
        content: if content_text.is_empty() {
            None
        } else {
            Some(content_text)
        },
        tool_calls,
        finish_reason: finish_reason.to_string(),
        usage,
    })
}

// ---------------------------------------------------------------------------
// LLMProvider implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl LLMProvider for AnthropicProvider {
    async fn chat(
        &self,
        messages: &[Value],
        tools: Option<&[Value]>,
        model: Option<&str>,
        max_tokens: u32,
        temperature: f64,
        thinking_budget: Option<u32>,
    ) -> Result<LLMResponse> {
        let normalized = model.map(|m| normalize_claude_model(m.split('/').last().unwrap_or(m)));
        let model = normalized.as_deref().unwrap_or(&self.default_model);
        let url = format!("{}/v1/messages", ANTHROPIC_API_BASE);

        let oauth = is_oauth_token(&self.api_key);
        let (system, anthropic_messages) = translate_messages(messages);

        let mut body = json!({
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        });

        // Extended thinking: add thinking block with budget
        if let Some(budget) = thinking_budget {
            body["thinking"] = json!({
                "type": "enabled",
                "budget_tokens": budget,
            });
            // Anthropic requires temperature=1 when thinking is enabled
            body["temperature"] = json!(1);
        }

        // OAuth requires Claude Code identity in the system prompt.
        if oauth {
            let mut sys_blocks = vec![json!({
                "type": "text",
                "text": CLAUDE_CODE_IDENTITY,
            })];
            if let Some(system_text) = &system {
                sys_blocks.push(json!({
                    "type": "text",
                    "text": system_text,
                }));
            }
            body["system"] = Value::Array(sys_blocks);
        } else if let Some(system_text) = &system {
            body["system"] = json!(system_text);
        }

        if let Some(tool_defs) = tools {
            if !tool_defs.is_empty() {
                body["tools"] = Value::Array(translate_tools(tool_defs));
            }
        }

        debug!(
            "AnthropicProvider::chat model={} oauth={} messages={}",
            model,
            oauth,
            anthropic_messages.len()
        );

        let mut req = self
            .client
            .post(&url)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("Content-Type", "application/json");

        if oauth {
            req = req
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("anthropic-beta", OAUTH_BETA)
                .header("user-agent", CLAUDE_CODE_UA)
                .header("x-app", "cli");
        } else {
            req = req.header("x-api-key", &self.api_key);
        }

        let response = match req.json(&body).send().await {
            Ok(r) => r,
            Err(e) => {
                warn!("Anthropic API request failed: {}", e);
                return Ok(LLMResponse {
                    content: Some(format!("Error calling Anthropic API: {}", e)),
                    tool_calls: Vec::new(),
                    finish_reason: "error".to_string(),
                    usage: HashMap::new(),
                });
            }
        };

        let status = response.status();
        let response_text = response.text().await.unwrap_or_default();

        if !status.is_success() {
            warn!("Anthropic API returned {} : {}", status, response_text);
            return Ok(LLMResponse {
                content: Some(format!(
                    "Error calling Anthropic API (HTTP {}): {}",
                    status, response_text
                )),
                tool_calls: Vec::new(),
                finish_reason: "error".to_string(),
                usage: HashMap::new(),
            });
        }

        let data: Value = serde_json::from_str(&response_text)
            .context("Failed to parse Anthropic API response")?;

        parse_anthropic_response(&data)
    }

    async fn chat_stream(
        &self,
        messages: &[Value],
        tools: Option<&[Value]>,
        model: Option<&str>,
        max_tokens: u32,
        temperature: f64,
        thinking_budget: Option<u32>,
    ) -> Result<StreamHandle> {
        let normalized = model.map(|m| normalize_claude_model(m.split('/').last().unwrap_or(m)));
        let model = normalized.as_deref().unwrap_or(&self.default_model);
        let url = format!("{}/v1/messages", ANTHROPIC_API_BASE);

        let oauth = is_oauth_token(&self.api_key);
        let (system, anthropic_messages) = translate_messages(messages);

        let mut body = json!({
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": true,
        });

        // Extended thinking: add thinking block with budget
        if let Some(budget) = thinking_budget {
            body["thinking"] = json!({
                "type": "enabled",
                "budget_tokens": budget,
            });
            // Anthropic requires temperature=1 when thinking is enabled
            body["temperature"] = json!(1);
        }

        // OAuth requires Claude Code identity in the system prompt.
        if oauth {
            let mut sys_blocks = vec![json!({
                "type": "text",
                "text": CLAUDE_CODE_IDENTITY,
            })];
            if let Some(system_text) = &system {
                sys_blocks.push(json!({
                    "type": "text",
                    "text": system_text,
                }));
            }
            body["system"] = Value::Array(sys_blocks);
        } else if let Some(system_text) = &system {
            body["system"] = json!(system_text);
        }

        if let Some(tool_defs) = tools {
            if !tool_defs.is_empty() {
                body["tools"] = Value::Array(translate_tools(tool_defs));
            }
        }

        debug!(
            "AnthropicProvider::chat_stream model={} oauth={} messages={}",
            model,
            oauth,
            anthropic_messages.len()
        );

        let mut req = self
            .client
            .post(&url)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("Content-Type", "application/json");

        if oauth {
            req = req
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("anthropic-beta", OAUTH_BETA)
                .header("user-agent", CLAUDE_CODE_UA)
                .header("x-app", "cli");
        } else {
            req = req.header("x-api-key", &self.api_key);
        }

        let response = req.json(&body).send().await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            warn!(
                "Anthropic streaming API returned {}: {}",
                status, error_text
            );
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
            let _ = tx.send(StreamChunk::Done(LLMResponse {
                content: Some(format!(
                    "Error calling Anthropic API (HTTP {}): {}",
                    status, error_text
                )),
                tool_calls: Vec::new(),
                finish_reason: "error".to_string(),
                usage: HashMap::new(),
            }));
            return Ok(StreamHandle { rx });
        }

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let byte_stream = response.bytes_stream();

        tokio::spawn(async move {
            parse_anthropic_sse(byte_stream, tx).await;
        });

        Ok(StreamHandle { rx })
    }

    fn get_default_model(&self) -> &str {
        &self.default_model
    }
}

// ---------------------------------------------------------------------------
// SSE stream parsing for Anthropic Messages API
// ---------------------------------------------------------------------------

/// Parse Anthropic SSE streaming events.
///
/// Events:
/// - `message_start` — contains initial message metadata
/// - `content_block_start` — new content block (text or tool_use)
/// - `content_block_delta` — incremental content (text_delta or input_json_delta)
/// - `content_block_stop` — block complete
/// - `message_delta` — stop_reason, usage
/// - `message_stop` — stream complete
async fn parse_anthropic_sse(
    byte_stream: impl futures_util::Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Unpin,
    tx: tokio::sync::mpsc::UnboundedSender<StreamChunk>,
) {
    let mut line_buffer = String::new();
    let mut full_content = String::new();
    let mut finish_reason = String::from("stop");
    let mut usage: HashMap<String, i64> = HashMap::new();

    // Track content blocks by index.
    // For tool_use blocks: (id, name, accumulated_json_str)
    let mut tool_blocks: HashMap<u64, (String, String, String)> = HashMap::new();
    let mut current_block_index: u64 = 0;
    #[allow(unused_assignments)]
    let mut current_block_type = String::new();

    let mut stream = Box::pin(byte_stream);

    while let Some(result) = stream.next().await {
        let bytes = match result {
            Ok(b) => b,
            Err(e) => {
                warn!("Anthropic SSE stream error: {}", e);
                break;
            }
        };

        let text = String::from_utf8_lossy(&bytes);
        line_buffer.push_str(&text);

        while let Some(newline_pos) = line_buffer.find('\n') {
            let line = line_buffer[..newline_pos]
                .trim_end_matches('\r')
                .to_string();
            line_buffer = line_buffer[newline_pos + 1..].to_string();

            if line.is_empty() || line.starts_with("event:") {
                // event: lines tell us the type, but we can infer from the data.
                continue;
            }

            if !line.starts_with("data: ") {
                continue;
            }

            let data_str = &line[6..];
            let data: Value = match serde_json::from_str(data_str) {
                Ok(v) => v,
                Err(_) => continue,
            };

            let event_type = data["type"].as_str().unwrap_or("");

            match event_type {
                "message_start" => {
                    // Extract initial usage if present.
                    if let Some(u) = data["message"]["usage"].as_object() {
                        if let Some(n) = u.get("input_tokens").and_then(|v| v.as_i64()) {
                            usage.insert("prompt_tokens".to_string(), n);
                        }
                    }
                }
                "content_block_start" => {
                    current_block_index = data["index"].as_u64().unwrap_or(current_block_index);
                    let block = &data["content_block"];
                    current_block_type = block["type"].as_str().unwrap_or("text").to_string();

                    if current_block_type == "tool_use" {
                        let id = block["id"].as_str().unwrap_or("").to_string();
                        let name = block["name"].as_str().unwrap_or("").to_string();
                        tool_blocks.insert(current_block_index, (id, name, String::new()));
                    }
                }
                "content_block_delta" => {
                    let delta = &data["delta"];
                    let delta_type = delta["type"].as_str().unwrap_or("");

                    match delta_type {
                        "text_delta" => {
                            if let Some(text) = delta["text"].as_str() {
                                full_content.push_str(text);
                                let _ = tx.send(StreamChunk::TextDelta(text.to_string()));
                            }
                        }
                        "thinking_delta" => {
                            if let Some(thinking) = delta["thinking"].as_str() {
                                let _ = tx.send(StreamChunk::ThinkingDelta(thinking.to_string()));
                            }
                        }
                        "input_json_delta" => {
                            // Accumulate tool input JSON.
                            if let Some(partial) = delta["partial_json"].as_str() {
                                if let Some(entry) = tool_blocks.get_mut(&current_block_index) {
                                    entry.2.push_str(partial);
                                }
                            }
                        }
                        _ => {}
                    }
                }
                "content_block_stop" => {
                    // Block finished — nothing to do, tool_blocks already accumulated.
                }
                "message_delta" => {
                    if let Some(sr) = data["delta"]["stop_reason"].as_str() {
                        finish_reason = match sr {
                            "tool_use" => "tool_calls".to_string(),
                            "end_turn" | "stop_sequence" => "stop".to_string(),
                            "max_tokens" => "length".to_string(),
                            other => other.to_string(),
                        };
                    }
                    if let Some(u) = data["usage"].as_object() {
                        if let Some(n) = u.get("output_tokens").and_then(|v| v.as_i64()) {
                            usage.insert("completion_tokens".to_string(), n);
                        }
                    }
                }
                "message_stop" => {
                    // Stream complete — assemble final response.
                    let content = if full_content.is_empty() {
                        None
                    } else {
                        Some(full_content.clone())
                    };

                    let mut tool_calls = Vec::new();
                    let mut indices: Vec<u64> = tool_blocks.keys().copied().collect();
                    indices.sort();
                    for idx in indices {
                        let (id, name, args_str) = tool_blocks.remove(&idx).unwrap();
                        let arguments: HashMap<String, Value> =
                            serde_json::from_str(&args_str).unwrap_or_default();
                        tool_calls.push(ToolCallRequest {
                            id,
                            name,
                            arguments,
                        });
                    }

                    let _ = tx.send(StreamChunk::Done(LLMResponse {
                        content,
                        tool_calls,
                        finish_reason: finish_reason.clone(),
                        usage: usage.clone(),
                    }));
                    return;
                }
                _ => {}
            }
        }
    }

    // Stream ended without message_stop — emit what we have.
    let content = if full_content.is_empty() {
        None
    } else {
        Some(full_content)
    };

    let mut tool_calls = Vec::new();
    let mut indices: Vec<u64> = tool_blocks.keys().copied().collect();
    indices.sort();
    for idx in indices {
        let (id, name, args_str) = tool_blocks.remove(&idx).unwrap();
        let arguments: HashMap<String, Value> = serde_json::from_str(&args_str).unwrap_or_default();
        tool_calls.push(ToolCallRequest {
            id,
            name,
            arguments,
        });
    }

    let _ = tx.send(StreamChunk::Done(LLMResponse {
        content,
        tool_calls,
        finish_reason,
        usage,
    }));
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_translate_system_message() {
        let messages = vec![json!({"role": "system", "content": "You are helpful."})];
        let (system, msgs) = translate_messages(&messages);
        assert_eq!(system, Some("You are helpful.".to_string()));
        assert!(msgs.is_empty());
    }

    #[test]
    fn test_translate_user_assistant() {
        let messages = vec![
            json!({"role": "system", "content": "Be concise."}),
            json!({"role": "user", "content": "Hello"}),
            json!({"role": "assistant", "content": "Hi!"}),
        ];
        let (system, msgs) = translate_messages(&messages);
        assert_eq!(system, Some("Be concise.".to_string()));
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[0]["content"], "Hello");
        assert_eq!(msgs[1]["role"], "assistant");
    }

    #[test]
    fn test_translate_tool_calls() {
        let messages = vec![json!({
            "role": "assistant",
            "content": "I'll read that.",
            "tool_calls": [{
                "id": "tc_1",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": "{\"path\": \"/tmp/test\"}"
                }
            }]
        })];
        let (_, msgs) = translate_messages(&messages);
        assert_eq!(msgs.len(), 1);
        let content = msgs[0]["content"].as_array().unwrap();
        assert_eq!(content.len(), 2); // text + tool_use
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[1]["type"], "tool_use");
        assert_eq!(content[1]["name"], "read_file");
        assert_eq!(content[1]["id"], "tc_1");
    }

    #[test]
    fn test_translate_tool_result() {
        let messages = vec![json!({
            "role": "tool",
            "tool_call_id": "tc_1",
            "content": "file contents"
        })];
        let (_, msgs) = translate_messages(&messages);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["role"], "user");
        let content = msgs[0]["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "tool_result");
        assert_eq!(content[0]["tool_use_id"], "tc_1");
        assert_eq!(content[0]["content"], "file contents");
    }

    #[test]
    fn test_translate_consecutive_tool_results_merged() {
        let messages = vec![
            json!({"role": "tool", "tool_call_id": "tc_1", "content": "result 1"}),
            json!({"role": "tool", "tool_call_id": "tc_2", "content": "result 2"}),
        ];
        let (_, msgs) = translate_messages(&messages);
        // Both should merge into a single user message.
        assert_eq!(msgs.len(), 1);
        let content = msgs[0]["content"].as_array().unwrap();
        assert_eq!(content.len(), 2);
    }

    #[test]
    fn test_translate_tools() {
        let tools = vec![json!({
            "type": "function",
            "function": {
                "name": "shell",
                "description": "Run a command",
                "parameters": {
                    "type": "object",
                    "properties": { "command": { "type": "string" } },
                    "required": ["command"]
                }
            }
        })];
        let result = translate_tools(&tools);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0]["name"], "shell");
        assert_eq!(result[0]["description"], "Run a command");
        assert!(result[0]["input_schema"]["properties"]["command"].is_object());
    }

    #[test]
    fn test_parse_anthropic_response_text() {
        let data = json!({
            "content": [
                {"type": "text", "text": "Hello world"}
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        });
        let resp = parse_anthropic_response(&data).unwrap();
        assert_eq!(resp.content, Some("Hello world".to_string()));
        assert!(resp.tool_calls.is_empty());
        assert_eq!(resp.finish_reason, "stop");
        assert_eq!(resp.usage["prompt_tokens"], 10);
        assert_eq!(resp.usage["completion_tokens"], 5);
    }

    #[test]
    fn test_parse_anthropic_response_tool_use() {
        let data = json!({
            "content": [
                {"type": "text", "text": "I'll check."},
                {
                    "type": "tool_use",
                    "id": "tu_1",
                    "name": "shell",
                    "input": {"command": "ls"}
                }
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 20}
        });
        let resp = parse_anthropic_response(&data).unwrap();
        assert_eq!(resp.content, Some("I'll check.".to_string()));
        assert_eq!(resp.tool_calls.len(), 1);
        assert_eq!(resp.tool_calls[0].name, "shell");
        assert_eq!(resp.tool_calls[0].id, "tu_1");
        assert_eq!(resp.tool_calls[0].arguments["command"], "ls");
        assert_eq!(resp.finish_reason, "tool_calls");
    }

    #[test]
    fn test_parse_anthropic_response_multiple_tool_uses() {
        let data = json!({
            "content": [
                {
                    "type": "tool_use",
                    "id": "tu_1",
                    "name": "read_file",
                    "input": {"path": "/a"}
                },
                {
                    "type": "tool_use",
                    "id": "tu_2",
                    "name": "read_file",
                    "input": {"path": "/b"}
                }
            ],
            "stop_reason": "tool_use",
            "usage": {}
        });
        let resp = parse_anthropic_response(&data).unwrap();
        assert_eq!(resp.tool_calls.len(), 2);
        assert_eq!(resp.tool_calls[0].id, "tu_1");
        assert_eq!(resp.tool_calls[1].id, "tu_2");
    }

    #[test]
    fn test_provider_default_model() {
        let p = AnthropicProvider::new("test-key", None);
        assert_eq!(p.get_default_model(), "claude-opus-4-6");
    }

    #[test]
    fn test_provider_strips_prefix() {
        let p = AnthropicProvider::new("test-key", Some("anthropic/claude-sonnet-4-5"));
        assert_eq!(p.get_default_model(), "claude-sonnet-4-5");
    }

    #[test]
    fn test_normalize_short_names() {
        assert_eq!(normalize_claude_model("opus"), "claude-opus-4-6");
        assert_eq!(
            normalize_claude_model("sonnet"),
            "claude-sonnet-4-5-20250929"
        );
        assert_eq!(normalize_claude_model("haiku"), "claude-haiku-4-5-20251001");
    }

    #[test]
    fn test_normalize_partial_names() {
        assert_eq!(normalize_claude_model("opus-4-6"), "claude-opus-4-6");
        assert_eq!(
            normalize_claude_model("sonnet-4-5-20250929"),
            "claude-sonnet-4-5-20250929"
        );
    }

    #[test]
    fn test_normalize_already_qualified() {
        assert_eq!(normalize_claude_model("claude-opus-4-6"), "claude-opus-4-6");
        assert_eq!(
            normalize_claude_model("claude-sonnet-4-20250514"),
            "claude-sonnet-4-20250514"
        ); // legacy passthrough
    }

    #[test]
    fn test_is_oauth_token() {
        assert!(is_oauth_token("sk-ant-oat01-abc123"));
        assert!(is_oauth_token("sk-ant-oat02-xyz"));
        assert!(!is_oauth_token("sk-ant-api03-abc123"));
        assert!(!is_oauth_token("sk-proj-abc123"));
        assert!(!is_oauth_token(""));
    }

    #[test]
    fn test_normalize_non_claude() {
        assert_eq!(normalize_claude_model("gpt-4o"), "gpt-4o");
        assert_eq!(normalize_claude_model("llama-3"), "llama-3");
    }

    #[test]
    fn test_provider_normalizes_short_name() {
        let p = AnthropicProvider::new("test-key", Some("opus-4-6"));
        assert_eq!(p.get_default_model(), "claude-opus-4-6");
    }

    #[test]
    fn test_full_conversation_roundtrip() {
        // Simulate a full conversation with system, user, assistant with tool call,
        // tool result, and user follow-up.
        let messages = vec![
            json!({"role": "system", "content": "You are a helpful assistant."}),
            json!({"role": "user", "content": "List files in /tmp"}),
            json!({
                "role": "assistant",
                "content": "I'll list the files.",
                "tool_calls": [{
                    "id": "tc_1",
                    "type": "function",
                    "function": { "name": "shell", "arguments": "{\"command\": \"ls /tmp\"}" }
                }]
            }),
            json!({"role": "tool", "tool_call_id": "tc_1", "content": "file1.txt\nfile2.txt"}),
            json!({"role": "user", "content": "Thanks!"}),
        ];

        let (system, msgs) = translate_messages(&messages);
        assert_eq!(system, Some("You are a helpful assistant.".to_string()));
        assert_eq!(msgs.len(), 4); // user, assistant, user(tool_result), user
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[1]["role"], "assistant");
        assert_eq!(msgs[2]["role"], "user"); // tool result
        assert_eq!(msgs[3]["role"], "user"); // follow-up
    }
}
