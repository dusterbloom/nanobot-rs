//! OpenAI-compatible API provider.
//!
//! Replaces LiteLLMProvider by calling OpenAI-compatible APIs directly via reqwest.
//! Supports OpenRouter, Anthropic (OpenAI-compat endpoint), OpenAI, DeepSeek,
//! Groq, vLLM, and any other provider that implements the OpenAI chat completions
//! API format.

use std::collections::HashMap;

use anyhow::Result;
use async_trait::async_trait;
use futures_util::StreamExt;
use reqwest::Client;
use tracing::{debug, warn};

use super::base::{LLMProvider, LLMResponse, StreamChunk, StreamHandle, ToolCallRequest};

/// An LLM provider that talks to any OpenAI-compatible chat completions endpoint.
pub struct OpenAICompatProvider {
    api_key: String,
    api_base: String,
    default_model: String,
    client: Client,
}

/// Normalize Claude model short-names so the API always gets the canonical ID.
///
/// - `"opus"` / `"sonnet"` / `"haiku"` → latest canonical ID
/// - `"opus-4-6"`, `"sonnet-4-5-..."` etc. → prepend `claude-`
/// - Already-qualified names or non-Claude models pass through unchanged.
fn normalize_model_name(name: &str) -> String {
    let lower = name.to_lowercase();

    // Already has a provider prefix (e.g. "anthropic/claude-opus-4-6") or
    // already starts with "claude-" — pass through.
    if lower.contains('/') || lower.starts_with("claude-") {
        return name.to_string();
    }

    // Short aliases (bare word).
    match lower.as_str() {
        "opus" => return "claude-opus-4-6".to_string(),
        "sonnet" => return "claude-sonnet-4-5-20250929".to_string(),
        "haiku" => return "claude-haiku-4-5-20251001".to_string(),
        "local" => return name.to_string(),
        _ => {}
    }

    // Claude model without the "claude-" prefix (e.g. "opus-4-6", "sonnet-4-5-20250929").
    if lower.starts_with("opus")
        || lower.starts_with("sonnet")
        || lower.starts_with("haiku")
    {
        return format!("claude-{}", name);
    }

    name.to_string()
}

impl OpenAICompatProvider {
    /// Returns true when the provider supports Anthropic-style `cache_control`
    /// breakpoints. Currently: direct Anthropic API and OpenRouter.
    fn supports_cache_control(&self, model: &str) -> bool {
        let is_anthropic_direct = self.api_base.contains("anthropic");
        let is_openrouter = self.api_base.contains("openrouter");
        let is_claude_model = model.contains("claude") || model.contains("anthropic");
        is_anthropic_direct || (is_openrouter && is_claude_model)
    }

    /// Create a new provider.
    ///
    /// Provider detection logic (porting from `LiteLLMProvider.__init__`):
    /// - OpenRouter: detected by `sk-or-` key prefix or `openrouter` in api_base
    /// - DeepSeek: detected by `deepseek` in the default model name
    /// - vLLM / custom: when an explicit `api_base` is provided that isn't OpenRouter
    /// - Default fallback: OpenRouter (`https://openrouter.ai/api/v1`)
    pub fn new(api_key: &str, api_base: Option<&str>, default_model: Option<&str>) -> Self {
        let default_model = normalize_model_name(
            default_model.unwrap_or("anthropic/claude-opus-4-5"),
        );

        let resolved_base = if let Some(base) = api_base {
            // Use whatever was explicitly provided.
            base.trim_end_matches('/').to_string()
        } else if api_key.starts_with("sk-or-") {
            "https://openrouter.ai/api/v1".to_string()
        } else if api_key.starts_with("sk-ant-") {
            "https://api.anthropic.com/v1".to_string()
        } else if default_model.contains("deepseek") {
            "https://api.deepseek.com".to_string()
        } else if api_key.starts_with("gsk_") || default_model.contains("groq") {
            "https://api.groq.com/openai/v1".to_string()
        } else if api_key.starts_with("sk-") && !default_model.contains('/') {
            // Bare "sk-" prefix with a non-routed model name -> likely OpenAI direct.
            "https://api.openai.com/v1".to_string()
        } else {
            // Fallback: OpenRouter (supports routed model names like "anthropic/claude-...").
            "https://openrouter.ai/api/v1".to_string()
        };

        Self {
            api_key: api_key.to_string(),
            api_base: resolved_base,
            default_model,
            client: Client::new(),
        }
    }
}

#[async_trait]
impl LLMProvider for OpenAICompatProvider {
    async fn chat(
        &self,
        messages: &[serde_json::Value],
        tools: Option<&[serde_json::Value]>,
        model: Option<&str>,
        max_tokens: u32,
        temperature: f64,
        _thinking_budget: Option<u32>,
    ) -> Result<LLMResponse> {
        let normalized = model.map(|m| normalize_model_name(m));
        let raw_model = normalized.as_deref().unwrap_or(&self.default_model);
        // Strip "provider/" prefix for non-OpenRouter APIs (e.g. "anthropic/claude-opus-4-5"
        // becomes "claude-opus-4-5" when hitting api.anthropic.com directly).
        let model = if !self.api_base.contains("openrouter") {
            raw_model.split('/').last().unwrap_or(raw_model)
        } else {
            raw_model
        };
        let url = format!("{}/chat/completions", self.api_base);

        // Inject cache_control breakpoints for Anthropic prompt caching.
        let (cached_msgs, cached_tools) = if self.supports_cache_control(model) {
            inject_cache_control(messages, tools)
        } else {
            (messages.to_vec(), tools.map(|t| t.to_vec()))
        };

        // Build request body.
        let mut body = serde_json::json!({
            "model": model,
            "messages": cached_msgs,
            "max_tokens": max_tokens,
            "temperature": temperature,
        });

        if let Some(ref tool_defs) = cached_tools {
            if !tool_defs.is_empty() {
                body["tools"] = serde_json::Value::Array(tool_defs.clone());
                body["tool_choice"] = serde_json::json!("auto");
            }
        } else if let Some(tool_defs) = tools {
            if !tool_defs.is_empty() {
                body["tools"] = serde_json::Value::Array(tool_defs.to_vec());
                body["tool_choice"] = serde_json::json!("auto");
            }
        }

        let response = match self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => {
                warn!("HTTP request to LLM failed (base={}): {}", self.api_base, e);
                return Ok(LLMResponse {
                    content: Some(format!("Error calling LLM: {}", e)),
                    tool_calls: Vec::new(),
                    finish_reason: "error".to_string(),
                    usage: HashMap::new(),
                });
            }
        };

        let status = response.status();
        let response_text = match response.text().await {
            Ok(t) => t,
            Err(e) => {
                return Ok(LLMResponse {
                    content: Some(format!("Error reading LLM response: {}", e)),
                    tool_calls: Vec::new(),
                    finish_reason: "error".to_string(),
                    usage: HashMap::new(),
                });
            }
        };

        if !status.is_success() {
            warn!(
                "LLM API returned status {} (base={}): {}",
                status, self.api_base, response_text
            );
            return Ok(LLMResponse {
                content: Some(format!(
                    "Error calling LLM (HTTP {}): {}",
                    status, response_text
                )),
                tool_calls: Vec::new(),
                finish_reason: "error".to_string(),
                usage: HashMap::new(),
            });
        }

        let data: serde_json::Value = match serde_json::from_str(&response_text) {
            Ok(v) => v,
            Err(e) => {
                return Ok(LLMResponse {
                    content: Some(format!("Error parsing LLM response JSON: {}", e)),
                    tool_calls: Vec::new(),
                    finish_reason: "error".to_string(),
                    usage: HashMap::new(),
                });
            }
        };

        parse_response(&data)
    }

    async fn chat_stream(
        &self,
        messages: &[serde_json::Value],
        tools: Option<&[serde_json::Value]>,
        model: Option<&str>,
        max_tokens: u32,
        temperature: f64,
        _thinking_budget: Option<u32>,
    ) -> Result<StreamHandle> {
        let normalized = model.map(|m| normalize_model_name(m));
        let raw_model = normalized.as_deref().unwrap_or(&self.default_model);
        let model = if !self.api_base.contains("openrouter") {
            raw_model.split('/').last().unwrap_or(raw_model)
        } else {
            raw_model
        };
        let url = format!("{}/chat/completions", self.api_base);

        // Inject cache_control breakpoints for Anthropic prompt caching.
        let (cached_msgs, cached_tools) = if self.supports_cache_control(model) {
            inject_cache_control(messages, tools)
        } else {
            (messages.to_vec(), tools.map(|t| t.to_vec()))
        };

        let mut body = serde_json::json!({
            "model": model,
            "messages": cached_msgs,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": true,
        });

        if let Some(ref tool_defs) = cached_tools {
            if !tool_defs.is_empty() {
                body["tools"] = serde_json::Value::Array(tool_defs.clone());
                body["tool_choice"] = serde_json::json!("auto");
            }
        } else if let Some(tool_defs) = tools {
            if !tool_defs.is_empty() {
                body["tools"] = serde_json::Value::Array(tool_defs.to_vec());
                body["tool_choice"] = serde_json::json!("auto");
            }
        }

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            warn!("LLM streaming API returned status {} (base={}): {}", status, self.api_base, error_text);
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
            let _ = tx.send(StreamChunk::Done(LLMResponse {
                content: Some(format!("Error calling LLM (HTTP {}): {}", status, error_text)),
                tool_calls: Vec::new(),
                finish_reason: "error".to_string(),
                usage: HashMap::new(),
            }));
            return Ok(StreamHandle { rx });
        }

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

        // Spawn a task to parse the SSE stream
        let byte_stream = response.bytes_stream();
        tokio::spawn(async move {
            parse_sse_stream(byte_stream, tx).await;
        });

        Ok(StreamHandle { rx })
    }

    fn get_default_model(&self) -> &str {
        &self.default_model
    }

    fn get_api_base(&self) -> Option<&str> {
        Some(&self.api_base)
    }
}

/// Inject `cache_control` breakpoints into messages and tool definitions
/// for Anthropic prompt caching.
///
/// Transforms the system message content from a plain string to a content
/// array with `cache_control: {"type": "ephemeral"}`. This tells Anthropic
/// (and OpenRouter) to cache the system prompt prefix across turns, reducing
/// input token cost by ~90% for the cached portion.
///
/// Also marks the last tool definition with a cache breakpoint so tool schemas
/// are cached too.
fn inject_cache_control(
    messages: &[serde_json::Value],
    tools: Option<&[serde_json::Value]>,
) -> (Vec<serde_json::Value>, Option<Vec<serde_json::Value>>) {
    let mut msgs = messages.to_vec();

    // Transform system message (typically index 0) to use content array.
    if let Some(msg) = msgs.first_mut() {
        if msg.get("role").and_then(|r| r.as_str()) == Some("system") {
            if let Some(content) = msg.get("content").and_then(|c| c.as_str()) {
                msg["content"] = serde_json::json!([
                    {
                        "type": "text",
                        "text": content,
                        "cache_control": {"type": "ephemeral"}
                    }
                ]);
            }
        }
    }

    // Mark the last tool definition with cache_control so tool schemas are cached.
    let cached_tools = tools.map(|defs| {
        let mut tool_defs = defs.to_vec();
        if let Some(last) = tool_defs.last_mut() {
            last["cache_control"] = serde_json::json!({"type": "ephemeral"});
        }
        tool_defs
    });

    (msgs, cached_tools)
}

/// Parse the OpenAI-compatible JSON response into an `LLMResponse`.
fn parse_response(data: &serde_json::Value) -> Result<LLMResponse> {
    let choices = data
        .get("choices")
        .and_then(|c| c.as_array())
        .cloned()
        .unwrap_or_default();

    if choices.is_empty() {
        return Ok(LLMResponse {
            content: Some("Error: No choices in LLM response".to_string()),
            tool_calls: Vec::new(),
            finish_reason: "error".to_string(),
            usage: HashMap::new(),
        });
    }

    let choice = &choices[0];
    let message = choice.get("message").cloned().unwrap_or_default();
    let finish_reason = choice
        .get("finish_reason")
        .and_then(|v| v.as_str())
        .unwrap_or("stop")
        .to_string();

    // Extract content. reasoning_content (GLM-4.7, DeepSeek-R1 style) is logged
    // but NOT merged into the user-facing content — it's internal chain-of-thought
    // that pollutes display, TTS, and conversation history.
    if let Some(reasoning) = message
        .get("reasoning_content")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
    {
        debug!("Model returned reasoning_content ({} chars), discarding from output", reasoning.len());
    }
    let content = message
        .get("content")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string());

    // Extract tool calls.
    let mut tool_calls = Vec::new();
    if let Some(tc_array) = message.get("tool_calls").and_then(|v| v.as_array()) {
        for tc in tc_array {
            let id = tc
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            let function = tc.get("function").cloned().unwrap_or_default();
            let name = function
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            // Arguments come as a JSON string that we need to parse.
            let arguments_raw = function
                .get("arguments")
                .cloned()
                .unwrap_or(serde_json::Value::String("{}".to_string()));

            let arguments: HashMap<String, serde_json::Value> =
                if let Some(s) = arguments_raw.as_str() {
                    match serde_json::from_str(s) {
                        Ok(map) => map,
                        Err(_) => {
                            let mut m = HashMap::new();
                            m.insert("raw".to_string(), serde_json::Value::String(s.to_string()));
                            m
                        }
                    }
                } else if let Some(obj) = arguments_raw.as_object() {
                    obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
                } else {
                    HashMap::new()
                };

            tool_calls.push(ToolCallRequest {
                id,
                name,
                arguments,
            });
        }
    }

    // Extract usage.
    let mut usage = HashMap::new();
    if let Some(usage_obj) = data.get("usage").and_then(|v| v.as_object()) {
        for (key, value) in usage_obj {
            if let Some(n) = value.as_i64() {
                usage.insert(key.clone(), n);
            }
        }
    }

    Ok(LLMResponse {
        content,
        tool_calls,
        finish_reason,
        usage,
    })
}

/// Parse an SSE byte stream from an OpenAI-compatible streaming response.
///
/// Emits `TextDelta` for each content delta and `Done` at the end with the
/// fully assembled response. Tool call argument deltas are accumulated
/// internally and only emitted in the final `Done`.
async fn parse_sse_stream(
    byte_stream: impl futures_util::Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Unpin,
    tx: tokio::sync::mpsc::UnboundedSender<StreamChunk>,
) {
    let mut line_buffer = String::new();
    let mut full_content = String::new();
    let mut full_reasoning = String::new();
    let mut finish_reason = String::from("stop");
    let mut usage: HashMap<String, i64> = HashMap::new();

    // Tool call accumulation: index → (id, name, arguments_json_str)
    let mut tool_calls_acc: HashMap<u64, (String, String, String)> = HashMap::new();

    let mut stream = Box::pin(byte_stream);

    while let Some(result) = stream.next().await {
        let bytes = match result {
            Ok(b) => b,
            Err(e) => {
                warn!("SSE stream error: {}", e);
                break;
            }
        };

        let text = String::from_utf8_lossy(&bytes);
        line_buffer.push_str(&text);

        // Process complete lines
        while let Some(newline_pos) = line_buffer.find('\n') {
            let line = line_buffer[..newline_pos].trim_end_matches('\r').to_string();
            line_buffer = line_buffer[newline_pos + 1..].to_string();

            if line.is_empty() {
                continue;
            }

            if !line.starts_with("data: ") {
                continue;
            }

            let data = &line[6..];

            if data == "[DONE]" {
                // Stream is complete — discard reasoning_content (internal chain-of-thought)
                if !full_reasoning.is_empty() {
                    debug!("Streaming: discarding reasoning_content ({} chars)", full_reasoning.len());
                }
                let content = if !full_content.is_empty() {
                    Some(full_content.clone())
                } else {
                    None
                };

                let mut tool_calls = Vec::new();
                let mut indices: Vec<u64> = tool_calls_acc.keys().copied().collect();
                indices.sort();
                for idx in indices {
                    let (id, name, args_str) = tool_calls_acc.remove(&idx).unwrap();
                    let arguments: HashMap<String, serde_json::Value> =
                        match serde_json::from_str(&args_str) {
                            Ok(map) => map,
                            Err(_) => {
                                let mut m = HashMap::new();
                                m.insert(
                                    "raw".to_string(),
                                    serde_json::Value::String(args_str),
                                );
                                m
                            }
                        };
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

            // Parse JSON chunk
            let chunk: serde_json::Value = match serde_json::from_str(data) {
                Ok(v) => v,
                Err(e) => {
                    debug!("SSE parse error (skipping chunk): {}", e);
                    continue;
                }
            };

            // Extract from choices[0].delta
            if let Some(choices) = chunk.get("choices").and_then(|c| c.as_array()) {
                if let Some(choice) = choices.first() {
                    // Update finish_reason if present
                    if let Some(fr) = choice.get("finish_reason").and_then(|v| v.as_str()) {
                        finish_reason = fr.to_string();
                    }

                    if let Some(delta) = choice.get("delta") {
                        // Reasoning content delta (GLM-4.7, DeepSeek-R1 style)
                        if let Some(reasoning) = delta.get("reasoning_content").and_then(|v| v.as_str()) {
                            if !reasoning.is_empty() {
                                full_reasoning.push_str(reasoning);
                                let _ = tx.send(StreamChunk::ThinkingDelta(reasoning.to_string()));
                            }
                        }

                        // Text content delta
                        if let Some(content) = delta.get("content").and_then(|v| v.as_str()) {
                            if !content.is_empty() {
                                full_content.push_str(content);
                                let _ = tx.send(StreamChunk::TextDelta(content.to_string()));
                            }
                        }

                        // Tool call deltas
                        if let Some(tc_array) = delta.get("tool_calls").and_then(|v| v.as_array())
                        {
                            for tc in tc_array {
                                let index =
                                    tc.get("index").and_then(|v| v.as_u64()).unwrap_or(0);
                                let entry = tool_calls_acc
                                    .entry(index)
                                    .or_insert_with(|| (String::new(), String::new(), String::new()));

                                if let Some(id) = tc.get("id").and_then(|v| v.as_str()) {
                                    entry.0 = id.to_string();
                                }
                                if let Some(function) = tc.get("function") {
                                    if let Some(name) =
                                        function.get("name").and_then(|v| v.as_str())
                                    {
                                        entry.1 = name.to_string();
                                    }
                                    if let Some(args) =
                                        function.get("arguments").and_then(|v| v.as_str())
                                    {
                                        entry.2.push_str(args);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Extract usage if present (some providers include it in the last chunk)
            if let Some(usage_obj) = chunk.get("usage").and_then(|v| v.as_object()) {
                for (key, value) in usage_obj {
                    if let Some(n) = value.as_i64() {
                        usage.insert(key.clone(), n);
                    }
                }
            }
        }
    }

    // Stream ended without [DONE] — emit whatever we have, discarding reasoning_content
    if !full_reasoning.is_empty() {
        debug!("Streaming (no DONE): discarding reasoning_content ({} chars)", full_reasoning.len());
    }
    let content = if !full_content.is_empty() {
        Some(full_content)
    } else {
        None
    };

    let mut tool_calls = Vec::new();
    let mut indices: Vec<u64> = tool_calls_acc.keys().copied().collect();
    indices.sort();
    for idx in indices {
        let (id, name, args_str) = tool_calls_acc.remove(&idx).unwrap();
        let arguments: HashMap<String, serde_json::Value> = match serde_json::from_str(&args_str) {
            Ok(map) => map,
            Err(_) => {
                let mut m = HashMap::new();
                m.insert("raw".to_string(), serde_json::Value::String(args_str));
                m
            }
        };
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

#[cfg(test)]
mod tests {
    use super::super::base::LLMProvider;
    use super::*;

    // ── parse_response tests ──────────────────────────────────────

    #[test]
    fn test_parse_response_with_content_and_tool_calls() {
        let data = serde_json::json!({
            "choices": [{
                "message": {
                    "content": "Sure, let me look that up.",
                    "tool_calls": [{
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"location\": \"London\", \"units\": \"celsius\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 30,
                "total_tokens": 80
            }
        });

        let resp = parse_response(&data).expect("parse should succeed");
        assert_eq!(resp.content.as_deref(), Some("Sure, let me look that up."));
        assert_eq!(resp.finish_reason, "tool_calls");
        assert_eq!(resp.tool_calls.len(), 1);

        let tc = &resp.tool_calls[0];
        assert_eq!(tc.id, "call_abc123");
        assert_eq!(tc.name, "get_weather");
        assert_eq!(
            tc.arguments.get("location").and_then(|v| v.as_str()),
            Some("London")
        );
        assert_eq!(
            tc.arguments.get("units").and_then(|v| v.as_str()),
            Some("celsius")
        );

        // Verify usage was extracted.
        assert_eq!(resp.usage.get("prompt_tokens"), Some(&50));
        assert_eq!(resp.usage.get("completion_tokens"), Some(&30));
        assert_eq!(resp.usage.get("total_tokens"), Some(&80));
    }

    #[test]
    fn test_parse_response_content_only_no_tool_calls() {
        let data = serde_json::json!({
            "choices": [{
                "message": {
                    "content": "Hello! How can I help you today?"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 8,
                "total_tokens": 18
            }
        });

        let resp = parse_response(&data).expect("parse should succeed");
        assert_eq!(
            resp.content.as_deref(),
            Some("Hello! How can I help you today?")
        );
        assert_eq!(resp.finish_reason, "stop");
        assert!(resp.tool_calls.is_empty());
        assert_eq!(resp.usage.get("total_tokens"), Some(&18));
    }

    #[test]
    fn test_parse_response_tool_calls_without_content() {
        let data = serde_json::json!({
            "choices": [{
                "message": {
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "search",
                                "arguments": "{\"query\": \"rust async\"}"
                            }
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": "{\"path\": \"/tmp/test.txt\"}"
                            }
                        }
                    ]
                },
                "finish_reason": "tool_calls"
            }]
        });

        let resp = parse_response(&data).expect("parse should succeed");
        // Content should be None when the message has no "content" field.
        assert!(resp.content.is_none());
        assert_eq!(resp.tool_calls.len(), 2);
        assert_eq!(resp.tool_calls[0].name, "search");
        assert_eq!(resp.tool_calls[1].name, "read_file");
        assert_eq!(resp.tool_calls[1].id, "call_2");
        assert_eq!(resp.finish_reason, "tool_calls");
        // No usage block -> empty map.
        assert!(resp.usage.is_empty());
    }

    #[test]
    fn test_parse_response_empty_choices() {
        let data = serde_json::json!({
            "choices": []
        });

        let resp = parse_response(&data).expect("parse should succeed");
        assert_eq!(
            resp.content.as_deref(),
            Some("Error: No choices in LLM response")
        );
        assert_eq!(resp.finish_reason, "error");
        assert!(resp.tool_calls.is_empty());
    }

    #[test]
    fn test_parse_response_missing_choices_key() {
        // Completely missing "choices" key (e.g. malformed JSON from the API).
        let data = serde_json::json!({
            "error": "something went wrong"
        });

        let resp = parse_response(&data).expect("parse should succeed");
        assert_eq!(
            resp.content.as_deref(),
            Some("Error: No choices in LLM response")
        );
        assert_eq!(resp.finish_reason, "error");
    }

    #[test]
    fn test_parse_response_tool_call_with_unparseable_arguments() {
        // Arguments that are a string but not valid JSON should be stored under "raw".
        let data = serde_json::json!({
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call_bad",
                        "type": "function",
                        "function": {
                            "name": "broken_tool",
                            "arguments": "this is not json"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        });

        let resp = parse_response(&data).expect("parse should succeed");
        assert_eq!(resp.tool_calls.len(), 1);
        let tc = &resp.tool_calls[0];
        assert_eq!(tc.name, "broken_tool");
        // Unparseable JSON string should be stored under the "raw" key.
        assert_eq!(
            tc.arguments.get("raw").and_then(|v| v.as_str()),
            Some("this is not json")
        );
    }

    #[test]
    fn test_parse_response_reasoning_content_discarded() {
        // reasoning_content from reasoning models (GLM-4.7, DeepSeek-R1) should
        // be discarded, NOT merged into the user-facing content.
        let data = serde_json::json!({
            "choices": [{
                "message": {
                    "reasoning_content": "Let me think about this step by step...",
                    "content": "The answer is 42."
                },
                "finish_reason": "stop"
            }]
        });

        let resp = parse_response(&data).expect("parse should succeed");
        // Only the main content should appear — no <thinking> tags
        assert_eq!(resp.content.as_deref(), Some("The answer is 42."));
        assert!(!resp.content.unwrap_or_default().contains("<thinking>"));
    }

    #[test]
    fn test_parse_response_reasoning_only_no_content() {
        // If model returns ONLY reasoning_content with no main content,
        // the response should be None (not a <thinking> block).
        let data = serde_json::json!({
            "choices": [{
                "message": {
                    "reasoning_content": "Hmm, let me think...",
                },
                "finish_reason": "stop"
            }]
        });

        let resp = parse_response(&data).expect("parse should succeed");
        assert!(resp.content.is_none(), "reasoning-only response should have no content");
    }

    // ── Provider creation / detection tests ───────────────────────

    #[test]
    fn test_new_openrouter_by_key_prefix() {
        let provider = OpenAICompatProvider::new("sk-or-my-key", None, None);
        assert_eq!(provider.api_base, "https://openrouter.ai/api/v1");
        assert_eq!(provider.default_model, "anthropic/claude-opus-4-5");
    }

    #[test]
    fn test_new_openrouter_by_api_base() {
        let provider = OpenAICompatProvider::new(
            "some-key",
            Some("https://openrouter.ai/api/v1"),
            Some("meta-llama/llama-3-70b"),
        );
        assert_eq!(provider.api_base, "https://openrouter.ai/api/v1");
        assert_eq!(provider.default_model, "meta-llama/llama-3-70b");
    }

    #[test]
    fn test_new_deepseek_detection() {
        let provider = OpenAICompatProvider::new("sk-something", None, Some("deepseek-chat"));
        assert_eq!(provider.api_base, "https://api.deepseek.com");
        assert_eq!(provider.default_model, "deepseek-chat");
    }

    #[test]
    fn test_new_groq_detection() {
        let provider = OpenAICompatProvider::new("gsk_something", None, Some("groq/llama3"));
        assert_eq!(provider.api_base, "https://api.groq.com/openai/v1");
        assert_eq!(provider.default_model, "groq/llama3");
    }

    #[test]
    fn test_new_explicit_api_base_takes_precedence() {
        let provider = OpenAICompatProvider::new(
            "sk-or-key",
            Some("http://localhost:8000/v1/"),
            Some("my-local-model"),
        );
        // Trailing slash should be trimmed.
        assert_eq!(provider.api_base, "http://localhost:8000/v1");
        assert_eq!(provider.default_model, "my-local-model");
    }

    #[test]
    fn test_new_default_fallback_is_openrouter() {
        // Unknown key prefix + no api_base + routed model name -> OpenRouter.
        let provider =
            OpenAICompatProvider::new("random-key", None, Some("anthropic/claude-opus-4-5"));
        assert_eq!(provider.api_base, "https://openrouter.ai/api/v1");
    }

    #[test]
    fn test_new_anthropic_key_detection() {
        let provider =
            OpenAICompatProvider::new("sk-ant-abc123", None, Some("claude-sonnet-4-5-20250929"));
        assert_eq!(provider.api_base, "https://api.anthropic.com/v1");
    }

    #[test]
    fn test_new_openai_key_with_bare_model() {
        // sk- prefix with a non-routed model -> OpenAI direct.
        let provider = OpenAICompatProvider::new("sk-abc123", None, Some("gpt-4o"));
        assert_eq!(provider.api_base, "https://api.openai.com/v1");
    }

    #[test]
    fn test_new_sk_key_with_routed_model_is_openrouter() {
        // sk- prefix but model has "/" -> OpenRouter.
        let provider =
            OpenAICompatProvider::new("sk-abc123", None, Some("anthropic/claude-opus-4-5"));
        assert_eq!(provider.api_base, "https://openrouter.ai/api/v1");
    }

    #[test]
    fn test_get_default_model() {
        let provider = OpenAICompatProvider::new("sk-key", None, Some("gpt-4o"));
        assert_eq!(provider.get_default_model(), "gpt-4o");
    }

    #[test]
    fn test_get_default_model_uses_fallback() {
        let provider = OpenAICompatProvider::new("sk-key", None, None);
        assert_eq!(provider.get_default_model(), "anthropic/claude-opus-4-5");
    }

    // ── Model name normalization tests ─────────────────────────────

    #[test]
    fn test_normalize_short_aliases() {
        assert_eq!(normalize_model_name("opus"), "claude-opus-4-6");
        assert_eq!(normalize_model_name("sonnet"), "claude-sonnet-4-5-20250929");
        assert_eq!(normalize_model_name("haiku"), "claude-haiku-4-5-20251001");
        // Case-insensitive.
        assert_eq!(normalize_model_name("Opus"), "claude-opus-4-6");
        assert_eq!(normalize_model_name("SONNET"), "claude-sonnet-4-5-20250929");
    }

    #[test]
    fn test_normalize_missing_claude_prefix() {
        assert_eq!(normalize_model_name("opus-4-6"), "claude-opus-4-6");
        assert_eq!(
            normalize_model_name("sonnet-4-5-20250929"),
            "claude-sonnet-4-5-20250929"
        );
        assert_eq!(
            normalize_model_name("haiku-4-5-20251001"),
            "claude-haiku-4-5-20251001"
        );
    }

    #[test]
    fn test_normalize_already_correct() {
        assert_eq!(
            normalize_model_name("claude-opus-4-6"),
            "claude-opus-4-6"
        );
        assert_eq!(
            normalize_model_name("anthropic/claude-opus-4-6"),
            "anthropic/claude-opus-4-6"
        );
    }

    #[test]
    fn test_normalize_non_claude_passthrough() {
        assert_eq!(normalize_model_name("gpt-4o"), "gpt-4o");
        assert_eq!(normalize_model_name("deepseek-chat"), "deepseek-chat");
        assert_eq!(normalize_model_name("local"), "local");
        assert_eq!(
            normalize_model_name("meta-llama/llama-3-70b"),
            "meta-llama/llama-3-70b"
        );
    }

    #[test]
    fn test_provider_normalizes_default_model() {
        // Config has "opus-4-6" → provider should normalize to "claude-opus-4-6".
        let provider = OpenAICompatProvider::new("sk-or-key", None, Some("opus-4-6"));
        assert_eq!(provider.default_model, "claude-opus-4-6");
    }

    // ── Cache control tests ──────────────────────────────────────

    #[test]
    fn test_supports_cache_control_anthropic_direct() {
        let provider = OpenAICompatProvider::new("sk-ant-abc", None, Some("claude-opus-4-6"));
        assert!(provider.supports_cache_control("claude-opus-4-6"));
    }

    #[test]
    fn test_supports_cache_control_openrouter_with_claude() {
        let provider = OpenAICompatProvider::new("sk-or-abc", None, Some("anthropic/claude-opus-4-6"));
        assert!(provider.supports_cache_control("anthropic/claude-opus-4-6"));
    }

    #[test]
    fn test_no_cache_control_openrouter_non_claude() {
        let provider = OpenAICompatProvider::new("sk-or-abc", None, Some("meta-llama/llama-3-70b"));
        assert!(!provider.supports_cache_control("meta-llama/llama-3-70b"));
    }

    #[test]
    fn test_no_cache_control_local() {
        let provider = OpenAICompatProvider::new("none", Some("http://localhost:8080/v1"), Some("local"));
        assert!(!provider.supports_cache_control("local"));
    }

    #[test]
    fn test_inject_cache_control_system_message() {
        let messages = vec![
            serde_json::json!({"role": "system", "content": "You are helpful."}),
            serde_json::json!({"role": "user", "content": "Hello"}),
        ];
        let (cached, _) = inject_cache_control(&messages, None);

        // System message should now have content as array with cache_control.
        let sys_content = &cached[0]["content"];
        assert!(sys_content.is_array(), "system content should be array");
        let block = &sys_content[0];
        assert_eq!(block["type"], "text");
        assert_eq!(block["text"], "You are helpful.");
        assert_eq!(block["cache_control"]["type"], "ephemeral");

        // User message should be unchanged.
        assert_eq!(cached[1]["content"], "Hello");
    }

    #[test]
    fn test_inject_cache_control_tools() {
        let messages = vec![
            serde_json::json!({"role": "system", "content": "test"}),
        ];
        let tools = vec![
            serde_json::json!({"type": "function", "function": {"name": "tool_a"}}),
            serde_json::json!({"type": "function", "function": {"name": "tool_b"}}),
        ];
        let (_, cached_tools) = inject_cache_control(&messages, Some(&tools));

        let tools = cached_tools.unwrap();
        // First tool: no cache_control.
        assert!(tools[0].get("cache_control").is_none());
        // Last tool: has cache_control.
        assert_eq!(tools[1]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn test_inject_cache_control_no_system_message() {
        // Edge case: no system message — should not panic.
        let messages = vec![
            serde_json::json!({"role": "user", "content": "Hello"}),
        ];
        let (cached, _) = inject_cache_control(&messages, None);
        // User message should be unchanged (not treated as system).
        assert_eq!(cached[0]["content"], "Hello");
    }
}
