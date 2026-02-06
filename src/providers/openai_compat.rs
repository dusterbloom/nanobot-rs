//! OpenAI-compatible API provider.
//!
//! Replaces LiteLLMProvider by calling OpenAI-compatible APIs directly via reqwest.
//! Supports OpenRouter, Anthropic (OpenAI-compat endpoint), OpenAI, DeepSeek,
//! Groq, vLLM, and any other provider that implements the OpenAI chat completions
//! API format.

use std::collections::HashMap;

use anyhow::Result;
use async_trait::async_trait;
use reqwest::Client;
use tracing::warn;

use super::base::{LLMProvider, LLMResponse, ToolCallRequest};

/// An LLM provider that talks to any OpenAI-compatible chat completions endpoint.
pub struct OpenAICompatProvider {
    api_key: String,
    api_base: String,
    default_model: String,
    client: Client,
}

impl OpenAICompatProvider {
    /// Create a new provider.
    ///
    /// Provider detection logic (porting from `LiteLLMProvider.__init__`):
    /// - OpenRouter: detected by `sk-or-` key prefix or `openrouter` in api_base
    /// - DeepSeek: detected by `deepseek` in the default model name
    /// - vLLM / custom: when an explicit `api_base` is provided that isn't OpenRouter
    /// - Default fallback: OpenRouter (`https://openrouter.ai/api/v1`)
    pub fn new(
        api_key: &str,
        api_base: Option<&str>,
        default_model: Option<&str>,
    ) -> Self {
        let default_model = default_model
            .unwrap_or("anthropic/claude-opus-4-5")
            .to_string();

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
    ) -> Result<LLMResponse> {
        let raw_model = model.unwrap_or(&self.default_model);
        // Strip "provider/" prefix for non-OpenRouter APIs (e.g. "anthropic/claude-opus-4-5"
        // becomes "claude-opus-4-5" when hitting api.anthropic.com directly).
        let model = if !self.api_base.contains("openrouter") {
            raw_model.split('/').last().unwrap_or(raw_model)
        } else {
            raw_model
        };
        let url = format!("{}/chat/completions", self.api_base);

        // Build request body.
        let mut body = serde_json::json!({
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        });

        if let Some(tool_defs) = tools {
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
                warn!("HTTP request to LLM failed: {}", e);
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
            warn!("LLM API returned status {}: {}", status, response_text);
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

    fn get_default_model(&self) -> &str {
        &self.default_model
    }
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

    // Extract content.
    let content = message
        .get("content")
        .and_then(|v| v.as_str())
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

            let arguments: HashMap<String, serde_json::Value> = if let Some(s) =
                arguments_raw.as_str()
            {
                match serde_json::from_str(s) {
                    Ok(map) => map,
                    Err(_) => {
                        let mut m = HashMap::new();
                        m.insert("raw".to_string(), serde_json::Value::String(s.to_string()));
                        m
                    }
                }
            } else if let Some(obj) = arguments_raw.as_object() {
                obj.iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect()
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

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::base::LLMProvider;

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
        let provider =
            OpenAICompatProvider::new("sk-something", None, Some("deepseek-chat"));
        assert_eq!(provider.api_base, "https://api.deepseek.com");
        assert_eq!(provider.default_model, "deepseek-chat");
    }

    #[test]
    fn test_new_groq_detection() {
        let provider =
            OpenAICompatProvider::new("gsk_something", None, Some("groq/llama3"));
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
        let provider =
            OpenAICompatProvider::new("sk-abc123", None, Some("gpt-4o"));
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
}
