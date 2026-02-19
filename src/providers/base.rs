//! Base LLM provider interface.

use std::collections::HashMap;

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// A tool call request from the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallRequest {
    pub id: String,
    pub name: String,
    pub arguments: HashMap<String, serde_json::Value>,
}

impl ToolCallRequest {
    /// Convert to OpenAI function-call JSON format.
    pub fn to_openai_json(&self) -> serde_json::Value {
        serde_json::json!({
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": serde_json::to_string(&self.arguments)
                    .unwrap_or_else(|_| "{}".to_string()),
            }
        })
    }
}

/// Response from an LLM provider.
#[derive(Debug, Clone)]
pub struct LLMResponse {
    pub content: Option<String>,
    pub tool_calls: Vec<ToolCallRequest>,
    pub finish_reason: String,
    pub usage: HashMap<String, i64>,
}

impl LLMResponse {
    /// Check if response contains tool calls.
    pub fn has_tool_calls(&self) -> bool {
        !self.tool_calls.is_empty()
    }
}

/// A chunk from an SSE streaming response.
#[derive(Debug, Clone)]
pub enum StreamChunk {
    /// Incremental text content from the LLM.
    TextDelta(String),
    /// Incremental thinking/reasoning content (extended thinking).
    ThinkingDelta(String),
    /// Stream complete — contains the fully assembled response.
    Done(LLMResponse),
}

/// Handle to a streaming LLM response.
pub struct StreamHandle {
    pub rx: tokio::sync::mpsc::UnboundedReceiver<StreamChunk>,
}

/// Abstract base trait for LLM providers.
///
/// Implementations should handle the specifics of each provider's API
/// while maintaining a consistent interface.
#[async_trait]
pub trait LLMProvider: Send + Sync {
    /// Send a chat completion request.
    ///
    /// # Arguments
    /// * `messages` - List of message objects with `role` and `content`.
    /// * `tools` - Optional list of tool definitions in OpenAI format.
    /// * `model` - Model identifier (provider-specific).
    /// * `max_tokens` - Maximum tokens in response.
    /// * `temperature` - Sampling temperature.
    /// * `thinking_budget` - If Some, enable extended thinking with this token budget.
    async fn chat(
        &self,
        messages: &[serde_json::Value],
        tools: Option<&[serde_json::Value]>,
        model: Option<&str>,
        max_tokens: u32,
        temperature: f64,
        thinking_budget: Option<u32>,
    ) -> Result<LLMResponse>;

    /// Send a streaming chat completion request.
    ///
    /// Default implementation falls back to buffered `chat()`.
    async fn chat_stream(
        &self,
        messages: &[serde_json::Value],
        tools: Option<&[serde_json::Value]>,
        model: Option<&str>,
        max_tokens: u32,
        temperature: f64,
        thinking_budget: Option<u32>,
    ) -> Result<StreamHandle> {
        let response = self
            .chat(
                messages,
                tools,
                model,
                max_tokens,
                temperature,
                thinking_budget,
            )
            .await?;
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        if let Some(ref content) = response.content {
            let _ = tx.send(StreamChunk::TextDelta(content.clone()));
        }
        let _ = tx.send(StreamChunk::Done(response));
        Ok(StreamHandle { rx })
    }

    /// Get the default model for this provider.
    fn get_default_model(&self) -> &str;

    /// Get the API base URL (for health checks). Returns None for cloud providers.
    fn get_api_base(&self) -> Option<&str> {
        None
    }
}
