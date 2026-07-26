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

    /// Returns `true` if this response represents an LLM provider error.
    ///
    /// `finish_reason = "error"` is synthesized by the streaming layer when a
    /// stream ends without producing any content or tool-call payload (see
    /// `parse_sse_stream`); hard transport/provider failures are still returned
    /// as `Err(ProviderError)` from `chat()`.
    pub fn is_error(&self) -> bool {
        self.finish_reason == "error"
    }

    /// Returns the error detail if this response is an error, else `None`.
    ///
    /// See `is_error()` note.
    pub fn error_detail(&self) -> Option<&str> {
        if self.is_error() {
            Some(self.content.as_deref().unwrap_or("Unknown LLM error"))
        } else {
            None
        }
    }
}

/// A chunk from an SSE streaming response.
#[derive(Debug, Clone)]
pub enum StreamChunk {
    /// Transport-level liveness without semantic model progress, such as an
    /// SSE comment heartbeat sent while a local backend request is queued.
    TransportProgress,
    /// Incremental text content from the LLM.
    TextDelta(String),
    /// Incremental thinking/reasoning content (extended thinking).
    ThinkingDelta(String),
    /// A streamed tool-call fragment arrived (function name or arguments).
    /// Emitted so consumers can mark end-of-prefill and reset idle watchdogs:
    /// pure tool-call responses may produce no text/thinking deltas while still
    /// generating a large JSON argument payload. Full calls are delivered in
    /// `Done`.
    ToolCallDelta,
    /// Server-reported prefill progress (`prompt_progress` chunks from
    /// llama.cpp/higgs when the request set `return_progress`). `processed`
    /// counts cached + prefilled prompt tokens out of `total`.
    PrefillProgress { processed: u64, total: u64 },
    /// Stream complete — contains the fully assembled response.
    Done(LLMResponse),
}

/// Handle to a streaming LLM response.
pub struct StreamHandle {
    pub rx: tokio::sync::mpsc::UnboundedReceiver<StreamChunk>,
    /// Provider-owned parser/read task. Dropping the handle aborts it so local
    /// streams that stop making progress release their HTTP request and JIT
    /// permit immediately instead of lingering until the backend timeout.
    pub abort_on_drop: Option<tokio::task::JoinHandle<()>>,
}

impl Drop for StreamHandle {
    fn drop(&mut self) {
        if let Some(handle) = self.abort_on_drop.take() {
            handle.abort();
        }
    }
}

/// How the model should choose tools for a request. Maps to OpenAI `tool_choice`.
///
/// `Required` asks the server to force exactly one tool call; for a local Higgs
/// backend this triggers grammar-constrained decoding so the call is always
/// well-formed. The default `chat()` path and cloud providers use `Auto`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolChoice {
    /// Model decides whether to call a tool (OpenAI `"auto"`).
    Auto,
    /// Force exactly one tool call (OpenAI `"required"`).
    Required,
    /// Forbid tool calls (OpenAI `"none"`).
    None,
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
    /// * `top_p` - If Some, nucleus sampling probability mass.
    async fn chat(
        &self,
        messages: &[serde_json::Value],
        tools: Option<&[serde_json::Value]>,
        model: Option<&str>,
        max_tokens: u32,
        temperature: f64,
        thinking_budget: Option<u32>,
        top_p: Option<f64>,
    ) -> Result<LLMResponse>;

    /// Like [`chat`](Self::chat) but with an explicit [`ToolChoice`].
    ///
    /// The default implementation ignores `tool_choice` and delegates to
    /// `chat`, so providers that don't support forcing keep their existing
    /// behavior. `OpenAICompatProvider` overrides this to emit `tool_choice`
    /// for local backends.
    #[allow(clippy::too_many_arguments)]
    async fn chat_with_tool_choice(
        &self,
        messages: &[serde_json::Value],
        tools: Option<&[serde_json::Value]>,
        model: Option<&str>,
        max_tokens: u32,
        temperature: f64,
        thinking_budget: Option<u32>,
        top_p: Option<f64>,
        _tool_choice: ToolChoice,
    ) -> Result<LLMResponse> {
        self.chat(
            messages,
            tools,
            model,
            max_tokens,
            temperature,
            thinking_budget,
            top_p,
        )
        .await
    }

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
        top_p: Option<f64>,
    ) -> Result<StreamHandle> {
        let response = self
            .chat(
                messages,
                tools,
                model,
                max_tokens,
                temperature,
                thinking_budget,
                top_p,
            )
            .await?;
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        if let Some(ref content) = response.content {
            let _ = tx.send(StreamChunk::TextDelta(content.clone()));
        }
        let _ = tx.send(StreamChunk::Done(response));
        Ok(StreamHandle {
            rx,
            abort_on_drop: None,
        })
    }

    /// Get the default model for this provider.
    fn get_default_model(&self) -> &str;

    /// Get the API base URL (for health checks). Returns None for cloud providers.
    fn get_api_base(&self) -> Option<&str> {
        None
    }

    /// Whether this provider speaks the higgs retained-session protocol
    /// (`session_id` / `drop_session_id(s)` request fields). Drives KV-cache
    /// session-id rotation on the agent side. Defaults to `false`; the
    /// OpenAI-compatible provider overrides it to return its `higgs_session_cache`
    /// flag (set when `localBackend=higgs`), so the capability — not the port —
    /// decides whether the agent rotates the session id on compaction/trim.
    fn supports_higgs_session_cache(&self) -> bool {
        false
    }
}
