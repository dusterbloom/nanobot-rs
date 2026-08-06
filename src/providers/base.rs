//! Base LLM provider interface.

use std::collections::HashMap;

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::errors::ProviderError;

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

/// Why an LLM call finished.
///
/// Replaces the magic `String` on [`LLMResponse`]; the provider-visible wire
/// string is recovered with [`FinishReason::wire_str`] so model-visible bytes
/// are unchanged (error-protocol doc §2.3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinishReason {
    /// Model produced a complete response (`"stop"` on the wire).
    Stop,
    /// Output hit the token limit (`"length"` on the wire).
    Length,
    /// Response is a tool-call request (`"tool_calls"` on the wire).
    ToolCalls,
    /// Stream died before producing content/tool-calls
    /// (was the literal `"error"` string, `openai_compat.rs:1842`).
    ProviderFailure,
    /// Turn was aborted/cancelled — was magic strings matched at
    /// `agent_loop/response.rs:1024`.
    Aborted,
    Cancelled,
    /// Unknown provider value — kept so foreign backends don't break parsing.
    Other(String),
}

impl FinishReason {
    /// Parse a provider-reported `finish_reason` string at the wire boundary.
    /// Unknown values round-trip through [`FinishReason::Other`] so foreign
    /// backends keep working unchanged.
    #[must_use]
    pub fn parse_finish_reason(s: &str) -> Self {
        match s {
            "stop" => Self::Stop,
            "length" => Self::Length,
            "tool_calls" => Self::ToolCalls,
            "error" => Self::ProviderFailure,
            "aborted" => Self::Aborted,
            "cancelled" => Self::Cancelled,
            other => Self::Other(other.to_string()),
        }
    }

    /// The byte-identical wire string for this reason.
    #[must_use]
    pub fn wire_str(&self) -> &str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
            Self::ToolCalls => "tool_calls",
            Self::ProviderFailure => "error",
            Self::Aborted => "aborted",
            Self::Cancelled => "cancelled",
            Self::Other(s) => s,
        }
    }
}

impl std::fmt::Display for FinishReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.wire_str())
    }
}

/// Response from an LLM provider.
#[derive(Debug, Clone)]
pub struct LLMResponse {
    pub content: Option<String>,
    pub tool_calls: Vec<ToolCallRequest>,
    pub finish_reason: FinishReason,
    pub usage: HashMap<String, i64>,
}

impl LLMResponse {
    /// Check if response contains tool calls.
    pub fn has_tool_calls(&self) -> bool {
        !self.tool_calls.is_empty()
    }

    /// `Result` view: `Ok(())` for stop/length/tool_calls,
    /// `Err(ProviderError::EmptyStream)` for the dead-stream case
    /// (error-protocol doc §2.3).
    ///
    /// # Errors
    ///
    /// Returns `Err(ProviderError::EmptyStream(..))` when the stream died
    /// before producing content or tool-calls.
    pub fn outcome(&self) -> Result<(), ProviderError> {
        match self.finish_reason {
            FinishReason::ProviderFailure => Err(ProviderError::EmptyStream(
                self.content
                    .clone()
                    .unwrap_or_else(|| "Unknown LLM error".to_string()),
            )),
            _ => Ok(()),
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


#[cfg(test)]
mod tests {
    use super::FinishReason;

    /// Wire-stability guard (error-protocol doc §2.3 / §4): every wire string
    /// the streaming layer can produce must parse and re-serialize to the
    /// byte-identical value, including unknown provider values.
    #[test]
    fn finish_reason_wire_round_trip_is_byte_identical() {
        let known = [
            ("stop", FinishReason::Stop),
            ("length", FinishReason::Length),
            ("tool_calls", FinishReason::ToolCalls),
            ("error", FinishReason::ProviderFailure),
            ("aborted", FinishReason::Aborted),
            ("cancelled", FinishReason::Cancelled),
        ];
        for (wire, reason) in known {
            assert_eq!(FinishReason::parse_finish_reason(wire), reason, "parse {wire}");
            assert_eq!(reason.wire_str(), wire, "wire_str for {wire}");
            assert_eq!(reason.to_string(), wire, "Display for {wire}");
        }
        // Unknown provider values round-trip via Other(String).
        for wire in ["function_call", "content_filter", "eos_token", ""] {
            assert_eq!(
                FinishReason::parse_finish_reason(wire).wire_str(),
                wire,
                "unknown wire value {wire:?} must round-trip"
            );
        }
    }
}
