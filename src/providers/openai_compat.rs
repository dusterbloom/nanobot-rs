// Error-protocol layer-3 backlog (docs/research/2026-08-06-error-conventions-and-host-bridge.md §3.6):
// the deny regime in Cargo.toml is live; this module still carries pre-existing
// violations of the lints below. Remove this allow as the module migrates onto
// the regime.
// Tracking: docs/error-protocol-backlog.md
#![allow(
    clippy::as_conversions,
    clippy::indexing_slicing,
    clippy::shadow_reuse,
    clippy::shadow_unrelated
)]
//! OpenAI-compatible API provider.
//!
//! Replaces LiteLLMProvider by calling OpenAI-compatible APIs directly via reqwest.
//! Supports OpenRouter, Anthropic (OpenAI-compat endpoint), OpenAI, DeepSeek,
//! Groq, vLLM, and any other provider that implements the OpenAI chat completions
//! API format.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use futures_util::StreamExt;
use reqwest::Client;
use tracing::{debug, info, instrument, warn};

use backon::Retryable;

use super::base::{
    FinishReason, LLMProvider, LLMResponse, StreamChunk, StreamHandle, ToolCallRequest, ToolChoice,
};
use super::constants::{
    ANTHROPIC_API_BASE, DEEPSEEK_API_BASE, GROQ_API_BASE, OPENAI_API_BASE, OPENROUTER_API_BASE,
};
use super::jit_gate::JitGate;
use super::retry;

/// An LLM provider that talks to any OpenAI-compatible chat completions endpoint.
pub struct OpenAICompatProvider {
    api_key: String,
    api_base: String,
    default_model: String,
    client: Client,
    /// Optional JIT gate for serialising requests to JIT-loading servers (e.g. LM Studio).
    jit_gate: Option<Arc<JitGate>>,
    /// Minimum backoff delay for cloud provider retries (default: 1s).
    retry_provider_min_secs: u64,
    /// Maximum backoff delay for cloud provider retries (default: 30s).
    retry_provider_max_secs: u64,
    /// Minimum backoff delay for JIT model loading retries (default: 2s).
    retry_jit_min_secs: u64,
    /// Maximum backoff delay for JIT model loading retries (default: 8s).
    retry_jit_max_secs: u64,
    /// Timeout for native LMS probe requests in seconds (default: 2).
    lms_native_probe_secs: u64,
    /// Whether to emit `tool_choice: "required"` to local backends when a caller
    /// requests [`ToolChoice::Required`]. Default `true`; the config escape hatch
    /// (`agents.defaults.constrained_tool_calls = false`) sets it `false`.
    constrained_tool_calls: bool,
    /// Whether to forward nanobot's internal per-conversation marker as Higgs'
    /// top-level `session_id` extension. Disabled for every non-Higgs provider.
    higgs_session_cache: bool,
    /// Optional OpenAI-compatible repetition penalty override.
    repetition_penalty: Option<f64>,
    /// Optional OpenAI-compatible frequency penalty override.
    frequency_penalty: Option<f64>,
    /// Optional OpenAI-compatible presence penalty override.
    presence_penalty: Option<f64>,
}

pub(crate) const NANOBOT_HIGGS_SESSION_ID_FIELD: &str = "_nanobot_higgs_session_id";
pub(crate) const NANOBOT_HIGGS_DROP_SESSION_ID_FIELD: &str = "_nanobot_higgs_drop_session_id";
pub(crate) const NANOBOT_HIGGS_DROP_SESSION_IDS_FIELD: &str = "_nanobot_higgs_drop_session_ids";
pub(crate) const NANOBOT_HIGGS_SESSION_LEASE_FIELD: &str = "_nanobot_higgs_session_lease";
pub(crate) const NANOBOT_HIGGS_SESSION_CACHE_POLICY_FIELD: &str =
    "_nanobot_higgs_session_cache_policy";
pub(crate) const NANOBOT_HIGGS_MAX_PROMPT_TOKENS_FIELD: &str = "_nanobot_higgs_max_prompt_tokens";

fn build_http_client(timeout_secs: u64) -> Client {
    let timeout = std::time::Duration::from_secs(timeout_secs);
    Client::builder()
        .connect_timeout(timeout)
        .read_timeout(timeout)
        .build()
        .unwrap_or_else(|_| Client::new())
}

fn is_valid_tool_call_name(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    (first.is_ascii_alphabetic() || first == '_')
        && chars.all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
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
    if lower.starts_with("opus") || lower.starts_with("sonnet") || lower.starts_with("haiku") {
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
        let default_model =
            normalize_model_name(default_model.unwrap_or("anthropic/claude-opus-4-5"));

        let resolved_base = if let Some(base) = api_base {
            // Use whatever was explicitly provided.
            base.trim_end_matches('/').to_string()
        } else if api_key.starts_with("sk-or-") {
            OPENROUTER_API_BASE.to_string()
        } else if api_key.starts_with("sk-ant-") {
            ANTHROPIC_API_BASE.to_string()
        } else if default_model.contains("deepseek") {
            DEEPSEEK_API_BASE.to_string()
        } else if api_key.starts_with("gsk_") || default_model.contains("groq") {
            GROQ_API_BASE.to_string()
        } else if api_key.starts_with("sk-") && !default_model.contains('/') {
            // Bare "sk-" prefix with a non-routed model name -> likely OpenAI direct.
            OPENAI_API_BASE.to_string()
        } else {
            // Fallback: OpenRouter (supports routed model names like "anthropic/claude-...").
            OPENROUTER_API_BASE.to_string()
        };

        // Bound connection establishment and inactivity between response reads.
        // The read timeout is sliding, so SSE heartbeat bytes keep a healthy
        // queued/local generation alive without imposing an absolute deadline.
        let client = build_http_client(120);

        Self {
            api_key: api_key.to_string(),
            api_base: resolved_base,
            default_model,
            client,
            jit_gate: None,
            retry_provider_min_secs: 1,
            retry_provider_max_secs: 30,
            retry_jit_min_secs: 2,
            retry_jit_max_secs: 8,
            lms_native_probe_secs: 2,
            constrained_tool_calls: true,
            higgs_session_cache: false,
            repetition_penalty: None,
            frequency_penalty: None,
            presence_penalty: None,
        }
    }

    /// Enable or disable grammar-constrained tool calls for local backends.
    ///
    /// When `false`, [`ToolChoice::Required`] degrades to `"auto"` (the config
    /// escape hatch). No effect on cloud providers.
    pub fn with_constrained_tool_calls(mut self, enabled: bool) -> Self {
        self.constrained_tool_calls = enabled;
        self
    }

    /// Enable Higgs' cache-resident continuation extension for this provider.
    pub fn with_higgs_session_cache(mut self, enabled: bool) -> Self {
        self.higgs_session_cache = enabled;
        self
    }

    /// Attach OpenAI-compatible sampling penalties.
    pub fn with_sampling_penalties(
        mut self,
        repetition_penalty: Option<f64>,
        frequency_penalty: Option<f64>,
        presence_penalty: Option<f64>,
    ) -> Self {
        self.repetition_penalty = repetition_penalty;
        self.frequency_penalty = frequency_penalty;
        self.presence_penalty = presence_penalty;
        self
    }

    /// Override the HTTP connect and read-inactivity timeouts.
    ///
    /// Replaces the default 120s sliding timeout with the given value.
    pub fn with_timeout(mut self, timeout_secs: u64) -> Self {
        self.client = build_http_client(timeout_secs);
        self
    }

    /// Override the native LMS probe timeout.
    pub fn with_lms_native_probe_secs(mut self, secs: u64) -> Self {
        self.lms_native_probe_secs = secs;
        self
    }

    /// Attach a JIT gate for serialised access to a JIT-loading server.
    ///
    /// When set, every `chat()` and `chat_stream()` call acquires the gate's
    /// single permit before sending the HTTP request. Streaming holds the
    /// permit for the entire stream duration to prevent model switches mid-stream.
    pub fn with_jit_gate(mut self, gate: Arc<JitGate>) -> Self {
        self.jit_gate = Some(gate);
        self
    }

    /// Override the retry backoff parameters.
    ///
    /// Values come from `config.retry`; defaults match the original hardcoded values.
    pub fn with_retry_config(
        mut self,
        provider_min_secs: u64,
        provider_max_secs: u64,
        jit_min_secs: u64,
        jit_max_secs: u64,
    ) -> Self {
        self.retry_provider_min_secs = provider_min_secs;
        self.retry_provider_max_secs = provider_max_secs;
        self.retry_jit_min_secs = jit_min_secs;
        self.retry_jit_max_secs = jit_max_secs;
        self
    }
}

const THINK_OPEN_TAGS: [&str; 2] = ["<thinking>", "<think>"];
const THINK_CLOSE_TAGS: [&str; 2] = ["</thinking>", "</think>"];

#[derive(Default)]
pub(crate) struct ThinkSplitState {
    pub(crate) in_think_block: bool,
    pub(crate) carry: String,
}

/// Try to extract a retry-after duration from an error response body.
fn parse_retry_after_ms(response_text: &str) -> u64 {
    // Try JSON: {"error": {"retry_after": 1.5}} or {"retry_after": 2}
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(response_text) {
        if let Some(secs) = v
            .pointer("/error/retry_after")
            .or_else(|| v.get("retry_after"))
            .and_then(|v| v.as_f64())
        {
            return (secs * 1000.0) as u64;
        }
    }
    // Default: 1 second
    1000
}

/// Map a [`ToolChoice`] to the OpenAI `tool_choice` body value.
///
/// `Required` becomes `"required"` only for local backends when constrained tool
/// calls are enabled; otherwise it degrades to `"auto"`, leaving cloud behavior
/// unchanged and honoring the config escape hatch.
fn tool_choice_value(tc: ToolChoice, api_base: &str, constrained: bool) -> serde_json::Value {
    match tc {
        ToolChoice::Required if is_local_api_base(api_base) && constrained => {
            serde_json::json!("required")
        }
        ToolChoice::None => serde_json::json!("none"),
        _ => serde_json::json!("auto"),
    }
}

pub(crate) fn is_local_api_base(api_base: &str) -> bool {
    let lower = api_base.to_ascii_lowercase();
    lower.contains("localhost")
        || lower.contains("127.0.0.1")
        || lower.contains("0.0.0.0")
        || is_private_ip(&lower)
}

/// Fold `role:"developer"` messages into the system message for local endpoints.
///
/// Local servers (higgs, llama.cpp chat templates) only speak
/// system|user|assistant|tool and return 500 on `developer` — which surfaced as
/// "I encountered an error: Server error" assistant turns. Cloud APIs that
/// understand `developer` (OpenAI) receive it unchanged. Same pattern as the
/// Anthropic mid-conversation `system` handling: adapt at the wire, not in the
/// assembler.
fn fold_developer_role_for_local(
    messages: Vec<serde_json::Value>,
    api_base: &str,
) -> Vec<serde_json::Value> {
    if !is_local_api_base(api_base)
        || !messages
            .iter()
            .any(|m| m.get("role").and_then(|r| r.as_str()) == Some("developer"))
    {
        return messages;
    }
    let mut out: Vec<serde_json::Value> = Vec::with_capacity(messages.len());
    let mut dev_texts: Vec<String> = Vec::new();
    for msg in messages {
        if msg.get("role").and_then(|r| r.as_str()) == Some("developer") {
            if let Some(text) = msg.get("content").and_then(|c| c.as_str()) {
                if !text.is_empty() {
                    dev_texts.push(text.to_string());
                }
            }
            continue; // never forward the developer role itself
        }
        out.push(msg);
    }
    if dev_texts.is_empty() {
        return out;
    }
    let folded = dev_texts.join("\n\n");
    match out
        .iter_mut()
        .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("system"))
    {
        Some(sys) => {
            let existing = sys.get("content").and_then(|c| c.as_str()).unwrap_or("");
            sys["content"] = serde_json::json!(format!("{existing}\n\n{folded}"));
        }
        None => out.insert(0, serde_json::json!({"role": "system", "content": folded})),
    }
    out
}

/// Check if a URL contains a private/LAN IP (RFC 1918).
fn is_private_ip(url: &str) -> bool {
    // Extract host portion from URL (between :// and next : or /)
    let host = url
        .find("://")
        .map(|i| &url[i + 3..])
        .unwrap_or(url)
        .split(&[':', '/'][..])
        .next()
        .unwrap_or("");

    // 10.0.0.0/8
    if host.starts_with("10.") {
        return true;
    }
    // 192.168.0.0/16
    if host.starts_with("192.168.") {
        return true;
    }
    // 172.16.0.0/12 (172.16.x.x – 172.31.x.x)
    if let Some(rest) = host.strip_prefix("172.") {
        if let Some(second) = rest.split('.').next().and_then(|s| s.parse::<u8>().ok()) {
            return (16..=31).contains(&second);
        }
    }
    false
}

#[derive(Default)]
struct HiggsRequestControl {
    session_id: Option<u64>,
    drop_session_ids: Vec<u64>,
    session_lease: Option<serde_json::Value>,
    session_cache_policy: Option<String>,
    max_prompt_tokens: Option<u32>,
}

fn request_messages_and_higgs_session_id(
    messages: &[serde_json::Value],
) -> (Vec<serde_json::Value>, HiggsRequestControl) {
    let mut control = HiggsRequestControl::default();
    let cleaned = messages
        .iter()
        .enumerate()
        .map(|(index, msg)| {
            let mut msg = msg.clone();
            if let Some(obj) = msg.as_object_mut() {
                obj.remove("ok");
                let marker = obj.remove(NANOBOT_HIGGS_SESSION_ID_FIELD);
                if index == 0 {
                    control.session_id = marker.and_then(|v| v.as_u64());
                }
                let drop_marker = obj.remove(NANOBOT_HIGGS_DROP_SESSION_ID_FIELD);
                if index == 0 {
                    if let Some(drop_session_id) = drop_marker.and_then(|v| v.as_u64()) {
                        control.drop_session_ids.push(drop_session_id);
                    }
                }
                let drop_markers = obj.remove(NANOBOT_HIGGS_DROP_SESSION_IDS_FIELD);
                if index == 0 {
                    if let Some(markers) = drop_markers.and_then(|v| v.as_array().cloned()) {
                        control
                            .drop_session_ids
                            .extend(markers.into_iter().filter_map(|v| v.as_u64()));
                    }
                }
                let lease = obj.remove(NANOBOT_HIGGS_SESSION_LEASE_FIELD);
                if index == 0 {
                    control.session_lease = lease.filter(|value| {
                        value.get("session_id").and_then(|v| v.as_u64()).is_some()
                            && value
                                .get("ttl_seconds")
                                .and_then(|v| v.as_u64())
                                .and_then(|value| u32::try_from(value).ok())
                                .is_some()
                    });
                }
                let policy = obj.remove(NANOBOT_HIGGS_SESSION_CACHE_POLICY_FIELD);
                if index == 0 {
                    control.session_cache_policy = policy
                        .and_then(|v| v.as_str().map(ToOwned::to_owned))
                        .filter(|value| {
                            matches!(value.as_str(), "best_effort" | "require_continuation")
                        });
                }
                let max_prompt_tokens = obj.remove(NANOBOT_HIGGS_MAX_PROMPT_TOKENS_FIELD);
                if index == 0 {
                    control.max_prompt_tokens = max_prompt_tokens
                        .and_then(|v| v.as_u64())
                        .and_then(|value| u32::try_from(value).ok());
                }
            }
            msg
        })
        .collect();
    control.drop_session_ids.sort_unstable();
    control.drop_session_ids.dedup();
    (cleaned, control)
}

/// Models that should keep template thinking enabled by default on local
/// OpenAI-compatible servers because the server can split it into
/// `reasoning_content` for the UI.
fn model_prefers_hidden_reasoning(model: &str) -> bool {
    crate::agent::model_capabilities::prefers_hidden_reasoning(model)
}

fn resolve_request_and_policy_model<'a>(
    api_base: &str,
    stripped_model: &'a str,
) -> (&'a str, &'a str) {
    let provider_model = if api_base.contains("openrouter") || api_base.starts_with("http://") {
        // OpenRouter: keep org/model for routing.
        // Local HTTP servers (LMS, vLLM, higgs): keep the full identifier.
        stripped_model
    } else {
        // Cloud HTTPS APIs (Anthropic, OpenAI, etc.): strip org prefix
        // (e.g. "anthropic/claude-opus-4-5" -> "claude-opus-4-5").
        stripped_model.split('/').last().unwrap_or(stripped_model)
    };

    // higgs-nightly serves each model under its real id and 404s on any other
    // name, so the wire (request) model and the policy model are the SAME real
    // id. The legacy `"active"` transport alias (one virtual name for whatever
    // was loaded) is gone: ctx.core.model always carries the real served id,
    // populated by startup discovery/`/v1/models` adoption or by `/model`
    // switch (which sets local_model to higgs's response id).
    (provider_model, provider_model)
}

/// Apply local reasoning controls when talking to localhost.
///
/// - `chat_template_kwargs.enable_thinking` toggles model reasoning mode for
///   templates that support it (Qwen3, etc.).
/// - `reasoning_budget` enforces a token budget for reasoning traces.
/// - `reasoning_format` tells the local server how to split visible vs reasoning text.
///
/// `/think` remains the explicit budgeted mode. A small allowlist of local
/// reasoning-first models keeps hidden reasoning enabled without imposing a
/// nanobot budget, so servers like Higgs can continue splitting it into
/// `reasoning_content` on every turn.
fn apply_local_reasoning_controls(
    body: &mut serde_json::Value,
    api_base: &str,
    model: &str,
    thinking_budget: Option<u32>,
) {
    if !is_local_api_base(api_base) {
        return;
    }

    if let Some(budget) = thinking_budget {
        body["chat_template_kwargs"] = serde_json::json!({
            "enable_thinking": true
        });
        body["reasoning_budget"] = serde_json::json!(budget);
        body["reasoning_format"] = serde_json::json!("deepseek");
    } else if model_prefers_hidden_reasoning(model) {
        body["chat_template_kwargs"] = serde_json::json!({
            "enable_thinking": true
        });
        body["reasoning_format"] = serde_json::json!("deepseek");
    } else {
        // Models like Qwen3.5 think by default (template-level). Without
        // explicit `enable_thinking: false` they burn all output tokens on
        // `<think>` blocks and produce no visible response.
        // Sent unconditionally: name-based thinking detection misses fine-tunes
        // whose served name hides the family (e.g. "qwythos-9b" is a Qwen3.5
        // template), and templates that don't define `enable_thinking` ignore
        // the kwarg. The assistant-prefill approach (`<think>\n</think>`)
        // doesn't work on servers like oMLX that reject assistant prefill.
        body["chat_template_kwargs"] = serde_json::json!({
            "enable_thinking": false
        });
    }
}

/// Suppress degenerate repetition loops on local servers.
///
/// Qwen 3.x recommends an additive presence penalty of 1.5. Higgs implements
/// the OpenAI/vLLM names `presence_penalty` and `repetition_penalty`; its
/// request parser ignores llama.cpp's `repeat_penalty`, which made the old
/// local safeguard a no-op on our production path. Non-Qwen local backends
/// retain the backend-specific 1.1 control for compatibility.
fn apply_repetition_controls(body: &mut serde_json::Value, api_base: &str, policy_model: &str) {
    if !is_local_api_base(api_base) {
        return;
    }
    let policy_model = policy_model.to_ascii_lowercase();
    if policy_model.contains("bonsai") {
        // Bonsai is Qwen3-derived, but the 1-bit compaction fine-tune loops
        // under the family-wide presence penalty. Higgs accepts this vLLM
        // field (and ignores llama.cpp's `repeat_penalty`).
        body["repetition_penalty"] = serde_json::json!(1.1);
    } else if policy_model.contains("qwen3") || policy_model.contains("agents-a1") {
        body["presence_penalty"] = serde_json::json!(1.5);
        body["repetition_penalty"] = serde_json::json!(1.0);
    } else {
        // 1.1 is the llama.cpp default and matches Ollama's default behavior.
        body["repeat_penalty"] = serde_json::json!(1.1);
    }
}

/// Ensure every object schema carries a `required` key (recursively).
///
/// Apple FM's guided generation returns HTTP 400 ("Invalid tool definition")
/// for any object schema that omits `required` — even no-arg tools (verified
/// live: `{"type":"object","properties":{}}` 400s; adding `"required":[]` 200s).
/// An empty `required: []` is a no-op for spec-compliant servers (where
/// `required` is optional and empty == absent), so this is applied
/// unconditionally. Deterministic → emitted tool schemas stay byte-stable
/// across turns (prefix-cache friendly).
fn ensure_required_keys(schema: &mut serde_json::Value) {
    let Some(obj) = schema.as_object_mut() else {
        return;
    };
    let is_object_schema = obj.get("type").and_then(|t| t.as_str()) == Some("object")
        || obj.contains_key("properties");
    if is_object_schema && !obj.contains_key("required") {
        obj.insert("required".to_string(), serde_json::Value::Array(vec![]));
    }
    if is_object_schema && !obj.contains_key("properties") {
        obj.insert(
            "properties".to_string(),
            serde_json::Value::Object(serde_json::Map::new()),
        );
    }
    if let Some(props) = obj.get_mut("properties").and_then(|p| p.as_object_mut()) {
        for (_, v) in props.iter_mut() {
            ensure_required_keys(v);
        }
    }
    if let Some(items) = obj.get_mut("items") {
        ensure_required_keys(items);
    }
}

fn is_apple_fm_request(body: &serde_json::Value) -> bool {
    matches!(
        body.get("model").and_then(|m| m.as_str()),
        Some("system" | "pcc")
    )
}

fn apple_fm_tool_parameters_supported(params: &serde_json::Value) -> bool {
    let Some(obj) = params.as_object() else {
        return false;
    };
    if obj.get("type").and_then(|t| t.as_str()) != Some("object") {
        return false;
    }
    let Some(props) = obj.get("properties").and_then(|p| p.as_object()) else {
        return true;
    };
    props.values().all(apple_fm_property_schema_supported)
}

fn apple_fm_property_schema_supported(schema: &serde_json::Value) -> bool {
    let Some(obj) = schema.as_object() else {
        return false;
    };
    match obj.get("type").and_then(|t| t.as_str()) {
        Some("string" | "integer" | "number" | "boolean") => true,
        Some("array") => obj
            .get("items")
            .map(apple_fm_array_item_schema_supported)
            .unwrap_or(false),
        // Live `fm serve` rejects nested object parameters and arrays of
        // objects with HTTP 400 "Invalid tool definition" before generation.
        Some("object") => false,
        _ => false,
    }
}

fn apple_fm_array_item_schema_supported(schema: &serde_json::Value) -> bool {
    let Some(obj) = schema.as_object() else {
        return false;
    };
    matches!(
        obj.get("type").and_then(|t| t.as_str()),
        Some("string" | "integer" | "number" | "boolean")
    )
}

fn filter_apple_fm_tool_schemas(body: &mut serde_json::Value) {
    if !is_apple_fm_request(body) {
        return;
    }

    let mut had_tools = false;
    let mut remaining = 0usize;
    if let Some(tools) = body.get_mut("tools").and_then(|t| t.as_array_mut()) {
        had_tools = true;
        tools.retain(|tool| {
            let supported = tool
                .get("function")
                .and_then(|f| f.get("parameters"))
                .map(apple_fm_tool_parameters_supported)
                .unwrap_or(false);
            if !supported {
                let name = tool
                    .get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(|n| n.as_str())
                    .unwrap_or("<unknown>");
                tracing::debug!(tool = name, "apple_fm_tool_schema_filtered");
            }
            supported
        });
        remaining = tools.len();
    }

    if had_tools && remaining == 0 {
        if let Some(obj) = body.as_object_mut() {
            obj.remove("tools");
            obj.remove("tool_choice");
            obj.remove("parallel_tool_calls");
        }
    }
}

/// Apply provider compatibility normalization to every tool in `body`.
fn normalize_tool_schemas(body: &mut serde_json::Value) {
    if let Some(tools) = body.get_mut("tools").and_then(|t| t.as_array_mut()) {
        for tool in tools.iter_mut() {
            if let Some(params) = tool
                .get_mut("function")
                .and_then(|f| f.get_mut("parameters"))
            {
                ensure_required_keys(params);
            }
        }
    }
    filter_apple_fm_tool_schemas(body);
}

fn find_first_tag(haystack: &str, tags: &[&str]) -> Option<(usize, usize)> {
    let mut best: Option<(usize, usize)> = None;
    for tag in tags {
        if let Some(idx) = haystack.find(tag) {
            let should_replace = match best {
                None => true,
                Some((best_idx, best_len)) => {
                    idx < best_idx || (idx == best_idx && tag.len() > best_len)
                }
            };
            if should_replace {
                best = Some((idx, tag.len()));
            }
        }
    }
    best
}

fn trailing_partial_tag_len(buffer: &str, tags: &[&str]) -> usize {
    let Some(start) = buffer.rfind('<') else {
        return 0;
    };
    let suffix = &buffer[start..];
    if tags.iter().any(|tag| tag.starts_with(suffix)) {
        suffix.len()
    } else {
        0
    }
}

/// Split one streamed content delta into visible text and reasoning text by
/// extracting `<think>...</think>` / `<thinking>...</thinking>` blocks.
pub(crate) fn split_thinking_from_content_delta(
    state: &mut ThinkSplitState,
    delta: &str,
) -> (String, String) {
    state.carry.push_str(delta);
    let mut visible = String::new();
    let mut reasoning = String::new();

    loop {
        if state.in_think_block {
            if let Some((idx, close_len)) = find_first_tag(&state.carry, &THINK_CLOSE_TAGS) {
                reasoning.push_str(&state.carry[..idx]);
                state.carry = state.carry[idx + close_len..].to_string();
                state.in_think_block = false;
                continue;
            }

            let keep = trailing_partial_tag_len(&state.carry, &THINK_CLOSE_TAGS);
            let emit_len = state.carry.len().saturating_sub(keep);
            if emit_len > 0 {
                reasoning.push_str(&state.carry[..emit_len]);
                state.carry = state.carry[emit_len..].to_string();
            }
            break;
        }

        if let Some((idx, open_len)) = find_first_tag(&state.carry, &THINK_OPEN_TAGS) {
            visible.push_str(&state.carry[..idx]);
            state.carry = state.carry[idx + open_len..].to_string();
            state.in_think_block = true;
            continue;
        }

        let keep = trailing_partial_tag_len(&state.carry, &THINK_OPEN_TAGS);
        let emit_len = state.carry.len().saturating_sub(keep);
        if emit_len > 0 {
            visible.push_str(&state.carry[..emit_len]);
            state.carry = state.carry[emit_len..].to_string();
        }
        break;
    }

    (visible, reasoning)
}

pub(crate) fn flush_thinking_split_state(state: &mut ThinkSplitState) -> (String, String) {
    if state.carry.is_empty() {
        return (String::new(), String::new());
    }

    let tail = std::mem::take(&mut state.carry);
    if state.in_think_block {
        state.in_think_block = false;
        (String::new(), tail)
    } else {
        (tail, String::new())
    }
}

fn extract_reasoning_delta(delta: &serde_json::Value) -> Option<&str> {
    delta
        .get("reasoning_content")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .or_else(|| {
            delta
                .get("reasoning")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
        })
}

/// The two request shapes built by [`OpenAICompatProvider::build_chat_request`].
#[derive(Clone, Copy)]
enum RequestKind {
    /// Non-streaming (`stream:false`); honors the caller's [`ToolChoice`].
    Blocking { tool_choice: ToolChoice },
    /// SSE streaming (`stream:true` + usage + local prefill progress);
    /// `tool_choice` is always `"auto"` on this path.
    Streaming,
}

/// Map a non-success chat/completions response to a [`crate::errors::ProviderError`].
///
/// Shared by `chat_impl` and `chat_stream` so streaming and non-streaming can
/// never disagree on error classification. Also emits the local-400 role
/// diagnostic (chat-template failures) and the generic API-error warning.
fn map_status_to_provider_error(
    status_code: u16,
    response_text: String,
    api_base: &str,
    body: &serde_json::Value,
    model_str: &str,
) -> crate::errors::ProviderError {
    use crate::errors::ProviderError;
    let body_snippet: String = response_text.chars().take(500).collect();
    if status_code == 400 && is_local_api_base(api_base) {
        if let Some(msgs) = body.get("messages").and_then(|m| m.as_array()) {
            let roles: Vec<&str> = msgs
                .iter()
                .filter_map(|m| m.get("role").and_then(|r| r.as_str()))
                .collect();
            tracing::error!(
                model = %model_str,
                status = status_code,
                body = %body_snippet,
                roles = ?roles,
                message_count = msgs.len(),
                "local_llm_400_message_roles"
            );
        }
    }
    warn!(
        model = %model_str,
        api_base = %api_base,
        status = status_code,
        body = %body_snippet,
        "llm_api_error"
    );
    let error_code = serde_json::from_str::<serde_json::Value>(&response_text)
        .ok()
        .and_then(|value| {
            value
                .pointer("/error/code")
                .or_else(|| value.pointer("/error/type"))
                .or_else(|| value.get("code"))
                .and_then(serde_json::Value::as_str)
                .map(str::to_string)
        });
    match status_code {
        429 => ProviderError::RateLimited {
            status: status_code,
            retry_after_ms: parse_retry_after_ms(&response_text),
        },
        401 | 403 => ProviderError::AuthError {
            status: status_code,
            message: response_text,
        },
        500..=599 => ProviderError::ServerError {
            status: status_code,
            message: response_text,
        },
        _ => ProviderError::HttpStatus {
            status: status_code,
            code: error_code,
            message: response_text,
        },
    }
}

impl OpenAICompatProvider {
    /// Build the `chat/completions` request body shared by `chat_impl` and
    /// `chat_stream`. Returns `(resolved_model, body)`.
    #[allow(clippy::too_many_arguments)]
    fn build_chat_request(
        &self,
        messages: &[serde_json::Value],
        tools: Option<&[serde_json::Value]>,
        model: Option<&str>,
        max_tokens: u32,
        temperature: f64,
        thinking_budget: Option<u32>,
        top_p: Option<f64>,
        kind: RequestKind,
    ) -> (String, serde_json::Value) {
        let normalized = model.map(|m| normalize_model_name(m));
        let raw_model = normalized.as_deref().unwrap_or(&self.default_model);
        // Strip "local:" prefix (internal routing tag, not part of actual model name)
        // and "provider/" prefix for non-OpenRouter APIs (e.g. "anthropic/claude-opus-4-5"
        // becomes "claude-opus-4-5" when hitting api.anthropic.com directly).
        let stripped = raw_model.strip_prefix("local:").unwrap_or(raw_model);
        let (model, policy_model) = resolve_request_and_policy_model(&self.api_base, stripped);

        debug!(
            "chat request: api_base={} raw_model={} stripped={} model={} policy_model={} streaming={}",
            self.api_base,
            raw_model,
            stripped,
            model,
            policy_model,
            matches!(kind, RequestKind::Streaming)
        );

        let (request_messages, higgs_control) = request_messages_and_higgs_session_id(messages);
        let request_messages = fold_developer_role_for_local(request_messages, &self.api_base);

        // Inject cache_control breakpoints for Anthropic prompt caching.
        let (cached_msgs, cached_tools) = if self.supports_cache_control(model) {
            inject_cache_control(&request_messages, tools)
        } else {
            (request_messages, tools.map(|t| t.to_vec()))
        };

        // Log message roles for debugging chat template errors from local servers.
        if is_local_api_base(&self.api_base) {
            let roles: Vec<&str> = cached_msgs
                .iter()
                .filter_map(|m| m.get("role").and_then(|r| r.as_str()))
                .collect();
            tracing::debug!(
                model = model,
                message_count = cached_msgs.len(),
                roles = ?roles,
                "local_llm_request_roles"
            );
        }

        let mut body = serde_json::json!({
            "model": model,
            "messages": cached_msgs,
            "max_tokens": max_tokens,
            "temperature": temperature,
        });
        if is_local_api_base(&self.api_base) {
            // Local servers own per-model sampling: higgs' config.toml
            // generation_defaults (and LM Studio presets) carry the
            // model-tuned temperature. A client-side value silently
            // overrides it — nanobot's generic 0.7 was beating the
            // Liquid-recommended 0.1 on LFM2.5. Cloud APIs keep ours.
            if let Some(obj) = body.as_object_mut() {
                obj.remove("temperature");
            }
        }
        match kind {
            // Explicit non-streaming: the OpenAI spec defaults `stream` to
            // false, but Apple FM's `fm serve` defaults to SSE when the field
            // is absent, which the non-streaming path can't parse. Declaring
            // it is a no-op for compliant servers and required for Apple FM.
            RequestKind::Blocking { .. } => {
                body["stream"] = serde_json::json!(false);
            }
            RequestKind::Streaming => {
                body["stream"] = serde_json::json!(true);
                body["stream_options"] = serde_json::json!({ "include_usage": true });
                if is_local_api_base(&self.api_base) {
                    // Ask llama.cpp/higgs servers to stream prefill progress
                    // (`prompt_progress` chunks) so the REPL can show a real %.
                    // Local-only: cloud APIs may reject unknown fields, and their
                    // prefill is not the bottleneck. Servers without support
                    // (LM Studio) ignore the field.
                    body["return_progress"] = serde_json::json!(true);
                }
            }
        }
        if let Some(tp) = top_p {
            body["top_p"] = serde_json::json!(tp);
        }
        if self.higgs_session_cache {
            if let Some(session_id) = higgs_control.session_id {
                body["session_id"] = serde_json::json!(session_id);
            }
            match higgs_control.drop_session_ids.as_slice() {
                [] => {}
                [drop_session_id] => {
                    body["drop_session_id"] = serde_json::json!(drop_session_id);
                }
                drop_session_ids => {
                    body["drop_session_ids"] = serde_json::json!(drop_session_ids);
                }
            }
            if let Some(session_lease) = &higgs_control.session_lease {
                body["session_lease"] = session_lease.clone();
            }
            if let Some(session_cache_policy) = &higgs_control.session_cache_policy {
                body["session_cache_policy"] = serde_json::json!(session_cache_policy);
            }
            if let Some(max_prompt_tokens) = higgs_control.max_prompt_tokens {
                body["max_prompt_tokens"] = serde_json::json!(max_prompt_tokens);
            }
        }
        // Definitive cache-engagement diagnostic. Both `higgs_session_cache`
        // (the provider flag, set when localBackend=higgs) AND `session_id`
        // (the marker extracted from the message array at line ~795) must be
        // present for body["session_id"] to be sent. If this logs
        // session_id_set=false on a higgs backend, retained session-KV reuse is
        // not engaged; exact radix/disk prefix caches may still hit.
        // Cross-reference with the response-side `local_llm_raw_usage` log:
        // session_id_set=true + cache_read=0 ⇒ the server received the id but
        // is not serving a cached prefix.
        debug!(
            api_base = %self.api_base,
            higgs_session_cache = self.higgs_session_cache,
            session_id_set = body.get("session_id").is_some(),
            session_id_value = ?higgs_control.session_id,
            drop_session_id_set = body.get("drop_session_id").is_some(),
            drop_session_ids_set = body.get("drop_session_ids").is_some(),
            drop_session_ids_value = ?higgs_control.drop_session_ids,
            "higgs_session_cache_request"
        );
        apply_local_reasoning_controls(&mut body, &self.api_base, policy_model, thinking_budget);

        // Forced tool calls ("required") only engage for local backends with
        // constrained tool calls enabled; elsewhere this is "auto" (unchanged).
        // The streaming path never forces tools.
        let tc_value = match kind {
            RequestKind::Blocking { tool_choice } => {
                tool_choice_value(tool_choice, &self.api_base, self.constrained_tool_calls)
            }
            RequestKind::Streaming => serde_json::json!("auto"),
        };
        if let Some(ref tool_defs) = cached_tools {
            if !tool_defs.is_empty() {
                body["tools"] = serde_json::Value::Array(tool_defs.clone());
                body["tool_choice"] = tc_value.clone();
            }
        } else if let Some(tool_defs) = tools {
            if !tool_defs.is_empty() {
                body["tools"] = serde_json::Value::Array(tool_defs.to_vec());
                body["tool_choice"] = tc_value;
            }
        }
        // Local models can enter a sampling loop where a token sequence echoes
        // until max_tokens; use the model/backend's actual sampling fields.
        apply_repetition_controls(&mut body, &self.api_base, policy_model);
        if let Some(repetition_penalty) = self.repetition_penalty {
            body["repetition_penalty"] = serde_json::json!(repetition_penalty);
        }
        if let Some(frequency_penalty) = self.frequency_penalty {
            body["frequency_penalty"] = serde_json::json!(frequency_penalty);
        }
        if let Some(presence_penalty) = self.presence_penalty {
            body["presence_penalty"] = serde_json::json!(presence_penalty);
        }
        // Strict-validation servers (Apple FM) reject object schemas that omit a
        // `required` key; normalize every outgoing tool schema (no-op elsewhere).
        normalize_tool_schemas(&mut body);

        (model.to_string(), body)
    }

    #[instrument(skip(self, messages, tools), fields(api_base = %self.api_base))]
    #[allow(clippy::too_many_arguments)]
    async fn chat_impl(
        &self,
        messages: &[serde_json::Value],
        tools: Option<&[serde_json::Value]>,
        model: Option<&str>,
        max_tokens: u32,
        temperature: f64,
        thinking_budget: Option<u32>,
        top_p: Option<f64>,
        tool_choice: ToolChoice,
    ) -> Result<LLMResponse> {
        let (model, body) = self.build_chat_request(
            messages,
            tools,
            model,
            max_tokens,
            temperature,
            thinking_budget,
            top_p,
            RequestKind::Blocking { tool_choice },
        );
        let url = format!("{}/chat/completions", self.api_base);
        let carries_one_shot_lease = body.get("session_lease").is_some();

        // JIT gate: serialise access to JIT-loading servers.
        // Measure JIT wait separately from the actual API call.
        let jit_wait_start = std::time::Instant::now();
        let _jit_permit = match &self.jit_gate {
            Some(gate) => Some(gate.acquire().await),
            None => None,
        };
        let jit_wait_ms = if self.jit_gate.is_some() {
            jit_wait_start.elapsed().as_millis() as u64
        } else {
            0
        };
        let call_start = std::time::Instant::now();

        let backoff = if self.jit_gate.is_some() {
            retry::jit_backoff()
        } else {
            retry::provider_backoff()
        };

        // Clone Arc-based/cheap values for the retry closure.
        let client = self.client.clone();
        let retry_url = url.clone();
        let api_key = self.api_key.clone();
        let api_base = self.api_base.clone();
        let model_owned = model.clone();

        use crate::errors::ProviderError;
        let result: Result<LLMResponse, ProviderError> = (|| {
            let client = client.clone();
            let url = retry_url.clone();
            let api_key = api_key.clone();
            let api_base = api_base.clone();
            let model_str = model_owned.clone();
            let body = body.clone();
            async move {
                let response = client
                    .post(&url)
                    .header("Authorization", format!("Bearer {}", api_key))
                    .header("Content-Type", "application/json")
                    .json(&body)
                    .send()
                    .await
                    .map_err(|e| {
                        warn!("HTTP request to LLM failed (base={}): {}", api_base, e);
                        ProviderError::HttpError(format!("Error calling LLM: {}", e))
                    })?;

                let status = response.status();
                let response_text = response
                    .text()
                    .await
                    .map_err(|e| ProviderError::ResponseReadError(e.to_string()))?;

                if !status.is_success() {
                    return Err(map_status_to_provider_error(
                        status.as_u16(),
                        response_text,
                        &api_base,
                        &body,
                        &model_str,
                    ));
                }

                let data: serde_json::Value = serde_json::from_str(&response_text)
                    .map_err(|e| ProviderError::JsonParseError(e.to_string()))?;

                parse_response(&data).map_err(|e| match e.downcast::<ProviderError>() {
                    Ok(pe) => pe,
                    Err(e) => ProviderError::JsonParseError(e.to_string()),
                })
            }
        })
        .retry(backoff)
        .when(|e| !carries_one_shot_lease && e.is_retryable())
        .notify(|e, dur: std::time::Duration| {
            warn!(error = %e, delay_ms = dur.as_millis() as u64, "provider_retry");
        })
        .adjust(retry::adjust_for_rate_limit)
        .await;

        let elapsed_ms = call_start.elapsed().as_millis() as u64;
        match &result {
            Ok(resp) => {
                let prompt_tokens = resp.usage.get("prompt_tokens").copied().unwrap_or(0);
                let completion_tokens = resp.usage.get("completion_tokens").copied().unwrap_or(0);
                info!(
                    elapsed_ms = elapsed_ms,
                    jit_wait_ms = jit_wait_ms,
                    model = %model,
                    tokens_prompt = prompt_tokens,
                    tokens_completion = completion_tokens,
                    api_base = %self.api_base,
                    "llm_call_complete"
                );
            }
            Err(e) => {
                warn!(
                    elapsed_ms = elapsed_ms,
                    jit_wait_ms = jit_wait_ms,
                    model = %model,
                    api_base = %self.api_base,
                    error = %e,
                    "llm_call_failed"
                );
            }
        }
        result.map_err(|e| e.into())
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
        thinking_budget: Option<u32>,
        top_p: Option<f64>,
    ) -> Result<LLMResponse> {
        self.chat_impl(
            messages,
            tools,
            model,
            max_tokens,
            temperature,
            thinking_budget,
            top_p,
            ToolChoice::Auto,
        )
        .await
    }

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
        tool_choice: ToolChoice,
    ) -> Result<LLMResponse> {
        self.chat_impl(
            messages,
            tools,
            model,
            max_tokens,
            temperature,
            thinking_budget,
            top_p,
            tool_choice,
        )
        .await
    }

    #[instrument(skip(self, messages, tools), fields(api_base = %self.api_base))]
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
        let (model, body) = self.build_chat_request(
            messages,
            tools,
            model,
            max_tokens,
            temperature,
            thinking_budget,
            top_p,
            RequestKind::Streaming,
        );
        let url = format!("{}/chat/completions", self.api_base);
        let carries_one_shot_lease = body.get("session_lease").is_some();

        // JIT gate: serialise access to JIT-loading servers.
        // For streaming, the permit is moved into the spawned task so it's held
        // for the entire stream duration, preventing model switches mid-stream.
        // Measure JIT wait separately from the actual API call.
        let jit_wait_start = std::time::Instant::now();
        let jit_permit = match &self.jit_gate {
            Some(gate) => Some(gate.acquire().await),
            None => None,
        };
        let jit_wait_ms = if self.jit_gate.is_some() {
            jit_wait_start.elapsed().as_millis() as u64
        } else {
            0
        };
        let call_start = std::time::Instant::now();

        let backoff = if self.jit_gate.is_some() {
            retry::jit_backoff()
        } else {
            retry::provider_backoff()
        };

        use crate::errors::ProviderError;
        let client = self.client.clone();
        let stream_url = url.clone();
        let api_key = self.api_key.clone();
        let api_base = self.api_base.clone();
        let model_owned = model.clone();

        let response: Result<reqwest::Response, ProviderError> = (|| {
            let client = client.clone();
            let url = stream_url.clone();
            let api_key = api_key.clone();
            let api_base = api_base.clone();
            let model_str = model_owned.clone();
            let body = body.clone();
            async move {
                let response = client
                    .post(&url)
                    .header("Authorization", format!("Bearer {}", api_key))
                    .header("Content-Type", "application/json")
                    .json(&body)
                    .send()
                    .await
                    .map_err(|e| ProviderError::HttpError(format!("Error calling LLM: {}", e)))?;

                let status = response.status();
                if !status.is_success() {
                    let error_text = response.text().await.unwrap_or_default();
                    return Err(map_status_to_provider_error(
                        status.as_u16(),
                        error_text,
                        &api_base,
                        &body,
                        &model_str,
                    ));
                }

                Ok(response)
            }
        })
        .retry(backoff)
        .when(|e| !carries_one_shot_lease && e.is_retryable())
        .notify(|e, dur: std::time::Duration| {
            warn!(error = %e, delay_ms = dur.as_millis() as u64, "provider_stream_retry");
        })
        .adjust(retry::adjust_for_rate_limit)
        .await;

        let response = match response {
            Ok(r) => r,
            Err(e) => {
                drop(jit_permit);
                return Err(e.into());
            }
        };

        let ttfb_ms = call_start.elapsed().as_millis() as u64;
        info!(
            ttfb_ms = ttfb_ms,
            jit_wait_ms = jit_wait_ms,
            model = %model,
            api_base = %self.api_base,
            "llm_stream_started"
        );

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

        // Spawn a task to parse the SSE stream.
        // The JIT permit is moved into the task and held until the stream ends,
        // preventing other providers from switching models mid-stream.
        let byte_stream = response.bytes_stream();
        let abort_on_drop = tokio::spawn(async move {
            parse_sse_stream(byte_stream, tx).await;
            // Permit drops here when the stream is fully consumed.
            drop(jit_permit);
        });

        Ok(StreamHandle {
            rx,
            abort_on_drop: Some(abort_on_drop),
        })
    }

    fn get_default_model(&self) -> &str {
        &self.default_model
    }

    fn get_api_base(&self) -> Option<&str> {
        Some(&self.api_base)
    }

    fn supports_higgs_session_cache(&self) -> bool {
        self.higgs_session_cache
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

/// Parse tool-call `arguments` into a param map.
///
/// Local models sometimes double-encode the payload: the JSON string
/// decodes to another JSON string rather than an object. Unwrap one extra
/// layer before giving up, otherwise the arguments are silently lost.
fn parse_tool_arguments(s: &str) -> Result<HashMap<String, serde_json::Value>, String> {
    match serde_json::from_str::<serde_json::Value>(s) {
        Ok(serde_json::Value::Object(map)) => Ok(map.into_iter().collect()),
        Ok(serde_json::Value::String(inner)) => {
            match serde_json::from_str::<serde_json::Value>(&inner) {
                Ok(serde_json::Value::Object(map)) => Ok(map.into_iter().collect()),
                _ => Err("double-encoded arguments did not decode to an object".to_string()),
            }
        }
        Ok(_) => Err("arguments did not decode to an object".to_string()),
        Err(e) => Err(e.to_string()),
    }
}

/// Parse the OpenAI-compatible JSON response into an `LLMResponse`.

/// Parse LFM2-format tool calls from model content text.
///
/// LFM2 models (LFM2.5, Macaw) output Pythonic function calls between
/// `<|tool_call_start|>` and `<|tool_call_end|>` tokens:
///   `<|tool_call_start|>[func(arg='val')]<|tool_call_end|>`
///
/// When the model is trained with its native template (which renders
/// tool-call history in this format), its live output also uses this
/// format rather than OpenAI JSON `tool_calls`.
fn parse_lfm2_tool_calls(content: &str) -> Vec<crate::providers::base::ToolCallRequest> {
    use regex::Regex;
    use std::sync::LazyLock;

    static LFM2_RE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"<\|tool_call_start\|>\s*\[(.*?)\]\s*<\|tool_call_end\|>").unwrap()
    });

    let mut out = Vec::new();
    let Some(caps) = LFM2_RE.captures(content) else {
        return out;
    };
    let raw = caps.get(1).map(|m| m.as_str()).unwrap_or("");
    // Split comma-separated calls.  The list is Pythonic:
    //   func1(), func2(arg='val')
    // Commas inside quotes are not call separators.
    let calls = split_lfm2_calls(raw);
    for (idx, call) in calls.iter().enumerate() {
        let call = call.trim();
        if call.is_empty() {
            continue;
        }
        let (name, args_str) = match call.split_once('(') {
            Some((n, rest)) => {
                let rest = rest.strip_suffix(')').unwrap_or(rest);
                (n.trim().to_string(), rest.trim().to_string())
            }
            None => continue,
        };
        if !is_valid_tool_call_name(&name) {
            continue;
        }
        let args_map = parse_lfm2_kwargs(&args_str);
        out.push(crate::providers::base::ToolCallRequest {
            id: format!(
                "lfm2_call_{}_{}",
                idx,
                uuid::Uuid::new_v4()
                    .to_string()
                    .chars()
                    .take(8)
                    .collect::<String>()
            ),
            name,
            arguments: args_map,
        });
    }
    out
}

/// Split "func1(), func2(a='b')" into ["func1()", "func2(a='b')"].
/// Strip LFM2 tool-call markers from content so the model doesn't
/// see its own tool calls as raw text when they're echoed in history.
fn strip_lfm2_markers(content: &str) -> String {
    use regex::Regex;
    use std::sync::LazyLock;
    static LFM2_STRIP: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"<\|tool_call_start\|>.*?<\|tool_call_end\|>").unwrap());
    LFM2_STRIP.replace_all(content, "").trim().to_string()
}

fn split_lfm2_calls(raw: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth: i32 = 0;
    let mut in_string = false;
    let mut start = 0;
    for (i, ch) in raw.char_indices() {
        match ch {
            '\'' | '\"' => in_string = !in_string,
            '(' => {
                if !in_string {
                    depth += 1
                }
            }
            ')' => {
                if !in_string {
                    depth = depth.saturating_sub(1)
                }
            }
            ',' => {
                if depth == 0 && !in_string {
                    out.push(raw[start..i].to_string());
                    start = i + 1;
                }
            }
            _ => {}
        }
    }
    if start < raw.len() {
        out.push(raw[start..].to_string());
    }
    out
}

/// Parse keyword-style arguments: `a='val', b=42, c=True` -> HashMap.
fn parse_lfm2_kwargs(args: &str) -> std::collections::HashMap<String, serde_json::Value> {
    use std::collections::HashMap;
    let mut map = HashMap::new();
    if args.is_empty() {
        return map;
    }
    let mut key = String::new();
    let mut val = String::new();
    let mut in_key = true;
    let mut in_string = false;
    let mut string_char = '"';
    for ch in args.chars() {
        if in_string {
            if ch == string_char {
                in_string = false;
            } else {
                val.push(ch);
            }
            continue;
        }
        if in_key && ch == '=' {
            in_key = false;
            continue;
        }
        if !in_key && (ch == '\'' || ch == '"') {
            in_string = true;
            string_char = ch;
            continue;
        }
        if ch == ',' && !in_string {
            let k = key.trim().to_string();
            let v = val.trim().to_string();
            if !k.is_empty() {
                // Try as number/bool, fall back to string
                if let Ok(n) = v.parse::<i64>() {
                    map.insert(k, serde_json::Value::Number(n.into()));
                } else if v == "True" || v == "true" {
                    map.insert(k, serde_json::Value::Bool(true));
                } else if v == "False" || v == "false" {
                    map.insert(k, serde_json::Value::Bool(false));
                } else {
                    map.insert(k, serde_json::Value::String(v));
                }
            }
            key.clear();
            val.clear();
            in_key = true;
            continue;
        }
        if in_key {
            key.push(ch);
        } else {
            val.push(ch);
        }
    }
    // Last arg
    let k = key.trim().to_string();
    let v = val.trim().to_string();
    if !k.is_empty() {
        let parsed = if let Ok(n) = v.parse::<i64>() {
            serde_json::Value::Number(n.into())
        } else if v == "True" || v == "true" {
            serde_json::Value::Bool(true)
        } else if v == "False" || v == "false" {
            serde_json::Value::Bool(false)
        } else {
            serde_json::Value::String(v)
        };
        map.insert(k, parsed);
    }
    map
}

fn parse_response(data: &serde_json::Value) -> Result<LLMResponse> {
    let choices = data
        .get("choices")
        .and_then(|c| c.as_array())
        .cloned()
        .unwrap_or_default();

    if choices.is_empty() {
        return Err(crate::errors::ProviderError::JsonParseError(
            "No choices in LLM response".into(),
        )
        .into());
    }

    let choice = &choices[0];
    let message = choice.get("message").cloned().unwrap_or_default();
    let finish_reason = FinishReason::parse_finish_reason(
        choice
            .get("finish_reason")
            .and_then(|v| v.as_str())
            .unwrap_or("stop"),
    );

    // Extract reasoning_content (separate field used by reasoning models).
    let reasoning_text = message
        .get("reasoning_content")
        .or_else(|| message.get("reasoning"))
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string());

    let content = message
        .get("content")
        .and_then(|v| v.as_str())
        .and_then(|raw| {
            if raw.is_empty() {
                return None;
            }
            let mut split_state = ThinkSplitState::default();
            let (mut visible, mut inline_reasoning) =
                split_thinking_from_content_delta(&mut split_state, raw);
            let (tail_visible, tail_reasoning) = flush_thinking_split_state(&mut split_state);
            visible.push_str(&tail_visible);
            inline_reasoning.push_str(&tail_reasoning);
            if !inline_reasoning.is_empty() {
                debug!(
                    "Model returned inline think tags ({} chars visible, {} chars reasoning)",
                    visible.len(),
                    inline_reasoning.len()
                );
            }
            let cleaned = visible.trim().to_string();
            if cleaned.is_empty() {
                // Model put everything in <think> blocks with no visible output.
                // This happens when servers (e.g. oMLX) ignore enable_thinking:false
                // and Qwen3.5 thinks by default. Use inline reasoning as fallback
                // so the caller doesn't get an empty response.
                if !inline_reasoning.is_empty() {
                    debug!(
                        "content empty after think-strip, using inline reasoning ({} chars) as fallback",
                        inline_reasoning.len()
                    );
                    Some(inline_reasoning.trim().to_string()).filter(|s| !s.is_empty())
                } else {
                    None
                }
            } else {
                Some(cleaned)
            }
        });

    // Fallback: if content is empty but reasoning_content is present, use it.
    // Some models (e.g. NanBeige) put all output in reasoning_content with
    // empty content — without this fallback the model appears silent.
    let content = if content.is_none() {
        if let Some(ref reasoning) = reasoning_text {
            debug!(
                "content empty, using reasoning_content ({} chars) as fallback",
                reasoning.len()
            );
            Some(reasoning.trim().to_string()).filter(|s| !s.is_empty())
        } else {
            None
        }
    } else {
        if let Some(ref reasoning) = reasoning_text {
            debug!(
                "Model returned reasoning_content ({} chars), discarding from output",
                reasoning.len()
            );
        }
        content
    };

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
            if !is_valid_tool_call_name(&name) {
                warn!(
                    id = %id,
                    raw_name = %name,
                    "dropping_malformed_tool_call_name"
                );
                continue;
            }

            // Arguments come as a JSON string that we need to parse.
            let arguments_raw = function
                .get("arguments")
                .cloned()
                .unwrap_or(serde_json::Value::String("{}".to_string()));

            let arguments: HashMap<String, serde_json::Value> =
                if let Some(s) = arguments_raw.as_str() {
                    match parse_tool_arguments(s) {
                        Ok(map) => map,
                        Err(e) => {
                            warn!(
                                tool = %name,
                                error = %e,
                                raw_args = %s,
                                "malformed_tool_call_json"
                            );
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

    // LFM2 fallback: models with native templates (Macaw, LFM2.5)
    // output `<|tool_call_start|>[func(args)]<|tool_call_end|>` in the
    // content text body.  When the OpenAI `tool_calls` array is empty
    // but the content carries LFM2 markers, parse them.
    let (content, tool_calls) = if tool_calls.is_empty()
        && content
            .as_ref()
            .is_some_and(|c| c.contains("<|tool_call_start|>"))
    {
        let lfm2_calls = parse_lfm2_tool_calls(content.as_deref().unwrap_or(""));
        let cleaned = strip_lfm2_markers(content.as_deref().unwrap_or(""));
        (Some(cleaned), lfm2_calls)
    } else {
        (content, tool_calls)
    };

    // Extract usage.
    let mut usage = HashMap::new();
    if let Some(usage_obj) = data.get("usage").and_then(|v| v.as_object()) {
        extract_usage_numbers(usage_obj, &mut usage);
    }

    Ok(LLMResponse {
        content,
        tool_calls,
        finish_reason,
        usage,
    })
}

/// Normalize the two common OpenAI usage shapes into the flat map shared by
/// all nanobot providers. `prompt_tokens_details.cached_tokens` is emitted by
/// OpenAI-compatible local servers, while Anthropic-style providers already
/// use the flat `cache_read_input_tokens` name.
fn extract_usage_numbers(
    usage_obj: &serde_json::Map<String, serde_json::Value>,
    usage: &mut HashMap<String, i64>,
) {
    for (key, value) in usage_obj {
        if let Some(n) = value.as_i64() {
            usage.insert(key.clone(), n);
        }
    }
    let cached_tokens = usage_obj
        .get("prompt_tokens_details")
        .or_else(|| usage_obj.get("input_tokens_details"))
        .and_then(|details| details.get("cached_tokens"))
        .and_then(serde_json::Value::as_i64);
    if let Some(cached_tokens) = cached_tokens {
        usage.insert("cache_read_input_tokens".to_string(), cached_tokens);
    }
    if let Some(prompt_tokens) = usage.get("prompt_tokens").copied() {
        let cached_tokens = usage.get("cache_read_input_tokens").copied().unwrap_or(0);
        usage.insert("cache_read_input_tokens".to_string(), cached_tokens);
        usage.insert(
            "cache_creation_input_tokens".to_string(),
            prompt_tokens.saturating_sub(cached_tokens).max(0),
        );
    }
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
    let mut full_reasoning = String::new(); // API reasoning_content field only
    let mut full_inline_thinking = String::new(); // inline <think> tags — fallback when content empty
    let mut split_state = ThinkSplitState::default();
    let mut finish_reason = FinishReason::Stop;
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
            let line = line_buffer[..newline_pos]
                .trim_end_matches('\r')
                .to_string();
            line_buffer = line_buffer[newline_pos + 1..].to_string();

            if line.is_empty() {
                continue;
            }

            // SSE comments are transport heartbeats. Higgs emits `:` while a
            // request is queued or generating; preserve that liveness signal
            // so the agent watchdog does not cancel a healthy queued request.
            // This is deliberately not text, TTFT, or prefill progress.
            if line.starts_with(':') {
                let _ = tx.send(StreamChunk::TransportProgress);
                continue;
            }

            if !line.starts_with("data: ") {
                continue;
            }

            let data = &line[6..];

            if data == "[DONE]" {
                let (tail_content, tail_reasoning) = flush_thinking_split_state(&mut split_state);
                if !tail_reasoning.is_empty() {
                    full_inline_thinking.push_str(&tail_reasoning);
                    let _ = tx.send(StreamChunk::ThinkingDelta(tail_reasoning));
                }
                if !tail_content.is_empty() {
                    full_content.push_str(&tail_content);
                    let _ = tx.send(StreamChunk::TextDelta(tail_content));
                }

                // Fallback: if content is empty but reasoning is present, use reasoning.
                let content = if !full_content.is_empty() {
                    if !full_reasoning.is_empty() {
                        debug!(
                            "Streaming: discarding reasoning_content ({} chars)",
                            full_reasoning.len()
                        );
                    }
                    Some(full_content.clone())
                } else if !full_reasoning.is_empty() {
                    debug!(
                        "Streaming: content empty, using reasoning_content ({} chars) as fallback",
                        full_reasoning.len()
                    );
                    Some(full_reasoning.clone())
                } else if !full_inline_thinking.is_empty() {
                    // Model put everything in <think> blocks with no visible
                    // content — happens when oMLX ignores enable_thinking:false.
                    debug!(
                        "Streaming: content empty, using inline <think> ({} chars) as fallback",
                        full_inline_thinking.len()
                    );
                    Some(full_inline_thinking.clone())
                } else {
                    None
                };

                let mut tool_calls = Vec::new();
                let mut indices: Vec<u64> = tool_calls_acc.keys().copied().collect();
                indices.sort();
                for idx in indices {
                    let Some((id, name, args_str)) = tool_calls_acc.remove(&idx) else {
                        continue;
                    };
                    if !is_valid_tool_call_name(&name) {
                        warn!(
                            id = %id,
                            raw_name = %name,
                            "dropping_malformed_tool_call_name"
                        );
                        continue;
                    }
                    let arguments: HashMap<String, serde_json::Value> =
                        match parse_tool_arguments(&args_str) {
                            Ok(map) => map,
                            Err(e) => {
                                warn!(
                                    tool = %name,
                                    error = %e,
                                    raw_args = %args_str,
                                    "malformed_tool_call_json"
                                );
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

                // LFM2 fallback: same as parse_response path.
                let (content, tool_calls) = if tool_calls.is_empty()
                    && content
                        .as_ref()
                        .is_some_and(|c| c.contains("<|tool_call_start|>"))
                {
                    let lfm2_calls = parse_lfm2_tool_calls(content.as_deref().unwrap_or(""));
                    let cleaned = strip_lfm2_markers(content.as_deref().unwrap_or(""));
                    (Some(cleaned), lfm2_calls)
                } else {
                    (content, tool_calls)
                };

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
                    warn!("SSE parse error (skipping chunk): {}", e);
                    continue;
                }
            };

            // Prefill progress (llama.cpp/higgs `return_progress`): these
            // chunks have empty choices and carry only prompt_progress.
            if let Some(pp) = chunk.get("prompt_progress") {
                let processed = pp.get("processed").and_then(|v| v.as_u64()).unwrap_or(0);
                let total = pp.get("total").and_then(|v| v.as_u64()).unwrap_or(0);
                if total > 0 {
                    let _ = tx.send(StreamChunk::PrefillProgress { processed, total });
                }
            }

            // Extract from choices[0].delta
            if let Some(choices) = chunk.get("choices").and_then(|c| c.as_array()) {
                if let Some(choice) = choices.first() {
                    // Update finish_reason if present (wire boundary: parse once here).
                    if let Some(fr) = choice.get("finish_reason").and_then(|v| v.as_str()) {
                        finish_reason = FinishReason::parse_finish_reason(fr);
                    }

                    if let Some(delta) = choice.get("delta") {
                        // Reasoning content delta (reasoning_content / reasoning field).
                        if let Some(reasoning) = extract_reasoning_delta(delta) {
                            if !reasoning.is_empty() {
                                full_reasoning.push_str(reasoning);
                                let _ = tx.send(StreamChunk::ThinkingDelta(reasoning.to_string()));
                            }
                        }

                        // Text content delta (may include inline <think> blocks).
                        if let Some(content) = delta.get("content").and_then(|v| v.as_str()) {
                            if !content.is_empty() {
                                let (visible, inline_reasoning) =
                                    split_thinking_from_content_delta(&mut split_state, content);
                                if !inline_reasoning.is_empty() {
                                    full_inline_thinking.push_str(&inline_reasoning);
                                    let _ = tx.send(StreamChunk::ThinkingDelta(inline_reasoning));
                                }
                                if !visible.is_empty() {
                                    full_content.push_str(&visible);
                                    let _ = tx.send(StreamChunk::TextDelta(visible));
                                }
                            }
                        }

                        // Tool call deltas
                        if let Some(tc_array) = delta.get("tool_calls").and_then(|v| v.as_array()) {
                            for tc in tc_array {
                                let index = tc.get("index").and_then(|v| v.as_u64()).unwrap_or(0);
                                let entry = tool_calls_acc.entry(index).or_insert_with(|| {
                                    (String::new(), String::new(), String::new())
                                });

                                if let Some(id) = tc.get("id").and_then(|v| v.as_str()) {
                                    entry.0 = id.to_string();
                                }
                                if let Some(function) = tc.get("function") {
                                    if let Some(name) =
                                        function.get("name").and_then(|v| v.as_str())
                                    {
                                        entry.1 = name.to_string();
                                        // Name arrives in a call's first fragment —
                                        // signal end-of-prefill for TTFT tracking
                                        // (pure tool-call responses have no
                                        // text/thinking deltas).
                                        let _ = tx.send(StreamChunk::ToolCallDelta);
                                    }
                                    // Accumulate argument fragments. The OpenAI
                                    // standard streams `arguments` as string
                                    // fragments, but some local servers (oMLX/MLX
                                    // for qwen3.6 / diffusiongemma / GLM) send the
                                    // whole thing in one chunk as a JSON *object*.
                                    // `parse_response` (non-streaming) handles both;
                                    // mirror that here so the args aren't silently
                                    // dropped → empty params. See LM Studio #1868.
                                    if let Some(args_val) = function.get("arguments") {
                                        if let Some(s) = args_val.as_str() {
                                            entry.2.push_str(s);
                                            if !s.is_empty() {
                                                let _ = tx.send(StreamChunk::ToolCallDelta);
                                            }
                                        } else if args_val.is_object() || args_val.is_array() {
                                            entry.2.push_str(
                                                &serde_json::to_string(args_val)
                                                    .unwrap_or_default(),
                                            );
                                            let _ = tx.send(StreamChunk::ToolCallDelta);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Extract usage if present (some providers include it in the last chunk)
            if let Some(usage_obj) = chunk.get("usage").and_then(|v| v.as_object()) {
                extract_usage_numbers(usage_obj, &mut usage);
            }
        }
    }

    let (tail_content, tail_reasoning) = flush_thinking_split_state(&mut split_state);
    if !tail_reasoning.is_empty() {
        full_inline_thinking.push_str(&tail_reasoning);
        let _ = tx.send(StreamChunk::ThinkingDelta(tail_reasoning));
    }
    if !tail_content.is_empty() {
        full_content.push_str(&tail_content);
        let _ = tx.send(StreamChunk::TextDelta(tail_content));
    }

    // Stream ended without [DONE] — SLM may have crashed or dropped connection.
    // Treat an abnormal termination during content generation as "length" so
    // the auto-continue mechanism can detect and recover from it.
    if finish_reason == FinishReason::Stop {
        finish_reason = FinishReason::Length;
    }
    warn!(
        content_len = full_content.len(),
        reasoning_len = full_reasoning.len(),
        inline_thinking_len = full_inline_thinking.len(),
        tool_calls = tool_calls_acc.len(),
        "sse_stream_ended_without_done"
    );
    if full_content.is_empty()
        && full_reasoning.is_empty()
        && full_inline_thinking.is_empty()
        && tool_calls_acc.is_empty()
    {
        let _ = tx.send(StreamChunk::Done(LLMResponse {
            content: Some(
                "LLM stream ended before the backend produced any response content or tool-call payload."
                    .to_string(),
            ),
            tool_calls: Vec::new(),
            finish_reason: FinishReason::ProviderFailure,
            usage,
        }));
        return;
    }
    // Fallback chain: API reasoning_content first, then inline <think> blocks.
    let content = if !full_content.is_empty() {
        if !full_reasoning.is_empty() {
            debug!(
                "Streaming (no DONE): discarding reasoning_content ({} chars)",
                full_reasoning.len()
            );
        }
        Some(full_content)
    } else if !full_reasoning.is_empty() {
        debug!(
            "Streaming (no DONE): content empty, using reasoning_content ({} chars) as fallback",
            full_reasoning.len()
        );
        Some(full_reasoning)
    } else if !full_inline_thinking.is_empty() {
        debug!(
            "Streaming (no DONE): content empty, using inline <think> ({} chars) as fallback",
            full_inline_thinking.len()
        );
        Some(full_inline_thinking)
    } else {
        None
    };

    let mut tool_calls = Vec::new();
    let mut indices: Vec<u64> = tool_calls_acc.keys().copied().collect();
    indices.sort();
    for idx in indices {
        let Some((id, name, args_str)) = tool_calls_acc.remove(&idx) else {
            continue;
        };
        if !is_valid_tool_call_name(&name) {
            warn!(
                id = %id,
                raw_name = %name,
                "dropping_malformed_tool_call_name"
            );
            continue;
        }
        let arguments: HashMap<String, serde_json::Value> = match parse_tool_arguments(&args_str) {
            Ok(map) => map,
            Err(e) => {
                warn!(
                    tool = %name,
                    error = %e,
                    raw_args = %args_str,
                    "malformed_tool_call_json"
                );
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

    /// chat and chat_stream must build identical request bodies apart from
    /// the streaming-only fields — the whole point of build_chat_request.
    #[test]
    fn test_blocking_and_streaming_request_bodies_agree() {
        let provider =
            OpenAICompatProvider::new("local", Some("http://127.0.0.1:1234/v1"), Some("qwen3-8b"));
        let messages = vec![
            serde_json::json!({"role": "system", "content": "prefix"}),
            serde_json::json!({"role": "user", "content": "hi"}),
        ];
        let tools = vec![serde_json::json!({
            "type": "function",
            "function": {"name": "t", "description": "d",
                         "parameters": {"type": "object", "properties": {}}}
        })];

        let (model_b, mut blocking) = provider.build_chat_request(
            &messages,
            Some(&tools),
            Some("qwen3-8b"),
            256,
            0.7,
            None,
            Some(0.9),
            RequestKind::Blocking {
                tool_choice: ToolChoice::Auto,
            },
        );
        let (model_s, mut streaming) = provider.build_chat_request(
            &messages,
            Some(&tools),
            Some("qwen3-8b"),
            256,
            0.7,
            None,
            Some(0.9),
            RequestKind::Streaming,
        );

        assert_eq!(model_b, model_s);
        assert_eq!(blocking["stream"], serde_json::json!(false));
        assert_eq!(streaming["stream"], serde_json::json!(true));
        assert!(blocking.get("parallel_tool_calls").is_none());
        assert!(streaming.get("parallel_tool_calls").is_none());

        // Strip the fields that legitimately differ; the rest must be identical.
        for body in [&mut blocking, &mut streaming] {
            let obj = body.as_object_mut().unwrap();
            obj.remove("stream");
            obj.remove("stream_options");
            obj.remove("return_progress");
        }
        assert_eq!(blocking, streaming);
    }

    #[test]
    fn test_build_chat_request_strips_internal_tool_result_status() {
        let provider = OpenAICompatProvider::new("sk-test", Some(OPENAI_API_BASE), Some("gpt-4o"));
        let messages = vec![
            serde_json::json!({"role": "assistant", "content": "", "tool_calls": [{
                "id": "tc_1",
                "type": "function",
                "function": {"name": "read_file", "arguments": "{\"path\":\"x\"}"}
            }]}),
            serde_json::json!({
                "role": "tool",
                "tool_call_id": "tc_1",
                "name": "read_file",
                "ok": false,
                "content": "Error: File not found",
            }),
        ];

        let (_, body) = provider.build_chat_request(
            &messages,
            None,
            Some("gpt-4o"),
            64,
            0.0,
            None,
            None,
            RequestKind::Blocking {
                tool_choice: ToolChoice::Auto,
            },
        );

        let sent_tool = body["messages"][1].as_object().unwrap();
        assert!(!sent_tool.contains_key("ok"));
        assert_eq!(sent_tool.get("content").unwrap(), "Error: File not found");
    }

    #[test]
    fn test_build_chat_request_injects_sampling_penalties_when_set() {
        let provider =
            OpenAICompatProvider::new("local", Some("http://127.0.0.1:1234/v1"), Some("qwen3-8b"))
                .with_sampling_penalties(Some(1.15), Some(0.7), Some(-0.3));
        let messages = vec![serde_json::json!({"role": "user", "content": "hi"})];

        let (_, body) = provider.build_chat_request(
            &messages,
            None,
            Some("qwen3-8b"),
            256,
            0.2,
            None,
            Some(0.9),
            RequestKind::Blocking {
                tool_choice: ToolChoice::Auto,
            },
        );

        assert_eq!(body["repetition_penalty"], serde_json::json!(1.15));
        assert_eq!(body["frequency_penalty"], serde_json::json!(0.7));
        assert_eq!(body["presence_penalty"], serde_json::json!(-0.3));
    }

    #[test]
    fn test_build_chat_request_omits_sampling_penalties_when_unset() {
        let provider =
            OpenAICompatProvider::new("local", Some("http://127.0.0.1:1234/v1"), Some("qwen3-8b"));
        let messages = vec![serde_json::json!({"role": "user", "content": "hi"})];

        let (_, body) = provider.build_chat_request(
            &messages,
            None,
            Some("llama-3.2-1b"),
            256,
            0.2,
            None,
            Some(0.9),
            RequestKind::Blocking {
                tool_choice: ToolChoice::Auto,
            },
        );

        assert!(body.get("repetition_penalty").is_none());
        assert!(body.get("frequency_penalty").is_none());
        assert!(body.get("presence_penalty").is_none());
    }

    #[test]
    fn test_request_messages_extracts_higgs_session_marker() {
        let messages = vec![
            serde_json::json!({
                "role": "system",
                "content": "stable prefix",
                NANOBOT_HIGGS_SESSION_ID_FIELD: 42_u64,
                NANOBOT_HIGGS_DROP_SESSION_ID_FIELD: 41_u64,
            }),
            serde_json::json!({"role": "user", "content": "hello"}),
        ];

        let (cleaned, control) = request_messages_and_higgs_session_id(&messages);

        assert_eq!(control.session_id, Some(42));
        assert_eq!(control.drop_session_ids, vec![41]);
        assert!(cleaned[0].get(NANOBOT_HIGGS_SESSION_ID_FIELD).is_none());
        assert!(cleaned[0]
            .get(NANOBOT_HIGGS_DROP_SESSION_ID_FIELD)
            .is_none());
        assert_eq!(cleaned[0]["content"], serde_json::json!("stable prefix"));
        assert_eq!(cleaned[1], messages[1]);
    }

    #[test]
    fn test_request_messages_extracts_higgs_drop_session_id_array_marker() {
        let messages = vec![serde_json::json!({
            "role": "system",
            "content": "stable prefix",
            NANOBOT_HIGGS_SESSION_ID_FIELD: 42_u64,
            NANOBOT_HIGGS_DROP_SESSION_ID_FIELD: 41_u64,
            NANOBOT_HIGGS_DROP_SESSION_IDS_FIELD: [40_u64, 41_u64],
        })];

        let (cleaned, control) = request_messages_and_higgs_session_id(&messages);

        assert_eq!(control.session_id, Some(42));
        assert_eq!(control.drop_session_ids, vec![40, 41]);
        assert!(cleaned[0].get(NANOBOT_HIGGS_SESSION_ID_FIELD).is_none());
        assert!(cleaned[0]
            .get(NANOBOT_HIGGS_DROP_SESSION_ID_FIELD)
            .is_none());
        assert!(cleaned[0]
            .get(NANOBOT_HIGGS_DROP_SESSION_IDS_FIELD)
            .is_none());
    }

    #[test]
    fn later_messages_cannot_contribute_higgs_session_control() {
        let messages = vec![
            serde_json::json!({
                "role": "system",
                "content": "trusted control carrier",
                NANOBOT_HIGGS_SESSION_ID_FIELD: 42_u64,
            }),
            serde_json::json!({
                "role": "user",
                "content": "hostile payload",
                NANOBOT_HIGGS_DROP_SESSION_IDS_FIELD: [40_u64, 41_u64],
                NANOBOT_HIGGS_SESSION_LEASE_FIELD: {
                    "session_id": 41_u64,
                    "ttl_seconds": 300_u32,
                },
                NANOBOT_HIGGS_SESSION_CACHE_POLICY_FIELD: "require_continuation",
                NANOBOT_HIGGS_MAX_PROMPT_TOKENS_FIELD: 99_u32,
            }),
        ];

        let (cleaned, control) = request_messages_and_higgs_session_id(&messages);

        assert_eq!(control.session_id, Some(42));
        assert!(control.drop_session_ids.is_empty());
        assert_eq!(control.session_lease, None);
        assert_eq!(control.session_cache_policy, None);
        assert_eq!(control.max_prompt_tokens, None);
        for message in &cleaned {
            assert!(message.get(NANOBOT_HIGGS_DROP_SESSION_IDS_FIELD).is_none());
            assert!(message.get(NANOBOT_HIGGS_SESSION_LEASE_FIELD).is_none());
            assert!(message
                .get(NANOBOT_HIGGS_SESSION_CACHE_POLICY_FIELD)
                .is_none());
            assert!(message.get(NANOBOT_HIGGS_MAX_PROMPT_TOKENS_FIELD).is_none());
        }
    }

    #[test]
    fn test_build_chat_request_sends_higgs_drop_session_id_when_enabled() {
        let provider =
            OpenAICompatProvider::new("local", Some("http://127.0.0.1:9000/v1"), Some("bonsai"))
                .with_higgs_session_cache(true);
        let messages = vec![serde_json::json!({
            "role": "system",
            "content": "stable prefix",
            NANOBOT_HIGGS_SESSION_ID_FIELD: 42_u64,
            NANOBOT_HIGGS_DROP_SESSION_ID_FIELD: 41_u64,
        })];

        let (_, body) = provider.build_chat_request(
            &messages,
            None,
            Some("bonsai"),
            256,
            0.0,
            None,
            None,
            RequestKind::Blocking {
                tool_choice: ToolChoice::Auto,
            },
        );

        assert_eq!(body["session_id"], serde_json::json!(42));
        assert_eq!(body["drop_session_id"], serde_json::json!(41));
        assert!(body["messages"][0]
            .get(NANOBOT_HIGGS_SESSION_ID_FIELD)
            .is_none());
        assert!(body["messages"][0]
            .get(NANOBOT_HIGGS_DROP_SESSION_ID_FIELD)
            .is_none());
    }

    #[test]
    fn test_build_chat_request_sends_higgs_drop_session_ids_when_enabled() {
        let provider =
            OpenAICompatProvider::new("local", Some("http://127.0.0.1:9000/v1"), Some("bonsai"))
                .with_higgs_session_cache(true);
        let messages = vec![serde_json::json!({
            "role": "system",
            "content": "stable prefix",
            NANOBOT_HIGGS_SESSION_ID_FIELD: 42_u64,
            NANOBOT_HIGGS_DROP_SESSION_IDS_FIELD: [40_u64, 41_u64],
        })];

        let (_, body) = provider.build_chat_request(
            &messages,
            None,
            Some("bonsai"),
            256,
            0.0,
            None,
            None,
            RequestKind::Blocking {
                tool_choice: ToolChoice::Auto,
            },
        );

        assert_eq!(body["session_id"], serde_json::json!(42));
        assert_eq!(body["drop_session_ids"], serde_json::json!([40, 41]));
        assert!(body.get("drop_session_id").is_none());
        assert!(body["messages"][0]
            .get(NANOBOT_HIGGS_DROP_SESSION_IDS_FIELD)
            .is_none());
    }

    #[test]
    fn test_build_chat_request_sends_exact_higgs_lease_control_when_enabled() {
        let provider =
            OpenAICompatProvider::new("local", Some("http://127.0.0.1:9000/v1"), Some("bonsai"))
                .with_higgs_session_cache(true);
        let messages = vec![serde_json::json!({
            "role": "system",
            "content": "stable prefix",
            NANOBOT_HIGGS_SESSION_ID_FIELD: 42_u64,
            "_nanobot_higgs_session_lease": {"session_id": 41_u64, "ttl_seconds": 300_u32},
            "_nanobot_higgs_session_cache_policy": "best_effort",
            "_nanobot_higgs_max_prompt_tokens": 31_744_u32,
        })];

        let (_, body) = provider.build_chat_request(
            &messages,
            None,
            Some("bonsai"),
            1_024,
            0.0,
            None,
            None,
            RequestKind::Blocking {
                tool_choice: ToolChoice::Auto,
            },
        );

        assert_eq!(body["session_id"], serde_json::json!(42));
        assert_eq!(
            body["session_lease"],
            serde_json::json!({"session_id": 41_u64, "ttl_seconds": 300_u32})
        );
        assert_eq!(
            body["session_cache_policy"],
            serde_json::json!("best_effort")
        );
        assert_eq!(body["max_prompt_tokens"], serde_json::json!(31_744_u32));
        assert!(body["messages"][0]
            .get("_nanobot_higgs_session_lease")
            .is_none());
        assert!(body["messages"][0]
            .get("_nanobot_higgs_session_cache_policy")
            .is_none());
        assert!(body["messages"][0]
            .get("_nanobot_higgs_max_prompt_tokens")
            .is_none());
    }

    #[test]
    fn test_build_chat_request_strips_higgs_lease_control_when_disabled() {
        let provider =
            OpenAICompatProvider::new("local", Some("http://127.0.0.1:9000/v1"), Some("bonsai"));
        let messages = vec![serde_json::json!({
            "role": "system",
            "content": "stable prefix",
            NANOBOT_HIGGS_SESSION_ID_FIELD: 42_u64,
            "_nanobot_higgs_session_lease": {"session_id": 41_u64, "ttl_seconds": 300_u32},
            "_nanobot_higgs_session_cache_policy": "best_effort",
            "_nanobot_higgs_max_prompt_tokens": 31_744_u32,
        })];

        let (_, body) = provider.build_chat_request(
            &messages,
            None,
            Some("bonsai"),
            1_024,
            0.0,
            None,
            None,
            RequestKind::Blocking {
                tool_choice: ToolChoice::Auto,
            },
        );

        assert!(body.get("session_id").is_none());
        assert!(body.get("session_lease").is_none());
        assert!(body.get("session_cache_policy").is_none());
        assert!(body.get("max_prompt_tokens").is_none());
        assert!(body["messages"][0]
            .get("_nanobot_higgs_session_lease")
            .is_none());
        assert!(body["messages"][0]
            .get("_nanobot_higgs_session_cache_policy")
            .is_none());
        assert!(body["messages"][0]
            .get("_nanobot_higgs_max_prompt_tokens")
            .is_none());
    }

    #[test]
    fn malformed_higgs_lease_ack_is_not_treated_as_numeric_confirmation() {
        let usage_obj = serde_json::json!({
            "prompt_tokens": 12,
            "higgs_session_lease_active": "1",
        });
        let mut usage = std::collections::HashMap::new();

        extract_usage_numbers(usage_obj.as_object().unwrap(), &mut usage);

        assert_eq!(usage.get("prompt_tokens"), Some(&12));
        assert_eq!(usage.get("higgs_session_lease_active"), None);
    }

    #[test]
    fn cloud_provider_never_emits_higgs_lease_control() {
        let provider =
            OpenAICompatProvider::new("openai", Some("https://api.openai.com/v1"), Some("gpt-5"));
        let messages = vec![serde_json::json!({
            "role": "system",
            "content": "stable prefix",
            NANOBOT_HIGGS_SESSION_ID_FIELD: 42_u64,
            NANOBOT_HIGGS_SESSION_LEASE_FIELD: {"session_id": 41_u64, "ttl_seconds": 300_u32},
            NANOBOT_HIGGS_SESSION_CACHE_POLICY_FIELD: "best_effort",
            NANOBOT_HIGGS_MAX_PROMPT_TOKENS_FIELD: 31_744_u32,
        })];

        let (_, body) = provider.build_chat_request(
            &messages,
            None,
            Some("gpt-5"),
            1_024,
            0.0,
            None,
            None,
            RequestKind::Blocking {
                tool_choice: ToolChoice::Auto,
            },
        );

        for field in [
            "session_id",
            "session_lease",
            "session_cache_policy",
            "max_prompt_tokens",
        ] {
            assert!(body.get(field).is_none(), "cloud request emitted {field}");
        }
    }

    /// The retained-session protocol must be engaged by the provider's advertised
    /// capability (`higgs_session_cache`, set when `localBackend=higgs`), NOT by
    /// sniffing port 8091. A higgs-nightly server on any other port must still
    /// advertise support so the agent loop rotates its session id on
    /// compaction/trim instead of re-prefilling under a stale id.
    #[test]
    fn test_higgs_session_cache_capability_independent_of_port() {
        // Non-8091 port: capability still follows the flag, not the port.
        let on = OpenAICompatProvider::new("local", Some("http://127.0.0.1:8092/v1"), Some("m"))
            .with_higgs_session_cache(true);
        let off = OpenAICompatProvider::new("local", Some("http://127.0.0.1:8092/v1"), Some("m"));
        assert!(
            on.supports_higgs_session_cache(),
            "higgs backend on a non-8091 port must still advertise capability"
        );
        assert!(
            !off.supports_higgs_session_cache(),
            "non-higgs backend must not advertise capability"
        );
    }

    #[test]
    fn test_build_chat_request_strips_higgs_markers_when_disabled() {
        let provider =
            OpenAICompatProvider::new("local", Some("http://127.0.0.1:9000/v1"), Some("bonsai"));
        let messages = vec![serde_json::json!({
            "role": "system",
            "content": "stable prefix",
            NANOBOT_HIGGS_SESSION_ID_FIELD: 42_u64,
            NANOBOT_HIGGS_DROP_SESSION_IDS_FIELD: [40_u64, 41_u64],
        })];

        let (_, body) = provider.build_chat_request(
            &messages,
            None,
            Some("bonsai"),
            256,
            0.0,
            None,
            None,
            RequestKind::Blocking {
                tool_choice: ToolChoice::Auto,
            },
        );

        assert!(body.get("session_id").is_none());
        assert!(body.get("drop_session_id").is_none());
        assert!(body.get("drop_session_ids").is_none());
        assert!(body["messages"][0]
            .get(NANOBOT_HIGGS_SESSION_ID_FIELD)
            .is_none());
        assert!(body["messages"][0]
            .get(NANOBOT_HIGGS_DROP_SESSION_IDS_FIELD)
            .is_none());
    }

    // R1 — Apple FM rejects object schemas missing `required` (HTTP 400, verified
    // live). ensure_required_keys must add `required:[]` recursively while
    // preserving existing required arrays.
    #[test]
    fn test_ensure_required_keys_adds_missing_required() {
        // no-arg tool (the check_inbox case)
        let mut s = serde_json::json!({"type":"object","properties":{}});
        ensure_required_keys(&mut s);
        assert_eq!(
            s["required"],
            serde_json::json!([]),
            "no-arg object must gain required:[]"
        );

        // nested object property must also be normalized; existing required kept
        let mut s2 = serde_json::json!({
            "type":"object",
            "properties":{"inner":{"type":"object","properties":{"x":{"type":"string"}}}},
            "required":["inner"]
        });
        ensure_required_keys(&mut s2);
        assert_eq!(
            s2["required"],
            serde_json::json!(["inner"]),
            "existing required preserved"
        );
        assert_eq!(
            s2["properties"]["inner"]["required"],
            serde_json::json!([]),
            "nested object must gain required:[]"
        );
    }

    // normalize_tool_schemas walks body.tools[].function.parameters.
    #[test]
    fn test_normalize_tool_schemas_walks_body() {
        let mut body = serde_json::json!({
            "tools":[{"type":"function","function":{"name":"check_inbox","parameters":{"type":"object","properties":{}}}}]
        });
        normalize_tool_schemas(&mut body);
        assert_eq!(
            body["tools"][0]["function"]["parameters"]["required"],
            serde_json::json!([]),
            "tool parameters must be normalized in place"
        );
    }

    #[test]
    fn test_normalize_tool_schemas_filters_apple_fm_nested_objects() {
        let mut body = serde_json::json!({
            "model": "system",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "lines": {"type": "array", "items": {"type": "integer"}}
                            },
                            "required": ["path"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "batch",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "operations": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "tool": {"type": "string"},
                                            "args": {"type": "object"}
                                        },
                                        "required": ["tool", "args"]
                                    }
                                }
                            },
                            "required": ["operations"]
                        }
                    }
                }
            ],
            "tool_choice": "auto",
            "parallel_tool_calls": false
        });

        normalize_tool_schemas(&mut body);

        let tools = body["tools"].as_array().expect("tools kept");
        assert_eq!(tools.len(), 1, "Apple FM should keep only flat schemas");
        assert_eq!(tools[0]["function"]["name"], serde_json::json!("read_file"));
        assert_eq!(body["tool_choice"], serde_json::json!("auto"));
    }

    #[test]
    fn test_normalize_tool_schemas_removes_empty_apple_fm_tool_set() {
        let mut body = serde_json::json!({
            "model": "pcc",
            "tools": [{
                "type": "function",
                "function": {
                    "name": "nested",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "args": {
                                "type": "object",
                                "properties": {"path": {"type": "string"}},
                                "required": ["path"]
                            }
                        },
                        "required": ["args"]
                    }
                }
            }],
            "tool_choice": "auto",
            "parallel_tool_calls": false
        });

        normalize_tool_schemas(&mut body);

        assert!(body.get("tools").is_none());
        assert!(body.get("tool_choice").is_none());
        assert!(body.get("parallel_tool_calls").is_none());
    }

    #[test]
    fn test_normalize_tool_schemas_keeps_nested_tools_for_non_apple_models() {
        let mut body = serde_json::json!({
            "model": "qwen36-35b",
            "tools": [{
                "type": "function",
                "function": {
                    "name": "batch",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "operations": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {"tool": {"type": "string"}},
                                    "required": ["tool"]
                                }
                            }
                        },
                        "required": ["operations"]
                    }
                }
            }]
        });

        normalize_tool_schemas(&mut body);

        let tools = body["tools"].as_array().expect("tools kept");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["function"]["name"], serde_json::json!("batch"));
    }

    // e2e (live, gated) — proves R1 end-to-end through the real provider→Apple FM
    // send path: a no-arg tool whose schema omits `required` (free-form object)
    // previously 400'd ("Invalid tool definition"); after normalize_tool_schemas
    // adds `required:[]`, Apple FM accepts it.
    //   Run: cargo test --lib \
    //     providers::openai_compat::tests::test_e2e_applefm_noarg_tool_accepted \
    //     -- --ignored --nocapture
    #[tokio::test]
    #[ignore]
    async fn test_e2e_applefm_noarg_tool_accepted() {
        let base = std::env::var("NANOBOT_APPLEFM_BASE")
            .unwrap_or_else(|_| "http://127.0.0.1:1976/v1".to_string());
        let model = std::env::var("NANOBOT_APPLEFM_MODEL").unwrap_or_else(|_| "system".to_string());
        let provider = OpenAICompatProvider::new("local", Some(&base), Some(&model));
        let messages = vec![serde_json::json!({"role": "user", "content": "hi"})];
        // Hostile free-form object schema (no `properties` content, no `required`).
        let tools = vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "check_inbox",
                "description": "Check the inbox.",
                "parameters": {"type": "object", "properties": {}}
            }
        })];
        let res = provider
            .chat(&messages, Some(&tools), Some(&model), 16, 0.0, None, None)
            .await;
        assert!(
            res.is_ok(),
            "no-arg tool must be accepted by Apple FM after schema normalization; got: {:?}",
            res.err()
        );
    }

    #[tokio::test]
    #[ignore]
    async fn test_e2e_applefm_filters_nested_tool_schema() {
        let base = std::env::var("NANOBOT_APPLEFM_BASE")
            .unwrap_or_else(|_| "http://127.0.0.1:1976/v1".to_string());
        let model = std::env::var("NANOBOT_APPLEFM_MODEL").unwrap_or_else(|_| "system".to_string());
        let provider = OpenAICompatProvider::new("local", Some(&base), Some(&model));
        let messages = vec![serde_json::json!({"role": "user", "content": "Reply with only: ok"})];
        let tools = vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "batch",
                "description": "Run several tool operations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "tool": {"type": "string"},
                                    "args": {"type": "object"}
                                },
                                "required": ["tool", "args"]
                            }
                        }
                    },
                    "required": ["operations"]
                }
            }
        })];

        let res = provider
            .chat(&messages, Some(&tools), Some(&model), 16, 0.0, None, None)
            .await;
        assert!(
            res.is_ok(),
            "nested tool schema must be filtered before Apple FM request; got: {:?}",
            res.err()
        );
    }

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
        assert_eq!(resp.finish_reason, FinishReason::ToolCalls);
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
    fn test_parse_response_drops_malformed_tool_call_names() {
        let data = serde_json::json!({
            "choices": [{
                "message": {
                    "content": "Let me read that.",
                    "tool_calls": [{
                        "id": "call_bad",
                        "type": "function",
                        "function": {
                            "name": "read_file\n<parameter=path",
                            "arguments": "{}"
                        }
                    }, {
                        "id": "call_good",
                        "type": "function",
                        "function": {
                            "name": "exec",
                            "arguments": "{\"command\":\"pwd\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        });

        let resp = parse_response(&data).expect("parse should succeed");
        assert_eq!(resp.tool_calls.len(), 1);
        assert_eq!(resp.tool_calls[0].id, "call_good");
        assert_eq!(resp.tool_calls[0].name, "exec");
    }

    #[test]
    fn test_parse_response_flattens_openai_cached_prompt_tokens() {
        let data = serde_json::json!({
            "choices": [{
                "message": {"content": "cached"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 4,
                "prompt_tokens_details": {"cached_tokens": 96}
            }
        });

        let response = parse_response(&data).expect("parse should succeed");
        assert_eq!(response.usage.get("cache_read_input_tokens"), Some(&96));
        assert_eq!(
            response.usage.get("cache_creation_input_tokens"),
            Some(&24),
            "fresh cache tokens are the uncached portion of the prompt"
        );
    }

    #[test]
    fn test_parse_response_cache_creation_saturates_when_cached_exceeds_prompt() {
        let data = serde_json::json!({
            "choices": [{
                "message": {"content": "cached"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 80,
                "prompt_tokens_details": {"cached_tokens": 96}
            }
        });

        let response = parse_response(&data).expect("parse should succeed");
        assert_eq!(response.usage.get("cache_creation_input_tokens"), Some(&0));
    }

    #[test]
    fn test_parse_response_computes_creation_from_flat_cache_read_tokens() {
        let data = serde_json::json!({
            "choices": [{
                "message": {"content": "cached"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 120,
                "cache_read_input_tokens": 90
            }
        });

        let response = parse_response(&data).expect("parse should succeed");
        assert_eq!(response.usage.get("cache_creation_input_tokens"), Some(&30));
    }

    #[test]
    fn test_parse_response_cold_prompt_defaults_cache_read_to_zero() {
        let data = serde_json::json!({
            "choices": [{
                "message": {"content": "cold"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 4
            }
        });

        let response = parse_response(&data).expect("parse should succeed");
        assert_eq!(response.usage.get("cache_read_input_tokens"), Some(&0));
        assert_eq!(
            response.usage.get("cache_creation_input_tokens"),
            Some(&120),
            "a prompt with no cache read is entirely newly created cache input"
        );
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
        assert_eq!(resp.finish_reason, FinishReason::Stop);
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
        assert_eq!(resp.finish_reason, FinishReason::ToolCalls);
        // No usage block -> empty map.
        assert!(resp.usage.is_empty());
    }

    #[test]
    fn test_parse_response_empty_choices() {
        let data = serde_json::json!({
            "choices": []
        });

        let err = parse_response(&data).unwrap_err();
        let provider_err = err.downcast_ref::<crate::errors::ProviderError>();
        assert!(provider_err.is_some(), "Should be a ProviderError");
        assert!(err.to_string().contains("No choices"));
    }

    #[test]
    fn test_parse_response_missing_choices_key() {
        // Completely missing "choices" key (e.g. malformed JSON from the API).
        let data = serde_json::json!({
            "error": "something went wrong"
        });

        let err = parse_response(&data).unwrap_err();
        assert!(err.to_string().contains("No choices"));
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
    fn test_parse_response_double_encoded_arguments() {
        // Local models sometimes double-encode: `arguments` is a JSON string
        // that decodes to another JSON *string* rather than to an object.
        // Build that payload the same way the wire does, so the test cannot
        // silently degrade into the ordinary single-encoded case.
        let double_encoded = serde_json::to_string(r#"{"query": "news"}"#).unwrap();
        assert!(
            serde_json::from_str::<HashMap<String, serde_json::Value>>(&double_encoded).is_err(),
            "payload must not be plain single-encoded args"
        );
        let data = serde_json::json!({
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call_news",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": double_encoded
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        });

        let resp = parse_response(&data).expect("parse should succeed");
        assert_eq!(resp.tool_calls.len(), 1);
        let tc = &resp.tool_calls[0];
        assert_eq!(tc.name, "search");
        // Verify the parameter was correctly unwrapped.
        assert_eq!(
            tc.arguments.get("query").and_then(|v| v.as_str()),
            Some("news")
        );
        // Verify "raw" key is NOT present (not a fallback case).
        assert!(tc.arguments.get("raw").is_none());
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
    fn test_parse_response_reasoning_only_falls_back_to_reasoning() {
        // If model returns ONLY reasoning_content with no main content
        // (e.g. NanBeige), reasoning_content is used as fallback content.
        let data = serde_json::json!({
            "choices": [{
                "message": {
                    "reasoning_content": "Hmm, let me think...",
                },
                "finish_reason": "stop"
            }]
        });

        let resp = parse_response(&data).expect("parse should succeed");
        assert_eq!(
            resp.content.as_deref(),
            Some("Hmm, let me think..."),
            "reasoning-only response should use reasoning_content as fallback"
        );
    }

    #[test]
    fn test_parse_response_empty_content_string_with_reasoning_fallback() {
        // NanBeige returns content: "" (empty string) with reasoning_content.
        let data = serde_json::json!({
            "choices": [{
                "message": {
                    "reasoning_content": "Step 1: analyze the question. Step 2: the answer is 7.",
                    "content": ""
                },
                "finish_reason": "length"
            }]
        });

        let resp = parse_response(&data).expect("parse should succeed");
        assert_eq!(
            resp.content.as_deref(),
            Some("Step 1: analyze the question. Step 2: the answer is 7."),
            "empty content string should trigger reasoning_content fallback"
        );
    }

    #[test]
    fn test_parse_response_strips_inline_think_tags() {
        let data = serde_json::json!({
            "choices": [{
                "message": {
                    "content": "Answer: <think>chain of thought</think>42"
                },
                "finish_reason": "stop"
            }]
        });

        let resp = parse_response(&data).expect("parse should succeed");
        assert_eq!(resp.content.as_deref(), Some("Answer: 42"));
    }

    #[test]
    fn test_parse_response_think_only_content_falls_back_to_reasoning() {
        // oMLX ignores enable_thinking:false — model puts everything in <think>.
        // The inline reasoning should be used as fallback content.
        let data = serde_json::json!({
            "choices": [{
                "message": {
                    "content": "<think>The user wants to know about the codebase structure. Let me analyze it and provide a summary of key components.</think>"
                },
                "finish_reason": "stop"
            }]
        });

        let resp = parse_response(&data).expect("parse should succeed");
        assert_eq!(
            resp.content.as_deref(),
            Some("The user wants to know about the codebase structure. Let me analyze it and provide a summary of key components."),
            "think-only content should use inline reasoning as fallback"
        );
    }

    #[test]
    fn test_parse_response_think_with_visible_content_strips_think() {
        // When there IS visible content after stripping, only visible should remain.
        let data = serde_json::json!({
            "choices": [{
                "message": {
                    "content": "<think>reasoning</think>The answer is 42."
                },
                "finish_reason": "stop"
            }]
        });

        let resp = parse_response(&data).expect("parse should succeed");
        assert_eq!(resp.content.as_deref(), Some("The answer is 42."));
    }

    #[test]
    fn test_split_thinking_from_content_delta_handles_split_tags() {
        let mut state = ThinkSplitState::default();

        let (v1, r1) = split_thinking_from_content_delta(&mut state, "Hello <thi");
        assert_eq!(v1, "Hello ");
        assert!(r1.is_empty());

        let (v2, r2) = split_thinking_from_content_delta(&mut state, "nk>secret</th");
        assert!(v2.is_empty());
        assert_eq!(r2, "secret");

        let (v3, r3) = split_thinking_from_content_delta(&mut state, "ink> world");
        assert_eq!(v3, " world");
        assert!(r3.is_empty());
    }

    /// When in_think_block starts true but content has NO close tag,
    /// ALL content is classified as reasoning and NOTHING is visible.
    /// This was the root cause of the streaming display bug: mlx.rs
    /// set starts_in_think=true while the server had enable_thinking=false,
    /// so the entire response vanished into ThinkingDelta.
    #[test]
    fn test_split_thinking_pre_assumed_think_eats_all_content() {
        let mut state = ThinkSplitState {
            in_think_block: true,
            ..Default::default()
        };

        // Feed normal content — no <think> tags at all.
        let (v1, r1) = split_thinking_from_content_delta(&mut state, "Hello world");
        assert!(v1.is_empty(), "visible should be empty but got: {v1:?}");
        assert_eq!(r1, "Hello world", "all content goes to reasoning");

        let (v2, r2) = split_thinking_from_content_delta(&mut state, ", this is a test.");
        assert!(v2.is_empty());
        assert_eq!(r2, ", this is a test.");

        // Flush also goes to reasoning.
        let (v3, _r3) = flush_thinking_split_state(&mut state);
        assert!(v3.is_empty());
        // Carry may have trailing partial-tag buffer; reasoning gets the rest.
        // The key point: nothing ever becomes visible.
    }

    /// With in_think_block=false (the fix), normal content without tags
    /// is correctly classified as visible.
    #[test]
    fn test_split_thinking_default_state_passes_content_through() {
        let mut state = ThinkSplitState::default(); // in_think_block: false

        let (v1, r1) = split_thinking_from_content_delta(&mut state, "Hello world");
        assert_eq!(v1, "Hello world");
        assert!(r1.is_empty());

        let (v2, r2) = split_thinking_from_content_delta(&mut state, ", more text.");
        assert_eq!(v2, ", more text.");
        assert!(r2.is_empty());
    }

    /// With in_think_block=false, self-generated <think> tags in content
    /// are still correctly detected and split.
    #[test]
    fn test_split_thinking_detects_inline_tags_from_default_state() {
        let mut state = ThinkSplitState::default();

        let (v1, r1) = split_thinking_from_content_delta(
            &mut state,
            "Before <think>reasoning here</think> after",
        );
        assert_eq!(v1, "Before  after");
        assert_eq!(r1, "reasoning here");
    }

    // ── Provider creation / detection tests ───────────────────────

    #[test]
    fn test_new_openrouter_by_key_prefix() {
        let provider = OpenAICompatProvider::new("sk-or-my-key", None, None);
        assert_eq!(provider.api_base, OPENROUTER_API_BASE);
        assert_eq!(provider.default_model, "anthropic/claude-opus-4-5");
    }

    #[test]
    fn test_new_openrouter_by_api_base() {
        let provider = OpenAICompatProvider::new(
            "some-key",
            Some(OPENROUTER_API_BASE),
            Some("meta-llama/llama-3-70b"),
        );
        assert_eq!(provider.api_base, OPENROUTER_API_BASE);
        assert_eq!(provider.default_model, "meta-llama/llama-3-70b");
    }

    #[test]
    fn test_new_deepseek_detection() {
        let provider = OpenAICompatProvider::new("sk-something", None, Some("deepseek-chat"));
        assert_eq!(provider.api_base, DEEPSEEK_API_BASE);
        assert_eq!(provider.default_model, "deepseek-chat");
    }

    #[test]
    fn test_new_groq_detection() {
        let provider = OpenAICompatProvider::new("gsk_something", None, Some("groq/llama3"));
        assert_eq!(provider.api_base, GROQ_API_BASE);
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
        assert_eq!(provider.api_base, OPENROUTER_API_BASE);
    }

    #[test]
    fn test_new_anthropic_key_detection() {
        let provider =
            OpenAICompatProvider::new("sk-ant-abc123", None, Some("claude-sonnet-4-5-20250929"));
        assert_eq!(provider.api_base, ANTHROPIC_API_BASE);
    }

    #[test]
    fn test_new_openai_key_with_bare_model() {
        // sk- prefix with a non-routed model -> OpenAI direct.
        let provider = OpenAICompatProvider::new("sk-abc123", None, Some("gpt-4o"));
        assert_eq!(provider.api_base, OPENAI_API_BASE);
    }

    #[test]
    fn test_new_sk_key_with_routed_model_is_openrouter() {
        // sk- prefix but model has "/" -> OpenRouter.
        let provider =
            OpenAICompatProvider::new("sk-abc123", None, Some("anthropic/claude-opus-4-5"));
        assert_eq!(provider.api_base, OPENROUTER_API_BASE);
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
        assert_eq!(normalize_model_name("claude-opus-4-6"), "claude-opus-4-6");
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
        let provider =
            OpenAICompatProvider::new("sk-or-abc", None, Some("anthropic/claude-opus-4-6"));
        assert!(provider.supports_cache_control("anthropic/claude-opus-4-6"));
    }

    #[test]
    fn test_no_cache_control_openrouter_non_claude() {
        let provider = OpenAICompatProvider::new("sk-or-abc", None, Some("meta-llama/llama-3-70b"));
        assert!(!provider.supports_cache_control("meta-llama/llama-3-70b"));
    }

    #[test]
    fn test_no_cache_control_local() {
        let provider =
            OpenAICompatProvider::new("none", Some("http://localhost:8080/v1"), Some("local"));
        assert!(!provider.supports_cache_control("local"));
    }

    #[test]
    fn test_fold_developer_role_for_local() {
        let msgs = vec![
            serde_json::json!({"role": "system", "content": "sys"}),
            serde_json::json!({"role": "developer", "content": "protocol"}),
            serde_json::json!({"role": "user", "content": "hi"}),
        ];
        // Local endpoint: developer folded into system, role never forwarded.
        let local = fold_developer_role_for_local(msgs.clone(), "http://127.0.0.1:8000/v1");
        assert_eq!(local.len(), 2);
        assert_eq!(local[0]["role"], "system");
        let sys = local[0]["content"].as_str().unwrap();
        assert!(sys.contains("sys") && sys.contains("protocol"));
        assert!(local.iter().all(|m| m["role"] != "developer"));

        // Remote endpoint: untouched (OpenAI understands developer).
        let remote = fold_developer_role_for_local(msgs, "https://api.openai.com/v1");
        assert_eq!(remote.len(), 3);
        assert_eq!(remote[1]["role"], "developer");

        // Local, no system message: a system message is synthesized at front.
        let no_sys = vec![
            serde_json::json!({"role": "developer", "content": "protocol"}),
            serde_json::json!({"role": "user", "content": "hi"}),
        ];
        let folded = fold_developer_role_for_local(no_sys, "http://localhost:1234/v1");
        assert_eq!(folded[0]["role"], "system");
        assert_eq!(folded[0]["content"], "protocol");
        assert_eq!(folded[1]["role"], "user");
    }

    #[test]
    fn test_apply_local_reasoning_controls_local_and_remote() {
        let mut local_body = serde_json::json!({"model": "qwen3-1.7b"});
        apply_local_reasoning_controls(
            &mut local_body,
            "http://localhost:18080/v1",
            "qwen3-1.7b",
            Some(4096),
        );
        assert_eq!(local_body["chat_template_kwargs"]["enable_thinking"], true);
        assert_eq!(local_body["reasoning_budget"], 4096);
        assert_eq!(local_body["reasoning_format"], "deepseek");

        let mut remote_body = serde_json::json!({"model": "gpt-4o"});
        apply_local_reasoning_controls(
            &mut remote_body,
            "https://api.openai.com/v1",
            "gpt-4o",
            Some(4096),
        );
        assert!(remote_body.get("chat_template_kwargs").is_none());
        assert!(remote_body.get("reasoning_budget").is_none());
        assert!(remote_body.get("reasoning_format").is_none());
    }

    // --- reasoning-controls tests ---

    #[test]
    fn test_thinking_disabled_by_default_regardless_of_model_name() {
        // Name-based detection misses fine-tunes (e.g. "qwythos-9b" is a
        // Qwen3.5 template that thinks by default). enable_thinking:false is
        // sent unconditionally; templates without the flag ignore it.
        for model in ["nanbeige-16b", "qwythos-9b"] {
            let mut body = serde_json::json!({"model": model, "messages": []});
            apply_local_reasoning_controls(&mut body, "http://localhost:1234", model, None);
            assert!(body.get("reasoning_budget").is_none());
            assert!(body.get("reasoning_format").is_none());
            assert_eq!(body["chat_template_kwargs"]["enable_thinking"], false);
        }
    }

    #[test]
    fn test_reasoning_params_sent_for_thinking_model() {
        let mut body =
            serde_json::json!({"model": "qwen3-1.7b", "messages": [], "temperature": 0.2});
        apply_local_reasoning_controls(
            &mut body,
            "http://localhost:1234",
            "qwen3-1.7b",
            Some(1024),
        );
        assert_eq!(body["reasoning_budget"], 1024);
        assert_eq!(body["reasoning_format"], "deepseek");
        assert_eq!(body["chat_template_kwargs"]["enable_thinking"], true);
    }

    #[test]
    fn test_reasoning_disabled_for_thinking_model() {
        let mut body = serde_json::json!({"model": "qwen3-1.7b", "messages": []});
        apply_local_reasoning_controls(&mut body, "http://localhost:1234", "qwen3-1.7b", None);
        assert!(
            body.get("reasoning_budget").is_none(),
            "reasoning_budget should not be sent"
        );
        assert!(
            body.get("reasoning_format").is_none(),
            "reasoning_format should not be sent"
        );
    }

    #[test]
    fn test_vibethinker_keeps_hidden_reasoning_enabled_by_default() {
        let mut body = serde_json::json!({
            "model": "VibeThinker-3B-mlx-8Bit",
            "messages": [],
            "temperature": 0.2
        });
        apply_local_reasoning_controls(
            &mut body,
            "http://127.0.0.1:8000/v1",
            "VibeThinker-3B-mlx-8Bit",
            None,
        );
        assert_eq!(body["chat_template_kwargs"]["enable_thinking"], true);
        assert_eq!(body["reasoning_format"], "deepseek");
        assert!(
            body.get("reasoning_budget").is_none(),
            "default-on hidden reasoning must not impose a nanobot budget"
        );
    }

    #[test]
    fn test_build_chat_request_omits_temperature_for_local_servers() {
        // Local servers own per-model sampling (higgs config.toml
        // generation_defaults, LM Studio presets); a client-side temperature
        // would silently override the model-tuned value. Cloud keeps ours.
        let messages = vec![serde_json::json!({"role": "user", "content": "hi"})];

        let local = OpenAICompatProvider::new(
            "local",
            Some("http://127.0.0.1:9000/v1"),
            Some("lfm2.5-2.6b-8bit"),
        );
        let (_, body) = local.build_chat_request(
            &messages,
            None,
            Some("lfm2.5-2.6b-8bit"),
            256,
            0.7,
            None,
            None,
            RequestKind::Streaming,
        );
        assert!(
            body.get("temperature").is_none(),
            "local requests must not carry a client temperature"
        );
        assert_eq!(
            body["chat_template_kwargs"]["enable_thinking"], true,
            "LFM2.5 always thinks; the tracker must stay in sync"
        );

        let remote = OpenAICompatProvider::new("sk-test", Some(OPENAI_API_BASE), Some("gpt-4o"));
        let (_, body) = remote.build_chat_request(
            &messages,
            None,
            Some("gpt-4o"),
            256,
            0.7,
            None,
            None,
            RequestKind::Streaming,
        );
        assert_eq!(body["temperature"], serde_json::json!(0.7));
    }

    #[test]
    fn test_qwen_repetition_controls_use_openai_fields_for_higgs() {
        let mut body = serde_json::json!({"model": "active", "messages": []});
        apply_repetition_controls(
            &mut body,
            "http://localhost:8000",
            "mlx-community/Qwen3.6-35B-A3B-4bit",
        );
        assert_eq!(body["presence_penalty"], 1.5);
        assert_eq!(body["repetition_penalty"], 1.0);
        assert!(body.get("repeat_penalty").is_none());
        assert!(body.get("frequency_penalty").is_none());
    }

    #[test]
    fn test_bonsai_request_controls_for_higgs() {
        let mut body = serde_json::json!({
            "model": "Bonsai-8B-mlx-1bit",
            "messages": [],
            "temperature": 0.3
        });
        apply_local_reasoning_controls(
            &mut body,
            "http://127.0.0.1:8001/v1",
            "Bonsai-8B-mlx-1bit",
            None,
        );
        apply_repetition_controls(&mut body, "http://127.0.0.1:8001/v1", "Bonsai-8B-mlx-1bit");

        assert_eq!(body["chat_template_kwargs"]["enable_thinking"], false);
        assert_eq!(body["repetition_penalty"], 1.1);
        assert!(body.get("repeat_penalty").is_none());
        assert!(body.get("presence_penalty").is_none());
        assert!(body.get("frequency_penalty").is_none());
    }

    #[test]
    fn test_non_qwen_local_repetition_controls_keep_backend_field() {
        let mut body = serde_json::json!({"model": "llama", "messages": []});
        apply_repetition_controls(&mut body, "http://localhost:1234", "llama-3.2-3b");
        assert_eq!(body["repeat_penalty"], 1.1);
        assert!(body.get("presence_penalty").is_none());
    }

    #[test]
    fn test_repetition_controls_absent_for_cloud_api() {
        // Cloud APIs manage their own anti-repetition and would reject
        // llama.cpp-native fields, so they must not be sent.
        let mut body = serde_json::json!({"model": "gpt-4o", "messages": []});
        apply_repetition_controls(&mut body, "https://api.openai.com/v1", "gpt-4o");
        assert!(body.get("repeat_penalty").is_none());
        assert!(body.get("presence_penalty").is_none());
        assert!(body.get("repetition_penalty").is_none());
    }

    #[test]
    fn test_is_local_api_base_private_ips() {
        // RFC 1918 private ranges
        assert!(is_local_api_base("http://192.168.1.22:1234/v1"));
        assert!(is_local_api_base("http://10.0.0.5:8080/v1"));
        assert!(is_local_api_base("http://172.16.0.1:1234/v1"));
        assert!(is_local_api_base("http://172.31.255.1:1234/v1"));
        // Existing localhost checks
        assert!(is_local_api_base("http://localhost:8080/v1"));
        assert!(is_local_api_base("http://127.0.0.1:8080/v1"));
        // Cloud APIs must NOT match
        assert!(!is_local_api_base("https://api.openai.com/v1"));
        assert!(!is_local_api_base("https://openrouter.ai/api/v1"));
        // Edge: 172.15 and 172.32 are NOT private
        assert!(!is_local_api_base("http://172.15.0.1:1234/v1"));
        assert!(!is_local_api_base("http://172.32.0.1:1234/v1"));
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
        let messages = vec![serde_json::json!({"role": "system", "content": "test"})];
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
        let messages = vec![serde_json::json!({"role": "user", "content": "Hello"})];
        let (cached, _) = inject_cache_control(&messages, None);
        // User message should be unchanged (not treated as system).
        assert_eq!(cached[0]["content"], "Hello");
    }

    // ── B6: SLM Observability tests ──────────────────────────────────
    //
    // These tests exercise the failure paths that were previously silent.
    // They verify the data transformation is correct (malformed JSON → {"raw": ...})
    // AND that the code doesn't panic on degenerate SLM outputs.

    #[test]
    fn test_parse_response_malformed_tool_args_multiple_tools() {
        // SLMs often produce a mix: one tool with valid JSON, another with garbage.
        let data = serde_json::json!({
            "choices": [{
                "message": {
                    "tool_calls": [
                        {
                            "id": "call_ok",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": "{\"path\": \"/tmp/test\"}"
                            }
                        },
                        {
                            "id": "call_bad",
                            "type": "function",
                            "function": {
                                "name": "web_fetch",
                                "arguments": "the content is <html>..."
                            }
                        }
                    ]
                },
                "finish_reason": "tool_calls"
            }]
        });

        let resp = parse_response(&data).expect("parse should succeed despite malformed args");
        assert_eq!(resp.tool_calls.len(), 2);

        // First tool: valid JSON → parsed normally.
        assert_eq!(resp.tool_calls[0].name, "read_file");
        assert_eq!(
            resp.tool_calls[0]
                .arguments
                .get("path")
                .and_then(|v| v.as_str()),
            Some("/tmp/test")
        );

        // Second tool: malformed → wrapped in {"raw": ...}.
        assert_eq!(resp.tool_calls[1].name, "web_fetch");
        assert_eq!(
            resp.tool_calls[1]
                .arguments
                .get("raw")
                .and_then(|v| v.as_str()),
            Some("the content is <html>...")
        );
        // Must NOT have the parsed keys from the first tool.
        assert!(resp.tool_calls[1].arguments.get("path").is_none());
    }

    #[test]
    fn test_parse_response_empty_arguments_string() {
        // SLMs sometimes emit arguments: "" (empty string) instead of "{}".
        let data = serde_json::json!({
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call_empty",
                        "type": "function",
                        "function": {
                            "name": "wait",
                            "arguments": ""
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        });

        let resp = parse_response(&data).expect("parse should succeed");
        assert_eq!(resp.tool_calls.len(), 1);
        assert_eq!(resp.tool_calls[0].name, "wait");
        // Empty string is invalid JSON → wrapped in {"raw": ""}.
        assert_eq!(
            resp.tool_calls[0]
                .arguments
                .get("raw")
                .and_then(|v| v.as_str()),
            Some("")
        );
    }

    #[test]
    fn test_parse_response_function_call_tag_as_arguments() {
        // Nemotron-family SLMs wrap tool calls in <start_function_call> tags.
        let data = serde_json::json!({
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call_tag",
                        "type": "function",
                        "function": {
                            "name": "shell",
                            "arguments": "<start_function_call>{\"command\": \"ls\"}<end_function_call>"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        });

        let resp = parse_response(&data).expect("parse should succeed");
        assert_eq!(resp.tool_calls.len(), 1);
        // Tagged output is not valid JSON → wrapped in raw.
        assert!(resp.tool_calls[0].arguments.contains_key("raw"));
    }

    /// Helper: create a fake byte stream from SSE lines.
    fn sse_bytes(lines: &[&str]) -> Vec<Result<bytes::Bytes, reqwest::Error>> {
        lines
            .iter()
            .map(|l| Ok(bytes::Bytes::from(format!("{}\n", l))))
            .collect()
    }

    async fn spawn_timed_sse_server(
        frames: Vec<(std::time::Duration, &'static str)>,
    ) -> (String, tokio::task::JoinHandle<()>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind mock SSE server");
        let address = listener.local_addr().expect("mock SSE server address");
        let task = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accept SSE request");
            let mut request = Vec::new();
            let mut buffer = [0_u8; 1024];
            while !request.windows(4).any(|window| window == b"\r\n\r\n") {
                let read = socket.read(&mut buffer).await.expect("read SSE request");
                if read == 0 {
                    return;
                }
                request.extend_from_slice(&buffer[..read]);
            }

            socket
                .write_all(
                    b"HTTP/1.1 200 OK\r\n\
                      Content-Type: text/event-stream\r\n\
                      Connection: close\r\n\
                      \r\n",
                )
                .await
                .expect("write SSE headers");
            socket.flush().await.expect("flush SSE headers");

            for (delay, frame) in frames {
                tokio::time::sleep(delay).await;
                if socket.write_all(frame.as_bytes()).await.is_err() {
                    return;
                }
                if socket.flush().await.is_err() {
                    return;
                }
            }
        });

        (format!("http://{address}/v1"), task)
    }

    #[tokio::test]
    async fn one_shot_higgs_lease_is_not_retried_after_retryable_http_failure() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;
        use std::time::Duration;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind retry capture server");
        let address = listener.local_addr().expect("retry capture address");
        let request_count = Arc::new(AtomicUsize::new(0));
        let server_count = Arc::clone(&request_count);
        let server = tokio::spawn(async move {
            while let Ok(Ok((mut socket, _))) =
                tokio::time::timeout(Duration::from_secs(5), listener.accept()).await
            {
                let mut request = Vec::new();
                let mut buffer = [0_u8; 2048];
                while !request.windows(4).any(|window| window == b"\r\n\r\n") {
                    let read = socket.read(&mut buffer).await.expect("read retry request");
                    if read == 0 {
                        break;
                    }
                    request.extend_from_slice(&buffer[..read]);
                }
                server_count.fetch_add(1, Ordering::SeqCst);
                socket
                    .write_all(
                        b"HTTP/1.1 503 Service Unavailable\r\n\
                          Content-Length: 11\r\n\
                          Connection: close\r\n\
                          \r\n\
                          unavailable",
                    )
                    .await
                    .expect("write retry response");
            }
        });
        let provider = OpenAICompatProvider::new(
            "local",
            Some(&format!("http://{address}/v1")),
            Some("bonsai"),
        )
        .with_higgs_session_cache(true);
        let messages = vec![serde_json::json!({
            "role": "system",
            "content": "stable prefix",
            NANOBOT_HIGGS_SESSION_ID_FIELD: 42_u64,
            NANOBOT_HIGGS_SESSION_LEASE_FIELD: {
                "session_id": 41_u64,
                "ttl_seconds": 300_u32,
            },
            NANOBOT_HIGGS_SESSION_CACHE_POLICY_FIELD: "best_effort",
            NANOBOT_HIGGS_MAX_PROMPT_TOKENS_FIELD: 31_744_u32,
        })];

        let result = provider
            .chat(&messages, None, None, 16, 0.0, None, None)
            .await;

        assert!(result.is_err());
        assert_eq!(
            request_count.load(Ordering::SeqCst),
            1,
            "a one-shot lease request must be attempted exactly once"
        );
        server.abort();
    }

    #[tokio::test]
    async fn captured_http_requests_preserve_higgs_route_and_strip_other_backends() {
        use crate::config::schema::ProviderConfig;
        use crate::providers::factory::{create_openai_compat, ProviderSpec};
        use std::sync::{Arc, Mutex};
        use std::time::Duration;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind request capture server");
        let address = listener.local_addr().expect("request capture address");
        let captured = Arc::new(Mutex::new(Vec::<serde_json::Value>::new()));
        let server_captured = Arc::clone(&captured);
        let server = tokio::spawn(async move {
            for request_index in 0..4 {
                let (mut socket, _) =
                    tokio::time::timeout(Duration::from_secs(5), listener.accept())
                        .await
                        .expect("capture request timeout")
                        .expect("accept capture request");
                let mut request = Vec::new();
                let mut buffer = [0_u8; 4096];
                let header_end = loop {
                    let read = socket
                        .read(&mut buffer)
                        .await
                        .expect("read capture request");
                    assert_ne!(read, 0, "capture request ended before headers");
                    request.extend_from_slice(&buffer[..read]);
                    if let Some(index) = request.windows(4).position(|window| window == b"\r\n\r\n")
                    {
                        break index + 4;
                    }
                };
                let headers = String::from_utf8_lossy(&request[..header_end]);
                let content_length = headers
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().unwrap())
                    })
                    .expect("capture request content-length");
                while request.len() - header_end < content_length {
                    let read = socket.read(&mut buffer).await.expect("read capture body");
                    assert_ne!(read, 0, "capture request ended before body");
                    request.extend_from_slice(&buffer[..read]);
                }
                let body: serde_json::Value =
                    serde_json::from_slice(&request[header_end..header_end + content_length])
                        .expect("captured JSON request");
                server_captured.lock().unwrap().push(body);

                let (status, response_body) = if request_index == 0 {
                    (
                        "409 Conflict",
                        r#"{"error":{"code":"retained_session_unavailable","message":"retained session 41 unavailable"}}"#,
                    )
                } else {
                    (
                        "200 OK",
                        r#"{"id":"chatcmpl-capture","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":12,"completion_tokens":1,"total_tokens":13}}"#,
                    )
                };
                let response = format!(
                    "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{response_body}",
                    response_body.len()
                );
                socket
                    .write_all(response.as_bytes())
                    .await
                    .expect("write capture response");
            }
        });

        let api_base = format!("http://{address}/v1");
        let tools = vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "list_dir",
                "description": "List a directory",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"]
                }
            }
        })];
        let retained_messages = vec![
            serde_json::json!({
                "role": "system",
                "content": "stable system prompt",
                NANOBOT_HIGGS_SESSION_ID_FIELD: 41_u64,
                NANOBOT_HIGGS_DROP_SESSION_ID_FIELD: 43_u64,
                NANOBOT_HIGGS_DROP_SESSION_IDS_FIELD: [44_u64, 43_u64],
                NANOBOT_HIGGS_SESSION_LEASE_FIELD: {"session_id": 41_u64, "ttl_seconds": 300_u32},
                NANOBOT_HIGGS_SESSION_CACHE_POLICY_FIELD: "require_continuation",
                NANOBOT_HIGGS_MAX_PROMPT_TOKENS_FIELD: 31_744_u32,
            }),
            serde_json::json!({"role": "user", "content": "inspect the workspace"}),
        ];
        let higgs = create_openai_compat(
            ProviderSpec::local(&api_base, Some("model")).with_higgs_session_cache(true),
        );
        assert!(higgs
            .chat(
                &retained_messages,
                Some(&tools),
                None,
                1_024,
                0.0,
                None,
                None
            )
            .await
            .is_err());

        let fallback_messages = vec![
            serde_json::json!({
                "role": "system",
                "content": "stable system prompt",
                NANOBOT_HIGGS_SESSION_ID_FIELD: 42_u64,
                NANOBOT_HIGGS_DROP_SESSION_ID_FIELD: 41_u64,
                NANOBOT_HIGGS_DROP_SESSION_IDS_FIELD: [41_u64, 43_u64],
                NANOBOT_HIGGS_SESSION_CACHE_POLICY_FIELD: "best_effort",
                NANOBOT_HIGGS_MAX_PROMPT_TOKENS_FIELD: 31_744_u32,
            }),
            retained_messages[1].clone(),
        ];
        assert_eq!(
            higgs
                .chat(
                    &fallback_messages,
                    Some(&tools),
                    None,
                    1_024,
                    0.0,
                    None,
                    None
                )
                .await
                .unwrap()
                .content
                .as_deref(),
            Some("ok")
        );

        let non_higgs = create_openai_compat(ProviderSpec::local(&api_base, Some("model")));
        non_higgs
            .chat(
                &retained_messages,
                Some(&tools),
                None,
                1_024,
                0.0,
                None,
                None,
            )
            .await
            .unwrap();
        let cloud_config = ProviderConfig {
            api_key: "sk-cloud-test".to_owned(),
            api_base: Some(api_base.clone()),
        };
        let mut cloud_spec = ProviderSpec::from_config(&cloud_config, None);
        cloud_spec.model = Some("gpt-5".to_owned());
        let cloud = create_openai_compat(cloud_spec);
        cloud
            .chat(
                &retained_messages,
                Some(&tools),
                None,
                1_024,
                0.0,
                None,
                None,
            )
            .await
            .unwrap();
        server.await.unwrap();

        let captured = captured.lock().unwrap();
        assert_eq!(captured.len(), 4);
        let expected_messages = serde_json::json!([
            {"role": "system", "content": "stable system prompt"},
            {"role": "user", "content": "inspect the workspace"}
        ]);
        let expected_base = |model: &str| {
            serde_json::json!({
                "model": model,
                "messages": expected_messages,
                "max_tokens": 1_024,
                "stream": false,
                "chat_template_kwargs": {"enable_thinking": false},
                "repeat_penalty": 1.1,
                "tools": tools,
                "tool_choice": "auto"
            })
        };
        let mut expected_retained = expected_base("model");
        expected_retained["session_id"] = serde_json::json!(41);
        expected_retained["drop_session_ids"] = serde_json::json!([43, 44]);
        assert!(!expected_retained["drop_session_ids"]
            .as_array()
            .unwrap()
            .contains(&serde_json::json!(41)));
        expected_retained["session_lease"] =
            serde_json::json!({"session_id": 41, "ttl_seconds": 300});
        expected_retained["session_cache_policy"] = serde_json::json!("require_continuation");
        expected_retained["max_prompt_tokens"] = serde_json::json!(31_744);
        assert_eq!(captured[0], expected_retained);

        let mut expected_fallback = expected_base("model");
        expected_fallback["session_id"] = serde_json::json!(42);
        expected_fallback["drop_session_ids"] = serde_json::json!([41, 43]);
        expected_fallback["session_cache_policy"] = serde_json::json!("best_effort");
        expected_fallback["max_prompt_tokens"] = serde_json::json!(31_744);
        assert_eq!(captured[1], expected_fallback);

        assert_eq!(captured[2], expected_base("model"));
        assert_eq!(captured[3], expected_base("gpt-5"));
    }

    #[tokio::test]
    async fn test_stream_read_timeout_slides_on_heartbeats_and_bounds_inactivity() {
        use std::time::{Duration, Instant};

        let (heartbeat_base, heartbeat_server) = spawn_timed_sse_server(vec![
            (Duration::from_millis(400), ":\n\n"),
            (Duration::from_millis(400), ":\n\n"),
            (Duration::from_millis(400), ":\n\n"),
            (
                Duration::from_millis(400),
                "data: {\"choices\":[{\"delta\":{\"content\":\"ready\"},\"index\":0}]}\n\n\
                 data: [DONE]\n\n",
            ),
        ])
        .await;
        let heartbeat_provider =
            OpenAICompatProvider::new("local", Some(&heartbeat_base), Some("mock")).with_timeout(1);
        let messages = vec![serde_json::json!({"role": "user", "content": "hello"})];
        let heartbeat_started = Instant::now();
        let mut heartbeat_stream = heartbeat_provider
            .chat_stream(&messages, None, None, 16, 0.0, None, None)
            .await
            .expect("heartbeat stream starts");
        let heartbeat_response = tokio::time::timeout(Duration::from_secs(4), async {
            loop {
                if let Some(StreamChunk::Done(response)) = heartbeat_stream.rx.recv().await {
                    break response;
                }
            }
        })
        .await
        .expect("heartbeat stream completes");
        assert!(
            heartbeat_started.elapsed() > Duration::from_secs(1),
            "heartbeats must allow total wall time to exceed the read timeout"
        );
        assert_eq!(heartbeat_response.finish_reason, FinishReason::Stop);
        assert_eq!(heartbeat_response.content.as_deref(), Some("ready"));
        heartbeat_server.await.expect("heartbeat server task");

        let (idle_base, idle_server) = spawn_timed_sse_server(vec![(
            Duration::from_millis(1_500),
            "data: {\"choices\":[{\"delta\":{\"content\":\"too late\"},\"index\":0}]}\n\n\
             data: [DONE]\n\n",
        )])
        .await;
        let idle_provider =
            OpenAICompatProvider::new("local", Some(&idle_base), Some("mock")).with_timeout(1);
        let mut idle_stream = idle_provider
            .chat_stream(&messages, None, None, 16, 0.0, None, None)
            .await
            .expect("idle stream starts");
        let idle_response = tokio::time::timeout(Duration::from_secs(3), async {
            loop {
                if let Some(StreamChunk::Done(response)) = idle_stream.rx.recv().await {
                    break response;
                }
            }
        })
        .await
        .expect("idle stream must end at the inactivity timeout");
        assert_eq!(idle_response.finish_reason, FinishReason::ProviderFailure);
        idle_server.abort();
    }

    #[tokio::test]
    async fn test_sse_heartbeat_emits_transport_progress() {
        let chunks = sse_bytes(&[
            ":",
            "data: {\"choices\":[{\"delta\":{\"content\":\"ready\"},\"index\":0}]}",
            "data: [DONE]",
        ]);
        let stream = futures_util::stream::iter(chunks);
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();

        parse_sse_stream(stream, tx).await;

        let mut saw_transport_progress = false;
        while let Ok(chunk) = rx.try_recv() {
            if matches!(chunk, StreamChunk::TransportProgress) {
                saw_transport_progress = true;
            }
        }
        assert!(
            saw_transport_progress,
            "an SSE comment heartbeat must keep the local stream watchdog alive"
        );
    }

    #[tokio::test]
    async fn test_sse_stream_no_done_produces_response() {
        // Simulate an SLM that streams content then drops the connection
        // without sending [DONE]. The parse_sse_stream should still
        // produce a Done chunk with whatever was accumulated.
        let chunks = sse_bytes(&[
            "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"},\"index\":0}]}",
            "data: {\"choices\":[{\"delta\":{\"content\":\" world\"},\"index\":0}]}",
            // No "data: [DONE]" — connection dropped.
        ]);

        let stream = futures_util::stream::iter(chunks);
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();

        parse_sse_stream(stream, tx).await;

        // Collect all chunks.
        let mut deltas = Vec::new();
        let mut done_response = None;
        while let Ok(chunk) = rx.try_recv() {
            match chunk {
                StreamChunk::TextDelta(d) => deltas.push(d),
                StreamChunk::Done(resp) => done_response = Some(resp),
                _ => {}
            }
        }

        // Must have received content deltas.
        assert!(!deltas.is_empty(), "should have received text deltas");
        let full: String = deltas.join("");
        assert!(full.contains("Hello"), "content should contain 'Hello'");

        // Must have received a Done with assembled content.
        let resp = done_response.expect("should have received Done chunk despite no [DONE]");
        assert_eq!(resp.content.as_deref(), Some("Hello world"));
        // Abnormal termination without [DONE] must report "length" so the
        // auto-continue mechanism can detect truncation (Bug 2 regression guard).
        assert_eq!(
            resp.finish_reason,
            FinishReason::Length,
            "stream ending without [DONE] must yield finish_reason=length"
        );
    }

    #[tokio::test]
    async fn test_sse_stream_with_done_keeps_finish_reason_stop() {
        // A well-formed stream that sends [DONE] should preserve the
        // finish_reason received in the chunks (not override it with "length").
        let chunks = sse_bytes(&[
            "data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"},\"index\":0}]}",
            "data: {\"choices\":[{\"finish_reason\":\"stop\",\"index\":0}]}",
            "data: [DONE]",
        ]);

        let stream = futures_util::stream::iter(chunks);
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();

        parse_sse_stream(stream, tx).await;

        let mut done_response = None;
        while let Ok(chunk) = rx.try_recv() {
            if let StreamChunk::Done(resp) = chunk {
                done_response = Some(resp);
            }
        }

        let resp = done_response.expect("should have received Done chunk");
        assert_eq!(
            resp.finish_reason,
            FinishReason::Stop,
            "normal stream with [DONE] must keep finish_reason=stop"
        );
    }

    #[tokio::test]
    async fn test_sse_stream_with_done_keeps_finish_reason_length() {
        // A stream that sends finish_reason=length and then [DONE] (model hit
        // token limit gracefully) must preserve "length".
        let chunks = sse_bytes(&[
            "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"},\"index\":0}]}",
            "data: {\"choices\":[{\"finish_reason\":\"length\",\"index\":0}]}",
            "data: [DONE]",
        ]);

        let stream = futures_util::stream::iter(chunks);
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();

        parse_sse_stream(stream, tx).await;

        let mut done_response = None;
        while let Ok(chunk) = rx.try_recv() {
            if let StreamChunk::Done(resp) = chunk {
                done_response = Some(resp);
            }
        }

        let resp = done_response.expect("should have received Done chunk");
        assert_eq!(
            resp.finish_reason,
            FinishReason::Length,
            "token-limit response must keep finish_reason=length"
        );
    }

    #[tokio::test]
    async fn test_sse_stream_malformed_tool_args_in_done() {
        // SLM streams a tool call with malformed arguments, then sends [DONE].
        let chunks = sse_bytes(&[
            "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"},\"index\":0}]}",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"tc1\",\"function\":{\"name\":\"shell\",\"arguments\":\"\"}}]},\"index\":0}]}",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"not json at all\"}}]},\"index\":0}]}",
            "data: {\"choices\":[{\"finish_reason\":\"tool_calls\",\"index\":0}]}",
            "data: [DONE]",
        ]);

        let stream = futures_util::stream::iter(chunks);
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();

        parse_sse_stream(stream, tx).await;

        let mut done_response = None;
        while let Ok(chunk) = rx.try_recv() {
            if let StreamChunk::Done(resp) = chunk {
                done_response = Some(resp);
            }
        }

        let resp = done_response.expect("should have received Done chunk");
        assert_eq!(resp.tool_calls.len(), 1);
        assert_eq!(resp.tool_calls[0].name, "shell");
        // Accumulated argument string "not json at all" is not valid JSON → {"raw": ...}.
        assert!(
            resp.tool_calls[0].arguments.contains_key("raw"),
            "malformed tool args should be wrapped in 'raw' key"
        );
    }

    #[tokio::test]
    async fn test_sse_stream_tool_call_emits_prefill_marker() {
        // A response that goes straight into a tool call (no text/thinking
        // deltas) must emit ToolCallDelta when the function name arrives so
        // the agent loop can record TTFT — otherwise tool-call turns never
        // measure prefill, hiding exactly the slow calls.
        let chunks = sse_bytes(&[
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"tc1\",\"function\":{\"name\":\"exec\",\"arguments\":\"\"}}]},\"index\":0}]}",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{}\"}}]},\"index\":0}]}",
            "data: {\"choices\":[{\"finish_reason\":\"tool_calls\",\"index\":0}]}",
            "data: [DONE]",
        ]);

        let stream = futures_util::stream::iter(chunks);
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        parse_sse_stream(stream, tx).await;

        let mut received = Vec::new();
        while let Ok(chunk) = rx.try_recv() {
            received.push(chunk);
        }
        let marker_pos = received
            .iter()
            .position(|c| matches!(c, StreamChunk::ToolCallDelta))
            .expect("tool-call stream must emit ToolCallDelta so TTFT is recorded");
        let done_pos = received
            .iter()
            .position(|c| matches!(c, StreamChunk::Done(_)))
            .expect("should have received Done");
        assert!(
            marker_pos < done_pos,
            "ToolCallDelta must arrive before Done (it marks end of prefill)"
        );
    }

    #[tokio::test]
    async fn test_sse_stream_no_done_with_tool_calls() {
        // SLM streams a tool call then drops without [DONE].
        let chunks = sse_bytes(&[
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"tc1\",\"function\":{\"name\":\"read_file\",\"arguments\":\"\"}}]},\"index\":0}]}",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"path\\\":\\\"\"}}]},\"index\":0}]}",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"/tmp/test\\\"\"}}]},\"index\":0}]}",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"}\"}}]},\"index\":0}]}",
            // No [DONE]
        ]);

        let stream = futures_util::stream::iter(chunks);
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();

        parse_sse_stream(stream, tx).await;

        let mut done_response = None;
        let mut tool_call_deltas = 0usize;
        while let Ok(chunk) = rx.try_recv() {
            match chunk {
                StreamChunk::ToolCallDelta => tool_call_deltas += 1,
                StreamChunk::Done(resp) => done_response = Some(resp),
                _ => {}
            }
        }

        assert!(
            tool_call_deltas >= 4,
            "name and argument fragments should reset the stream watchdog"
        );
        let resp = done_response.expect("should have received Done despite no [DONE]");
        assert_eq!(resp.tool_calls.len(), 1);
        assert_eq!(resp.tool_calls[0].name, "read_file");
        // The accumulated args should be valid JSON: {"path":"/tmp/test"}
        assert_eq!(
            resp.tool_calls[0]
                .arguments
                .get("path")
                .and_then(|v| v.as_str()),
            Some("/tmp/test")
        );
    }

    #[tokio::test]
    async fn test_sse_stream_tool_call_object_arguments() {
        // Some local servers (oMLX/MLX for qwen3.6 / diffusiongemma / GLM) stream
        // `arguments` as a whole JSON *object* in one chunk instead of string
        // fragments. The streaming parser must accept that, mirroring the
        // non-streaming path, or the params are dropped → empty query (LM Studio #1868).
        let chunks = sse_bytes(&[
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"tc1\",\"function\":{\"name\":\"web_search\",\"arguments\":{\"query\":\"latest space news\"}}}]},\"index\":0}]}",
            "data: [DONE]",
        ]);

        let stream = futures_util::stream::iter(chunks);
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();

        parse_sse_stream(stream, tx).await;

        let mut done_response = None;
        while let Ok(chunk) = rx.try_recv() {
            if let StreamChunk::Done(resp) = chunk {
                done_response = Some(resp);
            }
        }

        let resp = done_response.expect("should have received Done");
        assert_eq!(resp.tool_calls.len(), 1);
        assert_eq!(resp.tool_calls[0].name, "web_search");
        assert_eq!(
            resp.tool_calls[0]
                .arguments
                .get("query")
                .and_then(|v| v.as_str()),
            Some("latest space news"),
            "object-form streamed arguments must survive, not drop to empty"
        );
    }

    #[tokio::test]
    async fn test_sse_stream_completely_empty() {
        // SLM returns zero content — connection established then immediately dropped.
        let chunks: Vec<Result<bytes::Bytes, reqwest::Error>> = vec![];
        let stream = futures_util::stream::iter(chunks);
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();

        parse_sse_stream(stream, tx).await;

        let mut done_response = None;
        while let Ok(chunk) = rx.try_recv() {
            if let StreamChunk::Done(resp) = chunk {
                done_response = Some(resp);
            }
        }

        let resp = done_response.expect("should produce Done even for empty stream");
        assert_eq!(resp.finish_reason, FinishReason::ProviderFailure);
        assert!(
            resp.content
                .as_deref()
                .is_some_and(|content| content.contains("backend produced any response")),
            "empty stream should explain the provider failure"
        );
        assert!(
            resp.tool_calls.is_empty(),
            "empty stream should have no tool calls"
        );
    }

    // --- tool_choice_value mapping / gating ---

    const LOCAL: &str = "http://localhost:8000/v1";
    const CLOUD: &str = "https://api.openai.com/v1";

    #[test]
    fn active_local_transport_keeps_semantic_policy_model() {
        // higgs-nightly serves models under their real id and rejects the legacy
        // "active" alias, so the wire model must be the real id (not "active").
        // `default_model == "active"` (an unresolved sentinel) no longer affects
        // the wire — ctx.core.model's id is sent directly.
        let (request, policy) =
            resolve_request_and_policy_model(LOCAL, "usermma/VibeThinker-3B-mlx-8Bit");

        assert_eq!(request, "usermma/VibeThinker-3B-mlx-8Bit");
        assert_eq!(policy, "usermma/VibeThinker-3B-mlx-8Bit");
    }

    #[test]
    fn non_active_local_transport_uses_same_request_and_policy_model() {
        let (request, policy) = resolve_request_and_policy_model(LOCAL, "qwen3-8b");

        assert_eq!(request, "qwen3-8b");
        assert_eq!(policy, "qwen3-8b");
    }

    #[test]
    fn tool_choice_required_local_constrained_is_required() {
        assert_eq!(
            tool_choice_value(ToolChoice::Required, LOCAL, true),
            serde_json::json!("required")
        );
    }

    #[test]
    fn tool_choice_required_local_escape_hatch_degrades_to_auto() {
        // constrained=false (config escape hatch) → no forcing.
        assert_eq!(
            tool_choice_value(ToolChoice::Required, LOCAL, false),
            serde_json::json!("auto")
        );
    }

    #[test]
    fn tool_choice_required_cloud_stays_auto() {
        // Tier 1 is local-only; cloud behavior is unchanged.
        assert_eq!(
            tool_choice_value(ToolChoice::Required, CLOUD, true),
            serde_json::json!("auto")
        );
    }

    #[test]
    fn tool_choice_auto_and_none_map_directly() {
        assert_eq!(
            tool_choice_value(ToolChoice::Auto, LOCAL, true),
            serde_json::json!("auto")
        );
        assert_eq!(
            tool_choice_value(ToolChoice::None, LOCAL, true),
            serde_json::json!("none")
        );
    }
}
