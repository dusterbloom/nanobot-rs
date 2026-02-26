//! Configuration schema for nanobot.
//!
//! All structs use `#[serde(rename_all = "camelCase")]` so that the JSON config
//! file can use camelCase keys while Rust code uses snake_case fields.

use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Channel configs
// ---------------------------------------------------------------------------

/// WhatsApp channel configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WhatsAppConfig {
    #[serde(default)]
    pub enabled: bool,
    /// Explicit bridge URL. If not set, derived from `bridge_port`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bridge_url: Option<String>,
    #[serde(default = "default_whatsapp_bridge_port")]
    pub bridge_port: u16,
    #[serde(default)]
    pub allow_from: Vec<String>,
}

fn default_whatsapp_bridge_port() -> u16 {
    3001
}

impl WhatsAppConfig {
    /// Get the effective bridge WebSocket URL.
    pub fn effective_bridge_url(&self) -> String {
        self.bridge_url
            .clone()
            .unwrap_or_else(|| format!("ws://localhost:{}", self.bridge_port))
    }
}

impl Default for WhatsAppConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            bridge_url: None,
            bridge_port: default_whatsapp_bridge_port(),
            allow_from: Vec::new(),
        }
    }
}

/// Telegram channel configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TelegramConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub token: String,
    #[serde(default)]
    pub allow_from: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proxy: Option<String>,
}

impl Default for TelegramConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            token: String::new(),
            allow_from: Vec::new(),
            proxy: None,
        }
    }
}

/// Feishu/Lark channel configuration using WebSocket long connection.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FeishuConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub app_id: String,
    #[serde(default)]
    pub app_secret: String,
    #[serde(default)]
    pub encrypt_key: String,
    #[serde(default)]
    pub verification_token: String,
    #[serde(default)]
    pub allow_from: Vec<String>,
}

impl Default for FeishuConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            app_id: String::new(),
            app_secret: String::new(),
            encrypt_key: String::new(),
            verification_token: String::new(),
            allow_from: Vec::new(),
        }
    }
}

/// Email channel configuration (IMAP polling + SMTP sending).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EmailConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub imap_host: String,
    #[serde(default = "default_imap_port")]
    pub imap_port: u16,
    #[serde(default)]
    pub smtp_host: String,
    #[serde(default = "default_smtp_port")]
    pub smtp_port: u16,
    #[serde(default)]
    pub username: String,
    #[serde(default)]
    pub password: String,
    #[serde(default = "default_poll_interval")]
    pub poll_interval_secs: u64,
    #[serde(default)]
    pub allow_from: Vec<String>,
}

fn default_imap_port() -> u16 {
    993
}

fn default_smtp_port() -> u16 {
    587
}

fn default_poll_interval() -> u64 {
    30
}

impl Default for EmailConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            imap_host: String::new(),
            imap_port: default_imap_port(),
            smtp_host: String::new(),
            smtp_port: default_smtp_port(),
            username: String::new(),
            password: String::new(),
            poll_interval_secs: default_poll_interval(),
            allow_from: Vec::new(),
        }
    }
}

/// Configuration for chat channels.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChannelsConfig {
    #[serde(default)]
    pub whatsapp: WhatsAppConfig,
    #[serde(default)]
    pub telegram: TelegramConfig,
    #[serde(default)]
    pub feishu: FeishuConfig,
    #[serde(default)]
    pub email: EmailConfig,
}

// ---------------------------------------------------------------------------
// Agent configs
// ---------------------------------------------------------------------------

/// Default agent configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentDefaults {
    #[serde(default = "default_workspace")]
    pub workspace: String,
    #[serde(default = "default_model")]
    pub model: String,
    /// Preferred local GGUF model filename (e.g. "Qwen3-8B-Q4_K_M.gguf").
    /// Empty = use hardcoded DEFAULT_LOCAL_MODEL fallback.
    #[serde(default)]
    pub local_model: String,
    /// Custom API base for local inference (e.g. "http://192.168.1.22:1234/v1").
    /// When set, local mode uses this instead of LM Studio on localhost.
    /// All trio roles (main, router, specialist) share this endpoint; model
    /// differentiation happens via the `model` field in each API request (JIT loading).
    #[serde(default)]
    pub local_api_base: String,
    /// Context window size for local models (default: 32768).
    /// Separate from maxContextTokens so cloud (512K) and local (32K) coexist.
    #[serde(default = "default_local_max_context_tokens")]
    pub local_max_context_tokens: usize,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    #[serde(default = "default_max_tool_iterations")]
    pub max_tool_iterations: u32,
    #[serde(default = "default_max_context_tokens")]
    pub max_context_tokens: usize,
    #[serde(default = "default_max_concurrent_chats")]
    pub max_concurrent_chats: usize,
    /// Max characters for inline tool results before truncation (default: 30000).
    #[serde(default = "default_max_tool_result_chars")]
    pub max_tool_result_chars: usize,
    /// LM Studio model identifier for the main model (e.g. "gemma-3n-e4b-it").
    /// When empty, derived from local_model via strip_gguf_suffix.
    #[serde(default)]
    pub lms_main_model: String,
    /// Port for the LM Studio server when managed by lms CLI (default: 1234).
    #[serde(default = "default_lms_port")]
    pub lms_port: u16,
    /// Inference engine preference: "auto" | "lms".
    /// Currently only LM Studio ("lms") is supported.
    #[serde(default = "default_inference_engine")]
    pub inference_engine: String,
    /// Path to a YAML instruction profiles file for model-specific prompt
    /// engineering. When set, profiles are loaded at startup and applied to
    /// every LLM call based on the active model name and task kind.
    /// Example: "~/.nanobot/instructions.yaml"
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions_path: Option<String>,
    /// Runtime flag: skip JitGate when models are pre-loaded by lms.
    /// Not serialized to config.json.
    #[serde(skip)]
    pub skip_jit_gate: bool,
}

fn default_workspace() -> String {
    "~/.nanobot/workspace".to_string()
}

fn default_model() -> String {
    "anthropic/claude-opus-4-5".to_string()
}

fn default_local_max_context_tokens() -> usize {
    32768
}

fn default_max_tokens() -> u32 {
    2048
}

fn default_temperature() -> f64 {
    0.7
}

fn default_max_tool_iterations() -> u32 {
    20
}

fn default_max_context_tokens() -> usize {
    128000
}

fn default_max_concurrent_chats() -> usize {
    4
}

pub const DEFAULT_MAX_TOOL_RESULT_CHARS: usize = 10_000;

fn default_max_tool_result_chars() -> usize {
    DEFAULT_MAX_TOOL_RESULT_CHARS
}

fn default_lms_port() -> u16 {
    1234
}

fn default_inference_engine() -> String {
    "auto".to_string()
}

impl Default for AgentDefaults {
    fn default() -> Self {
        Self {
            workspace: default_workspace(),
            model: default_model(),
            local_model: String::new(),
            local_api_base: String::new(),
            local_max_context_tokens: default_local_max_context_tokens(),
            max_tokens: default_max_tokens(),
            temperature: default_temperature(),
            max_tool_iterations: default_max_tool_iterations(),
            max_context_tokens: default_max_context_tokens(),
            max_concurrent_chats: default_max_concurrent_chats(),
            max_tool_result_chars: default_max_tool_result_chars(),
            lms_main_model: String::new(),
            lms_port: default_lms_port(),
            inference_engine: default_inference_engine(),
            instructions_path: None,
            skip_jit_gate: false,
        }
    }
}

/// Agent configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentsConfig {
    #[serde(default)]
    pub defaults: AgentDefaults,
}

// ---------------------------------------------------------------------------
// Provider configs
// ---------------------------------------------------------------------------

/// LLM provider configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderConfig {
    #[serde(default)]
    pub api_key: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_base: Option<String>,
}

/// Configuration for LLM providers.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProvidersConfig {
    #[serde(default)]
    pub anthropic: ProviderConfig,
    #[serde(default)]
    pub openai: ProviderConfig,
    #[serde(default)]
    pub openrouter: ProviderConfig,
    #[serde(default)]
    pub deepseek: ProviderConfig,
    #[serde(default)]
    pub groq: ProviderConfig,
    #[serde(default)]
    pub zhipu: ProviderConfig,
    #[serde(default)]
    pub zhipu_coding: ProviderConfig,
    #[serde(default)]
    pub vllm: ProviderConfig,
    #[serde(default)]
    pub gemini: ProviderConfig,
    #[serde(default)]
    pub huggingface: ProviderConfig,
}

/// Known provider prefixes and their default base URLs.
///
/// Single source of truth used by both `Config::resolve_provider_for_model`
/// and `SubagentManager::resolve_provider_for_model`.
pub const PROVIDER_PREFIXES: &[(&str, fn(&ProvidersConfig) -> &ProviderConfig, &str)] = &[
    ("groq/", |p| &p.groq, "https://api.groq.com/openai/v1"),
    (
        "gemini/",
        |p| &p.gemini,
        "https://generativelanguage.googleapis.com/v1beta/openai",
    ),
    ("openai/", |p| &p.openai, "https://api.openai.com/v1"),
    (
        "anthropic/",
        |p| &p.anthropic,
        "https://api.anthropic.com/v1",
    ),
    ("deepseek/", |p| &p.deepseek, "https://api.deepseek.com"),
    (
        "huggingface/",
        |p| &p.huggingface,
        "https://router.huggingface.co/v1",
    ),
    (
        "zhipu-coding/",
        |p| &p.zhipu_coding,
        "https://api.z.ai/api/coding/paas/v4",
    ),
    ("zhipu/", |p| &p.zhipu, "https://api.z.ai/api/paas/v4"),
    (
        "openrouter/",
        |p| &p.openrouter,
        "https://openrouter.ai/api/v1",
    ),
];

impl ProvidersConfig {
    /// Resolve a model string with a provider prefix (e.g. `groq/llama-3.3-70b`)
    /// to `(api_key, api_base, stripped_model)`.
    ///
    /// Returns `None` if the prefix isn't recognized or the provider has no API key.
    pub fn resolve_model_prefix(&self, model: &str) -> Option<(String, String, String)> {
        for (prefix, accessor, default_base) in PROVIDER_PREFIXES {
            if let Some(rest) = model.strip_prefix(prefix) {
                let cfg = accessor(self);
                if !cfg.api_key.is_empty() {
                    let base = cfg.api_base.as_deref().unwrap_or(default_base);
                    return Some((cfg.api_key.clone(), base.to_string(), rest.to_string()));
                }
            }
        }
        None
    }
}

// ---------------------------------------------------------------------------
// Gateway config
// ---------------------------------------------------------------------------

/// Gateway/server configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GatewayConfig {
    #[serde(default = "default_gateway_host")]
    pub host: String,
    #[serde(default = "default_gateway_port")]
    pub port: u16,
}

fn default_gateway_host() -> String {
    "0.0.0.0".to_string()
}

fn default_gateway_port() -> u16 {
    18790
}

impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            host: default_gateway_host(),
            port: default_gateway_port(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tools configs
// ---------------------------------------------------------------------------

/// Web search tool configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WebSearchConfig {
    #[serde(default)]
    pub api_key: String,
    #[serde(default = "default_max_results")]
    pub max_results: u32,
    #[serde(default = "default_search_provider")]
    pub provider: String,
    #[serde(default = "default_searxng_url")]
    pub searxng_url: String,
}

fn default_max_results() -> u32 {
    5
}

fn default_search_provider() -> String {
    "searxng".to_string()
}

fn default_searxng_url() -> String {
    "http://localhost:8888".to_string()
}

impl Default for WebSearchConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            max_results: default_max_results(),
            provider: default_search_provider(),
            searxng_url: default_searxng_url(),
        }
    }
}

/// Web tools configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WebToolsConfig {
    #[serde(default)]
    pub search: WebSearchConfig,
}

/// Shell exec tool configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExecToolConfig {
    #[serde(default = "default_exec_timeout")]
    pub timeout: u64,
    #[serde(default)]
    pub restrict_to_workspace: bool,
}

fn default_exec_timeout() -> u64 {
    60
}

impl Default for ExecToolConfig {
    fn default() -> Self {
        Self {
            timeout: default_exec_timeout(),
            restrict_to_workspace: false,
        }
    }
}

/// Tools configuration.
///
/// Note: the `exec` field from Python is renamed to `exec_` in Rust to avoid
/// the reserved keyword, but serializes as `"exec"` in JSON.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolsConfig {
    #[serde(default)]
    pub web: WebToolsConfig,
    #[serde(default, rename = "exec")]
    pub exec_: ExecToolConfig,
}

// ---------------------------------------------------------------------------
// Trio router config
// ---------------------------------------------------------------------------

fn default_trio_router_port() -> u16 {
    8094
}

fn default_trio_router_ctx_tokens() -> usize {
    4096
}

fn default_trio_router_temperature() -> f64 {
    0.2
}

fn default_trio_router_top_p() -> f64 {
    0.95
}

fn default_trio_router_no_think() -> bool {
    true
}

fn default_trio_main_no_think() -> bool {
    true
}

fn default_trio_specialist_port() -> u16 {
    8095
}

fn default_trio_specialist_ctx_tokens() -> usize {
    8192
}

fn default_trio_specialist_temperature() -> f64 {
    0.7
}

fn default_vram_cap_gb() -> f64 {
    16.0
}

/// Circuit breaker tuning. Nested under trio.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct CircuitBreakerConfig {
    /// Number of consecutive failures before tripping (default: 3).
    pub threshold: u32,
    /// Cooldown period in seconds after tripping (default: 300).
    pub cooldown_secs: u64,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            threshold: 3,
            cooldown_secs: 300,
        }
    }
}

/// A URL + model pair identifying a specific model on a specific server.
///
/// Used for trio roles (router, specialist) so that both single-server (LM Studio)
/// and multi-server (llama.cpp) setups are expressed the same way.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelEndpoint {
    /// Full API base URL, e.g. "http://localhost:1234/v1".
    pub url: String,
    /// Model identifier sent in the API request, e.g. "nvidia_orchestrator-8b".
    pub model: String,
}

/// Configuration for the SLM trio (router + specialist helpers).
/// Default trio: gemma-3n-e4b-it (main) + nvidia_orchestrator-8b (router) + ministral-3-8b (specialist).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrioConfig {
    /// Enable the trio workflow (defaults to false).
    #[serde(default)]
    pub enabled: bool,
    /// Use /no_think mode for main model (gemma-3n) to output directly to content.
    #[serde(default = "default_trio_main_no_think")]
    pub main_no_think: bool,
    /// Local GGUF filename for the router (nvidia_orchestrator-8b). Stored in ~/models/.
    #[serde(default)]
    pub router_model: String,
    /// TCP port for the router server (default: 8094).
    #[serde(default = "default_trio_router_port")]
    pub router_port: u16,
    /// Context size for the router (default: 4096).
    #[serde(default = "default_trio_router_ctx_tokens")]
    pub router_ctx_tokens: usize,
    /// Temperature for router sampling (default: 0.6).
    #[serde(default = "default_trio_router_temperature")]
    pub router_temperature: f64,
    /// Top-p for router sampling (default: 0.95).
    #[serde(default = "default_trio_router_top_p")]
    pub router_top_p: f64,
    /// Use /no_think mode for direct JSON output (default: true).
    #[serde(default = "default_trio_router_no_think")]
    pub router_no_think: bool,
    /// Specialist SLM (summary/coder) filename stored in ~/models/.
    #[serde(default)]
    pub specialist_model: String,
    /// Port for the specialist server (default: 8095).
    #[serde(default = "default_trio_specialist_port")]
    pub specialist_port: u16,
    /// Context size for the specialist (default: 8192).
    #[serde(default = "default_trio_specialist_ctx_tokens")]
    pub specialist_ctx_tokens: usize,
    /// Temperature for the specialist LLM (default: 0.7).
    #[serde(default = "default_trio_specialist_temperature")]
    pub specialist_temperature: f64,
    /// Explicit endpoint for the router role (takes priority over router_port + router_model).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub router_endpoint: Option<ModelEndpoint>,
    /// Explicit endpoint for the specialist role (takes priority over specialist_port + specialist_model).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub specialist_endpoint: Option<ModelEndpoint>,
    /// VRAM budget cap in GB (default: 16). Context sizes auto-computed to fit.
    #[serde(default = "default_vram_cap_gb")]
    pub vram_cap_gb: f64,
    /// Anti-drift hooks for SLM context quality stabilization.
    #[serde(default)]
    pub anti_drift: AntiDriftConfig,
    /// Circuit breaker tuning for trio provider health tracking.
    #[serde(default)]
    pub circuit_breaker: CircuitBreakerConfig,
    /// When true, specialist is instructed to return a strict JSON envelope
    /// (`SpecialistResponse`) and the raw output is parsed accordingly.
    /// Defaults to false for backward compatibility.
    #[serde(default)]
    pub specialist_output_schema: bool,
    #[serde(default)]
    pub trace_log: bool,
}

/// Anti-drift configuration for SLM context stabilization.
///
/// Pre/post completion hooks that score turn quality, evict pollution,
/// collapse repetition, re-inject format anchors, and strip thinking artifacts.
/// Zero extra LLM calls — all heuristic-based.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AntiDriftConfig {
    /// Enable anti-drift hooks (default: true).
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Inject a format anchor every N iterations (default: 3).
    #[serde(default = "default_anchor_interval")]
    pub anchor_interval: u32,
    /// Pollution score threshold to evict a turn (default: 0.6, requires 2+ signals).
    #[serde(default = "default_pollution_threshold")]
    pub pollution_threshold: f32,
    /// Max word count before babble collapse fires (default: 200).
    #[serde(default = "default_babble_max_tokens")]
    pub babble_max_tokens: usize,
    /// Minimum consecutive identical fingerprints to trigger collapse (default: 3).
    #[serde(default = "default_repetition_min_count")]
    pub repetition_min_count: usize,
}

fn default_anchor_interval() -> u32 {
    3
}

fn default_pollution_threshold() -> f32 {
    0.6
}

fn default_babble_max_tokens() -> usize {
    200
}

fn default_repetition_min_count() -> usize {
    3
}

impl Default for AntiDriftConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            anchor_interval: default_anchor_interval(),
            pollution_threshold: default_pollution_threshold(),
            babble_max_tokens: default_babble_max_tokens(),
            repetition_min_count: default_repetition_min_count(),
        }
    }
}

impl Default for TrioConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            main_no_think: default_trio_main_no_think(),
            router_model: String::new(),
            router_port: default_trio_router_port(),
            router_ctx_tokens: default_trio_router_ctx_tokens(),
            router_temperature: default_trio_router_temperature(),
            router_top_p: default_trio_router_top_p(),
            router_no_think: default_trio_router_no_think(),
            specialist_model: String::new(),
            specialist_port: default_trio_specialist_port(),
            specialist_ctx_tokens: default_trio_specialist_ctx_tokens(),
            specialist_temperature: default_trio_specialist_temperature(),
            router_endpoint: None,
            specialist_endpoint: None,
            vram_cap_gb: default_vram_cap_gb(),
            anti_drift: AntiDriftConfig::default(),
            circuit_breaker: CircuitBreakerConfig::default(),
            specialist_output_schema: false,
            trace_log: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Memory config
// ---------------------------------------------------------------------------

/// Tuning knobs for context compaction. Nested under memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct CompactionTuning {
    /// Maximum merge rounds during compaction (default: 6).
    pub max_merge_rounds: usize,
}

impl Default for CompactionTuning {
    fn default() -> Self {
        Self {
            max_merge_rounds: 6,
        }
    }
}

/// Tuning knobs for session management. Nested under memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct SessionTuning {
    /// Rotate session file when it exceeds this size in bytes (default: 1_000_000).
    pub rotation_size_bytes: usize,
    /// Number of recent messages to carry into a new session (default: 10).
    pub rotation_carry_messages: usize,
}

impl Default for SessionTuning {
    fn default() -> Self {
        Self {
            rotation_size_bytes: 1_000_000,
            rotation_carry_messages: 10,
        }
    }
}

/// Tuning knobs for context hygiene. Nested under memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ContextHygieneConfig {
    /// Number of recent messages to keep untruncated (default: 20).
    pub keep_last_messages: usize,
}

impl Default for ContextHygieneConfig {
    fn default() -> Self {
        Self {
            keep_last_messages: 20,
        }
    }
}

/// Configuration for the observational memory system.
///
/// Observations are LLM-generated summaries of conversations, saved after
/// context compaction. A background reflector periodically condenses
/// observations into long-term memory (`MEMORY.md`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryConfig {
    /// Enable/disable observational memory (default: true).
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Model to use for memory operations (observation + reflection).
    /// If empty: Anthropic/OpenRouter defaults to "haiku", other cloud providers
    /// fall back to the main model, local defaults to trio specialist if available.
    /// Override with any model name, e.g. "gemini/gemini-2.5-flash".
    #[serde(default)]
    pub model: String,

    /// Optional separate provider for memory operations.
    /// If not set, reuses the main agent's provider.
    /// Allows pointing memory at a local LM Studio or cheap cloud API.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<ProviderConfig>,

    /// Deprecated: observations are no longer injected into the system prompt.
    /// Kept for backward compatibility with existing config files.
    #[serde(default = "default_observation_budget")]
    pub observation_budget: usize,

    /// Max tokens for working memory (per-session state) in the system prompt (default: 1500).
    #[serde(default = "default_working_memory_budget")]
    pub working_memory_budget: usize,

    /// Token threshold to trigger reflection (default: 20000).
    #[serde(default = "default_reflection_threshold")]
    pub reflection_threshold: usize,

    /// Seconds of inactivity before auto-completing a working memory session (default: 3600).
    #[serde(default = "default_session_complete_after_secs")]
    pub session_complete_after_secs: u64,

    /// Turns of inactivity before clearing the current session's working memory (default: 15).
    #[serde(default = "default_stale_memory_turn_threshold")]
    pub stale_memory_turn_threshold: u64,

    /// Compaction threshold as a percentage of available context (default: 66.6%).
    /// Compaction fires when this OR `compaction_threshold_tokens` is exceeded.
    #[serde(default = "default_compaction_threshold_percent")]
    pub compaction_threshold_percent: f64,

    /// Compaction threshold in absolute tokens (default: 100000).
    /// Compaction fires when this OR `compaction_threshold_percent` is exceeded.
    /// For large contexts (1M), the percent threshold alone would be too late.
    #[serde(default = "default_compaction_threshold_tokens")]
    pub compaction_threshold_tokens: usize,

    /// Maximum age (in turns) before messages are preferred for eviction (default: 50).
    /// Messages older than this are dropped first during trim_to_fit.
    #[serde(default = "default_max_message_age_turns")]
    pub max_message_age_turns: usize,

    /// Maximum number of user turns to load from session history (default: 10).
    /// Working memory carries context from older turns, so loading fewer turns
    /// saves context budget for the current conversation.
    #[serde(default = "default_max_history_turns")]
    pub max_history_turns: usize,

    /// When true, skills are loaded as names+descriptions only (not full content).
    /// The agent fetches full skill content on demand via the `read_skill` tool.
    /// This keeps the system prompt lean (RLM pattern: context as variable).
    #[serde(default)]
    pub lazy_skills: bool,

    /// Context window (tokens) of the compaction/memory model.
    /// Set when the memory model differs from the main model (e.g. a 2K summarizer).
    /// Default: 0 (use main model's context size).
    #[serde(default)]
    pub compaction_model_context_size: usize,

    /// Tuning knobs for context compaction.
    #[serde(default)]
    pub compaction: CompactionTuning,

    /// Tuning knobs for session management.
    #[serde(default)]
    pub session: SessionTuning,

    /// Tuning knobs for context hygiene.
    #[serde(default)]
    pub hygiene: ContextHygieneConfig,
}

fn default_true() -> bool {
    true
}

fn default_observation_budget() -> usize {
    2000
}

fn default_working_memory_budget() -> usize {
    600
}

fn default_session_complete_after_secs() -> u64 {
    3600
}

fn default_stale_memory_turn_threshold() -> u64 {
    15
}

fn default_compaction_threshold_percent() -> f64 {
    66.6
}

fn default_compaction_threshold_tokens() -> usize {
    100_000
}

fn default_max_message_age_turns() -> usize {
    50
}

fn default_max_history_turns() -> usize {
    10
}

fn default_reflection_threshold() -> usize {
    20000
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            enabled: default_true(),
            model: String::new(),
            provider: None,
            observation_budget: default_observation_budget(),
            working_memory_budget: default_working_memory_budget(),
            reflection_threshold: default_reflection_threshold(),
            session_complete_after_secs: default_session_complete_after_secs(),
            stale_memory_turn_threshold: default_stale_memory_turn_threshold(),
            compaction_threshold_percent: default_compaction_threshold_percent(),
            compaction_threshold_tokens: default_compaction_threshold_tokens(),
            max_message_age_turns: default_max_message_age_turns(),
            max_history_turns: default_max_history_turns(),
            lazy_skills: true,
            compaction_model_context_size: 0,
            compaction: CompactionTuning::default(),
            session: SessionTuning::default(),
            hygiene: ContextHygieneConfig::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Provenance config
// ---------------------------------------------------------------------------

/// Configuration for the Agent Provenance Protocol.
///
/// When enabled, tool calls are recorded in an immutable audit log,
/// tool execution is shown in the REPL, and the agent's claims can be
/// mechanically verified against actual tool outputs.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProvenanceConfig {
    /// Enable the provenance system (default: true).
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Write an append-only audit log of all tool calls (default: true when enabled).
    #[serde(default = "default_true")]
    pub audit_log: bool,

    /// Show tool call start/end events in the REPL (default: true).
    #[serde(default = "default_true")]
    pub show_tool_calls: bool,

    /// Run mechanical claim verification on agent responses (default: true).
    #[serde(default = "default_true")]
    pub verify_claims: bool,

    /// Strict mode: redact unverified claims from responses (default: true).
    #[serde(default = "default_true")]
    pub strict_mode: bool,

    /// Inject verification rules into the system prompt (default: true).
    #[serde(default = "default_true")]
    pub system_prompt_rules: bool,

    /// Force a user-visible response after every exec/write_file call (default: true).
    #[serde(default = "default_true")]
    pub response_boundary: bool,
}

impl Default for ProvenanceConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            audit_log: true,
            show_tool_calls: true,
            verify_claims: true,
            strict_mode: true,
            system_prompt_rules: true,
            response_boundary: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Subagent tuning config
// ---------------------------------------------------------------------------

/// Tuning knobs for subagent execution. Nested under toolDelegation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct SubagentTuning {
    /// Maximum iterations for a subagent run (default: 15).
    pub max_iterations: u32,
    /// Maximum spawn depth for nested subagents (default: 3).
    pub max_spawn_depth: u32,
    /// Fallback context window for local subagents (default: 8192).
    pub local_fallback_context: usize,
    /// Minimum context window for local subagents (default: 2048).
    pub local_min_context: usize,
    /// Maximum response tokens for local subagents (default: 1024).
    pub local_max_response_tokens: u32,
    /// Minimum response tokens for local subagents (default: 256).
    pub local_min_response_tokens: u32,
}

impl Default for SubagentTuning {
    fn default() -> Self {
        Self {
            max_iterations: 15,
            max_spawn_depth: 3,
            local_fallback_context: 8192,
            local_min_context: 2048,
            local_max_response_tokens: 1024,
            local_min_response_tokens: 256,
        }
    }
}

// ---------------------------------------------------------------------------
// Tool delegation config
// ---------------------------------------------------------------------------

fn default_td_max_iterations() -> u32 {
    10
}

fn default_td_max_tokens() -> u32 {
    1024
}

/// High-level delegation mode that sets sensible defaults for the strict flags.
///
/// Use this instead of configuring individual `strict_*` booleans:
/// - **Inline**: Main model calls tools directly (no delegation).
/// - **Delegated**: Tools delegated to a cheaper tool runner model.
/// - **Trio**: Strict separation — main=conversation, router=dispatch, specialist=execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub enum DelegationMode {
    /// Main model calls tools directly (delegation disabled).
    Inline,
    /// Tools delegated to tool runner model (default).
    #[default]
    Delegated,
    /// Strict trio: main=orchestrator, router=dispatch, specialist=tools.
    Trio,
}

/// Configuration for delegating tool execution loops to a cheaper model.
///
/// When enabled, tool calls from the main LLM are handed off to a lightweight
/// model that executes the tools and interprets their results, conserving the
/// main model's context window for reasoning.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolDelegationConfig {
    /// High-level mode (overrides strict_* flags when set).
    /// Defaults to `Delegated`. Set to `trio` for strict separation or
    /// `inline` to disable delegation entirely.
    #[serde(default)]
    pub mode: DelegationMode,
    /// Enable tool delegation (default: true).
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Model to use for the tool runner. Empty string = use main model.
    #[serde(default)]
    pub model: String,

    /// Optional separate provider for the tool runner.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<ProviderConfig>,

    /// Max tool loop iterations for the runner (default: 10).
    #[serde(default = "default_td_max_iterations")]
    pub max_iterations: u32,

    /// Max tokens per runner LLM call (default: 4096).
    #[serde(default = "default_td_max_tokens")]
    pub max_tokens: u32,

    /// Inject only truncated previews of tool results into the main context
    /// instead of full output. The runner's summary carries the meaning.
    /// Default: true (the whole point of delegation is context savings).
    #[serde(default = "default_true")]
    pub slim_results: bool,

    /// Max chars per tool result preview injected into the main context
    /// when `slim_results` is enabled (default: 200).
    #[serde(default = "default_td_preview_chars")]
    pub max_result_preview_chars: usize,

    /// Auto-spawn a local delegation server when in local mode and no
    /// explicit provider is configured (default: true).
    #[serde(default = "default_true")]
    pub auto_local: bool,

    /// Maximum cost in USD per delegation round (default: 0.01 = 1 cent).
    /// Set to 0.0 to disable cost limiting. Prices fetched from OpenRouter.
    #[serde(default = "default_td_cost_budget")]
    pub cost_budget: f64,

    /// Default model for spawned subagents when no explicit model is provided.
    /// Prevents expensive main models from being used as workers.
    /// Example: "haiku", "zhipu/glm-4.5-air", "local".
    /// Empty string = fall back to main model (not recommended).
    #[serde(default)]
    pub default_subagent_model: String,

    /// When true, reject direct tool calls emitted by the main model.
    /// The main model is forced into conversation/orchestration-only behavior.
    #[serde(default)]
    pub strict_no_tools_main: bool,

    /// When true, require router outputs to match strict JSON schema.
    #[serde(default)]
    pub strict_router_schema: bool,

    /// When true, build and use role-scoped context packs per turn.
    #[serde(default)]
    pub role_scoped_context_packs: bool,

    /// When true, force all subagents/tools to local-only model choices.
    #[serde(default)]
    pub strict_local_only: bool,

    /// When true, validate normalized ToolPlan before any tool execution.
    #[serde(default = "default_true")]
    pub strict_toolplan_validation: bool,

    /// When true, use deterministic fallback routing when router output is invalid.
    #[serde(default = "default_true")]
    pub deterministic_router_fallback: bool,

    /// Maximum identical tool calls allowed in one turn (dedup guard).
    #[serde(default = "default_td_max_same_tool_call")]
    pub max_same_tool_call_per_turn: u32,

    /// Tuning knobs for subagent execution.
    #[serde(default)]
    pub subagent: SubagentTuning,

    /// When true (default), the specialist response is injected into messages
    /// and the main model synthesizes it in its own voice (Continue).
    /// When false, the specialist response goes directly to the user (Break).
    #[serde(default = "default_true")]
    pub specialist_synthesis: bool,
}

fn default_td_cost_budget() -> f64 {
    0.01
}

fn default_td_preview_chars() -> usize {
    200
}

fn default_td_max_same_tool_call() -> u32 {
    3
}

impl Default for ToolDelegationConfig {
    fn default() -> Self {
        Self {
            mode: DelegationMode::default(),
            enabled: true,
            model: String::new(),
            provider: None,
            max_iterations: default_td_max_iterations(),
            max_tokens: default_td_max_tokens(),
            slim_results: true,
            max_result_preview_chars: default_td_preview_chars(),
            auto_local: true,
            cost_budget: default_td_cost_budget(),
            default_subagent_model: String::new(),
            strict_no_tools_main: false,
            strict_router_schema: false,
            role_scoped_context_packs: false,
            strict_local_only: false,
            strict_toolplan_validation: true,
            deterministic_router_fallback: true,
            max_same_tool_call_per_turn: default_td_max_same_tool_call(),
            subagent: SubagentTuning::default(),
            specialist_synthesis: true,
        }
    }
}

impl ToolDelegationConfig {
    /// Apply the high-level `mode` to the individual strict flags.
    ///
    /// Call after deserialization to ensure the mode takes effect.
    /// Individual flag overrides in the JSON are clobbered by mode.
    pub fn apply_mode(&mut self) {
        match self.mode {
            DelegationMode::Inline => {
                self.enabled = false;
                self.strict_no_tools_main = false;
                self.strict_router_schema = false;
                self.role_scoped_context_packs = false;
            }
            DelegationMode::Delegated => {
                self.enabled = true;
                self.strict_no_tools_main = false;
                self.strict_router_schema = false;
            }
            DelegationMode::Trio => {
                self.enabled = true;
                self.strict_no_tools_main = true;
                self.strict_router_schema = true;
                self.role_scoped_context_packs = true;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Worker/Swarm config
// ---------------------------------------------------------------------------

/// Configuration for the Worker/Swarm system.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkerConfig {
    /// Enable the swarm worker system (delegate tool). Default: true.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Maximum delegation depth (how many levels of delegate). Default: 3.
    #[serde(default = "default_worker_max_depth")]
    pub max_depth: u32,
    /// Enable python_eval tool for workers. Default: true.
    #[serde(default = "default_true")]
    pub python: bool,
    /// Enable delegate tool (recursive workers). Default: true.
    #[serde(default = "default_true")]
    pub delegate: bool,
    /// Budget multiplier for children (0.0-1.0). Default: 0.5.
    #[serde(default = "default_budget_multiplier")]
    pub budget_multiplier: f32,
}

fn default_worker_max_depth() -> u32 {
    3
}

fn default_budget_multiplier() -> f32 {
    0.5
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_depth: 3,
            python: true,
            delegate: true,
            budget_multiplier: 0.5,
        }
    }
}

// ---------------------------------------------------------------------------
// Proprioception config
// ---------------------------------------------------------------------------

fn default_grounding_interval() -> u32 {
    8
}

fn default_raw_window() -> usize {
    5
}

fn default_light_window() -> usize {
    20
}

/// Configuration for the ensemble proprioception system.
///
/// Controls shared body awareness, tool scoping, audience-aware compaction,
/// heartbeat grounding, gradient memory, and priority interrupts.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProprioceptionConfig {
    /// Enable the proprioception system (default: true).
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Phase-aware tool scoping for delegation model (default: true).
    #[serde(default = "default_true")]
    pub dynamic_tool_scoping: bool,

    /// Audience-aware compaction prompts (default: true).
    #[serde(default = "default_true")]
    pub audience_aware_compaction: bool,

    /// Turns between grounding injections. 0 = disabled (default: 8).
    #[serde(default = "default_grounding_interval")]
    pub grounding_interval: u32,

    /// Enable gradient memory (3-tier compaction) (default: true).
    #[serde(default = "default_true")]
    pub gradient_memory: bool,

    /// Number of most recent turns kept raw (default: 5).
    #[serde(default = "default_raw_window")]
    pub raw_window: usize,

    /// Number of turns in the light-compression tier (default: 20).
    #[serde(default = "default_light_window")]
    pub light_window: usize,

    /// Enable the aha channel for priority interrupts (default: true).
    #[serde(default = "default_true")]
    pub aha_channel: bool,
}

impl Default for ProprioceptionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            dynamic_tool_scoping: true,
            audience_aware_compaction: true,
            grounding_interval: default_grounding_interval(),
            gradient_memory: true,
            raw_window: default_raw_window(),
            light_window: default_light_window(),
            aha_channel: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Voice config
// ---------------------------------------------------------------------------

/// Configuration for voice mode TTS/STT.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VoiceConfig {
    /// Default language for TTS. "en" = Pocket only (fast), "auto" or None = both engines.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
}

// ---------------------------------------------------------------------------
// LCM (Lossless Context Management) config
// ---------------------------------------------------------------------------

fn default_lcm_tau_soft() -> f64 {
    0.5
}

fn default_lcm_tau_hard() -> f64 {
    0.85
}

fn default_lcm_deterministic_target() -> usize {
    512
}

fn default_lcm_compaction_context_size() -> usize {
    4096
}

/// Configuration for Lossless Context Management.
///
/// LCM replaces destructive compaction with a dual-state memory:
/// an immutable store (session JSONL) + active context with hierarchical
/// summaries. Summaries contain pointers back to originals, so the LLM
/// can `lcm_expand` any summary to recover the full messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LcmSchemaConfig {
    /// Enable LCM (default: false for backward compatibility).
    #[serde(default)]
    pub enabled: bool,
    /// Soft threshold as fraction of available context (0.0-1.0).
    /// Triggers async (non-blocking) compaction. Default: 0.5 (50%).
    #[serde(default = "default_lcm_tau_soft")]
    pub tau_soft: f64,
    /// Hard threshold as fraction of available context (0.0-1.0).
    /// Triggers blocking compaction. Default: 0.85 (85%).
    #[serde(default = "default_lcm_tau_hard")]
    pub tau_hard: f64,
    /// Target tokens for Level 3 deterministic truncation (default: 512).
    #[serde(default = "default_lcm_deterministic_target")]
    pub deterministic_target: usize,
    /// Dedicated compaction endpoint (url + model). When set, LCM uses this
    /// model for summarization instead of the default memory/compaction model.
    /// Example: `{"url": "http://192.168.1.22:1234/v1", "model": "qwen3-0.6b"}`
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compaction_endpoint: Option<ModelEndpoint>,
    /// Context window size of the compaction model in tokens (default: 4096).
    /// Only used when `compaction_endpoint` is set.
    #[serde(default = "default_lcm_compaction_context_size")]
    pub compaction_context_size: usize,
}

impl Default for LcmSchemaConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            tau_soft: default_lcm_tau_soft(),
            tau_hard: default_lcm_tau_hard(),
            deterministic_target: default_lcm_deterministic_target(),
            compaction_endpoint: None,
            compaction_context_size: default_lcm_compaction_context_size(),
        }
    }
}

// ---------------------------------------------------------------------------
// Cluster config (distributed inference via Exo / LAN peers)
// ---------------------------------------------------------------------------

fn default_cluster_scan_ports() -> Vec<u16> {
    vec![52415, 1234, 8080, 1337]
}

fn default_cluster_scan_interval() -> u64 {
    60
}

/// Configuration for distributed inference cluster discovery and routing.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ClusterConfig {
    /// Enable cluster mode (default: false).
    pub enabled: bool,
    /// Enable mDNS + HTTP probe auto-discovery (default: true when enabled).
    pub auto_discover: bool,
    /// Manual peer endpoint URLs (e.g. ["http://192.168.1.50:52415"]).
    pub endpoints: Vec<String>,
    /// Ports to scan during HTTP probe discovery (default: [52415, 1234, 8080]).
    #[serde(default = "default_cluster_scan_ports")]
    pub scan_ports: Vec<u16>,
    /// Seconds between discovery scans (default: 60).
    #[serde(default = "default_cluster_scan_interval")]
    pub scan_interval_secs: u64,
    /// Prefer cluster over cloud when model is available on both (default: true).
    pub prefer_cluster: bool,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            auto_discover: true,
            endpoints: Vec::new(),
            scan_ports: default_cluster_scan_ports(),
            scan_interval_secs: default_cluster_scan_interval(),
            prefer_cluster: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Root config
// ---------------------------------------------------------------------------

/// Root configuration for nanobot.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Config {
    #[serde(default)]
    pub agents: AgentsConfig,
    #[serde(default)]
    pub channels: ChannelsConfig,
    #[serde(default)]
    pub providers: ProvidersConfig,
    #[serde(default)]
    pub gateway: GatewayConfig,
    #[serde(default)]
    pub tools: ToolsConfig,
    #[serde(default)]
    pub memory: MemoryConfig,
    #[serde(default)]
    pub tool_delegation: ToolDelegationConfig,
    #[serde(default)]
    pub provenance: ProvenanceConfig,
    #[serde(default)]
    pub voice: VoiceConfig,
    #[serde(default)]
    pub worker: WorkerConfig,
    #[serde(default)]
    pub proprioception: ProprioceptionConfig,
    #[serde(default)]
    pub trio: TrioConfig,
    #[serde(default)]
    pub cluster: ClusterConfig,
    #[serde(default)]
    pub lcm: LcmSchemaConfig,
    #[serde(default)]
    pub model_capabilities: HashMap<String, crate::agent::model_capabilities::ModelCapabilitiesOverride>,
}

impl Config {
    fn is_provider_key_enabled(key: &str) -> bool {
        !key.is_empty() && !key.eq_ignore_ascii_case("none")
    }

    /// Get the expanded workspace path.
    pub fn workspace_path(&self) -> PathBuf {
        let ws = &self.agents.defaults.workspace;
        expand_tilde(ws)
    }

    /// Get the API key in priority order:
    /// OpenRouter > DeepSeek > Anthropic > OpenAI > Gemini > Zhipu > Groq > vLLM.
    pub fn get_api_key(&self) -> Option<String> {
        let candidates = [
            &self.providers.openrouter.api_key,
            &self.providers.deepseek.api_key,
            &self.providers.anthropic.api_key,
            &self.providers.openai.api_key,
            &self.providers.gemini.api_key,
            &self.providers.zhipu.api_key,
            &self.providers.zhipu_coding.api_key,
            &self.providers.groq.api_key,
            &self.providers.vllm.api_key,
        ];
        for key in candidates {
            if Self::is_provider_key_enabled(key) {
                return Some(key.clone());
            }
        }
        None
    }

    /// Resolve a model string with a provider prefix to (api_key, api_base, stripped_model).
    ///
    /// Delegates to `ProvidersConfig::resolve_model_prefix`.
    pub fn resolve_provider_for_model(&self, model: &str) -> Option<(String, String, String)> {
        self.providers.resolve_model_prefix(model)
    }

    /// Get the API base URL for the active provider.
    ///
    /// Detection order matches `get_api_key()` priority so that the key and
    /// base always refer to the same provider.
    pub fn get_api_base(&self) -> Option<String> {
        if Self::is_provider_key_enabled(&self.providers.openrouter.api_key) {
            return Some(
                self.providers
                    .openrouter
                    .api_base
                    .clone()
                    .unwrap_or_else(|| "https://openrouter.ai/api/v1".to_string()),
            );
        }
        if Self::is_provider_key_enabled(&self.providers.deepseek.api_key) {
            return Some("https://api.deepseek.com".to_string());
        }
        if Self::is_provider_key_enabled(&self.providers.anthropic.api_key) {
            return Some("https://api.anthropic.com/v1".to_string());
        }
        if Self::is_provider_key_enabled(&self.providers.openai.api_key) {
            return Some("https://api.openai.com/v1".to_string());
        }
        if Self::is_provider_key_enabled(&self.providers.gemini.api_key) {
            return Some("https://generativelanguage.googleapis.com/v1beta/openai".to_string());
        }
        if Self::is_provider_key_enabled(&self.providers.zhipu.api_key) {
            return Some(
                self.providers
                    .zhipu
                    .api_base
                    .clone()
                    .unwrap_or_else(|| "https://api.z.ai/api/paas/v4".to_string()),
            );
        }
        if Self::is_provider_key_enabled(&self.providers.zhipu_coding.api_key) {
            return Some(
                self.providers
                    .zhipu_coding
                    .api_base
                    .clone()
                    .unwrap_or_else(|| "https://api.z.ai/api/coding/paas/v4".to_string()),
            );
        }
        if Self::is_provider_key_enabled(&self.providers.groq.api_key) {
            return Some(
                self.providers
                    .groq
                    .api_base
                    .clone()
                    .unwrap_or_else(|| "https://api.groq.com/openai/v1".to_string()),
            );
        }
        if self.providers.vllm.api_base.is_some() {
            return self.providers.vllm.api_base.clone();
        }
        None
    }
}

/// Expand a leading `~` to the user's home directory.
fn expand_tilde(path: &str) -> PathBuf {
    if let Some(rest) = path.strip_prefix("~/") {
        let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        home.join(rest)
    } else if path == "~" {
        dirs::home_dir().unwrap_or_else(|| PathBuf::from("."))
    } else {
        PathBuf::from(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_serialization_roundtrip() {
        let cfg = Config::default();
        let json = serde_json::to_string_pretty(&cfg).unwrap();
        let cfg2: Config = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg2.agents.defaults.model, "anthropic/claude-opus-4-5");
        assert_eq!(cfg2.gateway.port, 18790);
    }

    #[test]
    fn test_api_key_priority() {
        let mut cfg = Config::default();
        cfg.providers.anthropic.api_key = "anthropic-key".to_string();
        cfg.providers.openrouter.api_key = "openrouter-key".to_string();
        assert_eq!(cfg.get_api_key(), Some("openrouter-key".to_string()));
    }

    #[test]
    fn test_api_key_none_when_empty() {
        let cfg = Config::default();
        assert_eq!(cfg.get_api_key(), None);
    }

    #[test]
    fn test_api_base_openrouter() {
        let mut cfg = Config::default();
        cfg.providers.openrouter.api_key = "key".to_string();
        assert_eq!(
            cfg.get_api_base(),
            Some("https://openrouter.ai/api/v1".to_string())
        );
    }

    #[test]
    fn test_exec_rename() {
        let json = r#"{"exec": {"timeout": 30, "restrictToWorkspace": true}}"#;
        let tools: ToolsConfig = serde_json::from_str(json).unwrap();
        assert_eq!(tools.exec_.timeout, 30);
        assert!(tools.exec_.restrict_to_workspace);
    }

    #[test]
    fn test_workspace_path() {
        let cfg = Config::default();
        let ws = cfg.workspace_path();
        assert!(ws.ends_with(".nanobot/workspace"));
    }

    #[test]
    fn test_api_base_anthropic() {
        let mut cfg = Config::default();
        cfg.providers.anthropic.api_key = "sk-ant-key".to_string();
        assert_eq!(
            cfg.get_api_base(),
            Some("https://api.anthropic.com/v1".to_string())
        );
    }

    #[test]
    fn test_api_base_openai() {
        let mut cfg = Config::default();
        cfg.providers.openai.api_key = "sk-key".to_string();
        assert_eq!(
            cfg.get_api_base(),
            Some("https://api.openai.com/v1".to_string())
        );
    }

    #[test]
    fn test_api_base_groq() {
        let mut cfg = Config::default();
        cfg.providers.groq.api_key = "gsk_key".to_string();
        assert_eq!(
            cfg.get_api_base(),
            Some("https://api.groq.com/openai/v1".to_string())
        );
    }

    #[test]
    fn test_api_base_deepseek() {
        let mut cfg = Config::default();
        cfg.providers.deepseek.api_key = "sk-ds-key".to_string();
        assert_eq!(
            cfg.get_api_base(),
            Some("https://api.deepseek.com".to_string())
        );
    }

    #[test]
    fn test_api_base_none_when_no_provider() {
        let cfg = Config::default();
        assert_eq!(cfg.get_api_base(), None);
    }

    #[test]
    fn test_local_vllm_provider_selected_when_cloud_disabled() {
        let mut cfg = Config::default();
        cfg.providers.openrouter.api_key = "none".to_string();
        cfg.providers.anthropic.api_key.clear();
        cfg.providers.openai.api_key.clear();
        cfg.providers.groq.api_key = "none".to_string();
        cfg.providers.vllm.api_key = "local".to_string();
        cfg.providers.vllm.api_base = Some("http://127.0.0.1:18080/v1".to_string());

        assert_eq!(cfg.get_api_key(), Some("local".to_string()));
        assert_eq!(
            cfg.get_api_base(),
            Some("http://127.0.0.1:18080/v1".to_string())
        );
    }

    #[test]
    fn test_tool_delegation_config_defaults() {
        let td = ToolDelegationConfig::default();
        assert!(td.enabled);
        assert!(td.model.is_empty());
        assert!(td.provider.is_none());
        assert_eq!(td.max_iterations, 10);
        assert_eq!(td.max_tokens, 1024);
        assert!(td.slim_results);
        assert_eq!(td.max_result_preview_chars, 200);
        assert!(td.auto_local);
        assert!(!td.strict_no_tools_main);
        assert!(!td.strict_router_schema);
        assert!(!td.role_scoped_context_packs);
        assert!(!td.strict_local_only);
        assert!(td.strict_toolplan_validation);
        assert!(td.deterministic_router_fallback);
        assert_eq!(td.max_same_tool_call_per_turn, 3);
    }

    #[test]
    fn test_tool_delegation_config_roundtrip() {
        let td = ToolDelegationConfig {
            enabled: true,
            model: "qwen2-0.5b".to_string(),
            provider: Some(ProviderConfig {
                api_key: "local".to_string(),
                api_base: Some("http://localhost:8080/v1".to_string()),
            }),
            max_iterations: 10,
            max_tokens: 2048,
            slim_results: true,
            max_result_preview_chars: 300,
            auto_local: true,
            cost_budget: 0.01,
            default_subagent_model: String::new(),
            strict_no_tools_main: true,
            strict_router_schema: true,
            role_scoped_context_packs: true,
            strict_local_only: true,
            strict_toolplan_validation: true,
            deterministic_router_fallback: true,
            max_same_tool_call_per_turn: 1,
            specialist_synthesis: true,
            mode: DelegationMode::Trio,
            subagent: SubagentTuning::default(),
        };
        let json = serde_json::to_string(&td).unwrap();
        let td2: ToolDelegationConfig = serde_json::from_str(&json).unwrap();
        assert!(td2.enabled);
        assert_eq!(td2.model, "qwen2-0.5b");
        assert_eq!(td2.max_iterations, 10);
        assert_eq!(td2.max_tokens, 2048);
        assert!(td2.provider.is_some());
        assert!(td2.strict_no_tools_main);
        assert!(td2.strict_router_schema);
        assert!(td2.role_scoped_context_packs);
        assert!(td2.strict_local_only);
        assert!(td2.strict_toolplan_validation);
        assert!(td2.deterministic_router_fallback);
        assert_eq!(td2.max_same_tool_call_per_turn, 1);
    }

    #[test]
    fn test_tool_delegation_config_in_root() {
        let json =
            r#"{"toolDelegation": {"enabled": true, "model": "small-model", "maxIterations": 5}}"#;
        let cfg: Config = serde_json::from_str(json).unwrap();
        assert!(cfg.tool_delegation.enabled);
        assert_eq!(cfg.tool_delegation.model, "small-model");
        assert_eq!(cfg.tool_delegation.max_iterations, 5);
        assert_eq!(cfg.tool_delegation.max_tokens, 1024); // default
    }

    #[test]
    fn test_api_base_priority_matches_key_priority() {
        // When both OpenRouter and Anthropic keys are set, OpenRouter wins
        // (matching get_api_key priority).
        let mut cfg = Config::default();
        cfg.providers.openrouter.api_key = "or-key".to_string();
        cfg.providers.anthropic.api_key = "ant-key".to_string();
        assert_eq!(
            cfg.get_api_base(),
            Some("https://openrouter.ai/api/v1".to_string())
        );
    }

    #[test]
    fn test_provenance_config_defaults() {
        let pc = ProvenanceConfig::default();
        assert!(pc.enabled);
        assert!(pc.audit_log);
        assert!(pc.show_tool_calls);
        assert!(pc.verify_claims);
        assert!(pc.strict_mode);
        assert!(pc.system_prompt_rules);
        assert!(pc.response_boundary);
    }

    #[test]
    fn test_provenance_config_roundtrip() {
        let pc = ProvenanceConfig {
            enabled: true,
            audit_log: true,
            show_tool_calls: false,
            verify_claims: true,
            strict_mode: true,
            system_prompt_rules: false,
            response_boundary: true,
        };
        let json = serde_json::to_string(&pc).unwrap();
        let pc2: ProvenanceConfig = serde_json::from_str(&json).unwrap();
        assert!(pc2.enabled);
        assert!(!pc2.show_tool_calls);
        assert!(pc2.verify_claims);
        assert!(pc2.strict_mode);
        assert!(!pc2.system_prompt_rules);
        assert!(pc2.response_boundary);
    }

    #[test]
    fn test_provenance_config_in_root() {
        let json = r#"{"provenance": {"enabled": true, "verifyClaims": true}}"#;
        let cfg: Config = serde_json::from_str(json).unwrap();
        assert!(cfg.provenance.enabled);
        assert!(cfg.provenance.verify_claims);
        assert!(cfg.provenance.audit_log); // default true
        assert!(cfg.provenance.show_tool_calls); // default true
        assert!(cfg.provenance.strict_mode); // default true
        assert!(cfg.provenance.response_boundary); // default true
    }

    // -- auto_local config field tests --

    #[test]
    fn test_auto_local_defaults_to_true() {
        // When auto_local is absent from JSON, it should default to true
        let json = r#"{"enabled": true, "model": "small-model"}"#;
        let td: ToolDelegationConfig = serde_json::from_str(json).unwrap();
        assert!(
            td.auto_local,
            "auto_local should default to true when absent"
        );
    }

    #[test]
    fn test_auto_local_explicit_false() {
        let json = r#"{"enabled": true, "autoLocal": false}"#;
        let td: ToolDelegationConfig = serde_json::from_str(json).unwrap();
        assert!(
            !td.auto_local,
            "auto_local should be false when explicitly set"
        );
    }

    #[test]
    fn test_auto_local_explicit_true() {
        let json = r#"{"enabled": true, "autoLocal": true}"#;
        let td: ToolDelegationConfig = serde_json::from_str(json).unwrap();
        assert!(td.auto_local);
    }

    #[test]
    fn test_auto_local_in_root_config() {
        // auto_local should be accessible through the root Config object
        let json = r#"{"toolDelegation": {"enabled": true, "autoLocal": false}}"#;
        let cfg: Config = serde_json::from_str(json).unwrap();
        assert!(cfg.tool_delegation.enabled);
        assert!(!cfg.tool_delegation.auto_local);
    }

    #[test]
    fn test_voice_config_defaults() {
        let vc = VoiceConfig::default();
        assert!(vc.language.is_none());
    }

    #[test]
    fn test_voice_config_in_root() {
        let json = r#"{"voice": {"language": "en"}}"#;
        let cfg: Config = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.voice.language.as_deref(), Some("en"));
    }

    #[test]
    fn test_voice_config_absent_defaults_to_none() {
        let json = r#"{}"#;
        let cfg: Config = serde_json::from_str(json).unwrap();
        assert!(cfg.voice.language.is_none());
    }

    #[test]
    fn test_auto_local_roundtrip_preserves_value() {
        let td = ToolDelegationConfig {
            enabled: true,
            auto_local: false,
            ..Default::default()
        };
        let json = serde_json::to_string(&td).unwrap();
        let td2: ToolDelegationConfig = serde_json::from_str(&json).unwrap();
        assert!(
            !td2.auto_local,
            "Roundtrip should preserve auto_local=false"
        );
    }

    #[test]
    fn test_local_api_base_deserialization() {
        let json = r#"{"agents": {"defaults": {"localApiBase": "http://192.168.1.22:1234/v1"}}}"#;
        let cfg: Config = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.agents.defaults.local_api_base, "http://192.168.1.22:1234/v1");
        assert!(!cfg.agents.defaults.local_api_base.is_empty());
    }

    #[test]
    fn test_local_api_base_empty_by_default() {
        let json = r#"{}"#;
        let cfg: Config = serde_json::from_str(json).unwrap();
        assert!(cfg.agents.defaults.local_api_base.is_empty());
    }

    #[test]
    fn test_local_max_context_tokens_default() {
        let json = r#"{}"#;
        let cfg: Config = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.agents.defaults.local_max_context_tokens, 32768);
    }

    #[test]
    fn test_max_tool_result_chars_default_constant() {
        let cfg = Config::default();
        assert_eq!(
            cfg.agents.defaults.max_tool_result_chars,
            DEFAULT_MAX_TOOL_RESULT_CHARS
        );
    }

    // -- ModelEndpoint + TrioConfig endpoint tests --

    #[test]
    fn test_model_endpoint_deserialization() {
        let json = r#"{"url": "http://localhost:1234/v1", "model": "nvidia_orchestrator-8b"}"#;
        let ep: ModelEndpoint = serde_json::from_str(json).unwrap();
        assert_eq!(ep.url, "http://localhost:1234/v1");
        assert_eq!(ep.model, "nvidia_orchestrator-8b");
    }

    #[test]
    fn test_trio_config_endpoints_absent_by_default() {
        let json = r#"{}"#;
        let cfg: Config = serde_json::from_str(json).unwrap();
        assert!(cfg.trio.router_endpoint.is_none());
        assert!(cfg.trio.specialist_endpoint.is_none());
    }

    #[test]
    fn test_trio_config_router_endpoint_lmstudio() {
        // Single LM Studio server: both roles share same URL, different models.
        let json = r#"{
            "trio": {
                "enabled": true,
                "routerEndpoint": {
                    "url": "http://localhost:1234/v1",
                    "model": "nvidia_orchestrator-8b"
                },
                "specialistEndpoint": {
                    "url": "http://localhost:1234/v1",
                    "model": "ministral-3-8b-instruct-2512"
                }
            }
        }"#;
        let cfg: Config = serde_json::from_str(json).unwrap();
        assert!(cfg.trio.enabled);
        let re = cfg.trio.router_endpoint.as_ref().unwrap();
        assert_eq!(re.url, "http://localhost:1234/v1");
        assert_eq!(re.model, "nvidia_orchestrator-8b");
        let se = cfg.trio.specialist_endpoint.as_ref().unwrap();
        assert_eq!(se.url, "http://localhost:1234/v1");
        assert_eq!(se.model, "ministral-3-8b-instruct-2512");
    }

    #[test]
    fn test_trio_config_endpoint_separate_servers() {
        // llama.cpp: separate servers on different ports.
        let json = r#"{
            "trio": {
                "routerEndpoint": {
                    "url": "http://localhost:8094/v1",
                    "model": "orchestrator"
                },
                "specialistEndpoint": {
                    "url": "http://localhost:8095/v1",
                    "model": "specialist"
                }
            }
        }"#;
        let cfg: Config = serde_json::from_str(json).unwrap();
        let re = cfg.trio.router_endpoint.as_ref().unwrap();
        assert_eq!(re.url, "http://localhost:8094/v1");
        let se = cfg.trio.specialist_endpoint.as_ref().unwrap();
        assert_eq!(se.url, "http://localhost:8095/v1");
    }

    #[test]
    fn test_trio_config_backwards_compat_port_model() {
        // Old-style config with routerPort + routerModel still works.
        let json = r#"{
            "trio": {
                "enabled": true,
                "routerModel": "nemotron-orchestrator",
                "routerPort": 8094
            }
        }"#;
        let cfg: Config = serde_json::from_str(json).unwrap();
        assert!(cfg.trio.enabled);
        assert_eq!(cfg.trio.router_model, "nemotron-orchestrator");
        assert_eq!(cfg.trio.router_port, 8094);
        assert!(cfg.trio.router_endpoint.is_none(), "endpoint should be absent when not set");
    }

    #[test]
    fn test_trio_config_endpoint_roundtrip() {
        let trio = TrioConfig {
            enabled: true,
            router_endpoint: Some(ModelEndpoint {
                url: "http://localhost:1234/v1".to_string(),
                model: "router-model".to_string(),
            }),
            specialist_endpoint: Some(ModelEndpoint {
                url: "http://localhost:1234/v1".to_string(),
                model: "specialist-model".to_string(),
            }),
            ..Default::default()
        };
        let json = serde_json::to_string(&trio).unwrap();
        let trio2: TrioConfig = serde_json::from_str(&json).unwrap();
        assert!(trio2.router_endpoint.is_some());
        assert_eq!(trio2.router_endpoint.unwrap().model, "router-model");
        assert!(trio2.specialist_endpoint.is_some());
        assert_eq!(trio2.specialist_endpoint.unwrap().model, "specialist-model");
    }

    #[test]
    fn test_trio_vram_cap_gb_default() {
        let json = r#"{}"#;
        let cfg: Config = serde_json::from_str(json).unwrap();
        assert!((cfg.trio.vram_cap_gb - 16.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_trio_vram_cap_gb_roundtrip() {
        let mut trio = TrioConfig::default();
        trio.vram_cap_gb = 12.0;
        let json = serde_json::to_string(&trio).unwrap();
        let trio2: TrioConfig = serde_json::from_str(&json).unwrap();
        assert!((trio2.vram_cap_gb - 12.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_trio_vram_cap_gb_from_json() {
        let json = r#"{"trio": {"vramCapGb": 8.5}}"#;
        let cfg: Config = serde_json::from_str(json).unwrap();
        assert!((cfg.trio.vram_cap_gb - 8.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_trio_config_endpoint_not_serialized_when_none() {
        let trio = TrioConfig::default();
        let json = serde_json::to_string(&trio).unwrap();
        assert!(!json.contains("routerEndpoint"), "None endpoints should be skipped");
        assert!(!json.contains("specialistEndpoint"), "None endpoints should be skipped");
    }

    #[test]
    fn test_lcm_config_defaults() {
        let lcm = LcmSchemaConfig::default();
        assert!(!lcm.enabled);
        assert!((lcm.tau_soft - 0.5).abs() < f64::EPSILON);
        assert!((lcm.tau_hard - 0.85).abs() < f64::EPSILON);
        assert_eq!(lcm.deterministic_target, 512);
        assert!(lcm.compaction_endpoint.is_none());
        assert_eq!(lcm.compaction_context_size, 4096);
    }

    #[test]
    fn test_lcm_config_roundtrip() {
        let mut lcm = LcmSchemaConfig::default();
        lcm.enabled = true;
        lcm.tau_soft = 0.6;
        lcm.tau_hard = 0.9;
        lcm.deterministic_target = 256;
        let json = serde_json::to_string(&lcm).unwrap();
        let lcm2: LcmSchemaConfig = serde_json::from_str(&json).unwrap();
        assert!(lcm2.enabled);
        assert!((lcm2.tau_soft - 0.6).abs() < f64::EPSILON);
        assert!((lcm2.tau_hard - 0.9).abs() < f64::EPSILON);
        assert_eq!(lcm2.deterministic_target, 256);
    }

    #[test]
    fn test_lcm_config_from_root_json() {
        let json = r#"{"lcm": {"enabled": true, "tauSoft": 0.7, "tauHard": 0.9}}"#;
        let cfg: Config = serde_json::from_str(json).unwrap();
        assert!(cfg.lcm.enabled);
        assert!((cfg.lcm.tau_soft - 0.7).abs() < f64::EPSILON);
        assert!((cfg.lcm.tau_hard - 0.9).abs() < f64::EPSILON);
        assert_eq!(cfg.lcm.deterministic_target, 512); // default
    }

    #[test]
    fn test_lcm_absent_defaults_to_disabled() {
        let json = r#"{}"#;
        let cfg: Config = serde_json::from_str(json).unwrap();
        assert!(!cfg.lcm.enabled);
    }

    #[test]
    fn test_compaction_tuning_defaults() {
        let c = CompactionTuning::default();
        assert_eq!(c.max_merge_rounds, 6);
    }

    #[test]
    fn test_session_tuning_defaults() {
        let s = SessionTuning::default();
        assert_eq!(s.rotation_size_bytes, 1_000_000);
        assert_eq!(s.rotation_carry_messages, 10);
    }

    #[test]
    fn test_context_hygiene_config_defaults() {
        let h = ContextHygieneConfig::default();
        assert_eq!(h.keep_last_messages, 20);
    }

    #[test]
    fn test_memory_config_nested_tuning() {
        let json = r#"{"memory": {"compaction": {"maxMergeRounds": 10}, "session": {"rotationSizeBytes": 500000}, "hygiene": {"keepLastMessages": 30}}}"#;
        let cfg: Config = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.memory.compaction.max_merge_rounds, 10);
        assert_eq!(cfg.memory.session.rotation_size_bytes, 500000);
        assert_eq!(cfg.memory.hygiene.keep_last_messages, 30);
    }

    #[test]
    fn test_lcm_compaction_endpoint() {
        let json = r#"{
            "lcm": {
                "enabled": true,
                "compactionEndpoint": {
                    "url": "http://192.168.1.22:1234/v1",
                    "model": "qwen3-0.6b"
                },
                "compactionContextSize": 2048
            }
        }"#;
        let cfg: Config = serde_json::from_str(json).unwrap();
        assert!(cfg.lcm.enabled);
        let ep = cfg.lcm.compaction_endpoint.as_ref().unwrap();
        assert_eq!(ep.url, "http://192.168.1.22:1234/v1");
        assert_eq!(ep.model, "qwen3-0.6b");
        assert_eq!(cfg.lcm.compaction_context_size, 2048);
    }

    #[test]
    fn test_subagent_tuning_defaults() {
        let t = SubagentTuning::default();
        assert_eq!(t.max_iterations, 15);
        assert_eq!(t.max_spawn_depth, 3);
        assert_eq!(t.local_fallback_context, 8192);
        assert_eq!(t.local_min_context, 2048);
        assert_eq!(t.local_max_response_tokens, 1024);
        assert_eq!(t.local_min_response_tokens, 256);
    }

    #[test]
    fn test_subagent_tuning_in_root_config() {
        let json = r#"{"toolDelegation": {"subagent": {"maxIterations": 20, "maxSpawnDepth": 5}}}"#;
        let cfg: Config = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.tool_delegation.subagent.max_iterations, 20);
        assert_eq!(cfg.tool_delegation.subagent.max_spawn_depth, 5);
        // Unspecified fields get defaults
        assert_eq!(cfg.tool_delegation.subagent.local_fallback_context, 8192);
    }

    #[test]
    fn test_memory_tuning_in_root_config() {
        let json = r#"{"memory": {"compaction": {"maxMergeRounds": 10}, "session": {"rotationSizeBytes": 500000}}}"#;
        let cfg: Config = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.memory.compaction.max_merge_rounds, 10);
        assert_eq!(cfg.memory.session.rotation_size_bytes, 500000);
        // Unspecified fields keep defaults
        assert_eq!(cfg.memory.session.rotation_carry_messages, 10);
        assert_eq!(cfg.memory.hygiene.keep_last_messages, 20);
    }

    #[test]
    fn test_circuit_breaker_config_defaults() {
        let c = CircuitBreakerConfig::default();
        assert_eq!(c.threshold, 3);
        assert_eq!(c.cooldown_secs, 300);
    }

    #[test]
    fn test_circuit_breaker_config_in_root_config() {
        let json = r#"{"trio": {"circuitBreaker": {"threshold": 5, "cooldownSecs": 600}}}"#;
        let cfg: Config = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.trio.circuit_breaker.threshold, 5);
        assert_eq!(cfg.trio.circuit_breaker.cooldown_secs, 600);
    }
}
