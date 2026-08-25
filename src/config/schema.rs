//! Configuration schema for nanobot.
//!
//! All structs use `#[serde(rename_all = "camelCase")]` so that the JSON config
//! file can use camelCase keys while Rust code uses snake_case fields.

use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::providers::constants::{
    ANTHROPIC_API_BASE, DEEPSEEK_API_BASE, GEMINI_API_BASE, GROQ_API_BASE, HUGGINGFACE_API_BASE,
    OPENAI_API_BASE, OPENROUTER_API_BASE, ZHIPU_API_BASE, ZHIPU_CODING_API_BASE,
};
use crate::utils::helpers::expand_tilde;

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
#[derive(Clone, Serialize, Deserialize)]
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

impl fmt::Debug for TelegramConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TelegramConfig")
            .field("enabled", &self.enabled)
            .field("token", &crate::config::redact(&self.token))
            .field("allow_from", &self.allow_from)
            .field("proxy", &self.proxy)
            .finish()
    }
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

/// Email channel configuration (IMAP polling + SMTP sending).
#[derive(Clone, Serialize, Deserialize)]
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

impl fmt::Debug for EmailConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EmailConfig")
            .field("enabled", &self.enabled)
            .field("imap_host", &self.imap_host)
            .field("imap_port", &self.imap_port)
            .field("smtp_host", &self.smtp_host)
            .field("smtp_port", &self.smtp_port)
            .field("username", &self.username)
            .field("password", &crate::config::redact(&self.password))
            .field("poll_interval_secs", &self.poll_interval_secs)
            .field("allow_from", &self.allow_from)
            .finish()
    }
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
    pub email: EmailConfig,
}

impl ChannelsConfig {
    /// Enable exactly one channel, disabling all others.
    pub fn enable_exclusive(&mut self, channel: &str) {
        self.whatsapp.enabled = channel == "whatsapp";
        self.telegram.enabled = channel == "telegram";
        self.email.enabled = channel == "email";
    }
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
    /// Preferred local model name (e.g. "Qwen3.6-35B-A3B-4bit"; historically a
    /// GGUF filename like "Qwen3-8B-Q4_K_M.gguf").
    ///
    /// LEGACY: since discovery-first startup this is only the EXPECTED-model
    /// hint for endpoint selection (and the model-dir hint when spawning
    /// Higgs). The model actually requested is adopted from the discovered
    /// endpoint's served list, paired with that endpoint.
    #[serde(default = "default_local_model")]
    pub local_model: String,
    /// Custom API base for local inference (e.g. "http://192.168.1.22:1234/v1").
    /// Highest-priority discovery candidate: when set AND healthy, local mode
    /// uses it. When empty or dead, startup discovers Higgs / LM Studio /
    /// cluster peers instead. All trio roles (main, router, specialist) share
    /// this endpoint; model differentiation happens via the `model` field in
    /// each API request (JIT loading).
    #[serde(default)]
    pub local_api_base: String,
    /// API key for local inference server (default: "local").
    /// Required for servers like oMLX that enforce authentication.
    #[serde(default = "default_local_api_key")]
    pub local_api_key: String,
    /// Context window size for local models (default: 32768).
    /// Separate from maxContextTokens so cloud (512K) and local (32K) coexist.
    #[serde(default = "default_local_max_context_tokens")]
    pub local_max_context_tokens: usize,
    /// Grammar-constrained tool calls for local backends (default: true).
    /// When the router (or other forced-tool path) asks for a tool call, the
    /// local server constrains decoding so the call is always well-formed.
    /// Escape hatch: set false to fall back to unconstrained "auto" behavior.
    #[serde(default = "default_constrained_tool_calls")]
    pub constrained_tool_calls: bool,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    #[serde(
        rename = "repetition_penalty",
        alias = "repetitionPenalty",
        skip_serializing_if = "Option::is_none"
    )]
    pub repetition_penalty: Option<f64>,
    #[serde(
        rename = "frequency_penalty",
        alias = "frequencyPenalty",
        skip_serializing_if = "Option::is_none"
    )]
    pub frequency_penalty: Option<f64>,
    #[serde(
        rename = "presence_penalty",
        alias = "presencePenalty",
        skip_serializing_if = "Option::is_none"
    )]
    pub presence_penalty: Option<f64>,
    #[serde(default = "default_max_tool_iterations")]
    pub max_tool_iterations: u32,
    #[serde(default = "default_max_context_tokens")]
    pub max_context_tokens: usize,
    #[serde(default = "default_max_concurrent_chats")]
    pub max_concurrent_chats: usize,
    /// TUI color-scheme index (0..31), cycled with Ctrl+P. Defaults to 0 (teal).
    #[serde(default)]
    pub theme_index: u8,
    /// Max characters for inline tool results before truncation (default: 30000).
    #[serde(default = "default_max_tool_result_chars")]
    pub max_tool_result_chars: usize,
    /// LM Studio model identifier for the main model (e.g. "gemma-3n-e4b-it").
    /// When empty, derived from local_model via strip_gguf_suffix.
    ///
    /// LEGACY: since discovery-first startup, the served model id is adopted
    /// from the discovered endpoint at runtime; this field is only a hint for
    /// the expected model and a compat sink for the adopted id.
    #[serde(default)]
    pub lms_main_model: String,
    /// Sole local-server spawning policy for `-l`, `/local`, and `/restart`
    /// (default: off). Discovery always runs first; when no healthy endpoint
    /// is found, only the explicitly selected backend may be spawned. `off`
    /// never authorizes a spawn.
    #[serde(default)]
    pub local_autostart: LocalAutostart,
    /// Default LM Studio inference port (1234; explicit endpoints remain
    /// supported).
    #[serde(default = "default_lms_port")]
    pub lms_port: u16,
    /// Derived runtime identity: "lmstudio" (default) or "higgs".
    /// Discovery or an authorized `localAutostart` spawn sets this tag; it is
    /// not a second spawn authority. The provider layer reads it for
    /// backend-specific behavior such as Higgs session caching and JIT policy.
    #[serde(default = "default_local_backend")]
    pub local_backend: String,
    /// Port for the managed Higgs server (default: 8091).
    #[serde(default = "default_higgs_port")]
    pub higgs_port: u16,
    /// Path to MLX model directory (containing .safetensors + tokenizer.json).
    /// Default: ~/.cache/lm-studio/models/mlx-community/Qwen3.5-2B-MLX-8bit
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mlx_model_dir: Option<String>,
    /// Optional Higgs DFlash/dSpark drafter sidecar path for the main managed
    /// Higgs server. When set, nanobot exports it as HIGGS_DFLASH_PATH before
    /// spawning Higgs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub higgs_draft_model: Option<String>,
    /// Number of draft tokens per speculative decoding step (default: 4).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_draft_tokens: Option<u32>,
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
    /// Maximum number of continuation turns the agent may take after an
    /// initial response (default: 2). Prevents runaway loops when the model
    /// keeps appending rather than finishing.
    #[serde(default = "default_max_continuations")]
    pub max_continuations: u32,
    /// Hard cap on thinking budget tokens for small local models (default: 256).
    #[serde(default = "default_local_thinking_small_model_cap")]
    pub local_thinking_small_model_cap: u32,
    /// Minimum max_tokens when long-mode is active (default: 12288).
    #[serde(default = "default_adaptive_long_mode_min_tokens")]
    pub adaptive_long_mode_min_tokens: u32,
    /// Minimum max_tokens for long-form prompts (default: 6144).
    #[serde(default = "default_adaptive_long_form_min_tokens")]
    pub adaptive_long_form_min_tokens: u32,
    /// Character length threshold above which a prompt is considered long-form (default: 500).
    #[serde(default = "default_adaptive_long_form_trigger_chars")]
    pub adaptive_long_form_trigger_chars: u32,
    /// Maximum max_tokens cap when recent tool calls are heavy (default: 2048).
    #[serde(default = "default_adaptive_tool_heavy_max_tokens")]
    pub adaptive_tool_heavy_max_tokens: u32,
    /// Minimum max_tokens floor when recent tool calls are heavy (default: 1024).
    #[serde(default = "default_adaptive_tool_heavy_min_tokens")]
    pub adaptive_tool_heavy_min_tokens: u32,
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

fn default_constrained_tool_calls() -> bool {
    true
}

fn default_max_tokens() -> u32 {
    4096
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

fn default_max_continuations() -> u32 {
    6
}

fn default_lms_port() -> u16 {
    1234
}

fn default_higgs_port() -> u16 {
    8091
}

fn default_local_model() -> String {
    "gemma-3n-e4b-it-Q4_K_M.gguf".to_string()
}

fn default_local_api_key() -> String {
    "local".to_string()
}

fn default_local_backend() -> String {
    "lmstudio".to_string()
}

/// Whether the derived runtime identity is a managed Higgs server.
/// This classification does not authorize spawning; `localAutostart` does.
pub fn is_higgs_backend(backend: &str) -> bool {
    backend == "higgs"
}

/// Sole local-server spawning policy (`agents.defaults.localAutostart`).
///
/// `Higgs` is the default: an omitted `localAutostart` key spawns the managed
/// Higgs sidecar when discovery finds nothing. Discovery of an already
/// running endpoint always runs first regardless.
/// Unknown config values (typos) still deserialize to `Off` — only a
/// recognized value can enable spawning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum LocalAutostart {
    /// Spawn the managed Higgs sidecar when discovery finds nothing (default).
    #[default]
    Higgs,
    /// Spawn LM Studio when discovery finds nothing.
    Lmstudio,
    /// Never spawn autonomously. `#[serde(other)]` folds unknown config
    /// values here so a typo can't silently enable spawning.
    #[serde(other)]
    Off,
}

fn default_local_thinking_small_model_cap() -> u32 {
    256
}

fn default_adaptive_long_mode_min_tokens() -> u32 {
    12288
}

fn default_adaptive_long_form_min_tokens() -> u32 {
    6144
}

fn default_adaptive_long_form_trigger_chars() -> u32 {
    500
}

fn default_adaptive_tool_heavy_max_tokens() -> u32 {
    2048
}

fn default_adaptive_tool_heavy_min_tokens() -> u32 {
    1024
}

impl Default for AgentDefaults {
    fn default() -> Self {
        Self {
            workspace: default_workspace(),
            model: default_model(),
            local_model: default_local_model(),
            local_api_base: String::new(),
            local_api_key: default_local_api_key(),
            local_max_context_tokens: default_local_max_context_tokens(),
            constrained_tool_calls: default_constrained_tool_calls(),
            max_tokens: default_max_tokens(),
            temperature: default_temperature(),
            repetition_penalty: None,
            frequency_penalty: None,
            presence_penalty: None,
            max_tool_iterations: default_max_tool_iterations(),
            max_context_tokens: default_max_context_tokens(),
            max_concurrent_chats: default_max_concurrent_chats(),
            theme_index: 0,
            max_tool_result_chars: default_max_tool_result_chars(),
            max_continuations: default_max_continuations(),
            lms_main_model: String::new(),
            local_autostart: LocalAutostart::default(),
            lms_port: default_lms_port(),
            higgs_port: default_higgs_port(),
            local_backend: default_local_backend(),
            mlx_model_dir: None,
            higgs_draft_model: None,
            num_draft_tokens: None,
            instructions_path: None,
            skip_jit_gate: false,
            local_thinking_small_model_cap: default_local_thinking_small_model_cap(),
            adaptive_long_mode_min_tokens: default_adaptive_long_mode_min_tokens(),
            adaptive_long_form_min_tokens: default_adaptive_long_form_min_tokens(),
            adaptive_long_form_trigger_chars: default_adaptive_long_form_trigger_chars(),
            adaptive_tool_heavy_max_tokens: default_adaptive_tool_heavy_max_tokens(),
            adaptive_tool_heavy_min_tokens: default_adaptive_tool_heavy_min_tokens(),
        }
    }
}

/// Agent configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentsConfig {
    #[serde(default)]
    pub defaults: AgentDefaults,
    /// Default lane: "answer" or "action". None defaults to action.
    #[serde(default)]
    pub default_lane: Option<String>,
}

/// Token budget tuning for adaptive max_tokens and local-thinking behaviour.
///
/// Extracted from `AgentDefaults` so it can be carried through `SwappableCoreConfig`
/// as a single field without requiring callers to populate 8 separate fields.
#[derive(Debug, Clone)]
pub struct AdaptiveTokenConfig {
    pub local_thinking_small_model_cap: u32,
    pub adaptive_long_mode_min_tokens: u32,
    pub adaptive_long_form_min_tokens: u32,
    pub adaptive_long_form_trigger_chars: u32,
    pub adaptive_tool_heavy_max_tokens: u32,
    pub adaptive_tool_heavy_min_tokens: u32,
}

impl Default for AdaptiveTokenConfig {
    fn default() -> Self {
        Self {
            local_thinking_small_model_cap: 256,
            adaptive_long_mode_min_tokens: 12288,
            adaptive_long_form_min_tokens: 6144,
            adaptive_long_form_trigger_chars: 500,
            adaptive_tool_heavy_max_tokens: 2048,
            adaptive_tool_heavy_min_tokens: 1024,
        }
    }
}

impl AdaptiveTokenConfig {
    /// Build from `AgentDefaults`, keeping all values in sync.
    pub fn from_defaults(d: &AgentDefaults) -> Self {
        Self {
            local_thinking_small_model_cap: d.local_thinking_small_model_cap,
            adaptive_long_mode_min_tokens: d.adaptive_long_mode_min_tokens,
            adaptive_long_form_min_tokens: d.adaptive_long_form_min_tokens,
            adaptive_long_form_trigger_chars: d.adaptive_long_form_trigger_chars,
            adaptive_tool_heavy_max_tokens: d.adaptive_tool_heavy_max_tokens,
            adaptive_tool_heavy_min_tokens: d.adaptive_tool_heavy_min_tokens,
        }
    }
}

// ---------------------------------------------------------------------------
// Provider configs
// ---------------------------------------------------------------------------

/// LLM provider configuration.
#[derive(Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderConfig {
    #[serde(default)]
    pub api_key: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_base: Option<String>,
}

impl fmt::Debug for ProviderConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProviderConfig")
            .field("api_key", &crate::config::redact(&self.api_key))
            .field("api_base", &self.api_base)
            .finish()
    }
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
    ("groq/", |p| &p.groq, GROQ_API_BASE),
    ("gemini/", |p| &p.gemini, GEMINI_API_BASE),
    ("openai/", |p| &p.openai, OPENAI_API_BASE),
    ("anthropic/", |p| &p.anthropic, ANTHROPIC_API_BASE),
    ("deepseek/", |p| &p.deepseek, DEEPSEEK_API_BASE),
    ("huggingface/", |p| &p.huggingface, HUGGINGFACE_API_BASE),
    ("zhipu-coding/", |p| &p.zhipu_coding, ZHIPU_CODING_API_BASE),
    ("zhipu/", |p| &p.zhipu, ZHIPU_API_BASE),
    ("openrouter/", |p| &p.openrouter, OPENROUTER_API_BASE),
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

    /// Strip a known provider prefix from a model name, regardless of whether
    /// that provider has a key configured.  Returns `None` if no prefix matches.
    pub fn strip_known_prefix(model: &str) -> Option<&str> {
        for (prefix, _, _) in PROVIDER_PREFIXES {
            if let Some(rest) = model.strip_prefix(prefix) {
                return Some(rest);
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
#[derive(Clone, Serialize, Deserialize)]
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
    /// Auto-start SearXNG Docker container if not running. Default: true.
    #[serde(default = "default_true")]
    pub auto_start: bool,
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

impl fmt::Debug for WebSearchConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WebSearchConfig")
            .field("api_key", &crate::config::redact(&self.api_key))
            .field("max_results", &self.max_results)
            .field("provider", &self.provider)
            .field("searxng_url", &self.searxng_url)
            .field("auto_start", &self.auto_start)
            .finish()
    }
}

impl Default for WebSearchConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            max_results: default_max_results(),
            provider: default_search_provider(),
            searxng_url: default_searxng_url(),
            auto_start: default_true(),
        }
    }
}

/// Web tools configuration.
/// Web fetch tool configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WebFetchConfig {
    /// Base URL of a local crw-server (fastCRW) used for `/v1/scrape`;
    /// empty string disables crw and web_fetch uses the plain fetcher.
    #[serde(default = "default_crw_url")]
    pub crw_url: String,
    /// Auto-start crw-server at startup when the binary is installed.
    #[serde(default = "default_true")]
    pub auto_start: bool,
}

fn default_crw_url() -> String {
    "http://localhost:3000".to_string()
}

impl Default for WebFetchConfig {
    fn default() -> Self {
        Self {
            crw_url: default_crw_url(),
            auto_start: true,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WebToolsConfig {
    #[serde(default)]
    pub search: WebSearchConfig,
    #[serde(default)]
    pub fetch: WebFetchConfig,
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

/// Cua driver (local desktop computer-use) tool configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CuaToolConfig {
    /// When false, the `cua` tool is not registered.
    #[serde(default = "default_cua_enabled")]
    pub enabled: bool,
    /// Path to the cua-driver binary. `None` resolves `cua-driver` on PATH.
    #[serde(default)]
    pub binary_path: Option<String>,
    /// Daemon permission mode applied at launch: standard | bounded | unrestricted.
    #[serde(default = "default_cua_permission_mode")]
    pub permission_mode: String,
    /// Auto-start the cua-driver daemon when a call finds it not running.
    #[serde(default = "default_cua_daemon_auto_start")]
    pub daemon_auto_start: bool,
    /// Directory for screenshots. `None` defaults to `<workspace>/cua`.
    #[serde(default)]
    pub screenshot_dir: Option<PathBuf>,
}

fn default_cua_enabled() -> bool {
    true
}

fn default_cua_permission_mode() -> String {
    "standard".to_string()
}

fn default_cua_daemon_auto_start() -> bool {
    true
}

impl Default for CuaToolConfig {
    fn default() -> Self {
        Self {
            enabled: default_cua_enabled(),
            binary_path: None,
            permission_mode: default_cua_permission_mode(),
            daemon_auto_start: default_cua_daemon_auto_start(),
            screenshot_dir: None,
        }
    }
}

/// Code execution tool configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CodeExecutionConfig {
    /// When false (default), the execute_code tool is not registered.
    #[serde(default)]
    pub enabled: bool,
    /// Timeout in seconds for each script execution (default: 30).
    #[serde(default = "default_code_timeout")]
    pub timeout: u64,
    /// Maximum number of tool RPC calls a single script may make (default: 20).
    #[serde(default = "default_max_tool_calls")]
    pub max_tool_calls: usize,
}

fn default_code_timeout() -> u64 {
    30
}

fn default_max_tool_calls() -> usize {
    20
}

impl Default for CodeExecutionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            timeout: default_code_timeout(),
            max_tool_calls: default_max_tool_calls(),
        }
    }
}

// ---------------------------------------------------------------------------
// Python kernel config
// ---------------------------------------------------------------------------

/// Stateful Python kernel tool settings.
///
/// Unlike `execute_code` which spawns a fresh python3 per call,
/// the kernel holds a persistent CPython interpreter in-process
/// via PyO3. Variables, imports, and function definitions survive
/// across calls. Feature: `python-kernel`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PythonKernelConfig {
    /// When false (default), the `python` tool is not registered.
    #[serde(default)]
    pub enabled: bool,
    /// Per-call timeout in seconds (default: 30).
    #[serde(default = "default_kernel_timeout")]
    pub timeout: u64,
}

fn default_kernel_timeout() -> u64 {
    30
}

impl Default for PythonKernelConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            timeout: default_kernel_timeout(),
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
    /// Code execution (Python RPC) tool settings.
    #[serde(default)]
    pub code_execution: CodeExecutionConfig,
    /// Cua driver (local desktop computer-use) tool settings.
    #[serde(default)]
    pub cua: CuaToolConfig,
    /// Stateful Python kernel tool (PyO3). Feature: `python-kernel`.
    #[serde(default)]
    pub python_kernel: PythonKernelConfig,
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

fn default_trio_specialist_top_p() -> f64 {
    0.95
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
    /// top_p for the specialist LLM (default: 0.95). Wired into the specialist
    /// call so reasoning models (e.g. VibeThinker: temp 1.0 / top_p 0.95) sample
    /// per their model card instead of hardcoded values.
    #[serde(default = "default_trio_specialist_top_p")]
    pub specialist_top_p: f64,
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
    /// Pollution score threshold to evict a turn (default: 0.6).
    #[serde(default = "default_pollution_threshold")]
    pub pollution_threshold: f32,
    /// Max word count before babble collapse fires (default: 500).
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
    500
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
            specialist_top_p: default_trio_specialist_top_p(),
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

/// Configuration for SQLite working memory and curated cross-session memory.
///
/// LCM updates the concrete session's working-memory row. A background
/// reflector periodically distills completed rows into `MEMORY.md`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryConfig {
    /// Enable/disable working memory and reflection (default: true).
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Model to use for LCM compaction and reflection.
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

    /// Max tokens for working memory (per-session state) in the system prompt (default: 1500).
    #[serde(default = "default_working_memory_budget")]
    pub working_memory_budget: usize,

    /// Token threshold to trigger reflection (default: 5000).
    #[serde(default = "default_reflection_threshold")]
    pub reflection_threshold: usize,

    /// Seconds of inactivity before auto-completing a working memory session (default: 3600).
    #[serde(default = "default_session_complete_after_secs")]
    pub session_complete_after_secs: u64,

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
    /// The agent fetches full skill content on demand via the `get_skills` tool.
    /// This keeps the system prompt lean (RLM pattern: context as variable).
    #[deprecated(
        note = "Superseded by `skill_disclosure` enum; set skill_disclosure = \"compact\" instead"
    )]
    #[serde(default)]
    pub lazy_skills: bool,

    /// Controls how skills are disclosed in the system prompt.
    /// - "compact" (default): one-line index per skill (~20 tokens each)
    /// - "xml": full XML summary with descriptions and metadata (~150 tokens each)
    /// - "eager": full skill content loaded into the system prompt
    /// Overrides `lazy_skills` when set to "eager" (disables lazy loading).
    #[serde(default = "default_skill_disclosure")]
    pub skill_disclosure: String,

    /// Tuning knobs for context hygiene.
    #[serde(default)]
    pub hygiene: ContextHygieneConfig,
}

fn default_true() -> bool {
    true
}

fn default_working_memory_budget() -> usize {
    600
}

fn default_session_complete_after_secs() -> u64 {
    3600
}

/// Shared default for both `max_message_age_turns` and `max_history_turns`.
///
/// Single source of truth (rather than two constants a comment promises to
/// keep in sync): age-based eviction must not rewrite turns older than the
/// history load already drops them at, or trim busts the prefix cache for
/// turns the model never even sees. Keeps many turns append-only so the
/// inference server's prefix cache stays warm across a long session.
/// Capable long-context models (e.g. Qwen3.6, 256K) comfortably hold this.
const DEFAULT_RETENTION_TURNS: usize = 60;

fn default_max_message_age_turns() -> usize {
    DEFAULT_RETENTION_TURNS
}

fn default_max_history_turns() -> usize {
    DEFAULT_RETENTION_TURNS
}

fn default_reflection_threshold() -> usize {
    5000
}

fn default_skill_disclosure() -> String {
    "compact".to_string()
}

#[allow(deprecated)]
impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            enabled: default_true(),
            model: String::new(),
            provider: None,
            working_memory_budget: default_working_memory_budget(),
            reflection_threshold: default_reflection_threshold(),
            session_complete_after_secs: default_session_complete_after_secs(),
            max_message_age_turns: default_max_message_age_turns(),
            max_history_turns: default_max_history_turns(),
            lazy_skills: true,
            skill_disclosure: default_skill_disclosure(),
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
// Router tuning config
// ---------------------------------------------------------------------------

/// Tuning knobs for the LLM-based router decisions and tool result truncation.
/// Nested under `toolDelegation.routerTuning`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct RouterTuningConfig {
    /// Max tokens for a single router LLM call (default: 256).
    pub max_tokens: u32,
    /// Max characters kept from a tool result before injection into the router
    /// context (default: 2400).
    pub max_tool_result_chars: usize,
    /// Max characters per message in the conversation tail passed to the
    /// router (default: 200).
    pub tail_max_msg_chars: usize,
    /// Max total characters for the whole conversation tail (default: 800).
    pub tail_max_chars: usize,
}

fn default_router_max_tokens() -> u32 {
    256
}
fn default_router_max_tool_result_chars() -> usize {
    2400
}
fn default_router_tail_max_msg_chars() -> usize {
    200
}
fn default_router_tail_max_chars() -> usize {
    800
}

impl Default for RouterTuningConfig {
    fn default() -> Self {
        Self {
            max_tokens: default_router_max_tokens(),
            max_tool_result_chars: default_router_max_tool_result_chars(),
            tail_max_msg_chars: default_router_tail_max_msg_chars(),
            tail_max_chars: default_router_tail_max_chars(),
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

/// High-level delegation mode with its strict routing policy attached.
///
/// Use this instead of configuring individual `strict_*` booleans:
/// - **Inline**: Main model calls tools directly (no delegation).
/// - **Delegated**: Tools delegated to a cheaper tool runner model.
/// - **Trio**: Strict separation — main=conversation, router=dispatch, specialist=execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DelegationMode {
    /// Main model calls tools directly (delegation disabled).
    Inline(DelegationStrictPolicy),
    /// Tools delegated to tool runner model (default).
    Delegated(DelegationStrictPolicy),
    /// Strict trio: main=orchestrator, router=dispatch, specialist=tools.
    Trio(DelegationStrictPolicy),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DelegationStrictPolicy {
    strict_no_tools_main: bool,
    strict_router_schema: bool,
    strict_local_only: bool,
    strict_toolplan_validation: bool,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
enum DelegationModeName {
    Inline,
    #[default]
    Delegated,
    Trio,
}

#[derive(Default)]
struct LegacyStrictPolicy {
    strict_no_tools_main: Option<bool>,
    strict_router_schema: Option<bool>,
    strict_local_only: Option<bool>,
    strict_toolplan_validation: Option<bool>,
}

impl Default for DelegationMode {
    fn default() -> Self {
        Self::delegated()
    }
}

impl Serialize for DelegationMode {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.name().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for DelegationMode {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(match DelegationModeName::deserialize(deserializer)? {
            DelegationModeName::Inline => Self::inline(),
            DelegationModeName::Delegated => Self::delegated(),
            DelegationModeName::Trio => Self::trio(),
        })
    }
}

impl DelegationMode {
    pub const fn inline() -> Self {
        Self::Inline(DelegationStrictPolicy::inline())
    }

    pub const fn delegated() -> Self {
        Self::Delegated(DelegationStrictPolicy::delegated())
    }

    pub const fn trio() -> Self {
        Self::Trio(DelegationStrictPolicy::trio())
    }

    pub fn is_inline(&self) -> bool {
        matches!(self, Self::Inline(_))
    }

    pub fn is_delegated(&self) -> bool {
        matches!(self, Self::Delegated(_))
    }

    pub fn is_trio(&self) -> bool {
        matches!(self, Self::Trio(_))
    }

    pub fn strict_no_tools_main(&self) -> bool {
        self.strict_policy().strict_no_tools_main
    }

    pub fn strict_router_schema(&self) -> bool {
        self.strict_policy().strict_router_schema
    }

    pub fn strict_local_only(&self) -> bool {
        self.strict_policy().strict_local_only
    }

    pub fn strict_toolplan_validation(&self) -> bool {
        self.strict_policy().strict_toolplan_validation
    }

    fn name(&self) -> DelegationModeName {
        match self {
            Self::Inline(_) => DelegationModeName::Inline,
            Self::Delegated(_) => DelegationModeName::Delegated,
            Self::Trio(_) => DelegationModeName::Trio,
        }
    }

    fn strict_policy(&self) -> DelegationStrictPolicy {
        match *self {
            Self::Inline(policy) | Self::Delegated(policy) | Self::Trio(policy) => policy,
        }
    }

    fn with_legacy_strict(self, legacy: LegacyStrictPolicy) -> Self {
        let policy = self.strict_policy().with_legacy(legacy);
        match self {
            Self::Inline(_) => Self::Inline(policy),
            Self::Delegated(_) => Self::Delegated(policy),
            Self::Trio(_) => Self::Trio(policy),
        }
    }

    fn without_strict_router(self) -> Self {
        let policy = self.strict_policy().without_strict_router();
        match self {
            Self::Inline(_) => Self::Inline(policy),
            Self::Delegated(_) => Self::Delegated(policy),
            Self::Trio(_) => Self::Trio(policy),
        }
    }
}

impl DelegationStrictPolicy {
    const fn inline() -> Self {
        Self {
            strict_no_tools_main: false,
            strict_router_schema: false,
            strict_local_only: false,
            strict_toolplan_validation: true,
        }
    }

    const fn delegated() -> Self {
        Self::inline()
    }

    const fn trio() -> Self {
        Self {
            strict_no_tools_main: true,
            strict_router_schema: true,
            strict_local_only: false,
            strict_toolplan_validation: true,
        }
    }

    fn with_legacy(self, legacy: LegacyStrictPolicy) -> Self {
        Self {
            strict_no_tools_main: legacy
                .strict_no_tools_main
                .unwrap_or(self.strict_no_tools_main),
            strict_router_schema: legacy
                .strict_router_schema
                .unwrap_or(self.strict_router_schema),
            strict_local_only: legacy.strict_local_only.unwrap_or(self.strict_local_only),
            strict_toolplan_validation: legacy
                .strict_toolplan_validation
                .unwrap_or(self.strict_toolplan_validation),
        }
    }

    fn without_strict_router(self) -> Self {
        Self {
            strict_no_tools_main: false,
            strict_router_schema: false,
            ..self
        }
    }
}

/// Configuration for delegating tool execution loops to a cheaper model.
///
/// When enabled, tool calls from the main LLM are handed off to a lightweight
/// model that executes the tools and interprets their results, conserving the
/// main model's context window for reasoning.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolDelegationConfig {
    /// High-level mode and strict routing policy.
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

    /// When true, build and use role-scoped context packs per turn.
    #[serde(default)]
    pub role_scoped_context_packs: bool,

    /// When true, use deterministic fallback routing when router output is invalid.
    #[serde(default = "default_true")]
    pub deterministic_router_fallback: bool,

    /// Maximum identical tool calls allowed in one turn (dedup guard).
    #[serde(default = "default_td_max_same_tool_call")]
    pub max_same_tool_call_per_turn: u32,

    /// Tuning knobs for subagent execution.
    #[serde(default)]
    pub subagent: SubagentTuning,

    /// Tuning knobs for the LLM-based router (token budgets, context limits).
    #[serde(default)]
    pub router_tuning: RouterTuningConfig,

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
            role_scoped_context_packs: false,
            deterministic_router_fallback: true,
            max_same_tool_call_per_turn: default_td_max_same_tool_call(),
            subagent: SubagentTuning::default(),
            router_tuning: RouterTuningConfig::default(),
            specialist_synthesis: true,
        }
    }
}

#[derive(Deserialize)]
#[serde(default, rename_all = "camelCase")]
struct ToolDelegationConfigWire {
    mode: DelegationMode,
    enabled: bool,
    model: String,
    provider: Option<ProviderConfig>,
    max_iterations: u32,
    max_tokens: u32,
    slim_results: bool,
    max_result_preview_chars: usize,
    auto_local: bool,
    cost_budget: f64,
    default_subagent_model: String,
    strict_no_tools_main: Option<bool>,
    strict_router_schema: Option<bool>,
    role_scoped_context_packs: bool,
    strict_local_only: Option<bool>,
    strict_toolplan_validation: Option<bool>,
    deterministic_router_fallback: bool,
    max_same_tool_call_per_turn: u32,
    subagent: SubagentTuning,
    router_tuning: RouterTuningConfig,
    specialist_synthesis: bool,
}

impl Default for ToolDelegationConfigWire {
    fn default() -> Self {
        let default = ToolDelegationConfig::default();
        Self {
            mode: default.mode,
            enabled: default.enabled,
            model: default.model,
            provider: default.provider,
            max_iterations: default.max_iterations,
            max_tokens: default.max_tokens,
            slim_results: default.slim_results,
            max_result_preview_chars: default.max_result_preview_chars,
            auto_local: default.auto_local,
            cost_budget: default.cost_budget,
            default_subagent_model: default.default_subagent_model,
            strict_no_tools_main: None,
            strict_router_schema: None,
            role_scoped_context_packs: default.role_scoped_context_packs,
            strict_local_only: None,
            strict_toolplan_validation: None,
            deterministic_router_fallback: default.deterministic_router_fallback,
            max_same_tool_call_per_turn: default.max_same_tool_call_per_turn,
            subagent: default.subagent,
            router_tuning: default.router_tuning,
            specialist_synthesis: default.specialist_synthesis,
        }
    }
}

impl<'de> Deserialize<'de> for ToolDelegationConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = ToolDelegationConfigWire::deserialize(deserializer)?;
        let mode = wire.mode.with_legacy_strict(LegacyStrictPolicy {
            strict_no_tools_main: wire.strict_no_tools_main,
            strict_router_schema: wire.strict_router_schema,
            strict_local_only: wire.strict_local_only,
            strict_toolplan_validation: wire.strict_toolplan_validation,
        });

        Ok(Self {
            mode,
            enabled: wire.enabled,
            model: wire.model,
            provider: wire.provider,
            max_iterations: wire.max_iterations,
            max_tokens: wire.max_tokens,
            slim_results: wire.slim_results,
            max_result_preview_chars: wire.max_result_preview_chars,
            auto_local: wire.auto_local,
            cost_budget: wire.cost_budget,
            default_subagent_model: wire.default_subagent_model,
            role_scoped_context_packs: wire.role_scoped_context_packs,
            deterministic_router_fallback: wire.deterministic_router_fallback,
            max_same_tool_call_per_turn: wire.max_same_tool_call_per_turn,
            subagent: wire.subagent,
            router_tuning: wire.router_tuning,
            specialist_synthesis: wire.specialist_synthesis,
        })
    }
}

impl ToolDelegationConfig {
    /// Apply the high-level `mode` to the individual strict flags.
    ///
    /// Call after deserialization to ensure the mode takes effect.
    /// The strict policy itself lives inside `mode`; this only aligns the
    /// non-strict runtime switches that still derive from the preset.
    pub fn apply_mode(&mut self) {
        match self.mode {
            DelegationMode::Inline(_) => {
                self.enabled = false;
                self.role_scoped_context_packs = false;
            }
            DelegationMode::Delegated(_) => {
                self.enabled = true;
            }
            DelegationMode::Trio(_) => {
                self.enabled = true;
                self.role_scoped_context_packs = true;
            }
        }
    }

    pub fn strict_no_tools_main(&self) -> bool {
        self.mode.strict_policy().strict_no_tools_main
    }

    pub fn strict_router_schema(&self) -> bool {
        self.mode.strict_policy().strict_router_schema
    }

    pub fn strict_local_only(&self) -> bool {
        self.mode.strict_policy().strict_local_only
    }

    pub fn strict_toolplan_validation(&self) -> bool {
        self.mode.strict_policy().strict_toolplan_validation
    }

    pub fn clear_strict_router(&mut self) {
        self.mode = self.mode.without_strict_router();
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

/// Configuration for the ensemble proprioception system.
///
/// Controls shared body awareness, heartbeat grounding, and priority
/// interrupts.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProprioceptionConfig {
    /// Enable the proprioception system (default: true).
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Turns between grounding injections. 0 = disabled (default: 8).
    #[serde(default = "default_grounding_interval")]
    pub grounding_interval: u32,

    /// Enable the aha channel for priority interrupts (default: true).
    #[serde(default = "default_true")]
    pub aha_channel: bool,

    /// Enable proactive information retrieval before tool calls (default: true).
    #[serde(default = "default_true")]
    pub proactive_retrieval: bool,
}

impl Default for ProprioceptionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            grounding_interval: default_grounding_interval(),
            aha_channel: true,
            proactive_retrieval: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Idle agency config
// ---------------------------------------------------------------------------

fn default_idle_after_secs() -> u64 {
    900
}

fn default_idle_max_backoff_secs() -> u64 {
    3600
}

fn default_idle_max_turns_per_hour() -> u32 {
    4
}

fn default_idle_write_paths() -> Vec<String> {
    vec!["skills/**".to_string(), "MEMORY.md".to_string()]
}

/// Idle-window agency (v0.5 E1). When the designated session has been quiet
/// past `afterSecs` and the local inference server is already warm, the
/// gateway injects one self-directed turn into the same session loop.
/// Keep `afterSecs` well under `memory.sessionCompleteAfterSecs` (default
/// 3600) or the idle observation lands in a freshly rolled-over session.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IdleConfig {
    /// Enable idle turns (default: false). Gateway mode only.
    #[serde(default)]
    pub enabled: bool,
    /// Quiet time before the first idle turn (default: 900s).
    #[serde(default = "default_idle_after_secs")]
    pub after_secs: u64,
    /// Backoff ceiling between consecutive idle turns (default: 3600s).
    /// Actual wait doubles per consecutive fire and resets on any inbound.
    #[serde(default = "default_idle_max_backoff_secs")]
    pub max_backoff_secs: u64,
    /// Hard cap on fired idle turns per sliding hour (default: 4).
    #[serde(default = "default_idle_max_turns_per_hour")]
    pub max_turns_per_hour: u32,
    /// Designated idle session key ("channel:chat_id"). None = the most
    /// recently active session (default: None).
    #[serde(default)]
    pub session_key: Option<String>,
    /// Write allowlist enforced only during idle turns: workspace-relative
    /// subtree ("skills/**"), workspace-relative exact file ("MEMORY.md"),
    /// or absolute path. File tools deny everything else while idle.
    #[serde(default = "default_idle_write_paths")]
    pub write_paths: Vec<String>,
}

impl Default for IdleConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            after_secs: default_idle_after_secs(),
            max_backoff_secs: default_idle_max_backoff_secs(),
            max_turns_per_hour: default_idle_max_turns_per_hour(),
            session_key: None,
            write_paths: default_idle_write_paths(),
        }
    }
}

// ---------------------------------------------------------------------------
// Voice config
// ---------------------------------------------------------------------------

/// TTS engine selection for voice mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub enum TtsEngineConfig {
    #[default]
    Supertonic,
    /// macOS `say` command — native system TTS (Siri-quality neural voices).
    /// No model is loaded; `speak()` shells out per turn. macOS only.
    Say,
}

/// Configuration for voice mode TTS/STT.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VoiceConfig {
    /// Default language for TTS. `None` means auto-detect per utterance.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// TTS engine selection. Default: "supertonic".
    #[serde(default)]
    pub tts_engine: TtsEngineConfig,
    /// Voice ID for the selected TTS engine.
    /// - Supertonic: "F1", "F2", "M1", "M2", etc.
    /// - Say: macOS voice name, e.g. "Samantha" or "Alice"
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tts_voice: Option<String>,
}

impl Default for VoiceConfig {
    fn default() -> Self {
        Self {
            language: None,
            tts_engine: TtsEngineConfig::default(),
            tts_voice: None,
        }
    }
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

/// Configuration for Lossless Context Management.
///
/// LCM replaces destructive compaction with a dual-state memory:
/// immutable SQLite message rows + active context with hierarchical
/// summaries. Summaries contain pointers back to originals, so the LLM
/// can `lcm_expand` any summary to recover the full messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LcmSchemaConfig {
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
}

impl Default for LcmSchemaConfig {
    fn default() -> Self {
        Self {
            tau_soft: default_lcm_tau_soft(),
            tau_hard: default_lcm_tau_hard(),
            deterministic_target: default_lcm_deterministic_target(),
        }
    }
}

// ---------------------------------------------------------------------------
// Cluster config (distributed inference via Exo / LAN peers)
// ---------------------------------------------------------------------------

fn default_cluster_scan_ports() -> Vec<u16> {
    vec![52415, 1234, 8080, 1337, 18100]
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
    /// Enable mDNS + HTTP probe auto-discovery (default: false).
    ///
    /// When true, every startup scans the local /24 subnet (up to 254 IPs ×
    /// scan_ports, 3s connect timeout each) — tens to hundreds of seconds of
    /// pure overhead for a single-node setup with no inference peers, and it
    /// delays the first turn. Keep false unless you actually run LAN peers;
    /// manual `endpoints` are probed regardless of this flag.
    pub auto_discover: bool,
    /// Manual peer endpoint URLs (e.g. ["http://192.168.1.50:52415"]).
    pub endpoints: Vec<String>,
    /// Ports to scan during HTTP probe discovery.
    /// Defaults cover Exo, LM Studio, llama.cpp, Jan, and common dstack/dFlash servers.
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
            auto_discover: false,
            endpoints: Vec::new(),
            scan_ports: default_cluster_scan_ports(),
            scan_interval_secs: default_cluster_scan_interval(),
            prefer_cluster: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Reasoning engine config
// ---------------------------------------------------------------------------

/// Configuration for the branching reasoning engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReasoningConfig {
    /// Enable the reasoning engine (checkpoint/backtrack/plan tools).
    #[serde(default)]
    pub enabled: bool,

    /// Auto-decompose complex tasks for non-thinking models.
    #[serde(default)]
    pub auto_decompose: bool,

    /// Maximum number of checkpoints on the stack.
    #[serde(default = "default_max_checkpoints")]
    pub max_checkpoints: usize,

    /// Max iterations per plan step before marking it failed.
    #[serde(default = "default_step_budget")]
    pub step_budget: u32,

    /// Automatically checkpoint before exec/write_file tools.
    #[serde(default = "default_auto_checkpoint_before_exec")]
    pub auto_checkpoint_before_exec: bool,
}

fn default_max_checkpoints() -> usize {
    5
}
fn default_step_budget() -> u32 {
    5
}
fn default_auto_checkpoint_before_exec() -> bool {
    true
}

impl Default for ReasoningConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            auto_decompose: true,
            max_checkpoints: default_max_checkpoints(),
            step_budget: default_step_budget(),
            auto_checkpoint_before_exec: default_auto_checkpoint_before_exec(),
        }
    }
}

// ---------------------------------------------------------------------------
// Timeouts config
// ---------------------------------------------------------------------------

/// Timeout settings for various provider and tool operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TimeoutsConfig {
    /// HTTP timeout for LLM chat completion requests (default: 120s).
    #[serde(default = "default_provider_http_secs")]
    pub provider_http_secs: u64,
    /// Probe timeout for LM Studio native API endpoint (default: 2s).
    #[serde(default = "default_lms_native_probe_secs")]
    pub lms_native_probe_secs: u64,
    /// Timeout for loading a model via LM Studio REST API (default: 120s).
    #[serde(default = "default_lms_load_secs")]
    pub lms_load_secs: u64,
    /// Timeout for unloading a model via LM Studio REST API (default: 30s).
    #[serde(default = "default_lms_unload_secs")]
    pub lms_unload_secs: u64,
}

fn default_provider_http_secs() -> u64 {
    120
}
fn default_lms_native_probe_secs() -> u64 {
    2
}
fn default_lms_load_secs() -> u64 {
    120
}
fn default_lms_unload_secs() -> u64 {
    30
}

impl Default for TimeoutsConfig {
    fn default() -> Self {
        Self {
            provider_http_secs: default_provider_http_secs(),
            lms_native_probe_secs: default_lms_native_probe_secs(),
            lms_load_secs: default_lms_load_secs(),
            lms_unload_secs: default_lms_unload_secs(),
        }
    }
}

// ---------------------------------------------------------------------------
// Retry config
// ---------------------------------------------------------------------------

/// Retry backoff settings for provider and JIT operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RetryConfig {
    /// Minimum backoff delay for cloud provider retries (default: 1s).
    #[serde(default = "default_provider_retry_min_secs")]
    pub provider_min_secs: u64,
    /// Maximum backoff delay for cloud provider retries (default: 30s).
    #[serde(default = "default_provider_retry_max_secs")]
    pub provider_max_secs: u64,
    /// Minimum backoff delay for JIT model loading retries (default: 2s).
    #[serde(default = "default_jit_retry_min_secs")]
    pub jit_min_secs: u64,
    /// Maximum backoff delay for JIT model loading retries (default: 8s).
    #[serde(default = "default_jit_retry_max_secs")]
    pub jit_max_secs: u64,
}

fn default_provider_retry_min_secs() -> u64 {
    1
}
fn default_provider_retry_max_secs() -> u64 {
    30
}
fn default_jit_retry_min_secs() -> u64 {
    2
}
fn default_jit_retry_max_secs() -> u64 {
    8
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            provider_min_secs: default_provider_retry_min_secs(),
            provider_max_secs: default_provider_retry_max_secs(),
            jit_min_secs: default_jit_retry_min_secs(),
            jit_max_secs: default_jit_retry_max_secs(),
        }
    }
}

// ---------------------------------------------------------------------------
// Monitoring config
// ---------------------------------------------------------------------------

/// Configuration for health/heartbeat monitoring parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MonitoringConfig {
    /// Number of consecutive health-check failures before a probe is marked
    /// degraded (default: 3).
    #[serde(default = "default_degraded_threshold")]
    pub degraded_threshold: u32,
    /// Timeout in seconds for a single health-check HTTP request (default: 2).
    #[serde(default = "default_health_check_timeout_secs")]
    pub health_check_timeout_secs: u64,
    /// Seconds between health-poll cycles in the watchdog loop (default: 30).
    #[serde(default = "default_health_poll_interval_secs")]
    pub health_poll_interval_secs: u64,
    /// Interval in seconds between tool-heartbeat progress ticks (default: 2).
    #[serde(default = "default_tool_heartbeat_secs")]
    pub tool_heartbeat_secs: u64,
}

fn default_degraded_threshold() -> u32 {
    3
}

fn default_health_check_timeout_secs() -> u64 {
    2
}

fn default_health_poll_interval_secs() -> u64 {
    30
}

fn default_tool_heartbeat_secs() -> u64 {
    2
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            degraded_threshold: default_degraded_threshold(),
            health_check_timeout_secs: default_health_check_timeout_secs(),
            health_poll_interval_secs: default_health_poll_interval_secs(),
            tool_heartbeat_secs: default_tool_heartbeat_secs(),
        }
    }
}

// ---------------------------------------------------------------------------
// Hooks config
// ---------------------------------------------------------------------------

/// Configuration for PreToolUse / PostToolUse hook scripts.
///
/// Hook scripts are shell executables that run before and after every tool call.
/// They receive context via environment variables (`NANOBOT_TOOL_NAME`, etc.).
/// A PreToolUse hook that exits non-zero blocks the tool call.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HooksConfig {
    /// Path to a script run before every tool call. Exit non-zero to block.
    #[serde(default)]
    pub pre_tool_use: Option<String>,
    /// Path to a script run after every tool call (observational only).
    #[serde(default)]
    pub post_tool_use: Option<String>,
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
    pub idle: IdleConfig,
    #[serde(default)]
    pub trio: TrioConfig,
    #[serde(default)]
    pub cluster: ClusterConfig,
    #[serde(default)]
    pub lcm: LcmSchemaConfig,
    #[serde(default)]
    pub model_capabilities:
        HashMap<String, crate::agent::model_capabilities::ModelCapabilitiesOverride>,
    #[serde(default)]
    pub reasoning: ReasoningConfig,
    #[serde(default)]
    pub timeouts: TimeoutsConfig,
    #[serde(default)]
    pub retry: RetryConfig,
    #[serde(default)]
    pub monitoring: MonitoringConfig,
    #[serde(default)]
    pub hooks: HooksConfig,
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

    /// The active provider (name, key) in priority order:
    /// OpenRouter > DeepSeek > Anthropic > OpenAI > Gemini > Zhipu > ZhipuCoding > Groq > vLLM.
    fn active_provider(&self) -> Option<(&'static str, &str)> {
        let candidates = [
            ("openrouter", &self.providers.openrouter.api_key),
            ("deepseek", &self.providers.deepseek.api_key),
            ("anthropic", &self.providers.anthropic.api_key),
            ("openai", &self.providers.openai.api_key),
            ("gemini", &self.providers.gemini.api_key),
            ("zhipu", &self.providers.zhipu.api_key),
            ("zhipu-coding", &self.providers.zhipu_coding.api_key),
            ("groq", &self.providers.groq.api_key),
            ("vllm", &self.providers.vllm.api_key),
        ];
        candidates
            .into_iter()
            .find(|(_, key)| Self::is_provider_key_enabled(key))
            .map(|(name, key)| (name, key.as_str()))
    }

    /// Name of the provider `get_api_key()` resolves to, if any.
    pub fn active_provider_name(&self) -> Option<&'static str> {
        self.active_provider().map(|(name, _)| name)
    }

    /// Get the API key of the active provider (see `active_provider`).
    pub fn get_api_key(&self) -> Option<String> {
        self.active_provider().map(|(_, key)| key.to_string())
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
                    .unwrap_or_else(|| OPENROUTER_API_BASE.to_string()),
            );
        }
        if Self::is_provider_key_enabled(&self.providers.deepseek.api_key) {
            return Some(DEEPSEEK_API_BASE.to_string());
        }
        if Self::is_provider_key_enabled(&self.providers.anthropic.api_key) {
            return Some(ANTHROPIC_API_BASE.to_string());
        }
        if Self::is_provider_key_enabled(&self.providers.openai.api_key) {
            return Some(OPENAI_API_BASE.to_string());
        }
        if Self::is_provider_key_enabled(&self.providers.gemini.api_key) {
            return Some(GEMINI_API_BASE.to_string());
        }
        if Self::is_provider_key_enabled(&self.providers.zhipu.api_key) {
            return Some(
                self.providers
                    .zhipu
                    .api_base
                    .clone()
                    .unwrap_or_else(|| ZHIPU_API_BASE.to_string()),
            );
        }
        if Self::is_provider_key_enabled(&self.providers.zhipu_coding.api_key) {
            return Some(
                self.providers
                    .zhipu_coding
                    .api_base
                    .clone()
                    .unwrap_or_else(|| ZHIPU_CODING_API_BASE.to_string()),
            );
        }
        if Self::is_provider_key_enabled(&self.providers.groq.api_key) {
            return Some(
                self.providers
                    .groq
                    .api_base
                    .clone()
                    .unwrap_or_else(|| GROQ_API_BASE.to_string()),
            );
        }
        if self.providers.vllm.api_base.is_some() {
            return self.providers.vllm.api_base.clone();
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idle_config_defaults_and_parse() {
        let cfg = Config::default();
        assert!(!cfg.idle.enabled);
        assert_eq!(cfg.idle.after_secs, 900);
        assert_eq!(cfg.idle.max_backoff_secs, 3600);
        assert_eq!(cfg.idle.max_turns_per_hour, 4);
        assert_eq!(cfg.idle.session_key, None);
        assert_eq!(
            cfg.idle.write_paths,
            vec!["skills/**".to_string(), "MEMORY.md".to_string()]
        );

        let parsed: Config = serde_json::from_str(
            r#"{"idle": {"enabled": true, "afterSecs": 60, "sessionKey": "telegram:42",
                        "writePaths": ["skills/**", "/abs/path/**"]}}"#,
        )
        .unwrap();
        assert!(parsed.idle.enabled);
        assert_eq!(parsed.idle.after_secs, 60);
        assert_eq!(parsed.idle.session_key.as_deref(), Some("telegram:42"));
        assert_eq!(parsed.idle.max_turns_per_hour, 4, "unspecified keep defaults");
        assert_eq!(parsed.idle.write_paths.len(), 2);

        // Partial old config with no idle block still parses to defaults.
        let bare: Config = serde_json::from_str("{}").unwrap();
        assert!(!bare.idle.enabled);
    }

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
        assert_eq!(cfg.get_api_base(), Some(OPENROUTER_API_BASE.to_string()));
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
        assert_eq!(cfg.get_api_base(), Some(ANTHROPIC_API_BASE.to_string()));
    }

    #[test]
    fn test_api_base_openai() {
        let mut cfg = Config::default();
        cfg.providers.openai.api_key = "sk-key".to_string();
        assert_eq!(cfg.get_api_base(), Some(OPENAI_API_BASE.to_string()));
    }

    #[test]
    fn test_api_base_groq() {
        let mut cfg = Config::default();
        cfg.providers.groq.api_key = "gsk_key".to_string();
        assert_eq!(cfg.get_api_base(), Some(GROQ_API_BASE.to_string()));
    }

    #[test]
    fn test_api_base_deepseek() {
        let mut cfg = Config::default();
        cfg.providers.deepseek.api_key = "sk-ds-key".to_string();
        assert_eq!(cfg.get_api_base(), Some(DEEPSEEK_API_BASE.to_string()));
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
        assert!(!td.strict_no_tools_main());
        assert!(!td.strict_router_schema());
        assert!(!td.role_scoped_context_packs);
        assert!(!td.strict_local_only());
        assert!(td.strict_toolplan_validation());
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
            role_scoped_context_packs: true,
            deterministic_router_fallback: true,
            max_same_tool_call_per_turn: 1,
            specialist_synthesis: true,
            mode: DelegationMode::trio(),
            subagent: SubagentTuning::default(),
            router_tuning: RouterTuningConfig::default(),
        };
        let json = serde_json::to_string(&td).unwrap();
        let td2: ToolDelegationConfig = serde_json::from_str(&json).unwrap();
        assert!(td2.enabled);
        assert_eq!(td2.model, "qwen2-0.5b");
        assert_eq!(td2.max_iterations, 10);
        assert_eq!(td2.max_tokens, 2048);
        assert!(td2.provider.is_some());
        assert!(td2.strict_no_tools_main());
        assert!(td2.strict_router_schema());
        assert!(td2.role_scoped_context_packs);
        assert!(!td2.strict_local_only());
        assert!(td2.strict_toolplan_validation());
        assert!(td2.deterministic_router_fallback);
        assert_eq!(td2.max_same_tool_call_per_turn, 1);
    }

    #[test]
    fn test_tool_delegation_old_shape_strict_flags_deserialize() {
        let json = r#"{
            "enabled": true,
            "mode": "delegated",
            "strictNoToolsMain": true,
            "strictRouterSchema": true,
            "strictLocalOnly": true,
            "strictToolplanValidation": false
        }"#;
        let td: ToolDelegationConfig = serde_json::from_str(json).unwrap();

        assert!(td.enabled);
        assert!(td.mode.is_delegated());
        assert!(td.strict_no_tools_main());
        assert!(td.strict_router_schema());
        assert!(td.strict_local_only());
        assert!(!td.strict_toolplan_validation());
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
        assert_eq!(cfg.get_api_base(), Some(OPENROUTER_API_BASE.to_string()));
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
    fn test_voice_config_tts_engine_default_is_supertonic() {
        let vc = VoiceConfig::default();
        assert_eq!(vc.tts_engine, TtsEngineConfig::Supertonic);
    }

    #[test]
    fn test_voice_config_tts_voice_default_is_none() {
        let vc = VoiceConfig::default();
        assert!(vc.tts_voice.is_none());
    }

    #[test]
    fn test_voice_config_roundtrip() {
        let vc = VoiceConfig {
            language: Some("en".to_string()),
            tts_engine: TtsEngineConfig::Supertonic,
            tts_voice: Some("M2".to_string()),
        };
        let json = serde_json::to_string(&vc).unwrap();
        let vc2: VoiceConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(vc2.language.as_deref(), Some("en"));
        assert_eq!(vc2.tts_engine, TtsEngineConfig::Supertonic);
        assert_eq!(vc2.tts_voice.as_deref(), Some("M2"));
    }

    #[test]
    fn test_tts_engine_config_serialization() {
        assert_eq!(
            serde_json::to_string(&TtsEngineConfig::Supertonic).unwrap(),
            r#""supertonic""#
        );
        assert_eq!(
            serde_json::to_string(&TtsEngineConfig::Say).unwrap(),
            r#""say""#
        );
    }

    #[test]
    fn test_tts_engine_config_deserialization() {
        let supertonic: TtsEngineConfig = serde_json::from_str(r#""supertonic""#).unwrap();
        let say: TtsEngineConfig = serde_json::from_str(r#""say""#).unwrap();
        assert_eq!(supertonic, TtsEngineConfig::Supertonic);
        assert_eq!(say, TtsEngineConfig::Say);
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
        assert_eq!(
            cfg.agents.defaults.local_api_base,
            "http://192.168.1.22:1234/v1"
        );
        assert!(!cfg.agents.defaults.local_api_base.is_empty());
    }

    #[test]
    fn test_local_api_base_empty_by_default() {
        let json = r#"{}"#;
        let cfg: Config = serde_json::from_str(json).unwrap();
        assert!(cfg.agents.defaults.local_api_base.is_empty());
    }

    #[test]
    fn test_higgs_draft_model_deserialization() {
        let json = r#"{"agents":{"defaults":{"higgsDraftModel":"/models/Bonsai-27B-dspark-mlx"}}}"#;
        let cfg: Config = serde_json::from_str(json).unwrap();
        assert_eq!(
            cfg.agents.defaults.higgs_draft_model.as_deref(),
            Some("/models/Bonsai-27B-dspark-mlx")
        );

        let empty: Config = serde_json::from_str(r#"{}"#).unwrap();
        assert!(empty.agents.defaults.higgs_draft_model.is_none());
    }

    #[test]
    fn test_default_lms_port_is_1234() {
        let cfg: Config = serde_json::from_str(r#"{}"#).unwrap();
        assert_eq!(cfg.agents.defaults.lms_port, 1234);
    }

    #[test]
    fn test_local_autostart_defaults_to_higgs() {
        let json = r#"{}"#;
        let cfg: Config = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.agents.defaults.local_autostart, LocalAutostart::Higgs);
    }

    #[test]
    fn test_local_autostart_parses_known_values() {
        let higgs: Config =
            serde_json::from_str(r#"{"agents": {"defaults": {"localAutostart": "higgs"}}}"#)
                .unwrap();
        assert_eq!(higgs.agents.defaults.local_autostart, LocalAutostart::Higgs);

        let lms: Config =
            serde_json::from_str(r#"{"agents": {"defaults": {"localAutostart": "lmstudio"}}}"#)
                .unwrap();
        assert_eq!(
            lms.agents.defaults.local_autostart,
            LocalAutostart::Lmstudio
        );

        let off: Config =
            serde_json::from_str(r#"{"agents": {"defaults": {"localAutostart": "off"}}}"#).unwrap();
        assert_eq!(off.agents.defaults.local_autostart, LocalAutostart::Off);
    }

    #[test]
    fn test_local_autostart_unknown_value_falls_back_to_off() {
        // An unknown value must not brick config loading — and must never
        // silently enable spawning. It degrades to Off.
        let cfg: Config =
            serde_json::from_str(r#"{"agents": {"defaults": {"localAutostart": "omlx"}}}"#)
                .unwrap();
        assert_eq!(cfg.agents.defaults.local_autostart, LocalAutostart::Off);
    }

    #[test]
    fn test_local_autostart_serializes_camel_case() {
        let cfg = Config::default();
        let json = serde_json::to_string(&cfg).unwrap();
        assert!(json.contains(r#""localAutostart":"higgs""#));
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
        assert!(
            cfg.trio.router_endpoint.is_none(),
            "endpoint should be absent when not set"
        );
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
        assert!(
            !json.contains("routerEndpoint"),
            "None endpoints should be skipped"
        );
        assert!(
            !json.contains("specialistEndpoint"),
            "None endpoints should be skipped"
        );
    }

    #[test]
    fn test_lcm_config_defaults() {
        let lcm = LcmSchemaConfig::default();
        assert!((lcm.tau_soft - 0.5).abs() < f64::EPSILON);
        assert!((lcm.tau_hard - 0.85).abs() < f64::EPSILON);
        assert_eq!(lcm.deterministic_target, 512);
    }

    #[test]
    fn test_lcm_config_roundtrip() {
        let mut lcm = LcmSchemaConfig::default();
        lcm.tau_soft = 0.6;
        lcm.tau_hard = 0.9;
        lcm.deterministic_target = 256;
        let json = serde_json::to_string(&lcm).unwrap();
        let lcm2: LcmSchemaConfig = serde_json::from_str(&json).unwrap();
        assert!((lcm2.tau_soft - 0.6).abs() < f64::EPSILON);
        assert!((lcm2.tau_hard - 0.9).abs() < f64::EPSILON);
        assert_eq!(lcm2.deterministic_target, 256);
    }

    #[test]
    fn test_lcm_config_from_root_json() {
        let json = r#"{"lcm": {"tauSoft": 0.7, "tauHard": 0.9}}"#;
        let cfg: Config = serde_json::from_str(json).unwrap();
        assert!((cfg.lcm.tau_soft - 0.7).abs() < f64::EPSILON);
        assert!((cfg.lcm.tau_hard - 0.9).abs() < f64::EPSILON);
        assert_eq!(cfg.lcm.deterministic_target, 512); // default
    }

    #[test]
    fn test_lcm_absent_uses_threshold_defaults() {
        let json = r#"{}"#;
        let cfg: Config = serde_json::from_str(json).unwrap();
        assert!((cfg.lcm.tau_soft - 0.5).abs() < f64::EPSILON);
        assert!((cfg.lcm.tau_hard - 0.85).abs() < f64::EPSILON);
    }

    #[test]
    fn test_context_hygiene_config_defaults() {
        let h = ContextHygieneConfig::default();
        assert_eq!(h.keep_last_messages, 20);
    }

    #[test]
    fn test_memory_config_nested_tuning() {
        let json = r#"{"memory": {"hygiene": {"keepLastMessages": 30}}}"#;
        let cfg: Config = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.memory.hygiene.keep_last_messages, 30);
    }

    #[test]
    fn test_obsolete_lcm_sidecar_config_is_ignored() {
        let json = r#"{
            "lcm": {
                "compactionModelDir": "/models/qwen3-0.6b",
                "compactionPort": 8092,
                "compactionContextSize": 2048
            },
            "agents": {
                "defaults": {
                    "higgsCompactionModelDir": "/models/legacy",
                    "higgsCompactionPort": 8093
                }
            }
        }"#;
        let cfg: Config = serde_json::from_str(json).unwrap();
        let serialized = serde_json::to_string(&cfg).unwrap();
        assert!(
            !serialized.contains("compactionModelDir")
                && !serialized.contains("compactionPort")
                && !serialized.contains("higgsCompactionModelDir")
                && !serialized.contains("higgsCompactionPort")
                && !serialized.contains("compactionContextSize"),
            "obsolete sidecar settings must be accepted as inert input and never serialized"
        );
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

    #[test]
    fn test_default_max_continuations() {
        assert_eq!(default_max_continuations(), 6);
        let cfg = Config::default();
        assert_eq!(cfg.agents.defaults.max_continuations, 6);
    }

    #[test]
    fn test_default_max_tokens_raised_for_local_models() {
        // max_tokens raised from 2048 → 4096 to prevent premature EOS on
        // quantized local models that hit the cap before completing tool calls.
        assert_eq!(default_max_tokens(), 4096);
        let cfg = Config::default();
        assert_eq!(cfg.agents.defaults.max_tokens, 4096);
    }

    #[test]
    fn test_adaptive_tool_heavy_defaults_raised() {
        // Raised from 1024/512 → 2048/1024 to give tool-heavy sessions
        // enough output budget for complete tool-call JSON on small models.
        assert_eq!(default_adaptive_tool_heavy_max_tokens(), 2048);
        assert_eq!(default_adaptive_tool_heavy_min_tokens(), 1024);
        let cfg = Config::default();
        assert_eq!(cfg.agents.defaults.adaptive_tool_heavy_max_tokens, 2048);
        assert_eq!(cfg.agents.defaults.adaptive_tool_heavy_min_tokens, 1024);
    }

    #[test]
    fn test_adaptive_long_defaults_cover_rich_local_artifacts() {
        // Qwen3.6 generated a polished single-file Tetris game at ~16.5KB,
        // which required 6039 completion tokens for the write_file payload.
        // Keep the rich-artifact/long-mode floor comfortably above that.
        assert_eq!(default_adaptive_long_form_min_tokens(), 6144);
        assert_eq!(default_adaptive_long_mode_min_tokens(), 12288);
        let cfg = Config::default();
        assert_eq!(cfg.agents.defaults.adaptive_long_form_min_tokens, 6144);
        assert_eq!(cfg.agents.defaults.adaptive_long_mode_min_tokens, 12288);
    }

    #[test]
    fn test_adaptive_token_config_from_defaults_consistency() {
        // AdaptiveTokenConfig should faithfully reflect AgentDefaults values.
        let cfg = Config::default();
        let atc = AdaptiveTokenConfig::from_defaults(&cfg.agents.defaults);
        assert_eq!(
            atc.adaptive_tool_heavy_max_tokens,
            cfg.agents.defaults.adaptive_tool_heavy_max_tokens
        );
        assert_eq!(
            atc.adaptive_tool_heavy_min_tokens,
            cfg.agents.defaults.adaptive_tool_heavy_min_tokens
        );
    }

    #[test]
    fn test_debug_redacts_credentials() {
        let mut cfg = Config::default();
        let fake_key = "sk-super-secret-key-12345";
        cfg.providers.openrouter.api_key = fake_key.to_string();
        cfg.providers.anthropic.api_key = "anthropic-secret".to_string();
        cfg.channels.telegram.token = "bot-token-secret".to_string();
        cfg.channels.email.password = "email-password".to_string();
        cfg.tools.web.search.api_key = "search-key".to_string();

        let debug_output = format!("{:?}", cfg);

        // None of the secret values should appear in Debug output
        assert!(
            !debug_output.contains(fake_key),
            "openrouter api_key leaked"
        );
        assert!(
            !debug_output.contains("anthropic-secret"),
            "anthropic api_key leaked"
        );
        assert!(
            !debug_output.contains("bot-token-secret"),
            "telegram token leaked"
        );
        assert!(
            !debug_output.contains("email-password"),
            "email password leaked"
        );
        assert!(
            !debug_output.contains("search-key"),
            "web search api_key leaked"
        );

        // Redaction markers should be present
        assert!(
            debug_output.contains("[REDACTED"),
            "missing redaction markers"
        );
    }

    #[test]
    fn test_cua_config_roundtrip() {
        // Defaults.
        let default = CuaToolConfig::default();
        assert!(default.enabled);
        assert_eq!(default.permission_mode, "standard");
        assert!(default.daemon_auto_start);
        assert_eq!(default.binary_path, None);
        assert_eq!(default.screenshot_dir, None);

        // Explicit JSON (camelCase) parses.
        let json = r#"{
            "tools": {
                "cua": {
                    "enabled": false,
                    "binaryPath": "/opt/bin/cua-driver",
                    "permissionMode": "bounded",
                    "daemonAutoStart": false,
                    "screenshotDir": "/tmp/shots"
                }
            }
        }"#;
        let cfg: Config = serde_json::from_str(json).unwrap();
        let cua = &cfg.tools.cua;
        assert!(!cua.enabled);
        assert_eq!(cua.binary_path.as_deref(), Some("/opt/bin/cua-driver"));
        assert_eq!(cua.permission_mode, "bounded");
        assert!(!cua.daemon_auto_start);
        assert_eq!(
            cua.screenshot_dir.as_deref(),
            Some(std::path::Path::new("/tmp/shots"))
        );

        // Missing block falls back to defaults.
        let cfg2: Config = serde_json::from_str(r#"{"tools": {}}"#).unwrap();
        assert!(cfg2.tools.cua.enabled);
        assert!(cfg2.tools.cua.daemon_auto_start);
    }
}
