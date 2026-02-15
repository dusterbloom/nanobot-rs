//! Configuration schema for nanobot.
//!
//! All structs use `#[serde(rename_all = "camelCase")]` so that the JSON config
//! file can use camelCase keys while Rust code uses snake_case fields.

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
}

fn default_workspace() -> String {
    "~/.nanobot/workspace".to_string()
}

fn default_model() -> String {
    "anthropic/claude-opus-4-5".to_string()
}

fn default_max_tokens() -> u32 {
    8192
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

fn default_max_tool_result_chars() -> usize {
    30000
}

impl Default for AgentDefaults {
    fn default() -> Self {
        Self {
            workspace: default_workspace(),
            model: default_model(),
            max_tokens: default_max_tokens(),
            temperature: default_temperature(),
            max_tool_iterations: default_max_tool_iterations(),
            max_context_tokens: default_max_context_tokens(),
            max_concurrent_chats: default_max_concurrent_chats(),
            max_tool_result_chars: default_max_tool_result_chars(),
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
    ("gemini/", |p| &p.gemini, "https://generativelanguage.googleapis.com/v1beta/openai"),
    ("openai/", |p| &p.openai, "https://api.openai.com/v1"),
    ("anthropic/", |p| &p.anthropic, "https://api.anthropic.com/v1"),
    ("deepseek/", |p| &p.deepseek, "https://api.deepseek.com"),
    ("huggingface/", |p| &p.huggingface, "https://router.huggingface.co/v1"),
    ("zhipu/", |p| &p.zhipu, "https://api.z.ai/api/paas/v4"),
    ("openrouter/", |p| &p.openrouter, "https://openrouter.ai/api/v1"),
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
                    return Some((
                        cfg.api_key.clone(),
                        base.to_string(),
                        rest.to_string(),
                    ));
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
}

fn default_max_results() -> u32 {
    5
}

impl Default for WebSearchConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            max_results: default_max_results(),
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
// Memory config
// ---------------------------------------------------------------------------

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
    /// If empty, falls back to the main agent model.
    /// Recommended: a small, fast model like "google/gemini-2.5-flash".
    #[serde(default)]
    pub model: String,

    /// Optional separate provider for memory operations.
    /// If not set, reuses the main agent's provider.
    /// Allows pointing memory at a local llama.cpp or cheap cloud API.
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
}

fn default_true() -> bool {
    true
}

fn default_observation_budget() -> usize {
    2000
}

fn default_working_memory_budget() -> usize {
    1500
}

fn default_session_complete_after_secs() -> u64 {
    3600
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
            compaction_threshold_percent: default_compaction_threshold_percent(),
            compaction_threshold_tokens: default_compaction_threshold_tokens(),
            max_message_age_turns: default_max_message_age_turns(),
            max_history_turns: default_max_history_turns(),
            lazy_skills: false,
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
// Tool delegation config
// ---------------------------------------------------------------------------

fn default_td_max_iterations() -> u32 {
    10
}

fn default_td_max_tokens() -> u32 {
    4096
}

/// Configuration for delegating tool execution loops to a cheaper model.
///
/// When enabled, tool calls from the main LLM are handed off to a lightweight
/// model that executes the tools and interprets their results, conserving the
/// main model's context window for reasoning.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolDelegationConfig {
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
}

fn default_td_preview_chars() -> usize {
    200
}

impl Default for ToolDelegationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            model: String::new(),
            provider: None,
            max_iterations: default_td_max_iterations(),
            max_tokens: default_td_max_tokens(),
            slim_results: true,
            max_result_preview_chars: default_td_preview_chars(),
            auto_local: true,
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
}

impl Config {
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
            &self.providers.groq.api_key,
            &self.providers.vllm.api_key,
        ];
        for key in candidates {
            if !key.is_empty() {
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
        if !self.providers.openrouter.api_key.is_empty() {
            return Some(
                self.providers
                    .openrouter
                    .api_base
                    .clone()
                    .unwrap_or_else(|| "https://openrouter.ai/api/v1".to_string()),
            );
        }
        if !self.providers.deepseek.api_key.is_empty() {
            return Some("https://api.deepseek.com".to_string());
        }
        if !self.providers.anthropic.api_key.is_empty() {
            return Some("https://api.anthropic.com/v1".to_string());
        }
        if !self.providers.openai.api_key.is_empty() {
            return Some("https://api.openai.com/v1".to_string());
        }
        if !self.providers.gemini.api_key.is_empty() {
            return Some("https://generativelanguage.googleapis.com/v1beta/openai".to_string());
        }
        if !self.providers.zhipu.api_key.is_empty() {
            return Some(
                self.providers
                    .zhipu
                    .api_base
                    .clone()
                    .unwrap_or_else(|| "https://api.z.ai/api/paas/v4".to_string()),
            );
        }
        if !self.providers.groq.api_key.is_empty() {
            return Some("https://api.groq.com/openai/v1".to_string());
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
    fn test_tool_delegation_config_defaults() {
        let td = ToolDelegationConfig::default();
        assert!(td.enabled);
        assert!(td.model.is_empty());
        assert!(td.provider.is_none());
        assert_eq!(td.max_iterations, 10);
        assert_eq!(td.max_tokens, 4096);
        assert!(td.slim_results);
        assert_eq!(td.max_result_preview_chars, 200);
        assert!(td.auto_local);
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
        };
        let json = serde_json::to_string(&td).unwrap();
        let td2: ToolDelegationConfig = serde_json::from_str(&json).unwrap();
        assert!(td2.enabled);
        assert_eq!(td2.model, "qwen2-0.5b");
        assert_eq!(td2.max_iterations, 10);
        assert_eq!(td2.max_tokens, 2048);
        assert!(td2.provider.is_some());
    }

    #[test]
    fn test_tool_delegation_config_in_root() {
        let json = r#"{"toolDelegation": {"enabled": true, "model": "small-model", "maxIterations": 5}}"#;
        let cfg: Config = serde_json::from_str(json).unwrap();
        assert!(cfg.tool_delegation.enabled);
        assert_eq!(cfg.tool_delegation.model, "small-model");
        assert_eq!(cfg.tool_delegation.max_iterations, 5);
        assert_eq!(cfg.tool_delegation.max_tokens, 4096); // default
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
        assert!(td.auto_local, "auto_local should default to true when absent");
    }

    #[test]
    fn test_auto_local_explicit_false() {
        let json = r#"{"enabled": true, "autoLocal": false}"#;
        let td: ToolDelegationConfig = serde_json::from_str(json).unwrap();
        assert!(!td.auto_local, "auto_local should be false when explicitly set");
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
        assert!(!td2.auto_local, "Roundtrip should preserve auto_local=false");
    }
}
