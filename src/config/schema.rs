//! Configuration schema for nanoclaw.
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
    #[serde(default = "default_whatsapp_bridge_url")]
    pub bridge_url: String,
    #[serde(default)]
    pub allow_from: Vec<String>,
}

fn default_whatsapp_bridge_url() -> String {
    "ws://localhost:3001".to_string()
}

impl Default for WhatsAppConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            bridge_url: default_whatsapp_bridge_url(),
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
}

fn default_workspace() -> String {
    "~/.nanoclaw/workspace".to_string()
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

impl Default for AgentDefaults {
    fn default() -> Self {
        Self {
            workspace: default_workspace(),
            model: default_model(),
            max_tokens: default_max_tokens(),
            temperature: default_temperature(),
            max_tool_iterations: default_max_tool_iterations(),
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
// Root config
// ---------------------------------------------------------------------------

/// Root configuration for nanoclaw.
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
            return self.providers.zhipu.api_base.clone();
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
        assert!(ws.ends_with(".nanoclaw/workspace"));
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
}
