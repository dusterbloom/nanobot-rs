//! Environment scanner for detecting available tools and capabilities.
//!
//! Probes the system PATH for common development tools, checks runtime versions,
//! and detects API keys from environment variables. Results are cached to avoid
//! re-scanning on every startup.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::utils::helpers::ensure_dir;

/// Known tools to probe for.
const KNOWN_TOOLS: &[(&str, &str)] = &[
    ("git", "--version"),
    ("python3", "--version"),
    ("python", "--version"),
    ("node", "--version"),
    ("npm", "--version"),
    ("cargo", "--version"),
    ("rustc", "--version"),
    ("docker", "--version"),
    ("docker-compose", "--version"),
    ("kubectl", "version --client --short"),
    ("go", "version"),
    ("java", "--version"),
    ("gcc", "--version"),
    ("make", "--version"),
    ("cmake", "--version"),
    ("sqlite3", "--version"),
    ("ffmpeg", "-version"),
    ("curl", "--version"),
    ("wget", "--version"),
];

/// Known environment variable prefixes that indicate API keys.
const API_KEY_PATTERNS: &[&str] = &[
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENROUTER_API_KEY",
    "DEEPSEEK_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "BRAVE_API_KEY",
    "ZHIPU_API_KEY",
    "GROQ_API_KEY",
    "GITHUB_TOKEN",
    "DOCKER_TOKEN",
    "AWS_ACCESS_KEY_ID",
    "TELEGRAM_BOT_TOKEN",
];

/// Detected tool with version.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedTool {
    pub name: String,
    pub version: String,
    pub path: Option<String>,
}

/// Available API keys (names only, never values).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedApiKey {
    pub name: String,
    pub present: bool,
}

/// Full environment capabilities report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentCapabilities {
    pub tools: Vec<DetectedTool>,
    pub api_keys: Vec<DetectedApiKey>,
    pub os: String,
    pub shell: String,
    pub scanned_at: String,
}

/// Scans the environment for available tools and capabilities.
pub struct EnvironmentScanner {
    cache_path: PathBuf,
}

impl EnvironmentScanner {
    pub fn new(workspace: &Path) -> Self {
        let cache_dir = workspace.join("cache");
        Self {
            cache_path: cache_dir.join("environment.json"),
        }
    }

    /// Run a full environment scan.
    pub fn scan(&self) -> EnvironmentCapabilities {
        let mut tools = Vec::new();

        for (name, version_flag) in KNOWN_TOOLS {
            let args: Vec<&str> = version_flag.split_whitespace().collect();
            if let Ok(output) = Command::new(name).args(&args).output() {
                if output.status.success() {
                    let version = String::from_utf8_lossy(&output.stdout)
                        .lines()
                        .next()
                        .unwrap_or("")
                        .trim()
                        .to_string();

                    let path = Command::new("which")
                        .arg(name)
                        .output()
                        .ok()
                        .and_then(|o| {
                            if o.status.success() {
                                Some(
                                    String::from_utf8_lossy(&o.stdout)
                                        .trim()
                                        .to_string(),
                                )
                            } else {
                                None
                            }
                        });

                    debug!("Detected tool: {} ({})", name, version);
                    tools.push(DetectedTool {
                        name: name.to_string(),
                        version,
                        path,
                    });
                }
            }
        }

        let mut api_keys = Vec::new();
        for key_name in API_KEY_PATTERNS {
            let present = std::env::var(key_name).is_ok();
            api_keys.push(DetectedApiKey {
                name: key_name.to_string(),
                present,
            });
        }

        let os = std::env::consts::OS.to_string();
        let shell = std::env::var("SHELL").unwrap_or_else(|_| "unknown".to_string());
        let scanned_at = chrono::Utc::now().to_rfc3339();

        let caps = EnvironmentCapabilities {
            tools,
            api_keys,
            os,
            shell,
            scanned_at,
        };

        // Cache the results.
        self.save_cache(&caps);

        caps
    }

    /// Load cached capabilities (if fresh enough).
    pub fn load_cached(&self) -> Option<EnvironmentCapabilities> {
        if !self.cache_path.is_file() {
            return None;
        }

        let content = fs::read_to_string(&self.cache_path).ok()?;
        let caps: EnvironmentCapabilities = serde_json::from_str(&content).ok()?;

        // Consider cache fresh if less than 1 hour old.
        if let Ok(scanned) = chrono::DateTime::parse_from_rfc3339(&caps.scanned_at) {
            let age = chrono::Utc::now() - scanned.with_timezone(&chrono::Utc);
            if age.num_hours() < 1 {
                return Some(caps);
            }
        }

        None
    }

    /// Get capabilities (cached or fresh scan).
    pub fn get_capabilities(&self) -> EnvironmentCapabilities {
        if let Some(cached) = self.load_cached() {
            return cached;
        }
        self.scan()
    }

    /// Format capabilities as a concise context section for the system prompt.
    pub fn format_for_context(caps: &EnvironmentCapabilities) -> String {
        let mut parts = Vec::new();

        // Available tools.
        let tool_names: Vec<&str> = caps.tools.iter().map(|t| t.name.as_str()).collect();
        if !tool_names.is_empty() {
            parts.push(format!("Available tools: {}", tool_names.join(", ")));
        }

        // Available API keys.
        let available_keys: Vec<&str> = caps
            .api_keys
            .iter()
            .filter(|k| k.present)
            .map(|k| k.name.as_str())
            .collect();
        if !available_keys.is_empty() {
            parts.push(format!("API keys configured: {}", available_keys.join(", ")));
        }

        parts.push(format!("OS: {}, Shell: {}", caps.os, caps.shell));

        parts.join("\n")
    }

    fn save_cache(&self, caps: &EnvironmentCapabilities) {
        if let Some(parent) = self.cache_path.parent() {
            ensure_dir(parent);
        }
        if let Ok(json) = serde_json::to_string_pretty(caps) {
            if let Err(e) = fs::write(&self.cache_path, json) {
                debug!("Failed to cache environment: {}", e);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scan_finds_basic_tools() {
        let dir = tempfile::tempdir().unwrap();
        let scanner = EnvironmentScanner::new(dir.path());
        let caps = scanner.scan();

        // At minimum, we should find some tools on any dev machine.
        // Git is almost universally available.
        assert!(!caps.os.is_empty());
    }

    #[test]
    fn test_cache_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let scanner = EnvironmentScanner::new(dir.path());
        let caps = scanner.scan();

        // Save should have happened during scan.
        let loaded = scanner.load_cached();
        assert!(loaded.is_some());
        let loaded = loaded.unwrap();
        assert_eq!(loaded.os, caps.os);
    }

    #[test]
    fn test_format_for_context() {
        let caps = EnvironmentCapabilities {
            tools: vec![
                DetectedTool {
                    name: "git".to_string(),
                    version: "git version 2.40".to_string(),
                    path: Some("/usr/bin/git".to_string()),
                },
                DetectedTool {
                    name: "cargo".to_string(),
                    version: "cargo 1.75".to_string(),
                    path: None,
                },
            ],
            api_keys: vec![
                DetectedApiKey {
                    name: "OPENAI_API_KEY".to_string(),
                    present: true,
                },
                DetectedApiKey {
                    name: "ANTHROPIC_API_KEY".to_string(),
                    present: false,
                },
            ],
            os: "linux".to_string(),
            shell: "/bin/bash".to_string(),
            scanned_at: chrono::Utc::now().to_rfc3339(),
        };

        let ctx = EnvironmentScanner::format_for_context(&caps);
        assert!(ctx.contains("git"));
        assert!(ctx.contains("cargo"));
        assert!(ctx.contains("OPENAI_API_KEY"));
        assert!(!ctx.contains("ANTHROPIC_API_KEY")); // Not present.
    }

    #[test]
    fn test_get_capabilities_uses_cache() {
        let dir = tempfile::tempdir().unwrap();
        let scanner = EnvironmentScanner::new(dir.path());

        // First call scans.
        let caps1 = scanner.get_capabilities();
        // Second call should use cache.
        let caps2 = scanner.get_capabilities();
        assert_eq!(caps1.scanned_at, caps2.scanned_at);
    }
}
