//! Configuration loading and saving utilities.

use std::fs;
use std::path::{Path, PathBuf};

use tracing::warn;

use crate::config::schema::Config;
use crate::utils::helpers::get_data_path;

/// Get the default configuration file path (`~/.nanobot/config.json`).
pub fn get_config_path() -> PathBuf {
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    home.join(".nanobot").join("config.json")
}

/// Get the nanobot data directory (delegates to `utils::helpers::get_data_path`).
pub fn get_data_dir() -> PathBuf {
    get_data_path()
}

/// Strip `//` line comments and `/* … */` block comments from JSONC text,
/// leaving string literals untouched (so a `"http://…"` URL or a value like
/// `"a//b"` survives). The config is hand-edited and users naturally annotate
/// it; `serde_json` rejects comments, and a rejected config silently falls back
/// to defaults — discarding every setting. Pre-stripping keeps comments working.
fn strip_jsonc_comments(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();
    let mut in_string = false;
    let mut escaped = false;
    while let Some(c) = chars.next() {
        if in_string {
            out.push(c);
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        match c {
            '"' => {
                in_string = true;
                out.push(c);
            }
            '/' if chars.peek() == Some(&'/') => {
                // Line comment: drop through end of line (keep the newline).
                while let Some(&n) = chars.peek() {
                    if n == '\n' {
                        break;
                    }
                    chars.next();
                }
            }
            '/' if chars.peek() == Some(&'*') => {
                // Block comment: drop through the closing `*/`.
                chars.next();
                let mut prev = '\0';
                for n in chars.by_ref() {
                    if prev == '*' && n == '/' {
                        break;
                    }
                    prev = n;
                }
            }
            _ => out.push(c),
        }
    }
    out
}

/// Load configuration from a file, or return a default [`Config`] if the file
/// does not exist or cannot be parsed.
///
/// If `config_path` is `None`, the default path (`~/.nanobot/config.json`) is
/// used.
pub fn load_config(config_path: Option<&Path>) -> Config {
    let path = match config_path {
        Some(p) => p.to_path_buf(),
        None => get_config_path(),
    };

    if path.exists() {
        match fs::read_to_string(&path) {
            Ok(contents) => {
                match serde_json::from_str::<Config>(&strip_jsonc_comments(&contents)) {
                    Ok(mut cfg) => {
                        cfg.tool_delegation.apply_mode();
                        crate::agent::model_capabilities::set_global_overrides(
                            cfg.model_capabilities.clone(),
                        );
                        return cfg;
                    }
                    Err(e) => {
                        // A config that EXISTS but fails to parse means every user
                        // setting (provider keys, model, localBackend, voice) is about
                        // to be replaced by defaults — e.g. silently routing to a cloud
                        // provider the user never selected. The tracing warn! is
                        // invisible under the TUI, so also shout to stderr (pre-TUI).
                        warn!(
                            "Failed to parse config from {}: {}. Using default configuration.",
                            path.display(),
                            e
                        );
                        eprintln!(
                        "\x1b[31mnanobot: config at {} is invalid ({}).\n  Running with DEFAULTS — your model, localBackend, and keys are NOT applied.\x1b[0m",
                        path.display(),
                        e
                    );
                    }
                }
            }
            Err(e) => {
                warn!(
                    "Failed to read config from {}: {}. Using default configuration.",
                    path.display(),
                    e
                );
            }
        }
    }

    Config::default()
}

/// Save configuration to a JSON file.
///
/// If `config_path` is `None`, the default path (`~/.nanobot/config.json`) is
/// used. Parent directories are created if they don't exist.
pub fn save_config(config: &Config, config_path: Option<&Path>) {
    let path = match config_path {
        Some(p) => p.to_path_buf(),
        None => get_config_path(),
    };

    // Ensure parent directory exists.
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }

    match serde_json::to_string_pretty(config) {
        Ok(json) => {
            if let Err(e) = fs::write(&path, json) {
                warn!("Failed to write config to {}: {}", path.display(), e);
            }
        }
        Err(e) => {
            warn!("Failed to serialize config: {}", e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_nonexistent_returns_default() {
        let path = Path::new("/tmp/nanobot_test_does_not_exist_987654.json");
        let cfg = load_config(Some(path));
        assert_eq!(cfg.gateway.port, 18790);
    }

    #[test]
    fn strip_jsonc_strips_comments_but_preserves_strings() {
        // `//` and `/* */` are stripped, but `//` inside string values — URLs,
        // and a literal "a//b" — must survive. This is the regression that made
        // a commented config silently fall back to defaults (higgs ignored).
        let src = r#"{
            // leading line comment
            "url": "http://127.0.0.1:8000/v1", // trailing comment
            /* block
               comment */
            "name": "a//b"
        }"#;
        let v: serde_json::Value =
            serde_json::from_str(&strip_jsonc_comments(src)).expect("JSONC must parse after strip");
        assert_eq!(v["url"], "http://127.0.0.1:8000/v1");
        assert_eq!(v["name"], "a//b");
    }

    #[test]
    fn test_load_and_save_roundtrip() {
        let dir = std::env::temp_dir().join("nanobot_test_loader");
        let _ = fs::create_dir_all(&dir);
        let tmp_path = dir.join("config_roundtrip.json");

        let cfg = Config::default();
        save_config(&cfg, Some(&tmp_path));

        let loaded = load_config(Some(&tmp_path));
        assert_eq!(loaded.agents.defaults.model, cfg.agents.defaults.model);
        assert_eq!(loaded.gateway.port, cfg.gateway.port);

        // Clean up.
        let _ = fs::remove_file(&tmp_path);
        let _ = fs::remove_dir(&dir);
    }
}
