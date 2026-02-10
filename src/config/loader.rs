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
            Ok(contents) => match serde_json::from_str::<Config>(&contents) {
                Ok(cfg) => return cfg,
                Err(e) => {
                    warn!(
                        "Failed to parse config from {}: {}. Using default configuration.",
                        path.display(),
                        e
                    );
                }
            },
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
