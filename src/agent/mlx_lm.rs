//! Managed mlx-lm server subprocess for MLX model inference.
//!
//! Spawns `python3 -m mlx_lm.server` as a child process, manages its lifecycle,
//! and supports hot model switching by killing and respawning with a new model.

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// Managed mlx-lm server process.
pub struct MlxLmServer {
    child: Option<Child>,
    pub port: u16,
    pub model_dir: PathBuf,
    pub adapter_path: Option<PathBuf>,
}

impl MlxLmServer {
    /// Start mlx-lm server for the given model.
    pub fn start(model_dir: PathBuf, adapter_path: Option<PathBuf>, port: u16) -> Result<Self, String> {
        let mut server = MlxLmServer {
            child: None,
            port,
            model_dir: model_dir.clone(),
            adapter_path: adapter_path.clone(),
        };
        server.spawn_process()?;
        Ok(server)
    }

    fn spawn_process(&mut self) -> Result<(), String> {
        let mut cmd = Command::new("python3");
        cmd.args(["-m", "mlx_lm.server"])
            .arg("--model").arg(&self.model_dir)
            .arg("--port").arg(self.port.to_string())
            .arg("--host").arg("127.0.0.1");

        if let Some(ref adapter) = self.adapter_path {
            if adapter.join("adapters.safetensors").exists() {
                cmd.arg("--adapter-path").arg(adapter);
            }
        }

        cmd.stdout(Stdio::null()).stderr(Stdio::piped());

        let child = cmd.spawn().map_err(|e| format!("failed to spawn mlx_lm.server: {e}"))?;
        self.child = Some(child);

        // Wait for server to be ready (poll /health or /v1/models)
        let deadline = Instant::now() + Duration::from_secs(60);
        let url = format!("http://127.0.0.1:{}/v1/models", self.port);
        while Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(500));
            // Check if process died
            if let Some(ref mut c) = self.child {
                if let Ok(Some(status)) = c.try_wait() {
                    return Err(format!("mlx_lm.server exited with {status}"));
                }
            }
            if let Ok(resp) = reqwest::blocking::get(&url) {
                if resp.status().is_success() {
                    return Ok(());
                }
            }
        }
        self.kill();
        Err("mlx_lm.server failed to start within 60s".into())
    }

    /// Switch to a different model (kills and respawns).
    pub fn switch_model(&mut self, model_dir: PathBuf, adapter_path: Option<PathBuf>) -> Result<(), String> {
        self.kill();
        self.model_dir = model_dir;
        self.adapter_path = adapter_path;
        self.spawn_process()
    }

    /// Reload adapters by restarting with the same model but updated adapter path.
    pub fn reload_adapters(&mut self, adapter_path: PathBuf) -> Result<(), String> {
        let model = self.model_dir.clone();
        self.switch_model(model, Some(adapter_path))
    }

    pub fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    pub fn kill(&mut self) {
        if let Some(ref mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }

    pub fn is_running(&mut self) -> bool {
        if let Some(ref mut c) = self.child {
            matches!(c.try_wait(), Ok(None))
        } else {
            false
        }
    }
}

impl Drop for MlxLmServer {
    fn drop(&mut self) {
        self.kill();
    }
}

// ---------------------------------------------------------------------------
// Model discovery
// ---------------------------------------------------------------------------

/// An MLX model found on disk.
#[derive(Debug, Clone)]
pub struct MlxModelInfo {
    pub path: PathBuf,
    pub name: String,
    pub has_adapters: bool,
}

/// Scan for MLX model directories under `~/.cache/lm-studio/models/`.
///
/// An MLX model dir contains `*.safetensors` + `tokenizer.json` (no `.gguf`).
pub fn discover_mlx_models() -> Vec<MlxModelInfo> {
    let base = match dirs::home_dir() {
        Some(h) => h.join(".cache/lm-studio/models"),
        None => return vec![],
    };

    let mut models = Vec::new();
    discover_recursive(&base, &mut models);
    models.sort_by(|a, b| a.name.cmp(&b.name));
    models
}

fn discover_recursive(dir: &Path, out: &mut Vec<MlxModelInfo>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }

        // Check if this dir is an MLX model (has safetensors + tokenizer.json, no .gguf)
        if is_mlx_model_dir(&path) {
            let name = path.file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            let adapter_dir = path.join("adapters");
            let has_adapters = adapter_dir.join("adapters.safetensors").exists();
            out.push(MlxModelInfo { path, name, has_adapters });
        } else {
            // Recurse into org dirs (mlx-community/, lmstudio-community/, etc.)
            discover_recursive(&path, out);
        }
    }
}

fn is_mlx_model_dir(dir: &Path) -> bool {
    // Must have tokenizer.json
    if !dir.join("tokenizer.json").exists() {
        return false;
    }
    // Must have at least one .safetensors file (not just adapters)
    let has_st = std::fs::read_dir(dir)
        .map(|entries| {
            entries.flatten().any(|e| {
                e.path().extension().is_some_and(|ext| ext == "safetensors")
            })
        })
        .unwrap_or(false);
    if !has_st {
        return false;
    }
    // Must NOT have .gguf files (that's a GGUF model, not MLX)
    let has_gguf = std::fs::read_dir(dir)
        .map(|entries| {
            entries.flatten().any(|e| {
                e.path().extension().is_some_and(|ext| ext == "gguf")
            })
        })
        .unwrap_or(false);
    !has_gguf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discover_mlx_models() {
        let models = discover_mlx_models();
        // Should find at least the models we know exist
        eprintln!("discovered {} MLX models:", models.len());
        for m in &models {
            eprintln!("  {} (adapters: {})", m.name, m.has_adapters);
        }
        // On this machine we have several MLX models
        if !models.is_empty() {
            assert!(models.iter().any(|m| m.name.contains("MLX") || m.name.contains("mlx") || m.name.contains("8bit") || m.name.contains("4bit")),
                "should find at least one MLX model");
        }
    }

    #[test]
    fn test_is_mlx_model_dir() {
        let home = dirs::home_dir().unwrap();
        let qwen = home.join(".cache/lm-studio/models/mlx-community/Qwen3.5-2B-MLX-8bit");
        if qwen.exists() {
            assert!(is_mlx_model_dir(&qwen), "Qwen3.5-2B-MLX-8bit should be detected as MLX model");
        }
    }
}
