//! Managed MLX inference server subprocess.
//!
//! Supports two backends:
//!   - `mlx-lm` — `python3 -m mlx_lm.server` (default, supports adapter hot-reload)
//!   - `vllm-mlx` — `vllm-mlx serve` (continuous batching, native tool calling, prefix cache)
//!
//! Backend is selected via `mlxLmUrl` config: `"auto"` → mlx-lm, `"vllm-mlx"` → vllm-mlx.

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// Which inference server backend to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InferenceBackend {
    /// `python3 -m mlx_lm.server` — supports adapter hot-reload.
    #[default]
    MlxLm,
    /// `vllm-mlx serve` — continuous batching, tool calling, prefix cache.
    VllmMlx,
}

#[derive(Debug, Clone, Default)]
pub struct MlxLmServerOptions {
    pub decode_concurrency: Option<usize>,
    pub prompt_concurrency: Option<usize>,
    pub chat_template_args: Option<String>,
}

/// Options specific to vllm-mlx backend.
#[derive(Debug, Clone, Default)]
pub struct VllmMlxOptions {
    pub tool_call_parser: Option<String>,
    pub reasoning_parser: Option<String>,
    pub continuous_batching: bool,
    pub max_tokens: Option<u32>,
}

/// Managed MLX inference server process.
pub struct MlxLmServer {
    child: Option<Child>,
    pub port: u16,
    pub model_dir: PathBuf,
    pub adapter_path: Option<PathBuf>,
    pub backend: InferenceBackend,
    pub options: MlxLmServerOptions,
    pub vllm_options: VllmMlxOptions,
}

impl MlxLmServer {
    /// PID-file service name for this backend.
    fn pid_name(&self) -> &str {
        match self.backend {
            InferenceBackend::MlxLm => "mlx-lm",
            InferenceBackend::VllmMlx => "vllm-mlx",
        }
    }
}

impl MlxLmServer {
    /// Start an MLX inference server for the given model.
    pub fn start(
        model_dir: PathBuf,
        adapter_path: Option<PathBuf>,
        port: u16,
        backend: InferenceBackend,
        options: MlxLmServerOptions,
        vllm_options: VllmMlxOptions,
    ) -> Result<Self, String> {
        let mut server = MlxLmServer {
            child: None,
            port,
            model_dir: model_dir.clone(),
            adapter_path: adapter_path.clone(),
            backend,
            options,
            vllm_options,
        };
        server.spawn_process()?;
        Ok(server)
    }

    fn spawn_process(&mut self) -> Result<(), String> {
        // Kill any stale child processes tracked by PID files from previous runs.
        super::pid_file::cleanup_stale_pids();
        // Kill any stale process on the port from a previous run.
        Self::kill_stale_on_port(self.port);

        let mut cmd = match self.backend {
            InferenceBackend::MlxLm => self.build_mlx_lm_command(),
            InferenceBackend::VllmMlx => self.build_vllm_mlx_command()?,
        };

        let backend_name = match self.backend {
            InferenceBackend::MlxLm => "mlx_lm.server",
            InferenceBackend::VllmMlx => "vllm-mlx",
        };

        // Redirect stdout to null. Stderr goes to a log file so startup
        // failures are diagnosable (the pipe-buffer-full concern only applies
        // to piping, not to file I/O).
        cmd.stdout(Stdio::null());
        let log_path = dirs::home_dir()
            .map(|h| h.join(format!(".nanobot/{backend_name}.log")))
            .filter(|p| {
                p.parent()
                    .map_or(false, |d| d.exists() || std::fs::create_dir_all(d).is_ok())
            });
        if let Some(ref lp) = log_path {
            match std::fs::File::create(lp) {
                Ok(f) => cmd.stderr(Stdio::from(f)),
                Err(_) => cmd.stderr(Stdio::null()),
            };
            tracing::info!(log = %lp.display(), "server stderr → log file");
        } else {
            cmd.stderr(Stdio::null());
        }

        tracing::info!(backend = backend_name, cmd = ?cmd, "spawning inference server");

        let child = cmd
            .spawn()
            .map_err(|e| format!("failed to spawn {backend_name}: {e}"))?;
        super::pid_file::write_pid(self.pid_name(), self.port, child.id());
        self.child = Some(child);

        // Wait for server to be ready by polling TCP connect + simple HTTP GET.
        // Uses raw TcpStream + manual HTTP instead of reqwest::blocking to avoid
        // panicking inside a tokio async runtime context.
        let deadline = Instant::now() + Duration::from_secs(90);
        let addr = format!("127.0.0.1:{}", self.port);
        let http_req = format!(
            "GET /v1/models HTTP/1.1\r\nHost: 127.0.0.1:{}\r\nConnection: close\r\n\r\n",
            self.port,
        );
        while Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(500));
            // Check if process died
            if let Some(ref mut c) = self.child {
                if let Ok(Some(status)) = c.try_wait() {
                    let hint = log_path
                        .as_ref()
                        .map(|p| format!(" (see {})", p.display()))
                        .unwrap_or_default();
                    return Err(format!("{backend_name} exited with {status}{hint}"));
                }
            }
            // Try raw TCP connect + HTTP GET to check readiness
            if let Ok(mut stream) = std::net::TcpStream::connect_timeout(
                &addr.parse().unwrap(),
                Duration::from_millis(500),
            ) {
                use std::io::{Read, Write};
                let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
                if stream.write_all(http_req.as_bytes()).is_ok() {
                    let mut buf = [0u8; 256];
                    if let Ok(n) = stream.read(&mut buf) {
                        let resp = String::from_utf8_lossy(&buf[..n]);
                        if resp.contains("200 OK") || resp.contains("200") {
                            return Ok(());
                        }
                    }
                }
            }
        }
        self.kill();
        let hint = log_path
            .as_ref()
            .map(|p| format!(" (see {})", p.display()))
            .unwrap_or_default();
        Err(format!("{backend_name} failed to start within 90s{hint}"))
    }

    /// Build the `python3 -m mlx_lm.server` command.
    fn build_mlx_lm_command(&self) -> Command {
        let mut cmd = Command::new("python3");
        cmd.args(["-m", "mlx_lm.server"])
            .arg("--model")
            .arg(&self.model_dir)
            .arg("--port")
            .arg(self.port.to_string())
            .arg("--host")
            .arg("127.0.0.1");

        if let Some(ref adapter) = self.adapter_path {
            if adapter.join("adapters.safetensors").exists() {
                cmd.arg("--adapter-path").arg(adapter);
            }
        }
        if let Some(value) = self.options.decode_concurrency {
            cmd.arg("--decode-concurrency").arg(value.to_string());
        }
        if let Some(value) = self.options.prompt_concurrency {
            cmd.arg("--prompt-concurrency").arg(value.to_string());
        }
        if let Some(ref value) = self.options.chat_template_args {
            cmd.arg("--chat-template-args").arg(value);
        }
        cmd
    }

    /// Build the `vllm-mlx serve` command.
    fn build_vllm_mlx_command(&self) -> Result<Command, String> {
        let bin = find_vllm_mlx_binary().ok_or_else(|| {
            "vllm-mlx binary not found. Install with: pip install vllm-mlx".to_string()
        })?;

        let mut cmd = Command::new(bin);
        cmd.arg("serve")
            .arg(&self.model_dir)
            .arg("--port")
            .arg(self.port.to_string())
            .arg("--host")
            .arg("127.0.0.1");

        if self.vllm_options.continuous_batching {
            cmd.arg("--continuous-batching");
        }
        if let Some(max) = self.vllm_options.max_tokens {
            cmd.arg("--max-tokens").arg(max.to_string());
        }
        if let Some(ref parser) = self.vllm_options.tool_call_parser {
            cmd.arg("--enable-auto-tool-choice")
                .arg("--tool-call-parser")
                .arg(parser);
        }
        if let Some(ref parser) = self.vllm_options.reasoning_parser {
            cmd.arg("--reasoning-parser").arg(parser);
        }
        Ok(cmd)
    }

    /// Whether this backend supports LoRA adapter hot-reload.
    pub fn supports_adapters(&self) -> bool {
        self.backend == InferenceBackend::MlxLm
    }

    /// Switch to a different model (kills and respawns).
    pub fn switch_model(
        &mut self,
        model_dir: PathBuf,
        adapter_path: Option<PathBuf>,
    ) -> Result<(), String> {
        self.kill();
        self.model_dir = model_dir;
        self.adapter_path = adapter_path;
        self.spawn_process()
    }

    /// Reload adapters by restarting with the same model but updated adapter path.
    /// Only supported on mlx-lm backend; no-op on vllm-mlx.
    pub fn reload_adapters(&mut self, adapter_path: PathBuf) -> Result<(), String> {
        if !self.supports_adapters() {
            tracing::info!("vllm-mlx: adapter reload not supported, skipping");
            return Ok(());
        }
        let model = self.model_dir.clone();
        self.switch_model(model, Some(adapter_path))
    }

    pub fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    /// Kill any stale process on the given port.
    fn kill_stale_on_port(port: u16) {
        if let Ok(output) = Command::new("lsof")
            .args(["-ti", &format!(":{port}")])
            .output()
        {
            let pids = String::from_utf8_lossy(&output.stdout);
            for pid_str in pids.split_whitespace() {
                if let Ok(pid) = pid_str.parse::<i32>() {
                    unsafe {
                        libc::kill(pid, libc::SIGTERM);
                    }
                    std::thread::sleep(Duration::from_millis(100));
                }
            }
        }
    }

    pub fn kill(&mut self) {
        if let Some(ref mut child) = self.child.take() {
            // Graceful shutdown: SIGTERM first, then escalate to SIGKILL.
            let pid = child.id();
            unsafe {
                libc::kill(pid as i32, libc::SIGTERM);
            }
            let deadline = Instant::now() + Duration::from_secs(3);
            while Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(100));
                if let Ok(Some(_)) = child.try_wait() {
                    super::pid_file::remove_pid(self.pid_name(), self.port);
                    return;
                }
            }
            // Still alive after grace period — force kill
            let _ = child.kill();
            let _ = child.wait();
            super::pid_file::remove_pid(self.pid_name(), self.port);
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
// vllm-mlx binary discovery
// ---------------------------------------------------------------------------

/// Find the `vllm-mlx` binary. Checks:
/// 1. `.venv/bin/vllm-mlx` relative to `~/.nanobot/` (workspace venv)
/// 2. `.venv/bin/vllm-mlx` in the current working directory
/// 3. `vllm-mlx` on PATH (via `which`)
pub fn find_vllm_mlx_binary() -> Option<PathBuf> {
    // Check common venv locations.
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()));
    let candidates: Vec<PathBuf> = [
        dirs::home_dir().map(|h| h.join(".nanobot/.venv/bin/vllm-mlx")),
        std::env::current_dir()
            .ok()
            .map(|d| d.join(".venv/bin/vllm-mlx")),
        // Search upward from the executable's directory for a .venv
        exe_dir.as_ref().and_then(|d| {
            d.ancestors()
                .find(|a| a.join(".venv/bin/vllm-mlx").is_file())
                .map(|a| a.join(".venv/bin/vllm-mlx"))
        }),
    ]
    .into_iter()
    .flatten()
    .collect();

    for path in candidates {
        if path.is_file() {
            return Some(path);
        }
    }

    // Fall back to PATH lookup.
    Command::new("which")
        .arg("vllm-mlx")
        .output()
        .ok()
        .and_then(|o| {
            let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
            if o.status.success() && !s.is_empty() {
                Some(PathBuf::from(s))
            } else {
                None
            }
        })
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
            let name = path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            let adapter_dir = path.join("adapters");
            let has_adapters = adapter_dir.join("adapters.safetensors").exists();
            out.push(MlxModelInfo {
                path,
                name,
                has_adapters,
            });
        } else {
            // Recurse into org dirs (mlx-community/, lmstudio-community/, etc.)
            discover_recursive(&path, out);
        }
    }
}

pub fn is_mlx_model_dir(dir: &Path) -> bool {
    // Must have tokenizer.json
    if !dir.join("tokenizer.json").exists() {
        return false;
    }
    // Must have at least one .safetensors file (not just adapters)
    let has_st = std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .flatten()
                .any(|e| e.path().extension().is_some_and(|ext| ext == "safetensors"))
        })
        .unwrap_or(false);
    if !has_st {
        return false;
    }
    // Must NOT have .gguf files (that's a GGUF model, not MLX)
    let has_gguf = std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .flatten()
                .any(|e| e.path().extension().is_some_and(|ext| ext == "gguf"))
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
            assert!(
                models.iter().any(|m| m.name.contains("MLX")
                    || m.name.contains("mlx")
                    || m.name.contains("8bit")
                    || m.name.contains("4bit")),
                "should find at least one MLX model"
            );
        }
    }

    #[test]
    fn test_is_mlx_model_dir() {
        let home = dirs::home_dir().unwrap();
        let qwen = home.join(".cache/lm-studio/models/mlx-community/Qwen3.5-2B-MLX-8bit");
        if qwen.exists() {
            assert!(
                is_mlx_model_dir(&qwen),
                "Qwen3.5-2B-MLX-8bit should be detected as MLX model"
            );
        }
    }

    #[test]
    fn test_find_vllm_mlx_binary() {
        // Should find the vllm-mlx binary in .venv or PATH on this machine.
        let bin = find_vllm_mlx_binary();
        eprintln!("vllm-mlx binary: {:?}", bin);
        if let Some(ref path) = bin {
            assert!(path.is_file(), "vllm-mlx path should be a file");
            assert!(
                path.to_string_lossy().contains("vllm-mlx"),
                "path should contain vllm-mlx"
            );
        }
    }

    #[test]
    fn test_inference_backend_default_is_mlx_lm() {
        assert_eq!(InferenceBackend::default(), InferenceBackend::MlxLm);
    }
}
