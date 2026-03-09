#![allow(dead_code)]
//! LM Studio integration via CLI (server management) and HTTP API (model operations).
//!
//! The `lms` CLI is used only for server start/stop/status because those commands
//! work reliably from WSL2. All model operations (load, unload, list) use the
//! LM Studio REST API (`/api/v1/...`) because the CLI's IPC to the LMS daemon
//! hangs when called from WSL2.

use std::path::{Path, PathBuf};

/// Find the `lms` binary on the system.
///
/// Search order:
/// 1. `lms` / `lms.exe` on PATH (via `which`)
/// 2. Known Windows path: `/mnt/c/Users/PC/.cache/lm-studio/bin/lms.exe`
/// 3. Glob fallback: `/mnt/c/Users/*/.cache/lm-studio/bin/lms.exe`
pub(crate) fn find_lms_binary() -> Option<PathBuf> {
    // 1. Check PATH
    if let Ok(output) = std::process::Command::new("which").arg("lms").output() {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                return Some(PathBuf::from(path));
            }
        }
    }

    // Also check for lms.exe on PATH (WSL2 can run .exe)
    if let Ok(output) = std::process::Command::new("which").arg("lms.exe").output() {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                return Some(PathBuf::from(path));
            }
        }
    }

    // 2. Known Windows path
    let known = PathBuf::from("/mnt/c/Users/PC/.cache/lm-studio/bin/lms.exe");
    if known.exists() {
        return Some(known);
    }

    // 3. Glob fallback: /mnt/c/Users/*/.cache/lm-studio/bin/lms.exe
    let users_dir = Path::new("/mnt/c/Users");
    if users_dir.is_dir() {
        if let Ok(entries) = std::fs::read_dir(users_dir) {
            for entry in entries.flatten() {
                let candidate = entry.path().join(".cache/lm-studio/bin/lms.exe");
                if candidate.exists() {
                    return Some(candidate);
                }
            }
        }
    }

    None
}

// ============================================================================
// Server management (CLI-based — these work from WSL2)
// ============================================================================

/// Parse `lms server status --json --quiet` output.
///
/// Returns `Some(port)` if the server is running, `None` otherwise.
pub(crate) fn server_status(lms_bin: &Path) -> Option<u16> {
    let output = std::process::Command::new(lms_bin)
        .args(["server", "status", "--json", "--quiet"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let text = String::from_utf8_lossy(&output.stdout);
    let json: serde_json::Value = serde_json::from_str(text.trim()).ok()?;
    let running = json.get("running")?.as_bool()?;
    if running {
        json.get("port")?.as_u64().map(|p| p as u16)
    } else {
        None
    }
}

/// Start the lms server on the given port.
///
/// Waits for `/v1/models` to respond (up to 30 seconds).
pub(crate) async fn server_start(lms_bin: &Path, port: u16) -> Result<(), String> {
    // Check if already running on the right port
    if let Some(running_port) = server_status(lms_bin) {
        if running_port == port {
            return Ok(());
        }
        // Running on wrong port — stop and restart
        server_stop(lms_bin).ok();
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    }

    let output = tokio::process::Command::new(lms_bin)
        .args([
            "server",
            "start",
            "--port",
            &port.to_string(),
            "--bind",
            "0.0.0.0",
            "--cors",
        ])
        .output()
        .await
        .map_err(|e| format!("failed to run lms server start: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("lms server start failed: {}", stderr.trim()));
    }

    // Wait for readiness
    if !wait_for_ready(port, 30).await {
        return Err(format!(
            "lms server started but /v1/models not responding after 30s on port {}",
            port
        ));
    }

    Ok(())
}

/// Stop the lms server.
pub(crate) fn server_stop(lms_bin: &Path) -> Result<(), String> {
    let output = std::process::Command::new(lms_bin)
        .args(["server", "stop"])
        .output()
        .map_err(|e| format!("failed to run lms server stop: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("lms server stop failed: {}", stderr.trim()));
    }

    Ok(())
}

// ============================================================================
// Model operations (HTTP API — CLI IPC hangs from WSL2)
// ============================================================================

/// Build the LMS REST API base URL.
///
/// When `host` is non-empty it is used directly; otherwise `api_host()` is
/// called to resolve the WSL2 Windows host IP (or `"127.0.0.1"` on native Linux).
fn rest_base(host: &str, port: u16) -> String {
    let h = if host.is_empty() {
        api_host()
    } else {
        host.to_string()
    };
    format!("http://{}:{}", h, port)
}

/// Load a model via the LM Studio REST API.
///
/// Uses `POST /api/v1/models/load` with optional `context_length`.
/// Returns immediately if the model is already loaded.
///
/// `host` is the hostname or IP of the LM Studio server.  Pass `""` to use the
/// default resolved via `api_host()` (WSL2 Windows host or `127.0.0.1`).
pub(crate) async fn load_model(
    host: &str,
    port: u16,
    model: &str,
    context_length: Option<usize>,
    timeout_secs: u64,
) -> Result<(), String> {
    load_model_at(&rest_base(host, port), model, context_length).await
}

/// Like `load_model` but accepts a full base URL (e.g. `"http://192.168.1.22:1234"`).
pub(crate) async fn load_model_at(
    base_url: &str,
    model: &str,
    context_length: Option<usize>,
) -> Result<(), String> {
    // Check if already loaded with matching context
    let models = list_models_full_at(base_url).await;
    if should_skip_load(&models, model, context_length) {
        tracing::debug!(
            "lms skip load: model={} already loaded with matching context",
            model
        );
        return Ok(());
    }

    post_load_model_at(base_url, model, context_length).await
}

/// Reload a model with a specific context length.
///
/// Unlike `load_model`, this forces a reload when the model is already loaded,
/// so context-size changes actually take effect.
///
/// `host` is the hostname or IP of the LM Studio server.  Pass `""` to use the
/// default resolved via `api_host()`.
pub(crate) async fn reload_model_with_context(
    host: &str,
    port: u16,
    model: &str,
    context_length: usize,
    load_timeout_secs: u64,
    unload_timeout_secs: u64,
) -> Result<(), String> {
    reload_model_with_context_at(
        &rest_base(host, port),
        model,
        context_length,
        unload_timeout_secs,
    )
    .await
}

/// Like `reload_model_with_context` but accepts a full base URL.
pub(crate) async fn reload_model_with_context_at(
    base_url: &str,
    model: &str,
    context_length: usize,
    unload_timeout_secs: u64,
) -> Result<(), String> {
    let models = list_models_full_at(base_url).await;
    let loaded_keys: Vec<String> = models
        .iter()
        .filter(|m| m.loaded && model_matches(&m.key, model))
        .map(|m| m.key.clone())
        .collect();
    for loaded_model in &loaded_keys {
        if let Err(e) = unload_model_at(base_url, loaded_model).await {
            tracing::warn!(
                "lms unload before reload failed: model={} loaded_model={} error={}",
                model,
                loaded_model,
                e
            );
        }
    }

    post_load_model_at(base_url, model, Some(context_length)).await
}

pub(crate) fn model_matches(loaded: &str, model: &str) -> bool {
    loaded == model || loaded.contains(model) || model.contains(loaded)
}

fn should_skip_load(models: &[ModelInfo], model: &str, context_length: Option<usize>) -> bool {
    let found = models
        .iter()
        .find(|m| m.loaded && model_matches(&m.key, model));
    match found {
        None => false,
        Some(info) => match context_length {
            // No specific context requested — already loaded is fine
            None => true,
            // Context requested — skip only if it matches what's already loaded
            Some(requested) => info.loaded_context == Some(requested),
        },
    }
}

async fn post_load_model(
    host: &str,
    port: u16,
    model: &str,
    context_length: Option<usize>,
    timeout_secs: u64,
) -> Result<(), String> {
    post_load_model_at(&rest_base(host, port), model, context_length).await
}

async fn post_load_model_at(
    base_url: &str,
    model: &str,
    context_length: Option<usize>,
) -> Result<(), String> {
    let url = format!("{}/api/v1/models/load", base_url);
    let mut body = serde_json::json!({ "model": model });
    if let Some(ctx) = context_length {
        body["context_length"] = serde_json::json!(ctx);
    }

    tracing::info!(
        "lms load via HTTP: model={} ctx={:?}",
        model,
        context_length
    );

    let client = reqwest::Client::new();
    let resp = client
        .post(&url)
        .json(&body)
        .timeout(std::time::Duration::from_secs(120))
        .send()
        .await
        .map_err(|e| format!("HTTP load request failed: {}", e))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(format!(
            "lms load '{}' failed (HTTP {}): {}",
            model, status, text
        ));
    }

    let json: serde_json::Value = resp.json().await.unwrap_or_default();
    let load_time = json
        .get("load_time_seconds")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    tracing::info!(
        "lms load complete: model={} load_time={:.1}s",
        model,
        load_time
    );

    Ok(())
}

/// Unload a single model via the LM Studio REST API.
///
/// `host` is the hostname or IP of the LM Studio server.  Pass `""` to use the
/// default resolved via `api_host()`.
pub(crate) async fn unload_model(
    host: &str,
    port: u16,
    model: &str,
    timeout_secs: u64,
) -> Result<(), String> {
    unload_model_at(&rest_base(host, port), model).await
}

/// Like `unload_model` but accepts a full base URL.
///
/// Looks up the `instance_id` from `/api/v1/models` (required by newer LM Studio
/// versions) and falls back to identifier-only for older versions.
pub(crate) async fn unload_model_at(base_url: &str, model: &str) -> Result<(), String> {
    let url = format!("{}/api/v1/models/unload", base_url);

    // Look up the instance_id for this model.
    let instance_id = {
        let models = list_models_full_at(base_url).await;
        models
            .iter()
            .find(|m| m.loaded && model_matches(&m.key, model))
            .and_then(|m| m.instance_id.clone())
    };

    let body = if let Some(ref iid) = instance_id {
        serde_json::json!({ "identifier": model, "instance_id": iid })
    } else {
        serde_json::json!({ "identifier": model })
    };

    let client = reqwest::Client::new();
    let resp = client
        .post(&url)
        .json(&body)
        .timeout(std::time::Duration::from_secs(60))
        .send()
        .await
        .map_err(|e| format!("HTTP unload request failed: {}", e))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(format!(
            "lms unload '{}' failed (HTTP {}): {}",
            model, status, text
        ));
    }

    Ok(())
}

/// Unload all models by querying loaded models and unloading each.
///
/// `host` is the hostname or IP of the LM Studio server.  Pass `""` to use the
/// default resolved via `api_host()`.
pub(crate) async fn unload_all(host: &str, port: u16, timeout_secs: u64) -> Result<(), String> {
    let loaded = list_loaded(host, port).await;
    for model in &loaded {
        unload_model(host, port, model, timeout_secs).await?;
    }
    Ok(())
}

/// Model info from the LM Studio REST API.
pub(crate) struct ModelInfo {
    pub key: String,
    pub model_type: String,
    pub loaded: bool,
    /// Context length of the first loaded instance, if any.
    pub loaded_context: Option<usize>,
    /// Instance ID of the first loaded instance (required for unload API).
    pub instance_id: Option<String>,
}

/// List all models via `GET /api/v1/models`, returning full info.
async fn list_models_full(host: &str, port: u16) -> Vec<ModelInfo> {
    list_models_full_at(&rest_base(host, port)).await
}

async fn list_models_full_at(base_url: &str) -> Vec<ModelInfo> {
    let url = format!("{}/api/v1/models", base_url);
    let resp = match reqwest::get(&url).await {
        Ok(r) if r.status().is_success() => r,
        _ => return Vec::new(),
    };

    let json: serde_json::Value = match resp.json().await {
        Ok(j) => j,
        Err(_) => return Vec::new(),
    };

    let models = match json.get("models").and_then(|m| m.as_array()) {
        Some(arr) => arr,
        None => return Vec::new(),
    };

    models
        .iter()
        .filter_map(|m| {
            let key = m.get("key")?.as_str()?.to_string();
            let model_type = m.get("type")?.as_str()?.to_string();
            let instances = m.get("loaded_instances").and_then(|v| v.as_array());
            let loaded = instances.map(|a| !a.is_empty()).unwrap_or(false);
            let first_instance = instances.and_then(|arr| arr.first());
            let loaded_context = first_instance
                .and_then(|inst| inst.get("config"))
                .and_then(|c| c.get("context_length"))
                .and_then(|v| v.as_u64())
                .map(|n| n as usize);
            let instance_id = first_instance
                .and_then(|inst| inst.get("instance_id"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            Some(ModelInfo {
                key,
                model_type,
                loaded,
                loaded_context,
                instance_id,
            })
        })
        .collect()
}

/// List currently loaded model identifiers via the HTTP API.
///
/// `host` is the hostname or IP of the LM Studio server.  Pass `""` to use the
/// default resolved via `api_host()`.
pub(crate) async fn list_loaded(host: &str, port: u16) -> Vec<String> {
    list_models_full(host, port)
        .await
        .into_iter()
        .filter(|m| m.loaded)
        .map(|m| m.key)
        .collect()
}

/// List all available (downloaded) model identifiers via the HTTP API.
///
/// Filters out embedding models by default.
///
/// `host` is the hostname or IP of the LM Studio server.  Pass `""` to use the
/// default resolved via `api_host()`.
pub(crate) async fn list_available(host: &str, port: u16) -> Vec<String> {
    list_models_full(host, port)
        .await
        .into_iter()
        .filter(|m| m.model_type == "llm")
        .map(|m| m.key)
        .collect()
}

/// List all available models with full info.
///
/// `host` is the hostname or IP of the LM Studio server.  Pass `""` to use the
/// default resolved via `api_host()`.
pub(crate) async fn list_available_full(host: &str, port: u16) -> Vec<ModelInfo> {
    list_models_full(host, port)
        .await
        .into_iter()
        .filter(|m| m.model_type == "llm")
        .collect()
}

/// Check if a model identifier is available in LM Studio.
///
/// Performs case-insensitive substring matching against the list of available
/// models, so `"ministral-3-3b"` matches `"mistralai/ministral-3-3b"`.
pub(crate) fn is_model_available(available: &[String], model_id: &str) -> bool {
    let needle = model_id.to_lowercase();
    available.iter().any(|m| {
        let m_lower = m.to_lowercase();
        m_lower == needle || m_lower.contains(&needle) || needle.contains(&m_lower)
    })
}

/// Resolve a potentially stale model name (e.g. GGUF filename) to an LMS identifier.
///
/// Uses the available models list for fuzzy matching. Returns the LMS identifier
/// if found, otherwise returns the input as-is.
///
/// Matching strategy (first match wins):
/// 1. Exact match (case-insensitive)
/// 2. Substring match on full key
/// 3. Substring match ignoring org prefix (e.g. `qwen/` in `qwen/qwen2.5-vl-7b`)
/// 4. Base-name overlap (strips org prefix + common suffixes like `-Instruct`)
pub(crate) fn resolve_model_name(available: &[String], hint: &str) -> String {
    let needle = hint.to_lowercase();

    // 1. Exact match
    if let Some(m) = available.iter().find(|m| m.to_lowercase() == needle) {
        return m.clone();
    }

    // 2. Substring match on full key
    if let Some(m) = available.iter().find(|m| {
        let m_lower = m.to_lowercase();
        m_lower.contains(&needle) || needle.contains(&m_lower)
    }) {
        return m.clone();
    }

    // 3. Match ignoring org prefix (e.g. "qwen/" in "qwen/qwen2.5-vl-7b")
    if let Some(m) = available.iter().find(|m| {
        let m_lower = m.to_lowercase();
        let base = m_lower.rsplit('/').next().unwrap_or(&m_lower);
        base.contains(&needle) || needle.contains(base)
    }) {
        return m.clone();
    }

    // 4. Normalize both sides: strip org prefix and common GGUF suffixes
    //    e.g. "Qwen2.5-VL-7B-Instruct" → "qwen2.5-vl-7b" matches "qwen/qwen2.5-vl-7b"
    let needle_base = strip_match_suffixes(&needle);
    if let Some(m) = available.iter().find(|m| {
        let m_lower = m.to_lowercase();
        let m_base = strip_match_suffixes(m_lower.rsplit('/').next().unwrap_or(&m_lower));
        m_base.contains(&needle_base) || needle_base.contains(&m_base)
    }) {
        return m.clone();
    }

    // No match — return as-is
    hint.to_string()
}

/// Strip common suffixes that differ between GGUF filenames and LMS identifiers.
fn strip_match_suffixes(name: &str) -> String {
    let mut s = name.to_lowercase();
    for suffix in &["-instruct", "-chat", "-it", "-hf", "-gguf"] {
        if let Some(stripped) = s.strip_suffix(suffix) {
            s = stripped.to_string();
        }
    }
    s
}

// ============================================================================
// Host / network helpers
// ============================================================================

/// Resolve the API host for LM Studio.
///
/// On WSL2, LMS runs on the Windows host — returns the host IP from
/// `/etc/resolv.conf`. Otherwise returns `"127.0.0.1"`.
pub(crate) fn api_host() -> String {
    let proc_version = std::fs::read_to_string("/proc/version").unwrap_or_default();
    let resolv_conf = std::fs::read_to_string("/etc/resolv.conf").unwrap_or_default();
    parse_wsl_host_ip(&proc_version, &resolv_conf).unwrap_or_else(|| "127.0.0.1".to_string())
}

/// Pure parser: extract Windows host IP from /proc/version + /etc/resolv.conf.
fn parse_wsl_host_ip(proc_version: &str, resolv_conf: &str) -> Option<String> {
    if !proc_version.to_lowercase().contains("microsoft") {
        return None;
    }
    for line in resolv_conf.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("nameserver") {
            return trimmed.split_whitespace().nth(1).map(|s| s.to_string());
        }
    }
    None
}

/// Wait for the lms server to respond on `/v1/models`.
async fn wait_for_ready(port: u16, timeout_secs: u64) -> bool {
    let host = api_host();
    let url = format!("http://{}:{}/v1/models", host, port);
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(timeout_secs);

    while tokio::time::Instant::now() < deadline {
        if let Ok(resp) = reqwest::get(&url).await {
            if resp.status().is_success() {
                return true;
            }
        }
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    }

    false
}

// ============================================================================
// Jinja probe — detect --jinja support for native function calling
// ============================================================================

/// Probe an LM Studio endpoint to detect whether `--jinja` (native tool_calls)
/// is likely supported.
///
/// Strategy: send a minimal chat completion with `tools` parameter.  If the
/// server returns a 200 with valid JSON (even if no tool_calls), jinja is
/// available.  If it returns 400/422 or an error mentioning "tools", jinja
/// is not enabled.
///
/// Returns `true` if jinja appears supported, `false` otherwise.
/// Timeout: 3 seconds to avoid blocking startup.
pub(crate) async fn probe_jinja_support(api_base: &str) -> bool {
    let url = format!("{}/chat/completions", api_base.trim_end_matches('/'));
    let body = serde_json::json!({
        "model": "probe",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
        "tools": [{
            "type": "function",
            "function": {
                "name": "test",
                "description": "test",
                "parameters": {"type": "object", "properties": {}}
            }
        }]
    });

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .build()
        .unwrap_or_default();

    match client.post(&url).json(&body).send().await {
        Ok(resp) => {
            let status = resp.status();
            if status.is_success() {
                tracing::info!("jinja_probe: tools accepted (status {status})");
                return true;
            }
            // 4xx with error about tools → jinja not enabled
            let text = resp.text().await.unwrap_or_default();
            let no_jinja = text.contains("tools")
                || text.contains("jinja")
                || text.contains("not supported")
                || text.contains("unknown field");
            if no_jinja {
                tracing::warn!("jinja_probe: tools rejected — --jinja likely not enabled");
            }
            !no_jinja
        }
        Err(e) => {
            tracing::debug!("jinja_probe: connection failed: {e}");
            false
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_jinja_probe_unreachable_returns_false() {
        // Probe against a port nothing listens on → should return false, not panic.
        let result = probe_jinja_support("http://127.0.0.1:19999/v1").await;
        assert!(!result, "unreachable server should return false");
    }

    #[tokio::test]
    async fn test_jinja_probe_bad_url_returns_false() {
        let result = probe_jinja_support("not-a-url").await;
        assert!(!result, "invalid URL should return false");
    }

    async fn fetch_loaded_context_length(port: u16, model: &str) -> Option<usize> {
        let url = format!("{}/api/v1/models", rest_base("", port));
        let resp = reqwest::get(&url).await.ok()?;
        if !resp.status().is_success() {
            return None;
        }
        let json: serde_json::Value = resp.json().await.ok()?;
        let models = json.get("models")?.as_array()?;
        for m in models {
            let key = m.get("key")?.as_str()?;
            if !model_matches(key, model) {
                continue;
            }
            let instances = m.get("loaded_instances")?.as_array()?;
            if instances.is_empty() {
                return None;
            }
            return instances.iter().find_map(|inst| {
                inst.get("config")
                    .and_then(|c| c.get("context_length"))
                    .and_then(|v| v.as_u64())
                    .map(|n| n as usize)
            });
        }
        None
    }

    async fn fetch_max_context_length(port: u16, model: &str) -> Option<usize> {
        let url = format!("{}/api/v1/models", rest_base("", port));
        let resp = reqwest::get(&url).await.ok()?;
        if !resp.status().is_success() {
            return None;
        }
        let json: serde_json::Value = resp.json().await.ok()?;
        let models = json.get("models")?.as_array()?;
        for m in models {
            let key = m.get("key")?.as_str()?;
            if model_matches(key, model) {
                return m
                    .get("max_context_length")
                    .and_then(|v| v.as_u64())
                    .map(|n| n as usize);
            }
        }
        None
    }

    // -- is_model_available tests --

    #[test]
    fn test_is_model_available_exact_match() {
        let available = vec!["nanbeige4.1-3b".to_string(), "qwen3-1.7b".to_string()];
        assert!(is_model_available(&available, "nanbeige4.1-3b"));
        assert!(is_model_available(&available, "qwen3-1.7b"));
        assert!(!is_model_available(&available, "nonexistent"));
    }

    #[test]
    fn test_is_model_available_substring_match() {
        let available = vec!["mistralai/ministral-3-3b".to_string()];
        assert!(is_model_available(&available, "mistralai/ministral-3-3b"));
        assert!(is_model_available(&available, "ministral-3-3b"));
    }

    #[test]
    fn test_is_model_available_case_insensitive() {
        let available = vec!["NanBeige4.1-3B".to_string()];
        assert!(is_model_available(&available, "nanbeige4.1-3b"));
    }

    #[test]
    fn test_is_model_available_empty_list() {
        let available: Vec<String> = vec![];
        assert!(!is_model_available(&available, "anything"));
    }

    // -- parse_wsl_host_ip tests --

    #[test]
    fn test_parse_wsl_host_ip_wsl2() {
        let proc_version = "Linux version 5.15.167.4-microsoft-standard-WSL2";
        let resolv_conf =
            "# This file was automatically generated by WSL.\nnameserver 172.26.16.1\n";
        assert_eq!(
            parse_wsl_host_ip(proc_version, resolv_conf),
            Some("172.26.16.1".to_string())
        );
    }

    #[test]
    fn test_parse_wsl_host_ip_not_wsl() {
        let proc_version = "Linux version 6.1.0-26-amd64 (debian-kernel@lists.debian.org)";
        let resolv_conf = "nameserver 8.8.8.8\n";
        assert_eq!(parse_wsl_host_ip(proc_version, resolv_conf), None);
    }

    #[test]
    fn test_parse_wsl_host_ip_no_nameserver() {
        let proc_version = "Linux version 5.15.167.4-microsoft-standard-WSL2";
        let resolv_conf = "# empty resolv.conf\n";
        assert_eq!(parse_wsl_host_ip(proc_version, resolv_conf), None);
    }

    #[test]
    fn test_server_status_parsing() {
        // Test that our JSON parsing logic works
        let json_str = r#"{"running":true,"port":1234}"#;
        let json: serde_json::Value = serde_json::from_str(json_str).unwrap();
        assert_eq!(json["running"].as_bool(), Some(true));
        assert_eq!(json["port"].as_u64(), Some(1234));
    }

    #[test]
    fn test_server_status_not_running() {
        let json_str = r#"{"running":false,"port":1234}"#;
        let json: serde_json::Value = serde_json::from_str(json_str).unwrap();
        assert_eq!(json["running"].as_bool(), Some(false));
    }

    #[test]
    fn test_rest_base() {
        // Empty host falls back to api_host() (environment-dependent)
        let base = rest_base("", 1234);
        assert!(base.starts_with("http://"));
        assert!(base.ends_with(":1234"));
    }

    #[test]
    fn test_rest_base_explicit_host() {
        let base = rest_base("192.168.1.22", 1234);
        assert_eq!(base, "http://192.168.1.22:1234");
    }

    // -- resolve_model_name tests --

    #[test]
    fn test_resolve_exact_match() {
        let available = vec!["nvidia_orchestrator-8b".to_string()];
        assert_eq!(
            resolve_model_name(&available, "nvidia_orchestrator-8b"),
            "nvidia_orchestrator-8b"
        );
    }

    #[test]
    fn test_resolve_with_org_prefix() {
        let available = vec![
            "qwen/qwen2.5-vl-7b".to_string(),
            "nanbeige4.1-3b".to_string(),
        ];
        // Stripped GGUF name contains the LMS base name (after org prefix removed)
        assert_eq!(
            resolve_model_name(&available, "Qwen2.5-VL-7B-Instruct"),
            "qwen/qwen2.5-vl-7b"
        );
    }

    #[test]
    fn test_resolve_instruct_suffix_stripped() {
        let available = vec!["qwen/qwen2.5-vl-7b".to_string()];
        // GGUF has -Instruct suffix, LMS doesn't — strip_match_suffixes handles this
        assert_eq!(
            resolve_model_name(&available, "qwen2.5-vl-7b-instruct"),
            "qwen/qwen2.5-vl-7b"
        );
    }

    #[test]
    fn test_resolve_no_match_returns_input() {
        let available = vec!["nanbeige4.1-3b".to_string()];
        assert_eq!(
            resolve_model_name(&available, "nonexistent-model"),
            "nonexistent-model"
        );
    }

    #[test]
    fn test_resolve_prefers_exact_over_fuzzy() {
        let available = vec![
            "qwen3-1.7b".to_string(),
            "qwen/qwen3-30b-a3b-2507".to_string(),
        ];
        assert_eq!(resolve_model_name(&available, "qwen3-1.7b"), "qwen3-1.7b");
    }

    // -- strip_match_suffixes tests --

    #[test]
    fn test_strip_match_suffixes() {
        assert_eq!(
            strip_match_suffixes("qwen2.5-vl-7b-instruct"),
            "qwen2.5-vl-7b"
        );
        assert_eq!(strip_match_suffixes("model-chat"), "model");
        assert_eq!(strip_match_suffixes("model-it"), "model");
        assert_eq!(strip_match_suffixes("plain-model"), "plain-model");
    }

    fn make_model_info(key: &str, loaded: bool, loaded_context: Option<usize>) -> ModelInfo {
        ModelInfo {
            key: key.to_string(),
            model_type: "llm".to_string(),
            loaded,
            loaded_context,
            instance_id: None,
        }
    }

    #[test]
    fn test_should_skip_load_when_loaded_and_no_context_requested() {
        let models = vec![make_model_info(
            "qwen2.5-coder-7b-instruct-mlx",
            true,
            Some(4096),
        )];
        assert!(should_skip_load(
            &models,
            "qwen2.5-coder-7b-instruct-mlx",
            None
        ));
    }

    #[test]
    fn test_should_skip_load_when_context_matches() {
        let models = vec![make_model_info(
            "qwen2.5-coder-7b-instruct-mlx",
            true,
            Some(6144),
        )];
        assert!(should_skip_load(
            &models,
            "qwen2.5-coder-7b-instruct-mlx",
            Some(6144)
        ));
    }

    #[test]
    fn test_should_not_skip_load_when_context_differs() {
        let models = vec![make_model_info(
            "qwen2.5-coder-7b-instruct-mlx",
            true,
            Some(4096),
        )];
        assert!(!should_skip_load(
            &models,
            "qwen2.5-coder-7b-instruct-mlx",
            Some(6144)
        ));
    }

    #[test]
    fn test_should_not_skip_load_when_not_loaded() {
        let models = vec![make_model_info(
            "qwen2.5-coder-7b-instruct-mlx",
            false,
            None,
        )];
        assert!(!should_skip_load(
            &models,
            "qwen2.5-coder-7b-instruct-mlx",
            None
        ));
    }

    #[test]
    fn test_model_matches_supports_fuzzy_key_matching() {
        assert!(model_matches(
            "qwen/qwen3-4b-thinking-2507",
            "qwen3-4b-thinking-2507"
        ));
        assert!(model_matches(
            "qwen3-4b-thinking-2507",
            "qwen/qwen3-4b-thinking-2507"
        ));
    }

    #[tokio::test]
    #[ignore = "requires running LM Studio server and loaded model"]
    async fn test_reload_model_with_context_updates_loaded_instance_context_live() {
        let port: u16 = std::env::var("NANOBOT_TEST_LMS_PORT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1234);
        let model = std::env::var("NANOBOT_TEST_LOCAL_MODEL")
            .unwrap_or_else(|_| "qwen/qwen3-4b-thinking-2507".to_string());

        let max_ctx = fetch_max_context_length(port, &model)
            .await
            .expect("model not found on LM Studio /api/v1/models");

        let target_a = 4096.min(max_ctx).max(2048);
        let target_b = 6144.min(max_ctx).max(2048);
        let original = fetch_loaded_context_length(port, &model).await;

        reload_model_with_context("", port, &model, target_a, 120, 30)
            .await
            .expect("failed to reload model at target_a context");
        let observed_a = fetch_loaded_context_length(port, &model)
            .await
            .expect("model not loaded after target_a reload");
        assert_eq!(
            observed_a, target_a,
            "LM Studio loaded_instances context_length mismatch after first reload"
        );

        if target_b != target_a {
            reload_model_with_context("", port, &model, target_b, 120, 30)
                .await
                .expect("failed to reload model at target_b context");
            let observed_b = fetch_loaded_context_length(port, &model)
                .await
                .expect("model not loaded after target_b reload");
            assert_eq!(
                observed_b, target_b,
                "LM Studio loaded_instances context_length mismatch after second reload"
            );
        }

        if let Some(orig_ctx) = original {
            let _ = reload_model_with_context("", port, &model, orig_ctx, 120, 30).await;
        }
    }
}
