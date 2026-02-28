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
    if let Ok(output) = std::process::Command::new("which")
        .arg("lms")
        .output()
    {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                return Some(PathBuf::from(path));
            }
        }
    }

    // Also check for lms.exe on PATH (WSL2 can run .exe)
    if let Ok(output) = std::process::Command::new("which")
        .arg("lms.exe")
        .output()
    {
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
        .args(["server", "start", "--port", &port.to_string(), "--bind", "0.0.0.0", "--cors"])
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
fn rest_base(port: u16) -> String {
    format!("http://{}:{}", api_host(), port)
}

/// Load a model via the LM Studio REST API.
///
/// Uses `POST /api/v1/models/load` with optional `context_length`.
/// Returns immediately if the model is already loaded.
pub(crate) async fn load_model(
    port: u16,
    model: &str,
    context_length: Option<usize>,
) -> Result<(), String> {
    // Check if already loaded
    let loaded = list_loaded(port).await;
    if should_skip_load(&loaded, model, context_length) {
        return Ok(());
    }

    post_load_model(port, model, context_length).await
}

/// Reload a model with a specific context length.
///
/// Unlike `load_model`, this forces a reload when the model is already loaded,
/// so context-size changes actually take effect.
pub(crate) async fn reload_model_with_context(
    port: u16,
    model: &str,
    context_length: usize,
) -> Result<(), String> {
    let loaded = list_loaded(port).await;
    for loaded_model in loaded.iter().filter(|m| model_matches(m, model)) {
        if let Err(e) = unload_model(port, loaded_model).await {
            tracing::warn!(
                "lms unload before reload failed: model={} loaded_model={} error={}",
                model,
                loaded_model,
                e
            );
        }
    }

    post_load_model(port, model, Some(context_length)).await
}

fn model_matches(loaded: &str, model: &str) -> bool {
    loaded == model || loaded.contains(model) || model.contains(loaded)
}

fn should_skip_load(loaded: &[String], model: &str, _context_length: Option<usize>) -> bool {
    loaded.iter().any(|m| model_matches(m, model))
}

async fn post_load_model(
    port: u16,
    model: &str,
    context_length: Option<usize>,
) -> Result<(), String> {

    let url = format!("{}/api/v1/models/load", rest_base(port));
    let mut body = serde_json::json!({ "model": model });
    if let Some(ctx) = context_length {
        body["context_length"] = serde_json::json!(ctx);
    }

    tracing::info!("lms load via HTTP: model={} ctx={:?}", model, context_length);

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
        return Err(format!("lms load '{}' failed (HTTP {}): {}", model, status, text));
    }

    let json: serde_json::Value = resp.json().await.unwrap_or_default();
    let load_time = json.get("load_time_seconds")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    tracing::info!("lms load complete: model={} load_time={:.1}s", model, load_time);

    Ok(())
}

/// Unload a single model via the LM Studio REST API.
pub(crate) async fn unload_model(port: u16, model: &str) -> Result<(), String> {
    let url = format!("{}/api/v1/models/unload", rest_base(port));
    let body = serde_json::json!({ "instance_id": model });

    let client = reqwest::Client::new();
    let resp = client
        .post(&url)
        .json(&body)
        .timeout(std::time::Duration::from_secs(30))
        .send()
        .await
        .map_err(|e| format!("HTTP unload request failed: {}", e))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("lms unload '{}' failed (HTTP {}): {}", model, status, text));
    }

    Ok(())
}

/// Unload all models by querying loaded models and unloading each.
pub(crate) async fn unload_all(port: u16) -> Result<(), String> {
    let loaded = list_loaded(port).await;
    for model in &loaded {
        unload_model(port, model).await?;
    }
    Ok(())
}

/// Model info from the LM Studio REST API.
pub(crate) struct ModelInfo {
    pub key: String,
    pub model_type: String,
    pub loaded: bool,
}

/// List all models via `GET /api/v1/models`, returning full info.
async fn list_models_full(port: u16) -> Vec<ModelInfo> {
    let url = format!("{}/api/v1/models", rest_base(port));
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
            let loaded = m.get("loaded_instances")
                .and_then(|v| v.as_array())
                .map(|a| !a.is_empty())
                .unwrap_or(false);
            Some(ModelInfo { key, model_type, loaded })
        })
        .collect()
}

/// List currently loaded model identifiers via the HTTP API.
pub(crate) async fn list_loaded(port: u16) -> Vec<String> {
    list_models_full(port)
        .await
        .into_iter()
        .filter(|m| m.loaded)
        .map(|m| m.key)
        .collect()
}

/// List all available (downloaded) model identifiers via the HTTP API.
///
/// Filters out embedding models by default.
pub(crate) async fn list_available(port: u16) -> Vec<String> {
    list_models_full(port)
        .await
        .into_iter()
        .filter(|m| m.model_type == "llm")
        .map(|m| m.key)
        .collect()
}

/// List all available models with full info.
pub(crate) async fn list_available_full(port: u16) -> Vec<ModelInfo> {
    list_models_full(port)
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
    let deadline =
        tokio::time::Instant::now() + std::time::Duration::from_secs(timeout_secs);

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
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    async fn fetch_loaded_context_length(port: u16, model: &str) -> Option<usize> {
        let url = format!("{}/api/v1/models", rest_base(port));
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
            return instances
                .iter()
                .find_map(|inst| {
                    inst.get("config")
                        .and_then(|c| c.get("context_length"))
                        .and_then(|v| v.as_u64())
                        .map(|n| n as usize)
                });
        }
        None
    }

    async fn fetch_max_context_length(port: u16, model: &str) -> Option<usize> {
        let url = format!("{}/api/v1/models", rest_base(port));
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
        let resolv_conf = "# This file was automatically generated by WSL.\nnameserver 172.26.16.1\n";
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
        // Just verify the format (actual host depends on environment)
        let base = rest_base(1234);
        assert!(base.starts_with("http://"));
        assert!(base.ends_with(":1234"));
    }

    // -- resolve_model_name tests --

    #[test]
    fn test_resolve_exact_match() {
        let available = vec!["nvidia_orchestrator-8b".to_string()];
        assert_eq!(resolve_model_name(&available, "nvidia_orchestrator-8b"), "nvidia_orchestrator-8b");
    }

    #[test]
    fn test_resolve_with_org_prefix() {
        let available = vec!["qwen/qwen2.5-vl-7b".to_string(), "nanbeige4.1-3b".to_string()];
        // Stripped GGUF name contains the LMS base name (after org prefix removed)
        assert_eq!(resolve_model_name(&available, "Qwen2.5-VL-7B-Instruct"), "qwen/qwen2.5-vl-7b");
    }

    #[test]
    fn test_resolve_instruct_suffix_stripped() {
        let available = vec!["qwen/qwen2.5-vl-7b".to_string()];
        // GGUF has -Instruct suffix, LMS doesn't — strip_match_suffixes handles this
        assert_eq!(resolve_model_name(&available, "qwen2.5-vl-7b-instruct"), "qwen/qwen2.5-vl-7b");
    }

    #[test]
    fn test_resolve_no_match_returns_input() {
        let available = vec!["nanbeige4.1-3b".to_string()];
        assert_eq!(resolve_model_name(&available, "nonexistent-model"), "nonexistent-model");
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
        assert_eq!(strip_match_suffixes("qwen2.5-vl-7b-instruct"), "qwen2.5-vl-7b");
        assert_eq!(strip_match_suffixes("model-chat"), "model");
        assert_eq!(strip_match_suffixes("model-it"), "model");
        assert_eq!(strip_match_suffixes("plain-model"), "plain-model");
    }

    #[test]
    fn test_should_skip_load_when_loaded_and_no_context_requested() {
        let loaded = vec!["qwen2.5-coder-7b-instruct-mlx".to_string()];
        assert!(should_skip_load(
            &loaded,
            "qwen2.5-coder-7b-instruct-mlx",
            None
        ));
    }

    #[test]
    fn test_should_skip_load_when_loaded_and_context_requested() {
        // If the model is already loaded, skip regardless of context_length.
        // Callers that need a forced reload must use reload_model_with_context().
        let loaded = vec!["qwen2.5-coder-7b-instruct-mlx".to_string()];
        assert!(should_skip_load(
            &loaded,
            "qwen2.5-coder-7b-instruct-mlx",
            Some(6144)
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

        reload_model_with_context(port, &model, target_a)
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
            reload_model_with_context(port, &model, target_b)
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
            let _ = reload_model_with_context(port, &model, orig_ctx).await;
        }
    }
}
