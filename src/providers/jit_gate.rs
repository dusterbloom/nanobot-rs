//! JIT model safety gate for LM Studio and similar JIT-loading servers.
//!
//! When multiple providers share the same endpoint but request different models,
//! the JIT loader can crash from concurrent model switches. This module provides:
//!
//! - **JitGate**: A single-permit semaphore that serialises all LLM requests to
//!   one JIT server, preventing concurrent model loading.
//! - **warmup_jit_models**: Pre-loads models one at a time with minimal requests.
//! - **is_jit_loading_error**: Detects JIT-specific error strings in responses.

use std::sync::Arc;

use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tracing::{info, warn};

/// Single-permit semaphore shared across all providers pointing to a JIT server.
///
/// Ensures only one LLM request is in-flight at a time, preventing concurrent
/// model switches that crash LM Studio.
#[derive(Clone)]
pub struct JitGate {
    permit: Arc<Semaphore>,
}

impl JitGate {
    pub fn new() -> Self {
        Self {
            permit: Arc::new(Semaphore::new(1)),
        }
    }

    /// Acquire the single permit. Blocks until the previous request completes.
    pub async fn acquire(&self) -> OwnedSemaphorePermit {
        self.permit
            .clone()
            .acquire_owned()
            .await
            .expect("JitGate semaphore closed unexpectedly")
    }
}

/// Detect JIT-specific loading errors in response text.
///
/// LM Studio returns these when a model is still loading or failed to load.
/// Retryability is now handled by `ProviderError::is_retryable()` in `errors.rs`.
#[cfg(test)]
pub fn is_jit_loading_error(text: &str) -> bool {
    let lower = text.to_lowercase();
    lower.contains("no models loaded")
        || lower.contains("failed to load model")
        || lower.contains("error loading model")
        || lower.contains("model is loading")
        || lower.contains("model not found")
}

/// Send a minimal `max_tokens:1` request to force JIT model loading.
///
/// Models are warmed up one at a time with a 30s timeout per model.
/// Failures are logged but don't prevent startup.
pub async fn warmup_jit_models(base_url: &str, api_key: &str, models: &[&str]) {
    let client = reqwest::Client::new();
    let url = format!("{}/chat/completions", base_url);

    // Derive the native (non-versioned) base so we can query loaded models.
    // base_url typically ends with "/v1"; strip it to reach the LMS root.
    let native_base = base_url.trim_end_matches('/').trim_end_matches("/v1");
    let loaded_ids = fetch_jit_loaded_models(native_base).await;

    for model in models {
        // Skip if already loaded — fuzzy match (either ID contains the other).
        if loaded_ids.iter().any(|id| jit_model_matches(id, model)) {
            info!("JIT warmup: '{}' already loaded, skipping", model);
            continue;
        }
        info!("JIT warmup: loading model '{}'", model);
        let body = serde_json::json!({
            "model": model,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
            "temperature": 0.0,
        });

        let result = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            client
                .post(&url)
                .header("Authorization", format!("Bearer {}", api_key))
                .header("Content-Type", "application/json")
                .json(&body)
                .send(),
        )
        .await;

        match result {
            Ok(Ok(resp)) => {
                let status = resp.status();
                if status.is_success() {
                    info!("JIT warmup: '{}' loaded OK", model);
                } else {
                    let body = resp.text().await.unwrap_or_default();
                    warn!("JIT warmup: '{}' returned HTTP {} — {}", model, status, body);
                }
            }
            Ok(Err(e)) => {
                warn!("JIT warmup: '{}' request failed — {}", model, e);
            }
            Err(_) => {
                warn!("JIT warmup: '{}' timed out after 30s", model);
            }
        }
    }
}

/// Fetch currently-loaded model IDs from an LM Studio native API endpoint.
///
/// `native_base` is the root URL without trailing slash and without `/v1`
/// (e.g. `http://host:1234`). Returns an empty vec on any error.
async fn fetch_jit_loaded_models(native_base: &str) -> Vec<String> {
    let list_url = format!("{}/api/v1/models", native_base);
    let resp = match reqwest::get(&list_url).await {
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
            let loaded = m
                .get("loaded_instances")
                .and_then(|v| v.as_array())
                .map(|a| !a.is_empty())
                .unwrap_or(false);
            if loaded { Some(key) } else { None }
        })
        .collect()
}

/// Fuzzy model identity check: matches if either ID contains the other.
fn jit_model_matches(loaded: &str, model: &str) -> bool {
    loaded == model || loaded.contains(model) || model.contains(loaded)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_jit_loading_error_positive() {
        assert!(is_jit_loading_error("No models loaded"));
        assert!(is_jit_loading_error("Error: no models loaded on this server"));
        assert!(is_jit_loading_error("Failed to load model xyz"));
        assert!(is_jit_loading_error("error loading model"));
        assert!(is_jit_loading_error("Model is loading, please wait"));
        assert!(is_jit_loading_error("model not found"));
    }

    #[test]
    fn test_is_jit_loading_error_negative() {
        assert!(!is_jit_loading_error("The answer is 42."));
        assert!(!is_jit_loading_error("HTTP 200 OK"));
        assert!(!is_jit_loading_error("rate limit exceeded"));
        assert!(!is_jit_loading_error(""));
    }

    #[tokio::test]
    async fn test_jit_gate_serialises_access() {
        let gate = JitGate::new();

        // First acquire should succeed immediately.
        let permit1 = gate.acquire().await;

        // Second acquire should block until first is dropped.
        let gate2 = gate.clone();
        let handle = tokio::spawn(async move {
            let _permit2 = gate2.acquire().await;
            42
        });

        // Give the second task time to block.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert!(!handle.is_finished(), "second acquire should be blocked");

        // Release first permit.
        drop(permit1);

        // Second task should complete.
        let result = handle.await.unwrap();
        assert_eq!(result, 42);
    }
}
