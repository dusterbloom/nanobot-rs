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

    for model in models {
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
