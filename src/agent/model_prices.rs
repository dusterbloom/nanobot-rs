//! Model price fetching and caching from OpenRouter.
//!
//! Fetches per-token pricing for all models from the OpenRouter public API
//! (no auth required) and caches locally. Used by the tool runner to enforce
//! cost budgets on RLM delegation loops.

use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

/// Cached model prices with timestamp.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPrices {
    /// Map of model_id -> (prompt_cost_per_token, completion_cost_per_token).
    pub prices: HashMap<String, (f64, f64)>,
    /// Unix timestamp when prices were fetched.
    pub fetched_at: i64,
}

/// How old the cache can be before we re-fetch (24 hours).
const CACHE_MAX_AGE_SECS: i64 = 86400;

/// OpenRouter API response structures (minimal).
#[derive(Deserialize)]
struct OpenRouterModelsResponse {
    data: Vec<OpenRouterModel>,
}

#[derive(Deserialize)]
struct OpenRouterModel {
    id: String,
    pricing: Option<OpenRouterPricing>,
}

#[derive(Deserialize)]
struct OpenRouterPricing {
    prompt: Option<String>,
    completion: Option<String>,
}

impl ModelPrices {
    /// Create an empty price map.
    pub fn empty() -> Self {
        Self {
            prices: HashMap::new(),
            fetched_at: 0,
        }
    }

    /// Get the cache file path.
    fn cache_path() -> Option<PathBuf> {
        dirs::home_dir().map(|h| h.join(".nanobot").join("cache").join("model_prices.json"))
    }

    /// Load prices from cache, re-fetching if stale or missing.
    /// Falls back to empty prices on any error (non-blocking).
    pub async fn load() -> Self {
        // Try loading from cache first.
        if let Some(path) = Self::cache_path() {
            if let Ok(data) = tokio::fs::read_to_string(&path).await {
                if let Ok(cached) = serde_json::from_str::<ModelPrices>(&data) {
                    let now = chrono::Utc::now().timestamp();
                    if now - cached.fetched_at < CACHE_MAX_AGE_SECS {
                        debug!("Model prices loaded from cache ({} models)", cached.prices.len());
                        return cached;
                    }
                    debug!("Model price cache is stale, re-fetching");
                }
            }
        }

        // Fetch fresh prices.
        match Self::fetch().await {
            Ok(prices) => {
                // Save to cache (best-effort).
                if let Some(path) = Self::cache_path() {
                    if let Some(parent) = path.parent() {
                        let _ = tokio::fs::create_dir_all(parent).await;
                    }
                    if let Ok(json) = serde_json::to_string(&prices) {
                        let _ = tokio::fs::write(&path, json).await;
                    }
                }
                debug!("Fetched {} model prices from OpenRouter", prices.prices.len());
                prices
            }
            Err(e) => {
                warn!("Failed to fetch model prices: {} — using empty prices", e);
                Self::empty()
            }
        }
    }

    /// Fetch prices from OpenRouter API.
    pub async fn fetch() -> Result<Self, String> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .map_err(|e| format!("HTTP client error: {}", e))?;

        let resp = client
            .get("https://openrouter.ai/api/v1/models")
            .header("User-Agent", "nanobot")
            .send()
            .await
            .map_err(|e| format!("Request failed: {}", e))?;

        if !resp.status().is_success() {
            return Err(format!("HTTP {}", resp.status()));
        }

        let body: OpenRouterModelsResponse = resp
            .json()
            .await
            .map_err(|e| format!("Parse error: {}", e))?;

        let mut prices = HashMap::new();
        for model in body.data {
            if let Some(pricing) = model.pricing {
                let prompt = pricing
                    .prompt
                    .as_deref()
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.0)
                    .max(0.0); // Clamp negative prices (e.g. credit/reward models)
                let completion = pricing
                    .completion
                    .as_deref()
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.0)
                    .max(0.0);
                prices.insert(model.id, (prompt, completion));
            }
        }

        Ok(Self {
            prices,
            fetched_at: chrono::Utc::now().timestamp(),
        })
    }

    /// Calculate cost for a given model and token counts.
    /// Returns 0.0 if model not found (local models, unknown models).
    pub fn cost_of(&self, model: &str, prompt_tokens: i64, completion_tokens: i64) -> f64 {
        if let Some(&(prompt_price, completion_price)) = self.prices.get(model) {
            (prompt_tokens as f64) * prompt_price + (completion_tokens as f64) * completion_price
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cost_of_known_model() {
        let mut prices = ModelPrices::empty();
        // GLM-4.5-air: $0.13/MTok prompt, $0.85/MTok completion
        // Per-token: 0.00000013, 0.00000085
        prices.prices.insert(
            "z-ai/glm-4.5-air".to_string(),
            (0.00000013, 0.00000085),
        );

        // 1000 prompt tokens, 500 completion tokens
        let cost = prices.cost_of("z-ai/glm-4.5-air", 1000, 500);
        // Expected: 1000 * 0.00000013 + 500 * 0.00000085 = 0.00013 + 0.000425 = 0.000555
        assert!((cost - 0.000555).abs() < 1e-9, "cost was {}", cost);
    }

    #[test]
    fn test_cost_of_unknown_model() {
        let prices = ModelPrices::empty();
        let cost = prices.cost_of("local/my-model", 10000, 5000);
        assert_eq!(cost, 0.0);
    }

    #[test]
    fn test_cost_of_zero_tokens() {
        let mut prices = ModelPrices::empty();
        prices.prices.insert("test/model".to_string(), (0.001, 0.002));
        assert_eq!(prices.cost_of("test/model", 0, 0), 0.0);
    }

    #[test]
    fn test_cost_of_opus() {
        let mut prices = ModelPrices::empty();
        // Opus: $5/MTok prompt, $25/MTok completion
        prices.prices.insert(
            "anthropic/claude-opus-4.6".to_string(),
            (0.000005, 0.000025),
        );

        // 1000 prompt, 100 completion
        let cost = prices.cost_of("anthropic/claude-opus-4.6", 1000, 100);
        // 1000 * 0.000005 + 100 * 0.000025 = 0.005 + 0.0025 = 0.0075
        assert!((cost - 0.0075).abs() < 1e-9, "cost was {}", cost);
    }

    #[test]
    fn test_empty_prices() {
        let prices = ModelPrices::empty();
        assert!(prices.prices.is_empty());
        assert_eq!(prices.fetched_at, 0);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let mut prices = ModelPrices::empty();
        prices.prices.insert("test/a".to_string(), (0.001, 0.002));
        prices.prices.insert("test/b".to_string(), (0.0, 0.005));
        prices.fetched_at = 1700000000;

        let json = serde_json::to_string(&prices).unwrap();
        let parsed: ModelPrices = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.prices.len(), 2);
        assert_eq!(parsed.prices["test/a"], (0.001, 0.002));
        assert_eq!(parsed.fetched_at, 1700000000);
    }

    #[tokio::test]
    async fn test_fetch_live() {
        // Integration test — actually hits OpenRouter API.
        // Skip in CI by checking env var.
        if std::env::var("CI").is_ok() {
            return;
        }

        let result = ModelPrices::fetch().await;
        match result {
            Ok(prices) => {
                assert!(prices.prices.len() > 100, "Expected 100+ models, got {}", prices.prices.len());
                // Verify a known model exists.
                assert!(prices.prices.contains_key("anthropic/claude-opus-4.6")
                    || prices.prices.contains_key("anthropic/claude-3.5-sonnet"),
                    "Should contain at least one Anthropic model");
                // Verify prices are non-negative.
                for (model, (p, c)) in &prices.prices {
                    assert!(*p >= 0.0, "Negative prompt price for {}", model);
                    assert!(*c >= 0.0, "Negative completion price for {}", model);
                }
            }
            Err(e) => {
                // Network might not be available — don't fail hard.
                eprintln!("Fetch failed (expected in offline env): {}", e);
            }
        }
    }
}
