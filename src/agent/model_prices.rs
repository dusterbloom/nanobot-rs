// Error-protocol layer-3 backlog (docs/research/2026-08-06-error-conventions-and-host-bridge.md §3.6):
// the deny regime in Cargo.toml is live; this module still carries pre-existing
// violations of the lints below. Remove this allow as the module migrates onto
// the regime.
// Tracking: docs/error-protocol-backlog.md
#![allow(clippy::as_conversions)]
//! Per-token model pricing used by the tool runner to enforce cost
//! budgets on RLM delegation loops.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Cached model prices with timestamp.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPrices {
    /// Map of model_id -> (prompt_cost_per_token, completion_cost_per_token).
    pub prices: HashMap<String, (f64, f64)>,
    /// Unix timestamp when prices were fetched.
    pub fetched_at: i64,
}

impl ModelPrices {
    /// Create an empty price map.
    #[cfg(test)]
    pub fn empty() -> Self {
        Self {
            prices: HashMap::new(),
            fetched_at: 0,
        }
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
        prices
            .prices
            .insert("z-ai/glm-4.5-air".to_string(), (0.00000013, 0.00000085));

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
        prices
            .prices
            .insert("test/model".to_string(), (0.001, 0.002));
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
}
