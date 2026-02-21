#![allow(dead_code)]
//! Circuit breaker for LLM provider health tracking.
//!
//! Tracks consecutive failures per provider:model key and marks providers
//! as temporarily unavailable after exceeding a failure threshold. After a
//! cooldown period the provider becomes available again for retry.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::config::schema::CircuitBreakerConfig;

/// Per-provider health state.
struct ProviderState {
    consecutive_failures: u32,
    last_failure: Option<Instant>,
}

/// Tracks provider health and trips when failures exceed a threshold.
pub struct CircuitBreaker {
    states: HashMap<String, ProviderState>,
    threshold: u32,
    cooldown: Duration,
}

impl CircuitBreaker {
    /// Create a new circuit breaker from config (defaults: 3 failures, 5 min cooldown).
    pub fn new(config: &CircuitBreakerConfig) -> Self {
        Self {
            states: HashMap::new(),
            threshold: config.threshold,
            cooldown: Duration::from_secs(config.cooldown_secs),
        }
    }

    /// Create with custom threshold and cooldown.
    pub fn with_settings(threshold: u32, cooldown: Duration) -> Self {
        Self {
            states: HashMap::new(),
            threshold,
            cooldown,
        }
    }

    /// Check if a provider:model key is available (not tripped or cooldown elapsed).
    pub fn is_available(&self, key: &str) -> bool {
        let state = match self.states.get(key) {
            Some(s) => s,
            None => return true, // never seen = available
        };

        if state.consecutive_failures < self.threshold {
            return true;
        }

        // Tripped — check if cooldown has elapsed.
        match state.last_failure {
            Some(t) => t.elapsed() >= self.cooldown,
            None => true,
        }
    }

    /// Record a successful call, resetting the failure counter.
    pub fn record_success(&mut self, key: &str) {
        if let Some(state) = self.states.get_mut(key) {
            state.consecutive_failures = 0;
            state.last_failure = None;
        }
    }

    /// Record a failed call, incrementing the failure counter.
    pub fn record_failure(&mut self, key: &str) {
        let state = self.states.entry(key.to_string()).or_insert(ProviderState {
            consecutive_failures: 0,
            last_failure: None,
        });
        state.consecutive_failures += 1;
        state.last_failure = Some(Instant::now());
    }
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self::new(&CircuitBreakerConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_provider_is_available() {
        let cb = CircuitBreaker::default();
        assert!(cb.is_available("openrouter:gpt-4"));
    }

    #[test]
    fn test_record_failure_below_threshold() {
        let mut cb = CircuitBreaker::default();
        cb.record_failure("local:qwen");
        cb.record_failure("local:qwen");
        // 2 failures < threshold of 3, still available.
        assert!(cb.is_available("local:qwen"));
    }

    #[test]
    fn test_record_failure_above_threshold_trips() {
        let mut cb = CircuitBreaker::default();
        for _ in 0..3 {
            cb.record_failure("local:qwen");
        }
        // 3 failures = threshold, should be unavailable.
        assert!(!cb.is_available("local:qwen"));
    }

    #[test]
    fn test_recovery_after_cooldown() {
        let mut cb = CircuitBreaker::with_settings(2, Duration::from_millis(10));
        cb.record_failure("bad:model");
        cb.record_failure("bad:model");
        assert!(!cb.is_available("bad:model"));

        // Wait for cooldown.
        std::thread::sleep(Duration::from_millis(15));
        assert!(cb.is_available("bad:model"));
    }

    #[test]
    fn test_record_success_resets() {
        let mut cb = CircuitBreaker::default();
        cb.record_failure("flaky:model");
        cb.record_failure("flaky:model");
        cb.record_success("flaky:model");
        // Success resets counter, so even more failures start from 0.
        cb.record_failure("flaky:model");
        cb.record_failure("flaky:model");
        assert!(cb.is_available("flaky:model")); // only 2 since last reset
    }

    #[test]
    fn test_independent_keys() {
        let mut cb = CircuitBreaker::default();
        for _ in 0..3 {
            cb.record_failure("bad:model");
        }
        assert!(!cb.is_available("bad:model"));
        assert!(cb.is_available("good:model")); // different key is fine
    }
}
