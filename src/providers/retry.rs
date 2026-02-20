//! Shared retry helpers for LLM providers.
//!
//! Provides backoff configurations and a rate-limit-aware delay adjuster
//! for use with `backon::Retryable`.

use std::time::Duration;

use backon::ExponentialBuilder;

use crate::errors::ProviderError;

/// Standard backoff for cloud providers: 1s → 2s → 4s … capped at 30s, 3 retries, with jitter.
pub fn provider_backoff() -> ExponentialBuilder {
    ExponentialBuilder::new()
        .with_min_delay(Duration::from_secs(1))
        .with_max_delay(Duration::from_secs(30))
        .with_factor(2.0)
        .with_jitter()
        .with_max_times(3)
}

/// JIT-specific backoff for local model loading: 2s → 4s → 8s, 3 retries, with jitter.
pub fn jit_backoff() -> ExponentialBuilder {
    ExponentialBuilder::new()
        .with_min_delay(Duration::from_secs(2))
        .with_max_delay(Duration::from_secs(8))
        .with_factor(2.0)
        .with_jitter()
        .with_max_times(3)
}

/// If the error is `RateLimited`, ensure the delay is at least `retry_after_ms`.
///
/// Signature matches `backon::Retry::adjust`: returning `None` aborts the retry.
pub fn adjust_for_rate_limit(
    err: &ProviderError,
    dur: Option<Duration>,
) -> Option<Duration> {
    match (err, dur) {
        (ProviderError::RateLimited { retry_after_ms, .. }, Some(d)) => {
            let rate_limit_delay = Duration::from_millis(*retry_after_ms);
            Some(d.max(rate_limit_delay))
        }
        (_, dur) => dur,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adjust_rate_limited_uses_max() {
        let err = ProviderError::RateLimited { status: 429, retry_after_ms: 5000 };
        // Backoff suggests 1s, but rate limit says 5s → use 5s.
        let result = adjust_for_rate_limit(&err, Some(Duration::from_secs(1)));
        assert_eq!(result, Some(Duration::from_secs(5)));
    }

    #[test]
    fn test_adjust_rate_limited_backoff_already_larger() {
        let err = ProviderError::RateLimited { status: 429, retry_after_ms: 500 };
        // Backoff suggests 2s, rate limit says 0.5s → keep 2s.
        let result = adjust_for_rate_limit(&err, Some(Duration::from_secs(2)));
        assert_eq!(result, Some(Duration::from_secs(2)));
    }

    #[test]
    fn test_adjust_non_rate_limited_passes_through() {
        let err = ProviderError::ServerError { status: 503, message: "overloaded".into() };
        let result = adjust_for_rate_limit(&err, Some(Duration::from_secs(1)));
        assert_eq!(result, Some(Duration::from_secs(1)));
    }

    #[test]
    fn test_adjust_none_passes_through() {
        let err = ProviderError::ServerError { status: 500, message: "error".into() };
        let result = adjust_for_rate_limit(&err, None);
        assert_eq!(result, None);
    }
}
