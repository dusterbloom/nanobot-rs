//! Domain error types for nanobot.
//!
//! Typed errors at module boundaries replace string-encoded errors and
//! enable structured error handling via pattern matching.

#![allow(clippy::disallowed_types)] // anyhow is the app convention — the ban targets tool boundaries (error protocol §2.5)
use thiserror::Error;

// ---------------------------------------------------------------------------
// Provider errors
// ---------------------------------------------------------------------------

/// Errors from LLM provider operations.
///
/// Embedded in `anyhow::Error` so the `LLMProvider` trait signature
/// (`-> anyhow::Result<LLMResponse>`) stays unchanged while callers
/// can downcast: `e.downcast_ref::<ProviderError>()`.
#[derive(Debug, Error)]
pub enum ProviderError {
    #[error("HTTP request failed: {0}")]
    HttpError(String),

    #[error("Failed to read response body: {0}")]
    ResponseReadError(String),

    #[error("Failed to parse response JSON: {0}")]
    JsonParseError(String),

    #[error("Rate limited (status {status}): retry after {retry_after_ms}ms")]
    RateLimited { status: u16, retry_after_ms: u64 },

    #[error("Authentication failed (status {status}): {message}")]
    AuthError { status: u16, message: String },

    #[error("Server error (status {status}): {message}")]
    ServerError { status: u16, message: String },

    #[error("Request cancelled")]
    Cancelled,
}

impl ProviderError {
    /// Whether this error is transient and the request should be retried.
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::RateLimited { .. } => true,
            Self::ServerError { .. } => true,
            Self::HttpError(msg) => is_transient_http_error(msg),
            Self::ResponseReadError(_)
            | Self::JsonParseError(_)
            | Self::AuthError { .. }
            | Self::Cancelled => false,
        }
    }
}

/// Downcast an `anyhow::Error` and check retryability.
pub fn is_retryable_provider_error(err: &anyhow::Error) -> bool {
    err.downcast_ref::<ProviderError>()
        .map_or(false, |pe| pe.is_retryable())
}

/// Check if an HTTP error message indicates a transient/retryable condition.
fn is_transient_http_error(msg: &str) -> bool {
    let lower = msg.to_lowercase();
    // Connection errors
    lower.contains("connection refused")
        || lower.contains("connection reset")
        || lower.contains("timed out")
        || lower.contains("timeout")
        || lower.contains("broken pipe")
        // JIT model loading errors (LM Studio) — excludes "model not found"
        // which is ambiguous (could be a config typo on cloud APIs).
        // JIT "model not found" during loading surfaces as 5xx → ServerError
        // which is already retryable.
        || lower.contains("no models loaded")
        || lower.contains("failed to load model")
        || lower.contains("error loading model")
        || lower.contains("model is loading")
}

// ---------------------------------------------------------------------------
// Tool error classification
// ---------------------------------------------------------------------------

/// Categorised tool failure reasons.
///
/// Produced by [`classify_tool_error`] from the error string that tools
/// currently return via the `"Error: ..."` prefix convention.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ToolErrorKind {
    #[error("Command timed out after {0}s")]
    Timeout(u64),

    #[error("Permission denied: {0}")]
    PermissionDenied(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Invalid arguments: {0}")]
    InvalidArgs(String),

    #[error("Tool not found: {0}")]
    ToolNotFound(String),

    #[error("Execution failed: {0}")]
    ExecutionFailed(String),

    #[error("Network error: {0}")]
    NetworkError(String),

    #[error("Rate limited")]
    RateLimited,

    #[error("Service unavailable: {0}")]
    ServiceUnavailable(String),

    #[error("Missing required argument '{param}'; call as {example}")]
    MissingArg { param: String, example: String },
}

impl ToolErrorKind {
    /// Whether this tool error is transient and the operation should be retried.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::Timeout(_)
                | Self::NetworkError(_)
                | Self::RateLimited
                | Self::ServiceUnavailable(_)
        )
    }
}

/// Classify a tool error string into a structured [`ToolErrorKind`].
///
/// Matches on known substrings in the error message. Returns `None` for
/// unrecognised patterns (the caller still has the raw string).
pub fn classify_tool_error(error_msg: &str) -> Option<ToolErrorKind> {
    let lower = error_msg.to_lowercase();

    if lower.contains("timed out") || lower.contains("timeout") {
        // Try to extract the timeout duration.
        let secs = extract_timeout_secs(&lower).unwrap_or(0);
        return Some(ToolErrorKind::Timeout(secs));
    }

    if lower.contains("permission denied") {
        return Some(ToolErrorKind::PermissionDenied(error_msg.to_string()));
    }

    // Network errors (check before "not found" which would false-match).
    if lower.contains("connection refused")
        || lower.contains("connection reset")
        || lower.contains("check network")
        || lower.contains("dns")
        || lower.contains("broken pipe")
    {
        return Some(ToolErrorKind::NetworkError(error_msg.to_string()));
    }

    // Rate limiting.
    if lower.contains("rate limit")
        || lower.contains("429")
        || lower.contains("quota exceeded")
        || lower.contains("too many requests")
    {
        return Some(ToolErrorKind::RateLimited);
    }

    // Service unavailable.
    if lower.contains("503")
        || lower.contains("service unavailable")
        || lower.contains("no models loaded")
        || lower.contains("model is loading")
    {
        return Some(ToolErrorKind::ServiceUnavailable(error_msg.to_string()));
    }

    if lower.contains("no such file")
        || lower.contains("not found")
        || lower.contains("does not exist")
    {
        return Some(ToolErrorKind::NotFound(error_msg.to_string()));
    }

    if lower.contains("invalid") || lower.contains("missing required") || lower.contains("expected")
    {
        return Some(ToolErrorKind::InvalidArgs(error_msg.to_string()));
    }

    if lower.contains("unknown tool") || lower.contains("tool not found") {
        return Some(ToolErrorKind::ToolNotFound(error_msg.to_string()));
    }

    None
}

/// Try to extract a numeric timeout value from an error message.
fn extract_timeout_secs(msg: &str) -> Option<u64> {
    // Pattern: "timed out after 30 seconds" or "timeout after 30s"
    let patterns = ["after ", "timeout "];
    for pat in &patterns {
        if let Some(pos) = msg.find(pat) {
            let after = &msg[pos + pat.len()..];
            let num_str: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
            if let Ok(n) = num_str.parse::<u64>() {
                return Some(n);
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Typed tool error protocol (docs/research/2026-08-06-error-conventions-and-host-bridge.md §2)
// ---------------------------------------------------------------------------

/// The single typed failure for the tool layer.
///
/// Every failure that crosses a tool boundary is this enum: never a bare
/// string, never `anyhow::Error`, never a struct with an `ok: bool` hole.
/// Severity/action axes are collapsed into one enum so a single `match`
/// gives retryability, model-fixability, and infra attribution.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ToolError {
    // ---- model-recoverable (the model can repair its own call) ----
    #[error("Missing required argument '{param}'; call as {example}")]
    MissingArg { param: String, example: String },

    #[error("Invalid arguments: {message}")]
    InvalidArgs { message: String },

    // ---- infra / policy (the model cannot fix these) ----
    #[error("Tool '{name}' not found")]
    ToolNotFound { name: String },

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Permission denied: {0}")]
    PermissionDenied(String),

    // ---- transient (safe to retry) ----
    #[error("Command timed out after {0}s")]
    Timeout(u64),

    #[error("Network error: {0}")]
    Network(String),

    #[error("Rate limited")]
    RateLimited,

    #[error("Service unavailable: {0}")]
    ServiceUnavailable(String),

    /// Everything else. The registry converts panics here; unmigrated tools
    /// funnel legacy `"Error: ..."` strings through [`Self::from_legacy`] —
    /// the *only* string-to-error path left in the codebase.
    #[error("Execution failed: {message}")]
    Execution { message: String },
}

impl ToolError {
    /// Transient ⇒ the tool runner may retry with backoff.
    /// Mirrors today's `ToolErrorKind::is_retryable`.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::Timeout(_) | Self::Network(_) | Self::RateLimited | Self::ServiceUnavailable(_)
        )
    }

    /// Model-recoverable ⇒ the loop re-injects the call with a corrected
    /// shape instead of failing the turn. Replaces the `MissingArg` special
    /// case in the registry and the `"is required"` string fallback.
    pub fn is_model_fixable(&self) -> bool {
        matches!(self, Self::MissingArg { .. } | Self::InvalidArgs { .. })
    }

    /// The exact wire string the model sees. Byte-stable with today's
    /// `"Error: ..."` convention so `tool_result_ok` and the exact-substring
    /// tests keep passing unmodified.
    ///
    /// The message-carrying variants render the payload verbatim because
    /// [`Self::from_legacy`] passes the *full* original message through —
    /// `render()` reproduces the pre-migration byte string exactly.
    pub fn render(&self) -> String {
        match self {
            Self::MissingArg { param, example } => {
                format!("Error: '{}' parameter is required; call as {}", param, example)
            }
            Self::Execution { message }
            | Self::InvalidArgs { message }
            | Self::NotFound(message)
            | Self::PermissionDenied(message)
            | Self::ToolNotFound { name: message }
            | Self::Network(message)
            | Self::ServiceUnavailable(message) => format!("Error: {message}"),
            Self::Timeout(secs) => format!("Error: Command timed out after {secs}s"),
            Self::RateLimited => format!("Error: Rate limited"),
        }
    }

    /// The single legacy bridge. Called only by the migration adapter
    /// (default `Tool::execute_typed`). Maps the exact strings
    /// `classify_tool_error` matched today, so retry behavior is preserved.
    #[allow(clippy::disallowed_methods)] // legacy bridge — deleted in Phase 3
    pub fn from_legacy(msg: &str) -> Self {
        if let Some(kind) = classify_tool_error(msg) {
            return match kind {
                ToolErrorKind::Timeout(s) => Self::Timeout(s),
                ToolErrorKind::PermissionDenied(m) => Self::PermissionDenied(m),
                ToolErrorKind::NotFound(m) => Self::NotFound(m),
                ToolErrorKind::InvalidArgs(m) => Self::InvalidArgs { message: m },
                ToolErrorKind::ToolNotFound(m) => Self::ToolNotFound { name: m },
                ToolErrorKind::ExecutionFailed(m) => Self::Execution { message: m },
                ToolErrorKind::NetworkError(m) => Self::Network(m),
                ToolErrorKind::RateLimited => Self::RateLimited,
                ToolErrorKind::ServiceUnavailable(m) => Self::ServiceUnavailable(m),
                ToolErrorKind::MissingArg { param, example } => Self::MissingArg { param, example },
            };
        }
        Self::Execution { message: msg.to_string() }
    }
}

/// Reverse bridge: map a typed [`ToolError`] back to the legacy
/// [`ToolErrorKind`] so existing consumers (retry, audit, the registry's
/// worked-example append) keep working until Phase 3 deletes the legacy
/// taxonomy. `Execution` has no legacy kind — it is the "everything else"
/// bucket.
pub fn legacy_kind_from_tool_error(e: &ToolError) -> Option<ToolErrorKind> {
    match e {
        ToolError::MissingArg { param, example } => Some(ToolErrorKind::MissingArg {
            param: param.clone(),
            example: example.clone(),
        }),
        ToolError::InvalidArgs { message } => Some(ToolErrorKind::InvalidArgs(message.clone())),
        ToolError::ToolNotFound { name } => Some(ToolErrorKind::ToolNotFound(name.clone())),
        ToolError::NotFound(m) => Some(ToolErrorKind::NotFound(m.clone())),
        ToolError::PermissionDenied(m) => Some(ToolErrorKind::PermissionDenied(m.clone())),
        ToolError::Timeout(s) => Some(ToolErrorKind::Timeout(*s)),
        ToolError::Network(m) => Some(ToolErrorKind::NetworkError(m.clone())),
        ToolError::RateLimited => Some(ToolErrorKind::RateLimited),
        ToolError::ServiceUnavailable(m) => Some(ToolErrorKind::ServiceUnavailable(m.clone())),
        ToolError::Execution { .. } => None,
    }
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // tests pin classify_tool_error's mapping; deleted in Phase 3
mod tests {
    use super::*;

    // -- ProviderError tests --

    #[test]
    fn test_provider_error_display() {
        let e = ProviderError::HttpError("connection refused".into());
        assert_eq!(e.to_string(), "HTTP request failed: connection refused");
    }

    #[test]
    fn test_provider_error_rate_limited() {
        let e = ProviderError::RateLimited {
            status: 429,
            retry_after_ms: 5000,
        };
        assert!(e.to_string().contains("429"));
        assert!(e.to_string().contains("5000"));
    }

    #[test]
    fn test_provider_error_downcast() {
        let anyhow_err: anyhow::Error = ProviderError::AuthError {
            status: 401,
            message: "invalid key".into(),
        }
        .into();
        let downcasted = anyhow_err.downcast_ref::<ProviderError>();
        assert!(downcasted.is_some());
        assert!(matches!(
            downcasted.unwrap(),
            ProviderError::AuthError { status: 401, .. }
        ));
    }

    // -- classify_tool_error tests --

    #[test]
    fn test_classify_timeout() {
        let kind = classify_tool_error("Command timed out after 30 seconds");
        assert_eq!(kind, Some(ToolErrorKind::Timeout(30)));
    }

    #[test]
    fn test_classify_timeout_no_duration() {
        let kind = classify_tool_error("Operation timeout");
        assert_eq!(kind, Some(ToolErrorKind::Timeout(0)));
    }

    #[test]
    fn test_classify_permission_denied() {
        let kind = classify_tool_error("Permission denied: /etc/shadow");
        assert!(matches!(kind, Some(ToolErrorKind::PermissionDenied(_))));
    }

    #[test]
    fn test_classify_not_found() {
        let kind = classify_tool_error("No such file or directory: /tmp/missing");
        assert!(matches!(kind, Some(ToolErrorKind::NotFound(_))));
    }

    #[test]
    fn test_classify_not_found_variant() {
        let kind = classify_tool_error("File does not exist: README.md");
        assert!(matches!(kind, Some(ToolErrorKind::NotFound(_))));
    }

    #[test]
    fn test_classify_invalid_args() {
        let kind = classify_tool_error("Invalid path argument: cannot be empty");
        assert!(matches!(kind, Some(ToolErrorKind::InvalidArgs(_))));
    }

    #[test]
    fn test_classify_missing_required() {
        let kind = classify_tool_error("Missing required parameter: command");
        assert!(matches!(kind, Some(ToolErrorKind::InvalidArgs(_))));
    }

    #[test]
    fn test_classify_tool_not_found() {
        let kind = classify_tool_error("Unknown tool: magic_wand");
        assert!(matches!(kind, Some(ToolErrorKind::ToolNotFound(_))));
    }

    #[test]
    fn test_classify_unknown_error() {
        let kind = classify_tool_error("Something went wrong in an unusual way");
        assert_eq!(kind, None);
    }

    #[test]
    fn test_classify_case_insensitive() {
        let kind = classify_tool_error("PERMISSION DENIED accessing /root");
        assert!(matches!(kind, Some(ToolErrorKind::PermissionDenied(_))));
    }

    // -- extract_timeout_secs tests --

    #[test]
    fn test_extract_timeout_after_pattern() {
        assert_eq!(extract_timeout_secs("timed out after 60 seconds"), Some(60));
    }

    #[test]
    fn test_extract_timeout_no_number() {
        assert_eq!(extract_timeout_secs("timed out after many seconds"), None);
    }

    // -- is_retryable tests --

    #[test]
    fn test_retryable_rate_limited() {
        let e = ProviderError::RateLimited {
            status: 429,
            retry_after_ms: 1000,
        };
        assert!(e.is_retryable());
    }

    #[test]
    fn test_retryable_server_error() {
        let e = ProviderError::ServerError {
            status: 503,
            message: "overloaded".into(),
        };
        assert!(e.is_retryable());
    }

    #[test]
    fn test_retryable_http_connection_refused() {
        let e = ProviderError::HttpError("Error calling LLM: connection refused".into());
        assert!(e.is_retryable());
    }

    #[test]
    fn test_retryable_http_timeout() {
        let e = ProviderError::HttpError("request timed out".into());
        assert!(e.is_retryable());
    }

    #[test]
    fn test_retryable_http_jit_loading() {
        let e = ProviderError::HttpError("no models loaded on this server".into());
        assert!(e.is_retryable());
    }

    #[test]
    fn test_retryable_http_model_is_loading() {
        let e = ProviderError::HttpError("Model is loading, please wait".into());
        assert!(e.is_retryable());
    }

    #[test]
    fn test_not_retryable_model_not_found() {
        // "model not found" should NOT be retried — could be a config typo.
        // JIT loading errors surface as 5xx (ServerError) which is already retryable.
        let e = ProviderError::HttpError("HTTP 404: model not found".into());
        assert!(!e.is_retryable());
    }

    #[test]
    fn test_not_retryable_auth() {
        let e = ProviderError::AuthError {
            status: 401,
            message: "invalid key".into(),
        };
        assert!(!e.is_retryable());
    }

    #[test]
    fn test_not_retryable_json_parse() {
        let e = ProviderError::JsonParseError("unexpected token".into());
        assert!(!e.is_retryable());
    }

    #[test]
    fn test_not_retryable_cancelled() {
        assert!(!ProviderError::Cancelled.is_retryable());
    }

    #[test]
    fn test_not_retryable_generic_http() {
        let e = ProviderError::HttpError("HTTP 400: bad request".into());
        assert!(!e.is_retryable());
    }

    #[test]
    fn test_is_retryable_provider_error_downcast() {
        let anyhow_err: anyhow::Error = ProviderError::RateLimited {
            status: 429,
            retry_after_ms: 5000,
        }
        .into();
        assert!(is_retryable_provider_error(&anyhow_err));
    }

    #[test]
    fn test_is_retryable_provider_error_non_provider() {
        let anyhow_err = anyhow::anyhow!("some random error");
        assert!(!is_retryable_provider_error(&anyhow_err));
    }

    // -- ToolErrorKind classification: network/rate-limit/service-unavailable --

    #[test]
    fn test_classify_network_error() {
        let kind = classify_tool_error("connection refused by remote host");
        assert!(matches!(kind, Some(ToolErrorKind::NetworkError(_))));
    }

    #[test]
    fn test_classify_rate_limited() {
        let kind = classify_tool_error("429 rate limit exceeded");
        assert_eq!(kind, Some(ToolErrorKind::RateLimited));
    }

    #[test]
    fn test_classify_service_unavailable() {
        let kind = classify_tool_error("service unavailable, try again later");
        assert!(matches!(kind, Some(ToolErrorKind::ServiceUnavailable(_))));
    }

    // -- ToolErrorKind::is_retryable --

    #[test]
    fn test_retryable_tool_error_variants() {
        assert!(ToolErrorKind::Timeout(30).is_retryable());
        assert!(ToolErrorKind::NetworkError("conn reset".into()).is_retryable());
        assert!(ToolErrorKind::RateLimited.is_retryable());
        assert!(ToolErrorKind::ServiceUnavailable("503".into()).is_retryable());
    }

    #[test]
    fn test_not_retryable_tool_error_variants() {
        assert!(!ToolErrorKind::NotFound("missing".into()).is_retryable());
        assert!(!ToolErrorKind::PermissionDenied("denied".into()).is_retryable());
        assert!(!ToolErrorKind::InvalidArgs("bad arg".into()).is_retryable());
        assert!(!ToolErrorKind::ToolNotFound("no_tool".into()).is_retryable());
        assert!(!ToolErrorKind::ExecutionFailed("fail".into()).is_retryable());
    }

    // -- ToolError (typed protocol) --

    #[test]
    fn test_tool_error_render_preserves_legacy_bytes() {
        // Unclassified legacy strings funnel through Execution and must come
        // out byte-identical ("'task' parameter is required", "Spawn callback
        // not configured" etc. — the 297-site contract).
        let e = ToolError::from_legacy("'task' parameter is required");
        assert_eq!(e.render(), "Error: 'task' parameter is required");

        let e = ToolError::from_legacy("Spawn callback not configured");
        assert_eq!(e.render(), "Error: Spawn callback not configured");

        // Classified payload-carrying variants reproduce the full original
        // message (the payload IS the stripped message).
        let e = ToolError::from_legacy("File not found: /tmp/missing");
        assert_eq!(e.render(), "Error: File not found: /tmp/missing");
        assert!(matches!(e, ToolError::NotFound(_)));

        let e = ToolError::from_legacy("Permission denied: /etc/shadow");
        assert_eq!(e.render(), "Error: Permission denied: /etc/shadow");

        let e = ToolError::from_legacy("Invalid path argument: cannot be empty");
        assert_eq!(e.render(), "Error: Invalid path argument: cannot be empty");
    }

    #[test]
    fn test_tool_error_from_legacy_classification() {
        assert!(matches!(
            ToolError::from_legacy("Command timed out after 30 seconds"),
            ToolError::Timeout(30)
        ));
        assert!(matches!(
            ToolError::from_legacy("connection refused by remote host"),
            ToolError::Network(_)
        ));
        assert!(matches!(
            ToolError::from_legacy("429 rate limit exceeded"),
            ToolError::RateLimited
        ));
        assert!(matches!(
            ToolError::from_legacy("service unavailable, try again later"),
            ToolError::ServiceUnavailable(_)
        ));
        assert!(matches!(
            ToolError::from_legacy("Unknown tool: magic_wand"),
            ToolError::ToolNotFound { .. }
        ));
        assert!(matches!(
            ToolError::from_legacy("No such file or directory: /x"),
            ToolError::NotFound(_)
        ));
        assert!(matches!(
            ToolError::from_legacy("unusual failure"),
            ToolError::Execution { .. }
        ));
    }

    #[test]
    fn test_tool_error_structural_classification() {
        // Retryable.
        assert!(ToolError::Timeout(30).is_retryable());
        assert!(ToolError::Network("conn reset".into()).is_retryable());
        assert!(ToolError::RateLimited.is_retryable());
        assert!(ToolError::ServiceUnavailable("503".into()).is_retryable());
        // Model-fixable.
        assert!(ToolError::MissingArg {
            param: "query".into(),
            example: "recall({\"query\":\"...\"})".into(),
        }
        .is_model_fixable());
        assert!(ToolError::InvalidArgs { message: "bad".into() }.is_model_fixable());
        // Neither.
        assert!(!ToolError::NotFound("x".into()).is_retryable());
        assert!(!ToolError::NotFound("x".into()).is_model_fixable());
        assert!(!ToolError::Execution { message: "boom".into() }.is_retryable());
    }

    #[test]
    fn test_legacy_kind_from_tool_error_round_trip() {
        let e = ToolError::MissingArg {
            param: "facts".into(),
            example: "remember({\"facts\":[\"a concise fact\"]})".into(),
        };
        assert!(matches!(
            legacy_kind_from_tool_error(&e),
            Some(ToolErrorKind::MissingArg { ref param, .. }) if param == "facts"
        ));
        assert!(matches!(
            legacy_kind_from_tool_error(&ToolError::Timeout(42)),
            Some(ToolErrorKind::Timeout(42))
        ));
        assert_eq!(
            legacy_kind_from_tool_error(&ToolError::Execution { message: "x".into() }),
            None
        );
    }
}
