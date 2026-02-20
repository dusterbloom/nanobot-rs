//! Domain error types for nanobot.
//!
//! Typed errors at module boundaries replace string-encoded errors and
//! enable structured error handling via pattern matching.

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

    if lower.contains("no such file")
        || lower.contains("not found")
        || lower.contains("does not exist")
    {
        return Some(ToolErrorKind::NotFound(error_msg.to_string()));
    }

    if lower.contains("invalid")
        || lower.contains("missing required")
        || lower.contains("expected")
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

#[cfg(test)]
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
        let anyhow_err: anyhow::Error =
            ProviderError::AuthError {
                status: 401,
                message: "invalid key".into(),
            }
            .into();
        let downcasted = anyhow_err.downcast_ref::<ProviderError>();
        assert!(downcasted.is_some());
        assert!(matches!(downcasted.unwrap(), ProviderError::AuthError { status: 401, .. }));
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
}
