pub mod loader;
pub mod schema;

/// Redact a secret for Debug output: empty → `"[empty]"`, non-empty → `"[REDACTED len=N]"`.
pub fn redact(s: &str) -> String {
    if s.is_empty() {
        "[empty]".into()
    } else {
        format!("[REDACTED len={}]", s.len())
    }
}

/// Redact an optional secret for Debug output.
pub fn redact_opt(s: &Option<String>) -> String {
    match s {
        None => "[none]".into(),
        Some(s) => redact(s),
    }
}
