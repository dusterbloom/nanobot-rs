//! Cross-session continuity: a one-line summary of the previous session,
//! injected once into the system prompt of a fresh session so the agent can
//! answer "continue from the previous session" deterministically.
//!
//! Pure functions only — the DB access lives in `session::db`
//! (`latest_session_tails`) and the wiring in `prepare_context`.

use chrono::{DateTime, Duration, Utc};

use crate::session::db::SessionTail;

/// How the current session came into existence, from the DB's point of view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SessionStart {
    /// No prior messages under this session id — a brand-new session.
    Fresh,
    /// The session already has history (resumed via `--resume`/`--continue`,
    /// or simply any turn after the first).
    Resumed,
}

/// Classify a session by the number of messages it had BEFORE this turn.
pub(crate) fn classify_session_start(prior_history_len: usize) -> SessionStart {
    match prior_history_len {
        0 => SessionStart::Fresh,
        _ => SessionStart::Resumed,
    }
}

/// Human-readable age: "just now", "5m ago", "3h ago", "2d ago".
pub(crate) fn humanize_age(age: Duration) -> String {
    if age.num_days() >= 1 {
        return format!("{}d ago", age.num_days());
    }
    if age.num_hours() >= 1 {
        return format!("{}h ago", age.num_hours());
    }
    if age.num_minutes() >= 1 {
        return format!("{}m ago", age.num_minutes());
    }
    "just now".to_string()
}

/// Truncate to at most `max` characters (char-boundary safe), appending "…".
pub(crate) fn truncate_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let cut: String = s.chars().take(max).collect();
    format!("{cut}…")
}

/// One-line summary of a past session's final exchange.
///
/// `Previous session (<age>, key <key>): <last user> → <assistant tail>`
pub(crate) fn format_continuity_line(tail: &SessionTail, now: DateTime<Utc>) -> String {
    let age = humanize_age(now - tail.updated_at);
    let user = truncate_chars(tail.last_user.trim(), 200);
    let assistant = if tail.last_assistant.trim().is_empty() {
        "(no reply)".to_string()
    } else {
        truncate_chars(tail.last_assistant.trim(), 200)
    };
    format!(
        "Previous session ({age}, key {key}): {user} → {assistant}",
        key = tail.session_key
    )
}

/// Injection condition: only a FRESH session with a prior session gets the
/// continuity note. Resumed sessions already carry their own history.
pub(crate) fn continuity_note(
    start: SessionStart,
    tail: Option<&SessionTail>,
    now: DateTime<Utc>,
) -> Option<String> {
    match start {
        SessionStart::Resumed => None,
        SessionStart::Fresh => tail.map(|t| format_continuity_line(t, now)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::db::SessionTail;
    use chrono::{Duration, Utc};

    fn make_tail(user: &str, assistant: &str, age: Duration) -> SessionTail {
        SessionTail {
            session_id: "20260701_000000_abc".into(),
            session_key: "cli:oneshot-123".into(),
            updated_at: Utc::now() - age,
            last_user: user.into(),
            last_assistant: assistant.into(),
        }
    }

    // --- classify_session_start ---

    #[test]
    fn test_classify_fresh_vs_resumed() {
        assert_eq!(classify_session_start(0), SessionStart::Fresh);
        assert_eq!(classify_session_start(1), SessionStart::Resumed);
        assert_eq!(classify_session_start(42), SessionStart::Resumed);
    }

    // --- humanize_age ---

    #[test]
    fn test_humanize_age_units() {
        assert_eq!(humanize_age(Duration::seconds(30)), "just now");
        assert_eq!(humanize_age(Duration::minutes(5)), "5m ago");
        assert_eq!(humanize_age(Duration::hours(3)), "3h ago");
        assert_eq!(humanize_age(Duration::days(2)), "2d ago");
    }

    // --- format_continuity_line ---

    #[test]
    fn test_format_line_contains_key_age_and_exchange() {
        let tail = make_tail(
            "fix the login bug",
            "Fixed it by adding a guard.",
            Duration::hours(2),
        );
        let line = format_continuity_line(&tail, Utc::now());
        assert!(line.contains("cli:oneshot-123"), "line: {line}");
        assert!(line.contains("2h ago"), "line: {line}");
        assert!(line.contains("fix the login bug"), "line: {line}");
        assert!(line.contains("Fixed it by adding a guard."), "line: {line}");
    }

    #[test]
    fn test_format_line_truncates_long_content_on_char_boundary() {
        let long_user = "é".repeat(500);
        let long_assistant = "日".repeat(500);
        let tail = make_tail(&long_user, &long_assistant, Duration::minutes(10));
        let line = format_continuity_line(&tail, Utc::now());
        // Bounded: well under ~300 tokens (~1200 chars) even with both sides maxed.
        assert!(
            line.chars().count() < 600,
            "line too long: {} chars",
            line.chars().count()
        );
        // No panic on multibyte truncation, and content survives truncated.
        assert!(line.contains('é'));
        assert!(line.contains('日'));
    }

    #[test]
    fn test_format_line_handles_missing_assistant_reply() {
        // Crashed session: user asked, assistant never answered.
        let tail = make_tail("last question", "", Duration::minutes(1));
        let line = format_continuity_line(&tail, Utc::now());
        assert!(line.contains("last question"));
        assert!(line.contains("(no reply)"), "line: {line}");
    }

    // --- continuity_note (injection condition) ---

    #[test]
    fn test_note_injected_only_when_fresh_with_prior_session() {
        let tail = make_tail("q", "a", Duration::minutes(1));
        let now = Utc::now();

        // Fresh + prior session exists → inject.
        assert!(continuity_note(SessionStart::Fresh, Some(&tail), now).is_some());
        // Fresh + no prior session → skip.
        assert!(continuity_note(SessionStart::Fresh, None, now).is_none());
        // Resumed → skip regardless.
        assert!(continuity_note(SessionStart::Resumed, Some(&tail), now).is_none());
        assert!(continuity_note(SessionStart::Resumed, None, now).is_none());
    }
}
