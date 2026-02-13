//! Per-session working memory for active task state.
//!
//! Each session gets a file at `{workspace}/memory/sessions/SESSION_{hash}.md`
//! containing YAML frontmatter + markdown body. Compaction summaries are
//! appended here instead of being written to the observations directory.
//!
//! Completed sessions can be archived for later reflection by the reflector.

use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use sha2::{Sha256, Digest};
use tracing::warn;

use crate::agent::token_budget::TokenBudget;
use crate::utils::helpers::ensure_dir;

/// Status of a working session.
#[derive(Debug, Clone, PartialEq)]
pub enum SessionStatus {
    Active,
    Completed,
    Archived,
}

impl fmt::Display for SessionStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SessionStatus::Active => write!(f, "active"),
            SessionStatus::Completed => write!(f, "completed"),
            SessionStatus::Archived => write!(f, "archived"),
        }
    }
}

impl SessionStatus {
    fn from_str(s: &str) -> Self {
        match s.trim() {
            "completed" => SessionStatus::Completed,
            "archived" => SessionStatus::Archived,
            _ => SessionStatus::Active,
        }
    }
}

/// A single working session loaded from disk.
pub struct WorkingSession {
    pub session_key: String,
    pub created: DateTime<Utc>,
    pub updated: DateTime<Utc>,
    pub status: SessionStatus,
    /// Markdown body after frontmatter.
    pub content: String,
    pub path: PathBuf,
}

/// Persistent store for per-session working memory files.
pub struct WorkingMemoryStore {
    sessions_dir: PathBuf,
    archived_dir: PathBuf,
}

impl WorkingMemoryStore {
    /// Create a new store, ensuring directories exist.
    pub fn new(workspace: &Path) -> Self {
        let sessions_dir = ensure_dir(workspace.join("memory").join("sessions"));
        let archived_dir = sessions_dir.join("archived");
        Self {
            sessions_dir,
            archived_dir,
        }
    }

    /// Deterministic hash for a session key — first 8 hex chars of SHA-256.
    pub fn session_hash(session_key: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(session_key.as_bytes());
        let result = hasher.finalize();
        // Manual hex encode of first 4 bytes (= 8 hex chars).
        result[..4]
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect()
    }

    /// Path for a session file.
    fn session_path(&self, session_key: &str) -> PathBuf {
        let hash = Self::session_hash(session_key);
        self.sessions_dir.join(format!("SESSION_{}.md", hash))
    }

    /// Get or create a working session for the given key.
    pub fn get_or_create(&self, session_key: &str) -> WorkingSession {
        let path = self.session_path(session_key);
        if path.exists() {
            if let Some(session) = self.parse_session(&path) {
                return session;
            }
        }

        let now = Utc::now();
        WorkingSession {
            session_key: session_key.to_string(),
            created: now,
            updated: now,
            status: SessionStatus::Active,
            content: String::new(),
            path,
        }
    }

    /// Save a working session to disk.
    pub fn save(&self, session: &WorkingSession) {
        ensure_dir(&self.sessions_dir);
        let frontmatter = format!(
            "---\nsession_key: \"{}\"\ncreated: \"{}\"\nupdated: \"{}\"\nstatus: {}\n---\n",
            session.session_key,
            session.created.to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
            session.updated.to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
            session.status,
        );
        let full = if session.content.is_empty() {
            frontmatter
        } else {
            format!("{}\n{}", frontmatter, session.content)
        };
        if let Err(e) = fs::write(&session.path, &full) {
            warn!("Failed to save working session {}: {}", session.session_key, e);
        }
    }

    /// Get working memory context for injection into the system prompt.
    ///
    /// Returns the session content truncated to fit the token budget.
    pub fn get_context(&self, session_key: &str, budget: usize) -> String {
        let session = self.get_or_create(session_key);
        if session.content.is_empty() || budget == 0 {
            return String::new();
        }

        let tokens = TokenBudget::estimate_str_tokens(&session.content);
        if tokens <= budget {
            return session.content.clone();
        }

        // Truncate to fit budget.
        let max_chars = budget.saturating_mul(4);
        let marker = "\n\n[truncated to fit working memory budget]";
        let keep = max_chars.saturating_sub(marker.len());
        let mut out: String = session.content.chars().take(keep).collect();
        out.push_str(marker);
        out
    }

    /// Append a compaction summary to the session's working memory.
    pub fn update_from_compaction(&self, session_key: &str, summary: &str) {
        let mut session = self.get_or_create(session_key);
        let timestamp = Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);

        let entry = format!("\n## Compaction Summary ({})\n\n{}\n", timestamp, summary.trim());
        session.content.push_str(&entry);
        session.updated = Utc::now();
        self.save(&session);
    }

    /// Mark a session as completed.
    pub fn complete(&self, session_key: &str) {
        let mut session = self.get_or_create(session_key);
        session.status = SessionStatus::Completed;
        session.updated = Utc::now();
        self.save(&session);
    }

    /// Move a session to the archived directory.
    pub fn archive(&self, session_key: &str) -> anyhow::Result<()> {
        let path = self.session_path(session_key);
        if !path.exists() {
            return Ok(());
        }
        ensure_dir(&self.archived_dir);
        if let Some(filename) = path.file_name() {
            let dest = self.archived_dir.join(filename);
            fs::rename(&path, &dest)?;
        }
        Ok(())
    }

    /// List all active sessions.
    pub fn list_active(&self) -> Vec<WorkingSession> {
        self.list_sessions_by_status(SessionStatus::Active)
    }

    /// List all completed sessions.
    pub fn list_completed(&self) -> Vec<WorkingSession> {
        self.list_sessions_by_status(SessionStatus::Completed)
    }

    /// Total tokens across all sessions with the given status.
    pub fn total_tokens_by_status(&self, status: SessionStatus) -> usize {
        let sessions = self.list_sessions_by_status(status);
        sessions
            .iter()
            .map(|s| TokenBudget::estimate_str_tokens(&s.content))
            .sum()
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    fn list_sessions_by_status(&self, status: SessionStatus) -> Vec<WorkingSession> {
        if !self.sessions_dir.exists() {
            return Vec::new();
        }

        let mut sessions: Vec<WorkingSession> = fs::read_dir(&self.sessions_dir)
            .into_iter()
            .flatten()
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.extension().and_then(|e| e.to_str()) == Some("md")
                    && p.file_name()
                        .and_then(|n| n.to_str())
                        .map(|n| n.starts_with("SESSION_"))
                        .unwrap_or(false)
            })
            .filter_map(|p| self.parse_session(&p))
            .filter(|s| s.status == status)
            .collect();

        // Sort by updated time, newest first.
        sessions.sort_by(|a, b| b.updated.cmp(&a.updated));
        sessions
    }

    fn parse_session(&self, path: &Path) -> Option<WorkingSession> {
        let raw = fs::read_to_string(path).ok()?;

        let mut session_key = String::new();
        let mut created = Utc::now();
        let mut updated = Utc::now();
        let mut status = SessionStatus::Active;
        let mut content = raw.clone();

        if raw.starts_with("---") {
            if let Some(end) = raw[3..].find("---") {
                let frontmatter = &raw[3..3 + end];
                content = raw[3 + end + 3..].trim_start_matches('\n').to_string();

                for line in frontmatter.lines() {
                    let line = line.trim();
                    if let Some(val) = line.strip_prefix("session_key:") {
                        session_key = val.trim().trim_matches('"').to_string();
                    } else if let Some(val) = line.strip_prefix("created:") {
                        if let Ok(dt) = DateTime::parse_from_rfc3339(val.trim().trim_matches('"')) {
                            created = dt.with_timezone(&Utc);
                        }
                    } else if let Some(val) = line.strip_prefix("updated:") {
                        if let Ok(dt) = DateTime::parse_from_rfc3339(val.trim().trim_matches('"')) {
                            updated = dt.with_timezone(&Utc);
                        }
                    } else if let Some(val) = line.strip_prefix("status:") {
                        status = SessionStatus::from_str(val.trim());
                    }
                }
            }
        }

        Some(WorkingSession {
            session_key,
            created,
            updated,
            status,
            content,
            path: path.to_path_buf(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_store() -> (TempDir, WorkingMemoryStore) {
        let tmp = TempDir::new().unwrap();
        let store = WorkingMemoryStore::new(tmp.path());
        (tmp, store)
    }

    #[test]
    fn test_session_hash_deterministic() {
        let h1 = WorkingMemoryStore::session_hash("cli:default");
        let h2 = WorkingMemoryStore::session_hash("cli:default");
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 8);
    }

    #[test]
    fn test_session_hash_different_keys() {
        let h1 = WorkingMemoryStore::session_hash("cli:default");
        let h2 = WorkingMemoryStore::session_hash("telegram:12345");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_get_or_create_new_session() {
        let (_tmp, store) = make_store();
        let session = store.get_or_create("cli:default");
        assert_eq!(session.session_key, "cli:default");
        assert_eq!(session.status, SessionStatus::Active);
        assert!(session.content.is_empty());
    }

    #[test]
    fn test_save_and_load_roundtrip() {
        let (_tmp, store) = make_store();
        let mut session = store.get_or_create("cli:test");
        session.content = "Working on memory refactor.".to_string();
        store.save(&session);

        let loaded = store.get_or_create("cli:test");
        assert_eq!(loaded.session_key, "cli:test");
        assert_eq!(loaded.status, SessionStatus::Active);
        assert!(loaded.content.contains("Working on memory refactor."));
    }

    #[test]
    fn test_update_from_compaction() {
        let (_tmp, store) = make_store();
        store.update_from_compaction("cli:default", "User asked about Rust memory management.");

        let session = store.get_or_create("cli:default");
        assert!(session.content.contains("Compaction Summary"));
        assert!(session.content.contains("Rust memory management"));
    }

    #[test]
    fn test_multiple_compactions_append() {
        let (_tmp, store) = make_store();
        store.update_from_compaction("cli:default", "First summary.");
        store.update_from_compaction("cli:default", "Second summary.");

        let session = store.get_or_create("cli:default");
        assert!(session.content.contains("First summary."));
        assert!(session.content.contains("Second summary."));
    }

    #[test]
    fn test_get_context_empty() {
        let (_tmp, store) = make_store();
        let ctx = store.get_context("cli:default", 1000);
        assert!(ctx.is_empty());
    }

    #[test]
    fn test_get_context_within_budget() {
        let (_tmp, store) = make_store();
        store.update_from_compaction("cli:default", "Short summary.");
        let ctx = store.get_context("cli:default", 5000);
        assert!(ctx.contains("Short summary."));
        assert!(!ctx.contains("[truncated"));
    }

    #[test]
    fn test_get_context_truncated() {
        let (_tmp, store) = make_store();
        store.update_from_compaction("cli:default", &"x".repeat(10000));
        let ctx = store.get_context("cli:default", 50);
        assert!(ctx.contains("[truncated to fit working memory budget]"));
    }

    #[test]
    fn test_complete_session() {
        let (_tmp, store) = make_store();
        store.update_from_compaction("cli:default", "Some work.");
        store.complete("cli:default");

        let session = store.get_or_create("cli:default");
        assert_eq!(session.status, SessionStatus::Completed);
    }

    #[test]
    fn test_archive_session() {
        let (_tmp, store) = make_store();
        store.update_from_compaction("cli:default", "Some work.");
        let path = store.session_path("cli:default");
        assert!(path.exists());

        store.archive("cli:default").unwrap();
        assert!(!path.exists());
        assert!(store.archived_dir.exists());
    }

    #[test]
    fn test_list_active() {
        let (_tmp, store) = make_store();
        store.update_from_compaction("session:a", "Work A.");
        store.update_from_compaction("session:b", "Work B.");
        store.update_from_compaction("session:c", "Work C.");
        store.complete("session:b");

        let active = store.list_active();
        assert_eq!(active.len(), 2);
        let keys: Vec<&str> = active.iter().map(|s| s.session_key.as_str()).collect();
        assert!(keys.contains(&"session:a"));
        assert!(keys.contains(&"session:c"));
    }

    #[test]
    fn test_list_completed() {
        let (_tmp, store) = make_store();
        store.update_from_compaction("session:a", "Work A.");
        store.update_from_compaction("session:b", "Work B.");
        store.complete("session:a");

        let completed = store.list_completed();
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].session_key, "session:a");
    }
}
