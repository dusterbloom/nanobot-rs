//! Per-session working memory for active task state.
//!
//! Each session gets a file at `{workspace}/memory/sessions/SESSION_{hash}.md`
//! containing YAML frontmatter + markdown body. Compaction overwrites session
//! content with a structured template (CONTEXT.md protocol) and also writes
//! to `{workspace}/CONTEXT-{channel}.md` for direct system prompt injection.
//!
//! Completed sessions can be archived for later reflection by the reflector.

use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use sha2::{Digest, Sha256};
use tracing::warn;

use crate::agent::token_budget::TokenBudget;
use crate::utils::helpers::{ensure_dir, move_file};

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
    /// Turn number at which this session was last updated via compaction.
    pub last_updated_turn: u64,
}

/// Persistent store for per-session working memory files.
pub struct WorkingMemoryStore {
    workspace: PathBuf,
    sessions_dir: PathBuf,
    archived_dir: PathBuf,
}

impl WorkingMemoryStore {
    /// Create a new store, ensuring directories exist.
    pub fn new(workspace: &Path) -> Self {
        let sessions_dir = ensure_dir(workspace.join("memory").join("sessions"));
        let archived_dir = sessions_dir.join("archived");
        Self {
            workspace: workspace.to_path_buf(),
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
        result[..4].iter().map(|b| format!("{:02x}", b)).collect()
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
            last_updated_turn: 0,
        }
    }

    /// Save a working session to disk.
    pub fn save(&self, session: &WorkingSession) {
        ensure_dir(&self.sessions_dir);
        let frontmatter = format!(
            "---\nsession_key: \"{}\"\ncreated: \"{}\"\nupdated: \"{}\"\nstatus: {}\nlast_updated_turn: {}\n---\n",
            session.session_key,
            session
                .created
                .to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
            session
                .updated
                .to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
            session.status,
            session.last_updated_turn,
        );
        let full = if session.content.is_empty() {
            frontmatter
        } else {
            format!("{}\n{}", frontmatter, session.content)
        };
        if let Err(e) = fs::write(&session.path, &full) {
            warn!(
                "Failed to save working session {}: {}",
                session.session_key, e
            );
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

        // Silently truncate at line boundaries — no visible markers.
        let max_chars = budget.saturating_mul(4);
        let lines: Vec<&str> = session.content.lines().collect();
        let mut kept: Vec<&str> = Vec::new();
        let mut char_count = 0;

        for line in &lines {
            let line_cost = line.len() + 1;
            if char_count + line_cost > max_chars && !kept.is_empty() {
                break;
            }
            kept.push(line);
            char_count += line_cost;
        }

        kept.join("\n")
    }

    /// Replace working memory with the latest compaction summary.
    ///
    /// Overwrites instead of appending — the structured template output
    /// from each compaction is a complete snapshot, not an increment.
    /// Also writes to `{workspace}/CONTEXT-{channel}.md` for system prompt injection.
    pub fn update_from_compaction(&self, session_key: &str, summary: &str, turn: u64) {
        let mut session = self.get_or_create(session_key);

        // Overwrite, not append — each compaction is a complete snapshot.
        session.content = summary.trim().to_string();
        session.updated = Utc::now();
        session.last_updated_turn = turn;
        self.save(&session);

        // Write per-channel context file for system prompt injection.
        // Each channel gets its own file (CONTEXT-cli.md, CONTEXT-telegram.md)
        // so concurrent sessions don't clobber each other.
        let channel = session_key.split(':').next().unwrap_or("default");
        let context_filename = format!("CONTEXT-{}.md", channel);
        let context_path = self.workspace.join(&context_filename);
        if let Err(e) = std::fs::write(&context_path, summary.trim()) {
            warn!("Failed to write {}: {}", context_filename, e);
        }
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
            move_file(&path, &dest)?;
        }
        Ok(())
    }

    /// Clear working memory for a session (reset content, keep session file).
    ///
    /// Also removes the corresponding `CONTEXT-{channel}.md` file so stale
    /// compaction summaries are not re-injected into the next system prompt.
    pub fn clear(&self, session_key: &str) {
        let mut session = self.get_or_create(session_key);
        session.content = String::new();
        session.updated = Utc::now();
        self.save(&session);

        // Remove the CONTEXT file written by update_from_compaction().
        let channel = session_key.split(':').next().unwrap_or("default");
        let per_channel = self.workspace.join(format!("CONTEXT-{}.md", channel));
        if per_channel.exists() {
            let _ = fs::remove_file(&per_channel);
        }
        // Also remove legacy fallback if it exists.
        let legacy = self.workspace.join("CONTEXT.md");
        if legacy.exists() {
            let _ = fs::remove_file(&legacy);
        }
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
        let mut last_updated_turn: u64 = 0;

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
                    } else if let Some(val) = line.strip_prefix("last_updated_turn:") {
                        last_updated_turn = val.trim().parse().unwrap_or(0);
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
            last_updated_turn,
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
        store.update_from_compaction("cli:default", "User asked about Rust memory management.", 0);

        let session = store.get_or_create("cli:default");
        assert!(session.content.contains("Rust memory management"));
    }

    #[test]
    fn test_multiple_compactions_overwrite() {
        let (_tmp, store) = make_store();
        store.update_from_compaction("cli:default", "First summary.", 0);
        store.update_from_compaction("cli:default", "Second summary.", 0);

        let session = store.get_or_create("cli:default");
        // Overwrite semantics: only latest snapshot survives.
        assert!(!session.content.contains("First summary."));
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
        store.update_from_compaction("cli:default", "Short summary.", 0);
        let ctx = store.get_context("cli:default", 5000);
        assert!(ctx.contains("Short summary."));
        assert!(!ctx.contains("[truncated"));
    }

    #[test]
    fn test_get_context_truncated() {
        let (_tmp, store) = make_store();
        // Use multi-line content so line-boundary truncation kicks in.
        let content = (0..500)
            .map(|i| format!("line {}", i))
            .collect::<Vec<_>>()
            .join("\n");
        store.update_from_compaction("cli:default", &content, 0);
        let ctx = store.get_context("cli:default", 50);
        // Should be silently truncated — no marker visible to the model.
        assert!(!ctx.contains("[truncated"));
        assert!(ctx.contains("line 0")); // keeps from head
        assert!(ctx.len() < content.len()); // actually truncated
    }

    #[test]
    fn test_complete_session() {
        let (_tmp, store) = make_store();
        store.update_from_compaction("cli:default", "Some work.", 0);
        store.complete("cli:default");

        let session = store.get_or_create("cli:default");
        assert_eq!(session.status, SessionStatus::Completed);
    }

    #[test]
    fn test_archive_session() {
        let (_tmp, store) = make_store();
        store.update_from_compaction("cli:default", "Some work.", 0);
        let path = store.session_path("cli:default");
        assert!(path.exists());

        store.archive("cli:default").unwrap();
        assert!(!path.exists());
        assert!(store.archived_dir.exists());
    }

    #[test]
    fn test_list_active() {
        let (_tmp, store) = make_store();
        store.update_from_compaction("session:a", "Work A.", 0);
        store.update_from_compaction("session:b", "Work B.", 0);
        store.update_from_compaction("session:c", "Work C.", 0);
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
        store.update_from_compaction("session:a", "Work A.", 0);
        store.update_from_compaction("session:b", "Work B.", 0);
        store.complete("session:a");

        let completed = store.list_completed();
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].session_key, "session:a");
    }

    // ----- Integration tests: cross-module memory pipeline -----

    #[test]
    fn test_integration_working_memory_injected_into_context() {
        // Verify that ContextBuilder picks up working memory via get_context().
        let tmp = TempDir::new().unwrap();
        let wm = WorkingMemoryStore::new(tmp.path());
        wm.update_from_compaction("cli:test", "User prefers dark mode and Vim keybindings.", 0);

        // Build system prompt (won't have working memory — that's injected by agent_loop).
        // But verify get_context returns the right content for injection.
        let ctx = wm.get_context("cli:test", 5000);
        assert!(ctx.contains("dark mode"));
        assert!(ctx.contains("Vim keybindings"));

        // Simulate what agent_loop does: enrich system prompt.
        let base_prompt = "You are nanobot.";
        let enriched = format!(
            "{}\n\n---\n\n# Working Memory (Current Session)\n\n{}",
            base_prompt, ctx
        );
        assert!(enriched.contains("You are nanobot."));
        assert!(enriched.contains("Working Memory (Current Session)"));
        assert!(enriched.contains("dark mode"));
    }

    #[test]
    fn test_integration_compaction_to_working_memory_to_context() {
        // Full pipeline: compaction overwrites summary → working memory stores it →
        // get_context returns it for system prompt injection.
        // Also writes per-channel CONTEXT-{channel}.md to workspace root.
        let tmp = TempDir::new().unwrap();
        let wm = WorkingMemoryStore::new(tmp.path());

        // Simulate 3 compaction cycles — overwrite semantics.
        wm.update_from_compaction("cli:session1", "Discussed Rust ownership model.", 0);
        wm.update_from_compaction("cli:session1", "Implemented borrow checker example.", 0);
        wm.update_from_compaction("cli:session1", "User wants to learn async next.", 0);

        // Only the latest snapshot survives (overwrite, not append).
        let ctx = wm.get_context("cli:session1", 10000);
        assert!(
            !ctx.contains("ownership model"),
            "older snapshots should be overwritten"
        );
        assert!(ctx.contains("async next"), "latest snapshot should survive");

        // Verify the session file exists on disk.
        let hash = WorkingMemoryStore::session_hash("cli:session1");
        let session_file = tmp
            .path()
            .join("memory")
            .join("sessions")
            .join(format!("SESSION_{}.md", hash));
        assert!(session_file.exists(), "Session file should exist on disk");

        // Verify per-channel CONTEXT-cli.md was written (channel extracted from session key).
        let context_file = tmp.path().join("CONTEXT-cli.md");
        assert!(context_file.exists(), "CONTEXT-cli.md should exist");
        let context_content = std::fs::read_to_string(&context_file).unwrap();
        assert!(
            context_content.contains("async next"),
            "CONTEXT-cli.md should have latest snapshot"
        );
    }

    #[test]
    fn test_integration_complete_and_reflector_readiness() {
        // Verify that completing a session makes it visible to the reflector's
        // list_completed(), and that archiving removes it.
        let tmp = TempDir::new().unwrap();
        let wm = WorkingMemoryStore::new(tmp.path());

        wm.update_from_compaction("s:1", "Facts from session 1.", 0);
        wm.update_from_compaction("s:2", "Facts from session 2.", 0);
        wm.update_from_compaction("s:3", "Facts from session 3.", 0);

        // Only complete sessions are visible to the reflector.
        assert_eq!(wm.list_completed().len(), 0);
        assert_eq!(wm.list_active().len(), 3);

        wm.complete("s:1");
        wm.complete("s:2");

        let completed = wm.list_completed();
        assert_eq!(completed.len(), 2);
        assert_eq!(wm.list_active().len(), 1);

        // After archiving, completed list is empty (reflector has processed them).
        wm.archive("s:1").unwrap();
        wm.archive("s:2").unwrap();
        assert_eq!(wm.list_completed().len(), 0);
        assert_eq!(wm.list_active().len(), 1);

        // Archived files exist.
        let archived_dir = tmp.path().join("memory").join("sessions").join("archived");
        assert!(archived_dir.exists());
        let count = std::fs::read_dir(&archived_dir).unwrap().count();
        assert_eq!(count, 2);
    }

    #[test]
    fn test_integration_context_no_longer_has_observations() {
        // Verify the system prompt no longer loads observations even when they exist.
        use crate::agent::context::ContextBuilder;

        let tmp = TempDir::new().unwrap();
        let obs_dir = tmp.path().join("memory").join("observations");
        std::fs::create_dir_all(&obs_dir).unwrap();
        std::fs::write(
            obs_dir.join("20260101T000000Z_test.md"),
            "---\ntimestamp: 2026-01-01T00:00:00Z\nsession: test\n---\n\nOld observation data.",
        )
        .unwrap();

        // Also write long-term memory to verify it IS loaded.
        let mem_dir = tmp.path().join("memory");
        std::fs::write(mem_dir.join("MEMORY.md"), "- User likes Rust").unwrap();

        let cb = ContextBuilder::new(tmp.path());
        let prompt = cb.build_system_prompt(None, None);

        // Observations should NOT be in the prompt.
        assert!(
            !prompt.contains("Old observation data"),
            "Observations must not be in system prompt"
        );
        assert!(!prompt.contains("Observations from Past Conversations"));

        // Long-term memory SHOULD be in the prompt.
        assert!(
            prompt.contains("User likes Rust"),
            "Long-term memory should be in system prompt"
        );

        // Identity should mention layered memory system.
        assert!(
            prompt.contains("Working Memory"),
            "Identity should mention working memory"
        );
        assert!(
            prompt.contains("recall"),
            "Identity should mention recall tool"
        );
    }

    #[test]
    fn test_clear_session() {
        let (_tmp, store) = make_store();
        store.update_from_compaction("cli:default", "Important context.", 5);
        assert!(!store.get_context("cli:default", 5000).is_empty());
        store.clear("cli:default");
        assert!(store.get_context("cli:default", 5000).is_empty());
        let session = store.get_or_create("cli:default");
        assert_eq!(session.status, SessionStatus::Active);
    }

    #[test]
    fn test_clear_removes_context_file() {
        let (tmp, store) = make_store();
        store.update_from_compaction("cli:default", "Stale task context.", 3);

        let context_file = tmp.path().join("CONTEXT-cli.md");
        assert!(context_file.exists(), "CONTEXT-cli.md should exist after compaction");

        store.clear("cli:default");

        assert!(!context_file.exists(), "CONTEXT-cli.md should be removed after clear()");
    }

    #[test]
    fn test_clear_removes_legacy_context_file() {
        let (tmp, store) = make_store();
        let legacy = tmp.path().join("CONTEXT.md");
        std::fs::write(&legacy, "Legacy stale context.").unwrap();

        store.clear("cli:default");

        assert!(!legacy.exists(), "CONTEXT.md should be removed after clear()");
    }

    #[test]
    fn test_last_updated_turn_roundtrip() {
        let (_tmp, store) = make_store();
        store.update_from_compaction("cli:default", "Summary.", 42);
        let session = store.get_or_create("cli:default");
        assert_eq!(session.last_updated_turn, 42);
    }
}
