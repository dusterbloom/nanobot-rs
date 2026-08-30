//! Per-session working memory stored in the canonical sessions database.
//!
//! Each row is keyed by the concrete SQLite `session_id`. Channel/chat keys
//! are reusable, so using them here would allow an expired session's compacted
//! state to leak into the next session for the same chat.

use std::fmt;
use std::sync::Arc;

use chrono::{DateTime, Utc};

use crate::agent::token_budget::TokenBudget;
use crate::session::db::{SessionDb, WorkingMemoryRecord};

/// Lifecycle of a derived working-memory snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionStatus {
    Active,
    Completed,
    Reflected,
}

impl SessionStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Completed => "completed",
            Self::Reflected => "reflected",
        }
    }

    fn from_str(value: &str) -> Self {
        match value {
            "completed" => Self::Completed,
            "reflected" => Self::Reflected,
            _ => Self::Active,
        }
    }
}

impl fmt::Display for SessionStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A working-memory snapshot joined with its owning session metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkingSession {
    pub session_id: String,
    pub session_key: String,
    pub created: DateTime<Utc>,
    pub updated: DateTime<Utc>,
    pub status: SessionStatus,
    pub content: String,
    pub last_updated_turn: u64,
}

impl From<WorkingMemoryRecord> for WorkingSession {
    fn from(record: WorkingMemoryRecord) -> Self {
        Self {
            session_id: record.session_id,
            session_key: record.session_key,
            created: record.created_at,
            updated: record.updated_at,
            status: SessionStatus::from_str(&record.status),
            content: record.content,
            last_updated_turn: record.last_updated_turn,
        }
    }
}

/// Async facade over SQLite's per-session working-memory rows.
#[derive(Clone)]
pub struct WorkingMemoryStore {
    sessions: Arc<SessionDb>,
}

impl WorkingMemoryStore {
    pub fn new(sessions: Arc<SessionDb>) -> Self {
        Self { sessions }
    }

    /// Load or initialise the working row for a concrete session.
    /// Returns `None` when `session_id` does not exist.
    pub async fn get_or_create(
        &self,
        session_id: &str,
    ) -> rusqlite::Result<Option<WorkingSession>> {
        self.sessions
            .get_or_create_working_memory(session_id)
            .await
            .map(|record| record.map(WorkingSession::from))
    }

    pub async fn save(&self, session: &WorkingSession) -> rusqlite::Result<bool> {
        self.sessions
            .save_working_memory(
                &session.session_id,
                &session.content,
                session.status.as_str(),
                session.last_updated_turn,
            )
            .await
    }

    /// Return the current session snapshot truncated at line boundaries.
    pub async fn get_context(&self, session_id: &str, budget: usize) -> rusqlite::Result<String> {
        let Some(session) = self.get_or_create(session_id).await? else {
            return Ok(String::new());
        };
        if session.content.is_empty() || budget == 0 {
            return Ok(String::new());
        }

        if TokenBudget::estimate_str_tokens(&session.content) <= budget {
            return Ok(session.content);
        }

        let max_chars = budget.saturating_mul(4);
        let mut kept = Vec::new();
        let mut char_count = 0;
        for line in session.content.lines() {
            let line_cost = line.len() + 1;
            if char_count + line_cost > max_chars && !kept.is_empty() {
                break;
            }
            kept.push(line);
            char_count += line_cost;
        }
        Ok(kept.join("\n"))
    }

    pub async fn complete(&self, session_id: &str) -> rusqlite::Result<bool> {
        if self.get_or_create(session_id).await?.is_none() {
            return Ok(false);
        }
        self.sessions
            .set_working_memory_status(session_id, "completed")
            .await
    }

    /// Mark a completed snapshot consumed by the reflector. SQLite keeps the
    /// row for audit and lifecycle accounting.
    pub async fn mark_reflected(&self, session_id: &str) -> rusqlite::Result<bool> {
        self.sessions
            .set_working_memory_status(session_id, "reflected")
            .await
    }

    pub async fn mark_reflected_all(&self, session_ids: &[String]) -> rusqlite::Result<usize> {
        self.sessions
            .set_working_memory_status_batch(session_ids, "reflected")
            .await
    }

    pub async fn clear(&self, session_id: &str) -> rusqlite::Result<bool> {
        if self.get_or_create(session_id).await?.is_none() {
            return Ok(false);
        }
        self.sessions.clear_working_memory(session_id).await
    }

    pub async fn list_active(&self) -> rusqlite::Result<Vec<WorkingSession>> {
        self.list_sessions_by_status(SessionStatus::Active).await
    }

    pub async fn list_completed(&self) -> rusqlite::Result<Vec<WorkingSession>> {
        self.list_sessions_by_status(SessionStatus::Completed).await
    }

    pub async fn list_reflected(&self) -> rusqlite::Result<Vec<WorkingSession>> {
        self.list_sessions_by_status(SessionStatus::Reflected).await
    }

    pub async fn total_tokens_by_status(&self, status: SessionStatus) -> rusqlite::Result<usize> {
        Ok(self
            .list_sessions_by_status(status)
            .await?
            .iter()
            .map(|session| TokenBudget::estimate_str_tokens(&session.content))
            .sum())
    }

    /// Number of sessions in `status` — the "how much am I about to distill"
    /// figure shown before a reflection pass runs.
    pub async fn count_by_status(&self, status: SessionStatus) -> rusqlite::Result<usize> {
        Ok(self.list_sessions_by_status(status).await?.len())
    }

    async fn list_sessions_by_status(
        &self,
        status: SessionStatus,
    ) -> rusqlite::Result<Vec<WorkingSession>> {
        self.sessions
            .list_working_memory(Some(status.as_str()))
            .await
            .map(|records| records.into_iter().map(WorkingSession::from).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    async fn make_store() -> (TempDir, Arc<SessionDb>, WorkingMemoryStore, String) {
        let temp = TempDir::new().unwrap();
        let db = Arc::new(SessionDb::new(&temp.path().join("sessions.db")));
        let session = db.create_session("cli:test").await;
        (temp, db.clone(), WorkingMemoryStore::new(db), session.id)
    }

    #[tokio::test]
    async fn context_respects_budget_and_line_boundaries() {
        let (_temp, db, store, session_id) = make_store().await;
        db.save_working_memory(
            &session_id,
            "first short line\nsecond longer line\nthird line",
            "active",
            1,
        )
        .await
        .unwrap();
        let context = store.get_context(&session_id, 5).await.unwrap();
        assert!(!context.is_empty());
        assert!(context.len() < "first short line\nsecond longer line\nthird line".len());
        assert!(!context.ends_with("second long"));
    }

    #[tokio::test]
    async fn lifecycle_is_persisted_in_sqlite() {
        let (temp, db, store, session_id) = make_store().await;
        db.save_working_memory(&session_id, "facts", "active", 3)
            .await
            .unwrap();
        assert!(store.complete(&session_id).await.unwrap());
        assert_eq!(store.list_completed().await.unwrap().len(), 1);
        assert!(store.mark_reflected(&session_id).await.unwrap());
        assert!(store.list_completed().await.unwrap().is_empty());
        assert_eq!(store.list_reflected().await.unwrap().len(), 1);
        assert!(store.list_active().await.unwrap().is_empty());
        assert!(
            !temp.path().join("memory").join("sessions").exists(),
            "SQLite working memory must not recreate SESSION_*.md storage"
        );
    }

    #[tokio::test]
    async fn reflected_batch_rolls_back_if_any_session_is_missing() {
        let (_temp, db, store, session_id) = make_store().await;
        db.save_working_memory(&session_id, "facts", "active", 1)
            .await
            .unwrap();
        store.complete(&session_id).await.unwrap();

        let result = store
            .mark_reflected_all(&[session_id.clone(), "missing-session".to_string()])
            .await;
        assert!(result.is_err());
        let completed = store.list_completed().await.unwrap();
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].session_id, session_id);
        assert!(store.list_reflected().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn clear_resets_snapshot_but_keeps_session() {
        let (_temp, db, store, session_id) = make_store().await;
        db.save_working_memory(&session_id, "temporary state", "active", 4)
            .await
            .unwrap();
        store.complete(&session_id).await.unwrap();
        assert!(store.clear(&session_id).await.unwrap());

        let loaded = store.get_or_create(&session_id).await.unwrap().unwrap();
        assert!(loaded.content.is_empty());
        assert_eq!(loaded.status, SessionStatus::Active);
        assert_eq!(loaded.last_updated_turn, 0);
    }

    #[tokio::test]
    async fn unknown_session_never_creates_orphan_memory() {
        let (_temp, db, store, _session_id) = make_store().await;
        assert!(store
            .get_or_create("missing-session")
            .await
            .unwrap()
            .is_none());
        assert!(!db
            .save_working_memory("missing-session", "leak", "active", 1)
            .await
            .unwrap());
    }
}
