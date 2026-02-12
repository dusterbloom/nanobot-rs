//! SQLite-backed conversation store.
//!
//! All verbatim turns are stored in a queryable SQLite database alongside
//! session metadata. This runs in parallel with the existing JSONL session
//! manager — both receive the same data.
//!
//! DB location: `~/.nanobot/store.db`

use std::path::Path;
use std::sync::Mutex;

use chrono::Utc;
use rusqlite::{params, Connection};
use tracing::warn;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Row types
// ---------------------------------------------------------------------------

/// A row from the `sessions` table.
#[derive(Debug, Clone)]
pub struct SessionRow {
    pub id: String,
    pub channel: String,
    pub chat_id: String,
    pub started_at: String,
    pub ended_at: Option<String>,
    pub metadata: Option<String>,
}

/// A row from the `turns` table.
#[derive(Debug, Clone)]
pub struct TurnRow {
    pub id: String,
    pub session_id: String,
    pub seq: i64,
    pub role: String,
    pub content: Option<String>,
    pub tool_calls: Option<String>,
    pub tool_results: Option<String>,
    pub model: Option<String>,
    pub tokens_in: Option<i64>,
    pub tokens_out: Option<i64>,
    pub channel: Option<String>,
    pub language: Option<String>,
    pub voice_mode: bool,
    pub created_at: String,
    pub metadata: Option<String>,
}

// ---------------------------------------------------------------------------
// ConversationStore
// ---------------------------------------------------------------------------

/// Thread-safe SQLite conversation store.
///
/// Uses a sync `Mutex<Connection>` because rusqlite's `Connection` is `!Send`.
/// All public methods are synchronous — callers can wrap in `spawn_blocking`
/// or `block_in_place` when needed from async contexts.
pub struct ConversationStore {
    conn: Mutex<Connection>,
}

impl ConversationStore {
    /// Open (or create) the database at `db_path` and run migrations.
    pub fn new(db_path: &Path) -> anyhow::Result<Self> {
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let conn = Connection::open(db_path)?;

        // Performance pragmas.
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA synchronous = NORMAL;
             PRAGMA foreign_keys = ON;",
        )?;

        // Run migrations.
        Self::migrate(&conn)?;

        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    /// Run schema migrations (idempotent).
    fn migrate(conn: &Connection) -> anyhow::Result<()> {
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS sessions (
                 id TEXT PRIMARY KEY,
                 channel TEXT NOT NULL,
                 chat_id TEXT NOT NULL,
                 started_at TEXT NOT NULL,
                 ended_at TEXT,
                 metadata TEXT
             );

             CREATE TABLE IF NOT EXISTS turns (
                 id TEXT PRIMARY KEY,
                 session_id TEXT NOT NULL REFERENCES sessions(id),
                 seq INTEGER NOT NULL,
                 role TEXT NOT NULL,
                 content TEXT,
                 tool_calls TEXT,
                 tool_results TEXT,
                 model TEXT,
                 tokens_in INTEGER,
                 tokens_out INTEGER,
                 channel TEXT,
                 language TEXT,
                 voice_mode INTEGER DEFAULT 0,
                 created_at TEXT NOT NULL,
                 metadata TEXT
             );

             CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);
             CREATE INDEX IF NOT EXISTS idx_turns_created ON turns(created_at);
             CREATE INDEX IF NOT EXISTS idx_sessions_channel ON sessions(channel, chat_id);",
        )?;
        Ok(())
    }

    // ------------------------------------------------------------------
    // Session operations
    // ------------------------------------------------------------------

    /// Find an active (un-ended) session for the given channel+chat_id,
    /// or create a new one. Returns the session UUID.
    pub fn ensure_session(&self, channel: &str, chat_id: &str) -> String {
        let conn = self.conn.lock().unwrap();

        // Look for an active session.
        let existing: Option<String> = conn
            .query_row(
                "SELECT id FROM sessions
                 WHERE channel = ?1 AND chat_id = ?2 AND ended_at IS NULL
                 ORDER BY started_at DESC LIMIT 1",
                params![channel, chat_id],
                |row| row.get(0),
            )
            .ok();

        if let Some(id) = existing {
            return id;
        }

        // Create a new session.
        let id = Uuid::new_v4().to_string();
        let now = Utc::now().to_rfc3339();
        if let Err(e) = conn.execute(
            "INSERT INTO sessions (id, channel, chat_id, started_at) VALUES (?1, ?2, ?3, ?4)",
            params![id, channel, chat_id, now],
        ) {
            warn!("Failed to create session: {}", e);
        }
        id
    }

    /// Mark a session as ended.
    pub fn end_session(&self, session_id: &str) {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();
        if let Err(e) = conn.execute(
            "UPDATE sessions SET ended_at = ?1 WHERE id = ?2",
            params![now, session_id],
        ) {
            warn!("Failed to end session {}: {}", session_id, e);
        }
    }

    /// End all active (un-ended) sessions. Called when the agent loop stops.
    pub fn end_all_active_sessions(&self) {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();
        if let Err(e) = conn.execute(
            "UPDATE sessions SET ended_at = ?1 WHERE ended_at IS NULL",
            params![now],
        ) {
            warn!("Failed to end active sessions: {}", e);
        }
    }

    /// Get info about a session.
    pub fn get_session_info(&self, session_id: &str) -> Option<SessionRow> {
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT id, channel, chat_id, started_at, ended_at, metadata
             FROM sessions WHERE id = ?1",
            params![session_id],
            |row| {
                Ok(SessionRow {
                    id: row.get(0)?,
                    channel: row.get(1)?,
                    chat_id: row.get(2)?,
                    started_at: row.get(3)?,
                    ended_at: row.get(4)?,
                    metadata: row.get(5)?,
                })
            },
        )
        .ok()
    }

    /// List recent sessions for a channel.
    pub fn list_sessions(&self, channel: &str, limit: usize) -> Vec<SessionRow> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = match conn.prepare(
            "SELECT id, channel, chat_id, started_at, ended_at, metadata
             FROM sessions WHERE channel = ?1
             ORDER BY started_at DESC LIMIT ?2",
        ) {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };

        stmt.query_map(params![channel, limit as i64], |row| {
            Ok(SessionRow {
                id: row.get(0)?,
                channel: row.get(1)?,
                chat_id: row.get(2)?,
                started_at: row.get(3)?,
                ended_at: row.get(4)?,
                metadata: row.get(5)?,
            })
        })
        .map(|rows| rows.filter_map(|r| r.ok()).collect())
        .unwrap_or_default()
    }

    // ------------------------------------------------------------------
    // Turn operations
    // ------------------------------------------------------------------

    /// Add a turn to a session. Returns the turn UUID.
    #[allow(clippy::too_many_arguments)]
    pub fn add_turn(
        &self,
        session_id: &str,
        role: &str,
        content: Option<&str>,
        tool_calls: Option<&str>,
        tool_results: Option<&str>,
        model: Option<&str>,
        tokens_in: Option<i64>,
        tokens_out: Option<i64>,
        channel: Option<&str>,
        language: Option<&str>,
        voice_mode: bool,
        metadata: Option<&str>,
    ) -> String {
        let conn = self.conn.lock().unwrap();
        let turn_id = Uuid::new_v4().to_string();
        let now = Utc::now().to_rfc3339();

        // Get next sequence number for this session.
        let seq: i64 = conn
            .query_row(
                "SELECT COALESCE(MAX(seq), 0) + 1 FROM turns WHERE session_id = ?1",
                params![session_id],
                |row| row.get(0),
            )
            .unwrap_or(1);

        if let Err(e) = conn.execute(
            "INSERT INTO turns (id, session_id, seq, role, content, tool_calls, tool_results,
                                model, tokens_in, tokens_out, channel, language, voice_mode,
                                created_at, metadata)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)",
            params![
                turn_id,
                session_id,
                seq,
                role,
                content,
                tool_calls,
                tool_results,
                model,
                tokens_in,
                tokens_out,
                channel,
                language,
                voice_mode as i32,
                now,
                metadata,
            ],
        ) {
            warn!("Failed to add turn: {}", e);
        }

        turn_id
    }

    /// Get turns for a specific session (most recent last).
    pub fn get_session_turns(&self, session_id: &str, limit: usize) -> Vec<TurnRow> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = match conn.prepare(
            "SELECT id, session_id, seq, role, content, tool_calls, tool_results,
                    model, tokens_in, tokens_out, channel, language, voice_mode,
                    created_at, metadata
             FROM turns WHERE session_id = ?1
             ORDER BY seq ASC LIMIT ?2",
        ) {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };

        Self::collect_turns(&mut stmt, params![session_id, limit as i64])
    }

    /// Get recent turns across sessions for the same channel+chat_id.
    pub fn get_recent_turns(&self, channel: &str, chat_id: &str, limit: usize) -> Vec<TurnRow> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = match conn.prepare(
            "SELECT t.id, t.session_id, t.seq, t.role, t.content, t.tool_calls, t.tool_results,
                    t.model, t.tokens_in, t.tokens_out, t.channel, t.language, t.voice_mode,
                    t.created_at, t.metadata
             FROM turns t
             JOIN sessions s ON t.session_id = s.id
             WHERE s.channel = ?1 AND s.chat_id = ?2
             ORDER BY t.created_at DESC LIMIT ?3",
        ) {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };

        Self::collect_turns(&mut stmt, params![channel, chat_id, limit as i64])
    }

    /// Basic LIKE search across turn content (replaced by BM25 in Phase 1).
    pub fn search_turns(&self, query_text: &str, limit: usize) -> Vec<TurnRow> {
        let conn = self.conn.lock().unwrap();
        let pattern = format!("%{}%", query_text);
        let mut stmt = match conn.prepare(
            "SELECT id, session_id, seq, role, content, tool_calls, tool_results,
                    model, tokens_in, tokens_out, channel, language, voice_mode,
                    created_at, metadata
             FROM turns WHERE content LIKE ?1
             ORDER BY created_at DESC LIMIT ?2",
        ) {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };

        Self::collect_turns(&mut stmt, params![pattern, limit as i64])
    }

    /// Helper to collect TurnRow results from a prepared statement.
    fn collect_turns(
        stmt: &mut rusqlite::Statement<'_>,
        params: impl rusqlite::Params,
    ) -> Vec<TurnRow> {
        stmt.query_map(params, |row| {
            Ok(TurnRow {
                id: row.get(0)?,
                session_id: row.get(1)?,
                seq: row.get(2)?,
                role: row.get(3)?,
                content: row.get(4)?,
                tool_calls: row.get(5)?,
                tool_results: row.get(6)?,
                model: row.get(7)?,
                tokens_in: row.get(8)?,
                tokens_out: row.get(9)?,
                channel: row.get(10)?,
                language: row.get(11)?,
                voice_mode: row.get::<_, i32>(12).map(|v| v != 0).unwrap_or(false),
                created_at: row.get(13)?,
                metadata: row.get(14)?,
            })
        })
        .map(|rows| rows.filter_map(|r| r.ok()).collect())
        .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_store() -> (tempfile::TempDir, ConversationStore) {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test_store.db");
        let store = ConversationStore::new(&db_path).unwrap();
        (dir, store)
    }

    #[test]
    fn test_create_and_query_session() {
        let (_dir, store) = temp_store();

        let sid = store.ensure_session("telegram", "12345");
        assert!(!sid.is_empty());

        // Same channel+chat_id should return the same active session.
        let sid2 = store.ensure_session("telegram", "12345");
        assert_eq!(sid, sid2);

        // Different chat_id should create a new session.
        let sid3 = store.ensure_session("telegram", "99999");
        assert_ne!(sid, sid3);
    }

    #[test]
    fn test_end_session_creates_new() {
        let (_dir, store) = temp_store();

        let sid1 = store.ensure_session("cli", "direct");
        store.end_session(&sid1);

        // After ending, a new session should be created.
        let sid2 = store.ensure_session("cli", "direct");
        assert_ne!(sid1, sid2);
    }

    #[test]
    fn test_add_and_get_turns() {
        let (_dir, store) = temp_store();

        let sid = store.ensure_session("cli", "direct");

        let t1 = store.add_turn(&sid, "user", Some("Hello"), None, None, None, None, None, None, None, false, None);
        let t2 = store.add_turn(&sid, "assistant", Some("Hi!"), None, None, Some("gpt-4"), Some(10), Some(5), None, None, false, None);

        assert!(!t1.is_empty());
        assert!(!t2.is_empty());
        assert_ne!(t1, t2);

        let turns = store.get_session_turns(&sid, 100);
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].role, "user");
        assert_eq!(turns[0].content.as_deref(), Some("Hello"));
        assert_eq!(turns[0].seq, 1);
        assert_eq!(turns[1].role, "assistant");
        assert_eq!(turns[1].content.as_deref(), Some("Hi!"));
        assert_eq!(turns[1].seq, 2);
        assert_eq!(turns[1].model.as_deref(), Some("gpt-4"));
    }

    #[test]
    fn test_search_turns() {
        let (_dir, store) = temp_store();

        let sid = store.ensure_session("cli", "direct");
        store.add_turn(&sid, "user", Some("Tell me about Rust"), None, None, None, None, None, None, None, false, None);
        store.add_turn(&sid, "assistant", Some("Rust is a systems language"), None, None, None, None, None, None, None, false, None);
        store.add_turn(&sid, "user", Some("What about Python?"), None, None, None, None, None, None, None, false, None);

        let results = store.search_turns("Rust", 10);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_get_session_info() {
        let (_dir, store) = temp_store();

        let sid = store.ensure_session("whatsapp", "user1");
        let info = store.get_session_info(&sid).unwrap();
        assert_eq!(info.channel, "whatsapp");
        assert_eq!(info.chat_id, "user1");
        assert!(info.ended_at.is_none());
    }

    #[test]
    fn test_list_sessions() {
        let (_dir, store) = temp_store();

        store.ensure_session("telegram", "a");
        store.ensure_session("telegram", "b");
        store.ensure_session("whatsapp", "c");

        let tg_sessions = store.list_sessions("telegram", 10);
        assert_eq!(tg_sessions.len(), 2);

        let wa_sessions = store.list_sessions("whatsapp", 10);
        assert_eq!(wa_sessions.len(), 1);
    }

    #[test]
    fn test_get_recent_turns_across_sessions() {
        let (_dir, store) = temp_store();

        let s1 = store.ensure_session("cli", "direct");
        store.add_turn(&s1, "user", Some("First session msg"), None, None, None, None, None, None, None, false, None);
        store.end_session(&s1);

        let s2 = store.ensure_session("cli", "direct");
        store.add_turn(&s2, "user", Some("Second session msg"), None, None, None, None, None, None, None, false, None);

        let recent = store.get_recent_turns("cli", "direct", 10);
        assert_eq!(recent.len(), 2);
    }

    #[test]
    fn test_voice_mode_flag() {
        let (_dir, store) = temp_store();

        let sid = store.ensure_session("voice", "mic");
        store.add_turn(&sid, "user", Some("voice msg"), None, None, None, None, None, None, Some("en"), true, None);

        let turns = store.get_session_turns(&sid, 10);
        assert_eq!(turns.len(), 1);
        assert!(turns[0].voice_mode);
        assert_eq!(turns[0].language.as_deref(), Some("en"));
    }
}
