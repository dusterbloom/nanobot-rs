//! SQLite-backed session store.
//!
//! Replaces the JSONL `SessionManager` with a single SQLite database
//! at `~/.nanobot/sessions.db`. WAL mode for concurrent reads.
//!
//! # Design
//!
//! - One DB file for all sessions (no per-session files, no date rotation).
//! - Sessions are identified by a stable `session_key` (e.g. `cli:default`).
//! - `get_or_resume()` returns the most recent session for a key, creating one
//!   if none exists — no date boundary means multi-day conversations stay
//!   unbroken.
//! - `filter_history()` from the `filters` module applies the same
//!   windowing/clear-marker/orphan-skip logic that the JSONL manager used.

use std::path::Path;

use chrono::{DateTime, Utc};
use rusqlite::{params, Connection};
use serde_json::{json, Value};
use tokio::sync::Mutex;
use tracing::warn;

use super::filters::filter_history;

// ---------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS sessions (
    id            TEXT PRIMARY KEY,
    session_key   TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    message_count INTEGER DEFAULT 0,
    metadata      TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_sessions_key     ON sessions(session_key);
CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at DESC);

CREATE TABLE IF NOT EXISTS messages (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL REFERENCES sessions(id),
    role          TEXT NOT NULL,
    content       TEXT,
    tool_calls    TEXT,
    tool_call_id  TEXT,
    tool_name     TEXT,
    turn_tag      INTEGER,
    synthetic     INTEGER DEFAULT 0,
    timestamp     TEXT NOT NULL,
    metadata      TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_messages_session   ON messages(session_id, id);
CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content,
    role,
    content='messages',
    content_rowid='id',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content, role) VALUES (new.id, new.content, new.role);
END;

CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content, role) VALUES('delete', old.id, old.content, old.role);
END;

CREATE TABLE IF NOT EXISTS summary_nodes (
    id            INTEGER PRIMARY KEY,
    session_id    TEXT NOT NULL REFERENCES sessions(id),
    source_ids    TEXT NOT NULL,
    child_ids     TEXT DEFAULT '[]',
    text          TEXT NOT NULL,
    tokens        INTEGER NOT NULL,
    level         INTEGER NOT NULL,
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_summary_nodes_session ON summary_nodes(session_id);
"#;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Metadata for a session (returned by list/get operations).
#[derive(Debug, Clone)]
pub struct SessionMeta {
    pub id: String,
    pub session_key: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub message_count: usize,
}

/// A single search result from FTS5 full-text search.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub session_id: String,
    pub session_key: String,
    pub role: String,
    pub content: String,
    pub timestamp: String,
    pub snippet: String,
    pub rank: f64,
}

// ---------------------------------------------------------------------------
// SessionDb
// ---------------------------------------------------------------------------

/// SQLite-backed session store.
///
/// Thread-safe via a `tokio::sync::Mutex`. All public methods are `async` so
/// callers do not need to change their `.await` patterns relative to the old
/// `SessionManager`.
pub struct SessionDb {
    conn: Mutex<Connection>,
}

impl SessionDb {
    /// Open (or create) the database at `db_path`.
    ///
    /// Enables WAL journal mode and creates the schema on first run.
    pub fn new(db_path: &Path) -> Self {
        let conn = Connection::open(db_path).unwrap_or_else(|e| {
            panic!("Failed to open session DB at {}: {}", db_path.display(), e)
        });

        // Enable WAL for concurrent read access.
        conn.execute_batch("PRAGMA journal_mode=WAL;")
            .unwrap_or_else(|e| warn!("Could not enable WAL mode: {}", e));

        // Create schema.
        conn.execute_batch(SCHEMA)
            .unwrap_or_else(|e| panic!("Failed to initialise session DB schema: {}", e));

        Self {
            conn: Mutex::new(conn),
        }
    }

    // -----------------------------------------------------------------------
    // Session CRUD
    // -----------------------------------------------------------------------

    /// Create a brand-new session for `key` and return its metadata.
    pub async fn create_session(&self, key: &str) -> SessionMeta {
        let now = Utc::now();
        let id = generate_session_id(&now);
        let created_str = now.to_rfc3339();

        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT INTO sessions (id, session_key, created_at, updated_at, message_count, metadata) \
             VALUES (?1, ?2, ?3, ?3, 0, '{}')",
            params![id, key, created_str],
        )
        .unwrap_or_else(|e| {
            warn!("Failed to create session for key {}: {}", key, e);
            0
        });

        SessionMeta {
            id,
            session_key: key.to_string(),
            created_at: now,
            updated_at: now,
            message_count: 0,
        }
    }

    /// Return the most recent session for `key`, or create a new one.
    ///
    /// This is the primary entry point for gateway and default CLI usage.
    /// Unlike the old JSONL manager, there is no date-based rotation — a
    /// session for `key` lives until it is explicitly deleted or until
    /// `create_session()` is called explicitly.
    pub async fn get_or_resume(&self, key: &str) -> SessionMeta {
        if let Some(meta) = self.get_latest_session(key).await {
            return meta;
        }
        self.create_session(key).await
    }

    /// Load a session by its unique ID. Returns `None` if not found.
    pub async fn get_session(&self, id: &str) -> Option<SessionMeta> {
        let conn = self.conn.lock().await;
        conn.query_row(
            "SELECT id, session_key, created_at, updated_at, message_count \
             FROM sessions WHERE id = ?1",
            params![id],
            row_to_meta,
        )
        .ok()
    }

    /// Return the most recent session for `key` by `updated_at`, or `None`.
    pub async fn get_latest_session(&self, key: &str) -> Option<SessionMeta> {
        let conn = self.conn.lock().await;
        conn.query_row(
            "SELECT id, session_key, created_at, updated_at, message_count \
             FROM sessions WHERE session_key = ?1 \
             ORDER BY updated_at DESC LIMIT 1",
            params![key],
            row_to_meta,
        )
        .ok()
    }

    /// List sessions, optionally filtered to those whose `session_key` starts
    /// with `key_filter`. Results are ordered by `updated_at` descending.
    pub async fn list_sessions(&self, key_filter: Option<&str>, limit: usize) -> Vec<SessionMeta> {
        let conn = self.conn.lock().await;
        let mut stmt = match key_filter {
            Some(_) => conn
                .prepare(
                    "SELECT id, session_key, created_at, updated_at, message_count \
                     FROM sessions WHERE session_key LIKE ?1 \
                     ORDER BY updated_at DESC LIMIT ?2",
                )
                .ok(),
            None => conn
                .prepare(
                    "SELECT id, session_key, created_at, updated_at, message_count \
                     FROM sessions ORDER BY updated_at DESC LIMIT ?1",
                )
                .ok(),
        };

        let stmt = match stmt.as_mut() {
            Some(s) => s,
            None => return Vec::new(),
        };

        let rows: Result<Vec<SessionMeta>, _> = match key_filter {
            Some(filter) => {
                let pattern = format!("{}%", filter);
                stmt.query_map(params![pattern, limit as i64], row_to_meta)
                    .map(|rows| rows.flatten().collect())
            }
            None => stmt
                .query_map(params![limit as i64], row_to_meta)
                .map(|rows| rows.flatten().collect()),
        };

        rows.unwrap_or_default()
    }

    /// List sessions updated within a time range.
    pub async fn list_sessions_since(&self, since: &str, limit: usize) -> Vec<SessionMeta> {
        let conn = self.conn.lock().await;
        let mut stmt = conn
            .prepare(
                "SELECT id, session_key, created_at, updated_at, message_count \
                 FROM sessions WHERE updated_at >= ?1 \
                 ORDER BY updated_at DESC LIMIT ?2",
            )
            .unwrap();
        stmt.query_map(params![since, limit as i64], |row| {
            Ok(SessionMeta {
                id: row.get(0)?,
                session_key: row.get(1)?,
                created_at: row
                    .get::<_, String>(2)?
                    .parse()
                    .unwrap_or_else(|_| Utc::now()),
                updated_at: row
                    .get::<_, String>(3)?
                    .parse()
                    .unwrap_or_else(|_| Utc::now()),
                message_count: row.get::<_, i64>(4)? as usize,
            })
        })
        .unwrap()
        .filter_map(|r| r.ok())
        .collect()
    }

    // -----------------------------------------------------------------------
    // Message operations
    // -----------------------------------------------------------------------

    /// Return the filtered conversation history for `session_id`.
    ///
    /// Loads all messages from the DB, then applies the same multi-stage
    /// filtering pipeline (`filter_history`) used by the JSONL manager:
    /// max-messages window, clear-marker, orphaned-tool-result skipping,
    /// turn limit, and wire-format projection.
    pub async fn get_history(
        &self,
        session_id: &str,
        max_messages: usize,
        max_turns: usize,
    ) -> Vec<Value> {
        let raw = self.get_all_messages(session_id).await;
        filter_history(&raw, max_messages, max_turns)
    }

    /// Add a single raw JSON message to `session_id` and persist it.
    ///
    /// Extracts `role`, `content`, `tool_calls`, `tool_call_id`, `name`,
    /// `_turn`, `_synthetic`, and `timestamp` from the value. All other
    /// fields are stored in the `metadata` column as JSON.
    pub async fn add_message(&self, session_id: &str, msg: &Value) {
        let conn = self.conn.lock().await;
        insert_message_locked(&conn, session_id, msg);
    }

    /// Add a batch of raw JSON messages in a single transaction.
    pub async fn add_messages(&self, session_id: &str, msgs: &[Value]) {
        let conn = self.conn.lock().await;

        // Wrap in an explicit transaction so the whole batch is atomic and
        // the session's `updated_at` / `message_count` are updated once.
        let result = conn.execute_batch("BEGIN");
        if let Err(e) = result {
            warn!("Failed to begin transaction for batch insert: {}", e);
            return;
        }

        for msg in msgs {
            insert_message_locked(&conn, session_id, msg);
        }

        if let Err(e) = conn.execute_batch("COMMIT") {
            warn!("Failed to commit batch insert: {}", e);
            let _ = conn.execute_batch("ROLLBACK");
        }
    }

    /// Append a `role: "clear"` marker to `session_id`.
    ///
    /// Preserves the append-only audit trail: old messages remain in the DB
    /// but `get_history()` will ignore them (the filtering pipeline respects
    /// the most recent clear marker).
    pub async fn clear_history(&self, session_id: &str) {
        let clear_marker = json!({
            "role": "clear",
            "timestamp": Utc::now().to_rfc3339(),
        });
        self.add_message(session_id, &clear_marker).await;
    }

    /// Return all messages for `session_id` without any filtering, ordered by
    /// insertion order (ascending `id`). Used for export and LCM rebuild.
    pub async fn get_all_messages(&self, session_id: &str) -> Vec<Value> {
        let conn = self.conn.lock().await;
        let mut stmt = match conn.prepare(
            "SELECT role, content, tool_calls, tool_call_id, tool_name, \
                    turn_tag, synthetic, timestamp, metadata \
             FROM messages WHERE session_id = ?1 ORDER BY id ASC",
        ) {
            Ok(s) => s,
            Err(e) => {
                warn!("Failed to prepare get_all_messages query: {}", e);
                return Vec::new();
            }
        };

        let rows = stmt.query_map(params![session_id], |row| {
            let role: String = row.get(0)?;
            let content: Option<String> = row.get(1)?;
            let tool_calls_json: Option<String> = row.get(2)?;
            let tool_call_id: Option<String> = row.get(3)?;
            let tool_name: Option<String> = row.get(4)?;
            let turn_tag: Option<i64> = row.get(5)?;
            let synthetic: i64 = row.get(6)?;
            let timestamp: String = row.get(7)?;
            let metadata_json: String = row.get(8)?;
            Ok((
                role,
                content,
                tool_calls_json,
                tool_call_id,
                tool_name,
                turn_tag,
                synthetic,
                timestamp,
                metadata_json,
            ))
        });

        let rows = match rows {
            Ok(r) => r,
            Err(e) => {
                warn!("Failed to query messages for session {}: {}", session_id, e);
                return Vec::new();
            }
        };

        rows.flatten()
            .map(
                |(
                    role,
                    content,
                    tool_calls_json,
                    tool_call_id,
                    tool_name,
                    turn_tag,
                    synthetic,
                    timestamp,
                    metadata_json,
                )| {
                    reconstruct_message(
                        role,
                        content,
                        tool_calls_json,
                        tool_call_id,
                        tool_name,
                        turn_tag,
                        synthetic,
                        timestamp,
                        metadata_json,
                    )
                },
            )
            .collect()
    }

    pub async fn search_messages(
        &self,
        query: &str,
        limit: usize,
        session_key_filter: Option<&str>,
    ) -> Vec<SearchResult> {
        let conn = self.conn.lock().await;
        if let Some(key_filter) = session_key_filter {
            let sql = "SELECT m.session_id, s.session_key, m.role, m.content, m.timestamp,
                              snippet(messages_fts, 0, '>>>', '<<<', '...', 40) as snip, rank
                       FROM messages_fts
                       JOIN messages m ON m.id = messages_fts.rowid
                       JOIN sessions s ON s.id = m.session_id
                       WHERE messages_fts MATCH ?1 AND s.session_key LIKE ?2
                       ORDER BY rank LIMIT ?3";
            let pattern = format!("{}%", key_filter);
            let mut stmt = match conn.prepare(sql) {
                Ok(s) => s,
                Err(e) => {
                    warn!("FTS prepare failed: {}", e);
                    return Vec::new();
                }
            };
            stmt.query_map(params![query, pattern, limit as i64], |row| {
                Ok(SearchResult {
                    session_id: row.get(0)?,
                    session_key: row.get(1)?,
                    role: row.get(2)?,
                    content: row.get::<_, Option<String>>(3)?.unwrap_or_default(),
                    timestamp: row.get(4)?,
                    snippet: row.get::<_, Option<String>>(5)?.unwrap_or_default(),
                    rank: row.get(6)?,
                })
            })
            .map(|rows| rows.flatten().collect())
            .unwrap_or_default()
        } else {
            let sql = "SELECT m.session_id, s.session_key, m.role, m.content, m.timestamp,
                              snippet(messages_fts, 0, '>>>', '<<<', '...', 40) as snip, rank
                       FROM messages_fts
                       JOIN messages m ON m.id = messages_fts.rowid
                       JOIN sessions s ON s.id = m.session_id
                       WHERE messages_fts MATCH ?1
                       ORDER BY rank LIMIT ?2";
            let mut stmt = match conn.prepare(sql) {
                Ok(s) => s,
                Err(e) => {
                    warn!("FTS prepare failed: {}", e);
                    return Vec::new();
                }
            };
            stmt.query_map(params![query, limit as i64], |row| {
                Ok(SearchResult {
                    session_id: row.get(0)?,
                    session_key: row.get(1)?,
                    role: row.get(2)?,
                    content: row.get::<_, Option<String>>(3)?.unwrap_or_default(),
                    timestamp: row.get(4)?,
                    snippet: row.get::<_, Option<String>>(5)?.unwrap_or_default(),
                    rank: row.get(6)?,
                })
            })
            .map(|rows| rows.flatten().collect())
            .unwrap_or_default()
        }
    }

    // -----------------------------------------------------------------------
    // Summary DAG persistence (LCM)
    // -----------------------------------------------------------------------

    /// Persist a summary node for LCM's summary DAG.
    pub async fn save_summary_node(
        &self,
        session_id: &str,
        node_id: usize,
        source_ids: &[usize],
        child_ids: &[usize],
        text: &str,
        tokens: usize,
        level: u8,
    ) {
        let conn = self.conn.lock().await;
        let source_json = serde_json::to_string(source_ids).unwrap_or_else(|_| "[]".to_string());
        let child_json = serde_json::to_string(child_ids).unwrap_or_else(|_| "[]".to_string());
        let now = Utc::now().to_rfc3339();
        if let Err(e) = conn.execute(
            "INSERT OR REPLACE INTO summary_nodes \
             (id, session_id, source_ids, child_ids, text, tokens, level, created_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                node_id as i64,
                session_id,
                source_json,
                child_json,
                text,
                tokens as i64,
                level as i64,
                now,
            ],
        ) {
            warn!(
                "Failed to save summary node {} for session {}: {}",
                node_id, session_id, e
            );
        }
    }

    /// Load all summary nodes for a session, ordered by ID.
    ///
    /// Returns a vec of (node_id, source_ids, child_ids, text, tokens, level).
    pub async fn load_summary_nodes(
        &self,
        session_id: &str,
    ) -> Vec<(usize, Vec<usize>, Vec<usize>, String, usize, u8)> {
        let conn = self.conn.lock().await;
        let mut stmt = match conn.prepare(
            "SELECT id, source_ids, child_ids, text, tokens, level \
             FROM summary_nodes WHERE session_id = ?1 ORDER BY id ASC",
        ) {
            Ok(s) => s,
            Err(e) => {
                warn!("Failed to prepare load_summary_nodes query: {}", e);
                return Vec::new();
            }
        };

        let rows = stmt.query_map(params![session_id], |row| {
            let id: i64 = row.get(0)?;
            let source_str: String = row.get(1)?;
            let child_str: String = row.get(2)?;
            let text: String = row.get(3)?;
            let tokens: i64 = row.get(4)?;
            let level: i64 = row.get(5)?;
            Ok((id, source_str, child_str, text, tokens, level))
        });

        match rows {
            Ok(r) => r
                .flatten()
                .map(|(id, source_str, child_str, text, tokens, level)| {
                    let source_ids: Vec<usize> =
                        serde_json::from_str(&source_str).unwrap_or_default();
                    let child_ids: Vec<usize> =
                        serde_json::from_str(&child_str).unwrap_or_default();
                    (
                        id as usize,
                        source_ids,
                        child_ids,
                        text,
                        tokens as usize,
                        level as u8,
                    )
                })
                .collect(),
            Err(e) => {
                warn!(
                    "Failed to load summary nodes for session {}: {}",
                    session_id, e
                );
                Vec::new()
            }
        }
    }

    pub async fn rebuild_fts_index(&self) {
        let conn = self.conn.lock().await;
        conn.execute(
            "INSERT INTO messages_fts(messages_fts) VALUES('delete-all')",
            [],
        )
        .ok();
        conn.execute(
            "INSERT INTO messages_fts(rowid, content, role) SELECT id, content, role FROM messages",
            [],
        )
        .ok();
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Generate a session ID from the current timestamp.
///
/// Format: `YYYYMMDD_HHMMSS_XXXXXX` where the last segment is derived from
/// sub-second nanoseconds (ensures uniqueness within a single second).
fn generate_session_id(now: &DateTime<Utc>) -> String {
    let nanos = now.timestamp_subsec_nanos();
    format!("{}_{:06x}", now.format("%Y%m%d_%H%M%S"), nanos & 0xFF_FFFF)
}

/// Map a SQLite row to `SessionMeta`. Used by `query_row` / `query_map`.
fn row_to_meta(row: &rusqlite::Row<'_>) -> rusqlite::Result<SessionMeta> {
    let id: String = row.get(0)?;
    let session_key: String = row.get(1)?;
    let created_str: String = row.get(2)?;
    let updated_str: String = row.get(3)?;
    let message_count: i64 = row.get(4)?;

    let created_at = DateTime::parse_from_rfc3339(&created_str)
        .map(|dt| dt.with_timezone(&Utc))
        .unwrap_or_else(|_| Utc::now());
    let updated_at = DateTime::parse_from_rfc3339(&updated_str)
        .map(|dt| dt.with_timezone(&Utc))
        .unwrap_or_else(|_| Utc::now());

    Ok(SessionMeta {
        id,
        session_key,
        created_at,
        updated_at,
        message_count: message_count as usize,
    })
}

/// Insert a single message into the DB using an already-locked connection.
///
/// Called from both `add_message()` (which locks externally) and the batch
/// path in `add_messages()` (which holds the lock for the whole batch).
fn insert_message_locked(conn: &Connection, session_id: &str, msg: &Value) {
    let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("user");

    let content = msg.get("content").and_then(|v| v.as_str());

    let tool_calls_json: Option<String> = msg
        .get("tool_calls")
        .map(|tc| serde_json::to_string(tc).unwrap_or_default());

    let tool_call_id = msg.get("tool_call_id").and_then(|v| v.as_str());

    // `name` on tool-result messages maps to the `tool_name` column.
    let tool_name = msg.get("name").and_then(|v| v.as_str());

    let turn_tag: Option<i64> = msg.get("_turn").and_then(|v| v.as_i64());

    let synthetic: i64 = if msg
        .get("_synthetic")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
    {
        1
    } else {
        0
    };

    let timestamp = msg
        .get("timestamp")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| Utc::now().to_rfc3339());

    // Collect any remaining fields into `metadata` so nothing is lost.
    let reserved = [
        "role",
        "content",
        "tool_calls",
        "tool_call_id",
        "name",
        "_turn",
        "_synthetic",
        "timestamp",
    ];
    let metadata: serde_json::Map<String, Value> = msg
        .as_object()
        .map(|obj| {
            obj.iter()
                .filter(|(k, _)| !reserved.contains(&k.as_str()))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect()
        })
        .unwrap_or_default();
    let metadata_json = serde_json::to_string(&metadata).unwrap_or_else(|_| "{}".to_string());

    if let Err(e) = conn.execute(
        "INSERT INTO messages \
         (session_id, role, content, tool_calls, tool_call_id, tool_name, \
          turn_tag, synthetic, timestamp, metadata) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
        params![
            session_id,
            role,
            content,
            tool_calls_json,
            tool_call_id,
            tool_name,
            turn_tag,
            synthetic,
            timestamp,
            metadata_json,
        ],
    ) {
        warn!(
            "Failed to insert message into session {}: {}",
            session_id, e
        );
        return;
    }

    // Update the session's `updated_at` and increment `message_count`.
    let now_str = Utc::now().to_rfc3339();
    if let Err(e) = conn.execute(
        "UPDATE sessions SET updated_at = ?1, message_count = message_count + 1 WHERE id = ?2",
        params![now_str, session_id],
    ) {
        warn!(
            "Failed to update session metadata for {}: {}",
            session_id, e
        );
    }
}

/// Reconstruct a `serde_json::Value` from the columns stored in the `messages`
/// table. This is the inverse of the field extraction done in
/// `insert_message_locked()`.
fn reconstruct_message(
    role: String,
    content: Option<String>,
    tool_calls_json: Option<String>,
    tool_call_id: Option<String>,
    tool_name: Option<String>,
    turn_tag: Option<i64>,
    synthetic: i64,
    timestamp: String,
    metadata_json: String,
) -> Value {
    let mut msg = json!({
        "role": role,
        "content": content.unwrap_or_default(),
        "timestamp": timestamp,
    });

    if let Some(tc_str) = tool_calls_json {
        if let Ok(tc) = serde_json::from_str::<Value>(&tc_str) {
            msg["tool_calls"] = tc;
        }
    }

    if let Some(id) = tool_call_id {
        msg["tool_call_id"] = json!(id);
    }

    if let Some(name) = tool_name {
        msg["name"] = json!(name);
    }

    if let Some(turn) = turn_tag {
        msg["_turn"] = json!(turn);
    }

    if synthetic != 0 {
        msg["_synthetic"] = json!(true);
    }

    // Merge any extra metadata fields back into the top-level object.
    if let Ok(Value::Object(extra)) = serde_json::from_str::<Value>(&metadata_json) {
        if let Some(obj) = msg.as_object_mut() {
            for (k, v) in extra {
                obj.entry(k).or_insert(v);
            }
        }
    }

    msg
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::tempdir;

    fn make_db() -> (SessionDb, tempfile::TempDir) {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("sessions.db");
        let db = SessionDb::new(&db_path);
        (db, dir)
    }

    // -----------------------------------------------------------------------
    // Session lifecycle
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_create_session() {
        let (db, _dir) = make_db();
        let meta = db.create_session("cli:default").await;

        assert_eq!(meta.session_key, "cli:default");
        assert_eq!(meta.message_count, 0);
        assert!(!meta.id.is_empty());

        // Should be retrievable by ID.
        let loaded = db.get_session(&meta.id).await.expect("session must exist");
        assert_eq!(loaded.id, meta.id);
        assert_eq!(loaded.session_key, "cli:default");
    }

    #[tokio::test]
    async fn test_get_or_resume_idempotent() {
        let (db, _dir) = make_db();

        let first = db.get_or_resume("telegram:42").await;
        let second = db.get_or_resume("telegram:42").await;

        // Second call must return the SAME session, not a new one.
        assert_eq!(
            first.id, second.id,
            "get_or_resume must resume the existing session"
        );
    }

    #[tokio::test]
    async fn test_get_or_resume_creates_when_none() {
        let (db, _dir) = make_db();
        let meta = db.get_or_resume("new:channel").await;

        assert_eq!(meta.session_key, "new:channel");
        assert!(!meta.id.is_empty());
    }

    #[tokio::test]
    async fn test_get_session_returns_none_for_missing_id() {
        let (db, _dir) = make_db();
        let result = db.get_session("nonexistent_id_xyz").await;
        assert!(result.is_none());
    }

    // -----------------------------------------------------------------------
    // Message round-trip
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_add_and_get_messages() {
        let (db, _dir) = make_db();
        let meta = db.create_session("cli:test").await;

        db.add_message(&meta.id, &json!({"role": "user", "content": "hello"}))
            .await;
        db.add_message(
            &meta.id,
            &json!({"role": "assistant", "content": "hi there"}),
        )
        .await;

        let history = db.get_history(&meta.id, 100, 0).await;
        assert_eq!(history.len(), 2);
        assert_eq!(history[0]["role"], "user");
        assert_eq!(history[0]["content"], "hello");
        assert_eq!(history[1]["role"], "assistant");
        assert_eq!(history[1]["content"], "hi there");
    }

    #[tokio::test]
    async fn test_add_messages_batch() {
        let (db, _dir) = make_db();
        let meta = db.create_session("cli:batch").await;

        let msgs = vec![
            json!({"role": "user", "content": "q1"}),
            json!({"role": "assistant", "content": "a1"}),
            json!({"role": "user", "content": "q2"}),
            json!({"role": "assistant", "content": "a2"}),
        ];
        db.add_messages(&meta.id, &msgs).await;

        let all = db.get_all_messages(&meta.id).await;
        assert_eq!(all.len(), 4);
    }

    #[tokio::test]
    async fn test_tool_calls_preserved() {
        let (db, _dir) = make_db();
        let meta = db.create_session("test:tools").await;

        let msgs = vec![
            json!({"role": "user", "content": "Read /tmp/test.txt"}),
            json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "tc_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{\"path\":\"/tmp/test.txt\"}"}
                }]
            }),
            json!({
                "role": "tool",
                "tool_call_id": "tc_1",
                "name": "read_file",
                "content": "file contents here"
            }),
            json!({"role": "assistant", "content": "The file contains: file contents here"}),
        ];
        db.add_messages(&meta.id, &msgs).await;

        let history = db.get_history(&meta.id, 100, 0).await;
        assert_eq!(history.len(), 4);

        // tool_calls must survive the round-trip.
        assert!(
            history[1].get("tool_calls").is_some(),
            "tool_calls must be preserved on assistant message"
        );

        // tool_call_id and name must survive.
        assert_eq!(
            history[2].get("tool_call_id").and_then(|v| v.as_str()),
            Some("tc_1")
        );
        assert_eq!(
            history[2].get("name").and_then(|v| v.as_str()),
            Some("read_file")
        );
    }

    #[tokio::test]
    async fn test_turn_tag_preserved() {
        let (db, _dir) = make_db();
        let meta = db.create_session("test:turn_tag").await;

        db.add_message(
            &meta.id,
            &json!({"role": "user", "content": "hello", "_turn": 7}),
        )
        .await;

        let all = db.get_all_messages(&meta.id).await;
        assert_eq!(all[0].get("_turn").and_then(|v| v.as_i64()), Some(7));
    }

    #[tokio::test]
    async fn test_synthetic_flag_preserved() {
        let (db, _dir) = make_db();
        let meta = db.create_session("test:synthetic").await;

        db.add_message(
            &meta.id,
            &json!({"role": "user", "content": "injected", "_synthetic": true}),
        )
        .await;

        // get_all_messages returns it raw (with the flag).
        let all = db.get_all_messages(&meta.id).await;
        assert_eq!(
            all[0].get("_synthetic").and_then(|v| v.as_bool()),
            Some(true)
        );

        // get_history must filter it out.
        let history = db.get_history(&meta.id, 100, 0).await;
        assert!(
            history.is_empty(),
            "synthetic messages must be filtered by get_history"
        );
    }

    // -----------------------------------------------------------------------
    // Clear history
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_clear_history() {
        let (db, _dir) = make_db();
        let meta = db.create_session("cli:clear_test").await;

        db.add_messages(
            &meta.id,
            &[
                json!({"role": "user", "content": "old question"}),
                json!({"role": "assistant", "content": "old answer"}),
            ],
        )
        .await;

        db.clear_history(&meta.id).await;

        db.add_messages(
            &meta.id,
            &[
                json!({"role": "user", "content": "new question"}),
                json!({"role": "assistant", "content": "new answer"}),
            ],
        )
        .await;

        // Filtered history must only show the post-clear messages.
        let history = db.get_history(&meta.id, 100, 0).await;
        assert_eq!(history.len(), 2, "only post-clear messages should appear");
        assert_eq!(history[0]["content"], "new question");
        assert_eq!(history[1]["content"], "new answer");

        // Raw get_all_messages still has everything (clear marker too).
        let all = db.get_all_messages(&meta.id).await;
        assert_eq!(all.len(), 5, "raw history should include clear marker");
    }

    // -----------------------------------------------------------------------
    // list_sessions
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_list_sessions_no_filter() {
        let (db, _dir) = make_db();
        db.create_session("telegram:1").await;
        db.create_session("telegram:2").await;
        db.create_session("cli:default").await;

        let all = db.list_sessions(None, 100).await;
        assert_eq!(all.len(), 3);
    }

    #[tokio::test]
    async fn test_list_sessions_key_filter() {
        let (db, _dir) = make_db();
        db.create_session("telegram:1").await;
        db.create_session("telegram:2").await;
        db.create_session("cli:default").await;

        let telegram_only = db.list_sessions(Some("telegram"), 100).await;
        assert_eq!(telegram_only.len(), 2);
        assert!(telegram_only
            .iter()
            .all(|m| m.session_key.starts_with("telegram")));
    }

    #[tokio::test]
    async fn test_list_sessions_limit() {
        let (db, _dir) = make_db();
        for i in 0..5 {
            db.create_session(&format!("key:{}", i)).await;
        }

        let limited = db.list_sessions(None, 3).await;
        assert_eq!(limited.len(), 3);
    }

    // -----------------------------------------------------------------------
    // get_latest_session
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_get_latest_session() {
        let (db, _dir) = make_db();

        // Create two sessions for the same key, then update the second one.
        let first = db.create_session("cli:default").await;
        let second = db.create_session("cli:default").await;

        // Add a message to the second session to bump its updated_at.
        db.add_message(&second.id, &json!({"role": "user", "content": "bump"}))
            .await;

        let latest = db
            .get_latest_session("cli:default")
            .await
            .expect("must find a session");
        assert_eq!(
            latest.id, second.id,
            "get_latest_session must return the most recently updated session"
        );
        // Suppress unused-variable warning; we kept `first` to verify ordering.
        let _ = first;
    }

    // -----------------------------------------------------------------------
    // Cross-day resume — the key correctness goal
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_cross_day_resume_no_date_dependency() {
        // The SQLite store has no date-based rotation. A session created
        // "yesterday" (simulated by explicit `create_session`) must be
        // resumable today via `get_or_resume` without creating a new session.
        let (db, _dir) = make_db();

        let original = db.create_session("cli:default").await;
        db.add_message(
            &original.id,
            &json!({"role": "user", "content": "message from yesterday"}),
        )
        .await;

        // Simulate "today" resumption — must return the same session.
        let resumed = db.get_or_resume("cli:default").await;
        assert_eq!(
            resumed.id, original.id,
            "cross-day resume must return the same session, not create a new one"
        );

        let history = db.get_history(&resumed.id, 100, 0).await;
        assert_eq!(history.len(), 1);
        assert_eq!(history[0]["content"], "message from yesterday");
    }

    // -----------------------------------------------------------------------
    // message_count tracking
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_message_count_increments() {
        let (db, _dir) = make_db();
        let meta = db.create_session("cli:count").await;
        assert_eq!(meta.message_count, 0);

        db.add_message(&meta.id, &json!({"role": "user", "content": "one"}))
            .await;
        db.add_message(&meta.id, &json!({"role": "assistant", "content": "two"}))
            .await;

        let loaded = db.get_session(&meta.id).await.expect("session exists");
        assert_eq!(loaded.message_count, 2);
    }

    #[tokio::test]
    async fn test_fts_search_basic() {
        let (db, _dir) = make_db();
        let meta = db.create_session("cli:fts").await;
        db.add_messages(
            &meta.id,
            &[
                json!({"role": "user", "content": "What is the capital of France?"}),
                json!({"role": "assistant", "content": "The capital of France is Paris."}),
            ],
        )
        .await;
        let results = db.search_messages("Paris", 10, None).await;
        assert!(!results.is_empty());
        assert!(results[0].content.contains("Paris"));
    }

    #[tokio::test]
    async fn test_fts_search_no_match() {
        let (db, _dir) = make_db();
        let meta = db.create_session("cli:fts2").await;
        db.add_message(&meta.id, &json!({"role": "user", "content": "Hello world"}))
            .await;
        let results = db.search_messages("xyznonexistent", 10, None).await;
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_fts_search_key_filter() {
        let (db, _dir) = make_db();
        let cli = db.create_session("cli:default").await;
        let tg = db.create_session("telegram:42").await;
        db.add_message(
            &cli.id,
            &json!({"role": "user", "content": "CLI Rust question"}),
        )
        .await;
        db.add_message(
            &tg.id,
            &json!({"role": "user", "content": "Telegram Rust question"}),
        )
        .await;
        let results = db.search_messages("Rust", 10, Some("cli:")).await;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].session_key, "cli:default");
    }

    #[tokio::test]
    async fn test_fts_rebuild() {
        let (db, _dir) = make_db();
        let meta = db.create_session("cli:rebuild").await;
        db.add_message(
            &meta.id,
            &json!({"role": "user", "content": "Rebuild test message"}),
        )
        .await;
        db.rebuild_fts_index().await;
        let results = db.search_messages("Rebuild", 10, None).await;
        assert!(!results.is_empty());
    }

    // -----------------------------------------------------------------------
    // Migration correctness: verify SessionDb handles all SessionManager scenarios
    // -----------------------------------------------------------------------

    /// Integration test: SessionDb must handle the same session_key patterns as SessionManager.
    /// Keys like "cli:default", "telegram:12345", "disk:session" must all work.
    #[tokio::test]
    async fn test_session_key_patterns_match_legacy() {
        let (db, _dir) = make_db();

        let keys = vec![
            "cli:default",
            "telegram:12345",
            "disk:session",
            "email:user@example.com",
        ];
        let mut ids = Vec::new();

        for key in &keys {
            let meta = db.create_session(key).await;
            ids.push(meta.id);
            assert_eq!(
                meta.session_key, *key,
                "session_key must preserve the original key format"
            );
        }

        for (i, key) in keys.iter().enumerate() {
            let loaded = db.get_session(&ids[i]).await.expect("session must exist");
            assert_eq!(loaded.session_key, *key);
        }
    }

    /// Integration test: filters module must work with SessionDb (same as with JSONL).
    #[tokio::test]
    async fn test_filters_integration_with_db() {
        let (db, _dir) = make_db();
        let meta = db.create_session("cli:filter_test").await;

        db.add_messages(
            &meta.id,
            &[
                json!({"role": "user", "content": "q1"}),
                json!({"role": "assistant", "content": "a1"}),
                json!({"role": "clear", "timestamp": "2024-01-01T00:00:00Z"}),
                json!({"role": "user", "content": "q2"}),
                json!({"role": "assistant", "content": "a2"}),
            ],
        )
        .await;

        let all = db.get_all_messages(&meta.id).await;
        let filtered = filter_history(&all, 100, 0);

        assert_eq!(filtered.len(), 2, "clear marker must truncate history");
        assert_eq!(filtered[0]["content"], "q2");
    }

    // -----------------------------------------------------------------------
    // Summary DAG persistence
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_dag_persistence_roundtrip() {
        let (db, _dir) = make_db();
        let meta = db.create_session("cli:dag_test").await;

        // Save two summary nodes.
        db.save_summary_node(
            &meta.id,
            0,
            &[1, 2, 3, 4],
            &[],
            "Summary of greeting exchange.",
            15,
            1,
        )
        .await;

        db.save_summary_node(
            &meta.id,
            1,
            &[5, 6, 7],
            &[],
            "Summary of technical discussion.",
            12,
            2,
        )
        .await;

        // Load them back.
        let nodes = db.load_summary_nodes(&meta.id).await;
        assert_eq!(nodes.len(), 2);

        let (id0, src0, child0, text0, tokens0, level0) = &nodes[0];
        assert_eq!(*id0, 0);
        assert_eq!(*src0, vec![1, 2, 3, 4]);
        assert!(child0.is_empty());
        assert_eq!(text0, "Summary of greeting exchange.");
        assert_eq!(*tokens0, 15);
        assert_eq!(*level0, 1);

        let (id1, src1, _child1, text1, tokens1, level1) = &nodes[1];
        assert_eq!(*id1, 1);
        assert_eq!(*src1, vec![5, 6, 7]);
        assert_eq!(text1, "Summary of technical discussion.");
        assert_eq!(*tokens1, 12);
        assert_eq!(*level1, 2);
    }

    #[tokio::test]
    async fn test_dag_persistence_empty_session() {
        let (db, _dir) = make_db();
        let meta = db.create_session("cli:empty").await;

        let nodes = db.load_summary_nodes(&meta.id).await;
        assert!(nodes.is_empty());
    }

    #[tokio::test]
    async fn test_dag_persistence_upsert() {
        let (db, _dir) = make_db();
        let meta = db.create_session("cli:upsert").await;

        // Save, then overwrite with updated text.
        db.save_summary_node(&meta.id, 0, &[1, 2], &[], "Original.", 5, 1)
            .await;
        db.save_summary_node(&meta.id, 0, &[1, 2], &[], "Updated.", 6, 1)
            .await;

        let nodes = db.load_summary_nodes(&meta.id).await;
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].3, "Updated.");
        assert_eq!(nodes[0].4, 6);
    }
}
