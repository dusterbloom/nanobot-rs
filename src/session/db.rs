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

use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use rusqlite::{params, types::Value as SqlValue, Connection, OptionalExtension, Transaction};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tokio::sync::Mutex;
use tracing::warn;

use crate::agent::lcm::SummaryManifest;

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

CREATE TABLE IF NOT EXISTS tool_results (
    session_id    TEXT NOT NULL REFERENCES sessions(id),
    tool_call_id  TEXT NOT NULL,
    tool_name     TEXT NOT NULL,
    content       TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    PRIMARY KEY (session_id, tool_call_id)
);
CREATE INDEX IF NOT EXISTS idx_tool_results_session ON tool_results(session_id);

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
    id            INTEGER NOT NULL,
    session_id    TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    source_ids    TEXT NOT NULL,
    child_ids     TEXT DEFAULT '[]',
    text          TEXT NOT NULL,
    tokens        INTEGER NOT NULL,
    level         INTEGER NOT NULL,
    created_at    TEXT NOT NULL,
    id_kind       TEXT,
    manifest_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (session_id, id)
);
CREATE INDEX IF NOT EXISTS idx_summary_nodes_session ON summary_nodes(session_id);

CREATE TABLE IF NOT EXISTS session_snapshots (
    session_key     TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL,
    version         INTEGER NOT NULL,
    cwd             TEXT NOT NULL,
    model           TEXT NOT NULL,
    tui_mode        TEXT NOT NULL,
    show_thinking   INTEGER NOT NULL,
    input_draft     TEXT NOT NULL,
    prompt_history  TEXT NOT NULL,
    recent_paths    TEXT NOT NULL,
    recent_commands TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS working_memory (
    session_id        TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
    content           TEXT NOT NULL DEFAULT '',
    status            TEXT NOT NULL DEFAULT 'active'
                      CHECK(status IN ('active', 'completed', 'reflected')),
    last_updated_turn INTEGER NOT NULL DEFAULT 0,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_working_memory_status
    ON working_memory(status, updated_at DESC);

CREATE TABLE IF NOT EXISTS legacy_imports (
    source_path    TEXT PRIMARY KEY,
    content_sha256 TEXT NOT NULL,
    session_id     TEXT NOT NULL,
    message_count  INTEGER NOT NULL,
    imported_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_legacy_imports_session
    ON legacy_imports(session_id);
CREATE INDEX IF NOT EXISTS idx_legacy_imports_content
    ON legacy_imports(content_sha256);
"#;

const WORKING_MEMORY_COLUMNS: &str = "\
    SELECT wm.session_id, s.session_key, wm.content, wm.status, \
           wm.last_updated_turn, wm.created_at, wm.updated_at \
    FROM working_memory wm JOIN sessions s ON s.id = wm.session_id";
const WORKING_MEMORY_SELECT_ALL: &str = "\
    SELECT wm.session_id, s.session_key, wm.content, wm.status, \
           wm.last_updated_turn, wm.created_at, wm.updated_at \
    FROM working_memory wm JOIN sessions s ON s.id = wm.session_id \
    ORDER BY wm.updated_at DESC";
const WORKING_MEMORY_SELECT_WITH_STATUS: &str = "\
    SELECT wm.session_id, s.session_key, wm.content, wm.status, \
           wm.last_updated_turn, wm.created_at, wm.updated_at \
    FROM working_memory wm JOIN sessions s ON s.id = wm.session_id \
    WHERE wm.status = ?1 ORDER BY wm.updated_at DESC";

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

/// SQL-level constraints applied before FTS rank ordering and limiting.
///
/// Generic internal search intentionally keeps its historical all-message
/// behavior. Agent-facing conversation discovery uses the canonical scope so
/// active-turn echoes, tool payloads, and injected scaffolding cannot outrank
/// the actual past conversation.
#[derive(Debug, Clone, Copy, Default)]
struct MessageSearchScope<'a> {
    session_key_prefix: Option<&'a str>,
    exclude_session_id: Option<&'a str>,
    canonical_conversation_only: bool,
}

/// The final user/assistant exchange of a past session, used for
/// cross-session continuity and latest-session discovery.
#[derive(Debug, Clone)]
pub struct SessionTail {
    pub session_id: String,
    pub session_key: String,
    pub updated_at: DateTime<Utc>,
    pub last_user: String,
    pub last_assistant: String,
}

/// UI/workspace state restored when the TUI resumes a session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionSnapshot {
    pub version: u32,
    pub session_key: String,
    pub session_id: String,
    pub cwd: String,
    pub model: String,
    pub tui_mode: String,
    pub show_thinking: bool,
    pub input_draft: String,
    pub prompt_history: Vec<String>,
    pub recent_paths: Vec<String>,
    pub recent_commands: Vec<String>,
    pub updated_at: DateTime<Utc>,
}

/// SQLite row backing the active-task summary for one concrete session.
///
/// This is deliberately keyed by `session_id`, not the reusable channel/chat
/// key. A fresh session created after an idle timeout must never inherit the
/// previous session's compacted working state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkingMemoryRecord {
    pub session_id: String,
    pub session_key: String,
    pub content: String,
    pub status: String,
    pub last_updated_turn: u64,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Result of importing one legacy JSONL session file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LegacyImportOutcome {
    Imported {
        session_id: String,
        message_count: usize,
    },
    AlreadyImported {
        session_id: String,
        message_count: usize,
    },
}

/// A legacy file is immutable once imported. Reusing its path with different
/// bytes is rejected so migration never silently duplicates or rewrites a
/// historical session.
#[derive(Debug)]
pub enum LegacyImportError {
    Io(std::io::Error),
    InvalidJson {
        line: usize,
        source: serde_json::Error,
    },
    ChangedFile {
        path: PathBuf,
        imported_sha256: String,
        current_sha256: String,
    },
    Database(rusqlite::Error),
    MessageInsert {
        line: usize,
    },
    MissingSessionKey,
}

impl fmt::Display for LegacyImportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "legacy JSONL I/O failed: {error}"),
            Self::InvalidJson { line, source } => {
                write!(f, "legacy JSONL line {line} is invalid: {source}")
            }
            Self::ChangedFile { path, .. } => {
                write!(f, "legacy JSONL changed after import: {}", path.display())
            }
            Self::Database(error) => write!(f, "legacy JSONL database import failed: {error}"),
            Self::MessageInsert { line } => {
                write!(f, "legacy JSONL message insert failed at line {line}")
            }
            Self::MissingSessionKey => write!(
                f,
                "legacy JSONL has no metadata session_key and no fallback was supplied"
            ),
        }
    }
}

impl std::error::Error for LegacyImportError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::InvalidJson { source, .. } => Some(source),
            Self::Database(error) => Some(error),
            Self::ChangedFile { .. } | Self::MessageInsert { .. } | Self::MissingSessionKey => None,
        }
    }
}

impl From<std::io::Error> for LegacyImportError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<rusqlite::Error> for LegacyImportError {
    fn from(value: rusqlite::Error) -> Self {
        Self::Database(value)
    }
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
    path: PathBuf,
}

/// Outcome of [`SessionDb::store_tool_result_immutable`]. The tool-result
/// stash is keyed by `(session_id, tool_call_id)` and, under the
/// "handles-not-bodies" invariant, a prompt handle references the body by
/// digest. A silent overwrite would make that handle lie (point at different
/// bytes than its `sha256` claims), so storage is IMMUTABLE: identical retries
/// are accepted; conflicting bytes are rejected; SQLite failures surface
/// explicitly so the caller can fail the turn rather than show a raw body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StoredResult {
    /// Newly stored this call.
    Stored { digest: String },
    /// Key already present with byte-identical content (idempotent retry).
    Identical { digest: String },
    /// Key already present with DIFFERENT bytes — the body is ambiguous and a
    /// handle MUST NOT be emitted for it. The caller surfaces this.
    Conflict {
        existing_digest: String,
        attempted_digest: String,
    },
    /// SQLite failure (disk full, locked, etc.). The caller must fail the turn.
    Failed,
}

impl SessionDb {
    /// Open (or create) the database at `db_path`.
    ///
    /// Enables WAL journal mode and creates the schema on first run.
    pub fn new(db_path: &Path) -> Self {
        let conn = Connection::open(db_path).unwrap_or_else(|e| {
            panic!("Failed to open session DB at {}: {}", db_path.display(), e)
        });

        // WAL permits concurrent readers. Foreign keys make newly-created
        // databases self-cleaning; deletion methods below still delete every
        // dependent table explicitly so databases created by older versions
        // (whose foreign keys lacked ON DELETE clauses) behave identically.
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")
            .unwrap_or_else(|e| warn!("Could not enable WAL mode: {}", e));

        // Create schema.
        conn.execute_batch(SCHEMA)
            .unwrap_or_else(|e| panic!("Failed to initialise session DB schema: {}", e));

        // Migration: summary_nodes.id_kind (added when LCM MessageId became
        // the messages rowid). The error on already-migrated DBs (duplicate
        // column) is expected and ignored.
        let _ = conn.execute("ALTER TABLE summary_nodes ADD COLUMN id_kind TEXT", []);
        let _ = conn.execute(
            "ALTER TABLE summary_nodes ADD COLUMN manifest_json TEXT NOT NULL DEFAULT '{}'",
            [],
        );

        // Pre-migration rows (id_kind absent) carry POSITIONAL source_ids
        // that cannot be resolved against the db-id-keyed LCM store. Purge
        // them once here instead of skipping them on every load. Idempotent:
        // subsequent opens find no such rows and stay silent.
        match conn.execute(
            "DELETE FROM summary_nodes WHERE id_kind IS NULL OR id_kind != 'db_id'",
            [],
        ) {
            Ok(purged) if purged > 0 => warn!(
                "purged {} legacy summary nodes with unresolvable positional ids; \
                 their text remains in session history",
                purged
            ),
            Ok(_) => {}
            Err(e) => warn!("Failed to purge legacy summary nodes: {}", e),
        }

        // Early SQLite builds keyed LCM node ids globally even though the DAG
        // allocates them per session. Migrate once to the composite key before
        // any writer can replace another session's node with the same id.
        let session_id_is_key: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('summary_nodes') \
                 WHERE name = 'session_id' AND pk > 0",
                [],
                |row| row.get(0),
            )
            .unwrap_or(0);
        if session_id_is_key == 0 {
            conn.execute_batch(
                "BEGIN IMMEDIATE;
                 CREATE TABLE summary_nodes_v2 (
                     id INTEGER NOT NULL,
                     session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                     source_ids TEXT NOT NULL,
                     child_ids TEXT DEFAULT '[]',
                     text TEXT NOT NULL,
                     tokens INTEGER NOT NULL,
                     level INTEGER NOT NULL,
                     created_at TEXT NOT NULL,
                     id_kind TEXT,
                     manifest_json TEXT NOT NULL DEFAULT '{}',
                     PRIMARY KEY (session_id, id)
                 );
                 INSERT INTO summary_nodes_v2
                     (id, session_id, source_ids, child_ids, text, tokens, level, created_at, id_kind,
                      manifest_json)
                 SELECT id, session_id, source_ids, child_ids, text, tokens, level, created_at,
                        id_kind, COALESCE(manifest_json, '{}')
                 FROM summary_nodes;
                 DROP TABLE summary_nodes;
                 ALTER TABLE summary_nodes_v2 RENAME TO summary_nodes;
                 CREATE INDEX idx_summary_nodes_session ON summary_nodes(session_id);
                 COMMIT;",
            )
            .unwrap_or_else(|error| panic!("Failed to migrate LCM summary node keys: {error}"));
        }

        // Import provenance outlives imported sessions. Older schemas tied the
        // journal to `sessions` with ON DELETE CASCADE, which allowed deleting
        // a session and then importing the same immutable JSONL again. Detach
        // the journal while retaining its historical session id for audit.
        let legacy_imports_has_session_fk: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM pragma_foreign_key_list('legacy_imports')",
                [],
                |row| row.get(0),
            )
            .unwrap_or(0);
        if legacy_imports_has_session_fk > 0 {
            conn.execute_batch(
                "BEGIN IMMEDIATE;
                 CREATE TABLE legacy_imports_v2 (
                     source_path TEXT PRIMARY KEY,
                     content_sha256 TEXT NOT NULL,
                     session_id TEXT NOT NULL,
                     message_count INTEGER NOT NULL,
                     imported_at TEXT NOT NULL
                 );
                 INSERT INTO legacy_imports_v2
                     (source_path, content_sha256, session_id, message_count, imported_at)
                 SELECT source_path, content_sha256, session_id, message_count, imported_at
                 FROM legacy_imports;
                 DROP TABLE legacy_imports;
                 ALTER TABLE legacy_imports_v2 RENAME TO legacy_imports;
                 CREATE INDEX idx_legacy_imports_session ON legacy_imports(session_id);
                 CREATE INDEX idx_legacy_imports_content ON legacy_imports(content_sha256);
                 COMMIT;",
            )
            .unwrap_or_else(|error| panic!("Failed to detach legacy import provenance: {error}"));
        }

        Self {
            conn: Mutex::new(conn),
            path: db_path.to_path_buf(),
        }
    }

    /// Filesystem location backing this handle. Used by restart-safe tools
    /// that open a short-lived read handle without sharing the live mutex.
    pub fn path(&self) -> &Path {
        &self.path
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
    /// Resume the most recent session for `key`, or create a new one.
    ///
    /// If `max_idle_secs > 0` and the latest session hasn't been updated in
    /// that many seconds, a fresh session is created instead of resuming the
    /// stale one. Pass `0` to always resume (the old behaviour).
    pub async fn get_or_resume(&self, key: &str) -> SessionMeta {
        self.get_or_resume_with_idle(key, 0).await
    }

    /// Like [`get_or_resume`] but with an explicit idle timeout.
    pub async fn get_or_resume_with_idle(&self, key: &str, max_idle_secs: u64) -> SessionMeta {
        if let Some(meta) = self.get_latest_session(key).await {
            if max_idle_secs > 0 {
                let idle = chrono::Utc::now() - meta.updated_at;
                if idle.num_seconds() > max_idle_secs as i64 {
                    tracing::info!(
                        session_key = %key,
                        idle_secs = idle.num_seconds(),
                        max_idle_secs,
                        "session_expired: creating fresh session"
                    );
                    let now = Utc::now().to_rfc3339();
                    let conn = self.conn.lock().await;
                    if let Err(error) =
                        set_working_memory_status_locked(&conn, &meta.id, "completed", &now)
                    {
                        warn!(
                            session_id = %meta.id,
                            %error,
                            "failed to complete expired session working memory"
                        );
                    }
                    drop(conn);
                    return self.create_session(key).await;
                }
            }

            // A completed/reflected row is terminal for one inactive epoch, not
            // for the concrete session forever. Continuing the session reopens
            // its latest snapshot so the next LCM checkpoint can replace it.
            let now = Utc::now().to_rfc3339();
            let conn = self.conn.lock().await;
            if let Err(error) = reactivate_working_memory_locked(&conn, &meta.id, &now) {
                warn!(
                    session_id = %meta.id,
                    %error,
                    "failed to reactivate resumed session working memory"
                );
            }
            return meta;
        }
        self.create_session(key).await
    }

    /// Select one concrete session for explicit resume.
    ///
    /// Touching `updated_at` makes that exact ID the latest session for its
    /// reusable key before the normal message path resolves the key. The same
    /// transaction reopens working memory while preserving its content/turn.
    pub async fn resume_session(&self, id: &str) -> rusqlite::Result<Option<SessionMeta>> {
        let now = Utc::now().to_rfc3339();
        let mut conn = self.conn.lock().await;
        let tx = conn.transaction()?;
        let changed = tx.execute(
            "UPDATE sessions SET updated_at = ?2 WHERE id = ?1",
            params![id, now],
        )?;
        if changed == 0 {
            return Ok(None);
        }
        reactivate_working_memory_locked(&tx, id, &now)?;
        tx.commit()?;
        drop(conn);
        Ok(self.get_session(id).await)
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

    /// Delete one session and every derived/persisted row that belongs to it.
    ///
    /// The cascade is explicit and transactional for compatibility with
    /// databases created before foreign-key cascades were enabled.
    pub async fn delete_session(&self, session_id: &str) -> rusqlite::Result<bool> {
        let mut conn = self.conn.lock().await;
        let tx = conn.transaction()?;
        let deleted = delete_session_rows(&tx, session_id)?;
        tx.commit()?;
        Ok(deleted)
    }

    /// Delete sessions whose last update predates `cutoff` in one transaction.
    pub async fn purge_sessions_before(&self, cutoff: DateTime<Utc>) -> rusqlite::Result<usize> {
        let mut conn = self.conn.lock().await;
        let tx = conn.transaction()?;
        let session_ids = select_session_ids(
            &tx,
            "SELECT id FROM sessions WHERE updated_at < ?1 ORDER BY id",
            Some(cutoff.to_rfc3339()),
        )?;
        let mut deleted = 0;
        for session_id in session_ids {
            deleted += usize::from(delete_session_rows(&tx, &session_id)?);
        }
        tx.commit()?;
        Ok(deleted)
    }

    /// Delete all sessions and all session-owned rows in one transaction.
    /// Immutable legacy-import provenance is intentionally retained.
    pub async fn nuke_sessions(&self) -> rusqlite::Result<usize> {
        let mut conn = self.conn.lock().await;
        let tx = conn.transaction()?;
        let session_count: i64 =
            tx.query_row("SELECT COUNT(*) FROM sessions", [], |row| row.get(0))?;
        // Delete whole tables rather than iterating only known sessions so a
        // database created by an older version cannot retain orphan rows.
        tx.execute("DELETE FROM working_memory", [])?;
        tx.execute("DELETE FROM tool_results", [])?;
        tx.execute("DELETE FROM summary_nodes", [])?;
        tx.execute("DELETE FROM session_snapshots", [])?;
        tx.execute("DELETE FROM messages", [])?;
        tx.execute("DELETE FROM sessions", [])?;
        tx.commit()?;
        Ok(session_count.max(0) as usize)
    }

    /// Import an immutable legacy JSONL transcript as a new SQLite session.
    ///
    /// Content SHA-256 is the cross-path identity. Every observed source path
    /// is journaled as immutable: identical bytes at a new path are a no-op,
    /// while changing bytes at any journaled path is an error. Session creation,
    /// all messages, and the migration marker commit atomically.
    pub async fn import_legacy_jsonl(
        &self,
        source_path: &Path,
        session_key: &str,
    ) -> Result<LegacyImportOutcome, LegacyImportError> {
        let canonical_path = fs::canonicalize(source_path)?;
        let bytes = fs::read(&canonical_path)?;
        let content_sha256 = sha256_hex(&bytes);
        let source_path_text = canonical_path.to_string_lossy().into_owned();

        let mut conn = self.conn.lock().await;
        let tx = conn.transaction()?;
        let existing: Option<(String, String, i64)> = tx
            .query_row(
                "SELECT content_sha256, session_id, message_count \
                 FROM legacy_imports WHERE source_path = ?1",
                params![source_path_text],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .optional()?;

        if let Some((imported_sha256, session_id, message_count)) = existing {
            if imported_sha256 == content_sha256 {
                tx.commit()?;
                return Ok(LegacyImportOutcome::AlreadyImported {
                    session_id,
                    message_count: message_count as usize,
                });
            }
            return Err(LegacyImportError::ChangedFile {
                path: canonical_path,
                imported_sha256,
                current_sha256: content_sha256,
            });
        }

        // A renamed or copied legacy transcript must not create a duplicate
        // session. Record the alias path as immutable too, so later changing
        // that copy is rejected even when the original imported session has
        // since been purged.
        let identical_content: Option<(String, i64)> = tx
            .query_row(
                "SELECT session_id, message_count FROM legacy_imports \
                 WHERE content_sha256 = ?1 ORDER BY imported_at, source_path LIMIT 1",
                params![content_sha256],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?;
        if let Some((session_id, message_count)) = identical_content {
            tx.execute(
                "INSERT INTO legacy_imports \
                 (source_path, content_sha256, session_id, message_count, imported_at) \
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![
                    source_path_text,
                    content_sha256,
                    session_id,
                    message_count,
                    Utc::now().to_rfc3339(),
                ],
            )?;
            tx.commit()?;
            return Ok(LegacyImportOutcome::AlreadyImported {
                session_id,
                message_count: message_count.max(0) as usize,
            });
        }

        let text = std::str::from_utf8(&bytes).map_err(|error| {
            LegacyImportError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, error))
        })?;
        let mut messages = Vec::new();
        let mut metadata_session_key = None;
        for (index, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let value = serde_json::from_str::<Value>(line).map_err(|source| {
                LegacyImportError::InvalidJson {
                    line: index + 1,
                    source,
                }
            })?;
            if value.get("_type").and_then(Value::as_str) == Some("metadata") {
                if metadata_session_key.is_none() {
                    metadata_session_key = value
                        .get("session_key")
                        .and_then(Value::as_str)
                        .map(str::trim)
                        .filter(|key| !key.is_empty())
                        .map(str::to_string);
                }
                continue;
            }
            messages.push((index + 1, value));
        }

        // Metadata is the authoritative identity embedded by the legacy
        // writer. The caller-provided key is only a fallback for older files
        // that contain message rows alone.
        let effective_session_key = metadata_session_key
            .as_deref()
            .unwrap_or_else(|| session_key.trim());
        if effective_session_key.is_empty() {
            return Err(LegacyImportError::MissingSessionKey);
        }

        let now = Utc::now();
        let session_id = generate_session_id(&now);
        let now_text = now.to_rfc3339();
        tx.execute(
            "INSERT INTO sessions \
             (id, session_key, created_at, updated_at, message_count, metadata) \
             VALUES (?1, ?2, ?3, ?3, 0, '{}')",
            params![session_id, effective_session_key, now_text],
        )?;

        for (line, message) in &messages {
            if insert_message_locked(&tx, &session_id, message).is_none() {
                return Err(LegacyImportError::MessageInsert { line: *line });
            }
        }

        tx.execute(
            "INSERT INTO legacy_imports \
             (source_path, content_sha256, session_id, message_count, imported_at) \
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                source_path_text,
                content_sha256,
                session_id,
                messages.len() as i64,
                now_text,
            ],
        )?;
        tx.commit()?;

        Ok(LegacyImportOutcome::Imported {
            session_id,
            message_count: messages.len(),
        })
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
        let mut raw = self.get_all_messages(session_id).await;

        // The history filter still performs text-specific truncation and token
        // estimation. Give it a JSON text projection for structured content,
        // then restore the exact persisted value by stable row id before the
        // messages are replayed to the provider.
        let mut structured_content = HashMap::new();
        for message in &mut raw {
            let Some(content) = message.get("content") else {
                continue;
            };
            if content.is_string() {
                continue;
            }
            let Some(db_id) = message.get("_db_id").and_then(Value::as_i64) else {
                continue;
            };
            structured_content.insert(db_id, content.clone());
            message["content"] = Value::String(content.to_string());
        }

        let mut history = filter_history(&raw, max_messages, max_turns);
        for message in &mut history {
            let Some(db_id) = message.get("_db_id").and_then(Value::as_i64) else {
                continue;
            };
            if let Some(content) = structured_content.remove(&db_id) {
                message["content"] = content;
            }
        }
        history
    }

    /// Add a single raw JSON message to `session_id` and persist it.
    ///
    /// Extracts `role`, `content`, `tool_calls`, `tool_call_id`, `name`,
    /// `_turn`, `_synthetic`, and `timestamp` from the value. All other
    /// fields are stored in the `metadata` column as JSON.
    ///
    /// Returns the inserted rowid (the message's stable `_db_id`), or `None`
    /// if the insert failed.
    pub async fn add_message(&self, session_id: &str, msg: &Value) -> Option<i64> {
        let conn = self.conn.lock().await;
        insert_message_locked(&conn, session_id, msg)
    }

    /// Add a batch of raw JSON messages in one checked transaction.
    ///
    /// The returned row ids correspond one-for-one with `msgs`. Any failed
    /// insert aborts the transaction, which is required for assistant
    /// tool-call carriers and their result receipts: replay must observe the
    /// entire protocol group or none of it.
    pub(crate) async fn add_messages_checked(
        &self,
        session_id: &str,
        msgs: &[Value],
    ) -> anyhow::Result<Vec<i64>> {
        if msgs.is_empty() {
            return Ok(Vec::new());
        }
        let mut conn = self.conn.lock().await;
        let tx = conn.transaction()?;
        let mut row_ids = Vec::with_capacity(msgs.len());
        for msg in msgs {
            let row_id = insert_message_locked(&tx, session_id, msg).ok_or_else(|| {
                anyhow::anyhow!("failed to insert message in atomic protocol batch")
            })?;
            row_ids.push(row_id);
        }
        tx.commit()?;
        Ok(row_ids)
    }

    /// Add a batch of raw JSON messages in a single transaction.
    pub async fn add_messages(&self, session_id: &str, msgs: &[Value]) {
        if let Err(error) = self.add_messages_checked(session_id, msgs).await {
            warn!(%error, "Failed to persist atomic message batch");
        }
    }

    /// Load a complete tool output previously stored by [`store_tool_result_immutable`].
    pub async fn load_tool_result(&self, session_id: &str, tool_call_id: &str) -> Option<String> {
        let conn = self.conn.lock().await;
        conn.query_row(
            "SELECT content FROM tool_results WHERE session_id = ?1 AND tool_call_id = ?2",
            params![session_id, tool_call_id],
            |row| row.get(0),
        )
        .ok()
    }

    /// Store a tool result IMMUTABLY: never overwrite an existing
    /// `(session_id, tool_call_id)` row. Returns a [`StoredResult`] so the
    /// caller can prove the stash exists (and matches) before emitting a
    /// handle that references it. This is the durability foundation for the
    /// "handles-not-bodies" invariant — see
    /// docs/superpowers/plans/2026-07-30-tool-result-handles-not-bodies.md.
    ///
    /// Atomic in practice: the connection is held under the SessionDb Mutex
    /// across the INSERT-OR-IGNORE and the read-back, so no other writer can
    /// interleave on this connection.
    pub async fn store_tool_result_immutable(
        &self,
        session_id: &str,
        tool_call_id: &str,
        tool_name: &str,
        content: &str,
    ) -> StoredResult {
        let attempted_digest = sha256_hex(content.as_bytes());
        let mut conn = self.conn.lock().await;
        // Real transaction (BEGIN…COMMIT) so the insert+read-back is atomic
        // ACROSS connections — recall_tool_result and subagents open their own
        // SessionDb on the same file, and autocommit INSERT+SELECT would race
        // with them. The transaction's write lock is acquired by the INSERT,
        // so no other writer can interleave before COMMIT.
        let tx = match conn.transaction() {
            Ok(t) => t,
            Err(error) => {
                warn!(
                    "Failed to begin tool-result store tx session={} call={}: {}",
                    session_id, tool_call_id, error
                );
                return StoredResult::Failed;
            }
        };
        // INSERT OR IGNORE: never overwrite. rows_affected == 1 means newly
        // inserted; 0 means the key already existed.
        let inserted_rows = match tx.execute(
            "INSERT OR IGNORE INTO tool_results \
             (session_id, tool_call_id, tool_name, content, created_at) \
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![session_id, tool_call_id, tool_name, content, Utc::now().to_rfc3339()],
        ) {
            Ok(n) => n,
            Err(error) => {
                warn!(
                    "Failed to persist tool result (immutable) session={} call={}: {}",
                    session_id, tool_call_id, error
                );
                return StoredResult::Failed;
            }
        };
        // Read back what is now stored under the key (either what we just wrote
        // or the pre-existing bytes) — same transaction, no interleave.
        let stored_content: String = match tx.query_row(
            "SELECT content FROM tool_results WHERE session_id = ?1 AND tool_call_id = ?2",
            params![session_id, tool_call_id],
            |row| row.get(0),
        ) {
            Ok(c) => c,
            Err(error) => {
                warn!(
                    "Stored row vanished after insert session={} call={}: {}",
                    session_id, tool_call_id, error
                );
                return StoredResult::Failed;
            }
        };
        if let Err(error) = tx.commit() {
            warn!(
                "Failed to commit tool-result store tx session={} call={}: {}",
                session_id, tool_call_id, error
            );
            return StoredResult::Failed;
        }
        let existing_digest = sha256_hex(stored_content.as_bytes());
        if inserted_rows == 1 {
            StoredResult::Stored {
                digest: existing_digest,
            }
        } else if existing_digest == attempted_digest {
            StoredResult::Identical {
                digest: existing_digest,
            }
        } else {
            StoredResult::Conflict {
                existing_digest,
                attempted_digest,
            }
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
        let _ = self.add_message(session_id, &clear_marker).await;
    }

    /// Save or replace the latest TUI/workspace snapshot for a session key.
    pub async fn save_snapshot(&self, snapshot: &SessionSnapshot) {
        let conn = self.conn.lock().await;
        let prompt_history =
            serde_json::to_string(&snapshot.prompt_history).unwrap_or_else(|_| "[]".to_string());
        let recent_paths =
            serde_json::to_string(&snapshot.recent_paths).unwrap_or_else(|_| "[]".to_string());
        let recent_commands =
            serde_json::to_string(&snapshot.recent_commands).unwrap_or_else(|_| "[]".to_string());
        if let Err(e) = conn.execute(
            "INSERT INTO session_snapshots \
             (session_key, session_id, version, cwd, model, tui_mode, show_thinking, \
              input_draft, prompt_history, recent_paths, recent_commands, updated_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12) \
             ON CONFLICT(session_key) DO UPDATE SET \
              session_id=excluded.session_id, version=excluded.version, cwd=excluded.cwd, \
              model=excluded.model, tui_mode=excluded.tui_mode, \
              show_thinking=excluded.show_thinking, input_draft=excluded.input_draft, \
              prompt_history=excluded.prompt_history, recent_paths=excluded.recent_paths, \
              recent_commands=excluded.recent_commands, updated_at=excluded.updated_at",
            params![
                &snapshot.session_key,
                &snapshot.session_id,
                snapshot.version as i64,
                &snapshot.cwd,
                &snapshot.model,
                &snapshot.tui_mode,
                if snapshot.show_thinking { 1 } else { 0 },
                &snapshot.input_draft,
                prompt_history,
                recent_paths,
                recent_commands,
                snapshot.updated_at.to_rfc3339(),
            ],
        ) {
            warn!(
                "Failed to save session snapshot for {}: {}",
                snapshot.session_key, e
            );
        }
    }

    /// Load the latest TUI/workspace snapshot for a session key.
    pub async fn load_snapshot(&self, session_key: &str) -> Option<SessionSnapshot> {
        let conn = self.conn.lock().await;
        conn.query_row(
            "SELECT session_key, session_id, version, cwd, model, tui_mode, show_thinking, \
                    input_draft, prompt_history, recent_paths, recent_commands, updated_at \
             FROM session_snapshots WHERE session_key = ?1",
            params![session_key],
            |row| {
                let updated_at: String = row.get(11)?;
                Ok(SessionSnapshot {
                    session_key: row.get(0)?,
                    session_id: row.get(1)?,
                    version: row.get::<_, i64>(2)? as u32,
                    cwd: row.get(3)?,
                    model: row.get(4)?,
                    tui_mode: row.get(5)?,
                    show_thinking: row.get::<_, i64>(6)? != 0,
                    input_draft: row.get(7)?,
                    prompt_history: parse_json_vec(row.get::<_, String>(8)?),
                    recent_paths: parse_json_vec(row.get::<_, String>(9)?),
                    recent_commands: parse_json_vec(row.get::<_, String>(10)?),
                    updated_at: updated_at.parse().unwrap_or_else(|_| Utc::now()),
                })
            },
        )
        .ok()
    }

    // -----------------------------------------------------------------------
    // Per-session working memory
    // -----------------------------------------------------------------------

    /// Load the working-memory row for a concrete session, creating an empty
    /// active row when the session exists but has not compacted yet.
    pub async fn get_or_create_working_memory(
        &self,
        session_id: &str,
    ) -> rusqlite::Result<Option<WorkingMemoryRecord>> {
        let conn = self.conn.lock().await;
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "INSERT OR IGNORE INTO working_memory \
             (session_id, content, status, last_updated_turn, created_at, updated_at) \
             SELECT id, '', 'active', 0, ?2, ?2 FROM sessions WHERE id = ?1",
            params![session_id, now],
        )?;
        load_working_memory_locked(&conn, session_id)
    }

    /// Replace one session's complete working-memory snapshot.
    pub async fn save_working_memory(
        &self,
        session_id: &str,
        content: &str,
        status: &str,
        last_updated_turn: u64,
    ) -> rusqlite::Result<bool> {
        let conn = self.conn.lock().await;
        upsert_working_memory_locked(&conn, session_id, content, status, last_updated_turn)
    }

    /// Change only the lifecycle status for one session's working memory.
    pub async fn set_working_memory_status(
        &self,
        session_id: &str,
        status: &str,
    ) -> rusqlite::Result<bool> {
        validate_working_memory_status(status)?;
        let conn = self.conn.lock().await;
        let now = Utc::now().to_rfc3339();
        set_working_memory_status_locked(&conn, session_id, status, &now)
    }

    /// Change a set of working-memory rows in one transaction. Either every
    /// requested session advances or none do.
    pub async fn set_working_memory_status_batch(
        &self,
        session_ids: &[String],
        status: &str,
    ) -> rusqlite::Result<usize> {
        validate_working_memory_status(status)?;
        let mut conn = self.conn.lock().await;
        let tx = conn.transaction()?;
        let now = Utc::now().to_rfc3339();
        for session_id in session_ids {
            if !set_working_memory_status_locked(&tx, session_id, status, &now)? {
                return Err(rusqlite::Error::QueryReturnedNoRows);
            }
        }
        tx.commit()?;
        Ok(session_ids.len())
    }

    /// Clear the derived working snapshot while preserving its session row.
    pub async fn clear_working_memory(&self, session_id: &str) -> rusqlite::Result<bool> {
        let conn = self.conn.lock().await;
        let now = Utc::now().to_rfc3339();
        let changed = conn.execute(
            "UPDATE working_memory \
             SET content = '', status = 'active', last_updated_turn = 0, updated_at = ?2 \
             WHERE session_id = ?1",
            params![session_id, now],
        )?;
        Ok(changed > 0)
    }

    /// List working-memory rows in most-recently-updated order.
    pub async fn list_working_memory(
        &self,
        status: Option<&str>,
    ) -> rusqlite::Result<Vec<WorkingMemoryRecord>> {
        if let Some(status) = status {
            validate_working_memory_status(status)?;
        }
        let conn = self.conn.lock().await;
        let sql = if status.is_some() {
            WORKING_MEMORY_SELECT_WITH_STATUS
        } else {
            WORKING_MEMORY_SELECT_ALL
        };
        let mut stmt = conn.prepare(sql)?;
        if let Some(status) = status {
            let rows = stmt.query_map(params![status], row_to_working_memory)?;
            Ok(rows.filter_map(Result::ok).collect())
        } else {
            let rows = stmt.query_map([], row_to_working_memory)?;
            Ok(rows.filter_map(Result::ok).collect())
        }
    }

    /// Return all messages for `session_id` without any filtering, ordered by
    /// insertion order (ascending `id`). Used for export and LCM rebuild.
    pub async fn get_all_messages(&self, session_id: &str) -> Vec<Value> {
        let conn = self.conn.lock().await;
        let mut stmt = match conn.prepare(
            "SELECT id, role, content, tool_calls, tool_call_id, tool_name, \
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
            let id: i64 = row.get(0)?;
            let role: String = row.get(1)?;
            let content: SqlValue = row.get(2)?;
            let tool_calls_json: Option<String> = row.get(3)?;
            let tool_call_id: Option<String> = row.get(4)?;
            let tool_name: Option<String> = row.get(5)?;
            let turn_tag: Option<i64> = row.get(6)?;
            let synthetic: i64 = row.get(7)?;
            let timestamp: String = row.get(8)?;
            let metadata_json: String = row.get(9)?;
            Ok((
                id,
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
                    id,
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
                        id,
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

    /// Split a query string into tokens without filtering by length or stopwords.
    /// Used for interactive prefix search where single characters and common words matter.
    fn prefix_query_terms(raw: &str) -> Vec<String> {
        let mut terms: Vec<String> = raw
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .map(|t| t.to_lowercase())
            .filter(|t| !t.is_empty())
            .collect();
        terms.dedup();
        terms
    }

    /// Turn a (possibly verbose, natural-language) search string into two FTS5 MATCH
    /// expressions: an AND of the significant content keywords (precise) and an OR of
    /// the same keywords (high recall). FTS operators ("or"/"and"/"not") and common
    /// English function words are dropped so a sentence like "find the first session
    /// where I told the Diary of Two Threads story" collapses to `diary two threads`.
    fn build_recall_queries(raw: &str) -> (String, String) {
        let terms = Self::recall_keywords(raw);
        let quoted: Vec<String> = terms.iter().map(|t| format!("\"{}\"", t)).collect();
        let and_q = quoted.join(" ");
        let or_q = quoted.join(" OR ");
        (and_q, or_q)
    }

    /// Strip FTS operators and common English function words from a (possibly verbose,
    /// natural-language) query, leaving the significant content keywords. Used both by
    /// the FTS5 query builder and by `in_session` filtering so a sentence like "find the
    /// first session where I told the Diary of Two Threads story" collapses to
    /// `["diary", "two", "threads"]`.
    pub fn recall_keywords(raw: &str) -> Vec<String> {
        const STOP: &[&str] = &[
            "or",
            "and",
            "not",
            "the",
            "a",
            "an",
            "i",
            "you",
            "it",
            "to",
            "of",
            "in",
            "on",
            "for",
            "with",
            "is",
            "was",
            "were",
            "my",
            "me",
            "this",
            "that",
            "these",
            "those",
            "first",
            "time",
            "wrote",
            "written",
            "generated",
            "tell",
            "told",
            "find",
            "finds",
            "finding",
            "story",
            "stories",
            "session",
            "sessions",
            "share",
            "shared",
            "about",
            "what",
            "when",
            "where",
            "who",
            "how",
            "why",
            "which",
            "please",
            "could",
            "would",
            "can",
            "did",
            "do",
            "does",
            "get",
            "give",
            "gives",
            "make",
            "made",
            "set",
            "future",
            "past",
            "few",
            "days",
            "ago",
            "title",
            "titled",
            "called",
            "name",
            "named",
            "remember",
            "remembered",
            "recall",
            "memory",
            "context",
            "identity",
        ];
        let mut terms: Vec<String> = raw
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .map(|t| t.to_lowercase())
            .filter(|t| t.len() >= 2 && !STOP.contains(&t.as_str()))
            .collect();
        terms.dedup();
        terms
    }

    pub async fn search_messages(
        &self,
        query: &str,
        limit: usize,
        session_key_filter: Option<&str>,
    ) -> Vec<SearchResult> {
        let conn = self.conn.lock().await;
        Self::run_recall_search(
            &conn,
            query,
            limit,
            MessageSearchScope {
                session_key_prefix: session_key_filter,
                ..MessageSearchScope::default()
            },
        )
    }

    /// Search canonical past conversation messages for agent-facing retrieval.
    ///
    /// Filtering happens inside SQL before FTS ranking and `LIMIT`: only real
    /// user/assistant rows from non-active sessions may compete for a result.
    /// This prevents a short query echo, tool payload, or synthetic replay
    /// scaffold from displacing the longer source message the agent needs.
    pub async fn search_conversation_messages(
        &self,
        query: &str,
        limit: usize,
        session_key_filter: Option<&str>,
        exclude_session_id: Option<&str>,
    ) -> Vec<SearchResult> {
        let conn = self.conn.lock().await;
        Self::run_recall_search(
            &conn,
            query,
            limit,
            MessageSearchScope {
                session_key_prefix: session_key_filter,
                exclude_session_id,
                canonical_conversation_only: true,
            },
        )
    }

    /// Normalize a natural-language query once, try precise AND semantics
    /// first, then fall back to OR for recall. Scope is shared by both passes.
    fn run_recall_search(
        conn: &rusqlite::Connection,
        query: &str,
        limit: usize,
        scope: MessageSearchScope<'_>,
    ) -> Vec<SearchResult> {
        // FTS5 defaults to implicit AND between terms, so a verbose natural-language
        // query ("Diary of two threads story future session share first time I wrote
        // it...") requires every noise word to co-occur in one message and matches
        // nothing. Strip FTS operators + function words down to content keywords and
        // AND those; fall back to OR of the same keywords for maximum recall.
        let (and_q, or_q) = Self::build_recall_queries(query);
        if and_q.is_empty() {
            return Vec::new();
        }
        for match_expr in [&and_q, &or_q] {
            let results = Self::run_match(conn, match_expr, scope, limit);
            if !results.is_empty() {
                return results;
            }
        }
        Vec::new()
    }

    /// Incremental prefix search for the interactive session picker.
    ///
    /// Unlike [`search_messages`], which strips a natural-language sentence
    /// down to content keywords for agent recall, this matches what a human
    /// is typing: every token is a prefix, single characters count, and no
    /// stopword list can swallow the query.
    pub async fn search_messages_prefix(&self, query: &str, limit: usize) -> Vec<SearchResult> {
        let terms = Self::prefix_query_terms(query);
        if terms.is_empty() {
            return Vec::new();
        }
        // Build an FTS5 MATCH expression ANDing tokens as prefixes:
        // each token becomes "tok"* (double-quoted token immediately followed by asterisk).
        let quoted: Vec<String> = terms.iter().map(|t| format!("\"{}\"*", t)).collect();
        let match_expr = quoted.join(" ");
        let conn = self.conn.lock().await;
        Self::run_match(&conn, &match_expr, MessageSearchScope::default(), limit)
    }

    /// Run one FTS5 MATCH expression with SQL-level scope constraints.
    fn run_match(
        conn: &rusqlite::Connection,
        match_expr: &str,
        scope: MessageSearchScope<'_>,
        limit: usize,
    ) -> Vec<SearchResult> {
        let mut sql = "SELECT m.session_id, s.session_key, m.role,
                              CAST(m.content AS TEXT), m.timestamp,
                              snippet(messages_fts, 0, '', '', '...', 80) as snip, rank
                       FROM messages_fts
                       JOIN messages m ON m.id = messages_fts.rowid
                       JOIN sessions s ON s.id = m.session_id
                       WHERE messages_fts MATCH ?"
            .to_string();
        let mut params = vec![rusqlite::types::Value::Text(match_expr.to_string())];

        if let Some(key_prefix) = scope.session_key_prefix {
            sql.push_str(" AND s.session_key LIKE ?");
            params.push(rusqlite::types::Value::Text(format!("{key_prefix}%")));
        }
        if let Some(session_id) = scope.exclude_session_id.filter(|id| !id.is_empty()) {
            sql.push_str(" AND m.session_id != ?");
            params.push(rusqlite::types::Value::Text(session_id.to_string()));
        }
        if scope.canonical_conversation_only {
            sql.push_str(" AND m.role IN ('user', 'assistant') AND m.synthetic = 0");
        }
        sql.push_str(" ORDER BY rank LIMIT ?");
        params.push(rusqlite::types::Value::Integer(limit as i64));

        let mut stmt = match conn.prepare(&sql) {
            Ok(s) => s,
            Err(e) => {
                warn!("FTS prepare failed: {}", e);
                return Vec::new();
            }
        };
        let param_refs: Vec<&dyn rusqlite::types::ToSql> = params
            .iter()
            .map(|v| v as &dyn rusqlite::types::ToSql)
            .collect();
        stmt.query_map(param_refs.as_slice(), |row| {
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
        manifest: &SummaryManifest,
    ) {
        let conn = self.conn.lock().await;
        if let Err(e) = save_summary_node_locked(
            &conn, session_id, node_id, source_ids, child_ids, text, tokens, level, manifest,
        ) {
            warn!(
                "Failed to save summary node {} for session {}: {}",
                node_id, session_id, e
            );
        }
    }

    /// Atomically persist the durable half of an LCM checkpoint before its
    /// compacted message array is eligible for foreground installation.
    pub async fn save_compaction_checkpoint(
        &self,
        session_id: &str,
        node_id: usize,
        source_ids: &[usize],
        child_ids: &[usize],
        text: &str,
        tokens: usize,
        level: u8,
        manifest: &SummaryManifest,
        working_memory: Option<(&str, u64)>,
    ) -> rusqlite::Result<()> {
        let mut conn = self.conn.lock().await;
        let tx = conn.transaction()?;
        save_summary_node_locked(
            &tx, session_id, node_id, source_ids, child_ids, text, tokens, level, manifest,
        )?;
        if let Some((content, turn)) = working_memory {
            let updated = upsert_working_memory_locked(&tx, session_id, content, "active", turn)?;
            if !updated {
                let session_exists: bool = tx.query_row(
                    "SELECT EXISTS(SELECT 1 FROM sessions WHERE id = ?1)",
                    params![&session_id],
                    |row| row.get(0),
                )?;
                if !session_exists {
                    return Err(rusqlite::Error::QueryReturnedNoRows);
                }
            }
        }
        tx.commit()
    }

    /// Load all summary nodes for a session, ordered by ID.
    ///
    /// Returns a vec of (node_id, source_ids, child_ids, text, tokens, level,
    /// manifest, id_kind). `id_kind` is always `"db_id"`: source_ids are stable
    /// messages rowids. Pre-migration rows with positional source_ids are
    /// purged once when the DB is opened (see `SessionDb::new`); the WHERE
    /// clause keeps the invariant local.
    pub async fn load_summary_nodes(
        &self,
        session_id: &str,
    ) -> Vec<(
        usize,
        Vec<usize>,
        Vec<usize>,
        String,
        usize,
        u8,
        SummaryManifest,
        String,
    )> {
        let conn = self.conn.lock().await;
        let mut stmt = match conn.prepare(
            "SELECT id, source_ids, child_ids, text, tokens, level, manifest_json, id_kind \
             FROM summary_nodes WHERE session_id = ?1 AND id_kind = 'db_id' \
             ORDER BY id ASC",
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
            let manifest_json: Option<String> = row.get(6)?;
            let id_kind: String = row.get(7)?;
            Ok((
                id,
                source_str,
                child_str,
                text,
                tokens,
                level,
                manifest_json,
                id_kind,
            ))
        });

        match rows {
            Ok(r) => r
                .flatten()
                .map(
                    |(id, source_str, child_str, text, tokens, level, manifest_json, id_kind)| {
                        let source_ids: Vec<usize> =
                            serde_json::from_str(&source_str).unwrap_or_default();
                        let child_ids: Vec<usize> =
                            serde_json::from_str(&child_str).unwrap_or_default();
                        let manifest = manifest_json
                            .as_deref()
                            .and_then(|json| serde_json::from_str(json).ok())
                            .unwrap_or_default();
                        (
                            id as usize,
                            source_ids,
                            child_ids,
                            text,
                            tokens as usize,
                            level as u8,
                            manifest,
                            id_kind,
                        )
                    },
                )
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

    /// Return the last user/assistant exchange of the `n` most recently
    /// updated sessions, excluding `exclude_session_id` and sessions without
    /// any real (non-synthetic, non-empty) user or assistant message.
    ///
    /// Deterministic, pure SQL — ordered by `updated_at` descending.
    pub async fn latest_session_tails(
        &self,
        exclude_session_id: &str,
        n: usize,
    ) -> Vec<SessionTail> {
        const SQL: &str = "\
            SELECT * FROM ( \
                SELECT s.id, s.session_key, s.updated_at, \
                    (SELECT CAST(m.content AS TEXT) FROM messages m \
                     WHERE m.session_id = s.id AND m.role = 'user' \
                       AND m.synthetic = 0 AND m.content IS NOT NULL AND m.content != '' \
                     ORDER BY m.id DESC LIMIT 1) AS last_user, \
                    (SELECT CAST(m.content AS TEXT) FROM messages m \
                     WHERE m.session_id = s.id AND m.role = 'assistant' \
                       AND m.synthetic = 0 AND m.content IS NOT NULL AND m.content != '' \
                     ORDER BY m.id DESC LIMIT 1) AS last_assistant \
                FROM sessions s WHERE s.id != ?1 \
                ORDER BY s.updated_at DESC \
            ) WHERE last_user IS NOT NULL OR last_assistant IS NOT NULL \
            LIMIT ?2";

        let conn = self.conn.lock().await;
        let mut stmt = match conn.prepare(SQL) {
            Ok(s) => s,
            Err(e) => {
                warn!("latest_session_tails prepare failed: {}", e);
                return Vec::new();
            }
        };
        stmt.query_map(params![exclude_session_id, n as i64], |row| {
            let updated_str: String = row.get(2)?;
            Ok(SessionTail {
                session_id: row.get(0)?,
                session_key: row.get(1)?,
                updated_at: DateTime::parse_from_rfc3339(&updated_str)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now()),
                last_user: row.get::<_, Option<String>>(3)?.unwrap_or_default(),
                last_assistant: row.get::<_, Option<String>>(4)?.unwrap_or_default(),
            })
        })
        .map(|rows| rows.flatten().collect())
        .unwrap_or_default()
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

fn upsert_working_memory_locked(
    conn: &Connection,
    session_id: &str,
    content: &str,
    status: &str,
    last_updated_turn: u64,
) -> rusqlite::Result<bool> {
    validate_working_memory_status(status)?;
    let now = Utc::now().to_rfc3339();
    let changed = conn.execute(
        "INSERT INTO working_memory \
         (session_id, content, status, last_updated_turn, created_at, updated_at) \
         SELECT id, ?2, ?3, ?4, ?5, ?5 FROM sessions WHERE id = ?1 \
         ON CONFLICT(session_id) DO UPDATE SET \
           content=excluded.content, status=excluded.status, \
           last_updated_turn=excluded.last_updated_turn, updated_at=excluded.updated_at \
         WHERE excluded.last_updated_turn >= working_memory.last_updated_turn \
           AND CASE excluded.status \
                 WHEN 'active' THEN 0 WHEN 'completed' THEN 1 ELSE 2 END \
               >= CASE working_memory.status \
                 WHEN 'active' THEN 0 WHEN 'completed' THEN 1 ELSE 2 END",
        params![session_id, content, status, last_updated_turn as i64, now],
    )?;
    Ok(changed > 0)
}

fn set_working_memory_status_locked(
    conn: &Connection,
    session_id: &str,
    status: &str,
    now: &str,
) -> rusqlite::Result<bool> {
    let changed = conn.execute(
        "UPDATE working_memory SET status = ?2, updated_at = ?3 \
         WHERE session_id = ?1 \
           AND (status = ?2 \
                OR (status = 'active' AND ?2 = 'completed') \
                OR (status = 'completed' AND ?2 = 'reflected'))",
        params![session_id, status, now],
    )?;
    Ok(changed > 0)
}

/// Explicit resume is the sole backwards lifecycle transition. It preserves the
/// snapshot and turn while preventing a stale reflector batch from advancing a
/// newly-active row directly to `reflected`.
fn reactivate_working_memory_locked(
    conn: &Connection,
    session_id: &str,
    now: &str,
) -> rusqlite::Result<bool> {
    let changed = conn.execute(
        "UPDATE working_memory SET status = 'active', updated_at = ?2 \
         WHERE session_id = ?1 AND status != 'active'",
        params![session_id, now],
    )?;
    Ok(changed > 0)
}

fn save_summary_node_locked(
    conn: &Connection,
    session_id: &str,
    node_id: usize,
    source_ids: &[usize],
    child_ids: &[usize],
    text: &str,
    tokens: usize,
    level: u8,
    manifest: &SummaryManifest,
) -> rusqlite::Result<()> {
    let source_json = serde_json::to_string(source_ids).unwrap_or_else(|_| "[]".to_string());
    let child_json = serde_json::to_string(child_ids).unwrap_or_else(|_| "[]".to_string());
    let manifest_json = serde_json::to_string(manifest).unwrap_or_else(|_| "{}".to_string());
    let now = Utc::now().to_rfc3339();
    // `db_id` means source ids are stable messages rowids. Older positional
    // nodes lack the marker and are deliberately skipped during reconstruction.
    conn.execute(
        "INSERT INTO summary_nodes \
         (id, session_id, source_ids, child_ids, text, tokens, level, created_at, id_kind, \
          manifest_json) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 'db_id', ?9) \
         ON CONFLICT(session_id, id) DO UPDATE SET \
         source_ids = excluded.source_ids, child_ids = excluded.child_ids, \
         text = excluded.text, tokens = excluded.tokens, level = excluded.level, \
         created_at = excluded.created_at, id_kind = excluded.id_kind, \
         manifest_json = excluded.manifest_json",
        params![
            node_id as i64,
            session_id,
            source_json,
            child_json,
            text,
            tokens as i64,
            level as i64,
            now,
            manifest_json,
        ],
    )?;
    Ok(())
}

fn load_working_memory_locked(
    conn: &Connection,
    session_id: &str,
) -> rusqlite::Result<Option<WorkingMemoryRecord>> {
    let sql = format!("{WORKING_MEMORY_COLUMNS} WHERE wm.session_id = ?1");
    conn.query_row(&sql, params![session_id], row_to_working_memory)
        .optional()
}

fn row_to_working_memory(row: &rusqlite::Row<'_>) -> rusqlite::Result<WorkingMemoryRecord> {
    let created_at: String = row.get(5)?;
    let updated_at: String = row.get(6)?;
    Ok(WorkingMemoryRecord {
        session_id: row.get(0)?,
        session_key: row.get(1)?,
        content: row.get(2)?,
        status: row.get(3)?,
        last_updated_turn: row.get::<_, i64>(4)?.max(0) as u64,
        created_at: parse_datetime(&created_at),
        updated_at: parse_datetime(&updated_at),
    })
}

fn validate_working_memory_status(status: &str) -> rusqlite::Result<()> {
    if matches!(status, "active" | "completed" | "reflected") {
        Ok(())
    } else {
        Err(rusqlite::Error::InvalidParameterName(format!(
            "invalid working-memory status: {status}"
        )))
    }
}

fn parse_datetime(value: &str) -> DateTime<Utc> {
    DateTime::parse_from_rfc3339(value)
        .map(|date| date.with_timezone(&Utc))
        .unwrap_or_else(|_| Utc::now())
}

fn select_session_ids(
    tx: &Transaction<'_>,
    sql: &str,
    parameter: Option<String>,
) -> rusqlite::Result<Vec<String>> {
    let mut stmt = tx.prepare(sql)?;
    if let Some(parameter) = parameter {
        let rows = stmt.query_map(params![parameter], |row| row.get(0))?;
        rows.collect()
    } else {
        let rows = stmt.query_map([], |row| row.get(0))?;
        rows.collect()
    }
}

/// Remove all rows owned by `session_id`. Keep this explicit rather than
/// relying solely on `ON DELETE CASCADE`: older databases were created without
/// cascade clauses, and SQLite cannot retrofit them with `ALTER TABLE`.
fn delete_session_rows(tx: &Transaction<'_>, session_id: &str) -> rusqlite::Result<bool> {
    tx.execute(
        "DELETE FROM working_memory WHERE session_id = ?1",
        params![session_id],
    )?;
    tx.execute(
        "DELETE FROM tool_results WHERE session_id = ?1",
        params![session_id],
    )?;
    tx.execute(
        "DELETE FROM summary_nodes WHERE session_id = ?1",
        params![session_id],
    )?;
    tx.execute(
        "DELETE FROM session_snapshots WHERE session_id = ?1",
        params![session_id],
    )?;
    tx.execute(
        "DELETE FROM messages WHERE session_id = ?1",
        params![session_id],
    )?;
    let deleted = tx.execute("DELETE FROM sessions WHERE id = ?1", params![session_id])?;
    Ok(deleted > 0)
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

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

fn parse_json_vec(json: String) -> Vec<String> {
    serde_json::from_str(&json).unwrap_or_default()
}

/// Insert a single message into the DB using an already-locked connection.
///
/// Called from both `add_message()` (which locks externally) and the batch
/// path in `add_messages()` (which holds the lock for the whole batch).
///
/// Returns the inserted rowid, or `None` on failure.
fn insert_message_locked(conn: &Connection, session_id: &str, msg: &Value) -> Option<i64> {
    let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("user");

    // SQLite's dynamic value types provide an unambiguous tag without a
    // schema migration: ordinary strings remain TEXT for search/readability,
    // while every other explicit JSON value is serialized into a BLOB. This
    // distinguishes JSON null from a missing content field and prevents media
    // arrays/objects from being flattened into a string during replay.
    let content = match msg.get("content") {
        Some(Value::String(text)) => SqlValue::Text(text.clone()),
        Some(value) => match serde_json::to_vec(value) {
            Ok(encoded) => SqlValue::Blob(encoded),
            Err(e) => {
                warn!(
                    "Failed to encode message content for session {}: {}",
                    session_id, e
                );
                return None;
            }
        },
        None => SqlValue::Null,
    };

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
    // `_db_id` is excluded: it is the rowid assigned by THIS insert — a stale
    // copy from a reloaded message must never be persisted as metadata.
    let reserved = [
        "role",
        "content",
        "tool_calls",
        "tool_call_id",
        "name",
        "_turn",
        "_synthetic",
        "_db_id",
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
        return None;
    }
    let row_id = conn.last_insert_rowid();

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
    Some(row_id)
}

/// Reconstruct a `serde_json::Value` from the columns stored in the `messages`
/// table. This is the inverse of the field extraction done in
/// `insert_message_locked()`.
fn reconstruct_message(
    id: i64,
    role: String,
    content: SqlValue,
    tool_calls_json: Option<String>,
    tool_call_id: Option<String>,
    tool_name: Option<String>,
    turn_tag: Option<i64>,
    synthetic: i64,
    timestamp: String,
    metadata_json: String,
) -> Value {
    // `_db_id` is the stable rowid — the LCM engine's MessageId. It is an
    // internal-only field: filter_history preserves it, the protocol render
    // drops it before the wire call.
    // Rows written before structured-content support contain TEXT or NULL.
    // New non-string values are JSON BLOBs, so mixed old/new databases replay
    // without a migration while preserving the original JSON type exactly.
    let content = match content {
        SqlValue::Null => Value::String(String::new()),
        SqlValue::Text(text) => Value::String(text),
        SqlValue::Blob(encoded) => serde_json::from_slice(&encoded)
            .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&encoded).into_owned())),
        SqlValue::Integer(value) => json!(value),
        SqlValue::Real(value) => json!(value),
    };

    let mut msg = json!({
        "role": role,
        "content": content,
        "timestamp": timestamp,
        "_db_id": id,
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
    use crate::agent::lcm::ManifestItem;
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

    #[test]
    fn test_build_recall_queries_strips_noise() {
        // A verbose natural-language query collapses to its content keywords so
        // FTS5 (implicit AND) actually matches the story instead of requiring
        // every noise word to co-occur.
        let (and_q, or_q) = SessionDb::build_recall_queries(
            "find the first session where I told the Diary of Two Threads story",
        );
        assert_eq!(and_q, "\"diary\" \"two\" \"threads\"");
        assert_eq!(or_q, "\"diary\" OR \"two\" OR \"threads\"");

        // FTS operators written as prose must not break the query.
        let (and_q, _) = SessionDb::build_recall_queries(
            "Diary of two threads story future session share first time I wrote it or generated it for you.",
        );
        assert_eq!(and_q, "\"diary\" \"two\" \"threads\"");

        // A precise phrase query is preserved as keywords.
        let (and_q, _) = SessionDb::build_recall_queries("\"Diary of two threads\"");
        assert_eq!(and_q, "\"diary\" \"two\" \"threads\"");
    }

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
    async fn test_get_or_resume_with_idle_creates_fresh_when_stale() {
        let (db, _dir) = make_db();

        // Create a session and add a message so it looks real.
        let original = db.create_session("telegram:42").await;
        db.add_messages(
            &original.id,
            &[serde_json::json!({"role": "user", "content": "old message"})],
        )
        .await;

        // Backdate updated_at to 2 hours ago.
        {
            let old_time = (chrono::Utc::now() - chrono::Duration::hours(2)).to_rfc3339();
            let conn = db.conn.lock().await;
            conn.execute(
                "UPDATE sessions SET updated_at = ?1 WHERE id = ?2",
                rusqlite::params![old_time, original.id],
            )
            .unwrap();
        }

        // With max_idle_secs=3600 (1 hour), the 2-hour-old session is stale.
        let fresh = db.get_or_resume_with_idle("telegram:42", 3600).await;
        assert_ne!(
            fresh.id, original.id,
            "stale session should not be resumed; a new one should be created"
        );
        assert_eq!(fresh.message_count, 0);

        // Without idle timeout, the old session is still resumed.
        // (get_latest_session returns the newest by updated_at, which is the
        // fresh one we just created — so this confirms both exist.)
        let sessions = db.list_sessions(Some("telegram:42"), 10).await;
        assert_eq!(sessions.len(), 2, "both old and new sessions should exist");
    }

    #[tokio::test]
    async fn test_get_or_resume_with_idle_resumes_when_recent() {
        let (db, _dir) = make_db();

        let original = db.create_session("telegram:99").await;

        // Session was just created (updated_at is now). With 3600s idle timeout
        // it should be resumed, not replaced.
        let resumed = db.get_or_resume_with_idle("telegram:99", 3600).await;
        assert_eq!(
            resumed.id, original.id,
            "recent session must be resumed, not replaced"
        );
    }

    #[tokio::test]
    async fn idle_rollover_completes_the_expired_working_memory() {
        let (db, _dir) = make_db();
        let original = db.create_session("telegram:idle-memory").await;
        db.save_working_memory(&original.id, "durable session summary", "active", 7)
            .await
            .unwrap();
        {
            let old_time = (Utc::now() - chrono::Duration::hours(2)).to_rfc3339();
            let conn = db.conn.lock().await;
            conn.execute(
                "UPDATE sessions SET updated_at = ?1 WHERE id = ?2",
                params![old_time, original.id],
            )
            .unwrap();
        }

        let fresh = db
            .get_or_resume_with_idle("telegram:idle-memory", 3600)
            .await;
        assert_ne!(fresh.id, original.id);
        let expired = db
            .get_or_create_working_memory(&original.id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(expired.status, "completed");
        assert_eq!(expired.content, "durable session summary");
        assert_eq!(expired.last_updated_turn, 7);
    }

    #[tokio::test]
    async fn explicit_resume_selects_exact_id_and_reopens_working_memory() {
        let (db, _dir) = make_db();
        let requested = db.create_session("cli:shared-name").await;
        db.save_working_memory(&requested.id, "old snapshot", "active", 5)
            .await
            .unwrap();
        assert!(db
            .set_working_memory_status(&requested.id, "completed")
            .await
            .unwrap());
        assert!(db
            .set_working_memory_status(&requested.id, "reflected")
            .await
            .unwrap());
        let newer = db.create_session("cli:shared-name").await;
        assert_eq!(
            db.get_latest_session("cli:shared-name").await.unwrap().id,
            newer.id
        );

        let resumed = db
            .resume_session(&requested.id)
            .await
            .unwrap()
            .expect("requested concrete session exists");
        assert_eq!(resumed.id, requested.id);
        assert_eq!(
            db.get_latest_session("cli:shared-name").await.unwrap().id,
            requested.id,
            "the requested ID, not merely the latest reusable key, must resume"
        );
        let memory = db
            .get_or_create_working_memory(&requested.id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(memory.status, "active");
        assert_eq!(memory.content, "old snapshot");
        assert_eq!(memory.last_updated_turn, 5);
        assert!(db
            .save_working_memory(&requested.id, "continued snapshot", "active", 6)
            .await
            .unwrap());
        assert!(
            !db.set_working_memory_status(&requested.id, "reflected")
                .await
                .unwrap(),
            "a stale reflector batch must not advance a reactivated row"
        );
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

        let _ = db
            .add_message(&meta.id, &json!({"role": "user", "content": "hello"}))
            .await;
        let _ = db
            .add_message(
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
    async fn test_structured_message_content_round_trips_losslessly() {
        let (db, _dir) = make_db();
        let meta = db.create_session("cli:structured-content").await;
        let media_content = json!([
            {"type": "text", "text": "Describe this image and recording."},
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64,iVBORw0KGgo=",
                    "detail": "high"
                }
            },
            {
                "type": "input_audio",
                "input_audio": {"data": "UklGRg==", "format": "wav"}
            }
        ]);
        let object_content = json!({"type": "text", "text": "object payload"});
        let messages = vec![
            json!({"role": "user", "content": "plain text"}),
            json!({"role": "assistant", "content": null}),
            json!({"role": "user", "content": media_content.clone()}),
            json!({"role": "assistant", "content": object_content.clone()}),
        ];

        db.add_messages(&meta.id, &messages).await;

        let replayed = db.get_all_messages(&meta.id).await;
        assert_eq!(replayed.len(), messages.len());
        assert_eq!(replayed[0]["content"], "plain text");
        assert_eq!(replayed[1]["content"], Value::Null);
        assert_eq!(replayed[2]["content"], media_content);
        assert_eq!(replayed[3]["content"], object_content);

        // The filtered replay path used by foreground inference must retain
        // the same multimodal array rather than converting it to JSON text.
        let history = db.get_history(&meta.id, 100, 0).await;
        assert_eq!(history[2]["content"], replayed[2]["content"]);

        // String-only convenience APIs expose structured content as JSON text
        // rather than dropping their rows because SQLite stored it as a BLOB.
        let search = db.search_messages("Describe", 10, None).await;
        assert_eq!(search.len(), 1);
        assert_eq!(
            serde_json::from_str::<Value>(&search[0].content).unwrap(),
            replayed[2]["content"]
        );
        let tails = db.latest_session_tails("different-session", 1).await;
        assert_eq!(tails.len(), 1);
        assert_eq!(
            serde_json::from_str::<Value>(&tails[0].last_user).unwrap(),
            replayed[2]["content"]
        );
        assert_eq!(
            serde_json::from_str::<Value>(&tails[0].last_assistant).unwrap(),
            replayed[3]["content"]
        );
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
    async fn checked_message_batch_rolls_back_when_middle_insert_fails() {
        let (db, _dir) = make_db();
        let meta = db.create_session("cli:atomic-batch-failure").await;
        {
            let conn = db.conn.lock().await;
            conn.execute_batch(
                "CREATE TRIGGER fail_rejected_protocol_receipt
                 BEFORE INSERT ON messages
                 WHEN NEW.content = 'forced failure'
                 BEGIN
                     SELECT RAISE(ABORT, 'forced middle insert failure');
                 END;",
            )
            .unwrap();
        }

        let messages = vec![
            json!({"role": "assistant", "content": "carrier"}),
            json!({"role": "tool", "content": "forced failure", "tool_call_id": "tc_fail"}),
            json!({"role": "tool", "content": "receipt", "tool_call_id": "tc_after"}),
        ];

        let result = db.add_messages_checked(&meta.id, &messages).await;

        assert!(result.is_err());
        assert!(
            db.get_all_messages(&meta.id).await.is_empty(),
            "a failed protocol group must leave no partial carrier or receipts"
        );
    }

    #[tokio::test]
    async fn oversized_tool_results_survive_database_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("sessions.db");
        let session_id = {
            let db = SessionDb::new(&db_path);
            let session = db.create_session("cli:durable-tool-result").await;
            assert!(matches!(
                db.store_tool_result_immutable(&session.id, "call_42", "read_file", "full exact body")
                    .await,
                StoredResult::Stored { .. }
            ));
            session.id
        };

        let reopened = SessionDb::new(&db_path);
        assert_eq!(
            reopened.load_tool_result(&session_id, "call_42").await,
            Some("full exact body".to_string())
        );
        assert_eq!(reopened.path(), db_path.as_path());
    }

    #[tokio::test]
    async fn immutable_store_returns_stored_then_identical_then_conflict() {
        let (db, _dir) = make_db();
        let session = db.create_session("cli:immutable").await;

        // First write: newly stored, with a real digest.
        let first = db
            .store_tool_result_immutable(&session.id, "call_x", "exec", "body bytes v1")
            .await;
        let digest = match first {
            StoredResult::Stored { digest } => digest,
            other => panic!("first store must be Stored, got {other:?}"),
        };
        assert!(!digest.is_empty());

        // Second write with IDENTICAL bytes: idempotent — Identical, same digest.
        let again = db
            .store_tool_result_immutable(&session.id, "call_x", "exec", "body bytes v1")
            .await;
        match again {
            StoredResult::Identical { digest: d } => assert_eq!(d, digest),
            other => panic!("identical retry must be Identical, got {other:?}"),
        }

        // The body actually stored is still the ORIGINAL — never overwritten.
        assert_eq!(
            db.load_tool_result(&session.id, "call_x").await,
            Some("body bytes v1".to_string()),
            "immutable store must never overwrite"
        );

        // Third write with DIFFERENT bytes under the same key: Conflict.
        let conflict = db
            .store_tool_result_immutable(&session.id, "call_x", "exec", "body bytes v2")
            .await;
        match conflict {
            StoredResult::Conflict {
                existing_digest,
                attempted_digest,
            } => {
                assert_eq!(existing_digest, digest, "existing must be the original v1");
                assert_ne!(attempted_digest, digest, "attempted must be the v2 digest");
            }
            other => panic!("different bytes must Conflict, got {other:?}"),
        }

        // Still never overwritten — the conflicting write was rejected.
        assert_eq!(
            db.load_tool_result(&session.id, "call_x").await,
            Some("body bytes v1".to_string()),
            "a conflicting write must not replace the stored body"
        );
    }

    #[tokio::test]
    async fn predictable_tool_call_ids_are_isolated_by_session() {
        let (db, _dir) = make_db();
        let first = db.create_session("cli:first-tool-session").await;
        let second = db.create_session("cli:second-tool-session").await;

        assert!(matches!(
            db.store_tool_result_immutable(&first.id, "call_0", "read_file", "first body")
                .await,
            StoredResult::Stored { .. }
        ));
        assert!(matches!(
            db.store_tool_result_immutable(&second.id, "call_0", "read_file", "second body")
                .await,
            StoredResult::Stored { .. }
        ));

        assert_eq!(
            db.load_tool_result(&first.id, "call_0").await.as_deref(),
            Some("first body")
        );
        assert_eq!(
            db.load_tool_result(&second.id, "call_0").await.as_deref(),
            Some("second body")
        );
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

        let _ = db
            .add_message(
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

        let _ = db
            .add_message(
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
        let _ = db
            .add_message(&second.id, &json!({"role": "user", "content": "bump"}))
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
        let _ = db
            .add_message(
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

        let _ = db
            .add_message(&meta.id, &json!({"role": "user", "content": "one"}))
            .await;
        let _ = db
            .add_message(&meta.id, &json!({"role": "assistant", "content": "two"}))
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
        let _ = db
            .add_message(&meta.id, &json!({"role": "user", "content": "Hello world"}))
            .await;
        let results = db.search_messages("xyznonexistent", 10, None).await;
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_fts_search_key_filter() {
        let (db, _dir) = make_db();
        let cli = db.create_session("cli:default").await;
        let tg = db.create_session("telegram:42").await;
        let _ = db
            .add_message(
                &cli.id,
                &json!({"role": "user", "content": "CLI Rust question"}),
            )
            .await;
        let _ = db
            .add_message(
                &tg.id,
                &json!({"role": "user", "content": "Telegram Rust question"}),
            )
            .await;
        let results = db.search_messages("Rust", 10, Some("cli:")).await;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].session_key, "cli:default");
    }

    #[tokio::test]
    async fn test_conversation_search_filters_noise_before_rank_limit() {
        let (db, _dir) = make_db();
        let past = db.create_session("cli:past").await;
        let current = db.create_session("cli:current").await;

        db.add_messages(
            &past.id,
            &[
                json!({
                    "role": "user",
                    "content": "Canonical source explains zephyrattestation with the complete design."
                }),
                json!({
                    "role": "user",
                    "content": "zephyrattestation",
                    "_synthetic": true
                }),
                json!({
                    "role": "tool",
                    "content": "zephyrattestation"
                }),
            ],
        )
        .await;
        db.add_messages(
            &current.id,
            &[
                json!({"role": "user", "content": "zephyrattestation"}),
                json!({"role": "tool", "content": "zephyrattestation"}),
            ],
        )
        .await;

        let results = db
            .search_conversation_messages("zephyrattestation", 1, Some("cli:"), Some(&current.id))
            .await;

        assert_eq!(results.len(), 1, "filters must run before LIMIT");
        assert_eq!(results[0].session_id, past.id);
        assert_eq!(results[0].role, "user");
        assert!(
            results[0].content.contains("Canonical source explains"),
            "short synthetic/tool/active echoes must not outrank the source: {:?}",
            results
        );
    }

    #[tokio::test]
    async fn test_fts_rebuild() {
        let (db, _dir) = make_db();
        let meta = db.create_session("cli:rebuild").await;
        let _ = db
            .add_message(
                &meta.id,
                &json!({"role": "user", "content": "Rebuild test message"}),
            )
            .await;
        db.rebuild_fts_index().await;
        let results = db.search_messages("Rebuild", 10, None).await;
        assert!(!results.is_empty());
    }

    #[test]
    fn test_prefix_query_terms_keeps_short_and_stopword_tokens() {
        // Prefix search keeps single characters and stopword tokens,
        // deliberately differing from recall_keywords.
        let terms = SessionDb::prefix_query_terms("a session");
        assert_eq!(terms, vec!["a", "session"]);

        // In contrast, recall_keywords strips both the single "a" (length < 2)
        // and "session" (it's a stopword), leaving nothing.
        let recall_terms = SessionDb::recall_keywords("a session");
        assert!(recall_terms.is_empty());
    }

    #[tokio::test]
    async fn test_search_messages_prefix_matches_partial_word() {
        let (db, _dir) = make_db();
        let meta = db.create_session("cli:prefix").await;
        db.add_messages(
            &meta.id,
            &[json!({"role": "user", "content": "This is a session about debugging"})],
        )
        .await;

        // Prefix search should match "sess" as a prefix of "session",
        // even though "session" is a stopword that recall_keywords filters out.
        let prefix_results = db.search_messages_prefix("sess", 10).await;
        assert!(
            !prefix_results.is_empty(),
            "prefix search for 'sess' should match 'session'"
        );

        // The existing search_messages should return empty because "session" is filtered.
        let recall_results = db.search_messages("sess", 10, None).await;
        assert!(
            recall_results.is_empty(),
            "recall search for 'sess' should not match (stopword)"
        );
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

    #[tokio::test]
    async fn test_session_snapshot_roundtrip_and_replace() {
        let (db, _dir) = make_db();
        let meta = db.create_session("cli:snapshot").await;
        let mut snapshot = SessionSnapshot {
            version: 1,
            session_key: meta.session_key.clone(),
            session_id: meta.id.clone(),
            cwd: "/tmp/project".to_string(),
            model: "model-a".to_string(),
            tui_mode: "inspect".to_string(),
            show_thinking: false,
            input_draft: "draft".to_string(),
            prompt_history: vec!["one".to_string(), "two".to_string()],
            recent_paths: vec!["src/main.rs".to_string()],
            recent_commands: vec!["/jobs".to_string()],
            updated_at: Utc::now(),
        };

        db.save_snapshot(&snapshot).await;
        let loaded = db.load_snapshot("cli:snapshot").await.unwrap();
        assert_eq!(loaded.session_key, snapshot.session_key);
        assert_eq!(loaded.prompt_history, snapshot.prompt_history);
        assert_eq!(loaded.recent_paths, snapshot.recent_paths);
        assert!(!loaded.show_thinking);

        snapshot.model = "model-b".to_string();
        snapshot.input_draft = "new draft".to_string();
        db.save_snapshot(&snapshot).await;
        let replaced = db.load_snapshot("cli:snapshot").await.unwrap();
        assert_eq!(replaced.model, "model-b");
        assert_eq!(replaced.input_draft, "new draft");
    }

    // -----------------------------------------------------------------------
    // Summary DAG persistence
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_dag_persistence_roundtrip() {
        let (db, _dir) = make_db();
        let meta = db.create_session("cli:dag_test").await;
        let manifest = SummaryManifest {
            open_loops: vec![ManifestItem {
                text: "Verify restart behavior".to_string(),
                sources: vec![2, 4],
            }],
            ..SummaryManifest::default()
        };

        // Save two summary nodes.
        db.save_summary_node(
            &meta.id,
            0,
            &[1, 2, 3, 4],
            &[],
            "Summary of greeting exchange.",
            15,
            1,
            &manifest,
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
            &SummaryManifest::default(),
        )
        .await;

        // Load them back.
        let nodes = db.load_summary_nodes(&meta.id).await;
        assert_eq!(nodes.len(), 2);

        let (id0, src0, child0, text0, tokens0, level0, manifest0, kind0) = &nodes[0];
        assert_eq!(*id0, 0);
        assert_eq!(*src0, vec![1, 2, 3, 4]);
        assert!(child0.is_empty());
        assert_eq!(text0, "Summary of greeting exchange.");
        assert_eq!(*tokens0, 15);
        assert_eq!(*level0, 1);
        assert_eq!(manifest0, &manifest);
        assert_eq!(kind0, "db_id", "save_summary_node must tag id_kind");

        let (id1, src1, _child1, text1, tokens1, level1, manifest1, _kind1) = &nodes[1];
        assert_eq!(*id1, 1);
        assert_eq!(*src1, vec![5, 6, 7]);
        assert_eq!(text1, "Summary of technical discussion.");
        assert_eq!(*tokens1, 12);
        assert_eq!(*level1, 2);
        assert_eq!(manifest1, &SummaryManifest::default());
    }

    /// Pre-migration summary_nodes rows (id_kind NULL or non-"db_id") carry
    /// POSITIONAL source_ids that cannot be resolved against the db-id-keyed
    /// LCM store. Opening the DB purges them once; db_id rows survive.
    #[tokio::test]
    async fn test_open_purges_legacy_summary_nodes() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("sessions.db");
        let meta = {
            let db = SessionDb::new(&db_path);
            let meta = db.create_session("cli:legacy_purge").await;
            db.save_summary_node(
                &meta.id,
                0,
                &[1, 2],
                &[],
                "modern",
                3,
                1,
                &SummaryManifest::default(),
            )
            .await;
            // Mimic a pre-migration DB: raw rows with NULL and stale id_kind.
            let conn = db.conn.lock().await;
            conn.execute(
                "INSERT INTO summary_nodes \
                 (id, session_id, source_ids, child_ids, text, tokens, level, created_at) \
                 VALUES (98, ?1, '[0,1]', '[]', 'null-kind positional', 4, 1, \
                 '2025-01-01T00:00:00Z')",
                rusqlite::params![meta.id],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO summary_nodes \
                 (id, session_id, source_ids, child_ids, text, tokens, level, created_at, \
                 id_kind) \
                 VALUES (99, ?1, '[0,1]', '[]', 'legacy positional', 4, 1, \
                 '2025-01-01T00:00:00Z', 'legacy')",
                rusqlite::params![meta.id],
            )
            .unwrap();
            meta
        };

        // Re-open: the migration path purges the legacy rows exactly once.
        let db = SessionDb::new(&db_path);
        let conn = db.conn.lock().await;
        let remaining: Vec<(i64, Option<String>)> = conn
            .prepare("SELECT id, id_kind FROM summary_nodes WHERE session_id = ?1")
            .unwrap()
            .query_map(rusqlite::params![meta.id], |row| {
                Ok((row.get(0)?, row.get(1)?))
            })
            .unwrap()
            .flatten()
            .collect();
        assert_eq!(
            remaining,
            vec![(0, Some("db_id".to_string()))],
            "legacy rows must be purged at open; db_id rows must survive"
        );
    }

    #[tokio::test]
    async fn summary_node_ids_are_isolated_per_session() {
        let (db, _dir) = make_db();
        let first = db.create_session("cli:dag-first").await;
        let second = db.create_session("cli:dag-second").await;

        db.save_summary_node(
            &first.id,
            0,
            &[11],
            &[],
            "first",
            1,
            1,
            &SummaryManifest::default(),
        )
        .await;
        db.save_summary_node(
            &second.id,
            0,
            &[22],
            &[],
            "second",
            1,
            1,
            &SummaryManifest::default(),
        )
        .await;

        let first_nodes = db.load_summary_nodes(&first.id).await;
        let second_nodes = db.load_summary_nodes(&second.id).await;
        assert_eq!(first_nodes.len(), 1);
        assert_eq!(second_nodes.len(), 1);
        assert_eq!(first_nodes[0].3, "first");
        assert_eq!(second_nodes[0].3, "second");
    }

    // -----------------------------------------------------------------------
    // latest_session_tails
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_latest_session_tails_empty_db() {
        let (db, _dir) = make_db();
        let tails = db.latest_session_tails("nonexistent", 3).await;
        assert!(tails.is_empty(), "empty db must yield no tails");
    }

    #[tokio::test]
    async fn test_latest_session_tails_picks_most_recent() {
        let (db, _dir) = make_db();

        let older = db.create_session("cli:oneshot-1").await;
        db.add_messages(
            &older.id,
            &[
                json!({"role": "user", "content": "old question"}),
                json!({"role": "assistant", "content": "old answer"}),
            ],
        )
        .await;

        let newer = db.create_session("cli:oneshot-2").await;
        db.add_messages(
            &newer.id,
            &[
                json!({"role": "user", "content": "new question"}),
                json!({"role": "assistant", "content": "new answer"}),
            ],
        )
        .await;

        let tails = db.latest_session_tails("some-other-id", 1).await;
        assert_eq!(tails.len(), 1);
        assert_eq!(tails[0].session_key, "cli:oneshot-2");
        assert_eq!(tails[0].last_user, "new question");
        assert_eq!(tails[0].last_assistant, "new answer");

        // With n=2, both come back, most recent first.
        let tails = db.latest_session_tails("some-other-id", 2).await;
        assert_eq!(tails.len(), 2);
        assert_eq!(tails[0].session_key, "cli:oneshot-2");
        assert_eq!(tails[1].session_key, "cli:oneshot-1");
    }

    #[tokio::test]
    async fn test_latest_session_tails_excludes_current_session() {
        let (db, _dir) = make_db();

        let prior = db.create_session("cli:oneshot-prior").await;
        db.add_messages(
            &prior.id,
            &[
                json!({"role": "user", "content": "prior question"}),
                json!({"role": "assistant", "content": "prior answer"}),
            ],
        )
        .await;

        let current = db.create_session("cli:oneshot-current").await;
        let _ = db
            .add_message(&current.id, &json!({"role": "user", "content": "hello"}))
            .await;

        let tails = db.latest_session_tails(&current.id, 3).await;
        assert_eq!(tails.len(), 1, "current session must be excluded");
        assert_eq!(tails[0].session_key, "cli:oneshot-prior");
    }

    #[tokio::test]
    async fn test_latest_session_tails_skips_empty_sessions() {
        let (db, _dir) = make_db();

        // A session with content, then a newer session with no messages at all
        // (e.g. created but crashed before the first exchange).
        let real = db.create_session("cli:real").await;
        db.add_messages(
            &real.id,
            &[
                json!({"role": "user", "content": "real question"}),
                json!({"role": "assistant", "content": "real answer"}),
            ],
        )
        .await;
        let _empty = db.create_session("cli:empty").await;

        let tails = db.latest_session_tails("other", 3).await;
        assert_eq!(tails.len(), 1, "message-less sessions must be skipped");
        assert_eq!(tails[0].session_key, "cli:real");
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
        db.save_summary_node(
            &meta.id,
            0,
            &[1, 2],
            &[],
            "Original.",
            5,
            1,
            &SummaryManifest::default(),
        )
        .await;
        db.save_summary_node(
            &meta.id,
            0,
            &[1, 2],
            &[],
            "Updated.",
            6,
            1,
            &SummaryManifest::default(),
        )
        .await;

        let nodes = db.load_summary_nodes(&meta.id).await;
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].3, "Updated.");
        assert_eq!(nodes[0].4, 6);
    }

    #[tokio::test]
    async fn working_memory_is_isolated_by_concrete_session_id() {
        let (db, _dir) = make_db();
        let first = db.create_session("telegram:shared-chat").await;
        let second = db.create_session("telegram:shared-chat").await;

        assert!(db
            .save_working_memory(&first.id, "first session state", "completed", 7)
            .await
            .unwrap());
        assert!(db
            .save_working_memory(&second.id, "second session state", "active", 2)
            .await
            .unwrap());

        let first_memory = db
            .get_or_create_working_memory(&first.id)
            .await
            .unwrap()
            .unwrap();
        let second_memory = db
            .get_or_create_working_memory(&second.id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(first_memory.content, "first session state");
        assert_eq!(first_memory.status, "completed");
        assert_eq!(first_memory.last_updated_turn, 7);
        assert_eq!(second_memory.content, "second session state");
        assert_eq!(second_memory.status, "active");
        assert_ne!(first_memory.session_id, second_memory.session_id);
    }

    #[tokio::test]
    async fn working_memory_updates_are_monotonic_and_lifecycle_terminal() {
        let (db, _dir) = make_db();
        let session = db.create_session("cli:monotonic-memory").await;
        assert!(db
            .save_working_memory(&session.id, "newer", "active", 10)
            .await
            .unwrap());
        assert!(!db
            .save_working_memory(&session.id, "stale", "active", 9)
            .await
            .unwrap());
        assert!(db
            .set_working_memory_status(&session.id, "completed")
            .await
            .unwrap());
        assert!(!db
            .save_working_memory(&session.id, "late active", "active", 11)
            .await
            .unwrap());
        assert!(db
            .set_working_memory_status(&session.id, "reflected")
            .await
            .unwrap());
        assert!(!db
            .save_working_memory(&session.id, "late completed", "completed", 12)
            .await
            .unwrap());
        assert!(!db
            .set_working_memory_status(&session.id, "active")
            .await
            .unwrap());

        let memory = db
            .get_or_create_working_memory(&session.id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(memory.content, "newer");
        assert_eq!(memory.last_updated_turn, 10);
        assert_eq!(memory.status, "reflected");
    }

    #[tokio::test]
    async fn lcm_checkpoint_commits_summary_and_working_memory_together() {
        let (db, _dir) = make_db();
        let session = db.create_session("cli:lcm-checkpoint").await;
        db.save_compaction_checkpoint(
            &session.id,
            4,
            &[11, 12],
            &[1],
            "durable summary",
            4,
            2,
            &SummaryManifest::default(),
            Some(("durable summary", 17)),
        )
        .await
        .unwrap();

        let nodes = db.load_summary_nodes(&session.id).await;
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].0, 4);
        assert_eq!(nodes[0].3, "durable summary");
        let memory = db
            .get_or_create_working_memory(&session.id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(memory.content, "durable summary");
        assert_eq!(memory.last_updated_turn, 17);

        assert!(db
            .save_compaction_checkpoint(
                "missing-session",
                5,
                &[21],
                &[],
                "must roll back",
                3,
                1,
                &SummaryManifest::default(),
                Some(("must roll back", 1)),
            )
            .await
            .is_err());
        let conn = db.conn.lock().await;
        let leaked: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM summary_nodes WHERE id = 5",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(leaked, 0);
    }

    #[tokio::test]
    async fn delete_session_transactionally_removes_all_owned_rows() {
        let (db, dir) = make_db();
        let jsonl = dir.path().join("legacy.jsonl");
        std::fs::write(&jsonl, "{\"role\":\"user\",\"content\":\"hello\"}\n").unwrap();
        let session_id = match db
            .import_legacy_jsonl(&jsonl, "legacy:cascade")
            .await
            .unwrap()
        {
            LegacyImportOutcome::Imported { session_id, .. } => session_id,
            LegacyImportOutcome::AlreadyImported { .. } => unreachable!(),
        };
        assert!(matches!(
            db.store_tool_result_immutable(&session_id, "call_1", "read_file", "raw body")
                .await,
            StoredResult::Stored { .. }
        ));
        db.save_summary_node(
            &session_id,
            901,
            &[1],
            &[],
            "summary",
            2,
            1,
            &SummaryManifest::default(),
        )
        .await;
        assert!(db
            .save_working_memory(&session_id, "working", "completed", 1)
            .await
            .unwrap());
        db.save_snapshot(&SessionSnapshot {
            version: 1,
            session_key: "legacy:cascade".to_string(),
            session_id: session_id.clone(),
            cwd: "/tmp".to_string(),
            model: "test".to_string(),
            tui_mode: "chat".to_string(),
            show_thinking: false,
            input_draft: String::new(),
            prompt_history: vec![],
            recent_paths: vec![],
            recent_commands: vec![],
            updated_at: Utc::now(),
        })
        .await;

        assert!(db.delete_session(&session_id).await.unwrap());
        assert!(!db.delete_session(&session_id).await.unwrap());
        assert!(db.get_session(&session_id).await.is_none());
        assert!(db.get_all_messages(&session_id).await.is_empty());
        assert!(db.load_tool_result(&session_id, "call_1").await.is_none());
        assert!(db.load_summary_nodes(&session_id).await.is_empty());
        assert!(db
            .get_or_create_working_memory(&session_id)
            .await
            .unwrap()
            .is_none());
        assert!(db.load_snapshot("legacy:cascade").await.is_none());

        {
            let conn = db.conn.lock().await;
            let imports: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM legacy_imports WHERE session_id = ?1",
                    params![session_id],
                    |row| row.get(0),
                )
                .unwrap();
            assert_eq!(imports, 1, "immutable import provenance must survive");
        }
        assert_eq!(
            db.import_legacy_jsonl(&jsonl, "legacy:cascade")
                .await
                .unwrap(),
            LegacyImportOutcome::AlreadyImported {
                session_id: session_id.clone(),
                message_count: 1,
            }
        );
        std::fs::write(&jsonl, "{\"role\":\"user\",\"content\":\"changed\"}\n").unwrap();
        assert!(matches!(
            db.import_legacy_jsonl(&jsonl, "legacy:cascade").await,
            Err(LegacyImportError::ChangedFile { .. })
        ));
    }

    #[tokio::test]
    async fn purge_and_nuke_cascade_in_single_database_transactions() {
        let (db, _dir) = make_db();
        let old = db.create_session("cli:old").await;
        let fresh = db.create_session("cli:fresh").await;
        db.save_working_memory(&old.id, "old", "active", 1)
            .await
            .unwrap();
        db.save_working_memory(&fresh.id, "fresh", "active", 1)
            .await
            .unwrap();
        {
            let conn = db.conn.lock().await;
            conn.execute(
                "UPDATE sessions SET updated_at = ?1 WHERE id = ?2",
                params![
                    (Utc::now() - chrono::Duration::days(10)).to_rfc3339(),
                    old.id
                ],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO legacy_imports \
                 (source_path, content_sha256, session_id, message_count, imported_at) \
                 VALUES ('/tmp/purged.jsonl', 'purged-hash', ?1, 1, ?2)",
                params![&old.id, Utc::now().to_rfc3339()],
            )
            .unwrap();
        }

        let removed = db
            .purge_sessions_before(Utc::now() - chrono::Duration::days(1))
            .await
            .unwrap();
        assert_eq!(removed, 1);
        assert!(db.get_session(&old.id).await.is_none());
        assert!(db.get_session(&fresh.id).await.is_some());
        assert_eq!(db.list_working_memory(None).await.unwrap().len(), 1);
        {
            let conn = db.conn.lock().await;
            let imports: i64 = conn
                .query_row("SELECT COUNT(*) FROM legacy_imports", [], |row| row.get(0))
                .unwrap();
            assert_eq!(imports, 1);
        }

        assert_eq!(db.nuke_sessions().await.unwrap(), 1);
        assert!(db.list_sessions(None, 100).await.is_empty());
        assert!(db.list_working_memory(None).await.unwrap().is_empty());
        let conn = db.conn.lock().await;
        let imports: i64 = conn
            .query_row("SELECT COUNT(*) FROM legacy_imports", [], |row| row.get(0))
            .unwrap();
        assert_eq!(imports, 1, "nuke must retain immutable import provenance");
    }

    #[tokio::test]
    async fn legacy_jsonl_import_is_idempotent_and_rejects_changed_file() {
        let (db, dir) = make_db();
        let path = dir.path().join("session.jsonl");
        std::fs::write(
            &path,
            concat!(
                "{\"_type\":\"metadata\",\"session_key\":\"cli:legacy\"}\n",
                "{\"role\":\"user\",\"content\":\"question\"}\n",
                "{\"role\":\"assistant\",\"content\":\"answer\"}\n"
            ),
        )
        .unwrap();

        let first = db
            .import_legacy_jsonl(&path, "cli:fallback-must-not-win")
            .await
            .unwrap();
        let (session_id, message_count) = match first {
            LegacyImportOutcome::Imported {
                session_id,
                message_count,
            } => (session_id, message_count),
            LegacyImportOutcome::AlreadyImported { .. } => unreachable!(),
        };
        assert_eq!(message_count, 2);
        assert_eq!(
            db.get_session(&session_id).await.unwrap().session_key,
            "cli:legacy",
            "legacy metadata session_key must be authoritative"
        );
        let second = db.import_legacy_jsonl(&path, "cli:legacy").await.unwrap();
        assert_eq!(
            second,
            LegacyImportOutcome::AlreadyImported {
                session_id: session_id.clone(),
                message_count: 2,
            }
        );
        let copied_path = dir.path().join("copied-session.jsonl");
        std::fs::copy(&path, &copied_path).unwrap();
        assert_eq!(
            db.import_legacy_jsonl(&copied_path, "cli:duplicate-copy")
                .await
                .unwrap(),
            LegacyImportOutcome::AlreadyImported {
                session_id: session_id.clone(),
                message_count: 2,
            },
            "identical bytes at another path must not create a second session"
        );
        assert_eq!(db.list_sessions(Some("cli:legacy"), 10).await.len(), 1);

        std::fs::write(
            &copied_path,
            "{\"role\":\"user\",\"content\":\"changed copy\"}\n",
        )
        .unwrap();
        assert!(matches!(
            db.import_legacy_jsonl(&copied_path, "cli:duplicate-copy")
                .await,
            Err(LegacyImportError::ChangedFile { .. })
        ));

        std::fs::write(&path, "{\"role\":\"user\",\"content\":\"changed\"}\n").unwrap();
        let changed = db.import_legacy_jsonl(&path, "cli:legacy").await;
        assert!(matches!(
            changed,
            Err(LegacyImportError::ChangedFile { .. })
        ));
        assert_eq!(db.get_all_messages(&session_id).await.len(), 2);
        assert_eq!(db.list_sessions(Some("cli:legacy"), 10).await.len(), 1);
    }

    #[tokio::test]
    async fn invalid_legacy_jsonl_rolls_back_without_creating_session() {
        let (db, dir) = make_db();
        let path = dir.path().join("broken.jsonl");
        std::fs::write(&path, "{\"role\":\"user\"}\nnot-json\n").unwrap();

        let result = db.import_legacy_jsonl(&path, "cli:broken").await;
        assert!(matches!(
            result,
            Err(LegacyImportError::InvalidJson { line: 2, .. })
        ));
        assert!(db.list_sessions(Some("cli:broken"), 10).await.is_empty());
    }

    #[tokio::test]
    async fn legacy_import_fk_migration_preserves_detached_provenance() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("sessions.db");
        {
            let conn = Connection::open(&path).unwrap();
            conn.execute_batch(
                "PRAGMA foreign_keys=ON;
                 CREATE TABLE sessions (
                    id TEXT PRIMARY KEY, session_key TEXT NOT NULL, created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL, message_count INTEGER DEFAULT 0,
                    metadata TEXT DEFAULT '{}'
                 );
                 CREATE TABLE legacy_imports (
                    source_path TEXT PRIMARY KEY, content_sha256 TEXT NOT NULL,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    message_count INTEGER NOT NULL, imported_at TEXT NOT NULL
                 );
                 INSERT INTO sessions VALUES
                    ('old-session', 'legacy:old', '2026-01-01T00:00:00Z',
                     '2026-01-01T00:00:00Z', 1, '{}');
                 INSERT INTO legacy_imports VALUES
                    ('/tmp/old.jsonl', 'abc123', 'old-session', 1,
                     '2026-01-01T00:00:00Z');",
            )
            .unwrap();
        }

        let db = SessionDb::new(&path);
        assert!(db.delete_session("old-session").await.unwrap());
        let conn = db.conn.lock().await;
        let foreign_keys: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM pragma_foreign_key_list('legacy_imports')",
                [],
                |row| row.get(0),
            )
            .unwrap();
        let records: i64 = conn
            .query_row("SELECT COUNT(*) FROM legacy_imports", [], |row| row.get(0))
            .unwrap();
        assert_eq!(foreign_keys, 0);
        assert_eq!(records, 1);
    }
}
