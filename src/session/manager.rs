//! Session management for conversation history.
//!
//! Sessions are stored as JSONL files. The first line is a metadata header
//! (with `_type: "metadata"`), followed by one JSON object per message.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Local};
use serde_json::{json, Value};
use tokio::sync::Mutex;
use tracing::warn;

use crate::utils::helpers::{ensure_dir, safe_filename};

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

/// A conversation session.
///
/// Stores messages in JSONL format for easy reading and persistence.
pub struct Session {
    /// Session key (usually `channel:chat_id`).
    pub key: String,
    /// Ordered list of messages. Each value is a JSON object with at least
    /// `role`, `content`, and `timestamp` fields.
    pub messages: Vec<Value>,
    /// When the session was originally created.
    pub created_at: DateTime<Local>,
    /// When the session was last updated (message added / cleared).
    pub updated_at: DateTime<Local>,
    /// Arbitrary metadata stored alongside the session.
    pub metadata: HashMap<String, Value>,
}

impl Session {
    /// Create a brand-new, empty session.
    pub fn new(key: &str) -> Self {
        let now = Local::now();
        Self {
            key: key.to_string(),
            messages: Vec::new(),
            created_at: now,
            updated_at: now,
            metadata: HashMap::new(),
        }
    }

    /// Add a message to the session with the current timestamp.
    pub fn add_message(&mut self, role: &str, content: &str) {
        let msg = json!({
            "role": role,
            "content": content,
            "timestamp": Local::now().to_rfc3339(),
        });
        self.messages.push(msg);
        self.updated_at = Local::now();
    }

    /// Return the last `max_messages` messages in LLM format (just `role` +
    /// `content`).
    pub fn get_history(&self, max_messages: usize) -> Vec<Value> {
        let start = if self.messages.len() > max_messages {
            self.messages.len() - max_messages
        } else {
            0
        };

        self.messages[start..]
            .iter()
            .map(|m| {
                json!({
                    "role": m.get("role").and_then(|v| v.as_str()).unwrap_or("user"),
                    "content": m.get("content").and_then(|v| v.as_str()).unwrap_or(""),
                })
            })
            .collect()
    }

    /// Clear all messages in the session.
    pub fn clear(&mut self) {
        self.messages.clear();
        self.updated_at = Local::now();
    }
}

// ---------------------------------------------------------------------------
// SessionManager
// ---------------------------------------------------------------------------

/// Manages conversation sessions.
///
/// Sessions are stored as JSONL files in `~/.nanoclaw/sessions`.
/// Thread-safe: the cache is protected by a Mutex so multiple tasks can
/// access sessions concurrently.
pub struct SessionManager {
    pub workspace: PathBuf,
    pub sessions_dir: PathBuf,
    cache: Mutex<HashMap<String, Session>>,
}

impl SessionManager {
    /// Create a new `SessionManager` rooted at `workspace`.
    pub fn new(workspace: &Path) -> Self {
        let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        let sessions_dir = ensure_dir(home.join(".nanoclaw").join("sessions"));

        Self {
            workspace: workspace.to_path_buf(),
            sessions_dir,
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// Get the history for a session, creating it if needed.
    pub async fn get_history(&self, key: &str, max_messages: usize) -> Vec<Value> {
        let mut cache = self.cache.lock().await;
        let session = Self::get_or_create_inner(&mut cache, key, &self.sessions_dir);
        session.get_history(max_messages)
    }

    /// Add a message to a session and persist it.
    pub async fn add_message_and_save(&self, key: &str, role: &str, content: &str) {
        let mut cache = self.cache.lock().await;
        let session = Self::get_or_create_inner(&mut cache, key, &self.sessions_dir);
        session.add_message(role, content);
        Self::save_session(session, &self.sessions_dir);
    }

    /// Add multiple messages to a session and persist it.
    pub async fn add_messages_and_save(&self, key: &str, messages: &[(&str, &str)]) {
        let mut cache = self.cache.lock().await;
        let session = Self::get_or_create_inner(&mut cache, key, &self.sessions_dir);
        for &(role, content) in messages {
            session.add_message(role, content);
        }
        Self::save_session(session, &self.sessions_dir);
    }

    /// Persist a session to disk as JSONL.
    ///
    /// The first line is a metadata header (`_type: "metadata"`), followed by
    /// one JSON object per message.
    pub fn save(&self, session: &Session) {
        Self::save_session(session, &self.sessions_dir);
    }

    /// Save a session that is already in the cache by its key.
    pub async fn save_cached(&self, key: &str) {
        let cache = self.cache.lock().await;
        if let Some(session) = cache.get(key) {
            Self::save_session(session, &self.sessions_dir);
        }
    }

    /// Delete a session from cache and disk.
    ///
    /// Returns `true` if the file was actually removed.
    pub async fn delete(&self, key: &str) -> bool {
        let mut cache = self.cache.lock().await;
        cache.remove(key);

        let path = Self::session_path(key, &self.sessions_dir);
        if path.exists() {
            let _ = fs::remove_file(&path);
            return true;
        }
        false
    }

    /// List all sessions found on disk.
    ///
    /// Returns a `Vec` of JSON objects with `key`, `created_at`, `updated_at`,
    /// and `path` fields, sorted by `updated_at` descending.
    pub fn list_sessions(&self) -> Vec<Value> {
        let mut sessions: Vec<Value> = Vec::new();

        let entries = match fs::read_dir(&self.sessions_dir) {
            Ok(e) => e,
            Err(_) => return sessions,
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
                continue;
            }
            if let Ok(content) = fs::read_to_string(&path) {
                if let Some(first_line) = content.lines().next() {
                    if let Ok(data) = serde_json::from_str::<Value>(first_line) {
                        if data.get("_type").and_then(|v| v.as_str()) == Some("metadata") {
                            let stem = path
                                .file_stem()
                                .unwrap_or_default()
                                .to_string_lossy()
                                .replace('_', ":");
                            sessions.push(json!({
                                "key": stem,
                                "created_at": data.get("created_at"),
                                "updated_at": data.get("updated_at"),
                                "path": path.to_string_lossy(),
                            }));
                        }
                    }
                }
            }
        }

        // Sort by updated_at descending.
        sessions.sort_by(|a, b| {
            let ua = a
                .get("updated_at")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let ub = b
                .get("updated_at")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            ub.cmp(ua)
        });

        sessions
    }

    // -----------------------------------------------------------------------
    // Private/internal helpers
    // -----------------------------------------------------------------------

    /// Get or create a session within an already-locked cache.
    fn get_or_create_inner<'a>(
        cache: &'a mut HashMap<String, Session>,
        key: &str,
        sessions_dir: &Path,
    ) -> &'a mut Session {
        if !cache.contains_key(key) {
            let session = Self::load_from_disk(key, sessions_dir)
                .unwrap_or_else(|| Session::new(key));
            cache.insert(key.to_string(), session);
        }
        cache.get_mut(key).expect("session must exist in cache")
    }

    /// Save a session to its JSONL file.
    fn save_session(session: &Session, sessions_dir: &Path) {
        let path = Self::session_path(&session.key, sessions_dir);

        let metadata_line = json!({
            "_type": "metadata",
            "created_at": session.created_at.to_rfc3339(),
            "updated_at": session.updated_at.to_rfc3339(),
            "metadata": Value::Object(
                session.metadata.iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect()
            ),
        });

        let mut lines = Vec::with_capacity(session.messages.len() + 1);
        lines.push(serde_json::to_string(&metadata_line).unwrap_or_default());
        for msg in &session.messages {
            lines.push(serde_json::to_string(msg).unwrap_or_default());
        }

        let content = lines.join("\n") + "\n";
        if let Err(e) = fs::write(&path, content) {
            warn!("Failed to save session {}: {}", session.key, e);
        }
    }

    /// Load a session from its JSONL file on disk.
    fn load_from_disk(key: &str, sessions_dir: &Path) -> Option<Session> {
        let path = Self::session_path(key, sessions_dir);
        if !path.exists() {
            return None;
        }

        let content = match fs::read_to_string(&path) {
            Ok(c) => c,
            Err(e) => {
                warn!("Failed to read session file {}: {}", key, e);
                return None;
            }
        };

        let mut messages: Vec<Value> = Vec::new();
        let mut metadata: HashMap<String, Value> = HashMap::new();
        let mut created_at: Option<DateTime<Local>> = None;

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            match serde_json::from_str::<Value>(line) {
                Ok(data) => {
                    if data.get("_type").and_then(|v| v.as_str()) == Some("metadata") {
                        if let Some(obj) = data.get("metadata").and_then(|v| v.as_object()) {
                            metadata = obj
                                .iter()
                                .map(|(k, v)| (k.clone(), v.clone()))
                                .collect();
                        }
                        if let Some(ts) = data.get("created_at").and_then(|v| v.as_str()) {
                            if let Ok(dt) = DateTime::parse_from_rfc3339(ts) {
                                created_at = Some(dt.with_timezone(&Local));
                            }
                        }
                    } else {
                        messages.push(data);
                    }
                }
                Err(e) => {
                    warn!("Skipping bad JSON line in session {}: {}", key, e);
                }
            }
        }

        Some(Session {
            key: key.to_string(),
            messages,
            created_at: created_at.unwrap_or_else(Local::now),
            updated_at: Local::now(),
            metadata,
        })
    }

    /// Compute the filesystem path for a given session key.
    fn session_path(key: &str, sessions_dir: &Path) -> PathBuf {
        let safe_key = safe_filename(&key.replace(':', "_"));
        sessions_dir.join(format!("{}.jsonl", safe_key))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_add_and_history() {
        let mut session = Session::new("test:123");
        session.add_message("user", "hello");
        session.add_message("assistant", "hi there");
        session.add_message("user", "how are you?");

        let history = session.get_history(2);
        assert_eq!(history.len(), 2);
        assert_eq!(
            history[0].get("role").and_then(|v| v.as_str()),
            Some("assistant")
        );
        assert_eq!(
            history[1].get("content").and_then(|v| v.as_str()),
            Some("how are you?")
        );
    }

    #[test]
    fn test_session_clear() {
        let mut session = Session::new("test:456");
        session.add_message("user", "hi");
        assert_eq!(session.messages.len(), 1);

        session.clear();
        assert!(session.messages.is_empty());
    }

    #[test]
    fn test_session_path() {
        let tmp = std::env::temp_dir().join("nanoclaw_test_session_path");
        let mgr = SessionManager::new(&tmp);
        let path = SessionManager::session_path("telegram:12345", &mgr.sessions_dir);
        assert!(path.to_string_lossy().ends_with("telegram_12345.jsonl"));
    }

    #[tokio::test]
    async fn test_get_history_creates_session() {
        let tmp = std::env::temp_dir().join("nanoclaw_test_get_history");
        let mgr = SessionManager::new(&tmp);
        let history = mgr.get_history("new:session", 100).await;
        assert!(history.is_empty());
    }

    #[tokio::test]
    async fn test_add_message_and_save() {
        let tmp = std::env::temp_dir().join("nanoclaw_test_add_msg");
        let mgr = SessionManager::new(&tmp);
        // Use a unique key to avoid interference from previous test runs.
        let key = format!("test:add_{}", uuid::Uuid::new_v4());
        mgr.add_message_and_save(&key, "user", "hello").await;
        let history = mgr.get_history(&key, 100).await;
        assert_eq!(history.len(), 1);
        assert_eq!(
            history[0].get("content").and_then(|v| v.as_str()),
            Some("hello")
        );
    }
}
