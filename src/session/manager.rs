#![allow(dead_code)]
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

    /// Return the last `max_messages` messages in LLM format.
    ///
    /// `max_turns` limits how many user turns to include (0 = no limit).
    /// Preserves `tool_calls` on assistant messages and `tool_call_id` on
    /// tool result messages so that resumed sessions include full tool context.
    pub fn get_history(&self, max_messages: usize, max_turns: usize) -> Vec<Value> {
        let start = if self.messages.len() > max_messages {
            self.messages.len() - max_messages
        } else {
            0
        };

        // Advance past any orphaned tool results at the window boundary.
        // A tool result at the boundary is orphaned because its matching
        // assistant+tool_calls message was before the window and got dropped.
        let mut safe_start = start;
        while safe_start < self.messages.len() {
            let role = self.messages[safe_start]
                .get("role")
                .and_then(|r| r.as_str())
                .unwrap_or("");
            if role == "tool" {
                safe_start += 1;
            } else {
                break;
            }
        }

        // Turn-based limit: scan backward counting user messages as turn boundaries.
        if max_turns > 0 {
            let mut turns_seen = 0;
            let mut turn_start = safe_start;
            for i in (safe_start..self.messages.len()).rev() {
                if self.messages[i].get("role").and_then(|r| r.as_str()) == Some("user") {
                    turns_seen += 1;
                    if turns_seen > max_turns {
                        break;
                    }
                    turn_start = i;
                }
            }
            safe_start = safe_start.max(turn_start);
        }

        self.messages[safe_start..]
            .iter()
            .map(|m| {
                let role = m.get("role").and_then(|v| v.as_str()).unwrap_or("user");
                let mut msg = json!({
                    "role": role,
                    "content": m.get("content").and_then(|v| v.as_str()).unwrap_or(""),
                });
                // Preserve tool_calls on assistant messages.
                if let Some(tc) = m.get("tool_calls") {
                    msg["tool_calls"] = tc.clone();
                }
                // Preserve tool_call_id on tool result messages.
                if let Some(id) = m.get("tool_call_id") {
                    msg["tool_call_id"] = id.clone();
                }
                // Preserve name on tool result messages.
                if let Some(name) = m.get("name") {
                    msg["name"] = name.clone();
                }
                msg
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

pub fn should_rotate(session: &Session, rotation_size_bytes: usize) -> bool {
    let age_hours = (Local::now() - session.updated_at).num_hours();
    if age_hours >= 24 {
        return true;
    }

    let size: usize = session.messages.iter().map(|m| m.to_string().len()).sum();
    if size > rotation_size_bytes {
        return true;
    }

    false
}

/// Manages conversation sessions.
///
/// Sessions are stored as JSONL files in `~/.nanobot/sessions`.
/// Thread-safe: the cache is protected by a Mutex so multiple tasks can
/// access sessions concurrently.
pub struct SessionManager {
    pub workspace: PathBuf,
    pub sessions_dir: PathBuf,
    cache: Mutex<HashMap<String, Session>>,
    /// Rotate session file when it exceeds this size in bytes.
    rotation_size_bytes: usize,
    /// Number of recent messages to carry into a new session on rotation.
    rotation_carry_messages: usize,
}

impl SessionManager {
    /// Create a new `SessionManager` rooted at `workspace` with default tuning.
    pub fn new(workspace: &Path) -> Self {
        let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        let sessions_dir = ensure_dir(home.join(".nanobot").join("sessions"));

        Self {
            workspace: workspace.to_path_buf(),
            sessions_dir,
            cache: Mutex::new(HashMap::new()),
            rotation_size_bytes: 1_000_000,
            rotation_carry_messages: 10,
        }
    }

    /// Create a new `SessionManager` with explicit tuning values from config.
    pub fn with_tuning(workspace: &Path, rotation_size_bytes: usize, rotation_carry_messages: usize) -> Self {
        let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        let sessions_dir = ensure_dir(home.join(".nanobot").join("sessions"));

        Self {
            workspace: workspace.to_path_buf(),
            sessions_dir,
            cache: Mutex::new(HashMap::new()),
            rotation_size_bytes,
            rotation_carry_messages,
        }
    }

    /// Get the history for a session, creating it if needed.
    ///
    /// `max_turns` limits how many user turns (conversation rounds) to load.
    /// Set to 0 to disable turn-based limiting.
    pub async fn get_history(
        &self,
        key: &str,
        max_messages: usize,
        max_turns: usize,
    ) -> Vec<Value> {
        let mut cache = self.cache.lock().await;
        let session = Self::get_or_create_inner(&mut cache, key, &self.sessions_dir, self.rotation_size_bytes, self.rotation_carry_messages);
        session.get_history(max_messages, max_turns)
    }

    /// Add a message to a session and persist it.
    pub async fn add_message_and_save(&self, key: &str, role: &str, content: &str) {
        let mut cache = self.cache.lock().await;
        let session = Self::get_or_create_inner(&mut cache, key, &self.sessions_dir, self.rotation_size_bytes, self.rotation_carry_messages);
        session.add_message(role, content);
        Self::save_session(session, &self.sessions_dir);
    }

    /// Add multiple messages to a session and persist it.
    pub async fn add_messages_and_save(&self, key: &str, messages: &[(&str, &str)]) {
        let mut cache = self.cache.lock().await;
        let session = Self::get_or_create_inner(&mut cache, key, &self.sessions_dir, self.rotation_size_bytes, self.rotation_carry_messages);
        for &(role, content) in messages {
            session.add_message(role, content);
        }
        Self::save_session(session, &self.sessions_dir);
    }

    /// Save a batch of raw JSON messages (preserving tool_calls, tool_call_id, etc.)
    pub async fn add_messages_raw(&self, key: &str, messages: &[Value]) {
        let mut cache = self.cache.lock().await;
        let session = Self::get_or_create_inner(&mut cache, key, &self.sessions_dir, self.rotation_size_bytes, self.rotation_carry_messages);
        for msg in messages {
            let mut m = msg.clone();
            // Add timestamp if not present.
            if m.get("timestamp").is_none() {
                m["timestamp"] = json!(Local::now().to_rfc3339());
            }
            session.messages.push(m);
        }
        session.updated_at = Local::now();
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
            let ua = a.get("updated_at").and_then(|v| v.as_str()).unwrap_or("");
            let ub = b.get("updated_at").and_then(|v| v.as_str()).unwrap_or("");
            ub.cmp(ua)
        });

        sessions
    }

    // -----------------------------------------------------------------------
    // Private/internal helpers
    // ---------------------------------------------------------------------------

    /// Get or create a session within an already-locked cache.
    ///
    /// Handles rotation:
    /// 1. Daily rotation: if session was created on a different day
    /// 2. Size rotation: if session exceeds `rotation_size_bytes`
    /// When rotating, carries over last `rotation_carry_messages` messages to new session.
    fn get_or_create_inner<'a>(
        cache: &'a mut HashMap<String, Session>,
        key: &str,
        sessions_dir: &Path,
        rotation_size_bytes: usize,
        rotation_carry_messages: usize,
    ) -> &'a mut Session {
        let today = Local::now().format("%Y-%m-%d").to_string();
        let mut carry_messages: Option<Vec<Value>> = None;

        if let Some(existing) = cache.get(key) {
            let session_date = existing.created_at.format("%Y-%m-%d").to_string();
            let size: usize = existing.messages.iter().map(|m| m.to_string().len()).sum();

            if session_date != today || size > rotation_size_bytes {
                Self::save_session(existing, sessions_dir);

                let carried: Vec<Value> = existing.messages
                    .iter()
                    .rev()
                    .take(rotation_carry_messages)
                    .rev()
                    .cloned()
                    .collect();

                if !carried.is_empty() && session_date == today {
                    carry_messages = Some(carried);
                }

                cache.remove(key);
            }
        }

        if !cache.contains_key(key) {
            let mut session =
                Self::load_from_disk(key, sessions_dir).unwrap_or_else(|| Session::new(key));
            
            if let Some(carried) = carry_messages {
                session.messages = carried;
                session.created_at = Local::now();
            }
            
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
        // Atomic write: write to temp file then rename to avoid corruption on crash.
        let tmp_path = path.with_extension("jsonl.tmp");
        if let Err(e) = fs::write(&tmp_path, &content) {
            warn!("Failed to write temp session {}: {}", session.key, e);
            return;
        }
        if let Err(e) = fs::rename(&tmp_path, &path) {
            warn!("Failed to rename session {}: {}", session.key, e);
        }
    }

    /// Load a session from its JSONL file on disk.
    ///
    /// Tries today's dated file first, then falls back to the legacy
    /// (undated) file for migration.
    fn load_from_disk(key: &str, sessions_dir: &Path) -> Option<Session> {
        let path = Self::session_path(key, sessions_dir);
        let path = if path.exists() {
            path
        } else {
            // Fall back to legacy undated path for migration.
            let legacy = Self::legacy_session_path(key, sessions_dir);
            if legacy.exists() {
                legacy
            } else {
                return None;
            }
        };

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
                            metadata = obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
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
    ///
    /// Uses daily rotation: `cli_default_2026-02-14.jsonl`.  Each day gets
    /// its own file so qmd indexes smaller, naturally-archived documents.
    fn session_path(key: &str, sessions_dir: &Path) -> PathBuf {
        let safe_key = safe_filename(&key.replace(':', "_"));
        let date = Local::now().format("%Y-%m-%d");
        sessions_dir.join(format!("{}_{}.jsonl", safe_key, date))
    }

    /// Compute the path for yesterday's session (for migration/fallback).
    #[allow(dead_code)]
    fn session_path_for_date(key: &str, sessions_dir: &Path, date: &str) -> PathBuf {
        let safe_key = safe_filename(&key.replace(':', "_"));
        sessions_dir.join(format!("{}_{}.jsonl", safe_key, date))
    }

    /// Legacy path without date suffix (for migration).
    fn legacy_session_path(key: &str, sessions_dir: &Path) -> PathBuf {
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

        let history = session.get_history(2, 0);
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
    fn test_session_path_includes_date() {
        let tmp = std::env::temp_dir().join("nanobot_test_session_path");
        let mgr = SessionManager::new(&tmp);
        let path = SessionManager::session_path("telegram:12345", &mgr.sessions_dir);
        let path_str = path.to_string_lossy().to_string();
        // Should contain the key and today's date.
        assert!(path_str.contains("telegram_12345_"), "path={}", path_str);
        assert!(path_str.ends_with(".jsonl"), "path={}", path_str);
        // Date portion should be YYYY-MM-DD format.
        let today = Local::now().format("%Y-%m-%d").to_string();
        assert!(path_str.contains(&today), "path={}", path_str);
    }

    #[test]
    fn test_legacy_session_path() {
        let tmp = std::env::temp_dir().join("nanobot_test_legacy_path");
        let mgr = SessionManager::new(&tmp);
        let path = SessionManager::legacy_session_path("telegram:12345", &mgr.sessions_dir);
        assert!(path.to_string_lossy().ends_with("telegram_12345.jsonl"));
    }

    #[tokio::test]
    async fn test_get_history_creates_session() {
        let tmp = std::env::temp_dir().join("nanobot_test_get_history");
        let mgr = SessionManager::new(&tmp);
        let history = mgr.get_history("new:session", 100, 0).await;
        assert!(history.is_empty());
    }

    #[tokio::test]
    async fn test_add_messages_raw_preserves_tool_calls() {
        let tmp = std::env::temp_dir().join("nanobot_test_raw_tool_calls");
        let mgr = SessionManager::new(&tmp);
        let key = format!("test:raw_{}", uuid::Uuid::new_v4());

        let messages = vec![
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
        mgr.add_messages_raw(&key, &messages).await;

        let history = mgr.get_history(&key, 100, 0).await;
        assert_eq!(history.len(), 4);

        // Verify tool_calls preserved on assistant message.
        let assistant_msg = &history[1];
        assert!(
            assistant_msg.get("tool_calls").is_some(),
            "tool_calls should be preserved"
        );

        // Verify tool_call_id preserved on tool result message.
        let tool_msg = &history[2];
        assert_eq!(
            tool_msg.get("tool_call_id").and_then(|v| v.as_str()),
            Some("tc_1")
        );
        assert_eq!(
            tool_msg.get("name").and_then(|v| v.as_str()),
            Some("read_file")
        );
    }

    #[tokio::test]
    async fn test_add_message_and_save() {
        let tmp = std::env::temp_dir().join("nanobot_test_add_msg");
        let mgr = SessionManager::new(&tmp);
        // Use a unique key to avoid interference from previous test runs.
        let key = format!("test:add_{}", uuid::Uuid::new_v4());
        mgr.add_message_and_save(&key, "user", "hello").await;
        let history = mgr.get_history(&key, 100, 0).await;
        assert_eq!(history.len(), 1);
        assert_eq!(
            history[0].get("content").and_then(|v| v.as_str()),
            Some("hello")
        );
    }

    #[test]
    fn test_get_history_skips_orphaned_tool_results_at_boundary() {
        let mut session = Session::new("test");
        // Build: user → assistant+tool_calls → tool result → assistant → user → assistant
        session
            .messages
            .push(json!({"role": "user", "content": "q1"}));
        session.messages.push(json!({
            "role": "assistant", "content": "",
            "tool_calls": [{"id": "tc_1", "type": "function", "function": {"name": "exec", "arguments": "{}"}}]
        }));
        session
            .messages
            .push(json!({"role": "tool", "tool_call_id": "tc_1", "name": "exec", "content": "ok"}));
        session
            .messages
            .push(json!({"role": "assistant", "content": "Done"}));
        session
            .messages
            .push(json!({"role": "user", "content": "q2"}));
        session
            .messages
            .push(json!({"role": "assistant", "content": "answer"}));

        // Window of 4: naive start=2 → messages[2] is the tool result (orphan).
        // Protocol-safe windowing should skip it.
        let history = session.get_history(4, 0);
        assert!(
            history
                .iter()
                .all(|m| m.get("role").and_then(|r| r.as_str()) != Some("tool")),
            "Orphaned tool results at window boundary should be skipped"
        );
        // Should have: assistant("Done"), user("q2"), assistant("answer") = 3 msgs
        assert_eq!(history.len(), 3);
    }

    #[test]
    fn test_get_history_preserves_complete_tool_groups() {
        let mut session = Session::new("test");
        // assistant+tool_calls → tool result → user → assistant
        session.messages.push(json!({
            "role": "assistant", "content": "",
            "tool_calls": [{"id": "tc_1", "type": "function", "function": {"name": "read", "arguments": "{}"}}]
        }));
        session.messages.push(
            json!({"role": "tool", "tool_call_id": "tc_1", "name": "read", "content": "data"}),
        );
        session
            .messages
            .push(json!({"role": "user", "content": "thanks"}));
        session
            .messages
            .push(json!({"role": "assistant", "content": "you're welcome"}));

        // All 4 messages fit — tool result is NOT orphaned because its assistant is in the window.
        let history = session.get_history(100, 0);
        assert_eq!(history.len(), 4);
        let has_tool = history
            .iter()
            .any(|m| m.get("role").and_then(|r| r.as_str()) == Some("tool"));
        assert!(has_tool, "Complete tool group should be preserved");
    }

    #[test]
    fn test_get_history_turn_limited() {
        let mut session = Session::new("test:turns");
        // Build 6 turns: user → assistant × 6
        for i in 0..6 {
            session
                .messages
                .push(json!({"role": "user", "content": format!("question {}", i)}));
            session
                .messages
                .push(json!({"role": "assistant", "content": format!("answer {}", i)}));
        }
        assert_eq!(session.messages.len(), 12);

        // max_turns=3 should return only the last 3 turns (6 messages).
        let history = session.get_history(100, 3);
        assert_eq!(history.len(), 6);
        assert_eq!(
            history[0].get("content").and_then(|v| v.as_str()),
            Some("question 3")
        );
        assert_eq!(
            history[5].get("content").and_then(|v| v.as_str()),
            Some("answer 5")
        );

        // max_turns=0 should return all messages (no turn limit).
        let history_all = session.get_history(100, 0);
        assert_eq!(history_all.len(), 12);

        // max_turns=1 should return the last turn (2 messages).
        let history_one = session.get_history(100, 1);
        assert_eq!(history_one.len(), 2);
        assert_eq!(
            history_one[0].get("content").and_then(|v| v.as_str()),
            Some("question 5")
        );
    }

    #[test]
    fn test_should_rotate_by_size() {
        let mut session = Session::new("test:large");
        // Create a session larger than 1MB
        let large_content = "x".repeat(100_000);
        for _ in 0..12 {
            session.add_message("user", &large_content);
        }

        assert!(should_rotate(&session, 1_000_000), "Should rotate when size > 1MB");
    }

    #[test]
    fn test_should_not_rotate_small_session() {
        let session = Session::new("test:small");
        // Fresh session with no messages should not rotate
        assert!(!should_rotate(&session, 1_000_000), "Should not rotate small session");
    }

    #[tokio::test]
    async fn test_rotation_carries_recent_messages() {
        let tmp = std::env::temp_dir().join("nanobot_test_rotation_carry");
        let mgr = SessionManager::new(&tmp);
        let key = format!("test:rotate_{}", uuid::Uuid::new_v4());
        
        // Add many messages to exceed 1MB
        let large_content = "x".repeat(100_000);
        for i in 0..12 {
            mgr.add_message_and_save(&key, "user", &format!("{}: {}", i, large_content)).await;
        }
        
        // Get history - should trigger rotation
        let history = mgr.get_history(&key, 100, 0).await;
        
        // Should have carried over last 10 messages
        assert!(history.len() <= 10, "After rotation should have at most 10 messages, got {}", history.len());
    }
}
