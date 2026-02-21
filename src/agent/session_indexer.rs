//! Session indexer: create searchable SESSION_*.md files from orphaned JSONL sessions.
//!
//! Reconciles `~/.nanobot/sessions/*.jsonl` with `workspace/memory/sessions/SESSION_*.md`.
//! For each JSONL file without a corresponding summary, extracts user+assistant messages
//! and writes a searchable .md file that the `recall` tool can grep.

use std::fs;
use std::path::Path;

use sha2::{Digest, Sha256};
use tracing::warn;

/// A single indexed session ready to be written as SESSION_*.md.
#[derive(Debug)]
pub struct IndexedSession {
    pub session_key: String,
    pub source_file: String,
    pub created: String,
    pub updated: String,
    pub message_count: usize,
    pub content: String,
}

/// Deterministic hash for a session key — first 8 hex chars of SHA-256.
///
/// Same algorithm as `WorkingMemoryStore::session_hash`.
pub fn session_hash(session_key: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(session_key.as_bytes());
    let result = hasher.finalize();
    result[..4].iter().map(|b| format!("{:02x}", b)).collect()
}

/// Derive a session key from a JSONL filename.
///
/// E.g. `"cli_default_2026-02-21.jsonl"` → `"cli:default"` (strip date suffix, restore colon).
/// Falls back to filename stem if no pattern matches.
fn session_key_from_filename(filename: &str) -> String {
    let stem = filename.trim_end_matches(".jsonl");
    // Try to extract session key from metadata line first — caller handles that.
    // Here we just derive from filename pattern: `channel_name_YYYY-MM-DD`
    // Strip trailing `_YYYY-MM-DD` date suffix if present.
    let key = if stem.len() > 11 {
        let maybe_date = &stem[stem.len() - 11..];
        if maybe_date.starts_with('_')
            && maybe_date[1..5].chars().all(|c| c.is_ascii_digit())
            && maybe_date.as_bytes()[5] == b'-'
        {
            &stem[..stem.len() - 11]
        } else {
            stem
        }
    } else {
        stem
    };
    // Convert first underscore to colon to restore `cli:default` form.
    if let Some(pos) = key.find('_') {
        format!("{}:{}", &key[..pos], &key[pos + 1..])
    } else {
        key.to_string()
    }
}

/// Extract searchable content from raw JSONL lines.
///
/// Pure function — takes lines and returns structured data.
/// Skips metadata lines, tool messages, and lines > 10KB.
/// Truncates individual messages to 500 chars and caps at `max_messages`.
pub fn extract_session_content(
    lines: &[&str],
    filename: &str,
    max_messages: usize,
) -> IndexedSession {
    let mut created = String::new();
    let mut updated = String::new();
    let mut session_key = session_key_from_filename(filename);
    let mut messages: Vec<String> = Vec::new();

    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        // Skip lines > 10KB (likely tool results with large output).
        if line.len() > 10_000 {
            continue;
        }

        let parsed: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };

        // Handle metadata lines.
        if parsed.get("_type").and_then(|v| v.as_str()) == Some("metadata") {
            if let Some(c) = parsed.get("created_at").and_then(|v| v.as_str()) {
                created = c.to_string();
            }
            if let Some(u) = parsed.get("updated_at").and_then(|v| v.as_str()) {
                updated = u.to_string();
            }
            if let Some(k) = parsed.get("session_key").and_then(|v| v.as_str()) {
                session_key = k.to_string();
            }
            continue;
        }

        // Only process user and assistant messages.
        let role = match parsed.get("role").and_then(|v| v.as_str()) {
            Some("user") | Some("assistant") => parsed["role"].as_str().unwrap(),
            _ => continue,
        };

        // Extract content, truncate to 500 chars.
        let content = parsed
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if content.is_empty() {
            continue;
        }

        let truncated: String = if content.len() > 500 {
            let cut: String = content.chars().take(500).collect();
            format!("{}...", cut)
        } else {
            content.to_string()
        };

        messages.push(format!("**{}:** {}", role, truncated));

        if messages.len() >= max_messages {
            break;
        }
    }

    IndexedSession {
        session_key,
        source_file: filename.to_string(),
        created,
        updated,
        message_count: messages.len(),
        content: messages.join("\n"),
    }
}

/// Scan sessions_dir for JSONL files without corresponding SESSION_*.md.
/// Create .md summaries for orphaned files.
///
/// Returns `(indexed_count, skipped_count, error_count)`.
pub fn index_sessions(
    sessions_dir: &Path,
    memory_sessions_dir: &Path,
) -> (usize, usize, usize) {
    let entries = match fs::read_dir(sessions_dir) {
        Ok(e) => e,
        Err(e) => {
            warn!("Cannot read sessions dir: {}", e);
            return (0, 0, 1);
        }
    };

    // Ensure memory sessions directory exists.
    if let Err(e) = fs::create_dir_all(memory_sessions_dir) {
        warn!("Cannot create memory sessions dir: {}", e);
        return (0, 0, 1);
    }

    let mut indexed = 0usize;
    let mut skipped = 0usize;
    let mut errors = 0usize;

    for entry in entries.flatten() {
        let path = entry.path();

        // Only process .jsonl files.
        let ext = path.extension().and_then(|e| e.to_str());
        if ext != Some("jsonl") {
            continue;
        }

        let filename = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n.to_string(),
            None => continue,
        };

        // Skip test rotation artifacts.
        if filename.contains("test_rotate") {
            skipped += 1;
            continue;
        }

        // Derive the session key from filename to compute the hash.
        // We need to peek at metadata to get the real session key.
        let content = match fs::read_to_string(&path) {
            Ok(c) => c,
            Err(e) => {
                warn!("Cannot read {}: {}", filename, e);
                errors += 1;
                continue;
            }
        };

        // Quick-extract session_key from first metadata line.
        let session_key = extract_session_key_from_content(&content, &filename);

        // Check if SESSION_*.md already exists.
        let hash = session_hash(&session_key);
        let md_filename = format!("SESSION_{}.md", hash);
        let md_path = memory_sessions_dir.join(&md_filename);

        if md_path.exists() {
            skipped += 1;
            continue;
        }

        // Extract content.
        let lines: Vec<&str> = content.lines().collect();
        let session = extract_session_content(&lines, &filename, 50);

        if session.content.is_empty() {
            skipped += 1;
            continue;
        }

        // Write SESSION_*.md with YAML frontmatter.
        let md_content = format!(
            "---\nsession_key: \"{}\"\ncreated: \"{}\"\nupdated: \"{}\"\nstatus: indexed\nsource: jsonl_extraction\nsource_file: \"{}\"\nmessage_count: {}\n---\n\n{}",
            session.session_key,
            session.created,
            session.updated,
            session.source_file,
            session.message_count,
            session.content,
        );

        if let Err(e) = fs::write(&md_path, &md_content) {
            warn!("Cannot write {}: {}", md_filename, e);
            errors += 1;
            continue;
        }

        indexed += 1;
    }

    (indexed, skipped, errors)
}

/// Extract session_key from the first metadata line in JSONL content.
fn extract_session_key_from_content(content: &str, filename: &str) -> String {
    for line in content.lines().take(3) {
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(line) {
            if parsed.get("_type").and_then(|v| v.as_str()) == Some("metadata") {
                if let Some(key) = parsed.get("session_key").and_then(|v| v.as_str()) {
                    return key.to_string();
                }
            }
        }
    }
    // Fallback: derive from filename.
    session_key_from_filename(filename)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    // --- extract_session_content (pure function tests) ---

    #[test]
    fn test_extract_user_and_assistant() {
        let lines = vec![
            r#"{"_type":"metadata","session_key":"cli:test","created_at":"2026-01-01T00:00:00Z","updated_at":"2026-01-01T01:00:00Z"}"#,
            r#"{"role":"user","content":"What is Rust?"}"#,
            r#"{"role":"assistant","content":"Rust is a systems programming language."}"#,
        ];
        let result = extract_session_content(&lines, "test.jsonl", 50);
        assert_eq!(result.session_key, "cli:test");
        assert_eq!(result.message_count, 2);
        assert!(result.content.contains("**user:** What is Rust?"));
        assert!(result.content.contains("**assistant:** Rust is a systems"));
        assert_eq!(result.created, "2026-01-01T00:00:00Z");
        assert_eq!(result.updated, "2026-01-01T01:00:00Z");
    }

    #[test]
    fn test_extract_skips_tool_messages() {
        let lines = vec![
            r#"{"role":"user","content":"Read the file"}"#,
            r#"{"role":"tool","name":"read_file","content":"file contents here..."}"#,
            r#"{"role":"assistant","content":"The file contains..."}"#,
        ];
        let result = extract_session_content(&lines, "test.jsonl", 50);
        assert_eq!(result.message_count, 2);
        assert!(!result.content.contains("tool"));
    }

    #[test]
    fn test_extract_skips_metadata() {
        let lines = vec![
            r#"{"_type":"metadata","session_key":"cli:x","created_at":"2026-01-01T00:00:00Z"}"#,
            r#"{"role":"user","content":"hello"}"#,
        ];
        let result = extract_session_content(&lines, "test.jsonl", 50);
        assert_eq!(result.message_count, 1);
        assert!(!result.content.contains("metadata"));
    }

    #[test]
    fn test_extract_skips_large_lines() {
        let big = "x".repeat(11_000);
        let big_line = format!(r#"{{"role":"user","content":"{}"}}"#, big);
        let lines = vec![
            big_line.as_str(),
            r#"{"role":"user","content":"small message"}"#,
        ];
        let result = extract_session_content(&lines, "test.jsonl", 50);
        assert_eq!(result.message_count, 1);
        assert!(result.content.contains("small message"));
    }

    #[test]
    fn test_extract_caps_messages() {
        let mut lines: Vec<String> = Vec::new();
        for i in 0..100 {
            lines.push(format!(r#"{{"role":"user","content":"msg {}"}}"#, i));
        }
        let refs: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();
        let result = extract_session_content(&refs, "test.jsonl", 5);
        assert_eq!(result.message_count, 5);
    }

    #[test]
    fn test_extract_truncates_content() {
        let long_text = "a".repeat(1000);
        let line = format!(r#"{{"role":"user","content":"{}"}}"#, long_text);
        let lines = vec![line.as_str()];
        let result = extract_session_content(&lines, "test.jsonl", 50);
        assert_eq!(result.message_count, 1);
        // 500 chars + "..." suffix
        assert!(result.content.len() < 520);
        assert!(result.content.contains("..."));
    }

    #[test]
    fn test_extract_empty_session() {
        let lines = vec![
            r#"{"_type":"metadata","session_key":"cli:empty","created_at":"2026-01-01T00:00:00Z"}"#,
        ];
        let result = extract_session_content(&lines, "empty.jsonl", 50);
        assert_eq!(result.message_count, 0);
        assert!(result.content.is_empty());
    }

    #[test]
    fn test_extract_handles_corrupt_lines() {
        let lines = vec![
            "not valid json at all",
            r#"{"role":"user","content":"valid"}"#,
            "{broken",
            r#"{"role":"assistant","content":"also valid"}"#,
        ];
        let result = extract_session_content(&lines, "test.jsonl", 50);
        assert_eq!(result.message_count, 2);
    }

    // --- session_key_from_filename ---

    #[test]
    fn test_session_key_from_filename_with_date() {
        assert_eq!(
            session_key_from_filename("cli_default_2026-02-21.jsonl"),
            "cli:default"
        );
    }

    #[test]
    fn test_session_key_from_filename_without_date() {
        assert_eq!(
            session_key_from_filename("cli_mytest.jsonl"),
            "cli:mytest"
        );
    }

    // --- session_hash ---

    #[test]
    fn test_session_hash_deterministic() {
        let h1 = session_hash("cli:default");
        let h2 = session_hash("cli:default");
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 8);
    }

    #[test]
    fn test_session_hash_matches_working_memory() {
        // Must produce the same hash as WorkingMemoryStore::session_hash.
        use crate::agent::working_memory::WorkingMemoryStore;
        let our_hash = session_hash("cli:test");
        let wm_hash = WorkingMemoryStore::session_hash("cli:test");
        assert_eq!(our_hash, wm_hash);
    }

    // --- index_sessions (filesystem integration tests) ---

    #[test]
    fn test_index_creates_md_files() {
        let sessions_dir = tempdir().unwrap();
        let memory_dir = tempdir().unwrap();

        // Write a JSONL file.
        let jsonl = r#"{"_type":"metadata","session_key":"cli:test","created_at":"2026-01-01T00:00:00Z","updated_at":"2026-01-01T01:00:00Z"}
{"role":"user","content":"Hello world"}
{"role":"assistant","content":"Hi there!"}"#;
        fs::write(sessions_dir.path().join("cli_test_2026-01-01.jsonl"), jsonl).unwrap();

        let (indexed, skipped, errors) =
            index_sessions(sessions_dir.path(), memory_dir.path());
        assert_eq!(indexed, 1);
        assert_eq!(errors, 0);

        // Verify the .md file was created.
        let hash = session_hash("cli:test");
        let md_path = memory_dir.path().join(format!("SESSION_{}.md", hash));
        assert!(md_path.exists(), "SESSION_{}.md should exist", hash);

        let content = fs::read_to_string(&md_path).unwrap();
        assert!(content.contains("session_key: \"cli:test\""));
        assert!(content.contains("status: indexed"));
        assert!(content.contains("**user:** Hello world"));
        // skipped should be 0 since no pre-existing md
        assert_eq!(skipped, 0);
    }

    #[test]
    fn test_index_skips_existing_summaries() {
        let sessions_dir = tempdir().unwrap();
        let memory_dir = tempdir().unwrap();

        let jsonl = r#"{"_type":"metadata","session_key":"cli:existing","created_at":"2026-01-01T00:00:00Z"}
{"role":"user","content":"test"}"#;
        fs::write(
            sessions_dir.path().join("cli_existing_2026-01-01.jsonl"),
            jsonl,
        )
        .unwrap();

        // Pre-create the SESSION_*.md.
        let hash = session_hash("cli:existing");
        fs::write(
            memory_dir.path().join(format!("SESSION_{}.md", hash)),
            "existing content",
        )
        .unwrap();

        let (indexed, skipped, _errors) =
            index_sessions(sessions_dir.path(), memory_dir.path());
        assert_eq!(indexed, 0);
        assert_eq!(skipped, 1);
    }

    #[test]
    fn test_index_skips_test_rotate() {
        let sessions_dir = tempdir().unwrap();
        let memory_dir = tempdir().unwrap();

        fs::write(
            sessions_dir.path().join("test_rotate_001.jsonl"),
            r#"{"role":"user","content":"test"}"#,
        )
        .unwrap();

        let (indexed, skipped, _errors) =
            index_sessions(sessions_dir.path(), memory_dir.path());
        assert_eq!(indexed, 0);
        assert_eq!(skipped, 1);
    }

    #[test]
    fn test_index_handles_corrupt_jsonl() {
        let sessions_dir = tempdir().unwrap();
        let memory_dir = tempdir().unwrap();

        // File with only garbage — should skip (empty content).
        fs::write(
            sessions_dir.path().join("corrupt.jsonl"),
            "not json\nalso not json\n",
        )
        .unwrap();

        let (indexed, skipped, errors) =
            index_sessions(sessions_dir.path(), memory_dir.path());
        assert_eq!(indexed, 0);
        assert_eq!(errors, 0);
        assert_eq!(skipped, 1); // skipped because content is empty
    }

    #[test]
    fn test_index_nonexistent_sessions_dir() {
        let memory_dir = tempdir().unwrap();
        let fake = Path::new("/tmp/nanobot_nonexistent_sessions_xyz");
        let (indexed, _skipped, errors) = index_sessions(fake, memory_dir.path());
        assert_eq!(indexed, 0);
        assert_eq!(errors, 1);
    }
}
