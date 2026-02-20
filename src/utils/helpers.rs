#![allow(dead_code)]
//! Utility functions for nanobot.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Result};
use chrono::Local;

/// Ensure a directory exists, creating it if necessary.
pub fn ensure_dir(path: impl AsRef<Path>) -> PathBuf {
    let path = path.as_ref().to_path_buf();
    if !path.exists() {
        let _ = fs::create_dir_all(&path);
    }
    path
}

/// Move a file, falling back to copy+remove when rename cannot cross devices.
pub fn move_file(src: &Path, dst: &Path) -> Result<()> {
    if let Some(parent) = dst.parent() {
        ensure_dir(parent);
    }

    match fs::rename(src, dst) {
        Ok(()) => Ok(()),
        Err(e) if is_cross_device_error(&e) => {
            fs::copy(src, dst)?;
            fs::remove_file(src)?;
            Ok(())
        }
        Err(e) => Err(e.into()),
    }
}

fn is_cross_device_error(err: &io::Error) -> bool {
    // EXDEV on Unix-like systems (Linux/macOS).
    err.raw_os_error() == Some(18)
}

/// Get the nanobot data directory (~/.nanobot).
///
/// Migrates from the old `~/.nanoclaw` directory if it exists and
/// `~/.nanobot` does not.
pub fn get_data_path() -> PathBuf {
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    let new_path = home.join(".nanobot");
    let old_path = home.join(".nanoclaw");
    if !new_path.exists() && old_path.exists() {
        if fs::rename(&old_path, &new_path).is_ok() {
            eprintln!("Migrated ~/.nanoclaw -> ~/.nanobot");
        }
    }
    ensure_dir(new_path)
}

/// Get the workspace path.
///
/// If `workspace` is provided, it is used (with `~` expansion).
/// Otherwise defaults to `~/.nanobot/workspace`.
pub fn get_workspace_path(workspace: Option<&str>) -> PathBuf {
    let path = match workspace {
        Some(ws) => expand_tilde(ws),
        None => {
            let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
            home.join(".nanobot").join("workspace")
        }
    };
    ensure_dir(path)
}

/// Get the sessions storage directory.
pub fn get_sessions_path() -> PathBuf {
    ensure_dir(get_data_path().join("sessions"))
}

/// Get the memory directory within the workspace.
pub fn get_memory_path(workspace: Option<&Path>) -> PathBuf {
    let ws = match workspace {
        Some(p) => p.to_path_buf(),
        None => get_workspace_path(None),
    };
    ensure_dir(ws.join("memory"))
}

/// Get the skills directory within the workspace.
pub fn get_skills_path(workspace: Option<&Path>) -> PathBuf {
    let ws = match workspace {
        Some(p) => p.to_path_buf(),
        None => get_workspace_path(None),
    };
    ensure_dir(ws.join("skills"))
}

/// Get today's date in YYYY-MM-DD format.
pub fn today_date() -> String {
    Local::now().format("%Y-%m-%d").to_string()
}

/// Get current timestamp in ISO format.
pub fn timestamp() -> String {
    Local::now().to_rfc3339()
}

/// Find the largest byte index `<= idx` that lies on a UTF-8 char boundary.
///
/// Equivalent to the nightly `str::floor_char_boundary`.
pub fn floor_char_boundary(s: &str, idx: usize) -> usize {
    if idx >= s.len() {
        return s.len();
    }
    let mut i = idx;
    while !s.is_char_boundary(i) && i > 0 {
        i -= 1;
    }
    i
}

/// Truncate a string to max length, adding a suffix if truncated.
pub fn truncate_string(s: &str, max_len: usize) -> String {
    let suffix = "...";
    if s.len() <= max_len {
        return s.to_string();
    }
    if max_len <= suffix.len() {
        let end = floor_char_boundary(s, max_len);
        return s[..end].to_string();
    }
    let end = floor_char_boundary(s, max_len - suffix.len());
    let mut result = s[..end].to_string();
    result.push_str(suffix);
    result
}

/// Convert a string to a safe filename by replacing unsafe characters with underscores.
pub fn safe_filename(name: &str) -> String {
    const UNSAFE_CHARS: &[char] = &['<', '>', ':', '"', '/', '\\', '|', '?', '*'];
    let mut result = name.to_string();
    for &ch in UNSAFE_CHARS {
        result = result.replace(ch, "_");
    }
    result.trim().to_string()
}

/// Parse a session key into (channel, chat_id).
///
/// The key must be in the format `"channel:chat_id"`.
pub fn parse_session_key(key: &str) -> Result<(String, String)> {
    match key.split_once(':') {
        Some((channel, chat_id)) => Ok((channel.to_string(), chat_id.to_string())),
        None => Err(anyhow!("Invalid session key: {}", key)),
    }
}

/// Expand a leading `~` to the user's home directory.
fn expand_tilde(path: &str) -> PathBuf {
    if let Some(rest) = path.strip_prefix("~/") {
        let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        home.join(rest)
    } else if path == "~" {
        dirs::home_dir().unwrap_or_else(|| PathBuf::from("."))
    } else {
        PathBuf::from(path)
    }
}

/// Append a JSON event as a single line to `events.jsonl` in the given workspace.
pub fn append_jsonl_event(workspace: &Path, event: &serde_json::Value) {
    use std::io::Write;
    let event_path = workspace.join("events.jsonl");
    let line = format!("{}\n", event);
    match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&event_path)
    {
        Ok(mut f) => {
            if let Err(e) = f.write_all(line.as_bytes()) {
                tracing::warn!("Failed to append event: {}", e);
            }
        }
        Err(e) => tracing::warn!("Failed to open events.jsonl: {}", e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_short_string() {
        assert_eq!(truncate_string("hello", 10), "hello");
    }

    #[test]
    fn test_truncate_long_string() {
        assert_eq!(truncate_string("hello world", 8), "hello...");
    }

    #[test]
    fn test_safe_filename() {
        assert_eq!(safe_filename("hello:world"), "hello_world");
        assert_eq!(safe_filename("a<b>c"), "a_b_c");
    }

    #[test]
    fn test_move_file_basic() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("a.txt");
        let dst = dir.path().join("nested").join("b.txt");
        fs::write(&src, "hello").unwrap();

        move_file(&src, &dst).unwrap();
        assert!(!src.exists());
        assert!(dst.exists());
        assert_eq!(fs::read_to_string(&dst).unwrap(), "hello");
    }

    #[test]
    fn test_parse_session_key_valid() {
        let (ch, id) = parse_session_key("telegram:12345").unwrap();
        assert_eq!(ch, "telegram");
        assert_eq!(id, "12345");
    }

    #[test]
    fn test_parse_session_key_with_colons() {
        let (ch, id) = parse_session_key("whatsapp:+1:234").unwrap();
        assert_eq!(ch, "whatsapp");
        assert_eq!(id, "+1:234");
    }

    #[test]
    fn test_parse_session_key_invalid() {
        assert!(parse_session_key("nodelimiter").is_err());
    }

    #[test]
    fn test_today_date_format() {
        let d = today_date();
        // Should be YYYY-MM-DD
        assert_eq!(d.len(), 10);
        assert_eq!(&d[4..5], "-");
        assert_eq!(&d[7..8], "-");
    }

    #[test]
    fn test_expand_tilde() {
        let p = expand_tilde("~/foo/bar");
        assert!(p.ends_with("foo/bar"));
        assert!(!p.to_string_lossy().contains('~'));
    }

    #[test]
    fn test_floor_char_boundary_ascii() {
        assert_eq!(floor_char_boundary("hello", 3), 3);
        assert_eq!(floor_char_boundary("hello", 10), 5);
        assert_eq!(floor_char_boundary("hello", 0), 0);
    }

    #[test]
    fn test_floor_char_boundary_multibyte() {
        // "héllo" — 'é' is 2 bytes (0xC3 0xA9), so byte indices 1..=2
        let s = "héllo";
        assert_eq!(s.len(), 6); // h(1) é(2) l(1) l(1) o(1)
        assert_eq!(floor_char_boundary(s, 2), 1); // byte 2 is mid-char, floor to 1
        assert_eq!(floor_char_boundary(s, 3), 3); // byte 3 is start of 'l'
    }

    #[test]
    fn test_floor_char_boundary_emoji() {
        // "a😀b" — '😀' is 4 bytes
        let s = "a😀b";
        assert_eq!(s.len(), 6); // a(1) 😀(4) b(1)
        assert_eq!(floor_char_boundary(s, 1), 1); // end of 'a'
        assert_eq!(floor_char_boundary(s, 2), 1); // mid-emoji, floor to 'a'
        assert_eq!(floor_char_boundary(s, 3), 1);
        assert_eq!(floor_char_boundary(s, 4), 1);
        assert_eq!(floor_char_boundary(s, 5), 5); // start of 'b'
    }

    #[test]
    fn test_truncate_string_multibyte() {
        // Should not panic on multi-byte strings
        let s = "café résumé";
        let t = truncate_string(s, 6);
        assert!(t.len() <= 9); // 6 + "..." at most
        assert!(t.ends_with("..."));
    }
}
