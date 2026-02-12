//! Shared scratchpad for inter-agent communication.
//!
//! A simple key-value store backed by markdown files in
//! `{workspace}/scratchpad/{key}.md`. Agents can write, read, append,
//! list, and delete entries.

use std::fs;
use std::path::{Path, PathBuf};

use crate::utils::helpers::ensure_dir;

/// Shared scratchpad backed by filesystem.
pub struct SharedScratchpad {
    dir: PathBuf,
}

impl SharedScratchpad {
    pub fn new(workspace: &Path) -> Self {
        Self {
            dir: workspace.join("scratchpad"),
        }
    }

    /// Write content to a key (overwrites existing).
    pub fn write(&self, key: &str, content: &str) -> String {
        ensure_dir(&self.dir);
        let path = self.key_path(key);
        match fs::write(&path, content) {
            Ok(_) => format!("Written to '{}'", key),
            Err(e) => format!("Error: Failed to write '{}': {}", key, e),
        }
    }

    /// Read content from a key.
    pub fn read(&self, key: &str) -> Option<String> {
        fs::read_to_string(self.key_path(key)).ok()
    }

    /// Append content to a key (creates if missing).
    pub fn append(&self, key: &str, content: &str) -> String {
        ensure_dir(&self.dir);
        let path = self.key_path(key);
        let existing = fs::read_to_string(&path).unwrap_or_default();
        let new_content = if existing.is_empty() {
            content.to_string()
        } else {
            format!("{}\n{}", existing, content)
        };
        match fs::write(&path, new_content) {
            Ok(_) => format!("Appended to '{}'", key),
            Err(e) => format!("Error: Failed to append to '{}': {}", key, e),
        }
    }

    /// List all keys.
    pub fn list(&self) -> Vec<String> {
        if !self.dir.is_dir() {
            return Vec::new();
        }

        let mut keys = Vec::new();
        if let Ok(entries) = fs::read_dir(&self.dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("md") {
                    if let Some(name) = path.file_stem().and_then(|n| n.to_str()) {
                        keys.push(name.to_string());
                    }
                }
            }
        }
        keys.sort();
        keys
    }

    /// Delete a key.
    pub fn delete(&self, key: &str) -> String {
        let path = self.key_path(key);
        if !path.is_file() {
            return format!("Error: Key '{}' not found", key);
        }
        match fs::remove_file(&path) {
            Ok(_) => format!("Deleted '{}'", key),
            Err(e) => format!("Error: Failed to delete '{}': {}", key, e),
        }
    }

    fn key_path(&self, key: &str) -> PathBuf {
        // Sanitize key for filesystem.
        let safe_key: String = key
            .chars()
            .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
            .collect();
        self.dir.join(format!("{}.md", safe_key))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_and_read() {
        let dir = tempfile::tempdir().unwrap();
        let pad = SharedScratchpad::new(dir.path());

        let result = pad.write("notes", "Hello world");
        assert!(result.contains("Written"));

        let content = pad.read("notes").unwrap();
        assert_eq!(content, "Hello world");
    }

    #[test]
    fn test_read_missing() {
        let dir = tempfile::tempdir().unwrap();
        let pad = SharedScratchpad::new(dir.path());

        assert!(pad.read("nonexistent").is_none());
    }

    #[test]
    fn test_append() {
        let dir = tempfile::tempdir().unwrap();
        let pad = SharedScratchpad::new(dir.path());

        pad.write("log", "Line 1");
        pad.append("log", "Line 2");

        let content = pad.read("log").unwrap();
        assert!(content.contains("Line 1"));
        assert!(content.contains("Line 2"));
    }

    #[test]
    fn test_append_creates() {
        let dir = tempfile::tempdir().unwrap();
        let pad = SharedScratchpad::new(dir.path());

        pad.append("new_key", "First entry");
        let content = pad.read("new_key").unwrap();
        assert_eq!(content, "First entry");
    }

    #[test]
    fn test_list() {
        let dir = tempfile::tempdir().unwrap();
        let pad = SharedScratchpad::new(dir.path());

        pad.write("alpha", "a");
        pad.write("beta", "b");

        let keys = pad.list();
        assert_eq!(keys, vec!["alpha", "beta"]);
    }

    #[test]
    fn test_delete() {
        let dir = tempfile::tempdir().unwrap();
        let pad = SharedScratchpad::new(dir.path());

        pad.write("temp", "temporary");
        assert!(pad.read("temp").is_some());

        let result = pad.delete("temp");
        assert!(result.contains("Deleted"));
        assert!(pad.read("temp").is_none());
    }

    #[test]
    fn test_delete_missing() {
        let dir = tempfile::tempdir().unwrap();
        let pad = SharedScratchpad::new(dir.path());

        let result = pad.delete("nonexistent");
        assert!(result.contains("Error"));
    }

    #[test]
    fn test_key_sanitization() {
        let dir = tempfile::tempdir().unwrap();
        let pad = SharedScratchpad::new(dir.path());

        pad.write("my/special key!", "content");
        // The key gets sanitized, so we need to use the sanitized version.
        let keys = pad.list();
        assert_eq!(keys.len(), 1);
    }
}
