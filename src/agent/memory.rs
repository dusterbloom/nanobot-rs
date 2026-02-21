#![allow(dead_code)]
//! Memory system for persistent agent memory.
//!
//! Supports daily notes (`memory/YYYY-MM-DD.md`) and long-term memory (`MEMORY.md`).

use std::fs;
use std::path::{Path, PathBuf};

use chrono::NaiveDate;

use crate::utils::helpers::ensure_dir;

/// Persistent memory store for the agent.
pub struct MemoryStore {
    /// Root workspace path.
    pub workspace: PathBuf,
    /// Directory that contains all memory files.
    pub memory_dir: PathBuf,
    /// Path to the long-term memory file.
    pub memory_file: PathBuf,
}

impl MemoryStore {
    /// Create a new `MemoryStore` for the given workspace.
    pub fn new(workspace: &Path) -> Self {
        let memory_dir = ensure_dir(workspace.join("memory"));
        let memory_file = memory_dir.join("MEMORY.md");
        Self {
            workspace: workspace.to_path_buf(),
            memory_dir,
            memory_file,
        }
    }

    /// Read long-term memory (`MEMORY.md`).
    pub fn read_long_term(&self) -> String {
        if self.memory_file.exists() {
            fs::read_to_string(&self.memory_file).unwrap_or_default()
        } else {
            String::new()
        }
    }

    /// Write to long-term memory (`MEMORY.md`), replacing existing content.
    ///
    /// Uses atomic write (temp file + rename) to avoid corruption on crash.
    pub fn write_long_term(&self, content: &str) {
        let tmp_path = self.memory_file.with_extension("md.tmp");
        if fs::write(&tmp_path, content).is_err() {
            return;
        }
        let _ = fs::rename(&tmp_path, &self.memory_file);
    }

    /// List all memory files sorted by date (newest first).
    pub fn list_memory_files(&self) -> Vec<PathBuf> {
        if !self.memory_dir.exists() {
            return Vec::new();
        }

        let mut files: Vec<PathBuf> = Vec::new();

        if let Ok(entries) = fs::read_dir(&self.memory_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    // Match YYYY-MM-DD.md pattern.
                    if name.len() == 13
                        && name.ends_with(".md")
                        && NaiveDate::parse_from_str(&name[..10], "%Y-%m-%d").is_ok()
                    {
                        files.push(path);
                    }
                }
            }
        }

        files.sort_by(|a, b| b.cmp(a));
        files
    }

    /// Read the most recent `n` daily notes, returning a combined string
    /// with date headers. Caps total output at approximately 200 tokens (~800 chars).
    pub fn read_recent_daily_notes(&self, n: usize) -> String {
        let files = self.list_memory_files(); // already sorted newest-first
        let mut result = String::new();
        let cap = 800; // ~200 tokens
        for path in files.into_iter().take(n) {
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                let date = &name[..10]; // YYYY-MM-DD
                if let Ok(content) = std::fs::read_to_string(&path) {
                    let entry = format!("### {}\n{}\n\n", date, content.trim());
                    if result.len() + entry.len() > cap {
                        // Add truncated remainder if there's room for at least the header
                        let remaining = cap.saturating_sub(result.len());
                        if remaining > 20 {
                            result.push_str(&entry[..remaining.min(entry.len())]);
                            result.push_str("...\n");
                        }
                        break;
                    }
                    result.push_str(&entry);
                }
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// Helper: create a MemoryStore backed by a temporary workspace.
    fn make_store() -> (TempDir, MemoryStore) {
        let tmp = TempDir::new().unwrap();
        let store = MemoryStore::new(tmp.path());
        (tmp, store)
    }

    // ----- construction -----

    #[test]
    fn test_new_creates_memory_dir() {
        let (tmp, store) = make_store();
        assert!(
            store.memory_dir.exists(),
            "memory directory should be created"
        );
        assert_eq!(store.memory_dir, tmp.path().join("memory"));
    }

    #[test]
    fn test_memory_file_path() {
        let (_tmp, store) = make_store();
        assert!(
            store.memory_file.ends_with("MEMORY.md"),
            "memory_file should point to MEMORY.md"
        );
    }

    // ----- write_long_term / read_long_term -----

    #[test]
    fn test_read_long_term_empty_initially() {
        let (_tmp, store) = make_store();
        assert_eq!(store.read_long_term(), "");
    }

    #[test]
    fn test_write_and_read_long_term_roundtrip() {
        let (_tmp, store) = make_store();
        store.write_long_term("User likes Rust.");
        assert_eq!(store.read_long_term(), "User likes Rust.");
    }

    #[test]
    fn test_write_long_term_overwrites() {
        let (_tmp, store) = make_store();
        store.write_long_term("first");
        store.write_long_term("second");
        assert_eq!(store.read_long_term(), "second");
    }

    // ----- list_memory_files -----

    #[test]
    fn test_list_memory_files_empty() {
        let (_tmp, store) = make_store();
        assert!(store.list_memory_files().is_empty());
    }

    #[test]
    fn test_list_memory_files_finds_dated_files() {
        let (_tmp, store) = make_store();
        // Create two dated files.
        fs::write(store.memory_dir.join("2025-01-01.md"), "jan1").unwrap();
        fs::write(store.memory_dir.join("2025-01-15.md"), "jan15").unwrap();
        // Also a non-date file that should be excluded.
        fs::write(store.memory_dir.join("MEMORY.md"), "long-term").unwrap();

        let files = store.list_memory_files();
        assert_eq!(files.len(), 2, "should find exactly two dated files");
    }

    #[test]
    fn test_list_memory_files_sorted_newest_first() {
        let (_tmp, store) = make_store();
        fs::write(store.memory_dir.join("2025-01-01.md"), "old").unwrap();
        fs::write(store.memory_dir.join("2025-06-15.md"), "new").unwrap();

        let files = store.list_memory_files();
        let names: Vec<String> = files
            .iter()
            .map(|f| f.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        assert_eq!(names[0], "2025-06-15.md");
        assert_eq!(names[1], "2025-01-01.md");
    }
}
