//! Memory system for persistent agent memory.
//!
//! Supports daily notes (`memory/YYYY-MM-DD.md`) and long-term memory (`MEMORY.md`).

use std::fs;
use std::path::{Path, PathBuf};

use chrono::{Local, NaiveDate};

use crate::utils::helpers::{ensure_dir, today_date};

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

    /// Get path to today's memory file.
    pub fn get_today_file(&self) -> PathBuf {
        self.memory_dir.join(format!("{}.md", today_date()))
    }

    /// Read today's memory notes. Returns empty string if no file exists.
    pub fn read_today(&self) -> String {
        let today_file = self.get_today_file();
        if today_file.exists() {
            fs::read_to_string(&today_file).unwrap_or_default()
        } else {
            String::new()
        }
    }

    /// Append content to today's memory notes.
    ///
    /// Creates the file with a date header if it does not exist yet.
    pub fn append_today(&self, content: &str) {
        let today_file = self.get_today_file();

        let full_content = if today_file.exists() {
            let existing = fs::read_to_string(&today_file).unwrap_or_default();
            format!("{}\n{}", existing, content)
        } else {
            let header = format!("# {}\n\n", today_date());
            format!("{}{}", header, content)
        };

        let _ = fs::write(&today_file, full_content);
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
    pub fn write_long_term(&self, content: &str) {
        let _ = fs::write(&self.memory_file, content);
    }

    /// Get memories from the last N days, concatenated with separators.
    pub fn get_recent_memories(&self, days: u32) -> String {
        let today = Local::now().date_naive();
        let mut memories: Vec<String> = Vec::new();

        for i in 0..days {
            let date = today - chrono::Duration::days(i64::from(i));
            let date_str = date.format("%Y-%m-%d").to_string();
            let file_path = self.memory_dir.join(format!("{}.md", date_str));

            if file_path.exists() {
                if let Ok(content) = fs::read_to_string(&file_path) {
                    memories.push(content);
                }
            }
        }

        memories.join("\n\n---\n\n")
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

    /// Get memory context for the agent prompt.
    ///
    /// Combines long-term memory and today's notes.
    pub fn get_memory_context(&self) -> String {
        let mut parts: Vec<String> = Vec::new();

        let long_term = self.read_long_term();
        if !long_term.is_empty() {
            parts.push(format!("## Long-term Memory\n{}", long_term));
        }

        let today = self.read_today();
        if !today.is_empty() {
            parts.push(format!("## Today's Notes\n{}", today));
        }

        if parts.is_empty() {
            String::new()
        } else {
            parts.join("\n\n")
        }
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

    // ----- append_today / read_today -----

    #[test]
    fn test_read_today_empty_initially() {
        let (_tmp, store) = make_store();
        assert_eq!(store.read_today(), "");
    }

    #[test]
    fn test_append_today_creates_file_with_header() {
        let (_tmp, store) = make_store();
        store.append_today("Did something important.");
        let content = store.read_today();
        let today_str = today_date();
        assert!(
            content.contains(&today_str),
            "today file should contain today's date in its header"
        );
        assert!(content.contains("Did something important."));
    }

    #[test]
    fn test_append_today_appends() {
        let (_tmp, store) = make_store();
        store.append_today("Line 1");
        store.append_today("Line 2");
        let content = store.read_today();
        assert!(content.contains("Line 1"));
        assert!(content.contains("Line 2"));
    }

    // ----- get_memory_context -----

    #[test]
    fn test_get_memory_context_empty() {
        let (_tmp, store) = make_store();
        assert_eq!(store.get_memory_context(), "");
    }

    #[test]
    fn test_get_memory_context_includes_long_term() {
        let (_tmp, store) = make_store();
        store.write_long_term("Likes cats.");
        let ctx = store.get_memory_context();
        assert!(ctx.contains("Long-term Memory"));
        assert!(ctx.contains("Likes cats."));
    }

    #[test]
    fn test_get_memory_context_includes_today() {
        let (_tmp, store) = make_store();
        store.append_today("Deployed v2.");
        let ctx = store.get_memory_context();
        assert!(ctx.contains("Today's Notes"));
        assert!(ctx.contains("Deployed v2."));
    }

    #[test]
    fn test_get_memory_context_combines_both() {
        let (_tmp, store) = make_store();
        store.write_long_term("Long-term note");
        store.append_today("Daily note");
        let ctx = store.get_memory_context();
        assert!(ctx.contains("Long-term Memory"));
        assert!(ctx.contains("Long-term note"));
        assert!(ctx.contains("Today's Notes"));
        assert!(ctx.contains("Daily note"));
    }

    // ----- get_recent_memories -----

    #[test]
    fn test_get_recent_memories_empty() {
        let (_tmp, store) = make_store();
        assert_eq!(store.get_recent_memories(7), "");
    }

    #[test]
    fn test_get_recent_memories_includes_today() {
        let (_tmp, store) = make_store();
        store.append_today("Today's work");
        let recent = store.get_recent_memories(1);
        assert!(
            recent.contains("Today's work"),
            "get_recent_memories(1) should include today's file"
        );
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

    // ----- get_today_file -----

    #[test]
    fn test_get_today_file_path() {
        let (_tmp, store) = make_store();
        let today_file = store.get_today_file();
        let expected_name = format!("{}.md", today_date());
        assert!(today_file.ends_with(&expected_name));
    }
}
