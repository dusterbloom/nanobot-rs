//! Memory system for persistent agent memory.
//!
//! `memory/MEMORY.md` contains curated cross-session facts. Session-scoped
//! working memory lives in SQLite and is handled by `WorkingMemoryStore`.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::utils::helpers::ensure_dir;

/// Serialize process-local read/modify/write transactions on `MEMORY.md`.
///
/// Reflection holds this across model inference so a manual `remember` update
/// cannot be based on, or overwritten by, a stale snapshot of curated memory.
pub(crate) fn memory_transaction_lock() -> &'static tokio::sync::Mutex<()> {
    static LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| tokio::sync::Mutex::new(()))
}

/// Persistent memory store for the agent.
pub struct MemoryStore {
    /// Path to the long-term memory file.
    memory_file: PathBuf,
}

impl MemoryStore {
    /// Create a new `MemoryStore` for the given workspace.
    pub fn new(workspace: &Path) -> Self {
        let memory_dir = ensure_dir(workspace.join("memory"));
        let memory_file = memory_dir.join("MEMORY.md");
        Self { memory_file }
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

    /// Append new facts under the existing content, skipping facts already
    /// present (normalized comparison). Returns how many were appended.
    ///
    /// Dream consolidation (agent::dream) uses this — unlike the reflector's
    /// full rewrite, appending cannot lose concurrent curation, and dedup
    /// makes repeated dreams idempotent.
    pub fn append_long_term_facts(&self, facts: &[String]) -> usize {
        let existing = self.read_long_term();
        let existing_norm: std::collections::HashSet<String> = existing
            .lines()
            .map(|l| l.trim().trim_start_matches("- ").trim().to_lowercase())
            .collect();
        let fresh: Vec<&String> = facts
            .iter()
            .filter(|f| {
                let t = f.trim();
                !t.is_empty() && !existing_norm.contains(&t.to_lowercase())
            })
            .collect();
        if fresh.is_empty() {
            return 0;
        }
        let mut next = existing;
        if !next.is_empty() && !next.ends_with('\n') {
            next.push('\n');
        }
        for fact in &fresh {
            next.push_str("- ");
            next.push_str(fact.trim());
            next.push('\n');
        }
        self.write_long_term(&next);
        fresh.len()
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
        let memory_dir = store.memory_file.parent().unwrap();
        assert!(memory_dir.exists(), "memory directory should be created");
        assert_eq!(memory_dir, tmp.path().join("memory"));
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
}
