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
