//! Observation store for cross-session memory.
//!
//! Observations are LLM-generated summaries of conversations, created during
//! context compaction. They provide cross-session awareness — the agent can
//! recall key facts from past conversations without replaying full history.
//!
//! File format: `{workspace}/memory/observations/{timestamp}_{session_key}.md`

use std::fs;
use std::path::{Path, PathBuf};

use crate::agent::token_budget::TokenBudget;
use crate::utils::helpers::{ensure_dir, move_file};

/// A single observation (loaded from disk).
pub struct Observation {
    pub timestamp: String,
    pub session_key: String,
    pub channel: Option<String>,
    pub content: String,
    pub path: PathBuf,
}

/// Persistent store for conversation observations.
pub struct ObservationStore {
    observations_dir: PathBuf,
    archived_dir: PathBuf,
}

impl ObservationStore {
    /// Create a new observation store for the given workspace.
    pub fn new(workspace: &Path) -> Self {
        let observations_dir = workspace.join("memory").join("observations");
        let archived_dir = observations_dir.join("archived");
        Self {
            observations_dir,
            archived_dir,
        }
    }

    /// Load recent observations, newest first.
    pub fn load_recent(&self, max_count: usize) -> Vec<Observation> {
        let mut files = self.list_observation_files();
        // Sort by filename descending (timestamps sort lexicographically).
        files.sort_by(|a, b| b.cmp(a));
        files.truncate(max_count);

        files
            .into_iter()
            .filter_map(|path| self.parse_observation(&path))
            .collect()
    }

    /// Get observation context for system prompt injection, respecting a token budget.
    pub fn get_context(&self, max_tokens: usize) -> String {
        let observations = self.load_recent(50); // load up to 50 recent
        if observations.is_empty() {
            return String::new();
        }

        let mut parts: Vec<String> = Vec::new();
        let mut total_tokens = 0;

        for obs in &observations {
            let entry = format!(
                "**[{}]** ({})\n{}",
                obs.timestamp, obs.session_key, obs.content,
            );
            let entry_tokens = TokenBudget::estimate_str_tokens(&entry);
            if total_tokens + entry_tokens > max_tokens {
                break;
            }
            parts.push(entry);
            total_tokens += entry_tokens;
        }

        parts.join("\n\n")
    }

    /// Count observation files on disk.
    pub fn count(&self) -> usize {
        self.list_observation_files().len()
    }

    /// Estimate total tokens across all observations.
    pub fn total_tokens(&self) -> usize {
        let files = self.list_observation_files();
        let mut total = 0;
        for path in &files {
            if let Ok(content) = fs::read_to_string(path) {
                total += TokenBudget::estimate_str_tokens(&content);
            }
        }
        total
    }

    /// Archive processed observations (move to archived/ subdirectory).
    pub fn archive(&self, paths: &[PathBuf]) -> anyhow::Result<()> {
        ensure_dir(&self.archived_dir);
        for path in paths {
            if !path.exists() {
                continue;
            }
            if let Some(filename) = path.file_name() {
                let dest = self.archived_dir.join(filename);
                move_file(path, &dest)?;
            }
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    /// List all `.md` files in the observations directory.
    fn list_observation_files(&self) -> Vec<PathBuf> {
        if !self.observations_dir.exists() {
            return Vec::new();
        }
        fs::read_dir(&self.observations_dir)
            .into_iter()
            .flatten()
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.extension().and_then(|e| e.to_str()) == Some("md")
                    && p.file_name()
                        .and_then(|n| n.to_str())
                        .map(|n| n != "archived")
                        .unwrap_or(true)
            })
            .collect()
    }

    /// Parse an observation file into an `Observation`.
    fn parse_observation(&self, path: &PathBuf) -> Option<Observation> {
        let raw = fs::read_to_string(path).ok()?;

        // Parse YAML frontmatter.
        let mut timestamp = String::new();
        let mut session_key = String::new();
        let mut channel = None;
        let mut content = raw.clone();

        if raw.starts_with("---") {
            if let Some(end) = raw[3..].find("---") {
                let frontmatter = &raw[3..3 + end];
                content = raw[3 + end + 3..].trim().to_string();

                for line in frontmatter.lines() {
                    let line = line.trim();
                    if let Some(val) = line.strip_prefix("timestamp:") {
                        timestamp = val.trim().to_string();
                    } else if let Some(val) = line.strip_prefix("session:") {
                        session_key = val.trim().to_string();
                    } else if let Some(val) = line.strip_prefix("channel:") {
                        channel = Some(val.trim().to_string());
                    }
                }
            }
        }

        Some(Observation {
            timestamp,
            session_key,
            channel,
            content,
            path: path.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_store() -> (TempDir, ObservationStore) {
        let tmp = TempDir::new().unwrap();
        let store = ObservationStore::new(tmp.path());
        (tmp, store)
    }

    #[test]
    fn test_load_recent_returns_newest_first() {
        let (_tmp, store) = make_store();
        // Create observations directory and write files with known timestamps.
        ensure_dir(&store.observations_dir);
        fs::write(
            store.observations_dir.join("20260101T000000Z_a.md"),
            "---\ntimestamp: 2026-01-01T00:00:00Z\nsession: a\n---\n\nOld observation",
        )
        .unwrap();
        fs::write(
            store.observations_dir.join("20260201T000000Z_b.md"),
            "---\ntimestamp: 2026-02-01T00:00:00Z\nsession: b\n---\n\nNew observation",
        )
        .unwrap();

        let observations = store.load_recent(10);
        assert_eq!(observations.len(), 2);
        assert_eq!(observations[0].session_key, "b");
        assert_eq!(observations[1].session_key, "a");
    }

    #[test]
    fn test_get_context_respects_token_limit() {
        let (_tmp, store) = make_store();
        ensure_dir(&store.observations_dir);
        // Create a large observation.
        let big_content = "x".repeat(10000);
        fs::write(
            store.observations_dir.join("20260101T000000Z_big.md"),
            format!(
                "---\ntimestamp: 2026-01-01T00:00:00Z\nsession: big\n---\n\n{}",
                big_content
            ),
        )
        .unwrap();

        // Request a very small token budget — should get empty or truncated.
        let context = store.get_context(10);
        let tokens = TokenBudget::estimate_str_tokens(&context);
        // Should respect the budget (may be empty if the single entry exceeds budget).
        assert!(tokens <= 10 || context.is_empty());
    }

    #[test]
    fn test_total_tokens_sums_all_observations() {
        let (_tmp, store) = make_store();
        ensure_dir(&store.observations_dir);
        fs::write(
            store.observations_dir.join("20260101T000000Z_a.md"),
            "Hello world",
        )
        .unwrap();
        fs::write(
            store.observations_dir.join("20260102T000000Z_b.md"),
            "Another observation",
        )
        .unwrap();

        let total = store.total_tokens();
        assert!(total > 0);
    }

    #[test]
    fn test_archive_moves_files() {
        let (_tmp, store) = make_store();
        ensure_dir(&store.observations_dir);
        let file_path = store.observations_dir.join("20260101T000000Z_test.md");
        fs::write(&file_path, "test content").unwrap();
        assert!(file_path.exists());

        store.archive(&[file_path.clone()]).unwrap();
        assert!(!file_path.exists());
        assert!(store.archived_dir.join("20260101T000000Z_test.md").exists());
    }
}
