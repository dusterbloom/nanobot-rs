#![allow(dead_code)]
//! Memory bulletin — periodic briefing synthesized by the compaction model.
//!
//! Every N minutes (default 60), when the system is idle, the bulletin service
//! queries active working sessions + long-term MEMORY.md and produces a ~500
//! word briefing. The result is cached in an `ArcSwap<String>` for zero-cost
//! reads on every turn — injected into the system prompt alongside memory.

use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use arc_swap::ArcSwap;
use serde_json::json;
use tracing::{info, warn};

use crate::agent::memory::MemoryStore;
use crate::agent::working_memory::WorkingMemoryStore;
use crate::providers::base::LLMProvider;

/// Default bulletin refresh interval: 60 minutes.
pub const DEFAULT_BULLETIN_INTERVAL_S: u64 = 60 * 60;

/// Prompt for bulletin generation.
const BULLETIN_PROMPT: &str = "\
Synthesize a concise briefing (~300 words max) from the user's memory and recent sessions.

Structure:
1. **Active context** — what the user is currently working on (from active sessions)
2. **Key facts** — the most important long-term facts (from memory)
3. **Recent patterns** — any recurring themes or preferences

Be direct. No preamble. Bullet points preferred.

Long-term memory:
{memory}

Active sessions:
{sessions}";

/// Cached bulletin for zero-cost injection into system prompts.
pub struct BulletinCache {
    cached: Arc<ArcSwap<String>>,
}

impl BulletinCache {
    /// Create a new empty cache.
    pub fn new() -> Self {
        Self {
            cached: Arc::new(ArcSwap::from_pointee(String::new())),
        }
    }

    /// Read the current bulletin (cheap Arc load).
    pub fn read(&self) -> Arc<String> {
        self.cached.load_full()
    }

    /// Update the cached bulletin.
    pub fn update(&self, content: String) {
        self.cached.store(Arc::new(content));
    }

    /// Get a cloneable handle for sharing across tasks.
    pub fn handle(&self) -> Arc<ArcSwap<String>> {
        self.cached.clone()
    }
}

/// Build the bulletin prompt from current memory state.
///
/// Pure function — no I/O, no LLM. Testable with synthetic data.
pub fn build_bulletin_prompt(
    long_term_memory: &str,
    active_sessions: &[(String, String)],
) -> String {
    let sessions_text = if active_sessions.is_empty() {
        "No active sessions.".to_string()
    } else {
        active_sessions
            .iter()
            .map(|(key, content)| format!("**{}**\n{}", key, content))
            .collect::<Vec<_>>()
            .join("\n\n")
    };

    BULLETIN_PROMPT
        .replace("{memory}", long_term_memory)
        .replace("{sessions}", &sessions_text)
}

/// Generate a bulletin using the compaction model.
///
/// Reads memory + active sessions, calls the LLM, returns the briefing text.
pub async fn generate_bulletin(
    provider: &dyn LLMProvider,
    model: &str,
    workspace: &Path,
) -> Result<String> {
    let memory_store = MemoryStore::new(workspace);
    let wm_store = WorkingMemoryStore::new(workspace);

    let long_term = memory_store.read_long_term();
    let active = wm_store.list_active();
    let sessions: Vec<(String, String)> = active
        .iter()
        .map(|s| (s.session_key.clone(), s.content.clone()))
        .collect();

    let prompt = build_bulletin_prompt(&long_term, &sessions);

    let messages = vec![
        json!({"role": "system", "content": "You are a memory briefing assistant. Be concise and factual."}),
        json!({"role": "user", "content": prompt}),
    ];

    let response = provider
        .chat(&messages, None, Some(model), 1024, 0.3, None, None)
        .await?;

    response
        .content
        .ok_or_else(|| anyhow::anyhow!("Bulletin generation returned no content"))
}

/// Refresh the bulletin cache if the compaction model is available.
///
/// Called from the heartbeat loop. Writes result to cache and to
/// `{workspace}/BULLETIN.md` for persistence across restarts.
pub async fn refresh_bulletin(
    provider: &dyn LLMProvider,
    model: &str,
    workspace: &Path,
    cache: &Arc<ArcSwap<String>>,
) -> Result<()> {
    let bulletin = generate_bulletin(provider, model, workspace).await?;

    // Update in-memory cache.
    cache.store(Arc::new(bulletin.clone()));

    // Persist to disk for warm starts.
    let path = workspace.join("BULLETIN.md");
    if let Err(e) = std::fs::write(&path, &bulletin) {
        warn!("Failed to write BULLETIN.md: {}", e);
    }

    info!("Memory bulletin refreshed ({} chars)", bulletin.len());
    Ok(())
}

/// Load persisted bulletin from disk (warm start after restart).
pub fn load_persisted_bulletin(workspace: &Path) -> Option<String> {
    let path = workspace.join("BULLETIN.md");
    std::fs::read_to_string(&path)
        .ok()
        .filter(|s| !s.trim().is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::base::{LLMProvider, LLMResponse};
    use async_trait::async_trait;
    use serde_json::Value;
    use tempfile::TempDir;

    // ---------------------------------------------------------------
    // build_bulletin_prompt (pure function tests)
    // ---------------------------------------------------------------

    #[test]
    fn test_prompt_no_sessions() {
        let prompt = build_bulletin_prompt("User prefers dark mode", &[]);
        assert!(prompt.contains("User prefers dark mode"));
        assert!(prompt.contains("No active sessions"));
    }

    #[test]
    fn test_prompt_with_sessions() {
        let sessions = vec![
            ("cli:default".into(), "Working on auth refactor".into()),
            ("telegram:123".into(), "Discussing API design".into()),
        ];
        let prompt = build_bulletin_prompt("", &sessions);
        assert!(prompt.contains("cli:default"));
        assert!(prompt.contains("Working on auth refactor"));
        assert!(prompt.contains("telegram:123"));
    }

    #[test]
    fn test_prompt_structure() {
        let prompt = build_bulletin_prompt("fact1", &[("s:1".into(), "content".into())]);
        // Should contain the template markers replaced
        assert!(!prompt.contains("{memory}"));
        assert!(!prompt.contains("{sessions}"));
    }

    // ---------------------------------------------------------------
    // BulletinCache tests
    // ---------------------------------------------------------------

    #[test]
    fn test_cache_starts_empty() {
        let cache = BulletinCache::new();
        assert!(cache.read().is_empty());
    }

    #[test]
    fn test_cache_update_and_read() {
        let cache = BulletinCache::new();
        cache.update("Briefing content here".into());
        assert_eq!(&*cache.read(), "Briefing content here");
    }

    #[test]
    fn test_cache_overwrite() {
        let cache = BulletinCache::new();
        cache.update("old".into());
        cache.update("new".into());
        assert_eq!(&*cache.read(), "new");
    }

    #[test]
    fn test_cache_handle_shared() {
        let cache = BulletinCache::new();
        let handle = cache.handle();
        cache.update("shared".into());
        assert_eq!(&*handle.load_full(), "shared");
    }

    // ---------------------------------------------------------------
    // Persistence tests
    // ---------------------------------------------------------------

    #[test]
    fn test_load_persisted_missing() {
        let tmp = TempDir::new().unwrap();
        assert!(load_persisted_bulletin(tmp.path()).is_none());
    }

    #[test]
    fn test_load_persisted_empty() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("BULLETIN.md"), "  \n").unwrap();
        assert!(load_persisted_bulletin(tmp.path()).is_none());
    }

    #[test]
    fn test_load_persisted_content() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("BULLETIN.md"), "Briefing text").unwrap();
        assert_eq!(
            load_persisted_bulletin(tmp.path()).unwrap(),
            "Briefing text"
        );
    }

    // ---------------------------------------------------------------
    // Integration: generate_bulletin with mock provider
    // ---------------------------------------------------------------

    struct MockBulletinProvider;

    #[async_trait]
    impl LLMProvider for MockBulletinProvider {
        async fn chat(
            &self,
            _messages: &[Value],
            _tools: Option<&[Value]>,
            _model: Option<&str>,
            _max_tokens: u32,
            _temperature: f64,
            _thinking_budget: Option<u32>,
            _top_p: Option<f64>,
        ) -> anyhow::Result<LLMResponse> {
            Ok(LLMResponse {
                content: Some("- User works on nanobot\n- Prefers Rust".into()),
                tool_calls: vec![],
                finish_reason: "stop".into(),
                usage: std::collections::HashMap::new(),
            })
        }

        fn get_default_model(&self) -> &str {
            "mock"
        }
    }

    #[tokio::test]
    async fn test_generate_bulletin_with_mock() {
        let tmp = TempDir::new().unwrap();
        // Write some memory
        let mem_path = tmp.path().join("memory");
        std::fs::create_dir_all(&mem_path).unwrap();
        std::fs::write(mem_path.join("MEMORY.md"), "User likes Rust").unwrap();

        let provider = MockBulletinProvider;
        let result = generate_bulletin(&provider, "test-model", tmp.path())
            .await
            .unwrap();
        assert!(result.contains("nanobot"));
    }

    #[tokio::test]
    async fn test_refresh_bulletin_updates_cache_and_disk() {
        let tmp = TempDir::new().unwrap();
        let mem_path = tmp.path().join("memory");
        std::fs::create_dir_all(&mem_path).unwrap();
        std::fs::write(mem_path.join("MEMORY.md"), "facts").unwrap();

        let cache = BulletinCache::new();
        let handle = cache.handle();

        let provider = MockBulletinProvider;
        refresh_bulletin(&provider, "test", tmp.path(), &handle)
            .await
            .unwrap();

        // Cache updated
        assert!(!cache.read().is_empty());
        // Disk updated
        let disk = std::fs::read_to_string(tmp.path().join("BULLETIN.md")).unwrap();
        assert!(disk.contains("nanobot"));
    }
}
