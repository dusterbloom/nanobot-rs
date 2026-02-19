//! Background reflector that distills working sessions into long-term factual memory.
//!
//! When completed working sessions accumulate past a token threshold, the
//! reflector reads current `MEMORY.md` + all completed sessions, calls the
//! memory model to extract reusable facts, writes the updated memory, and
//! archives the processed sessions.
//!
//! Also checks for legacy observation files and processes those during a
//! transition period.
//!
//! The reflector runs in a background `tokio::spawn` task and never blocks
//! user chat.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use serde_json::json;
use tracing::{debug, info, warn};

use crate::agent::memory::MemoryStore;
use crate::agent::observer::ObservationStore;
use crate::agent::working_memory::{SessionStatus, WorkingMemoryStore};
use crate::providers::base::LLMProvider;

/// Prompt sent to the memory model for facts-only reflection.
const REFLECTION_PROMPT: &str = "\
You are distilling conversation sessions into permanent factual memory.

RULES:
- Extract ONLY concrete, reusable facts
- NO session logs, task status, or temporary context
- Each fact should be independently useful in future conversations
- Use bullet points, one fact per line
- Remove facts that are outdated or contradicted by newer information

Good examples:
- User's name is Alex, prefers dark mode
- nanobot binary is installed at /usr/local/bin/nanobot
- edit_file tool is unreliable on large files, prefer write_file

Bad examples (DO NOT include):
- Currently working on memory refactor
- Last session discussed file handling

Current long-term memory:
{current_memory}

Recent session summaries:
{observations}

Write updated factual memory. Bullet points only. Be concise.";

/// Background reflector that crystallizes sessions into MEMORY.md.
pub struct Reflector {
    provider: Arc<dyn LLMProvider>,
    model: String,
    workspace: PathBuf,
    threshold_tokens: usize,
}

impl Reflector {
    /// Create a new reflector.
    pub fn new(
        provider: Arc<dyn LLMProvider>,
        model: String,
        workspace: &Path,
        threshold: usize,
    ) -> Self {
        Self {
            provider,
            model,
            workspace: workspace.to_path_buf(),
            threshold_tokens: threshold,
        }
    }

    /// Check whether accumulated sessions/observations exceed the reflection threshold.
    ///
    /// Checks both completed working sessions and legacy observation files.
    pub fn should_reflect(&self) -> bool {
        let wm = WorkingMemoryStore::new(&self.workspace);
        let wm_tokens = wm.total_tokens_by_status(SessionStatus::Completed);

        // Also check legacy observations during transition period.
        let observer = ObservationStore::new(&self.workspace);
        let obs_tokens = observer.total_tokens();

        let total = wm_tokens + obs_tokens;
        debug!(
            "Reflector: {} tokens ({}wm + {}obs, threshold: {})",
            total, wm_tokens, obs_tokens, self.threshold_tokens
        );
        total > self.threshold_tokens
    }

    /// Perform reflection: read completed sessions + legacy observations +
    /// current memory, call LLM, update MEMORY.md, archive processed sources.
    pub async fn reflect(&self) -> Result<()> {
        let memory_store = MemoryStore::new(&self.workspace);
        let wm = WorkingMemoryStore::new(&self.workspace);
        let observer = ObservationStore::new(&self.workspace);

        // Read current state.
        let current_memory = memory_store.read_long_term();

        // Gather summaries from completed working sessions.
        let completed_sessions = wm.list_completed();
        let mut summaries: Vec<String> = completed_sessions
            .iter()
            .map(|s| {
                format!(
                    "**Session: {}** ({})\n{}",
                    s.session_key,
                    s.updated.format("%Y-%m-%d %H:%M"),
                    s.content
                )
            })
            .collect();

        // Also gather legacy observations (transition period).
        let legacy_obs = observer.load_recent(usize::MAX);
        for obs in &legacy_obs {
            summaries.push(format!(
                "**[{}]** ({})\n{}",
                obs.timestamp, obs.session_key, obs.content
            ));
        }

        if summaries.is_empty() {
            debug!("Reflector: no sessions or observations to process");
            return Ok(());
        }

        info!(
            "Reflector: processing {} sources ({} sessions + {} legacy obs) into MEMORY.md",
            summaries.len(),
            completed_sessions.len(),
            legacy_obs.len()
        );

        let obs_text = summaries.join("\n\n");

        // Build the reflection prompt.
        let prompt = REFLECTION_PROMPT
            .replace("{current_memory}", &current_memory)
            .replace("{observations}", &obs_text);

        let messages = vec![
            json!({"role": "system", "content": "You are a memory management assistant. Extract only permanent facts."}),
            json!({"role": "user", "content": prompt}),
        ];

        let response = self
            .provider
            .chat(&messages, None, Some(&self.model), 2048, 0.3, None)
            .await?;

        let updated_memory = response
            .content
            .ok_or_else(|| anyhow::anyhow!("Reflection returned no content"))?;

        // Write updated memory (atomic — single write call).
        memory_store.write_long_term(&updated_memory);
        info!("Reflector: MEMORY.md updated");

        // Archive processed working sessions.
        for session in &completed_sessions {
            if let Err(e) = wm.archive(&session.session_key) {
                warn!("Failed to archive session {}: {}", session.session_key, e);
            }
        }
        if !completed_sessions.is_empty() {
            info!(
                "Reflector: archived {} working sessions",
                completed_sessions.len()
            );
        }

        // Archive legacy observations.
        if !legacy_obs.is_empty() {
            let paths: Vec<PathBuf> = legacy_obs.iter().map(|o| o.path.clone()).collect();
            observer.archive(&paths)?;
            info!("Reflector: archived {} legacy observations", paths.len());
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::base::{LLMProvider, LLMResponse};
    use async_trait::async_trait;
    use serde_json::Value;
    use std::collections::HashMap;
    use tempfile::TempDir;

    /// Mock provider that returns a fixed response.
    struct MockProvider {
        response: String,
    }

    impl MockProvider {
        fn new(response: &str) -> Self {
            Self {
                response: response.to_string(),
            }
        }
    }

    #[async_trait]
    impl LLMProvider for MockProvider {
        async fn chat(
            &self,
            _messages: &[Value],
            _tools: Option<&[Value]>,
            _model: Option<&str>,
            _max_tokens: u32,
            _temperature: f64,
            _thinking_budget: Option<u32>,
        ) -> Result<LLMResponse> {
            Ok(LLMResponse {
                content: Some(self.response.clone()),
                tool_calls: vec![],
                finish_reason: "stop".to_string(),
                usage: HashMap::new(),
            })
        }

        fn get_default_model(&self) -> &str {
            "mock"
        }
    }

    /// Mock provider that always fails.
    struct FailingProvider;

    #[async_trait]
    impl LLMProvider for FailingProvider {
        async fn chat(
            &self,
            _messages: &[Value],
            _tools: Option<&[Value]>,
            _model: Option<&str>,
            _max_tokens: u32,
            _temperature: f64,
            _thinking_budget: Option<u32>,
        ) -> Result<LLMResponse> {
            Err(anyhow::anyhow!("LLM unavailable"))
        }

        fn get_default_model(&self) -> &str {
            "mock"
        }
    }

    fn setup_workspace_with_sessions(tmp: &TempDir, count: usize, content_size: usize) -> PathBuf {
        let workspace = tmp.path().to_path_buf();
        let mem_dir = workspace.join("memory");
        std::fs::create_dir_all(&mem_dir).unwrap();

        let wm = WorkingMemoryStore::new(&workspace);
        for i in 0..count {
            let key = format!("test_session:{}", i);
            wm.update_from_compaction(&key, &"x".repeat(content_size));
            wm.complete(&key);
        }
        workspace
    }

    fn setup_workspace_with_legacy_observations(
        tmp: &TempDir,
        count: usize,
        content_size: usize,
    ) -> PathBuf {
        let workspace = tmp.path().to_path_buf();
        let obs_dir = workspace.join("memory").join("observations");
        std::fs::create_dir_all(&obs_dir).unwrap();
        let mem_dir = workspace.join("memory");
        std::fs::create_dir_all(&mem_dir).unwrap();

        for i in 0..count {
            let content = format!(
                "---\ntimestamp: 2026-01-{:02}T00:00:00Z\nsession: test_{}\n---\n\n{}",
                (i % 28) + 1,
                i,
                "x".repeat(content_size),
            );
            std::fs::write(
                obs_dir.join(format!("202601{:02}T000000Z_test_{}.md", (i % 28) + 1, i)),
                content,
            )
            .unwrap();
        }

        workspace
    }

    #[test]
    fn test_should_reflect_false_when_below_threshold() {
        let tmp = TempDir::new().unwrap();
        let workspace = setup_workspace_with_sessions(&tmp, 1, 10);
        let provider = Arc::new(MockProvider::new("memory"));
        let reflector = Reflector::new(provider, "test".into(), &workspace, 100_000);
        assert!(!reflector.should_reflect());
    }

    #[test]
    fn test_should_reflect_true_when_above_threshold() {
        let tmp = TempDir::new().unwrap();
        let workspace = setup_workspace_with_sessions(&tmp, 10, 1000);
        let provider = Arc::new(MockProvider::new("memory"));
        let reflector = Reflector::new(provider, "test".into(), &workspace, 100);
        assert!(reflector.should_reflect());
    }

    #[test]
    fn test_should_reflect_includes_legacy_observations() {
        let tmp = TempDir::new().unwrap();
        let workspace = setup_workspace_with_legacy_observations(&tmp, 10, 1000);
        let provider = Arc::new(MockProvider::new("memory"));
        let reflector = Reflector::new(provider, "test".into(), &workspace, 100);
        assert!(reflector.should_reflect());
    }

    #[tokio::test]
    async fn test_reflect_updates_memory_md_from_sessions() {
        let tmp = TempDir::new().unwrap();
        let workspace = setup_workspace_with_sessions(&tmp, 3, 100);
        let provider = Arc::new(MockProvider::new(
            "- User prefers Rust\n- Dark mode enabled",
        ));
        let reflector = Reflector::new(provider, "test".into(), &workspace, 0);

        reflector.reflect().await.unwrap();

        let memory = MemoryStore::new(&workspace);
        let content = memory.read_long_term();
        assert!(content.contains("User prefers Rust"));
    }

    #[tokio::test]
    async fn test_reflect_archives_completed_sessions() {
        let tmp = TempDir::new().unwrap();
        let workspace = setup_workspace_with_sessions(&tmp, 3, 100);
        let provider = Arc::new(MockProvider::new("Updated facts."));
        let reflector = Reflector::new(provider, "test".into(), &workspace, 0);

        reflector.reflect().await.unwrap();

        let wm = WorkingMemoryStore::new(&workspace);
        let remaining = wm.list_completed();
        assert!(
            remaining.is_empty(),
            "completed sessions should be archived after reflection"
        );

        // Archived directory should have files.
        let archived_dir = workspace.join("memory").join("sessions").join("archived");
        assert!(archived_dir.exists());
        let archived_count = std::fs::read_dir(&archived_dir).unwrap().count();
        assert_eq!(archived_count, 3);
    }

    #[tokio::test]
    async fn test_reflect_archives_legacy_observations() {
        let tmp = TempDir::new().unwrap();
        let workspace = setup_workspace_with_legacy_observations(&tmp, 2, 100);
        // Also need to init sessions dir.
        std::fs::create_dir_all(workspace.join("memory").join("sessions")).unwrap();

        let provider = Arc::new(MockProvider::new("Updated facts."));
        let reflector = Reflector::new(provider, "test".into(), &workspace, 0);

        reflector.reflect().await.unwrap();

        let observer = ObservationStore::new(&workspace);
        let remaining = observer.load_recent(100);
        assert!(
            remaining.is_empty(),
            "legacy observations should be archived"
        );
    }

    #[tokio::test]
    async fn test_reflect_graceful_on_failure() {
        let tmp = TempDir::new().unwrap();
        let workspace = setup_workspace_with_sessions(&tmp, 2, 100);
        let provider = Arc::new(FailingProvider);
        let reflector = Reflector::new(provider, "test".into(), &workspace, 0);

        let result = reflector.reflect().await;
        assert!(result.is_err());

        // Sessions should NOT be archived on failure.
        let wm = WorkingMemoryStore::new(&workspace);
        let remaining = wm.list_completed();
        assert_eq!(
            remaining.len(),
            2,
            "completed sessions should be preserved on failure"
        );
    }
}
