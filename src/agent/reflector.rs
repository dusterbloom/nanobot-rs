//! Background reflector that condenses observations into long-term memory.
//!
//! When observations accumulate past a token threshold, the reflector reads
//! current `MEMORY.md` + all observations, calls the memory model to produce
//! an updated long-term memory, writes it, and archives the processed
//! observations.
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
use crate::providers::base::LLMProvider;

/// Prompt sent to the memory model for reflection.
const REFLECTION_PROMPT: &str = "\
You are reflecting on a series of conversation observations to update long-term memory.

Categories to maintain:
- User preferences and communication style
- Recurring topics and interests
- Effective approaches (what works, what doesn't)
- Key facts and context about the user
- Important decisions and their rationale

Current long-term memory:
{current_memory}

Recent observations:
{observations}

Write an updated long-term memory. Be concise, specific, and preserve existing knowledge.
Organize by category. Remove outdated or contradicted information.
Output only the updated memory content (markdown format).";

/// Background reflector that crystallizes observations into MEMORY.md.
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

    /// Check whether accumulated observations exceed the reflection threshold.
    pub fn should_reflect(&self) -> bool {
        let observer = ObservationStore::new(&self.workspace);
        let total = observer.total_tokens();
        debug!(
            "Reflector: {} observation tokens (threshold: {})",
            total, self.threshold_tokens
        );
        total > self.threshold_tokens
    }

    /// Perform reflection: read observations + current memory, call LLM,
    /// update MEMORY.md, archive processed observations.
    pub async fn reflect(&self) -> Result<()> {
        let memory_store = MemoryStore::new(&self.workspace);
        let observer = ObservationStore::new(&self.workspace);

        // Read current state.
        let current_memory = memory_store.read_long_term();
        let observations = observer.load_recent(usize::MAX);

        if observations.is_empty() {
            debug!("Reflector: no observations to process");
            return Ok(());
        }

        info!(
            "Reflector: processing {} observations into MEMORY.md",
            observations.len()
        );

        // Format observations for the prompt.
        let obs_text: String = observations
            .iter()
            .map(|obs| {
                format!(
                    "**[{}]** ({})\n{}",
                    obs.timestamp, obs.session_key, obs.content
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n");

        // Build the reflection prompt.
        let prompt = REFLECTION_PROMPT
            .replace("{current_memory}", &current_memory)
            .replace("{observations}", &obs_text);

        let messages = vec![
            json!({"role": "system", "content": "You are a memory management assistant."}),
            json!({"role": "user", "content": prompt}),
        ];

        let response = self
            .provider
            .chat(&messages, None, Some(&self.model), 2048, 0.3)
            .await?;

        let updated_memory = response
            .content
            .ok_or_else(|| anyhow::anyhow!("Reflection returned no content"))?;

        // Write updated memory (atomic — single write call).
        memory_store.write_long_term(&updated_memory);
        info!("Reflector: MEMORY.md updated");

        // Archive processed observations.
        let paths: Vec<PathBuf> = observations.iter().map(|o| o.path.clone()).collect();
        observer.archive(&paths)?;
        info!("Reflector: archived {} observations", paths.len());

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
        ) -> Result<LLMResponse> {
            Err(anyhow::anyhow!("LLM unavailable"))
        }

        fn get_default_model(&self) -> &str {
            "mock"
        }
    }

    fn setup_workspace_with_observations(
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
        let workspace = setup_workspace_with_observations(&tmp, 1, 10);
        let provider = Arc::new(MockProvider::new("memory"));
        let reflector = Reflector::new(provider, "test".into(), &workspace, 100_000);
        assert!(!reflector.should_reflect());
    }

    #[test]
    fn test_should_reflect_true_when_above_threshold() {
        let tmp = TempDir::new().unwrap();
        let workspace = setup_workspace_with_observations(&tmp, 10, 1000);
        let provider = Arc::new(MockProvider::new("memory"));
        // Low threshold so the observations exceed it.
        let reflector = Reflector::new(provider, "test".into(), &workspace, 100);
        assert!(reflector.should_reflect());
    }

    #[tokio::test]
    async fn test_reflect_updates_memory_md() {
        let tmp = TempDir::new().unwrap();
        let workspace = setup_workspace_with_observations(&tmp, 3, 100);
        let provider = Arc::new(MockProvider::new("# Updated Memory\n\nUser prefers Rust."));
        let reflector = Reflector::new(provider, "test".into(), &workspace, 0);

        reflector.reflect().await.unwrap();

        let memory = MemoryStore::new(&workspace);
        let content = memory.read_long_term();
        assert!(content.contains("User prefers Rust."));
    }

    #[tokio::test]
    async fn test_reflect_archives_observations() {
        let tmp = TempDir::new().unwrap();
        let workspace = setup_workspace_with_observations(&tmp, 3, 100);
        let provider = Arc::new(MockProvider::new("Updated memory content."));
        let reflector = Reflector::new(provider, "test".into(), &workspace, 0);

        reflector.reflect().await.unwrap();

        // Observations should be archived.
        let observer = ObservationStore::new(&workspace);
        let remaining = observer.load_recent(100);
        assert!(
            remaining.is_empty(),
            "observations should be archived after reflection"
        );

        // Archived directory should have files.
        let archived_dir = workspace
            .join("memory")
            .join("observations")
            .join("archived");
        assert!(archived_dir.exists());
        let archived_count = std::fs::read_dir(&archived_dir).unwrap().count();
        assert_eq!(archived_count, 3);
    }

    #[tokio::test]
    async fn test_reflect_graceful_on_failure() {
        let tmp = TempDir::new().unwrap();
        let workspace = setup_workspace_with_observations(&tmp, 2, 100);
        let provider = Arc::new(FailingProvider);
        let reflector = Reflector::new(provider, "test".into(), &workspace, 0);

        let result = reflector.reflect().await;
        assert!(result.is_err());

        // Observations should NOT be archived on failure.
        let observer = ObservationStore::new(&workspace);
        let remaining = observer.load_recent(100);
        assert_eq!(
            remaining.len(),
            2,
            "observations should be preserved on failure"
        );
    }
}
