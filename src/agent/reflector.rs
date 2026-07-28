//! Background reflector that distills working sessions into long-term factual memory.
//!
//! When completed working sessions accumulate past a token threshold, the
//! reflector reads current `MEMORY.md` + all completed sessions, calls the
//! memory model to extract reusable facts, writes the updated memory, and
//! marks the processed SQLite rows reflected.
//!
//! The reflector runs in a background `tokio::spawn` task and never blocks
//! user chat.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use serde_json::json;
use tracing::{debug, info, warn};

use crate::agent::knowledge_graph::KnowledgeGraph;
use crate::agent::memory::{memory_transaction_lock, MemoryStore};
use crate::agent::working_memory::{SessionStatus, WorkingMemoryStore};
use crate::providers::base::LLMProvider;
use crate::session::db::SessionDb;

/// Prompt sent to the memory model for facts + entity extraction.
const REFLECTION_PROMPT: &str = "\
You are distilling conversation sessions into permanent factual memory.

## Part 1: Facts
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

## Part 2: Entities & Relations
After the facts, write a section starting with `## Entities` that lists key entities and their relationships.
Format each entity as: `- ENTITY_NAME (TYPE): brief description`
Format each relation as: `- FROM -> LABEL -> TO`

Types: person, tool, project, language, concept, service, file
Labels: uses, prefers, created, knows, works-on, depends-on, related-to

Example:
## Entities
- Alex (person): the user
- Rust (language): preferred systems programming language
- nanobot (project): AI assistant framework
- Alex -> prefers -> Rust
- nanobot -> depends-on -> Rust

Current long-term memory:
{current_memory}

Completed SQLite working-memory summaries:
{session_summaries}

Write updated factual memory (bullet points), then ## Entities section. Be concise.";

/// Background reflector that crystallizes sessions into MEMORY.md.
pub struct Reflector {
    provider: Arc<dyn LLMProvider>,
    model: String,
    workspace: PathBuf,
    threshold_tokens: usize,
    sessions: Arc<SessionDb>,
}

impl Reflector {
    /// Create a new reflector.
    pub fn new(
        provider: Arc<dyn LLMProvider>,
        model: String,
        workspace: &Path,
        threshold: usize,
        sessions: Arc<SessionDb>,
    ) -> Self {
        Self {
            provider,
            model,
            workspace: workspace.to_path_buf(),
            threshold_tokens: threshold,
            sessions,
        }
    }

    /// Check whether completed SQLite working-memory rows exceed the threshold.
    pub async fn should_reflect(&self) -> bool {
        Self::should_reflect_sessions(&self.sessions, self.threshold_tokens).await
    }

    /// Check reflection pressure without constructing a model-bound reflector
    /// or making a provider call.
    pub async fn should_reflect_sessions(sessions: &Arc<SessionDb>, threshold: usize) -> bool {
        let wm = WorkingMemoryStore::new(sessions.clone());
        let wm_tokens = match wm.total_tokens_by_status(SessionStatus::Completed).await {
            Ok(tokens) => tokens,
            Err(error) => {
                warn!("Reflector: failed to count completed sessions: {}", error);
                0
            }
        };

        let total = wm_tokens;
        debug!(
            "Reflector: {} completed working-memory tokens (threshold: {})",
            total, threshold
        );
        total > threshold
    }

    /// Distill completed SQLite working-memory rows into `MEMORY.md`.
    ///
    /// `MemoryStore::write_long_term` uses temp-file + rename. Only after that
    /// atomic replacement succeeds do we mark the source rows reflected.
    pub async fn reflect(&self) -> Result<()> {
        // Reflection is a single read/derive/write/status transaction at the
        // process level. All triggers share this lock so a slower run cannot
        // overwrite facts produced by a newer run or reflect the same rows
        // from the same stale MEMORY.md base.
        let _memory_guard = memory_transaction_lock().lock().await;
        let memory_store = MemoryStore::new(&self.workspace);
        let wm = WorkingMemoryStore::new(self.sessions.clone());

        // Read current state.
        let current_memory = memory_store.read_long_term();

        // Gather summaries from completed working sessions.
        let completed_sessions = wm.list_completed().await?;
        let summaries: Vec<String> = completed_sessions
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

        if summaries.is_empty() {
            debug!("Reflector: no completed sessions to process");
            return Ok(());
        }

        info!(
            "Reflector: processing {} completed sessions into MEMORY.md",
            completed_sessions.len()
        );

        let summaries_text = summaries.join("\n\n");

        // Build the reflection prompt.
        let prompt = REFLECTION_PROMPT
            .replace("{current_memory}", &current_memory)
            .replace("{session_summaries}", &summaries_text);

        let messages = vec![
            json!({"role": "system", "content": "You are a memory management assistant. Extract only permanent facts."}),
            json!({"role": "user", "content": prompt}),
        ];

        let response = self
            .provider
            .chat(&messages, None, Some(&self.model), 2048, 0.3, None, None)
            .await?;

        let updated_memory = response
            .content
            .ok_or_else(|| anyhow::anyhow!("Reflection returned no content"))?;

        // Split response into facts and entities sections.
        let (facts, entities_section) = split_entities_section(&updated_memory);

        // Atomic temp-file + rename; do not advance source state first.
        memory_store.write_long_term(&facts);
        if memory_store.read_long_term() != facts {
            return Err(anyhow::anyhow!(
                "atomic MEMORY.md replacement did not persist; completed sessions retained"
            ));
        }
        info!("Reflector: MEMORY.md updated");

        // Extract entities/relations into knowledge graph.
        if !entities_section.is_empty() {
            match KnowledgeGraph::open_default() {
                Ok(mut kg) => {
                    let (ent_count, rel_count) =
                        parse_entities_into_graph(&mut kg, &entities_section);
                    if ent_count > 0 || rel_count > 0 {
                        if let Err(e) = kg.save() {
                            warn!("Failed to save knowledge graph: {}", e);
                        } else {
                            info!(
                                "Reflector: updated knowledge graph ({} entities, {} relations)",
                                ent_count, rel_count
                            );
                        }
                    }
                }
                Err(e) => warn!("Failed to open knowledge graph: {}", e),
            }
        }

        // Advance source lifecycle only after the atomic memory replacement.
        let reflected_ids: Vec<String> = completed_sessions
            .iter()
            .map(|session| session.session_id.clone())
            .collect();
        wm.mark_reflected_all(&reflected_ids).await?;
        info!(
            "Reflector: marked {} working sessions reflected",
            completed_sessions.len()
        );

        Ok(())
    }
}

/// Split LLM response into facts (before `## Entities`) and entities section (after).
fn split_entities_section(response: &str) -> (String, String) {
    // Look for "## Entities" marker (case-insensitive).
    let lower = response.to_lowercase();
    if let Some(pos) = lower.find("## entities") {
        let facts = response[..pos].trim().to_string();
        let entities = response[pos..].to_string();
        (facts, entities)
    } else {
        (response.to_string(), String::new())
    }
}

/// Parse the entities section and upsert into the knowledge graph.
/// Returns (entities_added, relations_added).
fn parse_entities_into_graph(kg: &mut KnowledgeGraph, section: &str) -> (usize, usize) {
    let mut ent_count = 0;
    let mut rel_count = 0;

    for line in section.lines() {
        let line = line.trim().trim_start_matches('-').trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Try relation format: "FROM -> LABEL -> TO"
        let parts: Vec<&str> = line.split("->").map(|s| s.trim()).collect();
        if parts.len() == 3 && !parts[0].is_empty() && !parts[2].is_empty() {
            kg.add_relation(parts[0], parts[1], parts[2], "reflector");
            rel_count += 1;
            continue;
        }

        // Try entity format: "NAME (TYPE): description"
        if let Some((name_type, desc)) = line.split_once(':') {
            let desc = desc.trim();
            if let Some((name, kind)) = name_type.split_once('(') {
                let name = name.trim();
                let kind = kind.trim().trim_end_matches(')').trim();
                if !name.is_empty() && !kind.is_empty() {
                    kg.upsert_entity(name, kind, desc);
                    ent_count += 1;
                }
            }
        }
    }

    (ent_count, rel_count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::base::{LLMProvider, LLMResponse};
    use async_trait::async_trait;
    use serde_json::Value;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
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
            _top_p: Option<f64>,
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
            _top_p: Option<f64>,
        ) -> Result<LLMResponse> {
            Err(anyhow::anyhow!("LLM unavailable"))
        }

        fn get_default_model(&self) -> &str {
            "mock"
        }
    }

    struct CoordinatedProvider {
        calls: AtomicUsize,
        active: AtomicUsize,
        max_active: AtomicUsize,
    }

    #[async_trait]
    impl LLMProvider for CoordinatedProvider {
        async fn chat(
            &self,
            _messages: &[Value],
            _tools: Option<&[Value]>,
            _model: Option<&str>,
            _max_tokens: u32,
            _temperature: f64,
            _thinking_budget: Option<u32>,
            _top_p: Option<f64>,
        ) -> Result<LLMResponse> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.max_active.fetch_max(active, Ordering::SeqCst);
            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
            self.active.fetch_sub(1, Ordering::SeqCst);
            Ok(LLMResponse {
                content: Some("- serialized fact".to_string()),
                tool_calls: vec![],
                finish_reason: "stop".to_string(),
                usage: HashMap::new(),
            })
        }

        fn get_default_model(&self) -> &str {
            "mock"
        }
    }

    async fn setup_workspace_with_sessions(
        tmp: &TempDir,
        count: usize,
        content_size: usize,
    ) -> (PathBuf, Arc<SessionDb>) {
        let workspace = tmp.path().to_path_buf();
        let mem_dir = workspace.join("memory");
        std::fs::create_dir_all(&mem_dir).unwrap();

        let sessions = Arc::new(SessionDb::new(&workspace.join("sessions.db")));
        let wm = WorkingMemoryStore::new(sessions.clone());
        for i in 0..count {
            let key = format!("test_session:{}", i);
            let session = sessions.create_session(&key).await;
            sessions
                .save_working_memory(&session.id, &"x".repeat(content_size), "active", 0)
                .await
                .unwrap();
            wm.complete(&session.id).await.unwrap();
        }
        (workspace, sessions)
    }

    #[tokio::test]
    async fn test_should_reflect_false_when_below_threshold() {
        let tmp = TempDir::new().unwrap();
        let (workspace, sessions) = setup_workspace_with_sessions(&tmp, 1, 10).await;
        let provider = Arc::new(MockProvider::new("memory"));
        let reflector = Reflector::new(provider, "test".into(), &workspace, 100_000, sessions);
        assert!(!reflector.should_reflect().await);
    }

    #[tokio::test]
    async fn test_should_reflect_true_when_above_threshold() {
        let tmp = TempDir::new().unwrap();
        let (workspace, sessions) = setup_workspace_with_sessions(&tmp, 10, 1000).await;
        let provider = Arc::new(MockProvider::new("memory"));
        let reflector = Reflector::new(provider, "test".into(), &workspace, 100, sessions);
        assert!(reflector.should_reflect().await);
    }

    #[tokio::test]
    async fn test_reflect_updates_memory_md_from_sessions() {
        let tmp = TempDir::new().unwrap();
        let (workspace, sessions) = setup_workspace_with_sessions(&tmp, 3, 100).await;
        let provider = Arc::new(MockProvider::new(
            "- User prefers Rust\n- Dark mode enabled",
        ));
        let reflector = Reflector::new(provider, "test".into(), &workspace, 0, sessions);

        reflector.reflect().await.unwrap();

        let memory = MemoryStore::new(&workspace);
        let content = memory.read_long_term();
        assert!(content.contains("User prefers Rust"));
    }

    #[tokio::test]
    async fn test_reflect_marks_completed_sessions_reflected() {
        let tmp = TempDir::new().unwrap();
        let (workspace, sessions) = setup_workspace_with_sessions(&tmp, 3, 100).await;
        let provider = Arc::new(MockProvider::new("Updated facts."));
        let reflector = Reflector::new(provider, "test".into(), &workspace, 0, sessions.clone());

        reflector.reflect().await.unwrap();

        let wm = WorkingMemoryStore::new(sessions);
        let remaining = wm.list_completed().await.unwrap();
        assert!(
            remaining.is_empty(),
            "completed sessions should be consumed after reflection"
        );

        assert_eq!(wm.list_reflected().await.unwrap().len(), 3);
        assert!(!workspace.join("memory").join("sessions").exists());
    }

    #[tokio::test]
    async fn concurrent_reflection_is_serialized_across_triggers() {
        let tmp = TempDir::new().unwrap();
        let (workspace, sessions) = setup_workspace_with_sessions(&tmp, 1, 100).await;
        let provider = Arc::new(CoordinatedProvider {
            calls: AtomicUsize::new(0),
            active: AtomicUsize::new(0),
            max_active: AtomicUsize::new(0),
        });
        let first = Reflector::new(
            provider.clone(),
            "test".into(),
            &workspace,
            0,
            sessions.clone(),
        );
        let second = Reflector::new(provider.clone(), "test".into(), &workspace, 0, sessions);

        let (first_result, second_result) = tokio::join!(first.reflect(), second.reflect());
        first_result.unwrap();
        second_result.unwrap();

        assert_eq!(provider.max_active.load(Ordering::SeqCst), 1);
        assert_eq!(
            provider.calls.load(Ordering::SeqCst),
            1,
            "the second trigger must re-read SQLite after the first marks rows reflected"
        );
    }

    #[tokio::test]
    async fn test_reflect_graceful_on_failure() {
        let tmp = TempDir::new().unwrap();
        let (workspace, sessions) = setup_workspace_with_sessions(&tmp, 2, 100).await;
        let provider = Arc::new(FailingProvider);
        let reflector = Reflector::new(provider, "test".into(), &workspace, 0, sessions.clone());

        let result = reflector.reflect().await;
        assert!(result.is_err());

        // Sessions must remain completed when the memory update fails.
        let wm = WorkingMemoryStore::new(sessions);
        let remaining = wm.list_completed().await.unwrap();
        assert_eq!(
            remaining.len(),
            2,
            "completed sessions should be preserved on failure"
        );
    }

    // --- Entity extraction parsing tests ---

    #[test]
    fn test_split_entities_section() {
        let response = "- Fact one\n- Fact two\n\n## Entities\n- Alice (person): the user\n- Alice -> prefers -> Rust";
        let (facts, entities) = split_entities_section(response);
        assert!(facts.contains("Fact one"));
        assert!(!facts.contains("Entities"));
        assert!(entities.contains("Alice (person)"));
        assert!(entities.contains("-> prefers ->"));
    }

    #[test]
    fn test_split_entities_section_no_entities() {
        let response = "- Fact one\n- Fact two";
        let (facts, entities) = split_entities_section(response);
        assert_eq!(facts, response);
        assert!(entities.is_empty());
    }

    #[cfg(feature = "knowledge-graph")]
    #[test]
    fn test_parse_entities_into_graph_entities() {
        let tmp = TempDir::new().unwrap();
        let section = "## Entities\n- Alice (person): the user\n- Rust (language): systems lang";
        let mut kg = KnowledgeGraph::open(&tmp.path().join("test_parse_ent.json")).unwrap();
        let (ent, rel) = parse_entities_into_graph(&mut kg, section);
        assert_eq!(ent, 2);
        assert_eq!(rel, 0);
        assert_eq!(kg.entity_count(), 2);
    }

    #[cfg(feature = "knowledge-graph")]
    #[test]
    fn test_parse_entities_into_graph_relations() {
        let tmp = TempDir::new().unwrap();
        let section = "## Entities\n- Alice (person): user\n- Rust (language): lang\n- Alice -> prefers -> Rust\n- Alice -> uses -> nanobot";
        let mut kg = KnowledgeGraph::open(&tmp.path().join("test_parse_rel.json")).unwrap();
        let (ent, rel) = parse_entities_into_graph(&mut kg, section);
        assert_eq!(ent, 2);
        assert_eq!(rel, 2);
        // Relations auto-create entities for "nanobot".
        assert!(kg.entity_count() >= 3);
    }

    #[cfg(feature = "knowledge-graph")]
    #[test]
    fn test_parse_entities_skips_malformed_lines() {
        let tmp = TempDir::new().unwrap();
        let section = "## Entities\n- just some text without parens\n- (bad): no name\n- -> -> -> too many arrows";
        let mut kg = KnowledgeGraph::open(&tmp.path().join("test_parse_bad.json")).unwrap();
        let (ent, rel) = parse_entities_into_graph(&mut kg, section);
        assert_eq!(ent, 0);
        assert_eq!(rel, 0);
    }
}
