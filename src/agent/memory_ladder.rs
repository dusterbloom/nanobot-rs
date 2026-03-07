//! Memory Ladder: priority-ordered memory layer facade.
//!
//! Provides a unified query interface over 5 named memory layers with
//! budget-aware priority waterfall allocation. Layers fill highest-priority
//! first; when the token budget is exhausted, lower-priority layers are skipped.

use std::path::{Path, PathBuf};

use crate::agent::knowledge_store::KnowledgeStore;
use crate::agent::memory::MemoryStore;
use crate::agent::token_budget::TokenBudget;
use crate::agent::working_memory::WorkingMemoryStore;
use crate::session::db::SessionDb;

/// Named memory layers in priority order (lower discriminant = higher priority).
///
/// The Ord derivation on `#[repr(u8)]` gives us correct comparison:
/// `GroundTruth < WorkingSession < DurablePersonal < SearchIndex < Scratch`.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MemoryLayer {
    /// Long-term memory (MEMORY.md) -- always available.
    GroundTruth = 0,
    /// Per-session working memory -- always available.
    WorkingSession = 1,
    /// Knowledge graph entities -- requires `knowledge-graph` feature.
    DurablePersonal = 2,
    /// FTS5 semantic search -- requires `semantic` feature.
    SearchIndex = 3,
    /// Session history search -- always available.
    Scratch = 4,
}

/// Query parameters for the memory ladder.
pub struct MemoryQuery<'a> {
    pub session_key: &'a str,
    pub query: &'a str,
    pub total_budget: usize,
}

/// Result from a single layer after budget-constrained fetch.
#[derive(Debug)]
pub struct LayerResult {
    pub layer: MemoryLayer,
    pub content: String,
    pub tokens_used: usize,
}

/// Priority-ordered memory facade over all available stores.
///
/// Borrows from `SwappableCore` and `AgentLoopShared` -- lifetime `'a`
/// covers the turn in which the query executes.
pub struct MemoryLadder<'a> {
    workspace: PathBuf,
    working_memory: &'a WorkingMemoryStore,
    knowledge_store: Option<&'a KnowledgeStore>,
    session_db: &'a SessionDb,
}

impl<'a> MemoryLadder<'a> {
    pub fn new(
        workspace: &Path,
        working_memory: &'a WorkingMemoryStore,
        knowledge_store: Option<&'a KnowledgeStore>,
        session_db: &'a SessionDb,
    ) -> Self {
        Self {
            workspace: workspace.to_path_buf(),
            working_memory,
            knowledge_store,
            session_db,
        }
    }

    /// Returns the active layers for this build configuration.
    ///
    /// DurablePersonal requires `knowledge-graph`, SearchIndex requires `semantic`.
    pub fn available_layers(&self) -> Vec<MemoryLayer> {
        let mut layers = vec![MemoryLayer::GroundTruth, MemoryLayer::WorkingSession];

        #[cfg(feature = "knowledge-graph")]
        layers.push(MemoryLayer::DurablePersonal);

        #[cfg(feature = "semantic")]
        layers.push(MemoryLayer::SearchIndex);

        layers.push(MemoryLayer::Scratch);
        layers
    }

    /// Query all available layers with priority waterfall budget allocation.
    ///
    /// Iterates layers in priority order, allocating up to 50% of total budget
    /// per layer. When remaining budget reaches 0, lower layers are skipped.
    pub async fn query(&self, q: &MemoryQuery<'_>) -> Vec<LayerResult> {
        let mut results = Vec::new();
        let mut remaining = q.total_budget;

        for layer in self.available_layers() {
            if remaining == 0 {
                break;
            }

            // 50% soft cap: no single layer gets more than half the total budget.
            let allocation = remaining.min(q.total_budget / 2);
            // Edge case: if total_budget is 1, allocation would be 0. Ensure at least 1.
            let allocation = if allocation == 0 && remaining > 0 {
                remaining
            } else {
                allocation
            };

            let content = self.fetch_layer(layer, q.session_key, q.query, allocation).await;

            if !content.is_empty() {
                let tokens_used = TokenBudget::estimate_str_tokens(&content);
                remaining = remaining.saturating_sub(tokens_used);
                results.push(LayerResult {
                    layer,
                    content,
                    tokens_used,
                });
            }
        }

        results
    }

    /// Fetch content from a single layer, truncated to the given token budget.
    async fn fetch_layer(
        &self,
        layer: MemoryLayer,
        session_key: &str,
        query: &str,
        budget: usize,
    ) -> String {
        match layer {
            MemoryLayer::GroundTruth => {
                let raw = MemoryStore::new(&self.workspace).read_long_term();
                truncate_to_token_budget(&raw, budget)
            }
            MemoryLayer::WorkingSession => {
                self.working_memory.get_context(session_key, budget)
            }
            MemoryLayer::DurablePersonal => {
                #[cfg(feature = "knowledge-graph")]
                {
                    use crate::agent::knowledge_graph::KnowledgeGraph;
                    if let Ok(kg) = KnowledgeGraph::open_default() {
                        let entities = kg.search_entities(query);
                        if entities.is_empty() {
                            return String::new();
                        }
                        let formatted: Vec<String> = entities
                            .iter()
                            .map(|e| format!("- **{}** ({}): {}", e.name, e.kind, e.summary))
                            .collect();
                        truncate_to_token_budget(&formatted.join("\n"), budget)
                    } else {
                        String::new()
                    }
                }
                #[cfg(not(feature = "knowledge-graph"))]
                {
                    let _ = query;
                    unreachable!("DurablePersonal layer should not be available without knowledge-graph feature")
                }
            }
            MemoryLayer::SearchIndex => {
                #[cfg(feature = "semantic")]
                {
                    if let Some(ks) = self.knowledge_store {
                        if let Ok(hits) = ks.search(query, 10) {
                            if hits.is_empty() {
                                return String::new();
                            }
                            let formatted: Vec<String> = hits
                                .iter()
                                .map(|h| format!("[{}#{}] {}", h.source_name, h.chunk_idx, h.snippet))
                                .collect();
                            truncate_to_token_budget(&formatted.join("\n"), budget)
                        } else {
                            String::new()
                        }
                    } else {
                        String::new()
                    }
                }
                #[cfg(not(feature = "semantic"))]
                {
                    let _ = (query, budget);
                    unreachable!("SearchIndex layer should not be available without semantic feature")
                }
            }
            MemoryLayer::Scratch => {
                if query.is_empty() {
                    return String::new();
                }
                let results = self
                    .session_db
                    .search_messages(query, 10, None)
                    .await;
                if results.is_empty() {
                    return String::new();
                }
                let formatted: Vec<String> = results
                    .iter()
                    .map(|r| format!("[{}] {}: {}", r.timestamp, r.role, r.snippet))
                    .collect();
                truncate_to_token_budget(&formatted.join("\n"), budget)
            }
        }
    }
}

/// Truncate content to fit within a token budget, cutting at line boundaries.
fn truncate_to_token_budget(content: &str, budget: usize) -> String {
    if content.is_empty() || budget == 0 {
        return String::new();
    }
    let total = TokenBudget::estimate_str_tokens(content);
    if total <= budget {
        return content.to_string();
    }

    let mut kept = Vec::new();
    let mut accumulated = 0;
    for line in content.lines() {
        let line_tokens = TokenBudget::estimate_str_tokens(line) + 1; // +1 for newline
        if accumulated + line_tokens > budget && !kept.is_empty() {
            break;
        }
        kept.push(line);
        accumulated += line_tokens;
    }
    kept.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_layer_priority_ordering() {
        assert!(MemoryLayer::GroundTruth < MemoryLayer::WorkingSession);
        assert!(MemoryLayer::WorkingSession < MemoryLayer::DurablePersonal);
        assert!(MemoryLayer::DurablePersonal < MemoryLayer::SearchIndex);
        assert!(MemoryLayer::SearchIndex < MemoryLayer::Scratch);
    }

    #[test]
    fn test_all_layers_count() {
        // Explicitly construct all 5 variants to verify the enum has exactly 5.
        let all = [
            MemoryLayer::GroundTruth,
            MemoryLayer::WorkingSession,
            MemoryLayer::DurablePersonal,
            MemoryLayer::SearchIndex,
            MemoryLayer::Scratch,
        ];
        assert_eq!(all.len(), 5);
        // Verify discriminants are sequential 0..=4.
        for (i, layer) in all.iter().enumerate() {
            assert_eq!(*layer as u8, i as u8);
        }
    }

    #[test]
    fn test_available_layers_feature_gated() {
        // In the default test build (no knowledge-graph, no semantic features),
        // only GroundTruth, WorkingSession, and Scratch should be available.
        let tmp = TempDir::new().unwrap();
        let wm = WorkingMemoryStore::new(tmp.path());
        let db_path = tmp.path().join("sessions.db");
        let session_db = SessionDb::new(&db_path);

        let ladder = MemoryLadder::new(tmp.path(), &wm, None, &session_db);
        let layers = ladder.available_layers();

        assert_eq!(
            layers,
            vec![
                MemoryLayer::GroundTruth,
                MemoryLayer::WorkingSession,
                MemoryLayer::Scratch,
            ]
        );
    }

    #[tokio::test]
    async fn test_budget_waterfall_exhaustion() {
        // Create a workspace with a large MEMORY.md that fills the budget.
        let tmp = TempDir::new().unwrap();
        let mem_dir = tmp.path().join("memory");
        std::fs::create_dir_all(&mem_dir).unwrap();

        // Write enough content to consume the entire budget.
        let large_content = (0..500)
            .map(|i| format!("Important fact number {} about the user's preferences.", i))
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(mem_dir.join("MEMORY.md"), &large_content).unwrap();

        let wm = WorkingMemoryStore::new(tmp.path());
        // Add some working memory too.
        wm.update_from_compaction("test:session", "Working memory content here.", 0);

        let db_path = tmp.path().join("sessions.db");
        let session_db = SessionDb::new(&db_path);

        let ladder = MemoryLadder::new(tmp.path(), &wm, None, &session_db);
        let results = ladder
            .query(&MemoryQuery {
                session_key: "test:session",
                query: "",
                total_budget: 20, // Very small budget -- GroundTruth should consume most of it
            })
            .await;

        // GroundTruth should be present (it has content).
        assert!(
            results.iter().any(|r| r.layer == MemoryLayer::GroundTruth),
            "GroundTruth should be present"
        );

        // Total tokens used should not exceed budget.
        let total: usize = results.iter().map(|r| r.tokens_used).sum();
        assert!(
            total <= 20,
            "Total tokens {} should not exceed budget 20",
            total
        );
    }

    #[tokio::test]
    async fn test_soft_cap_enforcement() {
        // With total_budget=100, no single layer should get more than 50 tokens.
        let tmp = TempDir::new().unwrap();
        let mem_dir = tmp.path().join("memory");
        std::fs::create_dir_all(&mem_dir).unwrap();

        // Large MEMORY.md that could fill 100+ tokens.
        let large_content = (0..200)
            .map(|i| format!("Line {} with enough words to accumulate tokens quickly.", i))
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(mem_dir.join("MEMORY.md"), &large_content).unwrap();

        let wm = WorkingMemoryStore::new(tmp.path());
        wm.update_from_compaction("test:session", &large_content, 0);

        let db_path = tmp.path().join("sessions.db");
        let session_db = SessionDb::new(&db_path);

        let ladder = MemoryLadder::new(tmp.path(), &wm, None, &session_db);
        let results = ladder
            .query(&MemoryQuery {
                session_key: "test:session",
                query: "",
                total_budget: 100,
            })
            .await;

        for result in &results {
            assert!(
                result.tokens_used <= 50,
                "Layer {:?} used {} tokens, exceeding 50% soft cap",
                result.layer,
                result.tokens_used
            );
        }
    }

    #[test]
    fn test_truncate_to_token_budget_empty() {
        assert_eq!(truncate_to_token_budget("", 100), "");
        assert_eq!(truncate_to_token_budget("hello", 0), "");
    }

    #[test]
    fn test_truncate_to_token_budget_within() {
        let content = "Short content.";
        let result = truncate_to_token_budget(content, 1000);
        assert_eq!(result, content);
    }

    #[test]
    fn test_truncate_to_token_budget_over() {
        let content = (0..100)
            .map(|i| format!("Line {} with some content.", i))
            .collect::<Vec<_>>()
            .join("\n");
        let result = truncate_to_token_budget(&content, 10);
        assert!(result.len() < content.len(), "should be truncated");
        assert!(result.contains("Line 0"), "should keep from head");
    }
}
