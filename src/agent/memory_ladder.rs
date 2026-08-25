// Error-protocol layer-3 backlog (docs/research/2026-08-06-error-conventions-and-host-bridge.md §3.6):
// the deny regime in Cargo.toml is live; this module still carries pre-existing
// violations of the lints below. Remove this allow as the module migrates onto
// the regime.
// Tracking: docs/error-protocol-backlog.md
#![allow(clippy::shadow_reuse)]
//! Memory Ladder: priority-ordered memory layer facade.
//!
//! Provides a unified query interface over 4 named memory layers with
//! budget-aware priority waterfall allocation. Layers fill highest-priority
//! first; when the token budget is exhausted, lower-priority layers are skipped.
//!
//! Long-term memory (`MEMORY.md`) is NOT a ladder layer: it is injected
//! exactly once, by `ContextBuilder::collect_static_sections`'s
//! `PromptSection::MemoryBriefing` static section (both cloud and local
//! paths). A `GroundTruth` layer used to duplicate that same file into the
//! cloud path's runtime sections (`prepare_context::collect_cloud_runtime_sections`)
//! -- removed so `MEMORY.md` content reaches the wire exactly once.

use crate::agent::token_budget::TokenBudget;
use crate::agent::working_memory::WorkingMemoryStore;
use crate::session::db::SessionDb;

/// Named memory layers in priority order (lower discriminant = higher priority).
///
/// The Ord derivation on `#[repr(u8)]` gives us correct comparison:
/// `WorkingSession < DurablePersonal < SearchIndex < Scratch`.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MemoryLayer {
    /// Per-session working memory -- always available.
    WorkingSession = 0,
    /// Knowledge graph entities -- requires `knowledge-graph` feature.
    #[cfg_attr(not(feature = "knowledge-graph"), allow(dead_code))]
    DurablePersonal = 1,
    /// FTS5 semantic search -- requires `semantic` feature.
    #[cfg_attr(not(feature = "semantic"), allow(dead_code))]
    SearchIndex = 2,
    /// Session history search -- always available.
    Scratch = 3,
}

/// Query parameters for the memory ladder.
pub struct MemoryQuery<'a> {
    pub session_id: &'a str,
    pub query: &'a str,
    pub total_budget: usize,
}

/// Result from a single layer after budget-constrained fetch.
#[derive(Debug)]
pub struct LayerResult {
    pub layer: MemoryLayer,
    pub content: String,
}

/// Priority-ordered memory facade over all available stores.
///
/// Borrows from `SwappableCore` and `AgentLoopShared` -- lifetime `'a`
/// covers the turn in which the query executes.
pub struct MemoryLadder<'a> {
    working_memory: &'a WorkingMemoryStore,
    session_db: &'a SessionDb,
}

impl<'a> MemoryLadder<'a> {
    pub fn new(working_memory: &'a WorkingMemoryStore, session_db: &'a SessionDb) -> Self {
        Self {
            working_memory,
            session_db,
        }
    }

    /// Returns the active layers for this build configuration.
    ///
    /// DurablePersonal requires `knowledge-graph`, SearchIndex requires `semantic`.
    pub fn available_layers(&self) -> Vec<MemoryLayer> {
        let mut layers = vec![MemoryLayer::WorkingSession];

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
    ///
    #[allow(clippy::unreachable)] // feature-gated layers cannot exist without their feature
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

            let content = self
                .fetch_layer(layer, q.session_id, q.query, allocation)
                .await;

            if !content.is_empty() {
                let tokens_used = TokenBudget::estimate_str_tokens(&content);
                remaining = remaining.saturating_sub(tokens_used);
                results.push(LayerResult { layer, content });
            }
        }

        results
    }

    /// Fetch content from a single layer, truncated to the given token budget.
    #[allow(clippy::unreachable)] // feature-gated layers cannot exist without their feature
    async fn fetch_layer(
        &self,
        layer: MemoryLayer,
        session_id: &str,
        query: &str,
        budget: usize,
    ) -> String {
        match layer {
            MemoryLayer::WorkingSession => self
                .working_memory
                .get_context(session_id, budget)
                .await
                .unwrap_or_else(|error| {
                    tracing::warn!(%error, %session_id, "working-memory lookup failed");
                    String::new()
                }),
            MemoryLayer::DurablePersonal => {
                if query.trim().is_empty() {
                    return String::new();
                }
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
                if query.trim().is_empty() {
                    return String::new();
                }
                #[cfg(feature = "semantic")]
                {
                    if let Ok(ks) = crate::agent::knowledge_store::KnowledgeStore::open_default() {
                        let Ok(hits) = ks.search(query, 10) else {
                            return String::new();
                        };
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
                }
                #[cfg(not(feature = "semantic"))]
                {
                    let _ = (query, budget);
                    unreachable!(
                        "SearchIndex layer should not be available without semantic feature"
                    )
                }
            }
            MemoryLayer::Scratch => {
                if query.is_empty() {
                    return String::new();
                }
                let results = self.session_db.search_messages(query, 10, None).await;
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

    let mut kept = String::new();
    for line in content.lines() {
        let candidate = if kept.is_empty() {
            line.to_string()
        } else {
            format!("{kept}\n{line}")
        };

        if TokenBudget::estimate_str_tokens(&candidate) > budget {
            if kept.is_empty() {
                return truncate_line_to_token_budget(line, budget);
            }
            break;
        }

        kept = candidate;
    }
    kept
}

fn truncate_line_to_token_budget(line: &str, budget: usize) -> String {
    let mut kept = String::new();
    for word in line.split_whitespace() {
        let candidate = if kept.is_empty() {
            word.to_string()
        } else {
            format!("{kept} {word}")
        };

        if TokenBudget::estimate_str_tokens(&candidate) > budget {
            if kept.is_empty() {
                return truncate_chars_to_token_budget(word, budget);
            }
            break;
        }

        kept = candidate;
    }

    if kept.is_empty() {
        truncate_chars_to_token_budget(line, budget)
    } else {
        kept
    }
}

fn truncate_chars_to_token_budget(text: &str, budget: usize) -> String {
    let mut kept = String::new();
    for ch in text.chars() {
        let mut candidate = kept.clone();
        candidate.push(ch);
        if TokenBudget::estimate_str_tokens(&candidate) > budget {
            break;
        }
        kept = candidate;
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tempfile::TempDir;

    fn open_db(tmp: &TempDir) -> Arc<SessionDb> {
        Arc::new(SessionDb::new(&tmp.path().join("sessions.db")))
    }

    #[test]
    fn test_layer_priority_ordering() {
        assert!(MemoryLayer::WorkingSession < MemoryLayer::DurablePersonal);
        assert!(MemoryLayer::DurablePersonal < MemoryLayer::SearchIndex);
        assert!(MemoryLayer::SearchIndex < MemoryLayer::Scratch);
    }

    #[test]
    fn test_all_layers_count() {
        // Explicitly construct all 4 variants to verify the enum has exactly 4.
        let all = [
            MemoryLayer::WorkingSession,
            MemoryLayer::DurablePersonal,
            MemoryLayer::SearchIndex,
            MemoryLayer::Scratch,
        ];
        assert_eq!(all.len(), 4);
        // Verify discriminants are sequential 0..=3.
        for (i, layer) in all.iter().enumerate() {
            assert_eq!(*layer as u8, i as u8);
        }
    }

    #[test]
    fn test_available_layers_feature_gated() {
        let tmp = TempDir::new().unwrap();
        let session_db = open_db(&tmp);
        let wm = WorkingMemoryStore::new(session_db.clone());

        let ladder = MemoryLadder::new(&wm, session_db.as_ref());
        let layers = ladder.available_layers();

        let mut expected = vec![MemoryLayer::WorkingSession];
        #[cfg(feature = "knowledge-graph")]
        expected.push(MemoryLayer::DurablePersonal);
        #[cfg(feature = "semantic")]
        expected.push(MemoryLayer::SearchIndex);
        expected.push(MemoryLayer::Scratch);

        assert_eq!(layers, expected);
    }

    #[tokio::test]
    async fn test_budget_waterfall_exhaustion() {
        // Fill working memory (the highest-priority layer now that MEMORY.md
        // is injected exclusively by ContextBuilder's static MemoryBriefing
        // section, not through the ladder) with enough content to consume
        // the entire budget.
        let tmp = TempDir::new().unwrap();
        let session_db = open_db(&tmp);
        let session = session_db.create_session("test:session").await;
        let wm = WorkingMemoryStore::new(session_db.clone());

        let large_content = (0..500)
            .map(|i| format!("Important fact number {} about the user's preferences.", i))
            .collect::<Vec<_>>()
            .join("\n");
        session_db
            .save_working_memory(&session.id, &large_content, "active", 0)
            .await
            .unwrap();

        let ladder = MemoryLadder::new(&wm, session_db.as_ref());
        let results = ladder
            .query(&MemoryQuery {
                session_id: &session.id,
                query: "",
                total_budget: 20, // Very small budget -- WorkingSession should consume most of it
            })
            .await;

        // WorkingSession should be present (it has content).
        assert!(
            results
                .iter()
                .any(|r| r.layer == MemoryLayer::WorkingSession),
            "WorkingSession should be present"
        );

        // Total tokens used should not exceed budget.
        let total: usize = results
            .iter()
            .map(|r| TokenBudget::estimate_str_tokens(&r.content))
            .sum();
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
        let session_db = open_db(&tmp);
        let session = session_db.create_session("test:session").await;
        let wm = WorkingMemoryStore::new(session_db.clone());

        let large_content = (0..200)
            .map(|i| format!("Line {} with enough words to accumulate tokens quickly.", i))
            .collect::<Vec<_>>()
            .join("\n");
        session_db
            .save_working_memory(&session.id, &large_content, "active", 0)
            .await
            .unwrap();

        let ladder = MemoryLadder::new(&wm, session_db.as_ref());
        let results = ladder
            .query(&MemoryQuery {
                session_id: &session.id,
                query: "",
                total_budget: 100,
            })
            .await;

        for result in &results {
            let tokens_used = TokenBudget::estimate_str_tokens(&result.content);
            assert!(
                tokens_used <= 50,
                "Layer {:?} used {} tokens, exceeding 50% soft cap",
                result.layer,
                tokens_used
            );
        }
    }

    #[tokio::test]
    async fn test_scratch_query_uses_async_sqlite_search() {
        let tmp = TempDir::new().unwrap();
        let session_db = open_db(&tmp);
        let session = session_db.create_session("test:session").await;
        let wm = WorkingMemoryStore::new(session_db.clone());
        session_db
            .save_working_memory(&session.id, "A durable fact.", "active", 0)
            .await
            .unwrap();

        let ladder = MemoryLadder::new(&wm, session_db.as_ref());
        let results = ladder
            .query(&MemoryQuery {
                session_id: &session.id,
                query: "find anything",
                total_budget: 200,
            })
            .await;

        // WorkingSession still answers even when SQLite has no search hit.
        assert!(results
            .iter()
            .any(|r| r.layer == MemoryLayer::WorkingSession));
        assert!(!results.iter().any(|r| r.layer == MemoryLayer::Scratch));
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
        assert!(TokenBudget::estimate_str_tokens(&result) <= 10);
    }

    #[test]
    fn test_truncate_to_token_budget_splits_oversized_first_line() {
        let content = "This first line is intentionally too large for the tiny budget.";
        let result = truncate_to_token_budget(content, 3);
        assert!(!result.is_empty());
        assert!(TokenBudget::estimate_str_tokens(&result) <= 3);
    }
}
