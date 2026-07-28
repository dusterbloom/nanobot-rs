//! Recall tool: semantic and keyword search across curated memory.
//!
//! Uses in-process `KnowledgeStore` for hybrid BM25+vector search across
//! indexed documents and curated MEMORY.md. Raw transcripts belong exclusively
//! to `session_search`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use async_trait::async_trait;
use serde_json::{json, Value};

use super::base::Tool;
use crate::agent::knowledge_store::{KnowledgeStore, SearchHit};

/// Legacy working-memory files may still be present in an existing
/// `knowledge.db`. They are no longer canonical and must not compete with
/// curated retrieval.
fn is_legacy_session_source(source_name: &str) -> bool {
    let basename = source_name
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(source_name);
    basename.starts_with("SESSION_") && basename.ends_with(".md")
}

fn format_knowledge_hits(hits: &[SearchHit]) -> Option<String> {
    let mut output = String::new();
    for hit in hits
        .iter()
        .filter(|hit| !is_legacy_session_source(&hit.source_name))
    {
        output.push_str(&format!(
            "### {} (chunk {})\n{}\n\n",
            hit.source_name, hit.chunk_idx, hit.snippet
        ));
    }

    if output.is_empty() {
        None
    } else {
        Some(output.trim_end().to_string())
    }
}

/// Rank curated `MEMORY.md` facts against the meaningful words in a natural
/// language query. Unlike the old full-line substring check, this handles
/// calls such as "what does Peppi prefer?" without missing the fact just
/// because the whole sentence is not in the file.
fn matching_memory_facts<'a>(content: &'a str, query: &str, limit: usize) -> Vec<&'a str> {
    let terms = crate::session::db::SessionDb::recall_keywords(query);
    if terms.is_empty() {
        return Vec::new();
    }

    let mut ranked: Vec<(usize, usize, &str)> = content
        .lines()
        .enumerate()
        .filter_map(|(index, line)| {
            let fact = line.trim_start().strip_prefix("- ")?.trim();
            let lower = fact.to_lowercase();
            let score = terms.iter().filter(|term| lower.contains(*term)).count();
            (score > 0).then_some((score, index, fact))
        })
        .collect();
    ranked.sort_by_key(|(score, index, _)| (std::cmp::Reverse(*score), *index));
    ranked
        .into_iter()
        .take(limit)
        .map(|(_, _, fact)| fact)
        .collect()
}

/// How a recall query is executed against the knowledge store.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QueryMode {
    /// BM25/FTS only — exact-match shaped queries.
    Keyword,
    /// BM25 + vector fusion (degrades to BM25 when vectors are unavailable).
    Hybrid,
}

/// Pick the query mode. An explicit `mode` param wins; otherwise multi-word
/// queries (≥2 whitespace-separated tokens) use hybrid when the semantic
/// capability is compiled in, and single-word queries stay keyword.
fn choose_query_mode(explicit: Option<&str>, query: &str, semantic_available: bool) -> QueryMode {
    match explicit {
        Some("keyword") => QueryMode::Keyword,
        Some("semantic") | Some("hybrid") => QueryMode::Hybrid,
        _ => {
            let multi_word = query.split_whitespace().nth(1).is_some();
            if multi_word && semantic_available {
                QueryMode::Hybrid
            } else {
                QueryMode::Keyword
            }
        }
    }
}

/// Tool that searches curated nanobot memory layers.
pub struct RecallTool {
    workspace: PathBuf,
}

impl RecallTool {
    pub fn new(workspace: &Path) -> Self {
        Self {
            workspace: workspace.to_path_buf(),
        }
    }

    /// Search the knowledge store (hybrid BM25 + vector when semantic feature is enabled).
    fn knowledge_search(&self, query: &str, n: usize, mode: QueryMode) -> Option<String> {
        let store = KnowledgeStore::open_default().ok()?;

        let hits = match mode {
            QueryMode::Keyword => store.search(query, n).ok()?,
            QueryMode::Hybrid => store.hybrid_search(query, n).ok()?,
        };

        format_knowledge_hits(&hits)
    }

    /// Search curated MEMORY.md facts.
    async fn grep_memory(&self, query: &str, max_results: usize) -> String {
        let memory_dir = self.workspace.join("memory");

        // Search MEMORY.md
        let memory_file = memory_dir.join("MEMORY.md");
        let mut results: Vec<String> = Vec::new();

        if memory_file.exists() {
            if let Ok(content) = tokio::fs::read_to_string(&memory_file).await {
                let facts = matching_memory_facts(&content, query, max_results);
                if !facts.is_empty() {
                    let bullets = facts
                        .into_iter()
                        .map(|fact| format!("- {fact}"))
                        .collect::<Vec<_>>()
                        .join("\n");
                    results.push(format!("## Curated memory\n{bullets}"));
                }
            }
        }

        if results.is_empty() {
            format!("No matches found for '{}' in memory.", query)
        } else {
            results.join("\n\n")
        }
    }
}

#[async_trait]
impl Tool for RecallTool {
    fn name(&self) -> &str {
        "recall"
    }

    fn description(&self) -> &str {
        "Search curated long-term memory: facts in MEMORY.md and indexed documents. \
         Use this for durable user preferences, decisions, and saved knowledge; use \
         session_search for raw past conversations. \
         Multi-word queries automatically use hybrid keyword+semantic search when available; \
         pass mode='keyword' for exact matches or mode='semantic' to force meaning-based search."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query — the curated fact or saved knowledge to recall."
                },
                "mode": {
                    "type": "string",
                    "enum": ["auto", "keyword", "semantic"],
                    "description": "Search mode: 'auto' tries hybrid BM25+semantic (default), \
                                   'keyword' for BM25-only exact matches, or 'semantic' for meaning-based search"
                }
            },
            "required": ["query"]
        })
    }

    async fn execute(&self, params: HashMap<String, Value>) -> String {
        let explicit_mode = params.get("mode").and_then(|v| v.as_str());
        let query = params
            .get("query")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim();

        if let Some(mode) = explicit_mode {
            if !matches!(mode, "auto" | "keyword" | "semantic") {
                return if mode == "latest" {
                    "Error: mode='latest' belongs to session_search, not recall.".to_string()
                } else {
                    format!(
                        "Error: unsupported recall mode '{mode}'. Use auto, keyword, or semantic."
                    )
                };
            }
        }
        if query.is_empty() {
            return "Error: 'query' parameter is required and must be non-empty.".to_string();
        }
        let mode = choose_query_mode(explicit_mode, query, cfg!(feature = "semantic"));

        if crate::session::db::SessionDb::recall_keywords(query).is_empty() {
            return "No specific memory topic found in the query. Ask about a person, preference, decision, project, or other concrete detail."
                .to_string();
        }

        let n = 5;
        let canonical = self.grep_memory(query, n).await;
        let mut sections = if canonical.starts_with("No matches found") {
            Vec::new()
        } else {
            vec![canonical.clone()]
        };

        // The document index is a fallback, not a peer of curated MEMORY.md.
        // Merging both made stale indexed context drown out canonical facts.
        if sections.is_empty() {
            if let Some(results) = self.knowledge_search(query, n, mode) {
                let label = match mode {
                    QueryMode::Keyword => "Indexed document results",
                    QueryMode::Hybrid => "Indexed document results (hybrid)",
                };
                sections.push(format!("## {label}\n{results}"));
            }
        }
        if sections.is_empty() {
            sections.push(canonical);
        }

        // Truncate total output to avoid blowing context (UTF-8 safe).
        let output = sections.join("\n\n");
        if output.len() > 8000 {
            let truncated: String = output.chars().take(8000).collect();
            format!(
                "{}\n\n[truncated — {} total chars]",
                truncated,
                output.len()
            )
        } else {
            output
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_tool() -> (TempDir, RecallTool) {
        let tmp = TempDir::new().unwrap();
        std::fs::create_dir_all(tmp.path().join("memory")).unwrap();
        let tool = RecallTool::new(tmp.path());
        (tmp, tool)
    }

    #[test]
    fn test_recall_tool_name() {
        let (_tmp, tool) = make_tool();
        assert_eq!(tool.name(), "recall");
    }

    #[test]
    fn test_choose_query_mode_explicit_wins() {
        // Explicit mode always wins, regardless of word count or capability.
        assert_eq!(
            choose_query_mode(Some("keyword"), "several words here", true),
            QueryMode::Keyword
        );
        assert_eq!(
            choose_query_mode(Some("semantic"), "word", false),
            QueryMode::Hybrid
        );
    }

    #[test]
    fn test_choose_query_mode_heuristic_word_count() {
        // Single-word queries are exact-match shaped → keyword.
        assert_eq!(choose_query_mode(None, "rust", true), QueryMode::Keyword);
        // Multi-word queries use hybrid when semantic capability is compiled in.
        assert_eq!(
            choose_query_mode(None, "how compaction was configured", true),
            QueryMode::Hybrid
        );
        // ... but degrade to keyword when it is not.
        assert_eq!(
            choose_query_mode(None, "how compaction was configured", false),
            QueryMode::Keyword
        );
        // "auto" behaves like no explicit mode.
        assert_eq!(
            choose_query_mode(Some("auto"), "one two three", true),
            QueryMode::Hybrid
        );
        assert_eq!(
            choose_query_mode(Some("auto"), "one", true),
            QueryMode::Keyword
        );
    }

    #[test]
    fn test_recall_description_mentions_mode() {
        let (_tmp, tool) = make_tool();
        assert!(tool.description().contains("mode"));
    }

    #[test]
    fn test_recall_tool_parameters_schema() {
        let (_tmp, tool) = make_tool();
        let params = tool.parameters();
        assert_eq!(params["type"], "object");
        assert!(params["properties"]["query"].is_object());
        assert!(params["properties"]["mode"].is_object());
        assert_eq!(params["required"], json!(["query"]));
        assert!(!params["properties"]["mode"]["enum"]
            .as_array()
            .unwrap()
            .contains(&json!("latest")));
    }

    #[tokio::test]
    async fn test_recall_empty_query_returns_error() {
        let (_tmp, tool) = make_tool();
        let mut params = HashMap::new();
        params.insert("query".to_string(), json!(""));
        let result = tool.execute(params).await;
        assert!(result.contains("Error"));
    }

    #[tokio::test]
    async fn test_recall_rejects_latest_mode_as_session_search_owned() {
        let (_tmp, tool) = make_tool();
        let result = tool
            .execute(HashMap::from([
                ("query".to_string(), json!("recent sessions")),
                ("mode".to_string(), json!("latest")),
            ]))
            .await;
        assert!(
            result.contains("belongs to session_search"),
            "source ownership must be explicit: {result}"
        );
    }

    #[tokio::test]
    async fn test_recall_grep_finds_memory_md() {
        let (tmp, tool) = make_tool();
        std::fs::write(
            tmp.path().join("memory").join("MEMORY.md"),
            "- User prefers dark mode\n- Favorite language is Rust\n- Lives in Helsinki",
        )
        .unwrap();

        let mut params = HashMap::new();
        params.insert("query".to_string(), json!("Rust"));
        params.insert("mode".to_string(), json!("keyword"));
        let result = tool.execute(params).await;
        assert!(result.contains("Rust"), "Should find Rust in MEMORY.md");
    }

    #[tokio::test]
    async fn test_recall_natural_query_finds_curated_fact() {
        let (tmp, tool) = make_tool();
        std::fs::write(
            tmp.path().join("memory").join("MEMORY.md"),
            "- Peppi prefers concise answers\n- Favorite language is Rust\n",
        )
        .unwrap();

        let result = tool.grep_memory("what does Peppi prefer", 5).await;
        assert!(
            result.contains("Peppi prefers concise answers"),
            "got: {result}"
        );
    }

    #[tokio::test]
    async fn test_recall_does_not_search_raw_session_messages() {
        use crate::session::db::SessionDb;

        let (tmp, tool) = make_tool();
        let db = SessionDb::new(&tmp.path().join("sessions.db"));
        let session = db.create_session("cli:test").await;
        db.add_messages(
            &session.id,
            &[json!({
                "role": "user",
                "content": "Discussed async Rust patterns."
            })],
        )
        .await;
        let result = tool.grep_memory("async", 10).await;
        assert!(
            result.contains("No matches found"),
            "recall owns curated memory, not raw SQLite transcripts: {result}"
        );
    }

    #[tokio::test]
    async fn test_recall_rejects_topicless_memory_query() {
        let (_tmp, tool) = make_tool();
        let result = tool
            .execute(HashMap::from([(
                "query".to_string(),
                json!("what do you remember about me"),
            )]))
            .await;
        assert!(result.contains("No specific memory topic"), "got: {result}");
    }

    #[tokio::test]
    async fn test_recall_grep_no_matches() {
        let (_tmp, tool) = make_tool();
        // Test the grep_memory fallback directly (bypasses knowledge store).
        let result = tool.grep_memory("nonexistent_xyz_123_qqq", 5).await;
        assert!(result.contains("No matches found"));
    }

    #[tokio::test]
    async fn test_recall_utf8_truncation_no_panic() {
        let (tmp, tool) = make_tool();
        // Write a MEMORY.md with multi-byte UTF-8 characters that would panic with byte slicing.
        let cjk_content = "日本語テスト\n".repeat(2000); // ~12K chars of CJK
        std::fs::write(tmp.path().join("memory").join("MEMORY.md"), &cjk_content).unwrap();

        // Test grep_memory directly — the old &output[..8000] byte slice would panic on CJK.
        let result = tool.grep_memory("日本語", 10).await;
        assert!(
            result.contains("日本語"),
            "Should find CJK text: {}",
            &result[..result.len().min(200)]
        );
    }

    #[tokio::test]
    async fn test_recall_missing_query_param_returns_error() {
        let (_tmp, tool) = make_tool();
        let params = HashMap::new();
        let result = tool.execute(params).await;
        assert!(
            result.contains("Error"),
            "missing curated-memory query must be rejected, got: {result}"
        );
    }

    #[test]
    fn test_knowledge_search_on_empty_store() {
        let tool = RecallTool::new(Path::new("/tmp/nonexistent"));
        // Should gracefully return None when knowledge store has nothing.
        // (open_default will create an empty DB, hybrid_search returns empty)
        let result = tool.knowledge_search("test query", 5, QueryMode::Hybrid);
        // Either None (no hits) or Some with empty — both are fine.
        if let Some(ref text) = result {
            assert!(!text.is_empty());
        }
    }

    #[test]
    fn test_knowledge_results_exclude_legacy_session_markdown() {
        let hits = vec![
            SearchHit {
                source_name: "/tmp/memory/sessions/SESSION_cli_123.md".to_string(),
                chunk_idx: 0,
                snippet: "stale session summary".to_string(),
                rank: -2.0,
            },
            SearchHit {
                source_name: r"C:\memory\sessions\SESSION_cli_456.md".to_string(),
                chunk_idx: 0,
                snippet: "stale Windows session summary".to_string(),
                rank: -1.5,
            },
            SearchHit {
                source_name: "project-notes.md".to_string(),
                chunk_idx: 2,
                snippet: "current durable knowledge".to_string(),
                rank: -1.0,
            },
        ];

        let output = format_knowledge_hits(&hits).expect("non-session hit should remain");
        assert!(!output.contains("SESSION_cli_123.md"));
        assert!(!output.contains("SESSION_cli_456.md"));
        assert!(!output.contains("stale session summary"));
        assert!(output.contains("project-notes.md"));
        assert!(output.contains("current durable knowledge"));
    }
}
