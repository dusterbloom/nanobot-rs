//! Recall tool: semantic and keyword search across all memory.
//!
//! Uses in-process `KnowledgeStore` for hybrid BM25+vector search across
//! indexed documents, canonical SQLite session search, and curated MEMORY.md.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use async_trait::async_trait;
use serde_json::{json, Value};

use super::base::Tool;
use crate::agent::knowledge_store::{KnowledgeStore, SearchHit};

/// Legacy working-memory files may still be present in an existing
/// `knowledge.db`. They are no longer canonical and must not compete with
/// SQLite session history during recall.
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
/// calls such as "what does Peppi prefer?" without falling through to broad
/// session-history search just because the whole sentence is not in the file.
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

/// Top-level routing for a recall call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RecallRoute {
    /// Deterministic: the N most recent sessions straight from sessions.db —
    /// no FTS, no vectors, no query needed.
    Latest { count: usize },
    /// Search the knowledge store with the given query mode.
    Search(QueryMode),
}

/// Route a recall call. `mode="latest"` goes straight to the session DB
/// (count defaults to 3, clamped to 1..=10); everything else searches.
fn choose_route(
    explicit: Option<&str>,
    count: Option<u64>,
    query: &str,
    semantic_available: bool,
) -> RecallRoute {
    match explicit {
        Some("latest") => RecallRoute::Latest {
            count: count.unwrap_or(3).clamp(1, 10) as usize,
        },
        other => RecallRoute::Search(choose_query_mode(other, query, semantic_available)),
    }
}

/// Render the most recent sessions as one line each (key, age, last exchange).
fn format_latest_sessions(
    tails: &[crate::session::db::SessionTail],
    now: chrono::DateTime<chrono::Utc>,
) -> String {
    if tails.is_empty() {
        return "No previous sessions found in the session database.".to_string();
    }
    tails
        .iter()
        .map(|t| {
            format!(
                "- {}",
                crate::agent::continuity::format_continuity_line(t, now)
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
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

/// Tool that searches across all nanobot memory layers.
pub struct RecallTool {
    workspace: PathBuf,
    /// Path to sessions.db — enables the deterministic `mode="latest"` route.
    db_path: Option<PathBuf>,
}

impl RecallTool {
    pub fn new(workspace: &Path) -> Self {
        Self {
            workspace: workspace.to_path_buf(),
            db_path: None,
        }
    }

    /// Attach the session database used by `mode="latest"`.
    pub fn with_db(mut self, db_path: Option<PathBuf>) -> Self {
        self.db_path = db_path;
        self
    }

    /// Deterministic latest-session listing straight from sessions.db.
    async fn latest_sessions(&self, count: usize) -> String {
        let Some(ref db_path) = self.db_path else {
            return "Latest-session recall unavailable: no session database configured."
                .to_string();
        };
        let db = crate::session::db::SessionDb::new(db_path);
        let tails = db.latest_session_tails("", count).await;
        format_latest_sessions(&tails, chrono::Utc::now())
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

    /// Search curated MEMORY.md, falling back to canonical user messages.
    ///
    /// Curated facts are authoritative for cross-session recall. Raw history is
    /// searched only when no curated fact matches, and assistant/tool messages
    /// are excluded so old model claims and tool payloads do not masquerade as
    /// user memory.
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
            if let Some(db_path) = &self.db_path {
                let db = crate::session::db::SessionDb::new(db_path);
                for hit in db
                    .search_messages(query, max_results.saturating_mul(4), None)
                    .await
                {
                    if hit.role != "user" {
                        continue;
                    }
                    let snippet = if hit.snippet.is_empty() {
                        hit.content
                    } else {
                        hit.snippet
                    };
                    results.push(format!(
                        "## {} [{}]\n{}",
                        hit.session_key, hit.role, snippet
                    ));
                    if results.len() >= max_results {
                        break;
                    }
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
        "Search memory: curated long-term facts (MEMORY.md) and raw SQLite session history. \
         Use this to find past context, user preferences, or previous decisions. \
         Multi-word queries automatically use hybrid keyword+semantic search when available; \
         pass mode='keyword' for exact matches or mode='semantic' to force meaning-based search. \
         Pass mode='latest' (no query needed) to list the most recent sessions with their \
         last exchange — use this to continue from a previous session."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query — what you want to recall from memory. \
                                   Required except for mode='latest'."
                },
                "mode": {
                    "type": "string",
                    "enum": ["auto", "keyword", "semantic", "latest"],
                    "description": "Search mode: 'auto' tries hybrid BM25+semantic (default), \
                                   'keyword' for BM25-only exact matches, 'semantic' for meaning-based search, \
                                   'latest' for the most recent sessions (deterministic, no search)"
                },
                "count": {
                    "type": "integer",
                    "description": "For mode='latest': how many recent sessions to return (default 3, max 10)"
                }
            },
            "required": []
        })
    }

    async fn execute(&self, params: HashMap<String, Value>) -> String {
        let explicit_mode = params.get("mode").and_then(|v| v.as_str());
        let count = params.get("count").and_then(|v| v.as_u64());
        let has_query_param = params.contains_key("query");
        let query = params
            .get("query")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim();

        let effective_mode = if explicit_mode.is_none() && !has_query_param {
            Some("latest")
        } else {
            explicit_mode
        };

        let route = choose_route(effective_mode, count, query, cfg!(feature = "semantic"));
        let mode = match route {
            RecallRoute::Latest { count } => return self.latest_sessions(count).await,
            RecallRoute::Search(_) if query.is_empty() => {
                return "Error: 'query' parameter is required and must be non-empty.".to_string()
            }
            RecallRoute::Search(mode) => mode,
        };

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

        // The document index is a fallback, not a peer of curated memory and
        // raw user history. Merging all three made stale indexed context drown
        // out the canonical answer.
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

    async fn make_db_tool(session_key: &str, messages: &[Value]) -> (TempDir, RecallTool) {
        use crate::session::db::SessionDb;

        let tmp = TempDir::new().unwrap();
        std::fs::create_dir_all(tmp.path().join("memory")).unwrap();
        let db_path = tmp.path().join("sessions.db");
        let db = SessionDb::new(&db_path);
        let session = db.create_session(session_key).await;
        db.add_messages(&session.id, messages).await;
        let tool = RecallTool::new(tmp.path()).with_db(Some(db_path));
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

    // --- choose_route: mode="latest" is deterministic, never a search ---

    #[test]
    fn test_choose_route_latest_never_hits_search() {
        // "latest" routes to the deterministic DB path regardless of query
        // shape or semantic capability.
        assert_eq!(
            choose_route(Some("latest"), None, "anything at all", true),
            RecallRoute::Latest { count: 3 }
        );
        assert_eq!(
            choose_route(Some("latest"), None, "", false),
            RecallRoute::Latest { count: 3 }
        );
    }

    #[test]
    fn test_choose_route_latest_count_default_and_cap() {
        assert_eq!(
            choose_route(Some("latest"), Some(5), "", true),
            RecallRoute::Latest { count: 5 }
        );
        // Capped at 10.
        assert_eq!(
            choose_route(Some("latest"), Some(99), "", true),
            RecallRoute::Latest { count: 10 }
        );
        // Zero is nonsensical → clamp to 1.
        assert_eq!(
            choose_route(Some("latest"), Some(0), "", true),
            RecallRoute::Latest { count: 1 }
        );
    }

    #[test]
    fn test_choose_route_non_latest_delegates_to_query_mode() {
        assert_eq!(
            choose_route(Some("keyword"), None, "several words", true),
            RecallRoute::Search(QueryMode::Keyword)
        );
        assert_eq!(
            choose_route(None, None, "how compaction works", true),
            RecallRoute::Search(QueryMode::Hybrid)
        );
    }

    // --- format_latest_sessions ---

    #[test]
    fn test_format_latest_sessions_zero_sessions() {
        let out = format_latest_sessions(&[], chrono::Utc::now());
        assert!(out.contains("No previous sessions"), "got: {out}");
    }

    #[test]
    fn test_format_latest_sessions_lists_key_age_and_tail() {
        use crate::session::db::SessionTail;
        let now = chrono::Utc::now();
        let tails = vec![
            SessionTail {
                session_id: "s2".into(),
                session_key: "cli:oneshot-2".into(),
                updated_at: now - chrono::Duration::hours(1),
                last_user: "second q".into(),
                last_assistant: "second a".into(),
            },
            SessionTail {
                session_id: "s1".into(),
                session_key: "cli:oneshot-1".into(),
                updated_at: now - chrono::Duration::days(1),
                last_user: "first q".into(),
                last_assistant: "first a".into(),
            },
        ];
        let out = format_latest_sessions(&tails, now);
        assert!(out.contains("cli:oneshot-2"), "got: {out}");
        assert!(out.contains("1h ago"), "got: {out}");
        assert!(out.contains("second q"), "got: {out}");
        assert!(out.contains("cli:oneshot-1"), "got: {out}");
        assert!(out.contains("1d ago"), "got: {out}");
    }

    #[tokio::test]
    async fn test_recall_latest_without_db_reports_unavailable() {
        let (_tmp, tool) = make_tool(); // no db configured
        let mut params = HashMap::new();
        params.insert("mode".to_string(), json!("latest"));
        let result = tool.execute(params).await;
        assert!(
            result.contains("no session database"),
            "latest without db must degrade clearly, got: {result}"
        );
    }

    #[tokio::test]
    async fn test_recall_latest_reads_sessions_db() {
        let (_tmp, tool) = make_db_tool(
            "cli:oneshot-77",
            &[
                json!({"role": "user", "content": "unique latest question"}),
                json!({"role": "assistant", "content": "unique latest answer"}),
            ],
        )
        .await;
        let mut params = HashMap::new();
        params.insert("mode".to_string(), json!("latest"));
        let result = tool.execute(params).await;
        assert!(result.contains("cli:oneshot-77"), "got: {result}");
        assert!(result.contains("unique latest question"), "got: {result}");
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
    async fn test_recall_natural_query_prefers_curated_fact_over_session_noise() {
        let (tmp, tool) = make_db_tool(
            "cli:test",
            &[
                json!({"role": "user", "content": "old context about Rust tooling"}),
                json!({"role": "assistant", "content": "Peppi prefers an invented Rust workflow"}),
            ],
        )
        .await;
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
        assert!(!result.contains("invented Rust workflow"), "got: {result}");
        assert!(!result.contains("old context"), "got: {result}");
    }

    #[tokio::test]
    async fn test_recall_grep_finds_user_session_messages() {
        let (_tmp, tool) = make_db_tool(
            "cli:test",
            &[json!({
                "role": "user",
                "content": "Discussed async Rust patterns."
            })],
        )
        .await;
        let result = tool.grep_memory("async", 10).await;
        assert!(
            result.contains("async"),
            "Should find async in the SQLite session"
        );
    }

    #[tokio::test]
    async fn test_recall_session_fallback_excludes_assistant_claims() {
        let (_tmp, tool) = make_db_tool(
            "cli:test",
            &[
                json!({"role": "assistant", "content": "User definitely prefers chartreuse windows"}),
                json!({"role": "tool", "content": "chartreuse tool payload"}),
            ],
        )
        .await;
        let result = tool.grep_memory("chartreuse", 5).await;
        assert!(result.contains("No matches found"), "got: {result}");
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
    async fn test_recall_missing_query_param_defaults_to_latest() {
        let (_tmp, tool) = make_tool();
        let params = HashMap::new();
        let result = tool.execute(params).await;
        assert!(
            result.contains("Latest-session recall unavailable"),
            "missing query should take deterministic latest route, got: {result}"
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
