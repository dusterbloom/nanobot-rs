//! Recall tool: semantic and keyword search across all memory.
//!
//! Uses `qmd` (if available) for BM25 and vector search across the memory
//! collection, plus a direct grep fallback over session files and MEMORY.md.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::process::Command;

use super::base::Tool;

/// A step in the recall search pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SearchStep {
    /// BM25 keyword search via `qmd search`
    QmdKeyword,
    /// Vector/semantic search via `qmd vsearch`
    QmdVsearch,
    /// Direct grep over MEMORY.md and session files
    GrepFallback,
}

/// Returns the ordered list of search steps for the given mode.
///
/// In "auto" mode the order is: keyword (BM25) first, then semantic (vsearch),
/// with grep as the final fallback.  Keyword search works without embeddings
/// and covers all indexed sessions, so it should be attempted before vsearch.
pub fn search_order(mode: &str) -> Vec<SearchStep> {
    match mode {
        "semantic" => vec![SearchStep::QmdVsearch, SearchStep::GrepFallback],
        "keyword" => vec![SearchStep::QmdKeyword, SearchStep::GrepFallback],
        _ => vec![
            SearchStep::QmdKeyword,
            SearchStep::QmdVsearch,
            SearchStep::GrepFallback,
        ],
    }
}

/// Split a query into lowercase words suitable for grep matching.
/// Words shorter than 3 characters are excluded to reduce noise.
pub fn query_words(query: &str) -> Vec<String> {
    query
        .split_whitespace()
        .map(|w| w.to_lowercase())
        .filter(|w| w.len() >= 3)
        .collect()
}

/// Tool that searches across all nanobot memory layers.
pub struct RecallTool {
    workspace: PathBuf,
}

impl RecallTool {
    pub fn new(workspace: &Path) -> Self {
        Self {
            workspace: workspace.to_path_buf(),
        }
    }

    /// Try qmd search first (BM25). Returns None if qmd is not available.
    async fn qmd_search(&self, query: &str, n: usize) -> Option<String> {
        let output = Command::new("qmd")
            .args(["search", query, "-n", &n.to_string()])
            .output()
            .await
            .ok()?;

        if !output.status.success() {
            return None;
        }

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        if stdout.trim().is_empty() {
            return None;
        }
        Some(stdout)
    }

    /// Try qmd vsearch (vector/semantic). Returns None if unavailable or no embeddings.
    async fn qmd_vsearch(&self, query: &str, n: usize) -> Option<String> {
        let output = Command::new("qmd")
            .args(["vsearch", query, "-n", &n.to_string()])
            .output()
            .await
            .ok()?;

        if !output.status.success() {
            return None;
        }

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        if stdout.trim().is_empty() || stdout.contains("need embedding") {
            return None;
        }
        Some(stdout)
    }

    /// Fallback: grep through memory files directly.
    async fn grep_memory(&self, query: &str, max_results: usize) -> String {
        let memory_dir = self.workspace.join("memory");
        if !memory_dir.exists() {
            return "No memory directory found.".to_string();
        }

        // Search MEMORY.md
        let memory_file = memory_dir.join("MEMORY.md");
        let mut results: Vec<String> = Vec::new();

        if memory_file.exists() {
            if let Ok(content) = tokio::fs::read_to_string(&memory_file).await {
                let words = query_words(query);
                let matching_lines: Vec<&str> = content
                    .lines()
                    .filter(|line| {
                        let lower = line.to_lowercase();
                        words.iter().any(|w| lower.contains(w.as_str()))
                    })
                    .collect();
                if !matching_lines.is_empty() {
                    results.push(format!("## MEMORY.md\n{}", matching_lines.join("\n")));
                }
            }
        }

        // Search session files (active + archived).
        let sessions_dir = memory_dir.join("sessions");
        let search_dirs = [sessions_dir.clone(), sessions_dir.join("archived")];
        for search_dir in &search_dirs {
            if !search_dir.exists() {
                continue;
            }
            if let Ok(mut entries) = tokio::fs::read_dir(search_dir).await {
                while let Ok(Some(entry)) = entries.next_entry().await {
                    if results.len() >= max_results {
                        break;
                    }
                    let path = entry.path();
                    if path.extension().and_then(|e| e.to_str()) != Some("md") {
                        continue;
                    }
                    if let Ok(content) = tokio::fs::read_to_string(&path).await {
                        let words = query_words(query);
                        if words.is_empty() {
                            continue;
                        }
                        let lower_content = content.to_lowercase();
                        if words.iter().any(|w| lower_content.contains(w.as_str())) {
                            let filename = path
                                .file_name()
                                .and_then(|n| n.to_str())
                                .unwrap_or("unknown");
                            let snippet: Vec<&str> = content
                                .lines()
                                .filter(|line| {
                                    let lower = line.to_lowercase();
                                    words.iter().any(|w| lower.contains(w.as_str()))
                                })
                                .take(5)
                                .collect();
                            results.push(format!("## {}\n{}", filename, snippet.join("\n")));
                        }
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
        "Search memory: long-term facts (MEMORY.md), session summaries, and archived sessions. \
         Run /sessions index first to make historical conversations searchable. \
         Use this to find past context, user preferences, or previous decisions."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query — what you want to recall from memory"
                },
                "mode": {
                    "type": "string",
                    "enum": ["auto", "keyword", "semantic"],
                    "description": "Search mode: 'auto' tries keyword then semantic (default), \
                                   'keyword' for exact matches, 'semantic' for meaning-based search"
                }
            },
            "required": ["query"]
        })
    }

    async fn execute(&self, params: HashMap<String, Value>) -> String {
        let query = match params.get("query").and_then(|v| v.as_str()) {
            Some(q) if !q.trim().is_empty() => q.trim(),
            _ => return "Error: 'query' parameter is required and must be non-empty.".to_string(),
        };

        let mode = params
            .get("mode")
            .and_then(|v| v.as_str())
            .unwrap_or("auto");

        let n = 5;
        let mut sections: Vec<String> = Vec::new();
        let steps = search_order(mode);

        let mut found = false;

        for step in &steps {
            match step {
                SearchStep::QmdKeyword => {
                    if let Some(results) = self.qmd_search(query, n).await {
                        sections.push(format!("## Keyword Search Results\n{}", results));
                        found = true;
                    }
                }
                SearchStep::QmdVsearch => {
                    if let Some(results) = self.qmd_vsearch(query, n).await {
                        sections.push(format!("## Semantic Search Results\n{}", results));
                        found = true;
                    }
                }
                SearchStep::GrepFallback => {
                    if !found {
                        sections.push(self.grep_memory(query, n).await);
                    }
                }
            }
        }

        if sections.is_empty() {
            format!("No results found for '{}'.", query)
        } else {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_tool() -> (TempDir, RecallTool) {
        let tmp = TempDir::new().unwrap();
        // Create memory directory structure.
        let mem_dir = tmp.path().join("memory");
        std::fs::create_dir_all(mem_dir.join("sessions")).unwrap();
        let tool = RecallTool::new(tmp.path());
        (tmp, tool)
    }

    #[test]
    fn test_recall_tool_name() {
        let (_tmp, tool) = make_tool();
        assert_eq!(tool.name(), "recall");
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
    async fn test_recall_grep_finds_session_files() {
        let (tmp, tool) = make_tool();
        std::fs::write(
            tmp.path()
                .join("memory")
                .join("sessions")
                .join("SESSION_abc12345.md"),
            "---\nsession_key: \"cli:test\"\nstatus: active\n---\n\nDiscussed async Rust patterns.",
        )
        .unwrap();

        let mut params = HashMap::new();
        params.insert("query".to_string(), json!("async"));
        params.insert("mode".to_string(), json!("keyword"));
        let result = tool.execute(params).await;
        assert!(
            result.contains("async"),
            "Should find async in session file"
        );
    }

    #[tokio::test]
    async fn test_recall_grep_no_matches() {
        let (_tmp, tool) = make_tool();
        // Test the grep_memory fallback directly (bypasses qmd).
        let result = tool.grep_memory("nonexistent_xyz_123_qqq", 5).await;
        assert!(result.contains("No matches found"));
    }

    #[tokio::test]
    async fn test_recall_grep_finds_archived_sessions() {
        let (tmp, tool) = make_tool();
        let archived_dir = tmp.path().join("memory").join("sessions").join("archived");
        std::fs::create_dir_all(&archived_dir).unwrap();
        std::fs::write(
            archived_dir.join("SESSION_old.md"),
            "---\nsession_key: \"cli:old\"\nstatus: archived\n---\n\nDiscussed UTF-8 encoding.",
        )
        .unwrap();

        // Test grep_memory directly to bypass qmd.
        let result = tool.grep_memory("UTF-8", 10).await;
        assert!(
            result.contains("UTF-8"),
            "Should find UTF-8 in archived session file: {}",
            result
        );
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
    async fn test_recall_missing_query_param() {
        let (_tmp, tool) = make_tool();
        let params = HashMap::new();
        let result = tool.execute(params).await;
        assert!(result.contains("Error"));
    }

    // ---------------------------------------------------------------
    // search_order priority tests (pure function, no IO)
    // ---------------------------------------------------------------

    #[test]
    fn test_auto_mode_tries_keyword_before_vsearch() {
        let steps = search_order("auto");
        let kw_pos = steps.iter().position(|s| *s == SearchStep::QmdKeyword);
        let vs_pos = steps.iter().position(|s| *s == SearchStep::QmdVsearch);
        assert!(kw_pos.is_some(), "auto mode must include QmdKeyword");
        assert!(vs_pos.is_some(), "auto mode must include QmdVsearch");
        assert!(
            kw_pos.unwrap() < vs_pos.unwrap(),
            "In auto mode, keyword (BM25) must be tried BEFORE vsearch (semantic)"
        );
    }

    #[test]
    fn test_auto_mode_pipeline_order() {
        let steps = search_order("auto");
        assert_eq!(
            steps,
            vec![SearchStep::QmdKeyword, SearchStep::QmdVsearch, SearchStep::GrepFallback],
            "auto mode pipeline must be: keyword -> vsearch -> grep"
        );
    }

    #[test]
    fn test_keyword_mode_skips_vsearch() {
        let steps = search_order("keyword");
        assert!(!steps.contains(&SearchStep::QmdVsearch), "keyword mode must NOT include QmdVsearch");
        assert_eq!(steps, vec![SearchStep::QmdKeyword, SearchStep::GrepFallback]);
    }

    #[test]
    fn test_all_modes_end_with_grep_fallback() {
        for mode in &["auto", "keyword", "semantic"] {
            let steps = search_order(mode);
            assert_eq!(
                steps.last(),
                Some(&SearchStep::GrepFallback),
                "mode '{}' must end with GrepFallback",
                mode
            );
        }
    }

    // ---------------------------------------------------------------
    // query_words tests (pure function, no IO)
    // ---------------------------------------------------------------

    #[test]
    fn test_query_words_splits_and_lowercases() {
        let words = query_words("ZeroClaw Adoption Plan");
        assert_eq!(words, vec!["zeroclaw", "adoption", "plan"]);
    }

    #[test]
    fn test_query_words_filters_short_words() {
        let words = query_words("a is the ZeroClaw");
        assert_eq!(words, vec!["the", "zeroclaw"]);
    }

    #[test]
    fn test_query_words_empty_query() {
        let words = query_words("");
        assert!(words.is_empty());
    }

    #[tokio::test]
    async fn test_recall_grep_word_splitting() {
        let (tmp, tool) = make_tool();
        std::fs::write(
            tmp.path().join("memory").join("MEMORY.md"),
            "- Discussed ZeroClaw features\n- Compared with OpenFang\n- Unrelated line about weather",
        )
        .unwrap();

        let result = tool.grep_memory("zeroclaw adoption features", 5).await;
        assert!(result.contains("ZeroClaw"), "Should find line with 'zeroclaw': {}", result);
        assert!(!result.contains("weather"), "Should not match unrelated line: {}", result);
    }
}
