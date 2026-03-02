//! Recall tool: semantic and keyword search across all memory.
//!
//! Uses `qmd` (if available) for BM25 and vector search across the memory
//! collection, plus a direct grep fallback over session files and MEMORY.md.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::process::Command;

use chrono::{Local, NaiveDate, Duration as ChronoDuration};

use super::base::Tool;

/// A step in the recall search pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SearchStep {
    /// BM25 keyword search via `qmd search`
    QmdKeyword,
    /// Vector/semantic search via `qmd vsearch`
    QmdVsearch,
    /// Hybrid search via `qmd query` (BM25 + semantic + reranking)
    QmdHybrid,
    /// Date-filtered session search
    Temporal,
    /// Direct grep over MEMORY.md and session files
    GrepFallback,
}

/// Date range for temporal search queries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DateRange {
    Today,
    Yesterday,
    LastWeek,
    LastNDays(u32),
    Exact(String),
    Range(String, String),
}

/// Returns the ordered list of search steps for the given mode.
///
/// In "auto" mode the order is: hybrid (BM25+semantic+reranking) first, then
/// keyword-only as fallback (in case embeddings aren't ready), then grep.
pub fn search_order(mode: &str) -> Vec<SearchStep> {
    match mode {
        "semantic" => vec![SearchStep::QmdVsearch, SearchStep::GrepFallback],
        "keyword" => vec![SearchStep::QmdKeyword, SearchStep::GrepFallback],
        "hybrid" => vec![SearchStep::QmdHybrid, SearchStep::GrepFallback],
        "temporal" => vec![SearchStep::Temporal, SearchStep::GrepFallback],
        _ => vec![
            SearchStep::QmdHybrid,
            SearchStep::QmdKeyword,
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

/// Parse a temporal query into a date range and optional keyword filter.
///
/// Recognizes: "today", "yesterday", "last week", "last N days",
/// "YYYY-MM-DD", "YYYY-MM-DD to YYYY-MM-DD", and combinations with keywords.
pub fn parse_temporal_query(query: &str) -> (DateRange, String) {
    let q = query.trim().to_lowercase();

    if let Some(caps) = regex_date_range(&q) {
        return caps;
    }
    if let Some(caps) = regex_exact_date(&q) {
        return caps;
    }

    if q == "today" {
        return (DateRange::Today, String::new());
    }
    if q == "yesterday" {
        return (DateRange::Yesterday, String::new());
    }
    if q.starts_with("today ") {
        return (DateRange::Today, q[6..].trim().to_string());
    }
    if q.starts_with("yesterday ") {
        return (DateRange::Yesterday, q[10..].trim().to_string());
    }
    if q == "last week" {
        return (DateRange::LastWeek, String::new());
    }
    if q.starts_with("last week ") {
        return (DateRange::LastWeek, q[10..].trim().to_string());
    }

    if q.starts_with("last ") {
        let rest = &q[5..];
        if let Some(idx) = rest.find(" day") {
            if let Ok(n) = rest[..idx].trim().parse::<u32>() {
                let after_days = rest[idx..]
                    .trim_start_matches(" days")
                    .trim_start_matches(" day")
                    .trim();
                return (DateRange::LastNDays(n), after_days.to_string());
            }
        }
    }

    // No temporal pattern — treat everything as keyword for today
    (DateRange::Today, query.trim().to_string())
}

fn regex_date_range(q: &str) -> Option<(DateRange, String)> {
    let parts: Vec<&str> = q.splitn(2, " to ").collect();
    if parts.len() == 2 {
        let from = parts[0].trim();
        if from.len() == 10 && NaiveDate::parse_from_str(from, "%Y-%m-%d").is_ok() {
            let rest = parts[1].trim();
            let to_and_kw: Vec<&str> = rest.splitn(2, ' ').collect();
            let to = to_and_kw[0];
            if to.len() == 10 && NaiveDate::parse_from_str(to, "%Y-%m-%d").is_ok() {
                let kw = if to_and_kw.len() > 1 {
                    to_and_kw[1].trim().to_string()
                } else {
                    String::new()
                };
                return Some((DateRange::Range(from.to_string(), to.to_string()), kw));
            }
        }
    }
    None
}

fn regex_exact_date(q: &str) -> Option<(DateRange, String)> {
    if q.len() >= 10 {
        let candidate = &q[..10];
        if NaiveDate::parse_from_str(candidate, "%Y-%m-%d").is_ok() {
            let kw = q[10..].trim().to_string();
            return Some((DateRange::Exact(candidate.to_string()), kw));
        }
    }
    None
}

/// Extract a YYYY-MM-DD date from a session file path.
///
/// Session files are named like `cli-default-2026-02-19.jsonl` or
/// `telegram-380937266-2026-02-22.jsonl`.
pub fn extract_date_from_path(path: &str) -> Option<String> {
    let bytes = path.as_bytes();
    for i in 0..path.len().saturating_sub(9) {
        // Must start with a digit (YYYY-...)
        if !bytes[i].is_ascii_digit() {
            continue;
        }
        let candidate = &path[i..i + 10];
        if NaiveDate::parse_from_str(candidate, "%Y-%m-%d").is_ok() {
            return Some(candidate.to_string());
        }
    }
    None
}

/// Convert a DateRange to (start_date, end_date) as YYYY-MM-DD strings.
pub fn date_range_to_bounds(range: &DateRange) -> (String, String) {
    let today = Local::now().date_naive();
    match range {
        DateRange::Today => {
            let d = today.format("%Y-%m-%d").to_string();
            (d.clone(), d)
        }
        DateRange::Yesterday => {
            let d = (today - ChronoDuration::days(1)).format("%Y-%m-%d").to_string();
            (d.clone(), d)
        }
        DateRange::LastWeek => {
            let start = (today - ChronoDuration::days(7)).format("%Y-%m-%d").to_string();
            let end = today.format("%Y-%m-%d").to_string();
            (start, end)
        }
        DateRange::LastNDays(n) => {
            let start = (today - ChronoDuration::days(*n as i64)).format("%Y-%m-%d").to_string();
            let end = today.format("%Y-%m-%d").to_string();
            (start, end)
        }
        DateRange::Exact(d) => (d.clone(), d.clone()),
        DateRange::Range(from, to) => (from.clone(), to.clone()),
    }
}

/// Filter `--files` output lines by date range. Returns at most `n` lines.
pub fn filter_by_date(files_output: &str, range: &DateRange, n: usize) -> String {
    let (start, end) = date_range_to_bounds(range);
    let filtered: Vec<&str> = files_output
        .lines()
        .filter(|line| {
            if let Some(date) = extract_date_from_path(line) {
                date >= start && date <= end
            } else {
                false
            }
        })
        .take(n)
        .collect();
    filtered.join("\n")
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

    /// Hybrid search via `qmd query` (BM25 + semantic + reranking).
    async fn qmd_query(&self, query: &str, n: usize) -> Option<String> {
        let output = Command::new("qmd")
            .args(["query", query, "-n", &n.to_string()])
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

    /// Temporal search: find sessions by date range with optional keyword filter.
    async fn temporal_search(&self, query: &str, n: usize) -> Option<String> {
        let (date_range, keyword) = parse_temporal_query(query);
        let search_query = if keyword.is_empty() {
            "*".to_string()
        } else {
            keyword
        };

        let fetch_n = n * 4;
        let output = Command::new("qmd")
            .args([
                "search",
                &search_query,
                "-c",
                "sessions",
                "-n",
                &fetch_n.to_string(),
                "--files",
            ])
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

        let filtered = filter_by_date(&stdout, &date_range, n);
        if filtered.is_empty() {
            return None;
        }
        Some(filtered)
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
                    "enum": ["auto", "keyword", "semantic", "hybrid", "temporal"],
                    "description": "Search mode: 'auto' uses hybrid (BM25+semantic+reranking) with keyword fallback, \
                                   'keyword' for exact BM25 matches, 'semantic' for meaning-based, \
                                   'hybrid' for combined search with reranking, \
                                   'temporal' for date-filtered sessions (e.g. 'yesterday', 'last week', '2026-03-01')"
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
                SearchStep::QmdHybrid => {
                    if let Some(results) = self.qmd_query(query, n).await {
                        sections.push(format!("## Hybrid Search Results\n{}", results));
                        found = true;
                    }
                }
                SearchStep::Temporal => {
                    if let Some(results) = self.temporal_search(query, n).await {
                        sections.push(format!("## Temporal Search Results\n{}", results));
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
    fn test_auto_mode_tries_hybrid_first() {
        let steps = search_order("auto");
        assert_eq!(steps[0], SearchStep::QmdHybrid);
        assert_eq!(steps[1], SearchStep::QmdKeyword);
        assert_eq!(steps[2], SearchStep::GrepFallback);
    }

    #[test]
    fn test_auto_mode_pipeline_order() {
        let steps = search_order("auto");
        assert_eq!(
            steps,
            vec![SearchStep::QmdHybrid, SearchStep::QmdKeyword, SearchStep::GrepFallback],
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
        for mode in &["auto", "keyword", "semantic", "hybrid", "temporal"] {
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

    // ---------------------------------------------------------------
    // hybrid / temporal pipeline tests
    // ---------------------------------------------------------------

    #[test]
    fn test_hybrid_mode_pipeline() {
        let steps = search_order("hybrid");
        assert_eq!(steps, vec![SearchStep::QmdHybrid, SearchStep::GrepFallback]);
    }

    #[test]
    fn test_temporal_mode_pipeline() {
        let steps = search_order("temporal");
        assert_eq!(steps, vec![SearchStep::Temporal, SearchStep::GrepFallback]);
    }

    // ---------------------------------------------------------------
    // parse_temporal_query tests (pure function, no IO)
    // ---------------------------------------------------------------

    #[test]
    fn test_parse_temporal_today() {
        let (range, kw) = parse_temporal_query("today");
        assert_eq!(range, DateRange::Today);
        assert!(kw.is_empty());
    }

    #[test]
    fn test_parse_temporal_yesterday() {
        let (range, kw) = parse_temporal_query("yesterday");
        assert_eq!(range, DateRange::Yesterday);
        assert!(kw.is_empty());
    }

    #[test]
    fn test_parse_temporal_yesterday_with_keyword() {
        let (range, kw) = parse_temporal_query("yesterday authentication");
        assert_eq!(range, DateRange::Yesterday);
        assert_eq!(kw, "authentication");
    }

    #[test]
    fn test_parse_temporal_last_week() {
        let (range, kw) = parse_temporal_query("last week");
        assert_eq!(range, DateRange::LastWeek);
        assert!(kw.is_empty());
    }

    #[test]
    fn test_parse_temporal_last_n_days() {
        let (range, kw) = parse_temporal_query("last 3 days");
        assert_eq!(range, DateRange::LastNDays(3));
        assert!(kw.is_empty());
    }

    #[test]
    fn test_parse_temporal_exact_date() {
        let (range, kw) = parse_temporal_query("2026-03-01");
        assert_eq!(range, DateRange::Exact("2026-03-01".to_string()));
        assert!(kw.is_empty());
    }

    #[test]
    fn test_parse_temporal_exact_date_with_keyword() {
        let (range, kw) = parse_temporal_query("2026-03-01 deployment");
        assert_eq!(range, DateRange::Exact("2026-03-01".to_string()));
        assert_eq!(kw, "deployment");
    }

    #[test]
    fn test_parse_temporal_date_range() {
        let (range, kw) = parse_temporal_query("2026-02-25 to 2026-03-01");
        assert_eq!(range, DateRange::Range("2026-02-25".to_string(), "2026-03-01".to_string()));
        assert!(kw.is_empty());
    }

    // ---------------------------------------------------------------
    // extract_date_from_path tests (pure function, no IO)
    // ---------------------------------------------------------------

    #[test]
    fn test_extract_date_from_path_cli() {
        assert_eq!(
            extract_date_from_path("qmd://sessions/cli-default-2026-02-19.jsonl"),
            Some("2026-02-19".to_string())
        );
    }

    #[test]
    fn test_extract_date_from_path_telegram() {
        assert_eq!(
            extract_date_from_path("qmd://sessions/telegram-380937266-2026-02-22.jsonl"),
            Some("2026-02-22".to_string())
        );
    }

    #[test]
    fn test_extract_date_from_path_no_date() {
        assert_eq!(extract_date_from_path("qmd://sessions/unknown.jsonl"), None);
    }

    // ---------------------------------------------------------------
    // filter_by_date tests (pure function, no IO)
    // ---------------------------------------------------------------

    #[test]
    fn test_filter_by_date_exact() {
        let input = "docid1,0.95,qmd://sessions/cli-default-2026-03-01.jsonl,ctx\n\
                     docid2,0.90,qmd://sessions/cli-default-2026-02-28.jsonl,ctx\n\
                     docid3,0.85,qmd://sessions/cli-default-2026-02-27.jsonl,ctx";
        let filtered = filter_by_date(input, &DateRange::Exact("2026-03-01".to_string()), 5);
        assert!(filtered.contains("2026-03-01"));
        assert!(!filtered.contains("2026-02-28"));
    }

    #[test]
    fn test_filter_by_date_range() {
        let input = "docid1,0.95,qmd://sessions/cli-default-2026-03-01.jsonl,ctx\n\
                     docid2,0.90,qmd://sessions/cli-default-2026-02-28.jsonl,ctx\n\
                     docid3,0.85,qmd://sessions/cli-default-2026-02-20.jsonl,ctx";
        let filtered = filter_by_date(
            input,
            &DateRange::Range("2026-02-27".to_string(), "2026-03-01".to_string()),
            5,
        );
        assert!(filtered.contains("2026-03-01"));
        assert!(filtered.contains("2026-02-28"));
        assert!(!filtered.contains("2026-02-20"));
    }
}
