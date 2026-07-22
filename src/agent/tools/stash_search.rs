//! Bounded retrieval over stashed (truncated) tool results.
//!
//! Large tool outputs are stashed to SQLite and reduced to a head+tail preview
//! in context (`build_tool_result_preview` in `tool_engine.rs`). The existing
//! [`crate::agent::tools::recall_tool_result`] tool recovers the *full* body —
//! but that re-pollutes context. This module adds two tools that query the
//! stashed data *without* ever loading it whole:
//!
//! - [`SearchToolResultTool`]: grep within a stashed result (bounded).
//! - [`SliceToolResultTool`]: extract a specific line range.
//!
//! Both share [`query_stashed_lines`], a single helper that loads the stashed
//! body from SQLite once and hands it to a caller-supplied extraction
//! closure. This keeps line-iteration and bounding logic in one place (DRY)
//! and lets each tool own only its own query semantics (SRP).

use std::collections::HashMap;
use std::path::PathBuf;

use async_trait::async_trait;
use serde_json::{json, Value};

use crate::agent::tools::base::Tool;
use crate::session::db::SessionDb;

// ---------------------------------------------------------------------------
// Bounding constants — central source of truth for both tools.
// ---------------------------------------------------------------------------

/// Maximum matching lines a search may return.
const MAX_SEARCH_RESULTS: usize = 50;

/// Default (and minimum) `max_results` for a search.
const DEFAULT_SEARCH_RESULTS: usize = 10;

/// Maximum lines of surrounding context per match.
const MAX_CONTEXT_LINES: usize = 5;

/// Hard cap on total characters a single tool call may emit. Prevents a
/// high-entropy match set (e.g. a minified line repeated) from blowing up
/// context even after line-count bounding.
const MAX_OUTPUT_CHARS: usize = 4000;

/// Maximum number of lines a single slice may return.
const MAX_SLICE_LINES: usize = 2000;

// ---------------------------------------------------------------------------
// Shared line-query helper (DRY).
// ---------------------------------------------------------------------------

/// Load a stashed tool result from SQLite and apply an extraction closure that
/// receives a line iterator `(line_number, &str)`.
///
/// Returns `None` if the stashed body is missing. The closure produces the
/// final bounded string. Centralising the load + line-split keeps both tools
/// consistent and avoids two copies of the same plumbing.
async fn query_stashed_lines<F>(
    db_path: &PathBuf,
    session_id: &str,
    tool_call_id: &str,
    extract: F,
) -> Option<String>
where
    F: FnOnce(&[(usize, &str)]) -> String,
{
    let db = SessionDb::new(db_path);
    let body = db.load_tool_result(session_id, tool_call_id).await?;
    let lines: Vec<(usize, &str)> = body.lines().enumerate().map(|(i, l)| (i + 1, l)).collect();
    Some(extract(&lines))
}

/// Clamp `max_results` into the legal range `[1, MAX_SEARCH_RESULTS]`.
fn clamp_results(requested: Option<u64>) -> usize {
    requested
        .unwrap_or(DEFAULT_SEARCH_RESULTS as u64)
        .clamp(1, MAX_SEARCH_RESULTS as u64) as usize
}

/// Clamp `context_lines` into `[0, MAX_CONTEXT_LINES]`.
fn clamp_context(requested: Option<u64>) -> usize {
    requested
        .unwrap_or(0)
        .clamp(0, MAX_CONTEXT_LINES as u64) as usize
}

/// Build `context_lines` worth of surrounding lines, prefixed with the match
/// line itself. Lines are numbered and contiguous ranges collapse naturally.
fn render_with_context(
    line_no: usize,
    text: &str,
    lines: &[(usize, &str)],
    context_lines: usize,
) -> String {
    if context_lines == 0 {
        return format!("{}:{}", line_no, text);
    }
    let start = line_no.saturating_sub(context_lines).max(1);
    let end = (line_no + context_lines).min(lines.len());
    let mut out = String::new();
    for (n, l) in lines {
        if *n < start {
            continue;
        }
        if *n > end {
            break;
        }
        let marker = if *n == line_no { ">" } else { " " };
        out.push_str(&format!("{} {}:{}\n", marker, n, l));
    }
    out.trim_end().to_string()
}

/// Truncate `s` to `MAX_OUTPUT_CHARS` on a UTF-8 boundary with an overflow note.
fn bounded(s: String) -> String {
    if s.chars().count() <= MAX_OUTPUT_CHARS {
        return s;
    }
    let mut t: String = s.chars().take(MAX_OUTPUT_CHARS).collect();
    let overflow = s.chars().count() - MAX_OUTPUT_CHARS;
    t.push_str(&format!("\n... (truncated, {} more chars)", overflow));
    t
}

// ===========================================================================
// Tool 1: SearchToolResultTool
// ===========================================================================

/// Search WITHIN a stashed (truncated) tool result without loading the full
/// body into context. Returns matching lines with line numbers.
pub struct SearchToolResultTool {
    db_path: PathBuf,
    session_id: String,
}

impl SearchToolResultTool {
    pub fn with_db(db_path: PathBuf, session_id: String) -> Self {
        Self {
            db_path,
            session_id,
        }
    }
}

#[async_trait]
impl Tool for SearchToolResultTool {
    fn name(&self) -> &str {
        "search_tool_result"
    }

    fn description(&self) -> &str {
        "Search WITHIN a stashed (truncated) tool result without loading it \
         all into context. Use this instead of recall_tool_result when you \
         only need matching lines. Pass the tool_call_id from the \
         [truncated: ...] preview block. Returns up to max_results matching \
         lines with line numbers. Prefer this over recall_tool_result when \
         the full body would be wasteful."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "tool_call_id": {
                    "type": "string",
                    "description": "The tool_call_id from the [truncated: ...] preview block."
                },
                "pattern": {
                    "type": "string",
                    "description": "Substring to search for (case-insensitive)."
                },
                "max_results": {
                    "type": "integer",
                    "description": "Hard cap on matching lines returned (default 10, max 50)."
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Lines of context around each match (default 0, max 5). 0 = match line only."
                }
            },
            "required": ["tool_call_id", "pattern"]
        })
    }

    async fn execute(&self, params: HashMap<String, Value>) -> String {
        let id = match params.get("tool_call_id").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => {
                return "Error: tool_call_id is required. Pass the id from the \
                        [truncated: ...] preview block."
                    .to_string();
            }
        };
        let pattern = match params.get("pattern").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => return "Error: pattern is required.".to_string(),
        };
        let pattern_lower = pattern.to_lowercase();
        let max_results = clamp_results(params.get("max_results").and_then(|v| v.as_u64()));
        let context_lines = clamp_context(params.get("context_lines").and_then(|v| v.as_u64()));

        let result = query_stashed_lines(&self.db_path, &self.session_id, id, |lines| {
            let mut out = String::new();
            let mut shown = 0usize;
            let mut total_matches = 0usize;
            for (line_no, text) in lines {
                if text.to_lowercase().contains(&pattern_lower) {
                    total_matches += 1;
                    if shown >= max_results {
                        continue;
                    }
                    if !out.is_empty() {
                        out.push('\n');
                    }
                    out.push_str(&render_with_context(
                        *line_no,
                        text,
                        lines,
                        context_lines,
                    ));
                    shown += 1;
                }
            }
            if out.is_empty() {
                format!("No matching lines for '{}'.", pattern)
            } else if total_matches > shown {
                format!(
                    "{}\n[{} matches shown, {} total — refine the pattern or raise max_results]",
                    out, shown, total_matches
                )
            } else {
                out
            }
        })
        .await;

        match result {
            Some(s) => bounded(s),
            None => Self::not_found(id),
        }
    }
}

impl SearchToolResultTool {
    fn not_found(id: &str) -> String {
        format!(
            "No stored output for tool_call_id='{id}' in this session. It may \
             have been small enough that it was never stashed, or it may have \
             been removed."
        )
    }
}

// ===========================================================================
// Tool 2: SliceToolResultTool
// ===========================================================================

/// Extract a specific line range from a stashed tool result without loading
/// the full body into context.
pub struct SliceToolResultTool {
    db_path: PathBuf,
    session_id: String,
}

impl SliceToolResultTool {
    pub fn with_db(db_path: PathBuf, session_id: String) -> Self {
        Self {
            db_path,
            session_id,
        }
    }
}

#[async_trait]
impl Tool for SliceToolResultTool {
    fn name(&self) -> &str {
        "slice_tool_result"
    }

    fn description(&self) -> &str {
        "Extract a specific line range from a stashed (truncated) tool result \
         without loading the full body. Use after search_tool_result to read \
         a region in detail, or when you know the exact line numbers. \
         Pass the tool_call_id from the [truncated: ...] preview block."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "tool_call_id": {
                    "type": "string",
                    "description": "The tool_call_id from the [truncated: ...] preview block."
                },
                "start": {
                    "type": "integer",
                    "description": "Starting line number (1-indexed, inclusive)."
                },
                "end": {
                    "type": "integer",
                    "description": "Ending line number (1-indexed, inclusive). Defaults to start + 50."
                }
            },
            "required": ["tool_call_id", "start"]
        })
    }

    async fn execute(&self, params: HashMap<String, Value>) -> String {
        let id = match params.get("tool_call_id").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => {
                return "Error: tool_call_id is required. Pass the id from the \
                        [truncated: ...] preview block."
                    .to_string();
            }
        };
        let start = params
            .get("start")
            .and_then(|v| v.as_u64())
            .unwrap_or(1) as usize;
        let end = params
            .get("end")
            .and_then(|v| v.as_u64())
            .unwrap_or(start as u64 + DEFAULT_SLICE_SPAN as u64) as usize;

        let result = query_stashed_lines(&self.db_path, &self.session_id, id, |lines| {
            let total = lines.len();
            let clamped_start = start.max(1);
            let clamped_end = end.max(clamped_start).min(clamped_start + MAX_SLICE_LINES);

            let mut out = String::new();
            for (line_no, text) in lines {
                if *line_no < clamped_start {
                    continue;
                }
                if *line_no > clamped_end {
                    break;
                }
                if !out.is_empty() {
                    out.push('\n');
                }
                out.push_str(&format!("{}:{}", line_no, text));
            }
            if out.is_empty() {
                format!(
                    "Lines {}-{} are out of range (file has {} lines).",
                    clamped_start, clamped_end, total
                )
            } else {
                out
            }
        })
        .await;

        match result {
            Some(s) => bounded(s),
            None => SearchToolResultTool::not_found(id), // reuse the message
        }
    }
}

/// Default span when `end` is omitted.
const DEFAULT_SLICE_SPAN: usize = 50;

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    async fn make_db() -> (tempfile::TempDir, PathBuf, String) {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("sessions.db");
        let db = SessionDb::new(&db_path);
        let session = db.create_session("cli:search-test").await;
        (dir, db_path, session.id)
    }

    async fn seed(db_path: &PathBuf, session_id: &str, call_id: &str, body: &str) {
        let db = SessionDb::new(db_path);
        assert!(
            db.store_tool_result(session_id, call_id, "exec", body)
                .await
        );
    }

    #[tokio::test]
    async fn search_finds_matching_lines() {
        let (_dir, db_path, sid) = make_db().await;
        let body = "line one\nerror here\nline three\nerror again\n";
        seed(&db_path, &sid, "call_a", body).await;

        let tool = SearchToolResultTool::with_db(db_path, sid);
        let params = HashMap::from([
            ("tool_call_id".to_string(), json!("call_a")),
            ("pattern".to_string(), json!("error")),
        ]);
        let out = tool.execute(params).await;
        assert!(out.contains("2:error here"));
        assert!(out.contains("4:error again"));
    }

    #[tokio::test]
    async fn search_respects_max_results() {
        let (_dir, db_path, sid) = make_db().await;
        let body = "x\n".repeat(20);
        seed(&db_path, &sid, "call_b", &body).await;

        let tool = SearchToolResultTool::with_db(db_path, sid);
        let params = HashMap::from([
            ("tool_call_id".to_string(), json!("call_b")),
            ("pattern".to_string(), json!("x")),
            ("max_results".to_string(), json!(3)),
        ]);
        let out = tool.execute(params).await;
        // 3 shown + the "[3 matches shown, 20 total]" footer.
        assert!(out.contains("[3 matches shown, 20 total"));
    }

    #[tokio::test]
    async fn search_no_matches() {
        let (_dir, db_path, sid) = make_db().await;
        seed(&db_path, &sid, "call_c", "hello world\n").await;

        let tool = SearchToolResultTool::with_db(db_path, sid);
        let params = HashMap::from([
            ("tool_call_id".to_string(), json!("call_c")),
            ("pattern".to_string(), json!("zzz")),
        ]);
        let out = tool.execute(params).await;
        assert!(out.contains("No matching lines"));
    }

    #[tokio::test]
    async fn search_missing_id_returns_helpful_error() {
        let (_dir, db_path, sid) = make_db().await;
        let tool = SearchToolResultTool::with_db(db_path, sid);
        let params = HashMap::from([
            ("tool_call_id".to_string(), json!("ghost")),
            ("pattern".to_string(), json!("x")),
        ]);
        let out = tool.execute(params).await;
        assert!(out.contains("No stored output"));
    }

    #[tokio::test]
    async fn slice_extracts_line_range() {
        let (_dir, db_path, sid) = make_db().await;
        let body: Vec<&str> = (1..=100).map(|_| "data").collect();
        let body = body.join("\n");
        seed(&db_path, &sid, "call_d", &body).await;

        let tool = SliceToolResultTool::with_db(db_path, sid);
        let params = HashMap::from([
            ("tool_call_id".to_string(), json!("call_d")),
            ("start".to_string(), json!(10)),
            ("end".to_string(), json!(12)),
        ]);
        let out = tool.execute(params).await;
        assert!(out.contains("10:data"));
        assert!(out.contains("12:data"));
        assert!(!out.contains("13:data"));
    }

    #[tokio::test]
    async fn search_context_lines_surround_match() {
        let (_dir, db_path, sid) = make_db().await;
        let body = "a\nb\nTARGET\nc\nd\n";
        seed(&db_path, &sid, "call_e", body).await;

        let tool = SearchToolResultTool::with_db(db_path, sid);
        let params = HashMap::from([
            ("tool_call_id".to_string(), json!("call_e")),
            ("pattern".to_string(), json!("target")),
            ("context_lines".to_string(), json!(1)),
        ]);
        let out = tool.execute(params).await;
        assert!(out.contains("> 3:TARGET"));
        assert!(out.contains("  2:b"));
        assert!(out.contains("  4:c"));
    }

    #[tokio::test]
    async fn search_is_case_insensitive() {
        let (_dir, db_path, sid) = make_db().await;
        seed(&db_path, &sid, "call_f", "Fatal ERROR here\n").await;

        let tool = SearchToolResultTool::with_db(db_path, sid);
        let params = HashMap::from([
            ("tool_call_id".to_string(), json!("call_f")),
            ("pattern".to_string(), json!("fatal error")),
        ]);
        let out = tool.execute(params).await;
        assert!(out.contains("1:Fatal ERROR here"));
    }
}
