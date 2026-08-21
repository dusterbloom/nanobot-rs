// Error-protocol layer-3 backlog (docs/research/2026-08-06-error-conventions-and-host-bridge.md §3.6):
// the deny regime in Cargo.toml is live; this module still carries pre-existing
// violations of the lints below. Remove this allow as the module migrates onto
// the regime.
// Tracking: docs/error-protocol-backlog.md
#![allow(
    clippy::as_conversions,
    clippy::format_push_string,
    clippy::shadow_reuse
)]
//! Bounded retrieval over stashed (truncated) tool results.
//!
//! Large tool outputs are stashed to SQLite and represented in context by a
//! handle. [`SearchToolResultTool`] is exposed as `inspect_tool_result`: one
//! bounded operation that searches by query or reads a line range without ever
//! loading the exact body into the transcript. The shared
//! [`query_stashed_lines`] helper keeps storage and line iteration in one
//! place.

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

/// Default line span when an inspection omits `end_line`.
const DEFAULT_SLICE_SPAN: usize = 50;

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
    requested.unwrap_or(0).clamp(0, MAX_CONTEXT_LINES as u64) as usize
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

/// Render one self-contained, line-aligned artifact page. The continuation is
/// derived from the last COMPLETE source line that made it into the bounded
/// page; applying [`bounded`] afterwards would advertise rows the model never
/// saw and make them unreachable.
fn render_slice_page(
    lines: &[(usize, &str)],
    artifact_tool_call_id: &str,
    start: usize,
    end: usize,
) -> String {
    let total = lines.len();
    let clamped_start = start.max(1);
    let clamped_end = end
        .max(clamped_start)
        .min(clamped_start.saturating_add(MAX_SLICE_LINES.saturating_sub(1)));

    let header = |last_line: usize| {
        let page_status = if last_line < total {
            format!("next={}", last_line + 1)
        } else {
            "end".to_string()
        };
        format!(
            "[source={artifact_tool_call_id} lines {clamped_start}-{last_line}/{total} {page_status}]\n"
        )
    };

    if clamped_start > total {
        return format!(
            "[source={artifact_tool_call_id} lines {clamped_start}-{clamped_end} out of range (file has {total} lines)]"
        );
    }

    let mut rows = Vec::new();
    let mut last_line = None;
    for (line_no, text) in lines {
        if *line_no < clamped_start {
            continue;
        }
        if *line_no > clamped_end {
            break;
        }
        let row = format!("{}:{}", line_no, text);
        let mut candidate_rows = rows.clone();
        candidate_rows.push(row);
        let candidate = format!("{}{}", header(*line_no), candidate_rows.join("\n"));
        if candidate.chars().count() > MAX_OUTPUT_CHARS {
            break;
        }
        rows = candidate_rows;
        last_line = Some(*line_no);
    }

    match last_line {
        Some(last_line) => format!("{}{}", header(last_line), rows.join("\n")),
        None => format!(
            "[source={artifact_tool_call_id} line {clamped_start} exceeds the {MAX_OUTPUT_CHARS}-char page limit; next={clamped_start}]"
        ),
    }
}

// ===========================================================================
// Model-facing bounded result inspection
// ===========================================================================

/// Inspect a stashed tool result without loading its full body into context.
/// A query searches matching lines; omitting it returns a bounded line page.
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
        "inspect_tool_result"
    }

    fn description(&self) -> &str {
        "Read a small part of a prior tool result. Pass tool_call_id from its \
         TOOL_RESULT_HANDLE. Add query to find matching lines, or start_line \
         and end_line to read a range. Output is always bounded; the complete \
         source remains stored for another narrower inspection."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "tool_call_id": {
                    "type": "string",
                    "description": "The tool_call_id from the [truncated: ...] preview block."
                },
                "query": {
                    "type": "string",
                    "description": "Substring to search for (case-insensitive)."
                },
                "start_line": {
                    "type": "integer",
                    "description": "First line to read when query is omitted (default 1)."
                },
                "end_line": {
                    "type": "integer",
                    "description": "Last line to read when query is omitted (default start_line + 50)."
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
            "required": ["tool_call_id"]
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
        let query = params.get("query").and_then(|v| v.as_str());
        let result = if let Some(query) = query {
            let query_lower = query.to_lowercase();
            let max_results = clamp_results(params.get("max_results").and_then(|v| v.as_u64()));
            let context_lines = clamp_context(params.get("context_lines").and_then(|v| v.as_u64()));
            query_stashed_lines(&self.db_path, &self.session_id, id, |lines| {
                let mut out = String::new();
                let mut shown = 0usize;
                let mut total_matches = 0usize;
                for (line_no, text) in lines {
                    if text.to_lowercase().contains(&query_lower) {
                        total_matches += 1;
                        if shown >= max_results {
                            continue;
                        }
                        if !out.is_empty() {
                            out.push('\n');
                        }
                        out.push_str(&render_with_context(*line_no, text, lines, context_lines));
                        shown += 1;
                    }
                }
                if out.is_empty() {
                    format!("No matching lines for '{query}'.")
                } else if total_matches > shown {
                    format!(
                        "{}\n[{} matches shown, {} total — refine query or narrow the range]",
                        out, shown, total_matches
                    )
                } else {
                    out
                }
            })
            .await
        } else {
            let start = params
                .get("start_line")
                .and_then(|v| v.as_u64())
                .unwrap_or(1) as usize;
            let end = params
                .get("end_line")
                .and_then(|v| v.as_u64())
                .unwrap_or(start as u64 + DEFAULT_SLICE_SPAN as u64) as usize;
            query_stashed_lines(&self.db_path, &self.session_id, id, |lines| {
                render_slice_page(lines, id, start, end)
            })
            .await
        };

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
        assert!(matches!(
            db.store_tool_result_immutable(session_id, call_id, "exec", body)
                .await,
            crate::session::db::StoredResult::Stored { .. }
        ));
    }

    #[tokio::test]
    async fn search_finds_matching_lines() {
        let (_dir, db_path, sid) = make_db().await;
        let body = "line one\nerror here\nline three\nerror again\n";
        seed(&db_path, &sid, "call_a", body).await;

        let tool = SearchToolResultTool::with_db(db_path, sid);
        let params = HashMap::from([
            ("tool_call_id".to_string(), json!("call_a")),
            ("query".to_string(), json!("error")),
        ]);
        let out = tool.execute(params).await;
        assert_eq!(tool.name(), "inspect_tool_result");
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
            ("query".to_string(), json!("x")),
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
            ("query".to_string(), json!("zzz")),
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
            ("query".to_string(), json!("x")),
        ]);
        let out = tool.execute(params).await;
        assert!(out.contains("No stored output"));
    }

    #[tokio::test]
    async fn inspect_extracts_line_range() {
        let (_dir, db_path, sid) = make_db().await;
        let body: Vec<&str> = (1..=100).map(|_| "data").collect();
        let body = body.join("\n");
        seed(&db_path, &sid, "call_d", &body).await;

        let tool = SearchToolResultTool::with_db(db_path, sid);
        let params = HashMap::from([
            ("tool_call_id".to_string(), json!("call_d")),
            ("start_line".to_string(), json!(10)),
            ("end_line".to_string(), json!(12)),
        ]);
        let out = tool.execute(params).await;
        assert!(out.contains("10:data"));
        assert!(out.contains("12:data"));
        assert!(!out.contains("13:data"));
    }

    #[tokio::test]
    async fn inspect_reports_continuation_position() {
        let (_dir, db_path, sid) = make_db().await;
        let body: Vec<&str> = (1..=100).map(|_| "data").collect();
        seed(&db_path, &sid, "call_page", &body.join("\n")).await;

        let tool = SearchToolResultTool::with_db(db_path, sid);
        let first_page = tool
            .execute(HashMap::from([
                ("tool_call_id".to_string(), json!("call_page")),
                ("start_line".to_string(), json!(10)),
            ]))
            .await;
        assert!(
            first_page.starts_with("[source=call_page lines 10-60/100 next=61]"),
            "inspection must tell the model how to request the next page: {first_page}"
        );

        let last_page = tool
            .execute(HashMap::from([
                ("tool_call_id".to_string(), json!("call_page")),
                ("start_line".to_string(), json!(90)),
            ]))
            .await;
        assert!(
            last_page.starts_with("[source=call_page lines 90-100/100 end]"),
            "inspection must tell the model that paging is complete: {last_page}"
        );
    }

    #[tokio::test]
    async fn inspect_repeats_immutable_artifact_id_for_follow_up_calls() {
        let (_dir, db_path, sid) = make_db().await;
        seed(&db_path, &sid, "artifact_read_7", "alpha\nbeta\ngamma").await;

        let tool = SearchToolResultTool::with_db(db_path, sid);
        let out = tool
            .execute(HashMap::from([
                ("tool_call_id".to_string(), json!("artifact_read_7")),
                ("start_line".to_string(), json!(2)),
            ]))
            .await;

        assert!(
            out.starts_with("[source=artifact_read_7 "),
            "every page must repeat the immutable source artifact id: {out}"
        );
    }

    #[tokio::test]
    async fn inspect_wide_lines_never_advertises_unemitted_next_start() {
        let (_dir, db_path, sid) = make_db().await;
        let body = (1..=100)
            .map(|line| format!("{line}-{}", "x".repeat(180)))
            .collect::<Vec<_>>()
            .join("\n");
        seed(&db_path, &sid, "wide_artifact", &body).await;

        let tool = SearchToolResultTool::with_db(db_path, sid);
        let out = tool
            .execute(HashMap::from([
                ("tool_call_id".to_string(), json!("wide_artifact")),
                ("start_line".to_string(), json!(1)),
                ("end_line".to_string(), json!(100)),
            ]))
            .await;

        assert!(out.chars().count() <= MAX_OUTPUT_CHARS);
        let next_start = out
            .split("next=")
            .nth(1)
            .and_then(|tail| tail.split(|c: char| !c.is_ascii_digit()).next())
            .and_then(|n| n.parse::<usize>().ok())
            .expect("wide page must expose a continuation position");
        let last_emitted = out
            .lines()
            .filter_map(|line| line.split_once(':'))
            .filter_map(|(line_no, _)| line_no.parse::<usize>().ok())
            .last()
            .expect("wide page must contain at least one whole source row");
        assert_eq!(
            next_start,
            last_emitted + 1,
            "next_start must follow the final row actually emitted: {out}"
        );
    }

    #[tokio::test]
    async fn search_context_lines_surround_match() {
        let (_dir, db_path, sid) = make_db().await;
        let body = "a\nb\nTARGET\nc\nd\n";
        seed(&db_path, &sid, "call_e", body).await;

        let tool = SearchToolResultTool::with_db(db_path, sid);
        let params = HashMap::from([
            ("tool_call_id".to_string(), json!("call_e")),
            ("query".to_string(), json!("target")),
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
            ("query".to_string(), json!("fatal error")),
        ]);
        let out = tool.execute(params).await;
        assert!(out.contains("1:Fatal ERROR here"));
    }
}
