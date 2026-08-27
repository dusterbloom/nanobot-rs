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
use regex::{Regex, RegexBuilder};
use serde_json::{json, Value};

use crate::agent::tools::base::{Tool, ToolContext, ToolResult};
use crate::errors::ToolError;
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

/// Headroom reserved inside `MAX_OUTPUT_CHARS` for the loud paging-guidance
/// footer on sequential pages, so header + body + footer always stays within
/// the cap. Sized above the longest END/MORE footer (incl. multi-digit
/// offsets), so pages shrink by at most this much.
const GUIDANCE_RESERVE: usize = 200;

/// Maximum number of lines a single slice may return.
const MAX_SLICE_LINES: usize = 2000;

/// Default line span when an inspection omits `end_line`.
const DEFAULT_SLICE_SPAN: usize = 50;

// ---------------------------------------------------------------------------
// Shared line-query helper (DRY).
// ---------------------------------------------------------------------------

/// One physical source line and its zero-based character offset in the exact
/// stashed body. The offset lets a wide line fall back to a recoverable
/// character page without changing the normal line-range contract.
#[derive(Clone, Copy)]
struct StashedLine<'a> {
    number: usize,
    start_char: usize,
    text: &'a str,
}

/// Split like [`str::lines`] while retaining each line's character offset in
/// the original body. Empty output has no lines, as with [`str::lines`].
fn stashed_lines(body: &str) -> Vec<StashedLine<'_>> {
    if body.is_empty() {
        return Vec::new();
    }

    let mut start_char = 0;
    body.split_inclusive('\n')
        .enumerate()
        .map(|(index, fragment)| {
            let line = StashedLine {
                number: index + 1,
                start_char,
                text: fragment.trim_end_matches(|ch| ch == '\r' || ch == '\n'),
            };
            start_char += fragment.chars().count();
            line
        })
        .collect()
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
    lines: &[StashedLine<'_>],
    context_lines: usize,
) -> String {
    if context_lines == 0 {
        return format!("{}:{}", line_no, text);
    }
    let start = line_no.saturating_sub(context_lines).max(1);
    let end = (line_no + context_lines).min(lines.len());
    let mut out = String::new();
    for line in lines {
        if line.number < start {
            continue;
        }
        if line.number > end {
            break;
        }
        let marker = if line.number == line_no { ">" } else { " " };
        out.push_str(&format!("{} {}:{}\n", marker, line.number, line.text));
    }
    out.trim_end().to_string()
}

/// Render a bounded page from an exact body using a zero-based character
/// offset. This is the recovery path for a physical line that cannot fit on a
/// line page; `next_char` always advances beyond the visible text.
///
/// `paged` selects the loud sequential-paging guidance: when true, an
/// unmissable END-OF-SOURCE or MORE-CONTENT footer is appended (and body size
/// is reduced by `GUIDANCE_RESERVE` to keep the whole page within the cap).
/// Query-centered pages pass `paged = false` — they answer a search, they are
/// not a read-through, so "stop paging" guidance would be noise there.
fn render_char_page(body: &str, artifact_tool_call_id: &str, start_char: usize, paged: bool) -> String {
    let total = body.chars().count();
    if start_char >= total {
        return format!(
            "[source={artifact_tool_call_id} chars {start_char} out of range (output has {total} chars)]"
        );
    }

    let header = |end_char: usize| {
        let page_status = if end_char < total {
            format!("next_char={end_char}")
        } else {
            "end".to_string()
        };
        format!(
            "[source={artifact_tool_call_id} chars {start_char}..{end_char}/{total} {page_status}]\n"
        )
    };

    let page_budget = MAX_OUTPUT_CHARS.saturating_sub(if paged { GUIDANCE_RESERVE } else { 0 });
    let mut text = String::new();
    let mut end_char = start_char;
    for ch in body.chars().skip(start_char) {
        text.push(ch);
        let candidate_end = end_char + 1;
        if header(candidate_end).chars().count() + text.chars().count() > page_budget {
            text.pop();
            break;
        }
        end_char = candidate_end;
    }

    if text.is_empty() {
        return format!(
            "[source={artifact_tool_call_id} char page at {start_char} cannot fit within the {MAX_OUTPUT_CHARS}-char limit]"
        );
    }

    if !paged {
        return format!("{}{text}", header(end_char));
    }

    let footer = if end_char < total {
        format!(
            "\n[MORE CONTENT AHEAD — this page stops mid-source. To read the rest, call again with start_char={end_char}.]"
        )
    } else {
        "\n[END OF SOURCE — you have now read the complete stored output. Do not request further pages of this source; answer from what you have.]".to_string()
    };

    format!("{}{text}{footer}", header(end_char))
}

/// Render one self-contained, line-aligned artifact page. The continuation is
/// derived from the last COMPLETE source line that made it into the bounded
/// page; applying [`bounded`] afterwards would advertise rows the model never
/// saw and make them unreachable.
fn render_slice_page(
    body: &str,
    lines: &[StashedLine<'_>],
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
    for line in lines {
        if line.number < clamped_start {
            continue;
        }
        if line.number > clamped_end {
            break;
        }
        let row = format!("{}:{}", line.number, line.text);
        let mut candidate_rows = rows.clone();
        candidate_rows.push(row);
        let candidate = format!("{}{}", header(line.number), candidate_rows.join("\n"));
        if candidate.chars().count() > MAX_OUTPUT_CHARS.saturating_sub(GUIDANCE_RESERVE) {
            break;
        }
        rows = candidate_rows;
        last_line = Some(line.number);
    }

    match last_line {
        Some(last_line) => {
            let footer = if last_line < total {
                format!(
                    "\n[MORE CONTENT AHEAD — this page stops at line {last_line} of {total}. To read the rest, call again with start_line={}.]",
                    last_line + 1
                )
            } else {
                format!(
                    "\n[END OF SOURCE — you have now read all {total} lines of the stored output. Do not request further pages of this source; answer from what you have.]"
                )
            };
            format!("{}{}{}", header(last_line), rows.join("\n"), footer)
        }
        None => render_char_page(
            body,
            artifact_tool_call_id,
            lines
                .iter()
                .find(|line| line.number == clamped_start)
                .map_or(0, |line| line.start_char),
            true,
        ),
    }
}

/// Return a bounded query result. A match or requested context that cannot fit
/// becomes a character page beginning at the match, so no matching bytes are
/// silently truncated or made unreachable.
fn render_query_results(
    body: &str,
    lines: &[StashedLine<'_>],
    artifact_tool_call_id: &str,
    query: &str,
    matcher: &Regex,
    max_results: usize,
    context_lines: usize,
) -> String {
    let matches: Vec<(StashedLine<'_>, usize)> = lines
        .iter()
        .filter_map(|line| {
            matcher.find(line.text).map(|found| {
                let match_offset = line.text[..found.start()].chars().count();
                (*line, match_offset)
            })
        })
        .collect();

    let Some((first_match, first_match_offset)) = matches.first().copied() else {
        return format!(
            "No matching lines for '{query}'. Note: query is a literal substring (regex characters are escaped, so patterns like '\\.' search for the backslash-dot text itself). If you meant to read the source by position, omit query and use start_line/end_line or start_char."
        );
    };

    if let Some((wide_match, wide_match_offset)) = matches.iter().copied().find(|(line, _)| {
        render_with_context(line.number, line.text, lines, context_lines)
            .chars()
            .count()
            > MAX_OUTPUT_CHARS
    }) {
        return render_char_page(
            body,
            artifact_tool_call_id,
            wide_match.start_char + wide_match_offset,
            false,
        );
    }

    let total_matches = matches.len();
    let limit = total_matches.min(max_results);
    let mut rendered = String::new();
    let mut shown = 0usize;
    for (line, _) in matches.into_iter().take(limit) {
        let row = render_with_context(line.number, line.text, lines, context_lines);
        let candidate = if rendered.is_empty() {
            row
        } else {
            format!("{rendered}\n{row}")
        };
        let candidate_shown = shown + 1;
        let footer = if total_matches > candidate_shown {
            format!(
                "\n[{candidate_shown} matches shown, {total_matches} total — refine query or narrow the range]"
            )
        } else {
            String::new()
        };
        if candidate.chars().count() + footer.chars().count() > MAX_OUTPUT_CHARS {
            break;
        }
        rendered = candidate;
        shown = candidate_shown;
    }

    if shown == 0 {
        return render_char_page(
            body,
            artifact_tool_call_id,
            first_match.start_char + first_match_offset,
            false,
        );
    }
    if total_matches > shown {
        rendered.push_str(&format!(
            "\n[{shown} matches shown, {total_matches} total — refine query or narrow the range]"
        ));
    }
    rendered
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
         and end_line to read a range. If a page returns next_char, pass that \
         zero-based offset as start_char to continue a long line. Output is \
         always bounded; the complete source remains stored for another \
         narrower inspection."
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
                    "description": "Substring to find, case-insensitive. Matched as a LITERAL substring — regex metacharacters like '.', '|', '\\' are escaped, not interpreted; do not write regex patterns. To read by position instead of searching, omit this field and use start_line/end_line (or start_char after a next_char)."
                },
                "start_line": {
                    "type": "integer",
                    "description": "First line to read when query is omitted (default 1)."
                },
                "end_line": {
                    "type": "integer",
                    "description": "Last line to read when query is omitted (default start_line + 50)."
                },
                "start_char": {
                    "type": "integer",
                    "description": "Continue a long-line page at this zero-based character offset. Use only after output gives next_char."
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

    async fn execute(&self, params: HashMap<String, Value>, _ctx: &ToolContext) -> ToolResult {
        let id = match params.get("tool_call_id").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => {
                return Err(ToolError::InvalidArgs {
                    message: "tool_call_id is required. Pass the id from the \
                        [truncated: ...] preview block."
                        .to_string(),
                })
            }
        };
        let body = SessionDb::new(&self.db_path)
            .load_tool_result(&self.session_id, id)
            .await;
        let result = if let Some(body) = body {
            let lines = stashed_lines(&body);
            // An empty query is an omitted query: models send `query: ""`
            // alongside start_line when they mean line paging, and the empty
            // regex matches EVERY line — the wide-line fallback then returns
            // the same char window no matter what start_line says, stranding
            // the caller in a repeat loop (session 20260827_201333_10ec08).
            let query = params
                .get("query")
                .and_then(|v| v.as_str())
                .filter(|q| !q.trim().is_empty());
            if let Some(query) = query {
                let matcher = match RegexBuilder::new(&regex::escape(query))
                    .case_insensitive(true)
                    .build()
                {
                    Ok(matcher) => matcher,
                    Err(_) => {
                        return Err(ToolError::InvalidArgs {
                            message: "query could not be compiled.".to_string(),
                        })
                    }
                };
                let max_results = clamp_results(params.get("max_results").and_then(|v| v.as_u64()));
                let context_lines =
                    clamp_context(params.get("context_lines").and_then(|v| v.as_u64()));
                render_query_results(
                    &body,
                    &lines,
                    id,
                    query,
                    &matcher,
                    max_results,
                    context_lines,
                )
            } else if let Some(start_char) = params.get("start_char").and_then(|v| v.as_u64()) {
                render_char_page(&body, id, start_char as usize, true)
            } else {
                let start = params
                    .get("start_line")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(1) as usize;
                let end = params
                    .get("end_line")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(start as u64 + DEFAULT_SLICE_SPAN as u64)
                    as usize;
                render_slice_page(&body, &lines, id, start, end)
            }
        } else {
            // Success channel by design: teaches the model why nothing was
            // stashed instead of a bare error.
            Self::not_found(id)
        };
        Ok(result.into())
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
        let out = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
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
        let out = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
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
        let out = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
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
        let out = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
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
        let out = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
        assert!(out.contains("10:data"));
        assert!(out.contains("12:data"));
        assert!(!out.contains("13:data"));
    }

    #[tokio::test]
    async fn inspect_empty_query_falls_back_to_line_range() {
        // Session 20260827_201333_10ec08 trap: models send `query: ""` with
        // `start_line` meaning line paging. The empty regex matched every
        // line, and with a wide line present the wide-line fallback returned
        // the SAME char window for every start_line — an unrecoverable loop.
        // An empty query must behave like an omitted one.
        let (_dir, db_path, sid) = make_db().await;
        let mut body = "x".repeat(MAX_OUTPUT_CHARS + 1000); // one wide line
        for i in 2..=60 {
            body.push_str(&format!("\nline-{i}"));
        }
        seed(&db_path, &sid, "call_wide", &body).await;

        let tool = SearchToolResultTool::with_db(db_path, sid);
        for (start, expected) in [(50, "50:line-50"), (55, "55:line-55")] {
            let params = HashMap::from([
                ("tool_call_id".to_string(), json!("call_wide")),
                ("query".to_string(), json!("")),
                ("start_line".to_string(), json!(start)),
                ("max_results".to_string(), json!(80)),
                ("context_lines".to_string(), json!(3)),
            ]);
            let out = crate::agent::tools::base::render_result(
                tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                    .await,
            );
            assert!(out.contains(expected), "start_line={start}: {out}");
            assert!(
                out.contains(&format!("[source=call_wide lines {start}-")),
                "start_line={start} must page, not repeat a char window: {out}"
            );
        }
    }

    #[tokio::test]
    async fn inspect_sequential_pages_carry_loud_end_and_more_guidance() {
        let (_dir, db_path, sid) = make_db().await;
        // 200 x ~60-char lines ≈ 12K chars: forces a partial first page.
        let body: Vec<&str> = (0..200).map(|_| "data data data data data data").collect();
        seed(&db_path, &sid, "call_loud", &body.join("\n")).await;

        let tool = SearchToolResultTool::with_db(db_path, sid);
        // Partial page: must point at the exact next line.
        let partial = crate::agent::tools::base::render_result(
            tool.execute(
                HashMap::from([
                    ("tool_call_id".to_string(), json!("call_loud")),
                    ("start_line".to_string(), json!(1)),
                    ("end_line".to_string(), json!(100)),
                ]),
                &crate::agent::tools::base::ToolContext::sandbox(),
            )
            .await,
        );
        assert!(
            partial.contains("[MORE CONTENT AHEAD"),
            "partial page must warn there is more: {partial}"
        );

        // Page reaching the end: must say END OF SOURCE loudly.
        let last = crate::agent::tools::base::render_result(
            tool.execute(
                HashMap::from([
                    ("tool_call_id".to_string(), json!("call_loud")),
                    ("start_line".to_string(), json!(195)),
                    ("end_line".to_string(), json!(200)),
                ]),
                &crate::agent::tools::base::ToolContext::sandbox(),
            )
            .await,
        );
        assert!(
            last.contains("[END OF SOURCE"),
            "final page must announce the end: {last}"
        );
        assert!(
            last.chars().count() <= MAX_OUTPUT_CHARS,
            "guidance must fit inside the output cap"
        );
    }

    #[tokio::test]
    async fn inspect_reports_continuation_position() {
        let (_dir, db_path, sid) = make_db().await;
        let body: Vec<&str> = (1..=100).map(|_| "data").collect();
        seed(&db_path, &sid, "call_page", &body.join("\n")).await;

        let tool = SearchToolResultTool::with_db(db_path, sid);
        let first_page = crate::agent::tools::base::render_result(
            tool.execute(
                HashMap::from([
                    ("tool_call_id".to_string(), json!("call_page")),
                    ("start_line".to_string(), json!(10)),
                ]),
                &crate::agent::tools::base::ToolContext::sandbox(),
            )
            .await,
        );
        assert!(
            first_page.starts_with("[source=call_page lines 10-60/100 next=61]"),
            "inspection must tell the model how to request the next page: {first_page}"
        );

        let last_page = crate::agent::tools::base::render_result(
            tool.execute(
                HashMap::from([
                    ("tool_call_id".to_string(), json!("call_page")),
                    ("start_line".to_string(), json!(90)),
                ]),
                &crate::agent::tools::base::ToolContext::sandbox(),
            )
            .await,
        );
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
        let out = crate::agent::tools::base::render_result(
            tool.execute(
                HashMap::from([
                    ("tool_call_id".to_string(), json!("artifact_read_7")),
                    ("start_line".to_string(), json!(2)),
                ]),
                &crate::agent::tools::base::ToolContext::sandbox(),
            )
            .await,
        );

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
        let out = crate::agent::tools::base::render_result(
            tool.execute(
                HashMap::from([
                    ("tool_call_id".to_string(), json!("wide_artifact")),
                    ("start_line".to_string(), json!(1)),
                    ("end_line".to_string(), json!(100)),
                ]),
                &crate::agent::tools::base::ToolContext::sandbox(),
            )
            .await,
        );

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
    async fn inspect_single_wide_line_pages_by_character_offset() {
        let (_dir, db_path, sid) = make_db().await;
        let body = format!("{}TAIL", "x".repeat(MAX_OUTPUT_CHARS + 300));
        seed(&db_path, &sid, "wide_single_line", &body).await;

        let tool = SearchToolResultTool::with_db(db_path, sid);
        let first = crate::agent::tools::base::render_result(
            tool.execute(
                HashMap::from([("tool_call_id".to_string(), json!("wide_single_line"))]),
                &crate::agent::tools::base::ToolContext::sandbox(),
            )
            .await,
        );

        assert!(
            first.starts_with("[source=wide_single_line chars 0.."),
            "wide single-line output must expose a character page: {first}"
        );
        assert!(first.chars().count() <= MAX_OUTPUT_CHARS);
        let next_char = first
            .split("next_char=")
            .nth(1)
            .and_then(|tail| tail.split(|c: char| !c.is_ascii_digit()).next())
            .and_then(|n| n.parse::<usize>().ok())
            .expect("first character page must expose a continuation offset");

        let second = crate::agent::tools::base::render_result(
            tool.execute(
                HashMap::from([
                    ("tool_call_id".to_string(), json!("wide_single_line")),
                    ("start_char".to_string(), json!(next_char)),
                ]),
                &crate::agent::tools::base::ToolContext::sandbox(),
            )
            .await,
        );

        assert!(second.contains("TAIL"), "tail must be reachable: {second}");
        assert!(
            second.contains(" end]"),
            "last page must terminate: {second}"
        );
        assert!(second.chars().count() <= MAX_OUTPUT_CHARS);
    }

    #[tokio::test]
    async fn search_wide_line_centers_a_recoverable_page_on_the_match() {
        let (_dir, db_path, sid) = make_db().await;
        let body = format!("{}NEEDLE", "x".repeat(MAX_OUTPUT_CHARS + 200));
        seed(&db_path, &sid, "wide_query", &body).await;

        let tool = SearchToolResultTool::with_db(db_path, sid);
        let out = crate::agent::tools::base::render_result(
            tool.execute(
                HashMap::from([
                    ("tool_call_id".to_string(), json!("wide_query")),
                    ("query".to_string(), json!("needle")),
                ]),
                &crate::agent::tools::base::ToolContext::sandbox(),
            )
            .await,
        );

        assert!(
            out.starts_with("[source=wide_query chars 4200.."),
            "wide query must locate the matching character range: {out}"
        );
        assert!(
            out.contains("NEEDLE"),
            "matching text must be visible: {out}"
        );
        assert!(out.chars().count() <= MAX_OUTPUT_CHARS);
    }

    #[tokio::test]
    async fn search_finds_a_wide_match_after_a_short_match() {
        let (_dir, db_path, sid) = make_db().await;
        let body = format!(
            "TAIL_ONLY_MATCH short\n{}TAIL_ONLY_MATCH",
            "x".repeat(MAX_OUTPUT_CHARS + 200)
        );
        seed(&db_path, &sid, "mixed_query", &body).await;

        let tool = SearchToolResultTool::with_db(db_path, sid);
        let out = crate::agent::tools::base::render_result(
            tool.execute(
                HashMap::from([
                    ("tool_call_id".to_string(), json!("mixed_query")),
                    ("query".to_string(), json!("tail_only_match")),
                ]),
                &crate::agent::tools::base::ToolContext::sandbox(),
            )
            .await,
        );

        assert!(
            out.starts_with("[source=mixed_query chars "),
            "a later wide match must remain directly retrievable: {out}"
        );
        assert!(out.ends_with("TAIL_ONLY_MATCH"));
        assert!(out.chars().count() <= MAX_OUTPUT_CHARS);
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
        let out = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
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
        let out = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
        assert!(out.contains("1:Fatal ERROR here"));
    }
}
