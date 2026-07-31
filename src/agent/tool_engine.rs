//! Tool execution engine: delegated and inline paths.
//!
//! Extracted from `agent_loop.rs` to isolate tool execution logic.

use std::sync::atomic::Ordering;
use std::time::Duration;

use serde_json::{json, Value};
use tracing::{debug, info, instrument, warn, Instrument};

use crate::agent::agent_core::RuntimeCounters;
use crate::agent::audit::ToolEvent;
use crate::agent::context::ContextBuilder;
use crate::agent::markers::{
    TOOL_ANALYSIS_SUMMARY_PREFIX, TOOL_RUNNER_OUTPUT_PREFIX, TOOL_RUNNER_SUMMARY_PREFIX,
};
use crate::agent::role_policy;
use crate::agent::tool_runner::{self, Budget, ToolRunnerConfig};
use crate::providers::base::{LLMResponse, ToolCallRequest};
use std::sync::Arc;

use super::agent_loop::{ResponseBoundary, TurnContext};
use crate::agent::context_hygiene::{tool_result_ok, TOOL_RESULT_REPLAY_MAX_BYTES};
use crate::agent::tools::base::ToolConcurrency;

const LARGE_TOOL_RESULT_TOKEN_THRESHOLD: usize = 500;
/// Bound native multi-tool fan-out. Four concurrent reads/fetches keep local
/// resource use predictable while still collapsing the dominant serial waits.
const MAX_PARALLEL_TOOL_CALLS: usize = 4;
/// Minimum room for a compact receipt with a recall handle. Below this, the
/// prompt may save a few bytes while losing the exact retrieval path.
const MIN_BATCH_TOOL_RESULT_CAP_CHARS: usize = 320;

/// Per-tool token threshold above which a raw tool result is replaced by a
/// summary. Enumerative tools (`exec`, `list_dir`, `web_search`, `read_file`)
/// return specific strings — filenames, URLs, error lines — that the model
/// needs to quote verbatim. Summaries destroy them, which the model then
/// papers over by fabricating. Keep raw output for these up to ~4000 tokens.
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
    // No recall exemption: a recalled body can be hundreds of KB and must be
    // stashable under the recall's own id so slice_tool_result /
    // search_tool_result can query it. Exempting it left the raw body in live
    // context, which inflated a session to 77k tokens (2026-07-30).
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
/// `recall_tool_result({"tool_call_id": id})`.
pub(crate) const TOOL_RESULT_HANDLE_MARKER: &str = "TOOL_RESULT_HANDLE v1 |";

/// The explicit-retrieval tools whose results are ALREADY bounded (≤4KB) by
/// their own cap. Per plan §2 they carry their bounded excerpt directly —
/// wrapping them in a handle would be pointless double-indirection.
fn is_explicit_retrieval_tool(name: &str) -> bool {
    matches!(
        name,
        "recall_tool_result" | "slice_tool_result" | "search_tool_result"
    )
}

/// Render the canonical, write-once-stable handle for a tool result. Pure and
/// deterministic: identical inputs always produce byte-identical output, so a
/// handle rendered live at ingestion is identical to the one persisted to
/// SQLite and reloaded later — no prefix-cache drift (the root cause of the
/// `token_mismatch` desync class).
///
/// Built ONCE at ingestion from the exact stored bytes. The `sha256` is over
/// those bytes; `chars` is the Unicode char count; `args` is a fixed-scalar
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
        r#"{MARKER} id:{id_j} | tool:{tool_j} | ok:{ok} | chars:{chars} | sha256:{digest} | args:{args_j} | excerpt:{excerpt_j} | fetch:"recall_tool_result""#,
        MARKER = TOOL_RESULT_HANDLE_MARKER,
        id_j = serde_json::to_string(id).unwrap_or_else(|_| "\"\"".into()),
        tool_j = serde_json::to_string(tool).unwrap_or_else(|_| "\"\"".into()),
        args_j = serde_json::to_string(&tool_arg_summary(args)).unwrap_or_else(|_| "{}".into()),
        excerpt_j = serde_json::to_string(&excerpt).unwrap_or_else(|_| "\"\"".into()),
    )
}

/// Fixed-scalar allowlist for the handle's `args` field, in a fixed order.
/// Only these keys are surfaced (deterministic + compact); everything else is
/// dropped. JSON-escaped by the caller via `serde_json::to_string`.
fn tool_arg_summary(args: &std::collections::HashMap<String, Value>) -> Vec<(&'static str, String)> {
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
    ContextBuilder::add_tool_result_with_status(&mut ctx.messages, tool_id, tool_name, &msg, false);
}

/// Build a head+tail preview of `data` (≤ `cap` chars) with a `recall_tool_result`
/// pointer to `tool_call_id`. Assumes the full body is ALREADY stashed (by the
/// caller) when `data` was truncated — this only shapes the in-context preview.
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
    // A recalled body that is still too large for live context must NOT point
    // back at recall_tool_result (circular: the model just recalled it and
    // would loop). The full body is stashed under this id; direct the model at
    // slice_tool_result / search_tool_result to query it without reloading.
    let header = if tool_name == "recall_tool_result" {
        format!(
            "[recalled output still too large (~{estimated_tokens} tokens); \
             use slice_tool_result/search_tool_result with tool_call_id=\"{tool_call_id}\"]\n"
        )
    } else {
        format!(
            "[truncated: {tool_name}, ~{estimated_tokens} tokens; \
             recall_tool_result({{\"tool_call_id\": \"{tool_call_id}\"}}) for full]\n"
        )
    };
    let footer = "\n[...]\n";
    let fixed_chars = header.chars().count() + footer.chars().count();
    if fixed_chars >= cap {
        return header;
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
/// `recall_tool_result` to recover the middle. This bounds each result's
/// in-context cost to ~`cap` chars (so N tool calls cost ~N×cap, not N×full)
/// while keeping any result one tool call away — no re-run needed.
fn digest_tool_result(
    tool_name: &str,
    args: &std::collections::HashMap<String, Value>,
    data: &str,
    cap: usize,
    tool_call_id: &str,
) -> String {
    // No recall exemption: recalled bodies are shaped by the same cap as any
    // other result. build_tool_result_preview points recalled output at
    // slice/search (not back at recall) so the model cannot loop.
    let prompt_cap = cap.min(TOOL_RESULT_REPLAY_MAX_BYTES);
    let total_chars = data.chars().count();
    if total_chars <= prompt_cap && data.len() <= TOOL_RESULT_REPLAY_MAX_BYTES {
        return data.to_string();
    }
    build_tool_result_preview(tool_name, args, data, prompt_cap, tool_call_id)
}

/// Head+tail preview builder retained as the reference for the truncation
/// tests; production ingestion uses [`digest_tool_result`] (which adds
/// lossless retrieval).
#[allow(dead_code)]
fn compact_inline_tool_result(
    tool_name: &str,
    _args: &std::collections::HashMap<String, Value>,
    data: &str,
    max_chars: usize,
) -> String {
    let total_chars = data.chars().count();
    if total_chars <= max_chars {
        return data.to_string();
    }

    let estimated_tokens = crate::agent::token_budget::TokenBudget::estimate_str_tokens(data);
    let header = format!(
        "[truncated: {tool_name}, ~{estimated_tokens} tokens; \
         head+tail shown, re-request with a narrower range/query for the middle]\n"
    );

    let footer = "\n[...]\n";
    let fixed_chars = header.chars().count() + footer.chars().count();
    let preview_budget = max_chars.saturating_sub(fixed_chars).max(200);
    let head_chars = preview_budget * 2 / 3;
    let tail_chars = preview_budget.saturating_sub(head_chars);

    let head: String = data.chars().take(head_chars).collect();
    let tail_rev: Vec<char> = data.chars().rev().take(tail_chars).collect();
    let tail: String = tail_rev.into_iter().rev().collect();

    let mut out = format!("{header}{head}{footer}{tail}");
    if out.chars().count() > max_chars {
        out = out.chars().take(max_chars).collect();
    }
    out
}

pub(crate) fn local_model_key(model: &str) -> String {
    model
        .strip_prefix("local:")
        .unwrap_or(model)
        .trim()
        .to_ascii_lowercase()
}

/// True when tool delegation would reuse the SAME local model as the main agent.
///
/// Such delegation is pure prefix-cache poison: the delegation sub-loop runs
/// many distinct prompts on the same local server+model, evicting the main
/// conversation's KV/radix prefix and forcing a full re-prefill (~60-90s at
/// large context, measured) every tool round — for zero token-cost benefit
/// (same model). The caller runs tools inline instead, keeping one warm prefix.
/// A genuinely separate delegation model (different name) returns false, and a
/// cloud main model (`is_local == false`) returns false.
pub(crate) fn delegation_reuses_main_local_model(
    is_local: bool,
    main_model: &str,
    delegation_model: Option<&str>,
) -> bool {
    is_local && delegation_model.map(local_model_key) == Some(local_model_key(main_model))
}

/// Side-effect tools that arm (and are rejected by) the response boundary.
pub(crate) fn is_side_effect_tool(name: &str) -> bool {
    matches!(name, "exec" | "write_file" | "edit_file" | "apply_patch")
}

/// Tools whose result must be consumed and reported before another call of
/// the same class. Memory reads/writes are included because silently chaining
/// them was observed to leave a turn with no final answer and to amplify bad
/// recall/remember output.
pub(crate) fn requires_result_report(name: &str) -> bool {
    is_side_effect_tool(name) || matches!(name, "recall" | "remember")
}

/// Decide whether to arm the response boundary after a tool-execution round.
///
/// Behavioral, not positional: arm ONLY when a side-effect/write tool
/// actually ran AND the assistant produced no text report. A model that narrates
/// each step is not fabricating results, so it must not be throttled. Arming
/// blindly after every side-effect call (the prior behavior) rejected ~1/3 of
/// legitimate consecutive write/exec calls. `executed_tools` must contain
/// only the tools that actually executed — never boundary-rejected calls — so a
/// rejected call cannot re-arm the boundary.
fn should_arm_boundary(assistant_content: Option<&str>, executed_tools: &[&str]) -> bool {
    let reported = assistant_content.is_some_and(|c| !c.trim().is_empty());
    let ran_reportable_tool = executed_tools.iter().any(|n| requires_result_report(n));
    ran_reportable_tool && !reported
}

/// Decide whether an armed boundary blocks this response's side-effect calls.
///
/// Behavioral, mirroring [`should_arm_boundary`]: the armed call was nudged to
/// report as text. A response that carries a text report HAS complied — its
/// side-effect calls run normally. Only a silent armed response is rejected.
/// (The prior check was positional — Armed rejected every side-effect call,
/// so a model that narrated AND acted in one response ate a guaranteed
/// spurious rejection: the "always a bad exec before a good one" pattern.)
fn boundary_blocks_side_effects(armed: bool, assistant_content: Option<&str>) -> bool {
    let reported = assistant_content.is_some_and(|c| !c.trim().is_empty());
    armed && !reported
}

/// Execute tool calls via the delegation (tool-runner) path.
///
/// Returns `true` if delegation was used (caller should `continue` the main loop).
/// Returns `false` if delegation couldn't proceed (caller should fall through to inline).
#[instrument(
    name = "execute_tools_delegated",
    skip(ctx, counters, routed_tool_calls, response, delegation_provider, delegation_model),
    fields(
        tools = tracing::field::Empty,
        outcome = tracing::field::Empty,
    )
)]
pub(crate) async fn execute_tools_delegated(
    ctx: &mut TurnContext,
    counters: &RuntimeCounters,
    routed_tool_calls: &[ToolCallRequest],
    response: &LLMResponse,
    delegation_provider: &Option<Arc<dyn crate::providers::base::LLMProvider>>,
    delegation_model: &Option<String>,
) -> bool {
    let (tr_provider, tr_model) = match (delegation_provider.as_ref(), delegation_model.as_ref()) {
        (Some(p), Some(m)) => (p.clone(), m.clone()),
        _ => {
            tracing::Span::current().record("outcome", "skipped_no_provider");
            return false;
        }
    };

    let tool_names_summary: String = routed_tool_calls
        .iter()
        .map(|tc| tc.name.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    tracing::Span::current().record("tools", &tool_names_summary.as_str());

    debug!(
        "Delegating {} tool calls to tool runner (model: {})",
        routed_tool_calls.len(),
        tr_model
    );

    // Detect [VERBATIM] marker: the main model is asking for
    // raw tool output instead of a delegation summary.
    let verbatim = response
        .content
        .as_ref()
        .map(|c| c.contains("[VERBATIM]"))
        .unwrap_or(false);
    let same_local_model = ctx.core.mode().is_local()
        && local_model_key(&tr_model) == local_model_key(&ctx.core.model);
    let verbatim = verbatim || same_local_model;
    if same_local_model {
        debug!(
            "Delegation model is the main local model ({}); skipping scratch-pad LLM analysis",
            tr_model
        );
    }

    // Delegation models (Qwen, Nemotron, Claude) typically have 8K+ context.
    // Cap tool results to ~2000 tokens (~8000 chars) to allow meaningful content
    // while leaving room for system prompt, tool calls, and response.
    // Use the main model's limit only if it's already smaller.
    let delegation_result_limit = ctx.core.max_tool_result_chars.min(8000);

    let runner_config = ToolRunnerConfig {
        provider: tr_provider.clone(),
        model: tr_model.clone(),
        max_iterations: ctx.core.tool_delegation_config.max_iterations,
        max_tokens: ctx.core.tool_delegation_config.max_tokens,

        max_tool_result_chars: delegation_result_limit,
        short_circuit_chars: 200,
        depth: 0,
        cancellation_token: ctx.cancellation_token.clone(),
        verbatim,
        budget: {
            let cost_budget = ctx.core.tool_delegation_config.cost_budget;
            if cost_budget > 0.0 {
                let prices = crate::agent::model_prices::ModelPrices::load().await;
                Some(Budget::root_with_cost(
                    ctx.core.tool_delegation_config.max_iterations,
                    2,
                    cost_budget,
                    std::sync::Arc::new(prices),
                ))
            } else {
                Some(Budget::root(
                    ctx.core.tool_delegation_config.max_iterations,
                    2,
                ))
            }
        },
    };

    // Emit tool call start events for delegated calls.
    if let Some(ref tx) = ctx.tool_event_tx {
        for tc in routed_tool_calls {
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
    }

    // Build task description for the delegation model.
    let tool_names: Vec<&str> = routed_tool_calls
        .iter()
        .map(|tc| tc.name.as_str())
        .collect();
    let instructions = response
        .content
        .as_deref()
        .filter(|c| !c.trim().is_empty())
        .map(|c| c.chars().take(400).collect::<String>())
        .unwrap_or_else(|| ctx.user_content.chars().take(300).collect::<String>());
    let task_desc = if ctx.core.tool_delegation_config.role_scoped_context_packs {
        let task_state = format!(
            "Tool lane execution\nPlanned tools: {}",
            tool_names.join(", ")
        );
        role_policy::build_context_pack(
            role_policy::Role::Main,
            &instructions,
            "(live turn; summary omitted)",
            &task_state,
            &ctx.tools.tool_names(),
            2500,
        )
    } else {
        format!(
            "Instructions: {}\nTools to execute: {}",
            instructions,
            tool_names.join(", ")
        )
    };

    // Taint pre-check: warn if any delegated call is sensitive while context is tainted.
    for tc in routed_tool_calls {
        if let Some(_spans) = ctx.taint_state.check_sensitive(&tc.name) {
            warn!(
                "TAINT WARNING: Executing sensitive tool '{}' (delegated) with tainted context from: {}",
                tc.name,
                ctx.taint_state.taint_summary()
            );
        }
    }

    let delegation_start = std::time::Instant::now();

    // Journal the provider's tool-call carrier before the runner can perform
    // any side effect. This is the durable intent record for delegated tools.
    let tc_json: Vec<Value> = routed_tool_calls
        .iter()
        .map(|tc| tc.to_openai_json())
        .collect();
    ContextBuilder::add_assistant_message(
        &mut ctx.messages,
        response.content.as_deref(),
        Some(&tc_json),
    );
    ctx.persist_pending_protocol_messages().await;

    let run_result =
        tool_runner::run_tool_loop(&runner_config, routed_tool_calls, &ctx.tools, &task_desc).await;
    let delegation_elapsed_ms = delegation_start.elapsed().as_millis() as u64;

    // Only mark unhealthy on actual provider/tool-runner errors.
    let is_hard_failure = run_result.error.is_some();
    if is_hard_failure && !run_result.tool_results.is_empty() {
        let reason = format!(
            "delegation model errored: {}",
            run_result.error.as_deref().unwrap_or("unknown error")
        );
        let results_preview: String = run_result
            .tool_results
            .first()
            .map(|(_, name, data)| {
                format!("[{}]: {}", name, data.chars().take(200).collect::<String>())
            })
            .unwrap_or_default();
        warn!(
            "Delegation failed — {}. model={}, iterations={}, results={}, preview={}. \
             Marking unhealthy. Restart servers or toggle /local to recover.",
            reason,
            tr_model,
            run_result.iterations_used,
            run_result.tool_results.len(),
            results_preview,
        );
        counters.delegation_healthy.store(false, Ordering::Relaxed);
    } else if delegation_elapsed_ms > 30_000 {
        debug!(
            "Delegation run was slow ({} ms) but succeeded — keeping provider healthy",
            delegation_elapsed_ms,
        );
    } else if run_result.summary.is_none() && !run_result.tool_results.is_empty() {
        debug!(
            "Delegation returned no summary (model={}, iters={}), using results inline",
            tr_model, run_result.iterations_used,
        );
    } else if !counters.delegation_healthy.load(Ordering::Relaxed) {
        // Re-probe succeeded — server recovered!
        info!("Delegation provider recovered — re-enabling delegation");
        counters.delegation_healthy.store(true, Ordering::Relaxed);
        counters
            .delegation_retry_counter
            .store(0, Ordering::Relaxed);
    }

    debug!(
        "Tool runner completed: {} results in {} iterations",
        run_result.tool_results.len(),
        run_result.iterations_used
    );

    // Add tool results from the runner to the main context.
    let preview_max = ctx.core.tool_delegation_config.max_result_preview_chars;
    let routed_result_cap =
        inline_hot_prompt_result_cap_for_ctx_batch(ctx, routed_tool_calls.len());
    let force_routed_stash = routed_tool_calls.len() > 1;

    for tc in routed_tool_calls {
        let full_data = run_result
            .tool_results
            .iter()
            .find(|(id, _, _)| id == &tc.id)
            .map(|(_, _, data)| data.as_str())
            .unwrap_or("(no result)");

        let full_tokens = crate::agent::token_budget::TokenBudget::estimate_str_tokens(full_data);

        let threshold = summary_threshold_tokens(&tc.name);
        let cap = routed_result_cap;
        let injected_raw = if let Some(ref summary) = run_result.summary {
            // Summary exists from scratch-pad analysis.
            if full_tokens > threshold {
                // Large data + good summary available: use the summary so compaction
                // can never destroy the content by proportional truncation.
                format!(
                    "{}\n{}\n\n[Full output: {} chars, cached in context store]",
                    TOOL_ANALYSIS_SUMMARY_PREFIX,
                    summary,
                    full_data.len()
                )
            } else {
                // Small data — raw injection is safe; compaction won't truncate it.
                ctx.content_gate.admit_simple(full_data).into_text()
            }
        } else if ctx.core.specialist_provider.is_some() && full_tokens > threshold {
            ctx.content_gate
                .admit_with_specialist(
                    full_data,
                    ctx.core.specialist_provider.as_ref().unwrap().as_ref(),
                    ctx.core.specialist_model.as_deref().unwrap_or(""),
                )
                .await
                .into_text()
        } else {
            ctx.content_gate.admit_simple(full_data).into_text()
        };
        // Stash the RAW delegated output (pre-runner-summary / pre-gate) for
        // lossless recall — digesting `injected_raw` directly would store the
        // summary instead of the original. Then preview injected_raw and add a
        // recall pointer when the raw was stashed.
        let stashed_raw = match stash_tool_result_for_prompt_shaping(
            &ctx.core.sessions,
            &ctx.session_id,
            &tc.id,
            &tc.name,
            full_data,
            cap,
            force_routed_stash,
        )
        .await
        {
            Ok(b) => b,
            Err(sr) => {
                abort_turn_on_stash_failure(ctx, &tc.id, &tc.name, &sr);
                // Skip this iteration's post-stash shaping (the abort helper
                // already pushed an ok:false receipt). The loop-level
                // infra_error check in step_execute_tools finalizes the turn.
                continue;
            }
        };
        let mut injected = if stashed_raw
            && !is_explicit_retrieval_tool(&tc.name)
            && full_data.len() > TOOL_RESULT_REPLAY_MAX_BYTES
        {
            // Genuinely oversized (>8KB) non-retrieval delegated output →
            // handle only. Body lives in the stash; model recalls it. (A 95KB
            // result replayed raw was the cache-break class.)
            render_tool_result_handle(
                &tc.id,
                &tc.name,
                tool_result_ok(full_data),
                full_data.as_bytes(),
                &tc.arguments,
            )
        } else {
            // Medium (over hot-prompt cap but ≤8KB), retrieval tools (bounded
            // excerpt by design), or small — show actual CONTENT via the
            // deterministic head+tail preview. Handling medium results starved
            // the model of normal read/exec bytes (2026-07-31 regression).
            build_tool_result_preview(&tc.name, &tc.arguments, &injected_raw, cap, &tc.id)
        };
        // Prepend the per-lease progress signal so the model can see
        // remaining budget inline (B3 of the lease design — visible,
        // deterministic, helps the model self-regulate instead of being
        // interrupted). Recorded calls already incremented the counter,
        // so this describes the call that produced this result.
        let lease_signal = ctx.flow.lease.progress_signal();
        injected = format!("{lease_signal}\n{injected}");

        let ok = tool_result_ok(full_data);
        if ctx.core.provenance_config.enabled {
            ContextBuilder::add_tool_result_immutable_with_status(
                &mut ctx.messages,
                &tc.id,
                &tc.name,
                &injected,
                ok,
            );
        } else {
            ContextBuilder::add_tool_result_with_status(
                &mut ctx.messages,
                &tc.id,
                &tc.name,
                &injected,
                ok,
            );
        }
        ctx.flow.tool_guard.record_result_with_status(
            &tc.name,
            &tc.arguments,
            injected.clone(),
            ok,
        );
        ctx.used_tools.insert(tc.name.clone());
        ctx.persist_pending_protocol_messages().await;
    }

    // Inject the runner's summary so the main LLM knows what
    // the tools found without needing full output.
    let has_extra = run_result.tool_results.len() > routed_tool_calls.len();
    if run_result.summary.is_some() || has_extra {
        let summary_text = if has_extra {
            let extra = tool_runner::format_results_for_context(
                &run_result,
                preview_max,
                Some(&mut ctx.content_gate), // Wire ContentGate for budget-aware truncation
            );
            format!(
                "[Tool runner executed {} additional calls]\n{}",
                run_result.tool_results.len() - routed_tool_calls.len(),
                extra
            )
        } else {
            run_result.summary.clone().unwrap_or_default()
        };
        if !summary_text.is_empty() {
            let prefix = if verbatim {
                TOOL_RUNNER_OUTPUT_PREFIX
            } else {
                TOOL_RUNNER_SUMMARY_PREFIX
            };
            ctx.messages.push(crate::agent::markers::scaffold_user(format!(
                "{} {}",
                prefix, summary_text
            )));
            ctx.persist_pending_protocol_messages().await;
        }
    }

    // Record learning + audit for all tool results.
    let executor = format!("tool_runner:{}", tr_model);
    let n_results = run_result.tool_results.len().max(1) as u64;
    for (tool_call_id, tool_name, data) in &run_result.tool_results {
        let ok = !data.starts_with("Error:");
        let per_tool_ms = delegation_elapsed_ms / n_results;

        // Only render CallEnd in the TUI for results that the caller asked
        // for. Internal scratchpad calls the runner made on its own already
        // roll up into the runner-summary user message — emitting a CallEnd
        // for them produces a duplicate identical-duration block per tool.
        if let Some(ref tx) = ctx.tool_event_tx {
            if is_routed_call(tool_call_id, routed_tool_calls) {
                let _ = tx.send(ToolEvent::CallEnd {
                    tool_name: tool_name.clone(),
                    tool_call_id: tool_call_id.clone(),
                    result_data: data.clone(),
                    ok,
                    duration_ms: per_tool_ms,
                });
            }
        }

        if let Some(ref audit) = ctx.audit {
            let _ = audit.record(
                tool_name,
                tool_call_id,
                &json!({}),
                data,
                ok,
                per_tool_ms,
                &executor,
            );
        }

        ctx.used_tools.insert(tool_name.clone());
        ctx.turn_tool_entries
            .push(crate::agent::audit::TurnToolEntry {
                name: tool_name.clone(),
                id: tool_call_id.clone(),
                ok,
                duration_ms: per_tool_ms,
                result_chars: data.len(),
            });

        // Taint tracking: mark context tainted when a web tool ran via delegation.
        // We don't have the original arguments here, so pass None for detail.
        ctx.taint_state.mark_tainted(tool_name, None);
    }

    // Behavioral response-boundary arming (mirrors execute_tools_inline).
    // `response` is the main-model response that requested delegation.
    let executed: Vec<&str> = run_result
        .tool_results
        .iter()
        .map(|(_, tool_name, _)| tool_name.as_str())
        .collect();
    if should_arm_boundary(response.content.as_deref(), &executed) {
        ctx.flow.boundary = ResponseBoundary::Pending;
    }

    tracing::Span::current().record("outcome", "ok");
    true
}

/// Returns `true` when a delegated tool result corresponds to one of the
/// caller's routed tool calls (rather than an internal scratchpad call the
/// tool runner made on its own). The TUI should only render `CallEnd` for
/// these — internal extras already roll up into the runner summary message.
fn is_routed_call(tool_call_id: &str, routed_tool_calls: &[ToolCallRequest]) -> bool {
    routed_tool_calls.iter().any(|tc| tc.id == tool_call_id)
}

/// Collects everything produced by a single tool execution, ready for
/// sequential post-processing by `inject_tool_result`.
struct SingleToolResult {
    tool_name: String,
    tool_id: String,
    arguments: std::collections::HashMap<String, serde_json::Value>,
    result: crate::agent::tools::base::ToolExecutionResult,
    duration_ms: u64,
}

fn completed_reportable_tool(result: &SingleToolResult) -> Option<&str> {
    if !result.result.ok || !requires_result_report(&result.tool_name) {
        return None;
    }
    if result.tool_name == "write_file"
        && result
            .arguments
            .get("state")
            .and_then(Value::as_str)
            .is_some_and(|state| state.trim().eq_ignore_ascii_case("more"))
    {
        return None;
    }
    Some(&result.tool_name)
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
            return SingleToolResult {
                tool_name: tc.name.clone(),
                tool_id: tc.id.clone(),
                arguments: tc.arguments.clone(),
                result: crate::agent::tools::base::ToolExecutionResult::failure(
                    "tool call cancelled".to_string(),
                ),
                duration_ms: 0,
            };
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
            use crate::agent::tools::base::ToolExecutionContext;

            // Tool-call identity is part of the execution contract, not a UI
            // concern. A closed channel keeps headless calls event-free while
            // still carrying the provider ID needed for idempotent mutations.
            let event_tx = tool_event_tx.as_ref().cloned().unwrap_or_else(|| {
                let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
                drop(rx);
                tx
            });
            let exec_ctx = ToolExecutionContext {
                event_tx,
                cancellation_token: cancellation_token
                    .as_ref()
                    .map(|t| t.child_token())
                    .unwrap_or_else(tokio_util::sync::CancellationToken::new),
                tool_call_id: tc.id.clone(),
            };
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

        let duration_ms = start.elapsed().as_millis() as u64;
        tracing::Span::current().record("ok", result.ok);
        debug!(
            "Tool {} result ({}B, ok={}, {}ms)",
            tc.name,
            result.data.len(),
            result.ok,
            duration_ms
        );

        SingleToolResult {
            tool_name: tc.name.clone(),
            tool_id: tc.id.clone(),
            arguments: tc.arguments.clone(),
            result,
            duration_ms,
        }
    }
    .instrument(tool_span)
    .await
}

/// Execute calls in provider order. Adjacent `ParallelSafe` runs overlap with
/// bounded fan-out; every sequential tool is an ordering barrier. Each bounded
/// `join_all` preserves carrier order even when calls complete out of order.
async fn execute_tool_calls_ordered(
    calls: &[&ToolCallRequest],
    tools: &crate::agent::tools::registry::ToolRegistry,
    tool_event_tx: &Option<tokio::sync::mpsc::UnboundedSender<ToolEvent>>,
    cancellation_token: &Option<tokio_util::sync::CancellationToken>,
    tool_heartbeat_secs: u64,
    taints: Vec<Option<String>>,
) -> Vec<SingleToolResult> {
    let mut results = Vec::with_capacity(calls.len());
    let mut start = 0;

    while start < calls.len() {
        if tools.concurrency(&calls[start].name) == ToolConcurrency::ParallelSafe {
            let mut end = start + 1;
            while end < calls.len()
                && tools.concurrency(&calls[end].name) == ToolConcurrency::ParallelSafe
            {
                end += 1;
            }
            for chunk_start in (start..end).step_by(MAX_PARALLEL_TOOL_CALLS) {
                let chunk_end = (chunk_start + MAX_PARALLEL_TOOL_CALLS).min(end);
                let futures = calls[chunk_start..chunk_end]
                    .iter()
                    .zip(taints[chunk_start..chunk_end].iter().cloned())
                    .map(|(tc, taint)| {
                        execute_single_tool(
                            tc,
                            tools,
                            tool_event_tx,
                            cancellation_token,
                            tool_heartbeat_secs,
                            taint,
                        )
                    });
                results.extend(futures_util::future::join_all(futures).await);
            }
            start = end;
        } else {
            results.push(
                execute_single_tool(
                    calls[start],
                    tools,
                    tool_event_tx,
                    cancellation_token,
                    tool_heartbeat_secs,
                    taints[start].clone(),
                )
                .await,
            );
            start += 1;
        }
    }

    results
}

/// Post-process one completed tool result: gate content, inject into messages,
/// emit CallEnd, audit, update taint/learning/force_response.
///
/// This function must run sequentially (one result at a time) because it
/// mutates `ctx`.
async fn inject_tool_result(
    ctx: &mut TurnContext,
    r: &SingleToolResult,
    prompt_cap: usize,
    force_stash_raw: bool,
) {
    // For web_fetch/web_search: unwrap the JSON envelope so the model
    // sees clean article text rather than a JSON metadata summary.
    let result_data = if r.tool_name == "web_fetch" || r.tool_name == "web_search" {
        crate::agent::tools::web::extract_web_content(&r.result.data)
    } else {
        r.result.data.clone()
    };

    // Gate tool result through context budget.
    let threshold = summary_threshold_tokens(&r.tool_name);
    let cap = prompt_cap.max(1);
    let data = if r.tool_name == "recall_tool_result" {
        // A recalled body is verbatim what the model asked for, but a large
        // one cannot enter live context raw: a 172KB recall inflated a
        // session to 77k tokens and triggered a cache-dropping compaction
        // (session 20260730_094531_508b68, 2026-07-30). Route it through the
        // same stash+digest as any other oversized result — the full body is
        // stashed under this recall's id and the model queries it via
        // slice_tool_result / search_tool_result.
        let _stashed_raw = match stash_tool_result_for_prompt_shaping(
            &ctx.core.sessions,
            &ctx.session_id,
            &r.tool_id,
            &r.tool_name,
            &result_data,
            cap,
            force_stash_raw,
        )
        .await
        {
            Ok(b) => b,
            Err(sr) => {
                abort_turn_on_stash_failure(ctx, &r.tool_id, &r.tool_name, &sr);
                return;
            }
        };
        // Recall is an explicit-retrieval tool: its result carries a bounded
        // excerpt (digest_tool_result handles small-as-passthrough + large as
        // head+tail preview). The full body is stashed for slice/search.
        let prompt_data =
            digest_tool_result(&r.tool_name, &r.arguments, &result_data, cap, &r.tool_id);
        ctx.content_gate.admit_simple(&prompt_data).into_text()
    } else if ctx.core.specialist_provider.is_some()
        && crate::agent::token_budget::TokenBudget::estimate_str_tokens(&result_data) > threshold
    {
        // Specialist path: stash the RAW (pre-specialist) output for lossless
        // recall — without this, recall would return the specialist's summary
        // instead of the original. Then summarize and preview the summary.
        let stashed_raw = match stash_tool_result_for_prompt_shaping(
            &ctx.core.sessions,
            &ctx.session_id,
            &r.tool_id,
            &r.tool_name,
            &result_data,
            cap,
            force_stash_raw,
        )
        .await
        {
            Ok(b) => b,
            Err(sr) => {
                abort_turn_on_stash_failure(ctx, &r.tool_id, &r.tool_name, &sr);
                return;
            }
        };
        let summarized = ctx
            .content_gate
            .admit_with_specialist(
                &result_data,
                ctx.core.specialist_provider.as_ref().unwrap().as_ref(),
                ctx.core.specialist_model.as_deref().unwrap_or(""),
            )
            .await
            .into_text();
        let mut preview =
            build_tool_result_preview(&r.tool_name, &r.arguments, &summarized, cap, &r.tool_id);
        // If the summary fit under cap, build_tool_result_preview returned it
        // unchanged with no recall pointer — but the raw IS stashed, so tell
        // the model it can still recover the original.
        if stashed_raw && !preview.contains("recall_tool_result") {
            preview.push_str(&format!(
                "\n[full original output retrievable via recall_tool_result({{\"tool_call_id\": \"{}\"}})]",
                r.tool_id
            ));
        }
        preview
    } else {
        let stashed_raw = match stash_tool_result_for_prompt_shaping(
            &ctx.core.sessions,
            &ctx.session_id,
            &r.tool_id,
            &r.tool_name,
            &result_data,
            cap,
            force_stash_raw,
        )
        .await
        {
            Ok(b) => b,
            Err(sr) => {
                abort_turn_on_stash_failure(ctx, &r.tool_id, &r.tool_name, &sr);
                return;
            }
        };
        if stashed_raw && result_data.len() > TOOL_RESULT_REPLAY_MAX_BYTES {
            // GENUINELY oversized (>8KB): handle only. A single 95KB result
            // replayed raw was the cache-break class (token_mismatch under
            // ExactBootstrap). The body lives in the stash, fetchable via
            // recall_tool_result.
            let handle = render_tool_result_handle(
                &r.tool_id,
                &r.tool_name,
                r.result.ok,
                result_data.as_bytes(),
                &r.arguments,
            );
            ctx.content_gate.admit_simple(&handle).into_text()
        } else if stashed_raw {
            // Medium (over the hot-prompt char cap but ≤8KB replay cap): show
            // actual CONTENT via a deterministic head+tail preview. Handling
            // these deprived the model of normal read/exec bytes and forced
            // hallucination (2026-07-31 regression after the handles uproot).
            // Stable per-result (body doesn't change) → no cache drift.
            build_tool_result_preview(&r.tool_name, &r.arguments, &result_data, cap, &r.tool_id)
        } else {
            // Small result, not stashed — small enough to carry inline raw
            // (under both the char cap and the replay byte cap). A handle
            // would be larger than the body itself.
            ctx.content_gate.admit_simple(&result_data).into_text()
        }
    };

    if ctx.core.provenance_config.enabled {
        ContextBuilder::add_tool_result_immutable_with_status(
            &mut ctx.messages,
            &r.tool_id,
            &r.tool_name,
            &data,
            r.result.ok,
        );
    } else {
        ContextBuilder::add_tool_result_with_status(
            &mut ctx.messages,
            &r.tool_id,
            &r.tool_name,
            &data,
            r.result.ok,
        );
    }
    ctx.persist_pending_protocol_messages().await;
    ctx.flow.tool_guard.record_result_with_status(
        &r.tool_name,
        &r.arguments,
        data.clone(),
        r.result.ok,
    );

    // Emit CallEnd.
    if let Some(ref tx) = ctx.tool_event_tx {
        let _ = tx.send(ToolEvent::CallEnd {
            tool_name: r.tool_name.clone(),
            tool_call_id: r.tool_id.clone(),
            result_data: r.result.data.clone(),
            ok: r.result.ok,
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
            &r.result.data,
            r.result.ok,
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
            ok: r.result.ok,
            duration_ms: r.duration_ms,
            result_chars: r.result.data.len(),
        });

    // NOTE: response-boundary arming is NOT done here. This function sees only
    // one tool result and cannot tell whether the model reported its work in the
    // assistant turn. Arming is decided once, behaviorally, by the caller
    // (execute_tools_inline / execute_tools_delegated) which holds the assistant
    // `response`. Arming here (per-tool, unconditionally) was positional, not
    // behavioral — it throttled legitimate narrated side-effect chains.
}

/// First `max_chars` of the most recent real tool result in `messages`.
///
/// Skips prior boundary rejections (they are tool-role messages too — quoting
/// one back would nudge the model toward the rejection text instead of the
/// actual result). Returns `None` when no tool result exists yet.
fn last_tool_result_snippet(messages: &[Value], max_chars: usize) -> Option<String> {
    let content = messages.iter().rev().find_map(|m| {
        if m.get("role").and_then(|r| r.as_str()) != Some("tool") {
            return None;
        }
        let c = m.get("content").and_then(|c| c.as_str())?;
        if c.starts_with("response boundary:") {
            return None;
        }
        Some(c)
    })?;
    let cleaned = content
        .trim_start_matches("[VERBATIM TOOL OUTPUT — do not paraphrase]")
        .trim();
    if cleaned.is_empty() {
        return None;
    }
    let end = crate::utils::helpers::floor_char_boundary(cleaned, max_chars);
    Some(cleaned[..end].replace('\n', " "))
}

/// Inject an error result for a side-effect tool call rejected by the
/// response boundary.
///
/// Deliberately narrower than `inject_tool_result`: no learning record, no
/// taint, no audit, and — critically — it does NOT re-arm the boundary (a
/// rejected call must not extend its own boundary, or the loop would
/// livelock: nudge → reject → nudge → …).
fn inject_boundary_rejection(ctx: &mut TurnContext, tc: &ToolCallRequest) {
    // Quote the prior result inline: small local models don't reliably look
    // back through context to find what they should report on, so give them
    // the material to comply with right here at the tail.
    let prior = last_tool_result_snippet(&ctx.messages, 160);
    let msg = match prior {
        Some(snippet) => format!(
            "response boundary: {} was not executed — first respond with what the \
             previous tool results showed (last result began: \"{}\"); it can run \
             in a later step.",
            tc.name, snippet
        ),
        None => format!(
            "response boundary: {} was not executed — first respond with what the \
             previous tool results showed; it can run in a later step.",
            tc.name
        ),
    };
    if ctx.core.provenance_config.enabled {
        ContextBuilder::add_tool_result_immutable_with_status(
            &mut ctx.messages,
            &tc.id,
            &tc.name,
            &msg,
            false,
        );
    } else {
        ContextBuilder::add_tool_result_with_status(
            &mut ctx.messages,
            &tc.id,
            &tc.name,
            &msg,
            false,
        );
    }
    // The model attempted a tool but was prevented — suppress
    // ClaimedButNotExecuted validation for this turn.
    ctx.flow.tool_guard.had_blocked_calls = true;
    if let Some(ref tx) = ctx.tool_event_tx {
        // Emit CallStart too (same as execute_single_tool): renderers key the
        // tool row off CallStart's arguments preview, and an orphan CallEnd
        // both drops the command text and breaks the renderer's expectation
        // that new tool rows only appear at CallStart.
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
        let _ = tx.send(ToolEvent::CallEnd {
            tool_name: tc.name.clone(),
            tool_call_id: tc.id.clone(),
            result_data: msg,
            ok: false,
            duration_ms: 0,
        });
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
    response: &LLMResponse,
) {
    let tc_json: Vec<Value> = routed_tool_calls
        .iter()
        .map(|tc| tc.to_openai_json())
        .collect();

    ContextBuilder::add_assistant_message(
        &mut ctx.messages,
        response.content.as_deref(),
        Some(&tc_json),
    );
    // Durable intent precedes execution, including side-effect tools.
    ctx.persist_pending_protocol_messages().await;

    // Response boundary enforcement: when this call was nudged to respond,
    // side-effect tools are rejected with an error result instead of having
    // been stripped from the schema (schema churn changes the prompt head
    // and breaks server-side prefix caching; an error result appends at the
    // tail and is cache-safe).
    let blocks = boundary_blocks_side_effects(
        ctx.flow.boundary == ResponseBoundary::Armed,
        response.content.as_deref(),
    );
    let (blocked, allowed): (Vec<&ToolCallRequest>, Vec<&ToolCallRequest>) = routed_tool_calls
        .iter()
        .partition(|tc| blocks && requires_result_report(&tc.name));
    for tc in &blocked {
        inject_boundary_rejection(ctx, tc);
    }
    ctx.persist_pending_protocol_messages().await;

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

    let ordered_results = execute_tool_calls_ordered(
        &allowed,
        &ctx.tools,
        &ctx.tool_event_tx,
        &ctx.cancellation_token,
        ctx.core.tool_heartbeat_secs,
        taints,
    )
    .await;
    let result_cap = inline_hot_prompt_result_cap_for_ctx_batch(ctx, ordered_results.len());
    let force_stash_raw = ordered_results.len() > 1;
    for result in &ordered_results {
        inject_tool_result(ctx, result, result_cap, force_stash_raw).await;
    }

    // Behavioral response-boundary arming. `parallel`/`sequential` hold only the
    // EXECUTED (non-blocked) calls, so a boundary-rejected call cannot re-arm.
    // Failed side-effect calls (timeouts, errors) don't arm either: there is no
    // result to report, so retrying is the model's legitimate next move — the
    // duplicate-call guard still bounds runaway retries.
    // A staged write has not changed its target and must be allowed to receive
    // the next piece. Only the final/one-call write arms the report boundary.
    let executed: Vec<&str> = ordered_results
        .iter()
        .filter_map(completed_reportable_tool)
        .collect();

    // No tool actually ran this round — every call was boundary-rejected. The
    // round made no progress, so the loop should not count it as an iteration.
    if executed.is_empty() && !blocked.is_empty() {
        ctx.flow.round_executed_no_tools = true;
    }

    if should_arm_boundary(response.content.as_deref(), &executed) {
        ctx.flow.boundary = ResponseBoundary::Pending;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

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

        async fn execute(&self, _params: HashMap<String, Value>) -> String {
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
                "Error: probe failure".to_string()
            } else {
                format!("{} complete", self.name)
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
    fn test_compact_inline_tool_result_keeps_short_data_raw() {
        let args = HashMap::new();
        let data = "short result";
        assert_eq!(
            compact_inline_tool_result("read_file", &args, data, 100),
            data
        );
    }

    #[test]
    fn test_compact_inline_tool_result_caps_large_data() {
        let mut args = HashMap::new();
        args.insert("path".to_string(), serde_json::json!("src/lib.rs"));
        args.insert("lines".to_string(), serde_json::json!("1:1000"));
        let data = format!(
            "{}MIDDLE_SHOULD_BE_OMITTED{}",
            "head line\n".repeat(200),
            "tail line\n".repeat(200)
        );

        let compacted = compact_inline_tool_result("read_file", &args, &data, 900);

        assert!(compacted.chars().count() <= 900);
        assert!(compacted.contains("[truncated: read_file"));
        assert!(compacted.contains("re-request with a narrower range/query"));
        assert!(compacted.contains("\n[...]\n"));
        assert!(!compacted.contains("MIDDLE_SHOULD_BE_OMITTED"));
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
        assert!(injected.contains("recall_tool_result"));
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
    async fn replay_byte_cap_stashes_even_below_configured_char_cap() {
        let temp = tempfile::tempdir().unwrap();
        let sessions = crate::session::SessionDb::new(&temp.path().join("sessions.db"));
        let session = sessions.create_session("test:replay-byte-cap").await;
        let data = format!(
            "{}MIDDLE_SECRET{}",
            "head\n".repeat(1200),
            "tail\n".repeat(600)
        );
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
        assert!(injected.contains("recall_tool_result"));
        assert!(injected.contains("call_replay_cap"));
        assert!(!injected.contains("MIDDLE_SECRET"));
        let replay_body = crate::agent::context_hygiene::cap_tool_result_for_replay(&injected);
        assert!(
            replay_body.len() <= TOOL_RESULT_REPLAY_MAX_BYTES + 40,
            "final prompt body must be replay-cap stable"
        );
        assert!(replay_body.contains("recall_tool_result"));
        assert!(replay_body.contains("call_replay_cap"));
        assert_eq!(
            sessions
                .load_tool_result(&session.id, "call_replay_cap")
                .await
                .as_deref(),
            Some(data.as_str())
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
        // Points the model at the recall tool with the right id.
        assert!(out.contains("recall_tool_result"));
        assert!(out.contains("call_42"));
    }

    /// A recalled body that exceeds the replay cap must NOT enter live context
    /// raw — that is the 2026-07-30 regression (a 172KB recall inflated a
    /// session to 77k tokens and dropped the cache). The digest must (a) bound
    /// it, (b) point the model at slice/search against the recall's own id,
    /// and (c) NOT point back at recall_tool_result (which would loop).
    #[test]
    fn digest_tool_result_caps_recalled_body_and_points_to_slice_search() {
        let args = HashMap::new();
        // ~200KB body, far over the replay cap.
        let data = format!(
            "{}NEVER_INLINE_THIS_SECRET{}",
            "head line\n".repeat(10_000),
            "tail line\n".repeat(10_000)
        );
        let cap = TOOL_RESULT_REPLAY_MAX_BYTES;

        let out = digest_tool_result("recall_tool_result", &args, &data, cap, "recall_call_7");

        // (a) bounded — never the raw 200KB.
        assert!(
            out.chars().count() <= cap + 40,
            "recalled body must be capped to ~replay budget, got {} chars",
            out.chars().count()
        );
        assert!(
            !out.contains("NEVER_INLINE_THIS_SECRET"),
            "the oversized middle must not enter the preview"
        );
        // (b) the model can recover parts via slice/search against this id.
        assert!(out.contains("slice_tool_result") || out.contains("search_tool_result"));
        assert!(out.contains("recall_call_7"));
        // (c) NOT a circular recall pointer — the model just recalled it.
        assert!(
            !out.contains("call recall_tool_result"),
            "recalled-body preview must not point back at recall_tool_result (loop)"
        );
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
            assert!(out.contains("recall_tool_result"));
            assert!(out.contains(&format!("call_{idx}")));
            assert!(!out.contains("MIDDLE_SECRET"));
        }
    }

    #[test]
    fn test_local_model_key_strips_internal_prefix() {
        assert_eq!(
            local_model_key("local:Qwen3.6-35B-A3B-4bit"),
            "qwen3.6-35b-a3b-4bit"
        );
        assert_eq!(
            local_model_key("Qwen3.6-35B-A3B-4bit"),
            "qwen3.6-35b-a3b-4bit"
        );
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

    #[test]
    fn test_delegation_reuses_main_local_model() {
        // Same local model (with/without the "local:" prefix, any case) → reuse
        // → must NOT delegate (delegation would evict the main prefix cache).
        assert!(delegation_reuses_main_local_model(
            true,
            "local:qwen36-35b",
            Some("qwen36-35b")
        ));
        assert!(delegation_reuses_main_local_model(
            true,
            "Qwen36-35B",
            Some("local:qwen36-35b")
        ));
        // A genuinely separate delegation model → real offload → delegate.
        assert!(!delegation_reuses_main_local_model(
            true,
            "local:qwen36-35b",
            Some("qwen3.5-2b")
        ));
        // Cloud main model → no shared local KV cache → delegation is fine.
        assert!(!delegation_reuses_main_local_model(
            false,
            "claude-opus-4-6",
            Some("claude-opus-4-6")
        ));
        // No delegation model resolved → cannot reuse.
        assert!(!delegation_reuses_main_local_model(
            true,
            "local:qwen36-35b",
            None
        ));
    }

    #[test]
    fn test_last_tool_result_snippet_skips_rejections_and_truncates() {
        let messages = vec![
            serde_json::json!({"role": "user", "content": "hi"}),
            serde_json::json!({"role": "tool", "name": "exec", "content": "[VERBATIM TOOL OUTPUT — do not paraphrase] line one\nline two"}),
            serde_json::json!({"role": "tool", "name": "exec", "content": "response boundary: exec was not executed"}),
        ];
        // Boundary rejection is skipped; the real result is found, marker
        // stripped, newlines flattened.
        let s = last_tool_result_snippet(&messages, 160).unwrap();
        assert_eq!(s, "line one line two");
        // Truncation respects char boundaries.
        let s = last_tool_result_snippet(&messages, 4).unwrap();
        assert_eq!(s, "line");
        // No tool results at all → None.
        assert!(last_tool_result_snippet(&messages[..1], 160).is_none());
    }

    #[test]
    fn test_should_arm_boundary_behavioral_matrix() {
        // The response boundary is BEHAVIORAL: it arms only when a side-effect
        // tool (exec/write/edit_file) actually ran AND the assistant produced no text
        // report. This is the whole point of the fix — a model that narrates its
        // work must not be throttled.

        // ran side-effect + no report -> ARM (force a report next call)
        assert!(should_arm_boundary(None, &["exec"]));
        assert!(should_arm_boundary(Some(""), &["write_file"]));
        assert!(should_arm_boundary(None, &["edit_file"]));
        assert!(should_arm_boundary(Some("  \n\t "), &["exec", "read_file"]));

        // ran side-effect + reported -> do NOT arm (the regression being fixed:
        // narrated consecutive exec/write/edit chains were being rejected ~1/3
        // of the time)
        assert!(!should_arm_boundary(
            Some("Running wc -l to size the files."),
            &["exec"]
        ));
        assert!(!should_arm_boundary(
            Some("Writing the summary now."),
            &["write_file"]
        ));
        assert!(!should_arm_boundary(
            Some("Updating the file now."),
            &["edit_file"]
        ));

        // ordinary read tools do not arm
        assert!(!should_arm_boundary(None, &["read_file", "list_dir"]));
        assert!(!should_arm_boundary(
            Some("here are the files"),
            &["read_file"]
        ));
        assert!(!should_arm_boundary(None, &[]));

        // Memory operations must be consumed before another memory call. This
        // ensures a recall/remember round gets a clean answer attempt.
        assert!(should_arm_boundary(None, &["recall"]));
        assert!(should_arm_boundary(None, &["remember"]));
        assert!(!should_arm_boundary(
            Some("I found the requested preference."),
            &["recall"]
        ));
    }

    #[test]
    fn staged_write_does_not_arm_boundary_until_publish() {
        let result = |state: Option<&str>| {
            let mut arguments = HashMap::new();
            if let Some(state) = state {
                arguments.insert("state".to_string(), json!(state));
            }
            SingleToolResult {
                tool_name: "write_file".to_string(),
                tool_id: "call_write".to_string(),
                arguments,
                result: crate::agent::tools::base::ToolExecutionResult::success("ok".to_string()),
                duration_ms: 0,
            }
        };

        assert_eq!(completed_reportable_tool(&result(Some("more"))), None);
        assert_eq!(completed_reportable_tool(&result(Some(" MORE "))), None);
        assert_eq!(completed_reportable_tool(&result(Some("MoRe"))), None);
        assert_eq!(
            completed_reportable_tool(&result(Some("complete"))),
            Some("write_file")
        );
        assert_eq!(completed_reportable_tool(&result(None)), Some("write_file"));
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
            let result = execute_single_tool(&staged, &registry, &None, &None, 60, None).await;
            assert!(result.result.ok, "{:?}", result.result.error);
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
        let result = execute_single_tool(&final_piece, &registry, &None, &None, 60, None).await;
        assert!(result.result.ok, "{:?}", result.result.error);
        assert_eq!(std::fs::read_to_string(path).unwrap(), "once-done");
    }

    #[test]
    fn test_boundary_blocks_side_effects_is_behavioral() {
        // Armed + silent response -> block (the nudge was ignored).
        assert!(boundary_blocks_side_effects(true, None));
        assert!(boundary_blocks_side_effects(true, Some("")));
        assert!(boundary_blocks_side_effects(true, Some("  \n ")));

        // Armed + text report in the same response -> the model complied;
        // its side-effect calls must run (no spurious "bad exec before a
        // good one").
        assert!(!boundary_blocks_side_effects(
            true,
            Some("Empty response — let me check the service.")
        ));

        // Not armed -> never blocks.
        assert!(!boundary_blocks_side_effects(false, None));
        assert!(!boundary_blocks_side_effects(false, Some("hi")));
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
        assert!(!results[0].result.ok);
        assert!(results[1].result.ok);
        assert!(results[2].result.ok);
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
        assert!(results.iter().all(|result| !result.result.ok));
        assert!(results
            .iter()
            .all(|result| result.result.data.contains("cancelled")));
    }

    #[test]
    fn test_is_routed_call_matches_routed_id() {
        // A result whose id matches a routed tool call must be considered
        // routed (so the TUI gets a CallEnd for it).
        let routed = vec![make_tc("read_file", "tc_routed_1")];
        assert!(is_routed_call("tc_routed_1", &routed));
    }

    #[test]
    fn test_is_routed_call_skips_runner_scratchpad_id() {
        // Tool runner internal scratchpad calls use synthetic ids like
        // "sp0000001" (see tool_runner.rs). They must NOT be considered
        // routed — otherwise every delegated tool call produces a duplicate
        // CallEnd block in the TUI.
        let routed = vec![make_tc("read_file", "tc_routed_1")];
        assert!(!is_routed_call("sp0000001", &routed));
        assert!(!is_routed_call("tc_other", &routed));
    }

    #[test]
    fn test_is_routed_call_handles_multiple_routed() {
        let routed = vec![make_tc("read_file", "id_a"), make_tc("exec", "id_b")];
        assert!(is_routed_call("id_a", &routed));
        assert!(is_routed_call("id_b", &routed));
        assert!(!is_routed_call("id_c", &routed));
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

    /// STEP 2: the canonical handle must be a pure deterministic function of
    /// (id, tool, ok, stored_bytes, args). Same inputs → byte-identical output
    /// across calls (write-once-stable → no prefix-cache drift). The full body
    /// must NOT be a substring of the handle (handles carry only a tiny
    /// excerpt; the body lives in the stash).
    #[test]
    fn render_tool_result_handle_is_deterministic_and_hides_body() {
        let body = "line one with specific content\nline two\nline three";
        let mut args = HashMap::new();
        args.insert(
            "path".to_string(),
            Value::String("src/main.rs".to_string()),
        );
        args.insert(
            "ignored_thing".to_string(),
            Value::String("should not appear".to_string()),
        );

        let h1 = render_tool_result_handle("call_42", "read_file", true, body.as_bytes(), &args);
        let h2 = render_tool_result_handle("call_42", "read_file", true, body.as_bytes(), &args);

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
        // The handle points the model at recall_tool_result for the full body.
        assert!(
            h1.contains("recall_tool_result") && h1.contains("call_42"),
            "handle must reference recall_tool_result and the id; got: {h1}"
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
}
