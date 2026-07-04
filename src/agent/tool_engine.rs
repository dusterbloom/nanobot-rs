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

const LARGE_TOOL_RESULT_TOKEN_THRESHOLD: usize = 500;
const INLINE_TOOL_RESULT_HOT_PROMPT_MAX_CHARS: usize = 2_400;
const INLINE_TOOL_RESULT_HOT_PROMPT_MIN_CHARS: usize = 800;

/// Per-tool token threshold above which a raw tool result is replaced by a
/// summary. Enumerative tools (`exec`, `list_dir`, `web_search`, `read_file`)
/// return specific strings — filenames, URLs, error lines — that the model
/// needs to quote verbatim. Summaries destroy them, which the model then
/// papers over by fabricating. Keep raw output for these up to ~4000 tokens.
fn summary_threshold_tokens(tool_name: &str) -> usize {
    match tool_name {
        "exec" | "list_dir" | "find_files" | "search_files" | "search_context" | "file_info"
        | "file_preview" | "batch" | "workspace_diff" | "system_info" | "tool_status"
        | "web_search" | "read_file" => 4000,
        _ => LARGE_TOOL_RESULT_TOKEN_THRESHOLD,
    }
}

fn inline_hot_prompt_result_cap(ctx: &TurnContext) -> usize {
    ctx.core
        .max_tool_result_chars
        .min(INLINE_TOOL_RESULT_HOT_PROMPT_MAX_CHARS)
        .max(INLINE_TOOL_RESULT_HOT_PROMPT_MIN_CHARS)
}

fn tool_arg_summary(args: &std::collections::HashMap<String, Value>) -> String {
    let mut parts = Vec::new();
    for key in [
        "path",
        "lines",
        "query",
        "pattern",
        "url",
        "command",
        "glob",
        "max_lines",
    ] {
        if let Some(value) = args.get(key) {
            let rendered = value
                .as_str()
                .map(str::to_string)
                .unwrap_or_else(|| value.to_string());
            parts.push(format!("{}={}", key, rendered));
        }
    }

    let summary = if parts.is_empty() {
        "(arguments omitted)".to_string()
    } else {
        parts.join(", ")
    };

    summary.chars().take(260).collect()
}

fn compact_inline_tool_result(
    tool_name: &str,
    args: &std::collections::HashMap<String, Value>,
    data: &str,
    max_chars: usize,
) -> String {
    let total_chars = data.chars().count();
    if total_chars <= max_chars {
        return data.to_string();
    }

    let source = tool_arg_summary(args);
    let estimated_tokens = crate::agent::token_budget::TokenBudget::estimate_str_tokens(data);
    let header = format!(
        "[truncated: {tool_name}({source}) returned {total_chars} chars (~{estimated_tokens} tokens); \
         head+tail shown — re-request with a narrower range/query if the middle is needed]\n"
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
    matches!(name, "exec" | "write_file")
}

/// Decide whether to arm the response boundary after a tool-execution round.
///
/// Behavioral, not positional: arm ONLY when a side-effect tool (exec/write_file)
/// actually ran AND the assistant produced no text report. A model that narrates
/// each step is not fabricating results, so it must not be throttled. Arming
/// blindly after every side-effect call (the prior behavior) rejected ~1/3 of
/// legitimate consecutive exec/write_file calls. `executed_tools` must contain
/// only the tools that actually executed — never boundary-rejected calls — so a
/// rejected call cannot re-arm the boundary.
fn should_arm_boundary(assistant_content: Option<&str>, executed_tools: &[&str]) -> bool {
    let reported = assistant_content.is_some_and(|c| !c.trim().is_empty());
    let ran_side_effect = executed_tools.iter().any(|n| is_side_effect_tool(n));
    ran_side_effect && !reported
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

    // Build the assistant message with original tool_calls.
    let tc_json: Vec<Value> = routed_tool_calls
        .iter()
        .map(|tc| tc.to_openai_json())
        .collect();
    ContextBuilder::add_assistant_message(
        &mut ctx.messages,
        response.content.as_deref(),
        Some(&tc_json),
    );

    // Add tool results from the runner to the main context.
    let preview_max = ctx.core.tool_delegation_config.max_result_preview_chars;

    for tc in routed_tool_calls {
        let full_data = run_result
            .tool_results
            .iter()
            .find(|(id, _, _)| id == &tc.id)
            .map(|(_, _, data)| data.as_str())
            .unwrap_or("(no result)");

        let full_tokens = crate::agent::token_budget::TokenBudget::estimate_str_tokens(full_data);

        let threshold = summary_threshold_tokens(&tc.name);
        let cap = inline_hot_prompt_result_cap(ctx);
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
        let injected = compact_inline_tool_result(&tc.name, &tc.arguments, &injected_raw, cap);

        if ctx.core.provenance_config.enabled {
            ContextBuilder::add_tool_result_immutable(
                &mut ctx.messages,
                &tc.id,
                &tc.name,
                &injected,
            );
        } else {
            ContextBuilder::add_tool_result(&mut ctx.messages, &tc.id, &tc.name, &injected);
        }
        ctx.flow
            .tool_guard
            .record_result(&tc.name, &tc.arguments, injected.clone());
        ctx.used_tools.insert(tc.name.clone());
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
            ctx.messages.push(json!({
                "role": "user",
                "content": format!("{} {}", prefix, summary_text)
            }));
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

/// Returns `true` if a tool is safe to execute in parallel with other
/// parallel-safe tools. These are read-only operations that do not mutate
/// any shared state and can safely race each other.
fn is_parallel_safe(tool_name: &str) -> bool {
    matches!(
        tool_name,
        "read_file"
            | "file_preview"
            | "list_dir"
            | "find_files"
            | "search_files"
            | "search_context"
            | "file_info"
            | "batch"
            | "workspace_diff"
            | "system_info"
            | "tool_status"
            | "web_fetch"
            | "web_search"
            | "read_skill"
    )
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

        let result = if let Some(ref tx) = tool_event_tx {
            use crate::agent::tools::base::ToolExecutionContext;
            let exec_ctx = ToolExecutionContext {
                event_tx: tx.clone(),
                cancellation_token: cancellation_token
                    .as_ref()
                    .map(|t| t.child_token())
                    .unwrap_or_else(tokio_util::sync::CancellationToken::new),
                tool_call_id: tc.id.clone(),
            };
            tools
                .execute_with_context(&tc.name, tc.arguments.clone(), &exec_ctx)
                .await
        } else {
            tools.execute(&tc.name, tc.arguments.clone()).await
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

/// Post-process one completed tool result: gate content, inject into messages,
/// emit CallEnd, audit, update taint/learning/force_response.
///
/// This function must run sequentially (one result at a time) because it
/// mutates `ctx`.
async fn inject_tool_result(ctx: &mut TurnContext, r: &SingleToolResult) {
    // For web_fetch/web_search: unwrap the JSON envelope so the model
    // sees clean article text rather than a JSON metadata summary.
    let result_data = if r.tool_name == "web_fetch" || r.tool_name == "web_search" {
        crate::agent::tools::web::extract_web_content(&r.result.data)
    } else {
        r.result.data.clone()
    };

    // Gate tool result through context budget.
    let threshold = summary_threshold_tokens(&r.tool_name);
    let cap = inline_hot_prompt_result_cap(ctx);
    let data = if ctx.core.specialist_provider.is_some()
        && crate::agent::token_budget::TokenBudget::estimate_str_tokens(&result_data) > threshold
    {
        let summarized = ctx
            .content_gate
            .admit_with_specialist(
                &result_data,
                ctx.core.specialist_provider.as_ref().unwrap().as_ref(),
                ctx.core.specialist_model.as_deref().unwrap_or(""),
            )
            .await
            .into_text();
        compact_inline_tool_result(&r.tool_name, &r.arguments, &summarized, cap)
    } else {
        let prompt_data = compact_inline_tool_result(&r.tool_name, &r.arguments, &result_data, cap);
        ctx.content_gate.admit_simple(&prompt_data).into_text()
    };

    if ctx.core.provenance_config.enabled {
        ContextBuilder::add_tool_result_immutable(
            &mut ctx.messages,
            &r.tool_id,
            &r.tool_name,
            &data,
        );
    } else {
        ContextBuilder::add_tool_result(&mut ctx.messages, &r.tool_id, &r.tool_name, &data);
    }
    ctx.flow
        .tool_guard
        .record_result(&r.tool_name, &r.arguments, data.clone());

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

/// Inject an error result for a side-effect tool call rejected by the
/// response boundary.
///
/// Deliberately narrower than `inject_tool_result`: no learning record, no
/// taint, no audit, and — critically — it does NOT re-arm the boundary (a
/// rejected call must not extend its own boundary, or the loop would
/// livelock: nudge → reject → nudge → …).
fn inject_boundary_rejection(ctx: &mut TurnContext, tc: &ToolCallRequest) {
    let msg = format!(
        "response boundary: {} was not executed — first respond with what the \
         previous tool results showed; it can run in a later step.",
        tc.name
    );
    if ctx.core.provenance_config.enabled {
        ContextBuilder::add_tool_result_immutable(&mut ctx.messages, &tc.id, &tc.name, &msg);
    } else {
        ContextBuilder::add_tool_result(&mut ctx.messages, &tc.id, &tc.name, &msg);
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
/// Parallel-safe tools (`read_file`, `list_dir`, `web_fetch`, `web_search`,
/// `read_skill`) are executed concurrently via `join_all`. All other tools are
/// executed sequentially. Post-processing always runs sequentially so that
/// `ctx` mutations are safe.
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
        .partition(|tc| blocks && is_side_effect_tool(&tc.name));
    for tc in &blocked {
        inject_boundary_rejection(ctx, tc);
    }

    // Partition into parallel-safe and sequential tool calls.
    let (parallel, sequential): (Vec<_>, Vec<_>) = allowed
        .into_iter()
        .partition(|tc| is_parallel_safe(&tc.name));

    // Build taint warnings up-front (immutable borrow of ctx.taint_state).
    let parallel_taints: Vec<Option<String>> = parallel
        .iter()
        .map(|tc| {
            if ctx.taint_state.check_sensitive(&tc.name).is_some() {
                Some(ctx.taint_state.taint_summary())
            } else {
                None
            }
        })
        .collect();

    // Execute the parallel-safe batch concurrently.
    let parallel_results: Vec<SingleToolResult> = if !parallel.is_empty() {
        let futs = parallel
            .iter()
            .zip(parallel_taints.into_iter())
            .map(|(tc, taint)| {
                execute_single_tool(
                    tc,
                    &ctx.tools,
                    &ctx.tool_event_tx,
                    &ctx.cancellation_token,
                    ctx.core.tool_heartbeat_secs,
                    taint,
                )
            });
        futures_util::future::join_all(futs).await
    } else {
        vec![]
    };

    // Post-process parallel results sequentially (ctx mutation is safe here).
    for r in &parallel_results {
        inject_tool_result(ctx, r).await;
    }

    // Execute sequential tools one at a time.
    for tc in &sequential {
        let taint = if ctx.taint_state.check_sensitive(&tc.name).is_some() {
            Some(ctx.taint_state.taint_summary())
        } else {
            None
        };
        let r = execute_single_tool(
            tc,
            &ctx.tools,
            &ctx.tool_event_tx,
            &ctx.cancellation_token,
            ctx.core.tool_heartbeat_secs,
            taint,
        )
        .await;
        inject_tool_result(ctx, &r).await;
    }

    // Behavioral response-boundary arming. `parallel`/`sequential` hold only the
    // EXECUTED (non-blocked) calls, so a boundary-rejected call cannot re-arm.
    let executed: Vec<&str> = parallel
        .iter()
        .chain(sequential.iter())
        .map(|tc| tc.name.as_str())
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
        assert!(compacted.contains("[truncated: read_file("));
        assert!(compacted.contains("path=src/lib.rs"));
        assert!(compacted.contains("lines=1:1000"));
        assert!(compacted.contains("re-request with a narrower range/query"));
        assert!(compacted.contains("\n[...]\n"));
        assert!(!compacted.contains("MIDDLE_SHOULD_BE_OMITTED"));
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
    fn test_is_parallel_safe_classification() {
        // Parallel-safe tools
        assert!(is_parallel_safe("read_file"));
        assert!(is_parallel_safe("file_preview"));
        assert!(is_parallel_safe("list_dir"));
        assert!(is_parallel_safe("find_files"));
        assert!(is_parallel_safe("search_files"));
        assert!(is_parallel_safe("search_context"));
        assert!(is_parallel_safe("file_info"));
        assert!(is_parallel_safe("batch"));
        assert!(is_parallel_safe("workspace_diff"));
        assert!(is_parallel_safe("system_info"));
        assert!(is_parallel_safe("tool_status"));
        assert!(is_parallel_safe("web_fetch"));
        assert!(is_parallel_safe("web_search"));
        assert!(is_parallel_safe("read_skill"));
        // Must serialize
        assert!(!is_parallel_safe("exec"));
        assert!(!is_parallel_safe("write_file"));
        assert!(!is_parallel_safe("edit_file"));
        assert!(!is_parallel_safe("apply_patch"));
        assert!(!is_parallel_safe("spawn"));
        // Unknown defaults to serial
        assert!(!is_parallel_safe("unknown_tool"));
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
    fn test_should_arm_boundary_behavioral_matrix() {
        // The response boundary is BEHAVIORAL: it arms only when a side-effect
        // tool (exec/write_file) actually ran AND the assistant produced no text
        // report. This is the whole point of the fix — a model that narrates its
        // work must not be throttled.

        // ran side-effect + no report -> ARM (force a report next call)
        assert!(should_arm_boundary(None, &["exec"]));
        assert!(should_arm_boundary(Some(""), &["write_file"]));
        assert!(should_arm_boundary(Some("  \n\t "), &["exec", "read_file"]));

        // ran side-effect + reported -> do NOT arm (the regression being fixed:
        // narrated consecutive exec/write_file chains were being rejected ~1/3
        // of the time)
        assert!(!should_arm_boundary(
            Some("Running wc -l to size the files."),
            &["exec"]
        ));
        assert!(!should_arm_boundary(
            Some("Writing the summary now."),
            &["write_file"]
        ));

        // no side-effect tool ran -> never arm, report or not
        assert!(!should_arm_boundary(None, &["read_file", "list_dir"]));
        assert!(!should_arm_boundary(
            Some("here are the files"),
            &["read_file"]
        ));
        assert!(!should_arm_boundary(None, &[]));
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

    #[test]
    fn test_mixed_tools_partition_correctly() {
        let calls = vec![
            make_tc("read_file", "1"),
            make_tc("exec", "2"),
            make_tc("list_dir", "3"),
            make_tc("write_file", "4"),
            make_tc("find_files", "5"),
        ];
        let (par, seq): (Vec<_>, Vec<_>) = calls.iter().partition(|tc| is_parallel_safe(&tc.name));
        assert_eq!(par.len(), 3);
        assert_eq!(seq.len(), 2);
        assert_eq!(par[0].name, "read_file");
        assert_eq!(par[1].name, "list_dir");
        assert_eq!(par[2].name, "find_files");
        assert_eq!(seq[0].name, "exec");
        assert_eq!(seq[1].name, "write_file");
    }

    #[test]
    fn test_all_parallel_safe_no_sequential() {
        let calls = vec![
            make_tc("read_file", "1"),
            make_tc("list_dir", "2"),
            make_tc("web_search", "3"),
            make_tc("file_info", "4"),
        ];
        let (par, seq): (Vec<_>, Vec<_>) = calls.iter().partition(|tc| is_parallel_safe(&tc.name));
        assert_eq!(par.len(), 4);
        assert!(seq.is_empty());
    }

    #[test]
    fn test_all_sequential_no_parallel() {
        let calls = vec![make_tc("exec", "1"), make_tc("write_file", "2")];
        let (par, seq): (Vec<_>, Vec<_>) = calls.iter().partition(|tc| is_parallel_safe(&tc.name));
        assert!(par.is_empty());
        assert_eq!(seq.len(), 2);
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

    #[test]
    fn test_single_tool_partitions_correctly() {
        // Single parallel-safe tool
        let calls = vec![make_tc("read_file", "1")];
        let (par, seq): (Vec<_>, Vec<_>) = calls.iter().partition(|tc| is_parallel_safe(&tc.name));
        assert_eq!(par.len(), 1);
        assert!(seq.is_empty());

        // Single serial tool
        let calls = vec![make_tc("exec", "1")];
        let (par, seq): (Vec<_>, Vec<_>) = calls.iter().partition(|tc| is_parallel_safe(&tc.name));
        assert!(par.is_empty());
        assert_eq!(seq.len(), 1);
    }
}
