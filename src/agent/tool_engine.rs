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

use super::agent_loop::TurnContext;

const LARGE_TOOL_RESULT_TOKEN_THRESHOLD: usize = 500;

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

    // Local delegation providers require user-last messages
    // for tool→generate flow.
    // Check delegation provider's locality, not the main model's.
    let needs_user_cont = ctx.core.is_local || delegation_provider.is_some();

    // Detect [VERBATIM] marker: the main model is asking for
    // raw tool output instead of a delegation summary.
    let verbatim = response
        .content
        .as_ref()
        .map(|c| c.contains("[VERBATIM]"))
        .unwrap_or(false);

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
        needs_user_continuation: needs_user_cont,
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
            let preview: String = serde_json::to_string(&tc.arguments)
                .unwrap_or_default()
                .chars()
                .take(80)
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
    let run_result = tool_runner::run_tool_loop(
        &runner_config,
        routed_tool_calls,
        &ctx.tools,
        &task_desc,
    )
    .await;
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
                format!(
                    "[{}]: {}",
                    name,
                    data.chars().take(200).collect::<String>()
                )
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
        counters
            .delegation_healthy
            .store(false, Ordering::Relaxed);
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
        counters
            .delegation_healthy
            .store(true, Ordering::Relaxed);
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

        let full_tokens =
            crate::agent::token_budget::TokenBudget::estimate_str_tokens(full_data);

        let injected = if let Some(ref summary) = run_result.summary {
            // Summary exists from scratch-pad analysis.
            if full_tokens > LARGE_TOOL_RESULT_TOKEN_THRESHOLD {
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
        } else if ctx.core.specialist_provider.is_some()
            && full_tokens > LARGE_TOOL_RESULT_TOKEN_THRESHOLD
        {
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

        if ctx.core.provenance_config.enabled {
            ContextBuilder::add_tool_result_immutable(
                &mut ctx.messages,
                &tc.id,
                &tc.name,
                &injected,
            );
        } else {
            ContextBuilder::add_tool_result(
                &mut ctx.messages,
                &tc.id,
                &tc.name,
                &injected,
            );
        }
        ctx.flow.tool_guard.record_result(&tc.name, &tc.arguments, injected.clone());
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
                Some(&mut ctx.content_gate),  // Wire ContentGate for budget-aware truncation
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

        if let Some(ref tx) = ctx.tool_event_tx {
            let _ = tx.send(ToolEvent::CallEnd {
                tool_name: tool_name.clone(),
                tool_call_id: tool_call_id.clone(),
                result_data: data.clone(),
                ok,
                duration_ms: per_tool_ms,
            });
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

        ctx.core.learning.record_extended(
            tool_name,
            ok,
            &data.chars().take(100).collect::<String>(),
            if ok { None } else { Some(data) },
            Some(&tr_model),
            Some(&tr_model),
            None,
        );
        ctx.used_tools.insert(tool_name.clone());

        // Taint tracking: mark context tainted when a web tool ran via delegation.
        // We don't have the original arguments here, so pass None for detail.
        ctx.taint_state.mark_tainted(tool_name, None);
    }

    // Set response boundary flag if any delegated tool was exec/write_file.
    for (_, tool_name, _) in &run_result.tool_results {
        if tool_name == "exec" || tool_name == "write_file" {
            ctx.flow.force_response = true;
            break;
        }
    }

    tracing::Span::current().record("outcome", "ok");
    true
}

/// Returns `true` if a tool is safe to execute in parallel with other
/// parallel-safe tools. These are read-only operations that do not mutate
/// any shared state and can safely race each other.
fn is_parallel_safe(tool_name: &str) -> bool {
    matches!(
        tool_name,
        "read_file" | "list_dir" | "web_fetch" | "web_search" | "read_skill"
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
            let preview: String = serde_json::to_string(&tc.arguments)
                .unwrap_or_default()
                .chars()
                .take(80)
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
    let data = if ctx.core.specialist_provider.is_some()
        && crate::agent::token_budget::TokenBudget::estimate_str_tokens(&result_data) > 500
    {
        ctx.content_gate
            .admit_with_specialist(
                &result_data,
                ctx.core.specialist_provider.as_ref().unwrap().as_ref(),
                ctx.core.specialist_model.as_deref().unwrap_or(""),
            )
            .await
            .into_text()
    } else {
        ctx.content_gate.admit_simple(&result_data).into_text()
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

    // Learning.
    let context_str: String = r
        .arguments
        .values()
        .filter_map(|v| v.as_str())
        .next()
        .unwrap_or_default()
        .chars()
        .take(100)
        .collect();
    ctx.core.learning.record(
        &r.tool_name,
        r.result.ok,
        &context_str,
        r.result.error.as_deref(),
    );

    // Response boundary flag.
    if r.tool_name == "exec" || r.tool_name == "write_file" {
        ctx.flow.force_response = true;
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

    // Partition into parallel-safe and sequential tool calls.
    let (parallel, sequential): (Vec<_>, Vec<_>) = routed_tool_calls
        .iter()
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
    fn test_is_parallel_safe_classification() {
        // Parallel-safe tools
        assert!(is_parallel_safe("read_file"));
        assert!(is_parallel_safe("list_dir"));
        assert!(is_parallel_safe("web_fetch"));
        assert!(is_parallel_safe("web_search"));
        assert!(is_parallel_safe("read_skill"));
        // Must serialize
        assert!(!is_parallel_safe("exec"));
        assert!(!is_parallel_safe("write_file"));
        assert!(!is_parallel_safe("edit_file"));
        assert!(!is_parallel_safe("spawn"));
        // Unknown defaults to serial
        assert!(!is_parallel_safe("unknown_tool"));
    }

    #[test]
    fn test_mixed_tools_partition_correctly() {
        let calls = vec![
            make_tc("read_file", "1"),
            make_tc("exec", "2"),
            make_tc("list_dir", "3"),
            make_tc("write_file", "4"),
        ];
        let (par, seq): (Vec<_>, Vec<_>) =
            calls.iter().partition(|tc| is_parallel_safe(&tc.name));
        assert_eq!(par.len(), 2);
        assert_eq!(seq.len(), 2);
        assert_eq!(par[0].name, "read_file");
        assert_eq!(par[1].name, "list_dir");
        assert_eq!(seq[0].name, "exec");
        assert_eq!(seq[1].name, "write_file");
    }

    #[test]
    fn test_all_parallel_safe_no_sequential() {
        let calls = vec![
            make_tc("read_file", "1"),
            make_tc("list_dir", "2"),
            make_tc("web_search", "3"),
        ];
        let (par, seq): (Vec<_>, Vec<_>) =
            calls.iter().partition(|tc| is_parallel_safe(&tc.name));
        assert_eq!(par.len(), 3);
        assert!(seq.is_empty());
    }

    #[test]
    fn test_all_sequential_no_parallel() {
        let calls = vec![make_tc("exec", "1"), make_tc("write_file", "2")];
        let (par, seq): (Vec<_>, Vec<_>) =
            calls.iter().partition(|tc| is_parallel_safe(&tc.name));
        assert!(par.is_empty());
        assert_eq!(seq.len(), 2);
    }

    #[test]
    fn test_single_tool_partitions_correctly() {
        // Single parallel-safe tool
        let calls = vec![make_tc("read_file", "1")];
        let (par, seq): (Vec<_>, Vec<_>) =
            calls.iter().partition(|tc| is_parallel_safe(&tc.name));
        assert_eq!(par.len(), 1);
        assert!(seq.is_empty());

        // Single serial tool
        let calls = vec![make_tc("exec", "1")];
        let (par, seq): (Vec<_>, Vec<_>) =
            calls.iter().partition(|tc| is_parallel_safe(&tc.name));
        assert!(par.is_empty());
        assert_eq!(seq.len(), 1);
    }
}
