//! Tool execution engine: delegated and inline paths.
//!
//! Extracted from `agent_loop.rs` to isolate tool execution logic.

use std::sync::atomic::Ordering;
use std::time::Duration;

use serde_json::{json, Value};
use tracing::{debug, info, warn};

use crate::agent::agent_core::RuntimeCounters;
use crate::agent::audit::ToolEvent;
use crate::agent::context::ContextBuilder;
use crate::agent::role_policy;
use crate::agent::tool_runner::{self, Budget, ToolRunnerConfig};
use crate::providers::base::{LLMResponse, ToolCallRequest};
use std::sync::Arc;

use super::agent_loop::TurnContext;

/// Execute tool calls via the delegation (tool-runner) path.
///
/// Returns `true` if delegation was used (caller should `continue` the main loop).
/// Returns `false` if delegation couldn't proceed (caller should fall through to inline).
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
        _ => return false,
    };

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

    // The delegation model has a small context (8K tokens).
    // Cap tool results to ~750 tokens (~3000 chars) so the system
    // prompt, tool call messages, and response all fit comfortably.
    // Use the main model's limit only if it's already smaller.
    let delegation_result_limit = ctx.core.max_tool_result_chars.min(3000);

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

        let injected = ctx.content_gate.admit_simple(full_data).into_text();

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
                None, // TODO: wire ContentGate here
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
                "[tool runner output]"
            } else {
                "[tool runner summary]"
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
    }

    // Set response boundary flag if any delegated tool was exec/write_file.
    for (_, tool_name, _) in &run_result.tool_results {
        if tool_name == "exec" || tool_name == "write_file" {
            ctx.flow.force_response = true;
            break;
        }
    }

    true
}

/// Execute tool calls via the inline (direct) path.
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

    // Execute each tool call.
    for tc in routed_tool_calls {
        debug!("Executing tool: {} (id: {})", tc.name, tc.id);

        // Emit tool call start event.
        if let Some(ref tx) = ctx.tool_event_tx {
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

        // Spawn heartbeat that emits Progress every 2s for tools that
        // don't emit their own (everything except exec).
        let heartbeat = if let Some(ref tx) = ctx.tool_event_tx {
            let hb_tx = tx.clone();
            let hb_name = tc.name.clone();
            let hb_id = tc.id.clone();
            let hb_start = start;
            Some(tokio::spawn(async move {
                let mut interval = tokio::time::interval(Duration::from_secs(2));
                interval.tick().await; // skip first immediate tick
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

        let result = if let Some(ref tx) = ctx.tool_event_tx {
            use crate::agent::tools::base::ToolExecutionContext;
            let exec_ctx = ToolExecutionContext {
                event_tx: tx.clone(),
                cancellation_token: ctx
                    .cancellation_token
                    .as_ref()
                    .map(|t| t.child_token())
                    .unwrap_or_else(tokio_util::sync::CancellationToken::new),
                tool_call_id: tc.id.clone(),
            };
            ctx.tools
                .execute_with_context(&tc.name, tc.arguments.clone(), &exec_ctx)
                .await
        } else {
            ctx.tools.execute(&tc.name, tc.arguments.clone()).await
        };

        // Stop heartbeat when tool finishes.
        if let Some(hb) = heartbeat {
            hb.abort();
        }
        let duration_ms = start.elapsed().as_millis() as u64;
        debug!(
            "Tool {} result ({}B, ok={}, {}ms)",
            tc.name,
            result.data.len(),
            result.ok,
            duration_ms
        );
        // Gate tool result through context budget.
        let data = ctx.content_gate.admit_simple(&result.data).into_text();
        if ctx.core.provenance_config.enabled {
            ContextBuilder::add_tool_result_immutable(
                &mut ctx.messages,
                &tc.id,
                &tc.name,
                &data,
            );
        } else {
            ContextBuilder::add_tool_result(&mut ctx.messages, &tc.id, &tc.name, &data);
        }

        // Emit tool call end event.
        if let Some(ref tx) = ctx.tool_event_tx {
            let _ = tx.send(ToolEvent::CallEnd {
                tool_name: tc.name.clone(),
                tool_call_id: tc.id.clone(),
                result_data: result.data.clone(),
                ok: result.ok,
                duration_ms,
            });
        }

        // Record in audit log.
        if let Some(ref audit) = ctx.audit {
            let args_value = serde_json::to_value(&tc.arguments).unwrap_or(json!({}));
            let _ = audit.record(
                &tc.name,
                &tc.id,
                &args_value,
                &result.data,
                result.ok,
                duration_ms,
                "inline",
            );
        }

        // Track used tools.
        ctx.used_tools.insert(tc.name.clone());

        // Collect for turn audit summary.
        ctx.turn_tool_entries
            .push(crate::agent::audit::TurnToolEntry {
                name: tc.name.clone(),
                id: tc.id.clone(),
                ok: result.ok,
                duration_ms,
                result_chars: result.data.len(),
            });

        // Record tool outcome for learning.
        let context_str: String = tc
            .arguments
            .values()
            .filter_map(|v| v.as_str())
            .next()
            .unwrap_or_default()
            .chars()
            .take(100)
            .collect();
        ctx.core.learning.record(
            &tc.name,
            result.ok,
            &context_str,
            result.error.as_deref(),
        );

        // Set response boundary flag for exec/write_file.
        if tc.name == "exec" || tc.name == "write_file" {
            ctx.flow.force_response = true;
        }
    }
}
