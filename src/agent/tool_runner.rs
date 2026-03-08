#![allow(dead_code)]
//! Delegated tool execution loop.
//!
//! Receives initial tool calls from the main LLM, executes them via the
//! shared [`ToolRegistry`], and lets a cheap model decide if more tools are
//! needed. Returns aggregated results for injection into the main context.

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::{json, Value};
use tracing::{debug, info, warn};

use crate::agent::context::ContextBuilder;
use crate::agent::context_store::{self, ContextStore};
use crate::agent::sanitize::strip_tool_output;
use crate::agent::tools::ToolRegistry;
use crate::agent::worker_tools;
use crate::providers::base::{LLMProvider, ToolCallRequest};

/// Name of the delegate tool for recursive worker spawning.
pub const DELEGATE_TOOL: &str = "delegate";

/// Configuration for the tool runner loop.
pub struct ToolRunnerConfig {
    pub provider: Arc<dyn LLMProvider>,
    pub model: String,
    pub max_iterations: u32,
    pub max_tokens: u32,
    /// Maximum characters per tool result shown to the delegation model.
    /// Results exceeding this are truncated (with a marker) before the
    /// delegation model sees them, but full data is kept for the main model.
    pub max_tool_result_chars: usize,
    /// If all tool results from the initial execution are shorter than this
    /// threshold, skip the delegation LLM call entirely and return raw results.
    /// Set to 0 to disable short-circuiting. Default: 200.
    pub short_circuit_chars: usize,
    /// Recursion depth for ctx_summarize (default 0, max 2).
    pub depth: u32,
    /// Optional cancellation token. When cancelled, the loop stops between
    /// iterations and returns partial results collected so far.
    pub cancellation_token: Option<tokio_util::sync::CancellationToken>,
    /// If true, skip delegation LLM and return raw tool output unsummarized.
    /// Set by the main agent via `[VERBATIM]` marker in its response.
    pub verbatim: bool,
    /// Optional budget for recursive delegation. If None, delegate tool is disabled.
    pub budget: Option<Budget>,
}

/// Resource budget for a worker. Controls iterations, depth, cost, and timeouts.
#[derive(Debug, Clone)]
pub struct Budget {
    /// Maximum LLM iterations for this worker.
    pub max_iterations: u32,
    /// Maximum delegation depth (0 = can't delegate).
    pub max_depth: u32,
    /// Current depth in the delegation tree.
    pub current_depth: u32,
    /// Budget multiplier for children (0.5 = children get half).
    pub budget_multiplier: f32,
    /// Maximum cost in USD for this delegation (0.0 = unlimited).
    pub cost_limit: f64,
    /// Accumulated cost in USD so far.
    pub cost_spent: f64,
    /// Model prices for cost calculation.
    pub prices: Option<std::sync::Arc<crate::agent::model_prices::ModelPrices>>,
}

impl Budget {
    /// Create a root budget (depth 0).
    pub fn root(max_iterations: u32, max_depth: u32) -> Self {
        Self {
            max_iterations,
            max_depth,
            current_depth: 0,
            budget_multiplier: 0.5,
            cost_limit: 0.0,
            cost_spent: 0.0,
            prices: None,
        }
    }

    /// Create a root budget with cost tracking.
    pub fn root_with_cost(
        max_iterations: u32,
        max_depth: u32,
        cost_limit: f64,
        prices: std::sync::Arc<crate::agent::model_prices::ModelPrices>,
    ) -> Self {
        Self {
            max_iterations,
            max_depth,
            current_depth: 0,
            budget_multiplier: 0.5,
            cost_limit,
            cost_spent: 0.0,
            prices: Some(prices),
        }
    }

    /// Create a child budget with reduced iterations and incremented depth.
    pub fn child(&self) -> Option<Self> {
        if self.current_depth >= self.max_depth {
            return None;
        }
        Some(Self {
            max_iterations: ((self.max_iterations as f32) * self.budget_multiplier).max(1.0) as u32,
            max_depth: self.max_depth,
            current_depth: self.current_depth + 1,
            budget_multiplier: self.budget_multiplier,
            cost_limit: (self.cost_limit - self.cost_spent).max(0.0)
                * self.budget_multiplier as f64,
            cost_spent: 0.0,
            prices: self.prices.clone(),
        })
    }

    /// Check if delegation is allowed at current depth.
    pub fn can_delegate(&self) -> bool {
        self.current_depth < self.max_depth
    }

    /// Record cost from an LLM response and return the cost of this call.
    pub fn record_cost(
        &mut self,
        model: &str,
        usage: &std::collections::HashMap<String, i64>,
    ) -> f64 {
        let prices = match &self.prices {
            Some(p) => p,
            None => return 0.0,
        };
        let prompt_tokens = usage.get("prompt_tokens").copied().unwrap_or(0);
        let completion_tokens = usage.get("completion_tokens").copied().unwrap_or(0);
        let cost = prices.cost_of(model, prompt_tokens, completion_tokens);
        self.cost_spent += cost;
        cost
    }

    /// Check if the cost budget is exhausted.
    pub fn is_over_budget(&self) -> bool {
        self.cost_limit > 0.0 && self.cost_spent >= self.cost_limit
    }
}

/// Result of a delegated tool execution loop.
#[derive(Debug, Clone)]
pub struct ToolRunResult {
    /// Collected (tool_call_id, tool_name, result_data) from all iterations.
    pub tool_results: Vec<(String, String, String)>,
    /// Optional summary from the cheap model about what happened.
    pub summary: Option<String>,
    /// How many iterations were used.
    pub iterations_used: u32,
    /// Error message if the delegation LLM call failed.
    pub error: Option<String>,
}

/// Normalize a tool call key for dedup: sort JSON keys and use compact serialization.
pub(crate) fn normalize_call_key(name: &str, arguments: &HashMap<String, Value>) -> String {
    let mut sorted: Vec<_> = arguments.iter().collect();
    sorted.sort_by_key(|(k, _)| *k);
    let normalized = serde_json::to_string(&sorted).unwrap_or_default();
    format!("{}:{}", name, normalized)
}

/// Build a compact state string from ContextStore for the analysis model.
///
/// Includes variable metadata (name, length, preview) and accumulated
/// memory findings. Keeps output small (~200-400 chars) to leave the
/// model's context budget free for reasoning.
fn build_analysis_state(store: &ContextStore) -> String {
    let mut parts = Vec::new();

    // Variable metadata
    let vars = store.variable_metadata();
    if !vars.is_empty() {
        let var_lines: Vec<String> = vars
            .iter()
            .map(|(name, len, preview)| {
                format!("  {} ({} chars, preview: \"{}\")", name, len, preview)
            })
            .collect();
        parts.push(format!("Variables:\n{}", var_lines.join("\n")));
    }

    // Memory findings
    let mem = store.mem_entries();
    if !mem.is_empty() {
        let mut sorted_keys: Vec<&String> = mem.keys().collect();
        sorted_keys.sort();
        let mem_lines: Vec<String> = sorted_keys
            .iter()
            .map(|k| format!("  {}: {}", k, mem[*k]))
            .collect();
        parts.push(format!("Findings so far:\n{}", mem_lines.join("\n")));
    }

    parts.join("\n")
}

/// Run analysis using fresh single-turn LLM calls with scratch pad persistence.
///
/// Each round: build compact state from ContextStore → fresh LLM call →
/// execute tools → update scratch pad. The model gets its FULL context
/// budget every round (~300 tok input, ~3700 free).
async fn analyze_via_scratch_pad(
    config: &ToolRunnerConfig,
    context_store: &mut ContextStore,
    tool_defs: &[Value],
    allowed_tools: &std::collections::HashSet<&str>,
    tools: &ToolRegistry,
    task_context: &str,
    all_results: &mut Vec<(String, String, String)>,
    max_rounds: usize,
) -> Option<String> {
    let mut seen_calls: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut id_counter: usize = 1000; // offset to avoid collision with iteration-0 IDs

    // Cost tracking: accumulate across rounds, stop if budget exceeded.
    let mut cost_spent: f64 = 0.0;
    let cost_limit = config.budget.as_ref().map_or(0.0, |b| b.cost_limit);
    let prices = config.budget.as_ref().and_then(|b| b.prices.clone());

    let system_prompt = "You analyze stored data. Use micro-tools to inspect, mem_store to save findings.\nWhen you have enough information, write your final summary. STOP.\nPrevious findings are listed below — build on them, don't repeat work.\n/nothink";

    for round in 0..max_rounds {
        // Check cancellation.
        if config
            .cancellation_token
            .as_ref()
            .map_or(false, |t| t.is_cancelled())
        {
            return Some("Analysis cancelled.".to_string());
        }

        // Check cost budget.
        if cost_limit > 0.0 && cost_spent >= cost_limit {
            debug!(
                "Cost budget exhausted (${:.6} / ${:.6}) after {} rounds",
                cost_spent, cost_limit, round
            );
            break;
        }

        // Build compact state from ContextStore memory + variable metadata.
        let state = build_analysis_state(context_store);

        // Fresh messages each round — no growing conversation.
        let messages = vec![
            json!({ "role": "system", "content": system_prompt }),
            json!({
                "role": "user",
                "content": format!("Task: {}\n\n{}", task_context, state)
            }),
        ];

        let tool_defs_opt: Option<&[Value]> = if tool_defs.is_empty() {
            None
        } else {
            Some(tool_defs)
        };

        // Single LLM call with full context budget.
        let response = match config
            .provider
            .chat(
                &messages,
                tool_defs_opt,
                Some(&config.model),
                config.max_tokens,
                0.3,
                None,
                None,
            )
            .await
        {
            Ok(r) => r,
            Err(e) => {
                warn!(
                    "Scratch pad analysis LLM call failed (round {}): {}",
                    round, e
                );
                break; // Fall through to fallback
            }
        };

        // Track cost from this LLM call.
        if let Some(ref p) = prices {
            let prompt_tokens = response.usage.get("prompt_tokens").copied().unwrap_or(0);
            let completion_tokens = response
                .usage
                .get("completion_tokens")
                .copied()
                .unwrap_or(0);
            let call_cost = p.cost_of(&config.model, prompt_tokens, completion_tokens);
            cost_spent += call_cost;
            debug!(
                "Analysis round {} cost: ${:.6} (total: ${:.6} / limit: ${:.6})",
                round, call_cost, cost_spent, cost_limit
            );
        }

        if response.has_tool_calls() {
            // Pre-check: if ALL tool calls are duplicates or blocked, break early
            // to avoid burning rounds where the model keeps requesting the same calls.
            let all_skipped = response.tool_calls.iter().all(|tc| {
                if !allowed_tools.contains(tc.name.as_str()) {
                    return true;
                }
                let call_key = normalize_call_key(&tc.name, &tc.arguments);
                seen_calls.contains(&call_key)
            });
            if all_skipped {
                debug!(
                    "Scratch pad: all {} tool calls in round {} are duplicates/blocked — breaking",
                    response.tool_calls.len(),
                    round
                );
                break;
            }

            // Execute tool calls against ContextStore.
            for tc in &response.tool_calls {
                // Block disallowed tools.
                if !allowed_tools.contains(tc.name.as_str()) {
                    warn!("Scratch pad: blocked disallowed tool '{}'", tc.name);
                    continue;
                }

                // Loop detection across rounds.
                let call_key = normalize_call_key(&tc.name, &tc.arguments);
                if seen_calls.contains(&call_key) {
                    debug!("Scratch pad: duplicate call {} — skipping", call_key);
                    // Tell the model this call was already executed so it stops
                    // regenerating identical requests (saves tokens on local models).
                    let dedup_msg = format!(
                        "DUPLICATE: This exact {} call was already executed. Use the existing result from memory.",
                        tc.name
                    );
                    context_store.mem_store(&format!("dedup_{}", tc.name), dedup_msg);
                    continue;
                }
                seen_calls.insert(call_key);

                if tc.name == "ctx_summarize" {
                    let variable = tc
                        .arguments
                        .get("variable")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let instruction = tc
                        .arguments
                        .get("instruction")
                        .and_then(|v| v.as_str())
                        .unwrap_or("Summarize");
                    let result = context_store::execute_ctx_summarize(
                        context_store,
                        variable,
                        instruction,
                        &config.provider,
                        &config.model,
                        config.depth,
                        config.max_tokens,
                    )
                    .await;
                    // Store summary as new variable + save finding in memory.
                    let (var_name, _) = context_store.store(result.clone());
                    context_store.mem_store(&format!("summary_{}", var_name), result);
                } else if context_store::is_micro_tool(&tc.name) {
                    let result =
                        context_store::execute_micro_tool(context_store, &tc.name, &tc.arguments);
                    debug!("Scratch pad micro-tool {}: {} chars", tc.name, result.len());
                    // Auto-persist inspection results so the model sees them in
                    // subsequent rounds via build_analysis_state(). Without this,
                    // fresh-message rounds lose ctx_slice/ctx_grep results and the
                    // model re-requests identical calls until dedup blocks them.
                    match tc.name.as_str() {
                        "ctx_slice" | "ctx_grep" | "ctx_length" => {
                            let var = tc
                                .arguments
                                .get("variable")
                                .and_then(|v| v.as_str())
                                .unwrap_or("?");
                            let key = match tc.name.as_str() {
                                "ctx_slice" => {
                                    let s = tc
                                        .arguments
                                        .get("start")
                                        .and_then(|v| v.as_u64())
                                        .unwrap_or(0);
                                    let e = tc
                                        .arguments
                                        .get("end")
                                        .and_then(|v| v.as_u64())
                                        .unwrap_or(0);
                                    format!("slice_{}_{}-{}", var, s, e)
                                }
                                "ctx_grep" => {
                                    let pat = tc
                                        .arguments
                                        .get("pattern")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("?");
                                    format!("grep_{}_{}", var, pat)
                                }
                                _ => format!("len_{}", var),
                            };
                            context_store.mem_store(&key, result);
                        }
                        _ => {} // mem_store/mem_recall/set_phase handle persistence internally
                    }
                } else if worker_tools::is_worker_tool(&tc.name) {
                    let result =
                        worker_tools::execute_worker_tool(&tc.name, &tc.arguments, None).await;
                    let (_, _metadata) = context_store.store(result.clone());
                    let original_id = format!("sp{:07}", id_counter);
                    id_counter += 1;
                    all_results.push((original_id, tc.name.clone(), result));
                } else {
                    // Real tool — execute with retry on transient errors.
                    debug!("Scratch pad executing real tool: {}", tc.name);
                    let result = execute_with_retry(
                        tools,
                        &tc.name,
                        tc.arguments.clone(),
                        config.cancellation_token.as_ref(),
                        TOOL_MAX_RETRIES,
                    )
                    .await;
                    let (_, _metadata) = context_store.store(result.data.clone());
                    let original_id = format!("sp{:07}", id_counter);
                    id_counter += 1;
                    all_results.push((original_id, tc.name.clone(), result.data));
                }
            }
            // Continue to next round — fresh call with updated state.
        } else {
            // Model produced text — that's our summary.
            if let Some(text) = response.content {
                if !text.is_empty() {
                    debug!("Scratch pad analysis complete after {} rounds", round + 1);
                    return Some(text);
                }
            }
        }
    }

    // Phase 4: Graceful fallback — synthesize from accumulated memory findings.
    let keys = context_store.mem_keys();
    if !keys.is_empty() {
        let findings: Vec<String> = keys
            .iter()
            .filter_map(|k| {
                let v = context_store.mem_recall(k);
                if v.starts_with("Key '") {
                    None
                } else {
                    Some(format!("{}: {}", k, v))
                }
            })
            .collect();
        if !findings.is_empty() {
            debug!(
                "Scratch pad fallback: synthesizing from {} memory findings",
                findings.len()
            );
            return Some(findings.join("\n"));
        }
    }
    None
}

/// Pick a bounded scratch-pad analysis round budget from config + model family.
///
/// Small local models should spend fewer rounds "thinking about tool outputs"
/// and move on faster to actionable tool execution/results.
fn scratch_pad_round_budget(config: &ToolRunnerConfig) -> usize {
    let caps =
        crate::agent::model_capabilities::lookup(&config.model, &std::collections::HashMap::new());
    let rounds = config.max_iterations.clamp(1, 10) as usize;
    rounds.min(caps.scratch_pad_rounds).max(1)
}

/// Process tool calls from an LLM response: build assistant message,
/// execute each tool, and add results to messages.
///
/// This is the common core shared by subagent and pipeline.
/// Returns `true` if tool calls were processed, `false` if none were present.
///
/// **Protocol compliance** is handled by the caller: before the next LLM call,
/// pass `messages` through `protocol::render_to_wire()` to get the correct wire
/// format. Do NOT call `repair_for_local` here — rendering at call time is the
/// correct approach.
pub async fn process_tool_response(
    response: &crate::providers::base::LLMResponse,
    messages: &mut Vec<Value>,
    tools: &ToolRegistry,
) -> bool {
    if !response.has_tool_calls() {
        return false;
    }

    let tc_json: Vec<Value> = response
        .tool_calls
        .iter()
        .map(|tc| tc.to_openai_json())
        .collect();

    ContextBuilder::add_assistant_message(messages, response.content.as_deref(), Some(&tc_json));

    for tc in &response.tool_calls {
        let result = tools.execute(&tc.name, tc.arguments.clone()).await;
        ContextBuilder::add_tool_result(messages, &tc.id, &tc.name, &result.data);
    }

    true
}

/// Run a delegated tool execution loop.
///
/// Executes `initial_tool_calls` using the `tools` registry, then asks the
/// cheap model if more tool calls are needed. Repeats until the model
/// produces a text-only response or `max_iterations` is reached.
pub async fn run_tool_loop(
    config: &ToolRunnerConfig,
    initial_tool_calls: &[ToolCallRequest],
    tools: &ToolRegistry,
    system_context: &str,
) -> ToolRunResult {
    info!(
        role = "delegation",
        model = %config.model,
        tools = initial_tool_calls.len(),
        "tool_delegation_start"
    );
    let mut all_results: Vec<(String, String, String)> = Vec::new();
    let mut iterations_used: u32 = 0;
    let mut id_counter: usize = 0;
    // Track tool calls we've already executed to detect loops.
    let mut seen_calls: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut seen_results: std::collections::HashSet<u64> = std::collections::HashSet::new();

    // RLM ContextStore: stores full tool outputs as named variables.
    // The delegation model sees metadata for large results and can use
    // micro-tools (ctx_slice, ctx_grep, ctx_length) to inspect on demand.
    let mut context_store = ContextStore::new();

    // SAFETY: restrict the delegation model to ONLY the tools the main model
    // requested. This prevents prompt injection in tool results (web pages,
    // file contents) from tricking the small model into calling exec, write_file,
    // or other destructive tools that weren't part of the original request.
    let mut allowed_tools: std::collections::HashSet<&str> = initial_tool_calls
        .iter()
        .map(|tc| tc.name.as_str())
        .collect();
    // Micro-tools are always available to the delegation model.
    for name in context_store::MICRO_TOOLS {
        allowed_tools.insert(name);
    }
    // Worker tools are always available to the delegation model.
    for name in worker_tools::WORKER_TOOLS {
        allowed_tools.insert(name);
    }
    // Delegate tool is available if budget allows.
    if config.budget.as_ref().map_or(false, |b| b.can_delegate()) {
        allowed_tools.insert(DELEGATE_TOOL);
    }

    // Build a mini message history for the cheap model.
    // Keep the system prompt compact — the delegation model has limited context.
    let system_msg = json!({
        "role": "system",
        "content": "You are a tool result analyst. You receive tool outputs as named variables.\n\nRULES:\n1. If results are clear and complete, write a concise summary with specific data points. STOP.\n2. If results are too large, use ctx_grep(variable, pattern) to find relevant sections.\n3. If grep isn't enough, use ctx_slice(variable, start, end) to read specific ranges.\n4. If you need a focused summary of a large result, use ctx_summarize(variable, instruction).\n5. NEVER re-execute the same tool with identical arguments.\n6. NEVER execute tools not in your allowed list.\n7. Prefer grep → slice → summarize (cheapest first).\n\nOUTPUT FORMAT:\n- Lead with the answer/finding\n- Include specific numbers, paths, error messages verbatim\n- If multiple tool results, organize by tool call\n/nothink"
    });
    let mut messages: Vec<Value> = vec![system_msg];

    // Anthropic requires conversations to start with a user message.
    // Pass the task context (truncated to save delegation model context).
    let task_context: String = system_context.chars().take(500).collect();
    messages.push(json!({
        "role": "user",
        "content": task_context
    }));

    // Get tool definitions — filtered to ONLY the tools the main model requested.
    // The delegation model must not discover tools it wasn't asked to use.
    let all_tool_defs = tools.get_definitions();
    let mut tool_defs: Vec<Value> = all_tool_defs
        .into_iter()
        .filter(|def| {
            def.pointer("/function/name")
                .and_then(|v| v.as_str())
                .map(|name| allowed_tools.contains(name))
                .unwrap_or(false)
        })
        .collect();
    // Append micro-tool definitions so the delegation model can inspect variables.
    tool_defs.extend(context_store::micro_tool_definitions());
    tool_defs.extend(worker_tools::worker_tool_definitions());

    // Add delegate tool definition if budget allows delegation.
    if config.budget.as_ref().map_or(false, |b| b.can_delegate()) {
        tool_defs.push(json!({
            "type": "function",
            "function": {
                "name": "delegate",
                "description": "Spawn a child worker for a sub-task. The child inherits your model but gets a reduced budget. Returns the child's result.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {"type": "string", "description": "What the child worker should accomplish"},
                        "tools": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tool names the child can use (e.g. ['read_file', 'ctx_grep', 'python_eval'])"
                        },
                        "context": {"type": "string", "description": "Optional data to seed the child's context"}
                    },
                    "required": ["task"]
                }
            }
        }));
    }

    // Execute the initial tool calls from the main model.
    let mut pending_calls: Vec<ToolCallRequest> = initial_tool_calls.to_vec();

    // Deduplicate within the initial batch itself.
    let mut batch_seen = std::collections::HashSet::new();
    pending_calls.retain(|tc| {
        let key = normalize_call_key(&tc.name, &tc.arguments);
        batch_seen.insert(key)
    });
    if pending_calls.len() < initial_tool_calls.len() {
        tracing::warn!(
            before = initial_tool_calls.len(),
            after = pending_calls.len(),
            "Deduplicated identical tool calls in initial batch"
        );
    }

    // Seed seen_calls with initial calls so the model can't re-request them.
    for tc in &pending_calls {
        let call_key = normalize_call_key(&tc.name, &tc.arguments);
        seen_calls.insert(call_key);
    }

    for iteration in 0..config.max_iterations {
        iterations_used = iteration + 1;

        // Check cancellation before each iteration.
        if config
            .cancellation_token
            .as_ref()
            .map_or(false, |t| t.is_cancelled())
        {
            return ToolRunResult {
                tool_results: all_results,
                summary: Some("Tool execution cancelled.".to_string()),
                iterations_used,
                error: None,
            };
        }

        if pending_calls.is_empty() {
            break;
        }

        // Mistral/Ministral models require tool call IDs to be exactly
        // 9 alphanumeric characters. Normalize IDs for the internal
        // conversation while tracking the original↔normalized mapping.
        let mut id_map: HashMap<String, String> = HashMap::new();
        for tc in &mut pending_calls {
            let normalized = normalize_tool_call_id(id_counter);
            id_counter += 1;
            id_map.insert(normalized.clone(), tc.id.clone());
            tc.id = normalized;
        }

        // Build assistant message with tool_calls (using normalized IDs).
        let tc_json: Vec<Value> = pending_calls.iter().map(|tc| tc.to_openai_json()).collect();
        ContextBuilder::add_assistant_message(&mut messages, None, Some(&tc_json));

        // Execute each tool call with ContextStore-aware dispatch.
        // Micro-tools (ctx_slice, ctx_grep, ctx_length) execute against the
        // ContextStore and are internal to the delegation conversation.
        // Real tools execute via the registry; large results are stored as
        // variables and the delegation model sees metadata only.
        for tc in &pending_calls {
            if tc.name == "ctx_summarize" {
                // Async micro-tool: runs a sub-loop with the provider.
                let variable = tc
                    .arguments
                    .get("variable")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let instruction = tc
                    .arguments
                    .get("instruction")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Summarize");
                debug!(
                    "ctx_summarize: var={}, instruction={}, depth={}",
                    variable, instruction, config.depth
                );
                let result = context_store::execute_ctx_summarize(
                    &context_store,
                    variable,
                    instruction,
                    &config.provider,
                    &config.model,
                    config.depth,
                    config.max_tokens,
                )
                .await;
                // Store the summary as a new variable for subsequent micro-tool access.
                let (_, summary_metadata) = context_store.store(result.clone());
                let _ = summary_metadata; // metadata available if needed later
                ContextBuilder::add_tool_result(&mut messages, &tc.id, &tc.name, &result);
                // NOT added to all_results — ctx_summarize is internal to delegation.
            } else if tc.name == DELEGATE_TOOL {
                // Recursive delegation: spawn a child worker.
                let task = tc
                    .arguments
                    .get("task")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let context = tc
                    .arguments
                    .get("context")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                let child_budget = match config.budget.as_ref().and_then(|b| b.child()) {
                    Some(b) => b,
                    None => {
                        let result = "Error: delegation depth limit reached.".to_string();
                        ContextBuilder::add_tool_result(&mut messages, &tc.id, &tc.name, &result);
                        let original_id =
                            id_map.get(&tc.id).cloned().unwrap_or_else(|| tc.id.clone());
                        all_results.push((original_id, tc.name.clone(), result));
                        continue;
                    }
                };

                // Determine child tools — use requested subset or all allowed tools.
                let child_tool_names: Vec<String> = tc
                    .arguments
                    .get("tools")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(|s| s.to_string()))
                            .collect()
                    })
                    .unwrap_or_else(Vec::new);

                debug!(
                    "Delegate: task='{}', depth={}, budget={}, tools={:?}",
                    task, child_budget.current_depth, child_budget.max_iterations, child_tool_names
                );

                // Build child config — inherits provider/model from parent.
                let child_config = ToolRunnerConfig {
                    provider: config.provider.clone(),
                    model: config.model.clone(),
                    max_iterations: child_budget.max_iterations,
                    max_tokens: config.max_tokens,

                    max_tool_result_chars: config.max_tool_result_chars,
                    short_circuit_chars: config.short_circuit_chars,
                    depth: config.depth + 1,
                    cancellation_token: config.cancellation_token.clone(),
                    verbatim: false,
                    budget: Some(child_budget),
                };

                // Build child system prompt.
                let child_system = format!(
                    "You are a worker agent. Complete this task:\n\n{}\n\n{}",
                    task,
                    if context.is_empty() {
                        String::new()
                    } else {
                        format!("Context:\n{}", context)
                    }
                );

                // Ask the child model what tools to use.
                let child_msgs = vec![
                    json!({"role": "system", "content": child_system}),
                    json!({"role": "user", "content": "Begin working on the task. Use the available tools to complete it."}),
                ];

                // Get child tool definitions.
                let mut child_tool_defs: Vec<Value> = if child_tool_names.is_empty() {
                    // No specific tools requested — give same tools as parent (minus delegate to prevent deep recursion issues).
                    tool_defs
                        .iter()
                        .filter(|d| {
                            d.pointer("/function/name").and_then(|v| v.as_str())
                                != Some(DELEGATE_TOOL)
                                || child_config
                                    .budget
                                    .as_ref()
                                    .map_or(false, |b| b.can_delegate())
                        })
                        .cloned()
                        .collect()
                } else {
                    // Filter to only requested tools + always include micro-tools and worker tools.
                    let child_allowed: std::collections::HashSet<&str> =
                        child_tool_names.iter().map(|s| s.as_str()).collect();
                    tool_defs
                        .iter()
                        .filter(|d| {
                            let name = d
                                .pointer("/function/name")
                                .and_then(|v| v.as_str())
                                .unwrap_or("");
                            child_allowed.contains(name)
                                || context_store::is_micro_tool(name)
                                || worker_tools::is_worker_tool(name)
                        })
                        .cloned()
                        .collect()
                };

                // Always add micro-tool and worker tool defs if not already present.
                let existing_names: std::collections::HashSet<String> = child_tool_defs
                    .iter()
                    .filter_map(|d| {
                        d.pointer("/function/name")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string())
                    })
                    .collect();
                for def in context_store::micro_tool_definitions() {
                    let name = def
                        .pointer("/function/name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    if !existing_names.contains(name) {
                        child_tool_defs.push(def);
                    }
                }
                for def in worker_tools::worker_tool_definitions() {
                    let name = def
                        .pointer("/function/name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    if !existing_names.contains(name) {
                        child_tool_defs.push(def);
                    }
                }

                let child_tool_defs_opt: Option<&[Value]> = if child_tool_defs.is_empty() {
                    None
                } else {
                    Some(&child_tool_defs)
                };

                // Get the child's first response (with tool calls).
                let child_response = match config
                    .provider
                    .chat(
                        &child_msgs,
                        child_tool_defs_opt,
                        Some(&config.model),
                        config.max_tokens,
                        0.3,
                        None,
                        None,
                    )
                    .await
                {
                    Ok(r) => r,
                    Err(e) => {
                        let result = format!("Delegate error: {}", e);
                        ContextBuilder::add_tool_result(&mut messages, &tc.id, &tc.name, &result);
                        let original_id =
                            id_map.get(&tc.id).cloned().unwrap_or_else(|| tc.id.clone());
                        all_results.push((original_id, tc.name.clone(), result));
                        continue;
                    }
                };

                let result = if child_response.has_tool_calls() {
                    // Child wants to use tools — run the tool loop.
                    let child_result = Box::pin(run_tool_loop(
                        &child_config,
                        &child_response.tool_calls,
                        tools,
                        &child_system,
                    ))
                    .await;
                    // Return the child's summary, or concatenated results if no summary.
                    child_result.summary.unwrap_or_else(|| {
                        child_result
                            .tool_results
                            .iter()
                            .map(|(_, name, data)| {
                                format!("[{}]: {}", name, &data[..data.len().min(500)])
                            })
                            .collect::<Vec<_>>()
                            .join("\n")
                    })
                } else {
                    // Child produced a text response — return it directly.
                    child_response
                        .content
                        .unwrap_or_else(|| "No result from delegate.".to_string())
                };

                // Store delegate result like any other tool.
                let (_, metadata) = context_store.store(result.clone());
                let delegation_data = if result.len() > config.max_tool_result_chars {
                    metadata
                } else {
                    result.clone()
                };
                ContextBuilder::add_tool_result(&mut messages, &tc.id, &tc.name, &delegation_data);
                let original_id = id_map.get(&tc.id).cloned().unwrap_or_else(|| tc.id.clone());
                all_results.push((original_id, tc.name.clone(), result));
            } else if worker_tools::is_worker_tool(&tc.name) {
                // Async worker tool: runs a command/script and returns result.
                debug!("Worker tool: {} (id: {})", tc.name, tc.id);
                let result = worker_tools::execute_worker_tool(&tc.name, &tc.arguments, None).await;
                // Store result in ContextStore for subsequent micro-tool access.
                let (_, metadata) = context_store.store(result.clone());
                let delegation_data = if result.len() > config.max_tool_result_chars {
                    metadata
                } else {
                    result.clone()
                };
                ContextBuilder::add_tool_result(&mut messages, &tc.id, &tc.name, &delegation_data);
                let original_id = id_map.get(&tc.id).cloned().unwrap_or_else(|| tc.id.clone());
                all_results.push((original_id, tc.name.clone(), result));

                // Track result hashes for loop detection.
                use std::hash::{Hash, Hasher};
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                tc.name.hash(&mut hasher);
                all_results.last().unwrap().2.hash(&mut hasher);
                let result_hash = hasher.finish();
                if !seen_results.insert(result_hash) {
                    warn!(
                        "Worker tool '{}' produced identical results — likely loop",
                        tc.name
                    );
                }
            } else if context_store::is_micro_tool(&tc.name) {
                // Sync micro-tool: execute against ContextStore (internal to delegation).
                debug!("Micro-tool: {} (id: {})", tc.name, tc.id);
                let result =
                    context_store::execute_micro_tool(&mut context_store, &tc.name, &tc.arguments);
                ContextBuilder::add_tool_result(&mut messages, &tc.id, &tc.name, &result);
                // NOT added to all_results — micro-tools are internal.
            } else {
                // Real tool: execute with retry on transient errors.
                debug!("Tool runner executing: {} (id: {})", tc.name, tc.id);
                let result = execute_with_retry(
                    tools,
                    &tc.name,
                    tc.arguments.clone(),
                    config.cancellation_token.as_ref(),
                    TOOL_MAX_RETRIES,
                )
                .await;
                // For web_fetch/web_search: unwrap the JSON envelope so the model
                // sees clean article text rather than a JSON metadata summary.
                let raw_data = if tc.name == "web_fetch" || tc.name == "web_search" {
                    crate::agent::tools::web::extract_web_content(&result.data)
                } else {
                    result.data.clone()
                };
                let stripped = strip_tool_output(&raw_data);
                let (_, metadata) = context_store.store(stripped.clone());
                let delegation_data = if stripped.len() > config.max_tool_result_chars {
                    // Large result: model sees metadata, uses micro-tools to dig in.
                    metadata
                } else {
                    // Small result: model sees full result directly.
                    stripped.clone()
                };
                ContextBuilder::add_tool_result(&mut messages, &tc.id, &tc.name, &delegation_data);
                let original_id = id_map.get(&tc.id).cloned().unwrap_or_else(|| tc.id.clone());
                all_results.push((original_id, tc.name.clone(), stripped));

                // Track result hashes for loop detection.
                use std::hash::{Hash, Hasher};
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                tc.name.hash(&mut hasher);
                result.data.hash(&mut hasher);
                let result_hash = hasher.finish();
                if !seen_results.insert(result_hash) {
                    warn!(
                        "Tool '{}' produced identical results — likely loop",
                        tc.name
                    );
                }
            }
        }

        // Short-circuit: if this is the first iteration and ALL results
        // are short, skip the delegation LLM call entirely.
        // Trivial outputs (echo, simple commands) don't need summarization —
        // the main model can interpret them directly. This also avoids the
        // loop problem where small models don't know what to do with
        // one-line results. Set short_circuit_chars to 0 to disable.
        // Verbatim mode: skip delegation entirely, return raw results.
        if config.verbatim && iteration == 0 {
            debug!("Verbatim mode — skipping delegation LLM, returning raw results");
            return ToolRunResult {
                tool_results: all_results,
                summary: None,
                iterations_used,
                error: None,
            };
        }

        if config.short_circuit_chars > 0 && iteration == 0 {
            let threshold = config.short_circuit_chars;
            let all_short = all_results
                .iter()
                .all(|(_, _, data)| data.len() < threshold);
            // exec/web_search/web_fetch results need interpretation even when short
            // (e.g. a 50-char error message should be analyzed, not passed raw).
            let needs_interpretation = all_results
                .iter()
                .any(|(_, name, _)| matches!(name.as_str(), "exec" | "web_search" | "web_fetch"));
            if all_short && !needs_interpretation {
                debug!(
                    "All {} tool results are short (< {} chars) — skipping delegation LLM, returning raw results",
                    all_results.len(), threshold
                );
                return ToolRunResult {
                    tool_results: all_results,
                    summary: None,
                    iterations_used,
                    error: None,
                };
            }
        }

        // After executing initial tool calls (iteration 0), hand off to
        // scratch pad analysis. Each analysis round is a fresh single-turn
        // LLM call with the ContextStore's memory acting as persistent state.
        // This replaces the growing-message loop that caused context saturation.
        if iteration == 0 {
            let scratch_rounds = scratch_pad_round_budget(config);
            debug!(
                "Handing off to scratch pad analysis (up to {} rounds, model={})",
                scratch_rounds, config.model
            );
            let summary = analyze_via_scratch_pad(
                config,
                &mut context_store,
                &tool_defs,
                &allowed_tools,
                tools,
                &task_context,
                &mut all_results,
                scratch_rounds,
            )
            .await;
            return ToolRunResult {
                tool_results: all_results,
                summary,
                iterations_used: 1,
                error: None,
            };
        }

        // Fallback: iterations > 0 should not be reached (scratch pad
        // takes over after iteration 0), but kept for safety.
        break;
    }

    // Ran out of iterations or broke out of loop.
    ToolRunResult {
        tool_results: all_results,
        summary: None,
        iterations_used,
        error: None,
    }
}

/// Format tool run results for injection into the main LLM context.
///
/// If a `ContentGate` is provided, each result goes through the gate for
/// budget-aware sizing. Otherwise falls back to `max_result_chars` truncation.
pub fn format_results_for_context(
    result: &ToolRunResult,
    max_result_chars: usize,
    mut gate: Option<&mut crate::agent::context_gate::ContentGate>,
) -> String {
    let mut parts: Vec<String> = Vec::new();

    for (_, tool_name, data) in &result.tool_results {
        let sized = if let Some(gate) = gate.as_mut() {
            gate.admit_simple(data).into_text()
        } else {
            // Legacy fallback: hard char truncation.
            if data.len() > max_result_chars {
                let end = crate::utils::helpers::floor_char_boundary(data, max_result_chars);
                format!("{}… ({} chars total)", &data[..end], data.len())
            } else {
                data.clone()
            }
        };
        parts.push(format!("[{}]: {}", tool_name, sized));
    }

    if let Some(ref summary) = result.summary {
        parts.push(format!("\nSummary: {}", summary));
    }

    parts.join("\n")
}

/// Generate a Mistral-compatible tool call ID (exactly 9 alphanumeric chars).
///
/// Mistral/Ministral Jinja templates validate that tool call IDs are exactly
/// 9 alphanumeric characters. This function generates deterministic IDs from
/// a counter for use in the delegation model's internal conversation.
fn normalize_tool_call_id(counter: usize) -> String {
    format!("tc{:07}", counter)
}

/// Execute a tool with automatic retry on transient (retryable) errors.
///
/// Retries up to `max_retries` times with exponential backoff (500ms, 1s, 2s, …).
/// Respects the cancellation token between retries.
async fn execute_with_retry(
    tools: &ToolRegistry,
    name: &str,
    arguments: HashMap<String, serde_json::Value>,
    cancel: Option<&tokio_util::sync::CancellationToken>,
    max_retries: u32,
) -> crate::agent::tools::base::ToolExecutionResult {
    use crate::agent::tools::base::{ToolExecutionContext, ToolExecutionResult};

    let mut attempts = 0u32;
    loop {
        let result = if let Some(token) = cancel {
            let (event_tx, _event_rx) = tokio::sync::mpsc::unbounded_channel();
            let ctx = ToolExecutionContext {
                event_tx,
                cancellation_token: token.child_token(),
                tool_call_id: String::new(),
            };
            tools
                .execute_with_context(name, arguments.clone(), &ctx)
                .await
        } else {
            tools.execute(name, arguments.clone()).await
        };

        attempts += 1;

        if !result.is_retryable() || attempts > max_retries {
            return result;
        }

        // Check cancellation before sleeping.
        if let Some(token) = cancel {
            if token.is_cancelled() {
                return ToolExecutionResult::failure("Cancelled during retry".into());
            }
        }

        let backoff = std::time::Duration::from_millis(500 * (1 << (attempts - 1)));
        warn!(
            "Tool '{}' returned retryable error (attempt {}/{}), retrying in {:?}: {}",
            name,
            attempts,
            max_retries + 1,
            backoff,
            result.data
        );

        // Sleep with cancellation awareness.
        if let Some(token) = cancel {
            tokio::select! {
                _ = tokio::time::sleep(backoff) => {}
                _ = token.cancelled() => {
                    return ToolExecutionResult::failure("Cancelled during retry backoff".into());
                }
            }
        } else {
            tokio::time::sleep(backoff).await;
        }
    }
}

/// Default maximum retries for transient tool errors.
const TOOL_MAX_RETRIES: u32 = 3;

#[cfg(test)]
#[path = "tool_runner_tests.rs"]
mod tests;
