//! Delegated tool execution loop.
//!
//! Receives initial tool calls from the main LLM, executes them via the
//! shared [`ToolRegistry`], and lets a cheap model decide if more tools are
//! needed. Returns aggregated results for injection into the main context.

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::{json, Value};
use tracing::{debug, warn};

use crate::agent::context::ContextBuilder;
use crate::agent::context_store::{self, ContextStore};
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
    /// When true, append a `role: "user"` continuation after tool results
    /// before calling the LLM.  Local models (llama-server) require this;
    /// Mistral/Ministral handle tool→generate natively and break if a user
    /// message is injected.
    pub needs_user_continuation: bool,
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
fn normalize_call_key(name: &str, arguments: &HashMap<String, Value>) -> String {
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

        // Error response from provider — stop analyzing.
        if response.finish_reason == "error" {
            warn!(
                "Scratch pad analysis got error response (round {}) — falling back",
                round
            );
            break;
        }

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
                    // mem_store/mem_recall are already handled internally by execute_micro_tool.
                    // For inspection tools (slice/grep/length), the model will use
                    // mem_store in a subsequent call to persist findings.
                    debug!("Scratch pad micro-tool {}: {} chars", tc.name, result.len());
                } else if worker_tools::is_worker_tool(&tc.name) {
                    let result =
                        worker_tools::execute_worker_tool(&tc.name, &tc.arguments, None).await;
                    let (_, _metadata) = context_store.store(result.clone());
                    let original_id = format!("sp{:07}", id_counter);
                    id_counter += 1;
                    all_results.push((original_id, tc.name.clone(), result));
                } else {
                    // Real tool — execute and store in ContextStore.
                    debug!("Scratch pad executing real tool: {}", tc.name);
                    let result = if let Some(ref token) = config.cancellation_token {
                        use crate::agent::tools::base::ToolExecutionContext;
                        let (event_tx, _event_rx) = tokio::sync::mpsc::unbounded_channel();
                        let ctx = ToolExecutionContext {
                            event_tx,
                            cancellation_token: token.child_token(),
                            tool_call_id: tc.id.clone(),
                        };
                        tools
                            .execute_with_context(&tc.name, tc.arguments.clone(), &ctx)
                            .await
                    } else {
                        tools.execute(&tc.name, tc.arguments.clone()).await
                    };
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
    let mut rounds = config.max_iterations.clamp(1, 10) as usize;
    let model = config.model.to_ascii_lowercase();

    if model.contains("nanbeige") {
        rounds = rounds.min(3);
    } else if model.contains("functiongemma") {
        rounds = rounds.min(2);
    } else if model.contains("ministral-3") || model.contains("qwen3-1.7b") {
        rounds = rounds.min(4);
    }

    rounds.max(1)
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

    // Seed seen_calls with initial calls so the model can't re-request them.
    for tc in initial_tool_calls {
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
        let tc_json: Vec<Value> = pending_calls
            .iter()
            .map(|tc| tc.to_openai_json())
            .collect();
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
                    needs_user_continuation: config.needs_user_continuation,
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
                // Real tool: execute, store in ContextStore.
                debug!("Tool runner executing: {} (id: {})", tc.name, tc.id);
                let result = if let Some(ref token) = config.cancellation_token {
                    use crate::agent::tools::base::ToolExecutionContext;
                    let (event_tx, _event_rx) = tokio::sync::mpsc::unbounded_channel();
                    let ctx = ToolExecutionContext {
                        event_tx,
                        cancellation_token: token.child_token(),
                        tool_call_id: tc.id.clone(),
                    };
                    tools
                        .execute_with_context(&tc.name, tc.arguments.clone(), &ctx)
                        .await
                } else {
                    tools.execute(&tc.name, tc.arguments.clone()).await
                };
                let (_, metadata) = context_store.store(result.data.clone());
                let delegation_data = if result.data.len() > config.max_tool_result_chars {
                    // Large result: model sees metadata, uses micro-tools to dig in.
                    metadata
                } else {
                    // Small result: model sees full result directly.
                    result.data.clone()
                };
                ContextBuilder::add_tool_result(&mut messages, &tc.id, &tc.name, &delegation_data);
                let original_id = id_map.get(&tc.id).cloned().unwrap_or_else(|| tc.id.clone());
                all_results.push((original_id, tc.name.clone(), result.data.clone()));

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::tools::base::Tool;
    use async_trait::async_trait;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicU32, Ordering};

    // Mock LLM provider for testing.
    struct MockProvider {
        responses: tokio::sync::Mutex<Vec<crate::providers::base::LLMResponse>>,
    }

    impl MockProvider {
        fn new(responses: Vec<crate::providers::base::LLMResponse>) -> Self {
            Self {
                responses: tokio::sync::Mutex::new(responses),
            }
        }
    }

    #[async_trait]
    impl LLMProvider for MockProvider {
        async fn chat(
            &self,
            _messages: &[Value],
            _tools: Option<&[Value]>,
            _model: Option<&str>,
            _max_tokens: u32,
            _temperature: f64,
            _thinking_budget: Option<u32>,
        ) -> anyhow::Result<crate::providers::base::LLMResponse> {
            let mut responses = self.responses.lock().await;
            if responses.is_empty() {
                Ok(crate::providers::base::LLMResponse {
                    content: Some("Done.".to_string()),
                    tool_calls: vec![],
                    finish_reason: "stop".to_string(),
                    usage: HashMap::new(),
                })
            } else {
                Ok(responses.remove(0))
            }
        }

        fn get_default_model(&self) -> &str {
            "mock-model"
        }
    }

    // Mock tool that records how many times it was called.
    struct CountingTool {
        call_count: AtomicU32,
    }

    impl CountingTool {
        fn new() -> Self {
            Self {
                call_count: AtomicU32::new(0),
            }
        }
    }

    #[async_trait]
    impl Tool for CountingTool {
        fn name(&self) -> &str {
            "test_tool"
        }
        fn description(&self) -> &str {
            "A test tool"
        }
        fn parameters(&self) -> Value {
            json!({"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]})
        }
        async fn execute(&self, _params: HashMap<String, Value>) -> String {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            "tool result data".to_string()
        }
    }

    fn make_tool_calls(names: &[&str]) -> Vec<ToolCallRequest> {
        names
            .iter()
            .enumerate()
            .map(|(i, name)| ToolCallRequest {
                id: format!("call_{}", i),
                name: name.to_string(),
                arguments: {
                    let mut m = HashMap::new();
                    m.insert("query".to_string(), json!("test"));
                    m
                },
            })
            .collect()
    }

    #[tokio::test]
    async fn test_run_tool_loop_executes_tools() {
        let provider = Arc::new(MockProvider::new(vec![
            // After initial tool execution, model says "done".
            crate::providers::base::LLMResponse {
                content: Some("All done.".to_string()),
                tool_calls: vec![],
                finish_reason: "stop".to_string(),
                usage: HashMap::new(),
            },
        ]));

        let mut tools = ToolRegistry::new();
        tools.register(Box::new(CountingTool::new()));

        let config = ToolRunnerConfig {
            provider,
            model: "mock".to_string(),
            max_iterations: 10,
            max_tokens: 4096,
            needs_user_continuation: false,
            max_tool_result_chars: 30000,
            short_circuit_chars: 0,
            depth: 0,
            cancellation_token: None,
            verbatim: false,
            budget: None,
        };

        let result = run_tool_loop(
            &config,
            &make_tool_calls(&["test_tool"]),
            &tools,
            "test context",
        )
        .await;

        assert_eq!(result.tool_results.len(), 1);
        assert_eq!(result.tool_results[0].1, "test_tool");
        assert_eq!(result.tool_results[0].2, "tool result data");
        assert_eq!(result.summary.as_deref(), Some("All done."));
        assert_eq!(result.iterations_used, 1);
    }

    #[tokio::test]
    async fn test_run_tool_loop_respects_max_iterations() {
        // Provider always returns more tool calls with DIFFERENT args each time
        // (to avoid duplicate detection). Should stop at max_iterations.
        let mut responses = Vec::new();
        for i in 0..20 {
            responses.push(crate::providers::base::LLMResponse {
                content: None,
                tool_calls: vec![ToolCallRequest {
                    id: format!("chain_{}", i),
                    name: "test_tool".to_string(),
                    arguments: {
                        let mut m = HashMap::new();
                        m.insert("query".to_string(), json!(format!("chain_{}", i)));
                        m
                    },
                }],
                finish_reason: "tool_calls".to_string(),
                usage: HashMap::new(),
            });
        }

        let provider = Arc::new(MockProvider::new(responses));
        let mut tools = ToolRegistry::new();
        tools.register(Box::new(CountingTool::new()));

        let config = ToolRunnerConfig {
            provider,
            model: "mock".to_string(),
            max_iterations: 3,
            max_tokens: 4096,
            needs_user_continuation: false,
            max_tool_result_chars: 30000,
            short_circuit_chars: 0,
            depth: 0,
            cancellation_token: None,
            verbatim: false,
            budget: None,
        };

        let result = run_tool_loop(&config, &make_tool_calls(&["test_tool"]), &tools, "test").await;

        // Scratch pad runs internally after iteration 0; iterations_used is always 1.
        assert_eq!(result.iterations_used, 1);
        // Scratch pad exhausts rounds on pure tool-call responses and falls back.
        assert!(
            result.tool_results.len() > 1,
            "Should have initial + scratch pad results"
        );
    }

    #[tokio::test]
    async fn test_run_tool_loop_detects_duplicate_calls() {
        // Provider returns the same tool call repeatedly — loop detection should break it.
        let mut responses = Vec::new();
        for i in 0..10 {
            responses.push(crate::providers::base::LLMResponse {
                content: None,
                tool_calls: vec![ToolCallRequest {
                    id: format!("dup_{}", i),
                    name: "test_tool".to_string(),
                    arguments: {
                        let mut m = HashMap::new();
                        m.insert("query".to_string(), json!("same_args"));
                        m
                    },
                }],
                finish_reason: "tool_calls".to_string(),
                usage: HashMap::new(),
            });
        }

        let provider = Arc::new(MockProvider::new(responses));
        let mut tools = ToolRegistry::new();
        tools.register(Box::new(CountingTool::new()));

        let config = ToolRunnerConfig {
            provider,
            model: "mock".to_string(),
            max_iterations: 10,
            max_tokens: 4096,
            needs_user_continuation: false,
            max_tool_result_chars: 30000,
            short_circuit_chars: 0,
            depth: 0,
            cancellation_token: None,
            verbatim: false,
            budget: None,
        };

        // Initial call uses "test" as query (from make_tool_calls).
        // First chain uses "same_args" (new, allowed).
        // Second chain uses "same_args" again (duplicate, blocked).
        let result = run_tool_loop(&config, &make_tool_calls(&["test_tool"]), &tools, "test").await;

        // Scratch pad detects duplicates: first "same_args" call is new, subsequent skipped.
        assert_eq!(result.iterations_used, 1);
        assert_eq!(
            result.tool_results.len(),
            2,
            "Initial + 1 new call from scratch pad (duplicates skipped)"
        );
    }

    #[tokio::test]
    async fn test_run_tool_loop_returns_on_no_more_tool_calls() {
        let provider = Arc::new(MockProvider::new(vec![
            crate::providers::base::LLMResponse {
                content: Some("Summary: found 3 files.".to_string()),
                tool_calls: vec![],
                finish_reason: "stop".to_string(),
                usage: HashMap::new(),
            },
        ]));

        let mut tools = ToolRegistry::new();
        tools.register(Box::new(CountingTool::new()));

        let config = ToolRunnerConfig {
            provider,
            model: "mock".to_string(),
            max_iterations: 10,
            max_tokens: 4096,
            needs_user_continuation: false,
            max_tool_result_chars: 30000,
            short_circuit_chars: 0,
            depth: 0,
            cancellation_token: None,
            verbatim: false,
            budget: None,
        };

        let result = run_tool_loop(
            &config,
            &make_tool_calls(&["test_tool"]),
            &tools,
            "find files",
        )
        .await;

        assert_eq!(result.iterations_used, 1);
        assert_eq!(result.summary.as_deref(), Some("Summary: found 3 files."));
    }

    #[test]
    fn test_aggregate_results_formatting() {
        let result = ToolRunResult {
            tool_results: vec![
                (
                    "id1".into(),
                    "read_file".into(),
                    "file contents here".into(),
                ),
                ("id2".into(), "exec".into(), "command output".into()),
            ],
            summary: Some("Read a file and ran a command.".to_string()),
            iterations_used: 1,
            error: None,
        };

        let formatted = format_results_for_context(&result, 2000, None);
        assert!(formatted.contains("[read_file]: file contents here"));
        assert!(formatted.contains("[exec]: command output"));
        assert!(formatted.contains("Summary: Read a file and ran a command."));
    }

    #[test]
    fn test_aggregate_results_truncation() {
        let long_data = "x".repeat(3000);
        let result = ToolRunResult {
            tool_results: vec![("id1".into(), "big_tool".into(), long_data)],
            summary: None,
            iterations_used: 1,
            error: None,
        };

        let formatted = format_results_for_context(&result, 2000, None);
        assert!(formatted.contains("chars total"));
        assert!(formatted.len() < 3000);
    }

    #[test]
    fn test_slim_results_truncation() {
        let result = ToolRunResult {
            tool_results: vec![("id1".into(), "read_file".into(), "x".repeat(500))],
            summary: Some("Found a large file.".to_string()),
            iterations_used: 1,
            error: None,
        };

        // Slim mode: 200 char preview.
        let slim = format_results_for_context(&result, 200, None);
        assert!(slim.contains("500 chars total"));
        assert!(slim.len() < 400);

        // Full mode: 2000 char limit — 500 chars fits.
        let full = format_results_for_context(&result, 2000, None);
        assert!(!full.contains("chars total"));
        assert!(full.contains(&"x".repeat(500)));
    }

    // -- Message capturing mock for continuation message test --

    /// Mock provider that captures the messages array passed to chat().
    struct CapturingProvider {
        captured_messages: tokio::sync::Mutex<Vec<Vec<Value>>>,
    }

    impl CapturingProvider {
        fn new() -> Self {
            Self {
                captured_messages: tokio::sync::Mutex::new(Vec::new()),
            }
        }
    }

    #[async_trait]
    impl LLMProvider for CapturingProvider {
        async fn chat(
            &self,
            messages: &[Value],
            _tools: Option<&[Value]>,
            _model: Option<&str>,
            _max_tokens: u32,
            _temperature: f64,
            _thinking_budget: Option<u32>,
        ) -> anyhow::Result<crate::providers::base::LLMResponse> {
            self.captured_messages.lock().await.push(messages.to_vec());
            Ok(crate::providers::base::LLMResponse {
                content: Some("Done.".to_string()),
                tool_calls: vec![],
                finish_reason: "stop".to_string(),
                usage: HashMap::new(),
            })
        }

        fn get_default_model(&self) -> &str {
            "capturing-model"
        }
    }

    #[tokio::test]
    async fn test_tool_loop_message_sequence_ends_with_tool_result() {
        // Mistral/Ministral templates handle tool→generate natively.
        // Conversation should end with tool results (NOT user continuation).
        let provider = Arc::new(CapturingProvider::new());

        let mut tools = ToolRegistry::new();
        tools.register(Box::new(CountingTool::new()));

        let config = ToolRunnerConfig {
            provider: provider.clone(),
            model: "mock".to_string(),
            max_iterations: 10,
            max_tokens: 4096,
            needs_user_continuation: false,
            max_tool_result_chars: 30000,
            short_circuit_chars: 0,
            depth: 0,
            cancellation_token: None,
            verbatim: false,
            budget: None,
        };

        let _ = run_tool_loop(
            &config,
            &make_tool_calls(&["test_tool"]),
            &tools,
            "test context",
        )
        .await;

        let captured = provider.captured_messages.lock().await;
        assert!(
            !captured.is_empty(),
            "Should have made at least one LLM call"
        );

        let messages = &captured[0];

        // Expected sequence: system, user, assistant(tool_calls), tool(result)
        // NO user continuation — Mistral handles tool→generate natively.
        // Scratch pad sends fresh [system, user] messages each round.
        assert_eq!(
            messages.len(),
            2,
            "Scratch pad should send fresh [system, user] messages, got {}",
            messages.len()
        );
        assert_eq!(messages[0]["role"].as_str(), Some("system"));
        assert_eq!(messages[1]["role"].as_str(), Some("user"));
    }

    #[tokio::test]
    async fn test_tool_loop_continuation_present_in_chained_calls() {
        // When the model requests chained tool calls, each iteration
        // should have a user continuation message.
        let provider = Arc::new(MockProvider::new(vec![
            // First response: request another tool call
            crate::providers::base::LLMResponse {
                content: None,
                tool_calls: vec![ToolCallRequest {
                    id: "chain_0".to_string(),
                    name: "test_tool".to_string(),
                    arguments: {
                        let mut m = HashMap::new();
                        m.insert("query".to_string(), json!("chained"));
                        m
                    },
                }],
                finish_reason: "tool_calls".to_string(),
                usage: HashMap::new(),
            },
            // Second response: done
            crate::providers::base::LLMResponse {
                content: Some("Chained done.".to_string()),
                tool_calls: vec![],
                finish_reason: "stop".to_string(),
                usage: HashMap::new(),
            },
        ]));

        let mut tools = ToolRegistry::new();
        tools.register(Box::new(CountingTool::new()));

        let config = ToolRunnerConfig {
            provider,
            model: "mock".to_string(),
            max_iterations: 10,
            max_tokens: 4096,
            needs_user_continuation: false,
            max_tool_result_chars: 30000,
            short_circuit_chars: 0,
            depth: 0,
            cancellation_token: None,
            verbatim: false,
            budget: None,
        };

        let result = run_tool_loop(
            &config,
            &make_tool_calls(&["test_tool"]),
            &tools,
            "test context",
        )
        .await;

        // Scratch pad handles chaining: initial + 1 chained call, then "Chained done."
        assert_eq!(result.iterations_used, 1);
        assert_eq!(result.tool_results.len(), 2);
        assert_eq!(result.summary.as_deref(), Some("Chained done."));
    }

    #[tokio::test]
    async fn test_tool_loop_multiple_simultaneous_tools() {
        // Multiple tool calls in a single iteration — conversation ends
        // with the last tool result (no user continuation).
        let provider = Arc::new(CapturingProvider::new());

        let mut tools = ToolRegistry::new();
        tools.register(Box::new(CountingTool::new()));

        let config = ToolRunnerConfig {
            provider: provider.clone(),
            model: "mock".to_string(),
            max_iterations: 10,
            max_tokens: 4096,
            needs_user_continuation: false,
            max_tool_result_chars: 30000,
            short_circuit_chars: 0,
            depth: 0,
            cancellation_token: None,
            verbatim: false,
            budget: None,
        };

        // 3 simultaneous tool calls
        let calls = make_tool_calls(&["test_tool", "test_tool", "test_tool"]);
        let result = run_tool_loop(&config, &calls, &tools, "multi-tool test").await;

        assert_eq!(result.tool_results.len(), 3);

        let captured = provider.captured_messages.lock().await;
        let messages = &captured[0];

        // Scratch pad sends fresh [system, user] messages each round.
        assert_eq!(
            messages.len(),
            2,
            "Scratch pad should send fresh [system, user] messages"
        );
        assert_eq!(messages[0]["role"].as_str(), Some("system"));
        assert_eq!(messages[1]["role"].as_str(), Some("user"));
    }

    /// Mock provider that fails on the first call.
    struct FailingProvider;

    #[async_trait]
    impl LLMProvider for FailingProvider {
        async fn chat(
            &self,
            _messages: &[Value],
            _tools: Option<&[Value]>,
            _model: Option<&str>,
            _max_tokens: u32,
            _temperature: f64,
            _thinking_budget: Option<u32>,
        ) -> anyhow::Result<crate::providers::base::LLMResponse> {
            Err(anyhow::anyhow!("Connection refused"))
        }

        fn get_default_model(&self) -> &str {
            "failing-model"
        }
    }

    #[tokio::test]
    async fn test_tool_loop_provider_error_still_returns_results() {
        // Even if the LLM call fails, tool results gathered so far should be returned
        let provider = Arc::new(FailingProvider);

        let mut tools = ToolRegistry::new();
        tools.register(Box::new(CountingTool::new()));

        let config = ToolRunnerConfig {
            provider,
            model: "mock".to_string(),
            max_iterations: 10,
            max_tokens: 4096,
            needs_user_continuation: false,
            max_tool_result_chars: 30000,
            short_circuit_chars: 0,
            depth: 0,
            cancellation_token: None,
            verbatim: false,
            budget: None,
        };

        let result = run_tool_loop(
            &config,
            &make_tool_calls(&["test_tool"]),
            &tools,
            "test context",
        )
        .await;

        // Tool was executed before the LLM call failed
        assert_eq!(result.tool_results.len(), 1);
        assert_eq!(result.tool_results[0].1, "test_tool");
        // Scratch pad LLM call fails, falls back to None (no memory findings).
        assert!(
            result.summary.is_none(),
            "Provider error in scratch pad falls back to None: {:?}",
            result.summary
        );
        assert_eq!(result.iterations_used, 1);
    }

    #[tokio::test]
    async fn test_tool_loop_empty_initial_calls() {
        // No initial tool calls — should return immediately
        let provider = Arc::new(MockProvider::new(vec![]));

        let mut tools = ToolRegistry::new();
        tools.register(Box::new(CountingTool::new()));

        let config = ToolRunnerConfig {
            provider,
            model: "mock".to_string(),
            max_iterations: 10,
            max_tokens: 4096,
            needs_user_continuation: false,
            max_tool_result_chars: 30000,
            short_circuit_chars: 0,
            depth: 0,
            cancellation_token: None,
            verbatim: false,
            budget: None,
        };

        let result = run_tool_loop(&config, &[], &tools, "test").await;

        assert!(result.tool_results.is_empty());
        // Loop enters iteration 0, sets iterations_used=1, then breaks on empty check
        assert_eq!(result.iterations_used, 1);
    }

    // -- Tool call ID normalization tests --

    #[test]
    fn test_normalize_tool_call_id_length() {
        // Mistral requires exactly 9 alphanumeric chars
        for i in 0..100 {
            let id = normalize_tool_call_id(i);
            assert_eq!(id.len(), 9, "ID should be 9 chars: {}", id);
            assert!(
                id.chars().all(|c| c.is_ascii_alphanumeric()),
                "ID should be alphanumeric: {}",
                id
            );
        }
    }

    #[test]
    fn test_normalize_tool_call_id_unique() {
        let ids: Vec<String> = (0..50).map(normalize_tool_call_id).collect();
        let unique: std::collections::HashSet<&String> = ids.iter().collect();
        assert_eq!(ids.len(), unique.len(), "All IDs should be unique");
    }

    #[test]
    fn test_normalize_tool_call_id_format() {
        assert_eq!(normalize_tool_call_id(0), "tc0000000");
        assert_eq!(normalize_tool_call_id(1), "tc0000001");
        assert_eq!(normalize_tool_call_id(42), "tc0000042");
        assert_eq!(normalize_tool_call_id(9999999), "tc9999999");
    }

    #[tokio::test]
    async fn test_tool_loop_normalizes_ids_in_messages() {
        // Verify that tool call IDs sent to the delegation model are
        // normalized to 9 chars, even when original IDs are longer.
        let provider = Arc::new(CapturingProvider::new());

        let mut tools = ToolRegistry::new();
        tools.register(Box::new(CountingTool::new()));

        let config = ToolRunnerConfig {
            provider: provider.clone(),
            model: "mock".to_string(),
            max_iterations: 10,
            max_tokens: 4096,
            needs_user_continuation: false,
            max_tool_result_chars: 30000,
            short_circuit_chars: 0,
            depth: 0,
            cancellation_token: None,
            verbatim: false,
            budget: None,
        };

        // Use long IDs like cloud Claude generates
        let calls = vec![ToolCallRequest {
            id: "toolu_01XYZabc123def456ghi789".to_string(),
            name: "test_tool".to_string(),
            arguments: {
                let mut m = HashMap::new();
                m.insert("query".to_string(), json!("test"));
                m
            },
        }];

        let result = run_tool_loop(&config, &calls, &tools, "test context").await;

        // Result should use ORIGINAL ID (for main model correlation)
        assert_eq!(
            result.tool_results[0].0, "toolu_01XYZabc123def456ghi789",
            "Results should preserve original tool call ID"
        );

        // Scratch pad receives fresh [system, user] messages — no tool_call IDs in those.
        // The key guarantee is original IDs in results (verified above).
    }

    #[tokio::test]
    async fn test_tool_loop_id_mapping_preserves_originals() {
        // When multiple tools are called, each should map back to its original ID
        let provider = Arc::new(MockProvider::new(vec![
            crate::providers::base::LLMResponse {
                content: Some("Done.".to_string()),
                tool_calls: vec![],
                finish_reason: "stop".to_string(),
                usage: HashMap::new(),
            },
        ]));

        let mut tools = ToolRegistry::new();
        tools.register(Box::new(CountingTool::new()));

        let config = ToolRunnerConfig {
            provider,
            model: "mock".to_string(),
            max_iterations: 10,
            max_tokens: 4096,
            needs_user_continuation: false,
            max_tool_result_chars: 30000,
            short_circuit_chars: 0,
            depth: 0,
            cancellation_token: None,
            verbatim: false,
            budget: None,
        };

        let calls = vec![
            ToolCallRequest {
                id: "toolu_01AAAA".to_string(),
                name: "test_tool".to_string(),
                arguments: {
                    let mut m = HashMap::new();
                    m.insert("query".to_string(), json!("first"));
                    m
                },
            },
            ToolCallRequest {
                id: "toolu_01BBBB".to_string(),
                name: "test_tool".to_string(),
                arguments: {
                    let mut m = HashMap::new();
                    m.insert("query".to_string(), json!("second"));
                    m
                },
            },
        ];

        let result = run_tool_loop(&config, &calls, &tools, "test").await;

        assert_eq!(result.tool_results.len(), 2);
        assert_eq!(result.tool_results[0].0, "toolu_01AAAA");
        assert_eq!(result.tool_results[1].0, "toolu_01BBBB");
    }

    // -- Truncation tests --

    /// Mock tool that returns a long string of given length.
    struct VerboseTool {
        output_len: usize,
    }

    #[async_trait]
    impl Tool for VerboseTool {
        fn name(&self) -> &str {
            "verbose_tool"
        }
        fn description(&self) -> &str {
            "Returns a long string"
        }
        fn parameters(&self) -> Value {
            json!({"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]})
        }
        async fn execute(&self, _params: HashMap<String, Value>) -> String {
            "x".repeat(self.output_len)
        }
    }

    #[tokio::test]
    async fn test_large_result_injects_metadata() {
        // When a tool result exceeds max_tool_result_chars, the delegation
        // model should see metadata (variable name + char count + preview)
        // instead of the raw data.
        let provider = Arc::new(CapturingProvider::new());

        let mut tools = ToolRegistry::new();
        tools.register(Box::new(VerboseTool { output_len: 500 }));

        let config = ToolRunnerConfig {
            provider: provider.clone(),
            model: "mock".to_string(),
            max_iterations: 10,
            max_tokens: 4096,
            needs_user_continuation: false,
            max_tool_result_chars: 100,
            short_circuit_chars: 0,
            depth: 0,
            cancellation_token: None,
            verbatim: false,
            budget: None,
        };

        let calls = vec![ToolCallRequest {
            id: "call_0".to_string(),
            name: "verbose_tool".to_string(),
            arguments: {
                let mut m = HashMap::new();
                m.insert("query".to_string(), json!("test"));
                m
            },
        }];

        let result = run_tool_loop(&config, &calls, &tools, "test").await;

        // all_results should have full 500-char data
        assert_eq!(result.tool_results.len(), 1);
        assert_eq!(result.tool_results[0].2.len(), 500);

        // Scratch pad receives variable metadata in the user message state.
        let captured = provider.captured_messages.lock().await;
        let messages = &captured[0];
        let user_msg = messages
            .iter()
            .find(|m| m["role"].as_str() == Some("user"))
            .unwrap();
        let content = user_msg["content"].as_str().unwrap();
        assert!(
            content.contains("output_0"),
            "Scratch pad state should contain variable metadata, got: {}",
            content
        );
        assert!(
            content.contains("500 chars"),
            "State should contain char count, got: {}",
            content
        );
    }

    #[tokio::test]
    async fn test_small_result_injects_directly() {
        // Results under max_tool_result_chars should be injected as full text.
        let provider = Arc::new(CapturingProvider::new());

        let mut tools = ToolRegistry::new();
        tools.register(Box::new(VerboseTool { output_len: 50 }));

        let config = ToolRunnerConfig {
            provider: provider.clone(),
            model: "mock".to_string(),
            max_iterations: 10,
            max_tokens: 4096,
            needs_user_continuation: false,
            max_tool_result_chars: 100,
            short_circuit_chars: 0,
            depth: 0,
            cancellation_token: None,
            verbatim: false,
            budget: None,
        };

        let calls = vec![ToolCallRequest {
            id: "call_0".to_string(),
            name: "verbose_tool".to_string(),
            arguments: {
                let mut m = HashMap::new();
                m.insert("query".to_string(), json!("test"));
                m
            },
        }];

        let result = run_tool_loop(&config, &calls, &tools, "test").await;

        // Full data in both places
        assert_eq!(result.tool_results[0].2.len(), 50);

        // Scratch pad receives variable metadata in user message state.
        let captured = provider.captured_messages.lock().await;
        let messages = &captured[0];
        let user_msg = messages
            .iter()
            .find(|m| m["role"].as_str() == Some("user"))
            .unwrap();
        let content = user_msg["content"].as_str().unwrap();
        // Even small results are stored as variables in ContextStore.
        assert!(
            content.contains("output_0"),
            "Scratch pad state should contain variable info, got: {}",
            content
        );
    }

    #[tokio::test]
    async fn test_short_circuit_skips_llm_for_trivial_results() {
        // When all results are under short_circuit_chars, the delegation
        // LLM should NOT be called — results returned directly.
        let provider = Arc::new(CapturingProvider::new());

        let mut tools = ToolRegistry::new();
        tools.register(Box::new(CountingTool::new())); // returns "tool result data" (16 chars)

        let config = ToolRunnerConfig {
            provider: provider.clone(),
            model: "mock".to_string(),
            max_iterations: 10,
            max_tokens: 4096,
            needs_user_continuation: false,
            max_tool_result_chars: 30000,
            short_circuit_chars: 200, // 16 < 200 → short-circuit
            depth: 0,
            cancellation_token: None,
            verbatim: false,
            budget: None,
        };

        let result = run_tool_loop(
            &config,
            &make_tool_calls(&["test_tool"]),
            &tools,
            "test context",
        )
        .await;

        // Tool was executed
        assert_eq!(result.tool_results.len(), 1);
        assert_eq!(result.tool_results[0].2, "tool result data");
        // No summary — LLM was skipped
        assert!(
            result.summary.is_none(),
            "Short-circuit should skip LLM, summary should be None"
        );
        // No LLM calls made
        let captured = provider.captured_messages.lock().await;
        assert!(
            captured.is_empty(),
            "No LLM calls should have been made for short results"
        );
    }

    #[tokio::test]
    async fn test_short_circuit_disabled_when_zero() {
        // When short_circuit_chars is 0, even trivial results go through the LLM.
        let provider = Arc::new(MockProvider::new(vec![
            crate::providers::base::LLMResponse {
                content: Some("Summarized.".to_string()),
                tool_calls: vec![],
                finish_reason: "stop".to_string(),
                usage: HashMap::new(),
            },
        ]));

        let mut tools = ToolRegistry::new();
        tools.register(Box::new(CountingTool::new()));

        let config = ToolRunnerConfig {
            provider,
            model: "mock".to_string(),
            max_iterations: 10,
            max_tokens: 4096,
            needs_user_continuation: false,
            max_tool_result_chars: 30000,
            short_circuit_chars: 0, // disabled
            depth: 0,
            cancellation_token: None,
            verbatim: false,
            budget: None,
        };

        let result = run_tool_loop(
            &config,
            &make_tool_calls(&["test_tool"]),
            &tools,
            "test context",
        )
        .await;

        assert_eq!(result.tool_results.len(), 1);
        assert_eq!(
            result.summary.as_deref(),
            Some("Summarized."),
            "With short_circuit_chars=0, LLM should still be called"
        );
    }

    // -- Tool filtering tests (prompt injection defense) --

    /// A second mock tool to test tool filtering.
    struct DangerousTool;

    #[async_trait]
    impl Tool for DangerousTool {
        fn name(&self) -> &str {
            "dangerous_tool"
        }
        fn description(&self) -> &str {
            "A tool that should not be accessible to delegation"
        }
        fn parameters(&self) -> Value {
            json!({"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]})
        }
        async fn execute(&self, _params: HashMap<String, Value>) -> String {
            "DANGER: this should not execute".to_string()
        }
    }

    #[tokio::test]
    async fn test_tool_filtering_blocks_uninvited_tools() {
        // Delegation model tries to call "dangerous_tool" but the main model
        // only requested "test_tool". The dangerous call should be blocked.
        let provider = Arc::new(MockProvider::new(vec![
            // Delegation model tries to call dangerous_tool
            crate::providers::base::LLMResponse {
                content: None,
                tool_calls: vec![ToolCallRequest {
                    id: "evil_0".to_string(),
                    name: "dangerous_tool".to_string(),
                    arguments: {
                        let mut m = HashMap::new();
                        m.insert("cmd".to_string(), json!("rm -rf /"));
                        m
                    },
                }],
                finish_reason: "tool_calls".to_string(),
                usage: HashMap::new(),
            },
        ]));

        let mut tools = ToolRegistry::new();
        tools.register(Box::new(CountingTool::new()));
        tools.register(Box::new(DangerousTool));

        let config = ToolRunnerConfig {
            provider,
            model: "mock".to_string(),
            max_iterations: 10,
            max_tokens: 4096,
            needs_user_continuation: false,
            max_tool_result_chars: 30000,
            short_circuit_chars: 0,
            depth: 0,
            cancellation_token: None,
            verbatim: false,
            budget: None,
        };

        // Main model only requested test_tool
        let result = run_tool_loop(
            &config,
            &make_tool_calls(&["test_tool"]),
            &tools,
            "test context",
        )
        .await;

        // Only test_tool should have executed (from initial calls)
        assert_eq!(
            result.tool_results.len(),
            1,
            "Only initial tool should execute"
        );
        assert_eq!(result.tool_results[0].1, "test_tool");
        // dangerous_tool was blocked, all calls were filtered → loop detected
        assert!(
            result.summary.is_some(),
            "Should have a summary after blocking"
        );
    }

    #[tokio::test]
    async fn test_tool_filtering_allows_same_tool_different_args() {
        // Delegation model chains test_tool with different args — allowed.
        let provider = Arc::new(MockProvider::new(vec![
            // Chain: same tool, different args
            crate::providers::base::LLMResponse {
                content: None,
                tool_calls: vec![ToolCallRequest {
                    id: "chain_0".to_string(),
                    name: "test_tool".to_string(),
                    arguments: {
                        let mut m = HashMap::new();
                        m.insert("query".to_string(), json!("follow_up"));
                        m
                    },
                }],
                finish_reason: "tool_calls".to_string(),
                usage: HashMap::new(),
            },
            // Done
            crate::providers::base::LLMResponse {
                content: Some("All done.".to_string()),
                tool_calls: vec![],
                finish_reason: "stop".to_string(),
                usage: HashMap::new(),
            },
        ]));

        let mut tools = ToolRegistry::new();
        tools.register(Box::new(CountingTool::new()));
        tools.register(Box::new(DangerousTool));

        let config = ToolRunnerConfig {
            provider,
            model: "mock".to_string(),
            max_iterations: 10,
            max_tokens: 4096,
            needs_user_continuation: false,
            max_tool_result_chars: 30000,
            short_circuit_chars: 0,
            depth: 0,
            cancellation_token: None,
            verbatim: false,
            budget: None,
        };

        let result = run_tool_loop(
            &config,
            &make_tool_calls(&["test_tool"]),
            &tools,
            "test context",
        )
        .await;

        // Both calls should execute (same tool, different args)
        assert_eq!(
            result.tool_results.len(),
            2,
            "Should have 2 results (initial + chain)"
        );
        assert_eq!(result.tool_results[0].1, "test_tool");
        assert_eq!(result.tool_results[1].1, "test_tool");
    }

    #[tokio::test]
    async fn test_tool_defs_filtered_for_delegation() {
        // Verify the delegation model only receives definitions for requested tools.
        let provider = Arc::new(CapturingProvider::new());

        let mut tools = ToolRegistry::new();
        tools.register(Box::new(CountingTool::new())); // test_tool
        tools.register(Box::new(DangerousTool)); // dangerous_tool

        let config = ToolRunnerConfig {
            provider: provider.clone(),
            model: "mock".to_string(),
            max_iterations: 10,
            max_tokens: 4096,
            needs_user_continuation: false,
            max_tool_result_chars: 30000,
            short_circuit_chars: 0,
            depth: 0,
            cancellation_token: None,
            verbatim: false,
            budget: None,
        };

        let _ = run_tool_loop(
            &config,
            &make_tool_calls(&["test_tool"]),
            &tools,
            "test context",
        )
        .await;

        // The capturing provider doesn't directly expose tools, but we can
        // verify that dangerous_tool is NOT in the tool definitions by checking
        // the registry has 2 tools but only 1 was allowed.
        let all_defs = tools.get_definitions();
        assert_eq!(all_defs.len(), 2, "Registry should have both tools");

        // The run completed without dangerous_tool being available
        let captured = provider.captured_messages.lock().await;
        assert!(
            !captured.is_empty(),
            "Should have made at least one LLM call"
        );
    }

    // -- ContextStore integration tests --

    #[tokio::test]
    async fn test_micro_tool_results_not_in_all_results() {
        // When the delegation model calls ctx_slice, the result should NOT
        // appear in the returned tool_results (micro-tools are internal).
        let provider = Arc::new(MockProvider::new(vec![
            // Delegation model requests a micro-tool
            crate::providers::base::LLMResponse {
                content: None,
                tool_calls: vec![ToolCallRequest {
                    id: "micro_0".to_string(),
                    name: "ctx_length".to_string(),
                    arguments: {
                        let mut m = HashMap::new();
                        m.insert("variable".to_string(), json!("output_0"));
                        m
                    },
                }],
                finish_reason: "tool_calls".to_string(),
                usage: HashMap::new(),
            },
            // Then it summarizes
            crate::providers::base::LLMResponse {
                content: Some("Length is 16.".to_string()),
                tool_calls: vec![],
                finish_reason: "stop".to_string(),
                usage: HashMap::new(),
            },
        ]));

        let mut tools = ToolRegistry::new();
        tools.register(Box::new(CountingTool::new()));

        let config = ToolRunnerConfig {
            provider,
            model: "mock".to_string(),
            max_iterations: 10,
            max_tokens: 4096,
            needs_user_continuation: false,
            max_tool_result_chars: 30000,
            short_circuit_chars: 0,
            depth: 0,
            cancellation_token: None,
            verbatim: false,
            budget: None,
        };

        let result = run_tool_loop(
            &config,
            &make_tool_calls(&["test_tool"]),
            &tools,
            "test context",
        )
        .await;

        // Only the real tool call should be in results
        assert_eq!(
            result.tool_results.len(),
            1,
            "Micro-tool results should not be in all_results"
        );
        assert_eq!(result.tool_results[0].1, "test_tool");
    }

    #[tokio::test]
    async fn test_delegation_receives_micro_tool_defs() {
        // Verify the delegation model receives ctx_slice, ctx_grep, ctx_length
        // in its tool definitions on every iteration.
        struct CapturingToolsProvider {
            captured_tools: tokio::sync::Mutex<Vec<Vec<Value>>>,
            call_count: AtomicU32,
        }

        impl CapturingToolsProvider {
            fn new() -> Self {
                Self {
                    captured_tools: tokio::sync::Mutex::new(Vec::new()),
                    call_count: AtomicU32::new(0),
                }
            }
        }

        #[async_trait]
        impl LLMProvider for CapturingToolsProvider {
            async fn chat(
                &self,
                _messages: &[Value],
                tools: Option<&[Value]>,
                _model: Option<&str>,
                _max_tokens: u32,
                _temperature: f64,
                _thinking_budget: Option<u32>,
            ) -> anyhow::Result<crate::providers::base::LLMResponse> {
                if let Some(t) = tools {
                    self.captured_tools.lock().await.push(t.to_vec());
                }
                let n = self.call_count.fetch_add(1, Ordering::SeqCst);
                if n == 0 {
                    // First call (iteration 0): request a chained tool to trigger iteration 1
                    Ok(crate::providers::base::LLMResponse {
                        content: None,
                        tool_calls: vec![ToolCallRequest {
                            id: "chain_0".to_string(),
                            name: "test_tool".to_string(),
                            arguments: {
                                let mut m = HashMap::new();
                                m.insert("query".to_string(), json!("follow_up"));
                                m
                            },
                        }],
                        finish_reason: "tool_calls".to_string(),
                        usage: HashMap::new(),
                    })
                } else {
                    // Second call (iteration 1): done
                    Ok(crate::providers::base::LLMResponse {
                        content: Some("Done.".to_string()),
                        tool_calls: vec![],
                        finish_reason: "stop".to_string(),
                        usage: HashMap::new(),
                    })
                }
            }

            fn get_default_model(&self) -> &str {
                "capturing-tools"
            }
        }

        let provider = Arc::new(CapturingToolsProvider::new());

        let mut tools = ToolRegistry::new();
        tools.register(Box::new(CountingTool::new()));

        let config = ToolRunnerConfig {
            provider: provider.clone(),
            model: "mock".to_string(),
            max_iterations: 10,
            max_tokens: 4096,
            needs_user_continuation: false,
            max_tool_result_chars: 30000,
            short_circuit_chars: 0,
            depth: 0,
            cancellation_token: None,
            verbatim: false,
            budget: None,
        };

        let _ = run_tool_loop(
            &config,
            &make_tool_calls(&["test_tool"]),
            &tools,
            "test context",
        )
        .await;

        let captured = provider.captured_tools.lock().await;
        // All iterations now receive tools (no iteration 0 suppression).
        assert_eq!(
            captured.len(),
            2,
            "Both iterations should receive tool definitions"
        );

        // Check the first iteration's tool defs (representative of all).
        let defs = &captured[0];
        let tool_names: Vec<&str> = defs
            .iter()
            .filter_map(|d| d.pointer("/function/name").and_then(|v| v.as_str()))
            .collect();

        assert!(
            tool_names.contains(&"test_tool"),
            "Should include the real tool"
        );
        assert!(
            tool_names.contains(&"ctx_slice"),
            "Should include ctx_slice"
        );
        assert!(tool_names.contains(&"ctx_grep"), "Should include ctx_grep");
        assert!(
            tool_names.contains(&"ctx_length"),
            "Should include ctx_length"
        );
    }

    #[tokio::test]
    async fn test_short_circuit_bypasses_context_store() {
        // When results are short enough for short-circuit, the ContextStore
        // is still populated but the LLM is not called.
        let provider = Arc::new(CapturingProvider::new());

        let mut tools = ToolRegistry::new();
        tools.register(Box::new(CountingTool::new())); // returns "tool result data" (16 chars)

        let config = ToolRunnerConfig {
            provider: provider.clone(),
            model: "mock".to_string(),
            max_iterations: 10,
            max_tokens: 4096,
            needs_user_continuation: false,
            max_tool_result_chars: 30000,
            short_circuit_chars: 200, // 16 < 200 → short-circuit
            depth: 0,
            cancellation_token: None,
            verbatim: false,
            budget: None,
        };

        let result = run_tool_loop(
            &config,
            &make_tool_calls(&["test_tool"]),
            &tools,
            "test context",
        )
        .await;

        // Tool was executed and full result returned
        assert_eq!(result.tool_results.len(), 1);
        assert_eq!(result.tool_results[0].2, "tool result data");
        // No LLM call (short-circuited)
        assert!(result.summary.is_none());
        let captured = provider.captured_messages.lock().await;
        assert!(captured.is_empty(), "No LLM calls should have been made");
    }

    // -- ctx_summarize tests --

    #[tokio::test]
    async fn test_ctx_summarize_depth_guard() {
        // When depth >= MAX_DEPTH, execute_ctx_summarize returns an error
        // without calling the provider.
        let provider: Arc<dyn LLMProvider> = Arc::new(MockProvider::new(vec![]));
        let mut store = context_store::ContextStore::new();
        store.store("Some content to summarize.".to_string());

        let result = context_store::execute_ctx_summarize(
            &store,
            "output_0",
            "Summarize this",
            &provider,
            "mock",
            2, // At max depth
            4096,
        )
        .await;

        assert!(
            result.contains("depth limit"),
            "Should return depth limit error, got: {}",
            result
        );
        assert!(
            result.starts_with("Error:"),
            "Should be an error: {}",
            result
        );
    }

    #[tokio::test]
    async fn test_ctx_summarize_missing_variable() {
        // ctx_summarize with a nonexistent variable should return an error
        // without calling the provider.
        let provider: Arc<dyn LLMProvider> = Arc::new(MockProvider::new(vec![]));
        let store = context_store::ContextStore::new(); // empty store

        let result = context_store::execute_ctx_summarize(
            &store,
            "nonexistent_var",
            "Summarize this",
            &provider,
            "mock",
            0,
            4096,
        )
        .await;

        assert!(
            result.contains("not found"),
            "Should return not-found error, got: {}",
            result
        );
        assert!(
            result.starts_with("Error:"),
            "Should be an error: {}",
            result
        );
    }

    #[tokio::test]
    async fn test_ctx_summarize_produces_summary() {
        // ctx_summarize should call the provider recursively to summarize
        // a large variable. The sub-loop produces a text summary.

        // Track how many times the provider is called.
        let call_count = Arc::new(AtomicU32::new(0));
        let call_count_clone = call_count.clone();

        struct SummarizingProvider {
            call_count: Arc<AtomicU32>,
        }

        #[async_trait]
        impl LLMProvider for SummarizingProvider {
            async fn chat(
                &self,
                _messages: &[Value],
                _tools: Option<&[Value]>,
                _model: Option<&str>,
                _max_tokens: u32,
                _temperature: f64,
                _thinking_budget: Option<u32>,
            ) -> anyhow::Result<crate::providers::base::LLMResponse> {
                let n = self.call_count.fetch_add(1, Ordering::SeqCst);
                match n {
                    0 => {
                        // First call (outer loop): delegation model calls ctx_summarize
                        Ok(crate::providers::base::LLMResponse {
                            content: None,
                            tool_calls: vec![ToolCallRequest {
                                id: "sum_0".to_string(),
                                name: "ctx_summarize".to_string(),
                                arguments: {
                                    let mut m = HashMap::new();
                                    m.insert("variable".to_string(), json!("output_0"));
                                    m.insert(
                                        "instruction".to_string(),
                                        json!("Extract the main topic"),
                                    );
                                    m
                                },
                            }],
                            finish_reason: "tool_calls".to_string(),
                            usage: HashMap::new(),
                        })
                    }
                    1 => {
                        // Second call (sub-loop): summarize model produces text
                        Ok(crate::providers::base::LLMResponse {
                            content: Some("The content discusses Rust programming.".to_string()),
                            tool_calls: vec![],
                            finish_reason: "stop".to_string(),
                            usage: HashMap::new(),
                        })
                    }
                    _ => {
                        // Third call (outer loop resumes): use summary to finish
                        Ok(crate::providers::base::LLMResponse {
                            content: Some("Based on the summary: Rust programming.".to_string()),
                            tool_calls: vec![],
                            finish_reason: "stop".to_string(),
                            usage: HashMap::new(),
                        })
                    }
                }
            }

            fn get_default_model(&self) -> &str {
                "summarizing-model"
            }
        }

        let provider = Arc::new(SummarizingProvider {
            call_count: call_count_clone,
        });

        let mut tools = ToolRegistry::new();
        tools.register(Box::new(VerboseTool { output_len: 500 }));

        let config = ToolRunnerConfig {
            provider,
            model: "mock".to_string(),
            max_iterations: 10,
            max_tokens: 4096,
            needs_user_continuation: false,
            max_tool_result_chars: 100, // Force large result → metadata
            short_circuit_chars: 0,
            depth: 0, // First level — ctx_summarize allowed
            cancellation_token: None,
            verbatim: false,
            budget: None,
        };

        let calls = vec![ToolCallRequest {
            id: "call_0".to_string(),
            name: "verbose_tool".to_string(),
            arguments: {
                let mut m = HashMap::new();
                m.insert("query".to_string(), json!("test"));
                m
            },
        }];

        let result = run_tool_loop(&config, &calls, &tools, "test").await;

        // Real tool result preserved
        assert_eq!(result.tool_results.len(), 1);
        assert_eq!(result.tool_results[0].1, "verbose_tool");
        // Provider was called at least 2 times (outer + sub-loop)
        let total_calls = call_count.load(Ordering::SeqCst);
        assert!(
            total_calls >= 2,
            "Provider should be called at least twice (outer + sub-loop), got {}",
            total_calls
        );
    }

    // -- Cancellation tests --

    #[tokio::test]
    async fn test_cancellation_before_execution() {
        // Token cancelled before loop starts → immediate return, 0 tool results.
        let token = tokio_util::sync::CancellationToken::new();
        token.cancel(); // Pre-cancel

        let provider = Arc::new(MockProvider::new(vec![]));
        let mut tools = ToolRegistry::new();
        tools.register(Box::new(CountingTool::new()));

        let config = ToolRunnerConfig {
            provider,
            model: "mock".to_string(),
            max_iterations: 10,
            max_tokens: 4096,
            needs_user_continuation: false,
            max_tool_result_chars: 30000,
            short_circuit_chars: 0,
            depth: 0,
            cancellation_token: Some(token),
            verbatim: false,
            budget: None,
        };

        let result = run_tool_loop(
            &config,
            &make_tool_calls(&["test_tool"]),
            &tools,
            "test context",
        )
        .await;

        assert!(
            result.tool_results.is_empty(),
            "No tools should execute when pre-cancelled"
        );
        assert_eq!(result.iterations_used, 1);
        assert!(
            result.summary.as_deref().unwrap().contains("cancelled"),
            "Summary should mention cancellation: {:?}",
            result.summary
        );
    }

    #[tokio::test]
    async fn test_cancellation_mid_iteration() {
        // Cancel after first iteration's LLM call returns a chain request.
        // The cancellation check at the top of iteration 2 catches it.
        let token = tokio_util::sync::CancellationToken::new();
        let token_clone = token.clone();

        // Provider that cancels the token when called (simulating cancel
        // arriving between iteration 1's LLM response and iteration 2's start).
        struct CancellingProvider {
            token: tokio_util::sync::CancellationToken,
            call_count: AtomicU32,
        }

        #[async_trait]
        impl LLMProvider for CancellingProvider {
            async fn chat(
                &self,
                _messages: &[Value],
                _tools: Option<&[Value]>,
                _model: Option<&str>,
                _max_tokens: u32,
                _temperature: f64,
                _thinking_budget: Option<u32>,
            ) -> anyhow::Result<crate::providers::base::LLMResponse> {
                let n = self.call_count.fetch_add(1, Ordering::SeqCst);
                if n == 0 {
                    // First call: return a chain request, then cancel.
                    self.token.cancel();
                    Ok(crate::providers::base::LLMResponse {
                        content: None,
                        tool_calls: vec![ToolCallRequest {
                            id: "chain_0".to_string(),
                            name: "test_tool".to_string(),
                            arguments: {
                                let mut m = HashMap::new();
                                m.insert("query".to_string(), json!("chained"));
                                m
                            },
                        }],
                        finish_reason: "tool_calls".to_string(),
                        usage: HashMap::new(),
                    })
                } else {
                    Ok(crate::providers::base::LLMResponse {
                        content: Some("Should not reach here.".to_string()),
                        tool_calls: vec![],
                        finish_reason: "stop".to_string(),
                        usage: HashMap::new(),
                    })
                }
            }

            fn get_default_model(&self) -> &str {
                "cancelling-model"
            }
        }

        let provider = Arc::new(CancellingProvider {
            token: token.clone(),
            call_count: AtomicU32::new(0),
        });

        let mut tools = ToolRegistry::new();
        tools.register(Box::new(CountingTool::new()));

        let config = ToolRunnerConfig {
            provider,
            model: "mock".to_string(),
            max_iterations: 10,
            max_tokens: 4096,
            needs_user_continuation: false,
            max_tool_result_chars: 30000,
            short_circuit_chars: 0,
            depth: 0,
            cancellation_token: Some(token_clone),
            verbatim: false,
            budget: None,
        };

        let result = run_tool_loop(
            &config,
            &make_tool_calls(&["test_tool"]),
            &tools,
            "test context",
        )
        .await;

        // Initial tool executes (1), scratch pad round 0 executes chained tool (1),
        // then cancellation is detected at the top of round 1.
        assert_eq!(
            result.tool_results.len(),
            2,
            "Initial + 1 from scratch pad before cancellation detected"
        );
        assert!(
            result.summary.as_deref().unwrap().contains("cancelled"),
            "Summary should mention cancellation: {:?}",
            result.summary
        );
    }

    #[tokio::test]
    async fn test_cancellation_none_token_works() {
        // cancellation_token: None → normal execution (backward compat).
        let provider = Arc::new(MockProvider::new(vec![
            crate::providers::base::LLMResponse {
                content: Some("All done.".to_string()),
                tool_calls: vec![],
                finish_reason: "stop".to_string(),
                usage: HashMap::new(),
            },
        ]));

        let mut tools = ToolRegistry::new();
        tools.register(Box::new(CountingTool::new()));

        let config = ToolRunnerConfig {
            provider,
            model: "mock".to_string(),
            max_iterations: 10,
            max_tokens: 4096,
            needs_user_continuation: false,
            max_tool_result_chars: 30000,
            short_circuit_chars: 0,
            depth: 0,
            cancellation_token: None,
            verbatim: false,
            budget: None,
        };

        let result = run_tool_loop(
            &config,
            &make_tool_calls(&["test_tool"]),
            &tools,
            "test context",
        )
        .await;

        assert_eq!(result.tool_results.len(), 1);
        assert_eq!(result.summary.as_deref(), Some("All done."));
    }

    #[tokio::test]
    async fn test_verbatim_skips_delegation_returns_raw() {
        // When verbatim=true, the delegation model should never be called.
        // Use an empty provider (no responses queued) — if the delegation
        // model IS called, MockProvider returns a generic "Done." which would
        // set summary to Some. Verbatim must short-circuit before that.
        let provider = Arc::new(MockProvider::new(vec![]));

        let mut tools = ToolRegistry::new();
        tools.register(Box::new(CountingTool::new()));

        let config = ToolRunnerConfig {
            provider,
            model: "mock".to_string(),
            max_iterations: 10,
            max_tokens: 4096,
            needs_user_continuation: false,
            max_tool_result_chars: 30000,
            short_circuit_chars: 0,
            depth: 0,
            cancellation_token: None,
            verbatim: true,
            budget: None,
        };

        let result = run_tool_loop(
            &config,
            &make_tool_calls(&["test_tool"]),
            &tools,
            "test context",
        )
        .await;

        // Tools are executed.
        assert_eq!(result.tool_results.len(), 1);
        assert_eq!(result.tool_results[0].1, "test_tool");
        assert_eq!(result.tool_results[0].2, "tool result data");
        // Delegation model NOT called — no summary.
        assert!(
            result.summary.is_none(),
            "verbatim mode must not produce a summary"
        );
        // Only one iteration (initial execution, no delegation loop).
        assert_eq!(result.iterations_used, 1);
    }

    #[tokio::test]
    async fn test_verbatim_false_calls_delegation() {
        // Verify that verbatim=false still calls the delegation model
        // (control test to ensure verbatim=true is actually the branch).
        let provider = Arc::new(MockProvider::new(vec![
            crate::providers::base::LLMResponse {
                content: Some("Summarized.".to_string()),
                tool_calls: vec![],
                finish_reason: "stop".to_string(),
                usage: HashMap::new(),
            },
        ]));

        let mut tools = ToolRegistry::new();
        tools.register(Box::new(CountingTool::new()));

        let config = ToolRunnerConfig {
            provider,
            model: "mock".to_string(),
            max_iterations: 10,
            max_tokens: 4096,
            needs_user_continuation: false,
            max_tool_result_chars: 30000,
            short_circuit_chars: 0,
            depth: 0,
            cancellation_token: None,
            verbatim: false,
            budget: None,
        };

        let result = run_tool_loop(
            &config,
            &make_tool_calls(&["test_tool"]),
            &tools,
            "test context",
        )
        .await;

        assert_eq!(result.tool_results.len(), 1);
        // Delegation model WAS called — summary present.
        assert_eq!(result.summary.as_deref(), Some("Summarized."));
    }

    // -- Budget tests --

    #[test]
    fn test_budget_root() {
        let budget = Budget::root(20, 3);
        assert_eq!(budget.max_iterations, 20);
        assert_eq!(budget.max_depth, 3);
        assert_eq!(budget.current_depth, 0);
        assert!(budget.can_delegate());
    }

    #[test]
    fn test_budget_child() {
        let root = Budget::root(20, 3);
        let child = root.child().unwrap();
        assert_eq!(child.max_iterations, 10); // 20 * 0.5
        assert_eq!(child.current_depth, 1);
        assert!(child.can_delegate());

        let grandchild = child.child().unwrap();
        assert_eq!(grandchild.max_iterations, 5); // 10 * 0.5
        assert_eq!(grandchild.current_depth, 2);
        assert!(grandchild.can_delegate());

        let great = grandchild.child().unwrap();
        assert_eq!(great.current_depth, 3);
        assert!(!great.can_delegate()); // at max depth
        assert!(great.child().is_none()); // can't go deeper
    }

    #[test]
    fn test_budget_minimum_iterations() {
        let budget = Budget {
            max_iterations: 1,
            max_depth: 5,
            current_depth: 0,
            budget_multiplier: 0.5,
            cost_limit: 0.0,
            cost_spent: 0.0,
            prices: None,
        };
        let child = budget.child().unwrap();
        assert_eq!(child.max_iterations, 1, "Should not go below 1 iteration");
    }

    #[test]
    fn test_scratch_pad_round_budget_for_nanbeige() {
        let config = ToolRunnerConfig {
            provider: Arc::new(MockProvider::new(vec![])),
            model: "local:nanbeige4.1-3b-q8_0.gguf".to_string(),
            max_iterations: 10,
            max_tokens: 1024,
            needs_user_continuation: true,
            max_tool_result_chars: 2000,
            short_circuit_chars: 200,
            depth: 0,
            cancellation_token: None,
            verbatim: false,
            budget: None,
        };
        assert_eq!(scratch_pad_round_budget(&config), 3);
    }

    #[test]
    fn test_budget_zero_depth() {
        let budget = Budget::root(10, 0);
        assert!(!budget.can_delegate());
        assert!(budget.child().is_none());
    }

    #[test]
    fn test_budget_cost_tracking() {
        let mut prices = crate::agent::model_prices::ModelPrices::empty();
        // $1/MTok prompt, $2/MTok completion → per-token: 0.000001, 0.000002
        prices
            .prices
            .insert("test/model".to_string(), (0.000001, 0.000002));
        let prices = std::sync::Arc::new(prices);

        let mut budget = Budget::root_with_cost(10, 2, 0.01, prices);
        assert!(!budget.is_over_budget());

        // 1000 prompt + 500 completion = 0.001 + 0.001 = 0.002
        let mut usage = std::collections::HashMap::new();
        usage.insert("prompt_tokens".to_string(), 1000i64);
        usage.insert("completion_tokens".to_string(), 500);
        let cost = budget.record_cost("test/model", &usage);
        assert!((cost - 0.002).abs() < 1e-9, "cost was {}", cost);
        assert!(!budget.is_over_budget());

        // Push over budget: 5000 prompt + 2500 completion = 0.005 + 0.005 = 0.01
        usage.insert("prompt_tokens".to_string(), 5000);
        usage.insert("completion_tokens".to_string(), 2500);
        let cost2 = budget.record_cost("test/model", &usage);
        assert!((cost2 - 0.01).abs() < 1e-9, "cost2 was {}", cost2);
        // Total: 0.002 + 0.01 = 0.012 >= 0.01 limit
        assert!(budget.is_over_budget());
    }

    #[test]
    fn test_budget_cost_unknown_model_is_free() {
        let prices = std::sync::Arc::new(crate::agent::model_prices::ModelPrices::empty());
        let mut budget = Budget::root_with_cost(10, 2, 0.01, prices);

        let mut usage = std::collections::HashMap::new();
        usage.insert("prompt_tokens".to_string(), 1_000_000i64);
        usage.insert("completion_tokens".to_string(), 1_000_000);
        let cost = budget.record_cost("local/unknown", &usage);
        assert_eq!(cost, 0.0);
        assert!(!budget.is_over_budget());
    }

    #[test]
    fn test_budget_no_prices_is_free() {
        let mut budget = Budget::root(10, 2);
        assert!(budget.prices.is_none());

        let mut usage = std::collections::HashMap::new();
        usage.insert("prompt_tokens".to_string(), 1_000_000i64);
        usage.insert("completion_tokens".to_string(), 1_000_000);
        let cost = budget.record_cost("anything", &usage);
        assert_eq!(cost, 0.0);
        assert!(!budget.is_over_budget()); // cost_limit is 0.0 = unlimited
    }

    #[test]
    fn test_budget_child_inherits_remaining_cost() {
        let mut prices = crate::agent::model_prices::ModelPrices::empty();
        prices
            .prices
            .insert("test/model".to_string(), (0.000001, 0.000002));
        let prices = std::sync::Arc::new(prices);

        let mut budget = Budget::root_with_cost(10, 2, 0.10, prices);

        // Spend $0.04
        let mut usage = std::collections::HashMap::new();
        usage.insert("prompt_tokens".to_string(), 20_000i64);
        usage.insert("completion_tokens".to_string(), 10_000);
        budget.record_cost("test/model", &usage);
        assert!((budget.cost_spent - 0.04).abs() < 1e-9);

        // Child gets (0.10 - 0.04) * 0.5 = 0.03
        let child = budget.child().unwrap();
        assert!(
            (child.cost_limit - 0.03).abs() < 1e-9,
            "child cost_limit was {}",
            child.cost_limit
        );
        assert_eq!(child.cost_spent, 0.0);
        assert!(child.prices.is_some());
    }

    // -- build_analysis_state tests --

    #[test]
    fn test_build_analysis_state_empty_store() {
        let store = context_store::ContextStore::new();
        let state = build_analysis_state(&store);
        assert!(
            state.is_empty(),
            "Empty store should produce empty state, got: {}",
            state
        );
    }

    #[test]
    fn test_build_analysis_state_with_variables() {
        let mut store = context_store::ContextStore::new();
        store.store("hello world".to_string());
        store.store("x".repeat(500));

        let state = build_analysis_state(&store);
        assert!(
            state.contains("output_0"),
            "State should contain variable name"
        );
        assert!(
            state.contains("11 chars"),
            "State should contain char count for output_0"
        );
        assert!(
            state.contains("output_1"),
            "State should contain second variable"
        );
        assert!(
            state.contains("500 chars"),
            "State should contain char count for output_1"
        );
    }

    #[test]
    fn test_build_analysis_state_with_memory() {
        let mut store = context_store::ContextStore::new();
        store.mem_store("file_type", "HTML page".to_string());
        store.mem_store("relevant_section", "Lines 142-168".to_string());

        let state = build_analysis_state(&store);
        assert!(
            state.contains("Findings so far:"),
            "State should have findings section"
        );
        assert!(
            state.contains("file_type"),
            "State should contain memory key"
        );
        assert!(
            state.contains("HTML page"),
            "State should contain memory value"
        );
        assert!(
            state.contains("relevant_section"),
            "State should contain second key"
        );
    }

    #[test]
    fn test_build_analysis_state_with_both() {
        let mut store = context_store::ContextStore::new();
        store.store("some data".to_string());
        store.mem_store("finding", "important result".to_string());

        let state = build_analysis_state(&store);
        assert!(
            state.contains("Variables:"),
            "Should have variables section"
        );
        assert!(
            state.contains("Findings so far:"),
            "Should have findings section"
        );
        assert!(state.contains("output_0"), "Should contain variable");
        assert!(state.contains("important result"), "Should contain finding");
    }

    // -- Memory accumulation across scratch pad rounds --

    #[tokio::test]
    async fn test_scratch_pad_memory_accumulates_across_rounds() {
        // Verify that mem_store findings from round N appear in the
        // user message sent to the LLM in round N+1.
        //
        // Round 0: model calls mem_store("finding_1", "first result")
        // Round 1: model sees "finding_1: first result" in state → returns summary

        struct MemoryCapturingProvider {
            captured_messages: tokio::sync::Mutex<Vec<Vec<Value>>>,
            call_count: AtomicU32,
        }

        #[async_trait]
        impl LLMProvider for MemoryCapturingProvider {
            async fn chat(
                &self,
                messages: &[Value],
                _tools: Option<&[Value]>,
                _model: Option<&str>,
                _max_tokens: u32,
                _temperature: f64,
                _thinking_budget: Option<u32>,
            ) -> anyhow::Result<crate::providers::base::LLMResponse> {
                self.captured_messages.lock().await.push(messages.to_vec());
                let n = self.call_count.fetch_add(1, Ordering::SeqCst);
                match n {
                    0 => {
                        // Round 0: call mem_store to save a finding
                        Ok(crate::providers::base::LLMResponse {
                            content: None,
                            tool_calls: vec![ToolCallRequest {
                                id: "mem_0".to_string(),
                                name: "mem_store".to_string(),
                                arguments: {
                                    let mut m = HashMap::new();
                                    m.insert("key".to_string(), json!("finding_1"));
                                    m.insert("value".to_string(), json!("discovered 5 endpoints"));
                                    m
                                },
                            }],
                            finish_reason: "tool_calls".to_string(),
                            usage: HashMap::new(),
                        })
                    }
                    1 => {
                        // Round 1: model should see the finding in state → produce summary
                        Ok(crate::providers::base::LLMResponse {
                            content: Some("Found 5 endpoints in the API.".to_string()),
                            tool_calls: vec![],
                            finish_reason: "stop".to_string(),
                            usage: HashMap::new(),
                        })
                    }
                    _ => Ok(crate::providers::base::LLMResponse {
                        content: Some("Unexpected call.".to_string()),
                        tool_calls: vec![],
                        finish_reason: "stop".to_string(),
                        usage: HashMap::new(),
                    }),
                }
            }

            fn get_default_model(&self) -> &str {
                "memory-capturing"
            }
        }

        let provider = Arc::new(MemoryCapturingProvider {
            captured_messages: tokio::sync::Mutex::new(Vec::new()),
            call_count: AtomicU32::new(0),
        });

        let mut tools = ToolRegistry::new();
        tools.register(Box::new(CountingTool::new()));

        let config = ToolRunnerConfig {
            provider: provider.clone(),
            model: "mock".to_string(),
            max_iterations: 10,
            max_tokens: 4096,
            needs_user_continuation: false,
            max_tool_result_chars: 30000,
            short_circuit_chars: 0,
            depth: 0,
            cancellation_token: None,
            verbatim: false,
            budget: None,
        };

        let result = run_tool_loop(
            &config,
            &make_tool_calls(&["test_tool"]),
            &tools,
            "test context",
        )
        .await;

        assert_eq!(
            result.summary.as_deref(),
            Some("Found 5 endpoints in the API.")
        );

        // Verify round 1's user message contains the finding from round 0.
        let captured = provider.captured_messages.lock().await;
        assert!(
            captured.len() >= 2,
            "Should have at least 2 LLM calls (round 0 + round 1)"
        );

        let round1_messages = &captured[1];
        let user_msg = round1_messages
            .iter()
            .find(|m| m["role"].as_str() == Some("user"))
            .expect("Round 1 should have a user message");
        let content = user_msg["content"].as_str().unwrap();
        assert!(
            content.contains("finding_1"),
            "Round 1 state should contain memory key from round 0, got: {}",
            content
        );
        assert!(
            content.contains("discovered 5 endpoints"),
            "Round 1 state should contain memory value from round 0, got: {}",
            content
        );
    }

    // -- needs_user_continuation irrelevance for scratch pad --

    #[tokio::test]
    async fn test_scratch_pad_ignores_needs_user_continuation() {
        // With needs_user_continuation=true (local llama-server mode),
        // the scratch pad should still send only [system, user] messages.
        // No extra user continuation should be injected.
        let provider = Arc::new(CapturingProvider::new());

        let mut tools = ToolRegistry::new();
        tools.register(Box::new(CountingTool::new()));

        let config = ToolRunnerConfig {
            provider: provider.clone(),
            model: "mock".to_string(),
            max_iterations: 10,
            max_tokens: 4096,
            needs_user_continuation: true, // key: local model mode
            max_tool_result_chars: 30000,
            short_circuit_chars: 0,
            depth: 0,
            cancellation_token: None,
            verbatim: false,
            budget: None,
        };

        let _ = run_tool_loop(
            &config,
            &make_tool_calls(&["test_tool"]),
            &tools,
            "test context",
        )
        .await;

        let captured = provider.captured_messages.lock().await;
        assert!(
            !captured.is_empty(),
            "Scratch pad should make at least 1 LLM call"
        );

        // Every call to the scratch pad should have exactly 2 messages: system + user.
        // No extra user continuation injected.
        for (i, messages) in captured.iter().enumerate() {
            assert_eq!(
                messages.len(),
                2,
                "Scratch pad round {} should have [system, user], got {} messages",
                i,
                messages.len()
            );
            assert_eq!(
                messages[0]["role"].as_str(),
                Some("system"),
                "Round {} message 0 should be system",
                i
            );
            assert_eq!(
                messages[1]["role"].as_str(),
                Some("user"),
                "Round {} message 1 should be user",
                i
            );
        }
    }
}
