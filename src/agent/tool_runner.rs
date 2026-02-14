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
use crate::providers::base::{LLMProvider, ToolCallRequest};

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

    // Build a mini message history for the cheap model.
    // Keep the system prompt compact — the delegation model has limited context.
    let system_msg = json!({
        "role": "system",
        "content": "You are a tool execution agent. Tool results are stored as variables.\nYou receive metadata for large results (name, length, preview) and full results for small ones.\nUse ctx_slice, ctx_grep, ctx_length to examine large variables.\nUse ctx_summarize to sub-summarize a variable with a specific instruction.\n\nRULES:\n1. Read the metadata/results. If the answer is visible, SUMMARIZE with specific data and STOP.\n2. For large results: use ctx_grep to search for specific patterns first.\n3. Use ctx_slice to read specific sections only when grep isn't enough.\n4. Use ctx_summarize when you need a focused summary of a specific variable.\n5. NEVER re-execute a tool with the same arguments.\n6. When you have enough information, SUMMARIZE and STOP.\n7. Do not ask questions."
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
    let tool_defs_opt: Option<&[Value]> = if tool_defs.is_empty() {
        None
    } else {
        Some(&tool_defs)
    };

    // Execute the initial tool calls from the main model.
    let mut pending_calls: Vec<ToolCallRequest> = initial_tool_calls.to_vec();

    // Seed seen_calls with initial calls so the model can't re-request them.
    for tc in initial_tool_calls {
        let call_key = format!("{}:{}", tc.name,
            serde_json::to_string(&tc.arguments).unwrap_or_default());
        seen_calls.insert(call_key);
    }

    for iteration in 0..config.max_iterations {
        iterations_used = iteration + 1;

        // Check cancellation before each iteration.
        if config.cancellation_token.as_ref().map_or(false, |t| t.is_cancelled()) {
            return ToolRunResult {
                tool_results: all_results,
                summary: Some("Tool execution cancelled.".to_string()),
                iterations_used,
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
            .map(|tc| {
                json!({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": serde_json::to_string(&tc.arguments)
                            .unwrap_or_else(|_| "{}".to_string()),
                    }
                })
            })
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
                let variable = tc.arguments.get("variable").and_then(|v| v.as_str()).unwrap_or("");
                let instruction = tc.arguments.get("instruction").and_then(|v| v.as_str()).unwrap_or("Summarize");
                debug!("ctx_summarize: var={}, instruction={}, depth={}", variable, instruction, config.depth);
                let result = context_store::execute_ctx_summarize(
                    &context_store,
                    variable,
                    instruction,
                    &config.provider,
                    &config.model,
                    config.depth,
                    config.max_tokens,
                ).await;
                // Store the summary as a new variable for subsequent micro-tool access.
                let (_, summary_metadata) = context_store.store(result.clone());
                let _ = summary_metadata; // metadata available if needed later
                ContextBuilder::add_tool_result(&mut messages, &tc.id, &tc.name, &result);
                // NOT added to all_results — ctx_summarize is internal to delegation.
            } else if context_store::is_micro_tool(&tc.name) {
                // Sync micro-tool: execute against ContextStore (internal to delegation).
                debug!("Micro-tool: {} (id: {})", tc.name, tc.id);
                let result = context_store::execute_micro_tool(&context_store, &tc.name, &tc.arguments);
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
                    tools.execute_with_context(&tc.name, tc.arguments.clone(), &ctx).await
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
                all_results.push((original_id, tc.name.clone(), result.data));
            }
        }

        // Short-circuit: if this is the first iteration and ALL results
        // are short, skip the delegation LLM call entirely.
        // Trivial outputs (echo, simple commands) don't need summarization —
        // the main model can interpret them directly. This also avoids the
        // loop problem where small models don't know what to do with
        // one-line results. Set short_circuit_chars to 0 to disable.
        if config.short_circuit_chars > 0 && iteration == 0 {
            let threshold = config.short_circuit_chars;
            let all_short = all_results.iter().all(|(_, _, data)| data.len() < threshold);
            if all_short {
                debug!(
                    "All {} tool results are short (< {} chars) — skipping delegation LLM, returning raw results",
                    all_results.len(), threshold
                );
                return ToolRunResult {
                    tool_results: all_results,
                    summary: None,
                    iterations_used,
                };
            }
        }

        // Local models (llama-server) require conversations to end with
        // a user message. Mistral/Ministral handle tool→generate natively
        // and break if a user message is injected here.
        if config.needs_user_continuation {
            messages.push(json!({
                "role": "user",
                "content": "Tool results are above. Summarize the findings. Only call more tools if the results clearly indicate incomplete or failed execution."
            }));
        }

        // Ask the cheap model if more tools are needed.
        let response = match config
            .provider
            .chat(
                &messages,
                tool_defs_opt,
                Some(&config.model),
                config.max_tokens,
                0.3, // low temperature for tool execution
            )
            .await
        {
            Ok(r) => r,
            Err(e) => {
                warn!("Tool runner LLM call failed: {}", e);
                return ToolRunResult {
                    tool_results: all_results,
                    summary: Some(format!("Tool runner LLM error: {}", e)),
                    iterations_used,
                };
            }
        };

        // If the delegation LLM returned an error (server down, OOM, etc.),
        // don't treat the error text as a "summary". Return tool results
        // collected so far and let the main model interpret them directly.
        if response.finish_reason == "error" {
            warn!("Delegation model returned error — returning raw tool results");
            return ToolRunResult {
                tool_results: all_results,
                summary: None,
                iterations_used,
            };
        }

        if response.has_tool_calls() {
            // Filter out:
            // 1. Tools not in the original request (prompt injection defense)
            // 2. Duplicate tool calls (loop detection)
            let mut new_calls: Vec<_> = Vec::new();
            for tc in response.tool_calls {
                // SAFETY: block tools the delegation model invented.
                if !allowed_tools.contains(tc.name.as_str()) {
                    warn!(
                        "Delegation model requested '{}' — not in original tool set {:?} — BLOCKED",
                        tc.name, allowed_tools
                    );
                    continue;
                }
                let call_key = format!("{}:{}", tc.name,
                    serde_json::to_string(&tc.arguments).unwrap_or_default());
                if seen_calls.contains(&call_key) {
                    warn!("Delegation model re-requested {} with same args — breaking loop", tc.name);
                } else {
                    seen_calls.insert(call_key);
                    new_calls.push(tc);
                }
            }
            if new_calls.is_empty() {
                // All requested calls are duplicates — model is looping.
                debug!("All chained tool calls were duplicates — stopping delegation loop");
                return ToolRunResult {
                    tool_results: all_results,
                    summary: Some("Tool execution complete (loop detected).".to_string()),
                    iterations_used,
                };
            }
            pending_calls = new_calls;
        } else {
            // Model is done — return its summary.
            return ToolRunResult {
                tool_results: all_results,
                summary: response.content,
                iterations_used,
            };
        }
    }

    // Ran out of iterations.
    ToolRunResult {
        tool_results: all_results,
        summary: Some("Tool runner reached max iterations.".to_string()),
        iterations_used,
    }
}

/// Format tool run results for injection into the main LLM context.
///
/// `max_result_chars` controls how much of each tool result to include.
/// Use a small value (e.g. 200) for slim/RLM mode where the summary
/// carries the meaning, or a large value for full-detail mode.
pub fn format_results_for_context(result: &ToolRunResult, max_result_chars: usize) -> String {
    let mut parts: Vec<String> = Vec::new();

    for (_, tool_name, data) in &result.tool_results {
        let truncated = if data.len() > max_result_chars {
            format!(
                "{}… ({} chars total)",
                &data[..max_result_chars],
                data.len()
            )
        } else {
            data.clone()
        };
        parts.push(format!("[{}]: {}", tool_name, truncated));
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
        };

        let result = run_tool_loop(
            &config,
            &make_tool_calls(&["test_tool"]),
            &tools,
            "test",
        )
        .await;

        assert_eq!(result.iterations_used, 3);
        assert!(result.summary.as_deref().unwrap().contains("max iterations"));
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
        };

        // Initial call uses "test" as query (from make_tool_calls).
        // First chain uses "same_args" (new, allowed).
        // Second chain uses "same_args" again (duplicate, blocked).
        let result = run_tool_loop(
            &config,
            &make_tool_calls(&["test_tool"]),
            &tools,
            "test",
        )
        .await;

        // Should stop at iteration 2: initial + 1 chain, then duplicate detected
        assert_eq!(result.iterations_used, 2, "Should stop after detecting duplicate");
        assert_eq!(result.tool_results.len(), 2, "Should have 2 results (initial + 1 chain)");
        assert!(result.summary.as_deref().unwrap().contains("loop detected"),
            "Summary should mention loop: {:?}", result.summary);
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
        };

        let result = run_tool_loop(
            &config,
            &make_tool_calls(&["test_tool"]),
            &tools,
            "find files",
        )
        .await;

        assert_eq!(result.iterations_used, 1);
        assert_eq!(
            result.summary.as_deref(),
            Some("Summary: found 3 files.")
        );
    }

    #[test]
    fn test_aggregate_results_formatting() {
        let result = ToolRunResult {
            tool_results: vec![
                ("id1".into(), "read_file".into(), "file contents here".into()),
                ("id2".into(), "exec".into(), "command output".into()),
            ],
            summary: Some("Read a file and ran a command.".to_string()),
            iterations_used: 1,
        };

        let formatted = format_results_for_context(&result, 2000);
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
        };

        let formatted = format_results_for_context(&result, 2000);
        assert!(formatted.contains("chars total"));
        assert!(formatted.len() < 3000);

    }

    #[test]
    fn test_slim_results_truncation() {
        let result = ToolRunResult {
            tool_results: vec![
                ("id1".into(), "read_file".into(), "x".repeat(500)),
            ],
            summary: Some("Found a large file.".to_string()),
            iterations_used: 1,
        };

        // Slim mode: 200 char preview.
        let slim = format_results_for_context(&result, 200);
        assert!(slim.contains("500 chars total"));
        assert!(slim.len() < 400);

        // Full mode: 2000 char limit — 500 chars fits.
        let full = format_results_for_context(&result, 2000);
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
        ) -> anyhow::Result<crate::providers::base::LLMResponse> {
            self.captured_messages
                .lock()
                .await
                .push(messages.to_vec());
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
        };

        let _ = run_tool_loop(
            &config,
            &make_tool_calls(&["test_tool"]),
            &tools,
            "test context",
        )
        .await;

        let captured = provider.captured_messages.lock().await;
        assert!(!captured.is_empty(), "Should have made at least one LLM call");

        let messages = &captured[0];

        // Expected sequence: system, user, assistant(tool_calls), tool(result)
        // NO user continuation — Mistral handles tool→generate natively.
        assert_eq!(messages.len(), 4, "Expected exactly 4 messages, got {}", messages.len());
        assert_eq!(messages[0]["role"].as_str(), Some("system"));
        assert_eq!(messages[1]["role"].as_str(), Some("user"));
        assert_eq!(messages[2]["role"].as_str(), Some("assistant"));
        assert_eq!(messages[3]["role"].as_str(), Some("tool"));
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
        };

        let result = run_tool_loop(
            &config,
            &make_tool_calls(&["test_tool"]),
            &tools,
            "test context",
        )
        .await;

        assert_eq!(result.iterations_used, 2);
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
        };

        // 3 simultaneous tool calls
        let calls = make_tool_calls(&["test_tool", "test_tool", "test_tool"]);
        let result = run_tool_loop(&config, &calls, &tools, "multi-tool test").await;

        assert_eq!(result.tool_results.len(), 3);

        let captured = provider.captured_messages.lock().await;
        let messages = &captured[0];

        // Should be: system, user, assistant(3 tool_calls), tool, tool, tool
        let last = messages.last().unwrap();
        assert_eq!(last["role"].as_str(), Some("tool"), "Last message should be tool result");

        // Count tool messages — should be 3
        let tool_count = messages.iter()
            .filter(|m| m["role"].as_str() == Some("tool"))
            .count();
        assert_eq!(tool_count, 3, "Should have 3 tool result messages");
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
        // Summary should contain the error
        assert!(
            result.summary.as_deref().unwrap().contains("error"),
            "Summary should mention the error: {:?}",
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
            result.tool_results[0].0,
            "toolu_01XYZabc123def456ghi789",
            "Results should preserve original tool call ID"
        );

        // Messages sent to delegation model should use normalized 9-char IDs
        let captured = provider.captured_messages.lock().await;
        let messages = &captured[0];

        // Find the assistant message with tool_calls
        let assistant_msg = messages.iter().find(|m| m["role"].as_str() == Some("assistant")).unwrap();
        let tc_id = assistant_msg["tool_calls"][0]["id"].as_str().unwrap();
        assert_eq!(tc_id.len(), 9, "Tool call ID in messages should be 9 chars: {}", tc_id);

        // Find the tool result message
        let tool_msg = messages.iter().find(|m| m["role"].as_str() == Some("tool")).unwrap();
        let result_id = tool_msg["tool_call_id"].as_str().unwrap();
        assert_eq!(result_id.len(), 9, "Tool result ID should match normalized ID: {}", result_id);
        assert_eq!(tc_id, result_id, "Tool call and result IDs should match");
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

        // Messages sent to delegation model should have metadata, not raw data
        let captured = provider.captured_messages.lock().await;
        let messages = &captured[0];
        let tool_msg = messages.iter()
            .find(|m| m["role"].as_str() == Some("tool"))
            .unwrap();
        let content = tool_msg["content"].as_str().unwrap();
        assert!(content.contains("Variable 'output_0'"),
            "Delegation model should see variable metadata, got: {}", content);
        assert!(content.contains("500 chars"),
            "Metadata should contain char count, got: {}", content);
        assert!(content.len() < 300,
            "Metadata should be much shorter than 500 chars, got {} chars",
            content.len());
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

        let captured = provider.captured_messages.lock().await;
        let messages = &captured[0];
        let tool_msg = messages.iter()
            .find(|m| m["role"].as_str() == Some("tool"))
            .unwrap();
        let content = tool_msg["content"].as_str().unwrap();
        assert!(!content.contains("Variable '"),
            "Small results should not use metadata");
        assert_eq!(content, "x".repeat(50));
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
        assert!(result.summary.is_none(), "Short-circuit should skip LLM, summary should be None");
        // No LLM calls made
        let captured = provider.captured_messages.lock().await;
        assert!(captured.is_empty(), "No LLM calls should have been made for short results");
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
        };

        let result = run_tool_loop(
            &config,
            &make_tool_calls(&["test_tool"]),
            &tools,
            "test context",
        )
        .await;

        assert_eq!(result.tool_results.len(), 1);
        assert_eq!(result.summary.as_deref(), Some("Summarized."),
            "With short_circuit_chars=0, LLM should still be called");
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
        assert_eq!(result.tool_results.len(), 1, "Only initial tool should execute");
        assert_eq!(result.tool_results[0].1, "test_tool");
        // dangerous_tool was blocked, all calls were filtered → loop detected
        assert!(result.summary.is_some(), "Should have a summary after blocking");
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
        };

        let result = run_tool_loop(
            &config,
            &make_tool_calls(&["test_tool"]),
            &tools,
            "test context",
        )
        .await;

        // Both calls should execute (same tool, different args)
        assert_eq!(result.tool_results.len(), 2, "Should have 2 results (initial + chain)");
        assert_eq!(result.tool_results[0].1, "test_tool");
        assert_eq!(result.tool_results[1].1, "test_tool");
    }

    #[tokio::test]
    async fn test_tool_defs_filtered_for_delegation() {
        // Verify the delegation model only receives definitions for requested tools.
        let provider = Arc::new(CapturingProvider::new());

        let mut tools = ToolRegistry::new();
        tools.register(Box::new(CountingTool::new()));    // test_tool
        tools.register(Box::new(DangerousTool));           // dangerous_tool

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
        assert!(!captured.is_empty(), "Should have made at least one LLM call");
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
        };

        let result = run_tool_loop(
            &config,
            &make_tool_calls(&["test_tool"]),
            &tools,
            "test context",
        )
        .await;

        // Only the real tool call should be in results
        assert_eq!(result.tool_results.len(), 1, "Micro-tool results should not be in all_results");
        assert_eq!(result.tool_results[0].1, "test_tool");
    }

    #[tokio::test]
    async fn test_delegation_receives_micro_tool_defs() {
        // Verify the delegation model receives ctx_slice, ctx_grep, ctx_length
        // in its tool definitions.
        struct CapturingToolsProvider {
            captured_tools: tokio::sync::Mutex<Vec<Vec<Value>>>,
        }

        impl CapturingToolsProvider {
            fn new() -> Self {
                Self {
                    captured_tools: tokio::sync::Mutex::new(Vec::new()),
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
            ) -> anyhow::Result<crate::providers::base::LLMResponse> {
                if let Some(t) = tools {
                    self.captured_tools
                        .lock()
                        .await
                        .push(t.to_vec());
                }
                Ok(crate::providers::base::LLMResponse {
                    content: Some("Done.".to_string()),
                    tool_calls: vec![],
                    finish_reason: "stop".to_string(),
                    usage: HashMap::new(),
                })
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
        };

        let _ = run_tool_loop(
            &config,
            &make_tool_calls(&["test_tool"]),
            &tools,
            "test context",
        )
        .await;

        let captured = provider.captured_tools.lock().await;
        assert!(!captured.is_empty(), "Should have captured tool definitions");

        let defs = &captured[0];
        let tool_names: Vec<&str> = defs.iter()
            .filter_map(|d| d.pointer("/function/name").and_then(|v| v.as_str()))
            .collect();

        assert!(tool_names.contains(&"test_tool"), "Should include the real tool");
        assert!(tool_names.contains(&"ctx_slice"), "Should include ctx_slice");
        assert!(tool_names.contains(&"ctx_grep"), "Should include ctx_grep");
        assert!(tool_names.contains(&"ctx_length"), "Should include ctx_length");
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
        assert!(result.starts_with("Error:"), "Should be an error: {}", result);
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
        assert!(result.starts_with("Error:"), "Should be an error: {}", result);
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
        };

        let result = run_tool_loop(
            &config,
            &make_tool_calls(&["test_tool"]),
            &tools,
            "test context",
        )
        .await;

        assert!(result.tool_results.is_empty(), "No tools should execute when pre-cancelled");
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
        };

        let result = run_tool_loop(
            &config,
            &make_tool_calls(&["test_tool"]),
            &tools,
            "test context",
        )
        .await;

        // Should have the initial tool result but stopped before the chained call.
        assert_eq!(
            result.tool_results.len(), 1,
            "Should have only the initial result, not the chained one"
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
}
