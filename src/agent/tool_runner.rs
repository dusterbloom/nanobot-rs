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

    // Build a mini message history for the cheap model.
    let system_msg = json!({
        "role": "system",
        "content": format!(
            "You are a tool execution assistant. Execute tools to fulfill the user's request. \
             When all needed information has been gathered, respond with a structured summary:\n\
             - List key findings and data points (not just 'I read the file')\n\
             - Include relevant values, names, counts, or excerpts\n\
             - Be specific: 'Found 3 JSON files, config.json has 12 keys including apiKey' \
             not 'I found some files'\n\
             Do not ask the user questions — just execute tools and report results.\n\n\
             Context: {}",
            system_context
        )
    });
    let mut messages: Vec<Value> = vec![system_msg];

    // Anthropic requires conversations to start with a user message.
    // Add the original user request so the tool runner has context.
    messages.push(json!({
        "role": "user",
        "content": system_context
    }));

    // Get tool definitions for the cheap model.
    let tool_defs = tools.get_definitions();
    let tool_defs_opt: Option<&[Value]> = if tool_defs.is_empty() {
        None
    } else {
        Some(&tool_defs)
    };

    // Execute the initial tool calls from the main model.
    let mut pending_calls: Vec<ToolCallRequest> = initial_tool_calls.to_vec();

    for iteration in 0..config.max_iterations {
        iterations_used = iteration + 1;

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

        // Execute each tool call — store results with ORIGINAL IDs.
        for tc in &pending_calls {
            debug!("Tool runner executing: {} (id: {})", tc.name, tc.id);
            let result = tools.execute(&tc.name, tc.arguments.clone()).await;
            ContextBuilder::add_tool_result(&mut messages, &tc.id, &tc.name, &result.data);
            let original_id = id_map.get(&tc.id).cloned().unwrap_or_else(|| tc.id.clone());
            all_results.push((original_id, tc.name.clone(), result.data));
        }

        // Local models (llama-server) require conversations to end with
        // a user message. Mistral/Ministral handle tool→generate natively
        // and break if a user message is injected here.
        if config.needs_user_continuation {
            messages.push(json!({
                "role": "user",
                "content": "Based on the tool results above, decide: do you need to call more tools, or can you provide a summary of what was found?"
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

        if response.has_tool_calls() {
            pending_calls = response.tool_calls;
            // If the model also produced text alongside tool calls, ignore it for now.
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
        // Provider always returns more tool calls — should stop at max.
        let mut responses = Vec::new();
        for i in 0..20 {
            responses.push(crate::providers::base::LLMResponse {
                content: None,
                tool_calls: vec![ToolCallRequest {
                    id: format!("chain_{}", i),
                    name: "test_tool".to_string(),
                    arguments: {
                        let mut m = HashMap::new();
                        m.insert("query".to_string(), json!("chained"));
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
}
