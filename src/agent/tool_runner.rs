//! Delegated tool execution loop.
//!
//! Receives initial tool calls from the main LLM, executes them via the
//! shared [`ToolRegistry`], and lets a cheap model decide if more tools are
//! needed. Returns aggregated results for injection into the main context.

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

    // Build a mini message history for the cheap model.
    let system_msg = json!({
        "role": "system",
        "content": format!(
            "You are a tool execution assistant. Execute tools to fulfill the user's request. \
             When all needed information has been gathered, respond with a brief summary. \
             Do not ask the user questions — just execute tools and report results.\n\n\
             Context: {}",
            system_context
        )
    });
    let mut messages: Vec<Value> = vec![system_msg];

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

        // Build assistant message with tool_calls.
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

        // Execute each tool call.
        for tc in &pending_calls {
            debug!("Tool runner executing: {} (id: {})", tc.name, tc.id);
            let result = tools.execute(&tc.name, tc.arguments.clone()).await;
            ContextBuilder::add_tool_result(&mut messages, &tc.id, &tc.name, &result.data);
            all_results.push((tc.id.clone(), tc.name.clone(), result.data));
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
pub fn format_results_for_context(result: &ToolRunResult) -> String {
    let mut parts: Vec<String> = Vec::new();

    for (_, tool_name, data) in &result.tool_results {
        // Truncate very long results.
        let truncated = if data.len() > 2000 {
            format!("{}... (truncated)", &data[..2000])
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

        let formatted = format_results_for_context(&result);
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

        let formatted = format_results_for_context(&result);
        assert!(formatted.contains("(truncated)"));
        assert!(formatted.len() < 3000);
    }
}
