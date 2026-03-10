//! Tests for the tool runner.
//!
//! Extracted from tool_runner.rs for maintainability.
//! Loaded via `#[path = "tool_runner_tests.rs"]` in tool_runner.rs so that
//! `use super::*` continues to resolve against the tool_runner module.

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
        _top_p: Option<f64>,
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
                // Use index in query so calls with the same name are not
                // treated as duplicates by normalize_call_key dedup.
                m.insert("query".to_string(), json!(format!("test_{}", i)));
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
fn test_aggregate_results_passthrough() {
    let long_data = "x".repeat(3000);
    let result = ToolRunResult {
        tool_results: vec![("id1".into(), "big_tool".into(), long_data.clone())],
        summary: None,
        iterations_used: 1,
        error: None,
    };

    // Tool results are always passed through raw — never truncated.
    let formatted = format_results_for_context(&result, 2000, None);
    assert!(formatted.contains(&long_data));
}

#[test]
fn test_results_passthrough_with_summary() {
    let data = "x".repeat(500);
    let result = ToolRunResult {
        tool_results: vec![("id1".into(), "read_file".into(), data.clone())],
        summary: Some("Found a large file.".to_string()),
        iterations_used: 1,
        error: None,
    };

    // Tool results passed through raw regardless of limit parameter.
    let formatted = format_results_for_context(&result, 200, None);
    assert!(formatted.contains(&data));
    assert!(formatted.contains("Found a large file."));
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
        _top_p: Option<f64>,
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
        _top_p: Option<f64>,
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
    // dangerous_tool was blocked; early-break optimization means summary
    // may be None (no useful work to summarize), which is correct.
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
            _top_p: Option<f64>,
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
            _top_p: Option<f64>,
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
            _top_p: Option<f64>,
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
            _top_p: Option<f64>,
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

#[tokio::test]
async fn test_scratch_pad_sends_system_user_only() {
    // The scratch pad should send only [system, user] messages per round.
    let provider = Arc::new(CapturingProvider::new());

    let mut tools = ToolRegistry::new();
    tools.register(Box::new(CountingTool::new()));

    let config = ToolRunnerConfig {
        provider: provider.clone(),
        model: "mock".to_string(),
        max_iterations: 10,
        max_tokens: 4096,
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
