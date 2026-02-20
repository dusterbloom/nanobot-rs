//! Base class for agent tools.

use std::collections::HashMap;

use async_trait::async_trait;
use tokio::sync::mpsc::UnboundedSender;

use crate::agent::audit::ToolEvent;

/// Structured outcome for a tool invocation.
#[derive(Debug, Clone)]
pub struct ToolExecutionResult {
    pub ok: bool,
    pub data: String,
    pub error: Option<String>,
    /// Structured error classification when available.
    pub error_kind: Option<crate::errors::ToolErrorKind>,
}

impl ToolExecutionResult {
    pub fn success(data: String) -> Self {
        Self {
            ok: true,
            data,
            error: None,
            error_kind: None,
        }
    }

    pub fn failure(message: String) -> Self {
        let error_kind = crate::errors::classify_tool_error(&message);
        Self {
            ok: false,
            data: format!("Error: {}", message),
            error: Some(message),
            error_kind,
        }
    }
}

/// Context passed to tools during execution for progress reporting
/// and cancellation support.
pub struct ToolExecutionContext {
    /// Channel for emitting progress events to the REPL.
    pub event_tx: UnboundedSender<ToolEvent>,
    /// Token that signals the tool should abort gracefully.
    pub cancellation_token: tokio_util::sync::CancellationToken,
    /// The tool call ID for correlating events.
    pub tool_call_id: String,
}

/// Abstract base trait for agent tools.
///
/// Tools are capabilities that the agent can use to interact with
/// the environment, such as reading files, executing commands, etc.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Tool name used in function calls.
    fn name(&self) -> &str;

    /// Description of what the tool does.
    fn description(&self) -> &str;

    /// JSON Schema for tool parameters.
    fn parameters(&self) -> serde_json::Value;

    /// Execute the tool with given parameters.
    ///
    /// Returns the result as a string.
    async fn execute(&self, params: HashMap<String, serde_json::Value>) -> String;

    /// Execute and return a structured outcome.
    ///
    /// Tools can override this to report explicit success/failure semantics.
    /// The default implementation keeps backward compatibility and maps
    /// `Error:`-prefixed outputs to failures.
    async fn execute_with_result(
        &self,
        params: HashMap<String, serde_json::Value>,
    ) -> ToolExecutionResult {
        let out = self.execute(params).await;
        if let Some(err) = out.strip_prefix("Error:").map(|s| s.trim().to_string()) {
            let error_kind = crate::errors::classify_tool_error(&err);
            ToolExecutionResult {
                ok: false,
                data: out,
                error: Some(err),
                error_kind,
            }
        } else {
            ToolExecutionResult::success(out)
        }
    }

    /// Execute the tool with an execution context for progress reporting
    /// and cancellation.
    ///
    /// The default implementation ignores the context and delegates to
    /// [`execute`]. Tools that support streaming (like ExecTool) override
    /// this to emit [`ToolEvent::Progress`] events and check the
    /// cancellation token.
    async fn execute_with_context(
        &self,
        params: HashMap<String, serde_json::Value>,
        _ctx: &ToolExecutionContext,
    ) -> String {
        self.execute(params).await
    }

    /// Like [`execute_with_result`] but with an execution context.
    ///
    /// Default delegates to [`execute_with_context`] and maps `Error:`-prefixed
    /// outputs to failures, same as [`execute_with_result`].
    async fn execute_with_result_and_context(
        &self,
        params: HashMap<String, serde_json::Value>,
        ctx: &ToolExecutionContext,
    ) -> ToolExecutionResult {
        let out = self.execute_with_context(params, ctx).await;
        if let Some(err) = out.strip_prefix("Error:").map(|s| s.trim().to_string()) {
            let error_kind = crate::errors::classify_tool_error(&err);
            ToolExecutionResult {
                ok: false,
                data: out,
                error: Some(err),
                error_kind,
            }
        } else {
            ToolExecutionResult::success(out)
        }
    }

    /// Convert tool to OpenAI function schema format.
    fn to_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "function",
            "function": {
                "name": self.name(),
                "description": self.description(),
                "parameters": self.parameters(),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A mock tool for testing the Tool trait and to_schema().
    struct MockTool;

    #[async_trait]
    impl Tool for MockTool {
        fn name(&self) -> &str {
            "mock_tool"
        }

        fn description(&self) -> &str {
            "A mock tool for testing"
        }

        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "Test input"
                    }
                },
                "required": ["input"]
            })
        }

        async fn execute(&self, params: HashMap<String, serde_json::Value>) -> String {
            let input = params
                .get("input")
                .and_then(|v| v.as_str())
                .unwrap_or("none");
            format!("executed with: {}", input)
        }
    }

    #[test]
    fn test_mock_tool_name() {
        let tool = MockTool;
        assert_eq!(tool.name(), "mock_tool");
    }

    #[test]
    fn test_mock_tool_description() {
        let tool = MockTool;
        assert_eq!(tool.description(), "A mock tool for testing");
    }

    #[test]
    fn test_mock_tool_parameters() {
        let tool = MockTool;
        let params = tool.parameters();
        assert_eq!(params["type"], "object");
        assert!(params["properties"]["input"].is_object());
        let required = params["required"].as_array().unwrap();
        assert_eq!(required.len(), 1);
        assert_eq!(required[0], "input");
    }

    #[test]
    fn test_to_schema_structure() {
        let tool = MockTool;
        let schema = tool.to_schema();

        assert_eq!(schema["type"], "function");
        assert!(schema["function"].is_object());
        assert_eq!(schema["function"]["name"], "mock_tool");
        assert_eq!(schema["function"]["description"], "A mock tool for testing");
        assert_eq!(schema["function"]["parameters"]["type"], "object");
    }

    #[test]
    fn test_to_schema_contains_all_fields() {
        let tool = MockTool;
        let schema = tool.to_schema();
        let function = &schema["function"];

        // Verify all expected keys are present.
        assert!(function.get("name").is_some());
        assert!(function.get("description").is_some());
        assert!(function.get("parameters").is_some());
    }

    #[tokio::test]
    async fn test_mock_tool_execute() {
        let tool = MockTool;
        let mut params = HashMap::new();
        params.insert(
            "input".to_string(),
            serde_json::Value::String("hello".to_string()),
        );
        let result = tool.execute(params).await;
        assert_eq!(result, "executed with: hello");
    }

    #[tokio::test]
    async fn test_mock_tool_execute_missing_param() {
        let tool = MockTool;
        let params = HashMap::new();
        let result = tool.execute(params).await;
        assert_eq!(result, "executed with: none");
    }

    #[tokio::test]
    async fn test_mock_tool_execute_with_result_success() {
        let tool = MockTool;
        let mut params = HashMap::new();
        params.insert(
            "input".to_string(),
            serde_json::Value::String("hello".to_string()),
        );
        let result = tool.execute_with_result(params).await;
        assert!(result.ok);
        assert_eq!(result.data, "executed with: hello");
        assert!(result.error.is_none());
    }

    #[tokio::test]
    async fn test_mock_tool_execute_with_result_error_prefix() {
        struct ErrorTool;

        #[async_trait]
        impl Tool for ErrorTool {
            fn name(&self) -> &str {
                "error_tool"
            }
            fn description(&self) -> &str {
                "Returns an error string"
            }
            fn parameters(&self) -> serde_json::Value {
                serde_json::json!({"type": "object", "properties": {}})
            }
            async fn execute(&self, _params: HashMap<String, serde_json::Value>) -> String {
                "Error: bad input".to_string()
            }
        }

        let tool = ErrorTool;
        let result = tool.execute_with_result(HashMap::new()).await;
        assert!(!result.ok);
        assert_eq!(result.data, "Error: bad input");
        assert_eq!(result.error.as_deref(), Some("bad input"));
    }

    #[tokio::test]
    async fn test_execute_with_context_default_delegates_to_execute() {
        use crate::agent::audit::ToolEvent;

        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel::<ToolEvent>();
        let token = tokio_util::sync::CancellationToken::new();
        let ctx = ToolExecutionContext {
            event_tx: tx,
            cancellation_token: token,
            tool_call_id: "call_1".to_string(),
        };

        let tool = MockTool;
        let mut params = HashMap::new();
        params.insert(
            "input".to_string(),
            serde_json::Value::String("hello".to_string()),
        );

        // execute_with_context should return same result as execute
        let result = tool.execute_with_context(params.clone(), &ctx).await;
        let direct = tool.execute(params).await;
        assert_eq!(result, direct);
        assert_eq!(result, "executed with: hello");
    }

    #[tokio::test]
    async fn test_execute_with_result_and_context_default() {
        use crate::agent::audit::ToolEvent;

        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel::<ToolEvent>();
        let token = tokio_util::sync::CancellationToken::new();
        let ctx = ToolExecutionContext {
            event_tx: tx,
            cancellation_token: token,
            tool_call_id: "call_1".to_string(),
        };

        let tool = MockTool;
        let mut params = HashMap::new();
        params.insert(
            "input".to_string(),
            serde_json::Value::String("test".to_string()),
        );

        let result = tool.execute_with_result_and_context(params, &ctx).await;
        assert!(result.ok);
        assert_eq!(result.data, "executed with: test");
    }

    #[test]
    fn test_tool_execution_context_construction() {
        use crate::agent::audit::ToolEvent;

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<ToolEvent>();
        let token = tokio_util::sync::CancellationToken::new();

        let ctx = ToolExecutionContext {
            event_tx: tx,
            cancellation_token: token.clone(),
            tool_call_id: "call_123".to_string(),
        };

        // Verify fields are accessible
        assert_eq!(ctx.tool_call_id, "call_123");
        assert!(!ctx.cancellation_token.is_cancelled());

        // Can send events through the channel
        ctx.event_tx
            .send(ToolEvent::Progress {
                tool_name: "exec".to_string(),
                tool_call_id: "call_123".to_string(),
                elapsed_ms: 1000,
                output_preview: None,
            })
            .unwrap();

        let event = rx.try_recv().unwrap();
        match event {
            ToolEvent::Progress { elapsed_ms, .. } => assert_eq!(elapsed_ms, 1000),
            _ => panic!("Expected Progress"),
        }
    }

    #[test]
    fn test_cancellation_token_in_context() {
        use crate::agent::audit::ToolEvent;

        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel::<ToolEvent>();
        let token = tokio_util::sync::CancellationToken::new();

        let ctx = ToolExecutionContext {
            event_tx: tx,
            cancellation_token: token.clone(),
            tool_call_id: "call_456".to_string(),
        };

        assert!(!ctx.cancellation_token.is_cancelled());
        token.cancel();
        assert!(ctx.cancellation_token.is_cancelled());
    }
}
