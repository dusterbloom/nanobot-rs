//! Base class for agent tools.

use std::collections::HashMap;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::UnboundedSender;

use crate::agent::audit::ToolEvent;

/// Extract a required string parameter or early-return the canonical error.
///
/// `require_str!(params, "key")`          → `Error: 'key' parameter is required`
/// `require_str!(params, "key", " for X")`→ `Error: 'key' parameter is required for X`
/// `require_str!(params, "key", ".")`     → `Error: 'key' parameter is required.`
///
/// The suffix is appended verbatim so every pre-existing error string
/// (tests assert exact substrings) is preserved byte-for-byte.
macro_rules! require_str {
    ($params:expr, $key:literal) => {
        require_str!($params, $key, "")
    };
    ($params:expr, $key:literal, $suffix:literal) => {
        match $params.get($key).and_then(|v| v.as_str()) {
            Some(v) => v,
            None => {
                return concat!("Error: '", $key, "' parameter is required", $suffix).to_string()
            }
        }
    };
}
pub(crate) use require_str;

/// Permission level required to execute a tool.
///
/// Ordered from least to most privileged. A registry's `max_permission`
/// ceiling blocks any tool whose level exceeds it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum PermissionLevel {
    /// Read-only tools: read_file, list_dir, find_files, file_info,
    /// workspace_diff, system_info, recall, get_skills, session_search
    ReadOnly,
    /// Network access: web_search, web_fetch, browser
    Network,
    /// Write tools: write_file, edit_file, remember
    Write,
    /// Code execution: exec, execute_code
    Execute,
    /// System-level: spawn, message, cron_schedule, send_email
    System,
}

/// Whether a tool may overlap with adjacent calls from the same assistant
/// response. Defaulting to sequential keeps new and stateful tools safe until
/// their implementations are explicitly audited.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolConcurrency {
    Sequential,
    ParallelSafe,
}

/// Structured outcome for a tool invocation.
///
/// Invariant: `ok == true` implies `error.is_none()`, and every failure
/// carries a model-visible `Error: ...` string in `data`. The fields are
/// private so the invariant cannot be violated from outside this module —
/// the only ways to build a value are the constructors (all `#[must_use]`)
/// and the `From<ToolResult>` conversion introduced with the typed error
/// protocol. Readers use the accessors.
#[derive(Debug, Clone)]
pub struct ToolExecutionResult {
    ok: bool,
    data: String,
    error: Option<String>,
    /// Structured error classification when available.
    error_kind: Option<crate::errors::ToolErrorKind>,
}

impl ToolExecutionResult {
    /// Whether the tool call succeeded.
    #[must_use]
    pub fn ok(&self) -> bool {
        self.ok
    }

    /// The model-facing output text. On failure this carries the rendered
    /// `Error: ...` wire string.
    #[must_use]
    pub fn data(&self) -> &str {
        &self.data
    }

    /// The failure detail, when this is a failure.
    #[must_use]
    pub fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }

    /// The structured error classification, when known.
    #[must_use]
    pub fn error_kind(&self) -> Option<&crate::errors::ToolErrorKind> {
        self.error_kind.as_ref()
    }

    #[must_use]
    pub fn success(data: String) -> Self {
        Self {
            ok: true,
            data,
            error: None,
            error_kind: None,
        }
    }

    /// Whether this result represents a retryable (transient) error.
    #[must_use]
    pub fn is_retryable(&self) -> bool {
        self.error_kind.as_ref().map_or(false, |k| k.is_retryable())
    }

    /// Build a failure from a message plus a structural classification.
    ///
    /// The `data` is the full model-visible string (already `Error:`-prefixed);
    /// `error_kind` is produced at the source instead of by substring
    /// classification. Transitional: used by the legacy registry
    /// example-append path until Phase 3 deletes the legacy channel.
    #[must_use]
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn failure_with_kind(data: String, error_kind: crate::errors::ToolErrorKind) -> Self {
        Self {
            ok: false,
            data,
            error: None,
            error_kind: Some(error_kind),
        }
    }

    #[must_use]
    pub fn failure(message: String) -> Self {
        let error_kind = crate::errors::classify_tool_error(&message);
        Self {
            ok: false,
            data: format!("Error: {}", message),
            error: Some(message),
            error_kind,
        }
    }

    /// Append a corrective worked example to a failure's model-visible text
    /// (the registry's MissingArg path: `"... is required. Call as X."`).
    /// Only meaningful on failures.
    pub fn append_worked_example(&mut self, example: &str) {
        let base = self.data.trim_end_matches('.');
        self.data = format!("{}. Call as {}.", base, example);
    }
}

/// The typed success payload of a tool call (error protocol, §2.2).
///
/// A plain struct rather than `String` so the registry can attach audit
/// metadata later without a breaking change.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolOutput {
    /// The model-facing text. May carry `TOOL_RESULT_HANDLE v1` receipts,
    /// `[truncated: …]` markers, or raw output.
    pub text: String,
}

/// The one result type of the tool layer.
pub type ToolResult = Result<ToolOutput, crate::errors::ToolError>;

impl From<ToolResult> for ToolExecutionResult {
    fn from(r: ToolResult) -> Self {
        match r {
            Ok(out) => ToolExecutionResult {
                ok: true,
                data: out.text,
                error: None,
                error_kind: None,
            },
            Err(e) => ToolExecutionResult {
                ok: false,
                data: e.render(),
                error: Some(e.to_string()),
                error_kind: crate::errors::legacy_kind_from_tool_error(&e),
            },
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

    /// Execute and return the typed outcome (error protocol, Phase 1 —
    /// additive). This is the migration seam: tools land on `ToolResult` by
    /// overriding this method and building [`crate::errors::ToolError`]
    /// variants at their failure sites.
    ///
    /// The default implementation funnels the legacy `String` channel through
    /// [`crate::errors::ToolError::from_legacy`] so every unmigrated tool
    /// keeps working — and keeps producing byte-identical `Error: ...` wire
    /// strings — with no changes.
    async fn execute_typed(
        &self,
        params: HashMap<String, serde_json::Value>,
        ctx: &ToolExecutionContext,
    ) -> ToolResult {
        let out = self.execute_with_context(params, ctx).await;
        if let Some(err) = out.strip_prefix("Error:").map(|s| s.trim().to_string()) {
            Err(crate::errors::ToolError::from_legacy(&err))
        } else {
            Ok(ToolOutput { text: out })
        }
    }

    /// Execute and return a structured outcome.
    ///
    /// Default renders the typed [`execute_typed`] result into the legacy
    /// structured shape (`From<ToolResult> for ToolExecutionResult`).
    async fn execute_with_result(
        &self,
        params: HashMap<String, serde_json::Value>,
    ) -> ToolExecutionResult {
        let (event_tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let ctx = ToolExecutionContext {
            event_tx,
            cancellation_token: tokio_util::sync::CancellationToken::new(),
            tool_call_id: String::new(),
        };
        ToolExecutionResult::from(self.execute_typed(params, &ctx).await)
    }

    /// Like [`execute_with_result`] but with an execution context.
    ///
    /// Default renders the typed [`execute_typed`] result into the legacy
    /// structured shape.
    async fn execute_with_result_and_context(
        &self,
        params: HashMap<String, serde_json::Value>,
        ctx: &ToolExecutionContext,
    ) -> ToolExecutionResult {
        ToolExecutionResult::from(self.execute_typed(params, ctx).await)
    }

    /// The permission level required to execute this tool.
    ///
    /// The registry checks this against its `max_permission` ceiling before
    /// executing. Default is `ReadOnly` (least privileged).
    fn permission(&self) -> PermissionLevel {
        PermissionLevel::ReadOnly
    }

    /// Execution policy for adjacent tool calls in one assistant response.
    fn concurrency(&self) -> ToolConcurrency {
        ToolConcurrency::Sequential
    }

    /// Whether this tool is currently available for use.
    ///
    /// Tools can return `false` when their required backend is not configured.
    /// Unavailable tools are excluded from `get_definitions()` so the LLM
    /// never sees them, but they can still be executed directly if needed.
    ///
    /// Default implementation always returns `true`.
    fn is_available(&self) -> bool {
        true
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
        assert!(result.ok());
        assert_eq!(result.data(), "executed with: hello");
        assert!(result.error().is_none());
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
        assert!(!result.ok());
        // Wire string stays byte-identical through the typed funnel.
        assert_eq!(result.data(), "Error: bad input");
        // The `error` field now carries the typed Display (internal channel);
        // the model-visible string is `data()` (rendered).
        assert_eq!(result.error(), Some("Execution failed: bad input"));
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
        assert!(result.ok());
        assert_eq!(result.data(), "executed with: test");
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

    #[test]
    fn test_permission_level_ordering() {
        assert!(PermissionLevel::ReadOnly < PermissionLevel::Network);
        assert!(PermissionLevel::Network < PermissionLevel::Write);
        assert!(PermissionLevel::Write < PermissionLevel::Execute);
        assert!(PermissionLevel::Execute < PermissionLevel::System);
    }

    #[test]
    fn test_default_permission_is_read_only() {
        let tool = MockTool;
        assert_eq!(tool.permission(), PermissionLevel::ReadOnly);
    }

    #[test]
    fn test_tool_execution_result_is_retryable() {
        // Success is never retryable.
        assert!(!ToolExecutionResult::success("ok".into()).is_retryable());

        // Non-retryable failure (e.g. "not found" → NotFound).
        let nf = ToolExecutionResult::failure("No such file or directory".into());
        assert!(!nf.is_retryable());

        // Retryable failure (e.g. "connection refused" → NetworkError).
        let net = ToolExecutionResult::failure("connection refused".into());
        assert!(net.is_retryable());

        // Retryable failure (rate limit).
        let rl = ToolExecutionResult::failure("429 rate limit exceeded".into());
        assert!(rl.is_retryable());
    }
}
