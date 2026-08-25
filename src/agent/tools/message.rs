//! Message tool for sending messages to users.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

use super::base::{PermissionLevel, Tool, ToolContext, ToolOutput, ToolResult};
use crate::agent::host_bridge::{MessageHost, SendMessageRequest};
use crate::errors::ToolError;

/// Tool to send messages to users on chat channels.
///
/// Depends only on [`MessageHost`] (research §3.3 ISP/DIP): the production
/// host (`AgentHost` in `tool_wiring`) and test mocks are interchangeable.
/// Default channel/chat are baked in at the registry boundary; per-call
/// `channel`/`chat_id` params override them.
pub struct MessageTool {
    host: Arc<dyn MessageHost>,
    default_channel: String,
    default_chat_id: String,
}

impl MessageTool {
    /// Create a new message tool wired to a typed message host.
    pub fn new(host: Arc<dyn MessageHost>, default_channel: &str, default_chat_id: &str) -> Self {
        Self {
            host,
            default_channel: default_channel.to_string(),
            default_chat_id: default_chat_id.to_string(),
        }
    }
}
#[async_trait]
impl Tool for MessageTool {
    fn name(&self) -> &str {
        "message"
    }

    fn permission(&self) -> PermissionLevel {
        PermissionLevel::System
    }

    fn description(&self) -> &str {
        "Send an explicit out-of-band notification to a chat channel. Do not use this for the normal assistant response; answer with assistant text instead."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The message content to send"
                },
                "channel": {
                    "type": "string",
                    "description": "Optional: target channel (telegram, discord, etc.)"
                },
                "chat_id": {
                    "type": "string",
                    "description": "Optional: target chat/user ID"
                }
            },
            "required": ["content"]
        })
    }

    /// Typed entry point: resolves the target channel/chat (per-call params
    /// override the baked defaults; same `"No target channel/chat
    /// specified"` guard as legacy), then drives the send through the
    /// [`MessageHost`] trait — no callback slot, no `anyhow` at the tool
    /// boundary. The host's reply text is the model-visible output; host
    /// errors propagate typed and render byte-identically. The trait's
    /// default String `execute` renders this byte-for-byte.
    async fn execute(
        &self,
        params: HashMap<String, serde_json::Value>,
        _ctx: &ToolContext,
    ) -> ToolResult {
        let content = match params.get("content").and_then(|v| v.as_str()) {
            Some(c) => c.to_string(),
            None => {
                return Err(ToolError::MissingArg {
                    param: "content".to_string(),
                    example: r#"message({"content":"hello"})"#.to_string(),
                })
            }
        };

        let channel = params
            .get("channel")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| self.default_channel.clone());

        let chat_id = params
            .get("chat_id")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| self.default_chat_id.clone());

        if channel.is_empty() || chat_id.is_empty() {
            return Err(ToolError::Execution {
                message: "No target channel/chat specified".to_string(),
            });
        }

        let reply = self
            .host
            .send(SendMessageRequest {
                channel,
                chat_id,
                content,
            })
            .await?;
        Ok(ToolOutput { text: reply.text })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::host_bridge::SendMessageReply;

    /// Mock host echoing the legacy confirmation text back unchanged.
    struct MockHost;

    #[async_trait]
    impl MessageHost for MockHost {
        async fn send(&self, req: SendMessageRequest) -> Result<SendMessageReply, ToolError> {
            Ok(SendMessageReply {
                text: format!("Message sent to {}:{}", req.channel, req.chat_id),
            })
        }
    }

    /// Mock host whose send fails with a typed error.
    struct FailingHost;

    #[async_trait]
    impl MessageHost for FailingHost {
        async fn send(&self, _req: SendMessageRequest) -> Result<SendMessageReply, ToolError> {
            Err(ToolError::Execution {
                message: "Error sending message: network error".to_string(),
            })
        }
    }

    #[test]
    fn test_message_tool_name() {
        let tool = MessageTool::new(Arc::new(MockHost), "test_channel", "test_chat");
        assert_eq!(tool.name(), "message");
    }

    #[test]
    fn test_message_tool_description() {
        let tool = MessageTool::new(Arc::new(MockHost), "test_channel", "test_chat");
        assert!(!tool.description().is_empty());
    }

    #[test]
    fn test_message_tool_parameters() {
        let tool = MessageTool::new(Arc::new(MockHost), "test_channel", "test_chat");
        let params = tool.parameters();
        assert_eq!(params["type"], "object");
        assert!(params["properties"]["content"].is_object());
        let required = params["required"].as_array().unwrap();
        assert!(required.iter().any(|v| v == "content"));
    }

    #[tokio::test]
    async fn test_execute_missing_content() {
        let tool = MessageTool::new(Arc::new(MockHost), "chan", "chat");
        let params = HashMap::new();
        let result = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
        assert_eq!(
            result,
            "Error: 'content' parameter is required; call as message({\"content\":\"hello\"})"
        );
    }

    #[tokio::test]
    async fn test_execute_empty_channel_and_chat() {
        let tool = MessageTool::new(Arc::new(MockHost), "", "");
        let mut params = HashMap::new();
        params.insert(
            "content".to_string(),
            serde_json::Value::String("hello".to_string()),
        );
        let result = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
        assert_eq!(result, "Error: No target channel/chat specified");
    }

    #[tokio::test]
    async fn test_execute_uses_baked_defaults() {
        let tool = MessageTool::new(Arc::new(MockHost), "telegram", "12345");
        let mut params = HashMap::new();
        params.insert(
            "content".to_string(),
            serde_json::Value::String("hello!".to_string()),
        );
        let result = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
        assert_eq!(result, "Message sent to telegram:12345");
    }

    #[tokio::test]
    async fn test_execute_with_failing_host() {
        let tool = MessageTool::new(Arc::new(FailingHost), "discord", "999");
        let mut params = HashMap::new();
        params.insert(
            "content".to_string(),
            serde_json::Value::String("hello".to_string()),
        );
        let result = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
        assert!(result.contains("Error sending message"));
        assert!(result.contains("network error"));
    }

    #[tokio::test]
    async fn test_execute_with_channel_override() {
        let tool = MessageTool::new(Arc::new(MockHost), "default_chan", "default_chat");
        let mut params = HashMap::new();
        params.insert(
            "content".to_string(),
            serde_json::Value::String("hello".to_string()),
        );
        params.insert(
            "channel".to_string(),
            serde_json::Value::String("override_chan".to_string()),
        );
        params.insert(
            "chat_id".to_string(),
            serde_json::Value::String("override_chat".to_string()),
        );
        let result = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
        assert_eq!(result, "Message sent to override_chan:override_chat");
    }

    #[tokio::test]
    async fn test_empty_override_falls_back_to_default() {
        // An empty channel/chat param is treated as "use the default" —
        // same filter as the legacy surface.
        let tool = MessageTool::new(Arc::new(MockHost), "telegram", "42");
        let mut params = HashMap::new();
        params.insert(
            "content".to_string(),
            serde_json::Value::String("hi".to_string()),
        );
        params.insert(
            "channel".to_string(),
            serde_json::Value::String("".to_string()),
        );
        params.insert(
            "chat_id".to_string(),
            serde_json::Value::String("".to_string()),
        );
        let result = crate::agent::tools::base::render_result(
            tool.execute(params, &crate::agent::tools::base::ToolContext::sandbox())
                .await,
        );
        assert_eq!(result, "Message sent to telegram:42");
    }
}
