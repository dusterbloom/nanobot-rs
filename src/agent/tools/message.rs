#![allow(dead_code)]
//! Message tool for sending messages to users.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use tokio::sync::Mutex;

use super::base::Tool;
use crate::bus::events::OutboundMessage;

/// Type alias for the send callback.
pub type SendCallback =
    Arc<dyn Fn(OutboundMessage) -> Pin<Box<dyn Future<Output = Result<()>> + Send>> + Send + Sync>;

/// Tool to send messages to users on chat channels.
pub struct MessageTool {
    send_callback: Arc<Mutex<Option<SendCallback>>>,
    default_channel: Arc<Mutex<String>>,
    default_chat_id: Arc<Mutex<String>>,
}

impl MessageTool {
    /// Create a new message tool.
    pub fn new(
        send_callback: Option<SendCallback>,
        default_channel: &str,
        default_chat_id: &str,
    ) -> Self {
        Self {
            send_callback: Arc::new(Mutex::new(send_callback)),
            default_channel: Arc::new(Mutex::new(default_channel.to_string())),
            default_chat_id: Arc::new(Mutex::new(default_chat_id.to_string())),
        }
    }

    /// Set the current message context.
    pub async fn set_context(&self, channel: &str, chat_id: &str) {
        *self.default_channel.lock().await = channel.to_string();
        *self.default_chat_id.lock().await = chat_id.to_string();
    }

    /// Set the callback for sending messages.
    pub async fn set_send_callback(&self, callback: SendCallback) {
        *self.send_callback.lock().await = Some(callback);
    }
}

#[async_trait]
impl Tool for MessageTool {
    fn name(&self) -> &str {
        "message"
    }

    fn description(&self) -> &str {
        "Send a message to the user. Use this when you want to communicate something."
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

    async fn execute(&self, params: HashMap<String, serde_json::Value>) -> String {
        let content = match params.get("content").and_then(|v| v.as_str()) {
            Some(c) => c.to_string(),
            None => return "Error: 'content' parameter is required".to_string(),
        };

        let default_channel = self.default_channel.lock().await.clone();
        let default_chat_id = self.default_chat_id.lock().await.clone();

        let channel = params
            .get("channel")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .unwrap_or(default_channel);

        let chat_id = params
            .get("chat_id")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .unwrap_or(default_chat_id);

        if channel.is_empty() || chat_id.is_empty() {
            return "Error: No target channel/chat specified".to_string();
        }

        let callback_guard = self.send_callback.lock().await;
        let callback = match callback_guard.as_ref() {
            Some(cb) => cb.clone(),
            None => return "Error: Message sending not configured".to_string(),
        };
        // Drop the lock before awaiting the callback.
        drop(callback_guard);

        let msg = OutboundMessage::new(&channel, &chat_id, &content);

        match callback(msg).await {
            Ok(()) => format!("Message sent to {}:{}", channel, chat_id),
            Err(e) => format!("Error sending message: {}", e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_tool_name() {
        let tool = MessageTool::new(None, "test_channel", "test_chat");
        assert_eq!(tool.name(), "message");
    }

    #[test]
    fn test_message_tool_description() {
        let tool = MessageTool::new(None, "test_channel", "test_chat");
        assert!(!tool.description().is_empty());
    }

    #[test]
    fn test_message_tool_parameters() {
        let tool = MessageTool::new(None, "test_channel", "test_chat");
        let params = tool.parameters();
        assert_eq!(params["type"], "object");
        assert!(params["properties"]["content"].is_object());
        let required = params["required"].as_array().unwrap();
        assert!(required.iter().any(|v| v == "content"));
    }

    #[tokio::test]
    async fn test_execute_without_callback() {
        let tool = MessageTool::new(None, "chan", "chat");
        let mut params = HashMap::new();
        params.insert(
            "content".to_string(),
            serde_json::Value::String("hello".to_string()),
        );
        let result = tool.execute(params).await;
        assert!(result.contains("Message sending not configured"));
    }

    #[tokio::test]
    async fn test_execute_missing_content() {
        let tool = MessageTool::new(None, "chan", "chat");
        let params = HashMap::new();
        let result = tool.execute(params).await;
        assert!(result.contains("'content' parameter is required"));
    }

    #[tokio::test]
    async fn test_execute_empty_channel_and_chat() {
        let tool = MessageTool::new(None, "", "");
        let mut params = HashMap::new();
        params.insert(
            "content".to_string(),
            serde_json::Value::String("hello".to_string()),
        );
        let result = tool.execute(params).await;
        assert!(result.contains("No target channel/chat specified"));
    }

    #[tokio::test]
    async fn test_set_context() {
        let tool = MessageTool::new(None, "old_channel", "old_chat");
        tool.set_context("new_channel", "new_chat").await;

        // Verify context changed by executing without callback -- channel/chat
        // should now be "new_channel"/"new_chat" (we cannot directly observe
        // this, but we can verify it does not error with "No target channel").
        let mut params = HashMap::new();
        params.insert(
            "content".to_string(),
            serde_json::Value::String("test".to_string()),
        );
        let result = tool.execute(params).await;
        // Should not complain about missing channel -- it should fail on the
        // callback not being set instead.
        assert!(result.contains("Message sending not configured"));
    }

    #[tokio::test]
    async fn test_execute_with_mock_callback() {
        let callback: SendCallback = Arc::new(|_msg: OutboundMessage| Box::pin(async { Ok(()) }));
        let tool = MessageTool::new(Some(callback), "telegram", "12345");

        let mut params = HashMap::new();
        params.insert(
            "content".to_string(),
            serde_json::Value::String("hello!".to_string()),
        );
        let result = tool.execute(params).await;
        assert_eq!(result, "Message sent to telegram:12345");
    }

    #[tokio::test]
    async fn test_execute_with_failing_callback() {
        let callback: SendCallback = Arc::new(|_msg: OutboundMessage| {
            Box::pin(async { Err(anyhow::anyhow!("network error")) })
        });
        let tool = MessageTool::new(Some(callback), "discord", "999");

        let mut params = HashMap::new();
        params.insert(
            "content".to_string(),
            serde_json::Value::String("hello".to_string()),
        );
        let result = tool.execute(params).await;
        assert!(result.contains("Error sending message"));
        assert!(result.contains("network error"));
    }

    #[tokio::test]
    async fn test_set_send_callback() {
        let tool = MessageTool::new(None, "chan", "chat");

        // Initially no callback.
        let mut params = HashMap::new();
        params.insert(
            "content".to_string(),
            serde_json::Value::String("test".to_string()),
        );
        let result = tool.execute(params).await;
        assert!(result.contains("Message sending not configured"));

        // Set callback.
        let callback: SendCallback = Arc::new(|_msg: OutboundMessage| Box::pin(async { Ok(()) }));
        tool.set_send_callback(callback).await;

        // Now it should succeed.
        let mut params = HashMap::new();
        params.insert(
            "content".to_string(),
            serde_json::Value::String("test".to_string()),
        );
        let result = tool.execute(params).await;
        assert_eq!(result, "Message sent to chan:chat");
    }

    #[tokio::test]
    async fn test_execute_with_channel_override() {
        let callback: SendCallback = Arc::new(|_msg: OutboundMessage| Box::pin(async { Ok(()) }));
        let tool = MessageTool::new(Some(callback), "default_chan", "default_chat");

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
        let result = tool.execute(params).await;
        assert_eq!(result, "Message sent to override_chan:override_chat");
    }
}
