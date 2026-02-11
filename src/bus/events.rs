//! Event types for the message bus.

use std::collections::HashMap;

use chrono::{DateTime, Local};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Message received from a chat channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InboundMessage {
    /// Channel name (e.g. "telegram", "whatsapp", "feishu").
    pub channel: String,
    /// User identifier within the channel.
    pub sender_id: String,
    /// Chat/conversation identifier.
    pub chat_id: String,
    /// Message text content.
    pub content: String,
    /// When the message was received.
    #[serde(default = "now")]
    pub timestamp: DateTime<Local>,
    /// Media attachment URLs.
    #[serde(default)]
    pub media: Vec<String>,
    /// Channel-specific metadata.
    #[serde(default)]
    pub metadata: HashMap<String, Value>,
}

fn now() -> DateTime<Local> {
    Local::now()
}

impl InboundMessage {
    /// Create a new inbound message with required fields and sensible defaults.
    pub fn new(
        channel: impl Into<String>,
        sender_id: impl Into<String>,
        chat_id: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            channel: channel.into(),
            sender_id: sender_id.into(),
            chat_id: chat_id.into(),
            content: content.into(),
            timestamp: Local::now(),
            media: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Unique key for session identification (`"channel:chat_id"`).
    pub fn session_key(&self) -> String {
        format!("{}:{}", self.channel, self.chat_id)
    }
}

/// Message to send to a chat channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutboundMessage {
    /// Target channel name.
    pub channel: String,
    /// Target chat/conversation identifier.
    pub chat_id: String,
    /// Message text content.
    pub content: String,
    /// Optional message ID to reply to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reply_to: Option<String>,
    /// Media attachment URLs.
    #[serde(default)]
    pub media: Vec<String>,
    /// Channel-specific metadata.
    #[serde(default)]
    pub metadata: HashMap<String, Value>,
}

impl OutboundMessage {
    /// Create a new outbound message with required fields and sensible defaults.
    pub fn new(
        channel: impl Into<String>,
        chat_id: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            channel: channel.into(),
            chat_id: chat_id.into(),
            content: content.into(),
            reply_to: None,
            media: Vec::new(),
            metadata: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inbound_session_key() {
        let msg = InboundMessage::new("telegram", "user1", "chat42", "hello");
        assert_eq!(msg.session_key(), "telegram:chat42");
    }

    #[test]
    fn test_outbound_creation() {
        let msg = OutboundMessage::new("whatsapp", "+1234", "Hi there");
        assert_eq!(msg.channel, "whatsapp");
        assert_eq!(msg.chat_id, "+1234");
        assert_eq!(msg.content, "Hi there");
        assert!(msg.reply_to.is_none());
    }

    #[test]
    fn test_inbound_serialization_roundtrip() {
        let msg = InboundMessage::new("feishu", "u123", "c456", "test message");
        let json = serde_json::to_string(&msg).unwrap();
        let deserialized: InboundMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.channel, "feishu");
        assert_eq!(deserialized.content, "test message");
    }
}
