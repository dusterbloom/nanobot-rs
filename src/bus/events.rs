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

/// Coalesce multiple inbound messages from the same session into one.
///
/// Messages are merged in order with timing annotations so the LLM sees them
/// as a single turn with context. Media and metadata are combined.
pub fn coalesce_messages(mut messages: Vec<InboundMessage>) -> InboundMessage {
    assert!(!messages.is_empty(), "coalesce_messages requires at least 1 message");
    if messages.len() == 1 {
        return messages.remove(0);
    }
    let first = &messages[0];
    let channel = first.channel.clone();
    let sender_id = first.sender_id.clone();
    let chat_id = first.chat_id.clone();
    let timestamp = first.timestamp;

    let mut parts: Vec<String> = Vec::new();
    let mut all_media: Vec<String> = Vec::new();
    let mut merged_metadata = first.metadata.clone();

    for (i, msg) in messages.iter().enumerate() {
        if i == 0 {
            parts.push(msg.content.clone());
        } else {
            let delta_ms = msg.timestamp.signed_duration_since(timestamp).num_milliseconds();
            parts.push(format!("[+{}ms] {}", delta_ms, msg.content));
        }
        all_media.extend(msg.media.iter().cloned());
        // Merge metadata (later messages override)
        for (k, v) in &msg.metadata {
            merged_metadata.insert(k.clone(), v.clone());
        }
    }

    let mut merged = InboundMessage::new(channel, sender_id, chat_id, parts.join("\n"));
    merged.timestamp = timestamp;
    merged.media = all_media;
    merged.metadata = merged_metadata;
    merged
}

/// Returns true if this channel should bypass coalescing (single-user channels).
pub fn should_coalesce(channel: &str) -> bool {
    // CLI and voice are single-user, no rapid-fire messages to coalesce.
    !matches!(channel, "cli" | "voice" | "cron")
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

    // ---------------------------------------------------------------
    // coalesce_messages tests
    // ---------------------------------------------------------------

    #[test]
    fn test_coalesce_single_message_passthrough() {
        let msg = InboundMessage::new("telegram", "user1", "chat1", "hello");
        let result = coalesce_messages(vec![msg]);
        assert_eq!(result.content, "hello");
        assert_eq!(result.channel, "telegram");
    }

    #[test]
    fn test_coalesce_two_messages_adds_timing() {
        let mut m1 = InboundMessage::new("telegram", "user1", "chat1", "hello");
        let mut m2 = InboundMessage::new("telegram", "user1", "chat1", "world");
        // Simulate 200ms gap
        m2.timestamp = m1.timestamp + chrono::Duration::milliseconds(200);
        let result = coalesce_messages(vec![m1, m2]);
        assert!(result.content.contains("hello"));
        assert!(result.content.contains("[+200ms] world"));
    }

    #[test]
    fn test_coalesce_merges_media() {
        let mut m1 = InboundMessage::new("telegram", "user1", "chat1", "look");
        m1.media = vec!["photo1.jpg".into()];
        let mut m2 = InboundMessage::new("telegram", "user1", "chat1", "and this");
        m2.media = vec!["photo2.jpg".into()];
        let result = coalesce_messages(vec![m1, m2]);
        assert_eq!(result.media.len(), 2);
        assert!(result.media.contains(&"photo1.jpg".to_string()));
        assert!(result.media.contains(&"photo2.jpg".to_string()));
    }

    #[test]
    fn test_coalesce_preserves_first_timestamp() {
        let m1 = InboundMessage::new("telegram", "user1", "chat1", "first");
        let ts = m1.timestamp;
        let mut m2 = InboundMessage::new("telegram", "user1", "chat1", "second");
        m2.timestamp = ts + chrono::Duration::milliseconds(500);
        let result = coalesce_messages(vec![m1, m2]);
        assert_eq!(result.timestamp, ts);
    }

    #[test]
    fn test_coalesce_three_messages() {
        let m1 = InboundMessage::new("telegram", "user1", "chat1", "hey");
        let ts = m1.timestamp;
        let mut m2 = InboundMessage::new("telegram", "user1", "chat1", "check this");
        m2.timestamp = ts + chrono::Duration::milliseconds(100);
        let mut m3 = InboundMessage::new("telegram", "user1", "chat1", "please");
        m3.timestamp = ts + chrono::Duration::milliseconds(350);
        let result = coalesce_messages(vec![m1, m2, m3]);
        assert_eq!(result.content, "hey\n[+100ms] check this\n[+350ms] please");
    }

    // ---------------------------------------------------------------
    // should_coalesce tests
    // ---------------------------------------------------------------

    #[test]
    fn test_should_coalesce_telegram() {
        assert!(should_coalesce("telegram"));
    }

    #[test]
    fn test_should_coalesce_whatsapp() {
        assert!(should_coalesce("whatsapp"));
    }

    #[test]
    fn test_should_not_coalesce_cli() {
        assert!(!should_coalesce("cli"));
    }

    #[test]
    fn test_should_not_coalesce_voice() {
        assert!(!should_coalesce("voice"));
    }

    #[test]
    fn test_should_not_coalesce_cron() {
        assert!(!should_coalesce("cron"));
    }
}
