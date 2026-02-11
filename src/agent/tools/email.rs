//! Email tools for checking inbox and sending emails from the REPL agent.

use std::collections::HashMap;

use async_trait::async_trait;
use serde_json::{json, Value};

use super::base::Tool;
use crate::bus::events::{InboundMessage, OutboundMessage};
use crate::channels::email::{poll_inbox, poll_inbox_api, send_email};
use crate::config::schema::EmailConfig;

/// Tool to check the email inbox for unread messages.
pub struct CheckInboxTool {
    email_config: EmailConfig,
}

impl CheckInboxTool {
    pub fn new(email_config: EmailConfig) -> Self {
        Self { email_config }
    }
}

#[async_trait]
impl Tool for CheckInboxTool {
    fn name(&self) -> &str {
        "check_inbox"
    }

    fn description(&self) -> &str {
        "Check the email inbox for new/unread messages. Returns a list of unread emails with sender, subject, and body."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {}
        })
    }

    async fn execute(&self, _params: HashMap<String, Value>) -> String {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<InboundMessage>();

        let use_api = self.email_config.imap_host.contains("agentmail.to");
        let result = if use_api {
            poll_inbox_api(&self.email_config, &tx).await
        } else {
            poll_inbox(&self.email_config, &tx).await
        };

        drop(tx); // close sender so receiver drains

        if let Err(e) = result {
            return format!("Error checking inbox: {}", e);
        }

        let mut messages = Vec::new();
        while let Ok(msg) = rx.try_recv() {
            let from = &msg.sender_id;
            let subject = msg
                .metadata
                .get("subject")
                .and_then(|v| v.as_str())
                .unwrap_or("(no subject)");
            messages.push(format!(
                "From: {}\nSubject: {}\n\n{}",
                from, subject, msg.content
            ));
        }

        if messages.is_empty() {
            "No new messages.".to_string()
        } else {
            format!(
                "{} unread message(s):\n\n{}",
                messages.len(),
                messages.join("\n---\n")
            )
        }
    }
}

/// Tool to send an email via SMTP.
pub struct SendEmailTool {
    email_config: EmailConfig,
}

impl SendEmailTool {
    pub fn new(email_config: EmailConfig) -> Self {
        Self { email_config }
    }
}

#[async_trait]
impl Tool for SendEmailTool {
    fn name(&self) -> &str {
        "send_email"
    }

    fn description(&self) -> &str {
        "Send an email to a recipient."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "Recipient email address"
                },
                "subject": {
                    "type": "string",
                    "description": "Email subject line"
                },
                "body": {
                    "type": "string",
                    "description": "Email body text"
                },
                "reply_to_message_id": {
                    "type": "string",
                    "description": "Message-ID to reply to (for threading)"
                }
            },
            "required": ["to", "subject", "body"]
        })
    }

    async fn execute(&self, params: HashMap<String, Value>) -> String {
        let to = match params.get("to").and_then(|v| v.as_str()) {
            Some(t) => t,
            None => return "Error: 'to' parameter is required".to_string(),
        };
        let subject = match params.get("subject").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => return "Error: 'subject' parameter is required".to_string(),
        };
        let body = match params.get("body").and_then(|v| v.as_str()) {
            Some(b) => b,
            None => return "Error: 'body' parameter is required".to_string(),
        };
        let reply_to = params.get("reply_to_message_id").and_then(|v| v.as_str());

        let mut msg = OutboundMessage::new("email", format!("email:{}", to), body);
        msg.metadata.insert("subject".to_string(), json!(subject));
        if let Some(ref_id) = reply_to {
            msg.metadata.insert("message_id".to_string(), json!(ref_id));
        }

        match send_email(&self.email_config, &msg).await {
            Ok(()) => format!("Email sent to {}", to),
            Err(e) => format!("Error sending email: {}", e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_inbox_tool_name() {
        let config = EmailConfig::default();
        let tool = CheckInboxTool::new(config);
        assert_eq!(tool.name(), "check_inbox");
    }

    #[test]
    fn test_send_email_tool_name() {
        let config = EmailConfig::default();
        let tool = SendEmailTool::new(config);
        assert_eq!(tool.name(), "send_email");
    }

    #[test]
    fn test_check_inbox_schema() {
        let config = EmailConfig::default();
        let tool = CheckInboxTool::new(config);
        let schema = tool.to_schema();
        assert_eq!(schema["function"]["name"], "check_inbox");
        assert_eq!(schema["type"], "function");
    }

    #[test]
    fn test_send_email_schema() {
        let config = EmailConfig::default();
        let tool = SendEmailTool::new(config);
        let schema = tool.to_schema();
        assert_eq!(schema["function"]["name"], "send_email");
        let params = &schema["function"]["parameters"];
        assert!(params["properties"]["to"].is_object());
        assert!(params["properties"]["subject"].is_object());
        assert!(params["properties"]["body"].is_object());
        let required = params["required"].as_array().unwrap();
        assert!(required.contains(&json!("to")));
        assert!(required.contains(&json!("subject")));
        assert!(required.contains(&json!("body")));
    }

    #[tokio::test]
    async fn test_send_email_missing_to() {
        let config = EmailConfig::default();
        let tool = SendEmailTool::new(config);
        let params = HashMap::new();
        let result = tool.execute(params).await;
        assert!(
            result.contains("Error"),
            "Should error without 'to': {}",
            result
        );
    }

    #[tokio::test]
    async fn test_send_email_missing_subject() {
        let config = EmailConfig::default();
        let tool = SendEmailTool::new(config);
        let mut params = HashMap::new();
        params.insert("to".to_string(), json!("test@example.com"));
        let result = tool.execute(params).await;
        assert!(
            result.contains("Error"),
            "Should error without 'subject': {}",
            result
        );
    }

    #[tokio::test]
    async fn test_send_email_missing_body() {
        let config = EmailConfig::default();
        let tool = SendEmailTool::new(config);
        let mut params = HashMap::new();
        params.insert("to".to_string(), json!("test@example.com"));
        params.insert("subject".to_string(), json!("Test"));
        let result = tool.execute(params).await;
        assert!(
            result.contains("Error"),
            "Should error without 'body': {}",
            result
        );
    }

    #[tokio::test]
    async fn test_check_inbox_fails_gracefully_on_unreachable() {
        let config = EmailConfig {
            imap_host: "127.0.0.1".to_string(),
            imap_port: 1,
            username: "test".to_string(),
            password: "test".to_string(),
            ..Default::default()
        };
        let tool = CheckInboxTool::new(config);
        let result = tool.execute(HashMap::new()).await;
        assert!(
            result.starts_with("Error"),
            "Should return error: {}",
            result
        );
    }

    #[tokio::test]
    async fn test_send_email_fails_gracefully_on_invalid_from() {
        let config = EmailConfig {
            smtp_host: "smtp.example.com".to_string(),
            username: "not-valid".to_string(),
            password: "test".to_string(),
            ..Default::default()
        };
        let tool = SendEmailTool::new(config);
        let mut params = HashMap::new();
        params.insert("to".to_string(), json!("test@example.com"));
        params.insert("subject".to_string(), json!("Test"));
        params.insert("body".to_string(), json!("Hello"));
        let result = tool.execute(params).await;
        assert!(
            result.starts_with("Error"),
            "Should return error: {}",
            result
        );
    }
}
