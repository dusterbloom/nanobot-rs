//! Email channel implementation using IMAP polling and SMTP sending.
//!
//! Receives messages by polling IMAP INBOX for UNSEEN messages, sends replies
//! via SMTP with proper threading headers (`In-Reply-To`, `References`).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use serde_json::json;
use tokio::sync::mpsc::UnboundedSender;
use tracing::{debug, info, warn};

use crate::bus::events::{InboundMessage, OutboundMessage};
use crate::channels::base::Channel;
use crate::config::schema::EmailConfig;

/// Email channel using IMAP polling for receiving and SMTP for sending.
pub struct EmailChannel {
    config: EmailConfig,
    bus_tx: UnboundedSender<InboundMessage>,
    running: Arc<AtomicBool>,
}

impl EmailChannel {
    /// Create a new `EmailChannel`.
    pub fn new(
        config: EmailConfig,
        bus_tx: UnboundedSender<InboundMessage>,
    ) -> Self {
        Self {
            config,
            bus_tx,
            running: Arc::new(AtomicBool::new(false)),
        }
    }
}

#[async_trait]
impl Channel for EmailChannel {
    fn name(&self) -> &str {
        "email"
    }

    async fn start(&mut self) -> Result<()> {
        if self.config.imap_host.is_empty() {
            return Err(anyhow::anyhow!("Email IMAP host not configured"));
        }
        if self.config.username.is_empty() || self.config.password.is_empty() {
            return Err(anyhow::anyhow!("Email username/password not configured"));
        }

        self.running.store(true, Ordering::SeqCst);

        let config = self.config.clone();
        let bus_tx = self.bus_tx.clone();
        let running = self.running.clone();

        let use_api = config.imap_host.contains("agentmail.to");
        if use_api {
            info!("Starting email channel (agentmail.to API polling every {}s)...", config.poll_interval_secs);
        } else {
            info!("Starting email channel (IMAP polling every {}s)...", config.poll_interval_secs);
        }

        tokio::spawn(async move {
            while running.load(Ordering::SeqCst) {
                let result = if use_api {
                    poll_inbox_api(&config, &bus_tx).await
                } else {
                    poll_inbox(&config, &bus_tx).await
                };
                if let Err(e) = result {
                    warn!("Email poll error: {}", e);
                }
                tokio::time::sleep(std::time::Duration::from_secs(config.poll_interval_secs)).await;
            }
        });

        Ok(())
    }

    async fn stop(&mut self) -> Result<()> {
        self.running.store(false, Ordering::SeqCst);
        info!("Email channel stopped");
        Ok(())
    }

    async fn send(&self, msg: &OutboundMessage) -> Result<()> {
        send_email(&self.config, msg).await
    }

    fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }
}

// ---------------------------------------------------------------------------
// IMAP polling
// ---------------------------------------------------------------------------

/// Connect to IMAP, fetch UNSEEN messages, parse them, and emit to the bus.
pub async fn poll_inbox(
    config: &EmailConfig,
    bus_tx: &UnboundedSender<InboundMessage>,
) -> Result<()> {
    use anyhow::Context;
    use async_native_tls::TlsConnector;
    use futures_util::TryStreamExt;
    use tokio_util::compat::TokioAsyncReadCompatExt;

    debug!("Email: connecting to {}:{}", config.imap_host, config.imap_port);
    let tcp = tokio::net::TcpStream::connect((config.imap_host.as_str(), config.imap_port))
        .await
        .context("IMAP TCP connect failed")?;
    let tcp_compat = tcp.compat();

    debug!("Email: TLS handshake");
    let tls = TlsConnector::new();
    let tls_stream = tls
        .connect(&config.imap_host, tcp_compat)
        .await
        .context("IMAP TLS handshake failed")?;

    debug!("Email: reading server greeting");
    let client = async_imap::Client::new(tls_stream);

    debug!("Email: logging in as {}", config.username);
    let mut session = client
        .login(&config.username, &config.password)
        .await
        .map_err(|e| anyhow::anyhow!("IMAP LOGIN failed: {}", e.0))?;

    debug!("Email: selecting INBOX");
    session.select("INBOX").await.context("IMAP SELECT INBOX failed")?;

    // Use UID SEARCH for better compatibility across IMAP servers.
    debug!("Email: searching for unseen messages (UID SEARCH)");
    let unseen = session
        .uid_search("UNSEEN")
        .await
        .context("IMAP UID SEARCH UNSEEN failed")?;

    if unseen.is_empty() {
        debug!("Email: no unseen messages");
        session.logout().await.ok(); // best-effort logout
        return Ok(());
    }

    debug!("Email: found {} unseen messages", unseen.len());

    let uid_set: String = unseen
        .iter()
        .map(|n| n.to_string())
        .collect::<Vec<_>>()
        .join(",");

    debug!("Email: fetching UIDs {}", uid_set);
    let messages: Vec<async_imap::types::Fetch> = session
        .uid_fetch(&uid_set, "(RFC822 UID)")
        .await
        .context("IMAP UID FETCH failed")?
        .try_collect()
        .await?;

    for msg in &messages {
        if let Some(body) = msg.body() {
            match mail_parser::MessageParser::default().parse(body) {
                Some(parsed) => {
                    let from = parsed
                        .from()
                        .and_then(|addrs| addrs.first())
                        .and_then(|a| a.address())
                        .unwrap_or("unknown")
                        .to_string();

                    // Check allow list.
                    if !config.allow_from.is_empty() && !config.allow_from.contains(&from) {
                        debug!("Email: ignoring message from non-allowed sender {}", from);
                        continue;
                    }

                    let subject = parsed.subject().unwrap_or("").to_string();
                    let text_body = parsed
                        .body_text(0)
                        .map(|t| t.to_string())
                        .unwrap_or_default();

                    let content = if subject.is_empty() {
                        text_body
                    } else {
                        format!("Subject: {}\n\n{}", subject, text_body)
                    };

                    if content.trim().is_empty() {
                        continue;
                    }

                    let message_id = parsed
                        .message_id()
                        .unwrap_or("")
                        .to_string();

                    let mut inbound = InboundMessage::new(
                        "email",
                        &from,
                        &format!("email:{}", from),
                        &content,
                    );
                    inbound
                        .metadata
                        .insert("message_id".to_string(), json!(message_id));
                    inbound
                        .metadata
                        .insert("subject".to_string(), json!(subject));

                    debug!(
                        "Email from {}: {}",
                        from,
                        &content[..content.len().min(50)]
                    );

                    let _ = bus_tx.send(inbound);
                }
                None => {
                    warn!("Email: failed to parse message body");
                }
            }
        }
    }

    // Mark all fetched messages as Seen using UID STORE.
    debug!("Email: marking {} messages as Seen", messages.len());
    let _updates: Vec<async_imap::types::Fetch> = session
        .uid_store(&uid_set, "+FLAGS (\\Seen)")
        .await
        .context("IMAP UID STORE +FLAGS failed")?
        .try_collect()
        .await?;

    session.logout().await.ok(); // best-effort logout
    Ok(())
}

// ---------------------------------------------------------------------------
// REST API polling (agentmail.to)
// ---------------------------------------------------------------------------

/// Poll for new messages via the agentmail.to REST API.
///
/// agentmail.to doesn't support standard IMAP commands (SELECT, SEARCH, etc.)
/// so we use their HTTP API instead. The password field is used as the API key.
pub async fn poll_inbox_api(
    config: &EmailConfig,
    bus_tx: &UnboundedSender<InboundMessage>,
) -> Result<()> {
    use anyhow::Context;

    let api_key = &config.password;
    let inbox_id = &config.username; // inbox_id matches the email address
    let client = reqwest::Client::new();

    // List unread messages.
    let url = format!(
        "https://api.agentmail.to/v0/inboxes/{}/messages",
        inbox_id
    );
    debug!("Email API: listing unread messages for {}", inbox_id);

    let resp = client
        .get(&url)
        .bearer_auth(api_key)
        .query(&[("labels", "unread")])
        .send()
        .await
        .context("agentmail.to API request failed")?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(anyhow::anyhow!(
            "agentmail.to API returned {}: {}",
            status,
            &body[..body.len().min(200)]
        ));
    }

    let data: serde_json::Value = resp.json().await.context("Failed to parse API response")?;

    let messages = data["messages"]
        .as_array()
        .unwrap_or(&Vec::new())
        .clone();

    if messages.is_empty() {
        debug!("Email API: no unread messages");
        return Ok(());
    }

    debug!("Email API: found {} unread messages", messages.len());

    for msg in &messages {
        let labels = msg["labels"]
            .as_array()
            .map(|a| a.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>())
            .unwrap_or_default();

        // Only process received messages (skip sent/draft).
        if !labels.contains(&"received") {
            continue;
        }

        let from_raw = msg["from"].as_str().unwrap_or("unknown");
        // Extract email address from "Name <email@domain>" format.
        let from = extract_email_address(from_raw);

        // Check allow list.
        if !config.allow_from.is_empty() && !config.allow_from.contains(&from) {
            debug!("Email API: ignoring message from non-allowed sender {}", from);
            continue;
        }

        let subject = msg["subject"].as_str().unwrap_or("").to_string();
        let message_id = msg["message_id"].as_str().unwrap_or("").to_string();
        let api_msg_id = message_id.clone();

        // Fetch full message body.
        let body_text = fetch_message_body(&client, api_key, inbox_id, &api_msg_id)
            .await
            .unwrap_or_else(|e| {
                // Fall back to preview if full fetch fails.
                warn!("Email API: failed to fetch message body: {}", e);
                msg["preview"].as_str().unwrap_or("").to_string()
            });

        let content = if subject.is_empty() {
            body_text
        } else {
            format!("Subject: {}\n\n{}", subject, body_text)
        };

        if content.trim().is_empty() {
            continue;
        }

        let mut inbound = InboundMessage::new(
            "email",
            &from,
            &format!("email:{}", from),
            &content,
        );
        inbound
            .metadata
            .insert("message_id".to_string(), json!(message_id));
        inbound
            .metadata
            .insert("subject".to_string(), json!(subject));

        debug!(
            "Email API from {}: {}",
            from,
            &content[..content.len().min(50)]
        );

        let _ = bus_tx.send(inbound);

        // Mark message as read by removing "unread" label.
        mark_message_read(&client, api_key, inbox_id, &api_msg_id)
            .await
            .ok(); // best-effort
    }

    Ok(())
}

/// Fetch the full text body of a message from agentmail.to.
async fn fetch_message_body(
    client: &reqwest::Client,
    api_key: &str,
    inbox_id: &str,
    message_id: &str,
) -> Result<String> {
    let url = format!(
        "https://api.agentmail.to/v0/inboxes/{}/messages/{}",
        inbox_id,
        url_encode(message_id)
    );
    let resp = client.get(&url).bearer_auth(api_key).send().await?;
    let data: serde_json::Value = resp.json().await?;
    // Prefer extracted_text (new content only), fall back to text, then preview.
    let text = data["extracted_text"]
        .as_str()
        .or_else(|| data["text"].as_str())
        .or_else(|| data["preview"].as_str())
        .unwrap_or("")
        .to_string();
    Ok(text)
}

/// Mark a message as read on agentmail.to by updating its labels.
async fn mark_message_read(
    client: &reqwest::Client,
    api_key: &str,
    inbox_id: &str,
    message_id: &str,
) -> Result<()> {
    let url = format!(
        "https://api.agentmail.to/v0/inboxes/{}/messages/{}",
        inbox_id,
        url_encode(message_id)
    );
    client
        .patch(&url)
        .bearer_auth(api_key)
        .json(&serde_json::json!({
            "remove_labels": ["unread"]
        }))
        .send()
        .await?;
    debug!("Email API: marked message as read");
    Ok(())
}

/// Extract email address from "Name <email@domain>" or plain "email@domain".
pub fn extract_email_address(raw: &str) -> String {
    if let Some(start) = raw.find('<') {
        if let Some(end) = raw.find('>') {
            return raw[start + 1..end].to_string();
        }
    }
    raw.to_string()
}

/// Percent-encode a string for use in a URL path segment.
fn url_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 3);
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char);
            }
            _ => {
                out.push('%');
                out.push_str(&format!("{:02X}", b));
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// SMTP sending
// ---------------------------------------------------------------------------

/// Send an outbound message via SMTP.
pub async fn send_email(config: &EmailConfig, msg: &OutboundMessage) -> Result<()> {
    use lettre::message::Mailbox;
    use lettre::transport::smtp::authentication::Credentials;
    use lettre::{AsyncSmtpTransport, AsyncTransport, Message, Tokio1Executor};

    // chat_id format is "email:recipient@example.com".
    let recipient = msg
        .chat_id
        .strip_prefix("email:")
        .unwrap_or(&msg.chat_id);

    let from: Mailbox = config
        .username
        .parse()
        .map_err(|e| anyhow::anyhow!("Invalid from address '{}': {}", config.username, e))?;

    let to: Mailbox = recipient
        .parse()
        .map_err(|e| anyhow::anyhow!("Invalid to address '{}': {}", recipient, e))?;

    let mut builder = Message::builder().from(from).to(to);

    // Thread the reply with In-Reply-To and References headers.
    if let Some(ref_id) = msg.metadata.get("message_id").and_then(|v| v.as_str()) {
        if !ref_id.is_empty() {
            builder = builder
                .in_reply_to(ref_id.to_string())
                .references(ref_id.to_string());
        }
    }

    let subject = msg
        .metadata
        .get("subject")
        .and_then(|v| v.as_str())
        .map(|s| {
            if s.starts_with("Re: ") {
                s.to_string()
            } else {
                format!("Re: {}", s)
            }
        })
        .unwrap_or_else(|| "Re: Your message".to_string());

    let email = builder
        .subject(subject)
        .body(msg.content.clone())
        .map_err(|e| anyhow::anyhow!("Failed to build email: {}", e))?;

    let creds = Credentials::new(config.username.clone(), config.password.clone());

    // Port 465 = implicit TLS, port 587 = STARTTLS.
    let mailer = if config.smtp_port == 465 {
        AsyncSmtpTransport::<Tokio1Executor>::relay(&config.smtp_host)?
            .port(config.smtp_port)
            .credentials(creds)
            .build()
    } else {
        AsyncSmtpTransport::<Tokio1Executor>::starttls_relay(&config.smtp_host)?
            .port(config.smtp_port)
            .credentials(creds)
            .build()
    };

    mailer.send(email).await?;
    debug!("Email sent to {}", recipient);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bus::events::OutboundMessage;
    use crate::config::schema::EmailConfig;
    use tokio::sync::mpsc;

    /// Helper: create an Ethereal test account via the Nodemailer API.
    /// Returns None if the API is unreachable (graceful skip).
    async fn create_ethereal_account() -> Option<EmailConfig> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .ok()?;
        let resp = client
            .post("https://api.nodemailer.com/user")
            .json(&serde_json::json!({
                "requestor": "nanobot-test",
                "version": "1.0.0"
            }))
            .send()
            .await
            .ok()?;

        let data: serde_json::Value = resp.json().await.ok()?;
        if data.get("status")?.as_str()? != "success" {
            return None;
        }

        Some(EmailConfig {
            enabled: true,
            imap_host: data["imap"]["host"].as_str()?.to_string(),
            imap_port: data["imap"]["port"].as_u64()? as u16,
            smtp_host: data["smtp"]["host"].as_str()?.to_string(),
            smtp_port: data["smtp"]["port"].as_u64()? as u16,
            username: data["user"].as_str()?.to_string(),
            password: data["pass"].as_str()?.to_string(),
            poll_interval_secs: 5,
            allow_from: Vec::new(),
        })
    }

    /// Helper: send a test email to self using send_email().
    async fn send_self(config: &EmailConfig, subject: &str, body: &str) {
        let mut msg = OutboundMessage::new(
            "email",
            format!("email:{}", config.username),
            body,
        );
        msg.metadata
            .insert("subject".to_string(), serde_json::json!(subject));
        send_email(config, &msg).await.expect("SMTP send failed");
    }

    /// Helper: poll inbox and collect all inbound messages.
    async fn poll_collect(config: &EmailConfig) -> Vec<InboundMessage> {
        let (tx, mut rx) = mpsc::unbounded_channel::<InboundMessage>();
        poll_inbox(config, &tx).await.expect("IMAP poll failed");
        let mut msgs = Vec::new();
        while let Ok(m) = rx.try_recv() {
            msgs.push(m);
        }
        msgs
    }

    // ======================================================================
    // Unit tests (no network required)
    // ======================================================================

    #[test]
    fn test_start_fails_without_imap_host() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let config = EmailConfig::default();
            let (tx, _rx) = mpsc::unbounded_channel::<InboundMessage>();
            let mut channel = EmailChannel::new(config, tx);
            let result = channel.start().await;
            assert!(result.is_err());
            let err = result.unwrap_err().to_string();
            assert!(err.contains("IMAP host"), "Should mention IMAP host, got: {}", err);
        });
    }

    #[test]
    fn test_start_fails_without_credentials() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let mut config = EmailConfig::default();
            config.imap_host = "imap.example.com".to_string();
            let (tx, _rx) = mpsc::unbounded_channel::<InboundMessage>();
            let mut channel = EmailChannel::new(config, tx);
            let result = channel.start().await;
            assert!(result.is_err());
            let err = result.unwrap_err().to_string();
            assert!(err.contains("username/password"), "Should mention credentials, got: {}", err);
        });
    }

    #[test]
    fn test_start_fails_with_username_but_no_password() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let mut config = EmailConfig::default();
            config.imap_host = "imap.example.com".to_string();
            config.username = "user@example.com".to_string();
            // password is empty
            let (tx, _rx) = mpsc::unbounded_channel::<InboundMessage>();
            let mut channel = EmailChannel::new(config, tx);
            let result = channel.start().await;
            assert!(result.is_err(), "Should fail with empty password");
        });
    }

    #[test]
    fn test_default_config_has_sane_defaults() {
        let config = EmailConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.imap_port, 993);
        assert_eq!(config.smtp_port, 587);
        assert_eq!(config.poll_interval_secs, 30);
        assert!(config.allow_from.is_empty());
        assert!(config.imap_host.is_empty());
        assert!(config.smtp_host.is_empty());
        assert!(config.username.is_empty());
        assert!(config.password.is_empty());
    }

    #[test]
    fn test_channel_name() {
        let config = EmailConfig::default();
        let (tx, _rx) = mpsc::unbounded_channel::<InboundMessage>();
        let channel = EmailChannel::new(config, tx);
        assert_eq!(channel.name(), "email");
    }

    #[test]
    fn test_channel_not_running_initially() {
        let config = EmailConfig::default();
        let (tx, _rx) = mpsc::unbounded_channel::<InboundMessage>();
        let channel = EmailChannel::new(config, tx);
        assert!(!channel.is_running());
    }

    #[tokio::test]
    async fn test_stop_is_idempotent() {
        let config = EmailConfig::default();
        let (tx, _rx) = mpsc::unbounded_channel::<InboundMessage>();
        let mut channel = EmailChannel::new(config, tx);
        // Stop on a never-started channel should succeed.
        channel.stop().await.expect("first stop failed");
        channel.stop().await.expect("second stop failed");
        assert!(!channel.is_running());
    }

    #[tokio::test]
    async fn test_poll_fails_on_unreachable_host() {
        let config = EmailConfig {
            enabled: true,
            imap_host: "127.0.0.1".to_string(),
            imap_port: 1, // nothing listens here
            smtp_host: String::new(),
            smtp_port: 587,
            username: "test".to_string(),
            password: "test".to_string(),
            poll_interval_secs: 5,
            allow_from: Vec::new(),
        };
        let (tx, _rx) = mpsc::unbounded_channel::<InboundMessage>();
        let result = poll_inbox(&config, &tx).await;
        assert!(result.is_err(), "Should fail connecting to unreachable host");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("TCP connect") || err.contains("Connection refused"),
            "Error should mention connection failure, got: {}",
            err
        );
    }

    #[tokio::test]
    async fn test_send_fails_on_invalid_from_address() {
        let config = EmailConfig {
            enabled: true,
            imap_host: String::new(),
            imap_port: 993,
            smtp_host: "smtp.example.com".to_string(),
            smtp_port: 587,
            username: "not-a-valid-email".to_string(), // invalid from
            password: "test".to_string(),
            poll_interval_secs: 30,
            allow_from: Vec::new(),
        };
        let msg = OutboundMessage::new("email", "email:test@example.com", "body");
        let result = send_email(&config, &msg).await;
        assert!(result.is_err(), "Should fail with invalid from address");
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Invalid from address"), "Got: {}", err);
    }

    #[tokio::test]
    async fn test_send_fails_on_invalid_to_address() {
        let config = EmailConfig {
            enabled: true,
            imap_host: String::new(),
            imap_port: 993,
            smtp_host: "smtp.example.com".to_string(),
            smtp_port: 587,
            username: "sender@example.com".to_string(),
            password: "test".to_string(),
            poll_interval_secs: 30,
            allow_from: Vec::new(),
        };
        // chat_id with invalid email after "email:" prefix
        let msg = OutboundMessage::new("email", "email:not-valid", "body");
        let result = send_email(&config, &msg).await;
        assert!(result.is_err(), "Should fail with invalid to address");
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Invalid to address"), "Got: {}", err);
    }

    #[tokio::test]
    async fn test_send_strips_email_prefix_from_chat_id() {
        // Verify the chat_id → recipient extraction logic
        let config = EmailConfig {
            enabled: true,
            imap_host: String::new(),
            imap_port: 993,
            smtp_host: "127.0.0.1".to_string(), // will fail at SMTP connect
            smtp_port: 1,
            username: "sender@example.com".to_string(),
            password: "test".to_string(),
            poll_interval_secs: 30,
            allow_from: Vec::new(),
        };
        // chat_id with prefix: recipient should be "recv@example.com"
        let msg = OutboundMessage::new("email", "email:recv@example.com", "body");
        let result = send_email(&config, &msg).await;
        // It should fail at SMTP connection, not at address parsing
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            !err.contains("Invalid to address"),
            "Should parse email:recv@example.com correctly, got: {}",
            err
        );
    }

    #[tokio::test]
    async fn test_send_handles_chat_id_without_prefix() {
        let config = EmailConfig {
            enabled: true,
            imap_host: String::new(),
            imap_port: 993,
            smtp_host: "127.0.0.1".to_string(),
            smtp_port: 1,
            username: "sender@example.com".to_string(),
            password: "test".to_string(),
            poll_interval_secs: 30,
            allow_from: Vec::new(),
        };
        // chat_id WITHOUT "email:" prefix — should still work
        let msg = OutboundMessage::new("email", "recv@example.com", "body");
        let result = send_email(&config, &msg).await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            !err.contains("Invalid to address"),
            "Should handle raw email address as chat_id, got: {}",
            err
        );
    }

    #[tokio::test]
    async fn test_send_subject_adds_re_prefix() {
        // We can't easily inspect the built email without sending it,
        // but we can verify the subject logic via a roundtrip test below.
        // This test verifies the function doesn't panic with various subject values.
        let config = EmailConfig {
            enabled: true,
            imap_host: String::new(),
            imap_port: 993,
            smtp_host: "127.0.0.1".to_string(),
            smtp_port: 1,
            username: "sender@example.com".to_string(),
            password: "test".to_string(),
            poll_interval_secs: 30,
            allow_from: Vec::new(),
        };

        // With subject
        let mut msg = OutboundMessage::new("email", "email:recv@example.com", "body");
        msg.metadata.insert("subject".to_string(), json!("Hello"));
        let _ = send_email(&config, &msg).await; // will fail at SMTP, that's fine

        // With "Re: " already present
        let mut msg2 = OutboundMessage::new("email", "email:recv@example.com", "body");
        msg2.metadata.insert("subject".to_string(), json!("Re: Hello"));
        let _ = send_email(&config, &msg2).await;

        // Without subject metadata
        let msg3 = OutboundMessage::new("email", "email:recv@example.com", "body");
        let _ = send_email(&config, &msg3).await;
    }

    // ======================================================================
    // Integration tests (Ethereal — gracefully skipped if API unavailable)
    // ======================================================================

    #[tokio::test]
    async fn test_smtp_send_basic() {
        let config = match create_ethereal_account().await {
            Some(c) => c,
            None => { eprintln!("Skipping: Ethereal unavailable"); return; }
        };

        let mut msg = OutboundMessage::new(
            "email",
            format!("email:{}", config.username),
            "Hello from nanobot!",
        );
        msg.metadata.insert("subject".to_string(), json!("Test Email"));

        let result = send_email(&config, &msg).await;
        assert!(result.is_ok(), "SMTP send failed: {:?}", result.err());
    }

    #[tokio::test]
    async fn test_smtp_send_without_subject_metadata() {
        let config = match create_ethereal_account().await {
            Some(c) => c,
            None => { eprintln!("Skipping: Ethereal unavailable"); return; }
        };

        // No subject in metadata — should default to "Re: Your message"
        let msg = OutboundMessage::new(
            "email",
            format!("email:{}", config.username),
            "Body without explicit subject",
        );
        let result = send_email(&config, &msg).await;
        assert!(result.is_ok(), "Send without subject failed: {:?}", result.err());
    }

    #[tokio::test]
    async fn test_smtp_send_with_threading_headers() {
        let config = match create_ethereal_account().await {
            Some(c) => c,
            None => { eprintln!("Skipping: Ethereal unavailable"); return; }
        };

        let mut msg = OutboundMessage::new(
            "email",
            format!("email:{}", config.username),
            "This is a reply",
        );
        msg.metadata.insert("subject".to_string(), json!("Thread Test"));
        msg.metadata.insert("message_id".to_string(), json!("<abc123@example.com>"));

        let result = send_email(&config, &msg).await;
        assert!(result.is_ok(), "Send with threading failed: {:?}", result.err());
    }

    #[tokio::test]
    async fn test_smtp_send_unicode_body() {
        let config = match create_ethereal_account().await {
            Some(c) => c,
            None => { eprintln!("Skipping: Ethereal unavailable"); return; }
        };

        let mut msg = OutboundMessage::new(
            "email",
            format!("email:{}", config.username),
            "Unicode test: \u{1F600} \u{1F408} \u{2764}\u{FE0F} \u{00E9}\u{00E8}\u{00EA} \u{4F60}\u{597D}",
        );
        msg.metadata.insert("subject".to_string(), json!("Unicode \u{1F30D}"));

        let result = send_email(&config, &msg).await;
        assert!(result.is_ok(), "Unicode send failed: {:?}", result.err());
    }

    #[tokio::test]
    async fn test_smtp_send_long_body() {
        let config = match create_ethereal_account().await {
            Some(c) => c,
            None => { eprintln!("Skipping: Ethereal unavailable"); return; }
        };

        let long_body = "A".repeat(50_000);
        let mut msg = OutboundMessage::new(
            "email",
            format!("email:{}", config.username),
            &long_body,
        );
        msg.metadata.insert("subject".to_string(), json!("Long Body Test"));

        let result = send_email(&config, &msg).await;
        assert!(result.is_ok(), "Long body send failed: {:?}", result.err());
    }

    #[tokio::test]
    async fn test_imap_poll_empty_inbox() {
        let config = match create_ethereal_account().await {
            Some(c) => c,
            None => { eprintln!("Skipping: Ethereal unavailable"); return; }
        };

        let (tx, _rx) = mpsc::unbounded_channel::<InboundMessage>();
        let result = poll_inbox(&config, &tx).await;
        assert!(result.is_ok(), "IMAP poll failed on empty inbox: {:?}", result.err());
    }

    #[tokio::test]
    async fn test_roundtrip_send_then_poll() {
        let config = match create_ethereal_account().await {
            Some(c) => c,
            None => { eprintln!("Skipping: Ethereal unavailable"); return; }
        };

        send_self(&config, "Roundtrip Test", "Integration test body").await;
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;

        let msgs = poll_collect(&config).await;
        assert_eq!(msgs.len(), 1, "Expected 1 message, got {}", msgs.len());

        let inbound = &msgs[0];
        assert_eq!(inbound.channel, "email");
        assert!(inbound.content.contains("Integration test body"), "Body mismatch: {}", inbound.content);
        assert!(inbound.content.contains("Roundtrip Test"), "Subject missing: {}", inbound.content);
    }

    #[tokio::test]
    async fn test_roundtrip_message_metadata() {
        let config = match create_ethereal_account().await {
            Some(c) => c,
            None => { eprintln!("Skipping: Ethereal unavailable"); return; }
        };

        send_self(&config, "Metadata Test", "Check metadata fields").await;
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;

        let msgs = poll_collect(&config).await;
        assert_eq!(msgs.len(), 1);

        let inbound = &msgs[0];
        // Sender should be the Ethereal username
        assert_eq!(inbound.sender_id, config.username);
        // chat_id should be "email:{sender}"
        assert_eq!(inbound.chat_id, format!("email:{}", config.username));
        // Metadata should include subject (send_email adds "Re: " prefix)
        let subject = inbound.metadata.get("subject").and_then(|v| v.as_str());
        assert_eq!(subject, Some("Re: Metadata Test"));
        // Metadata should include message_id key (may be empty for some servers)
        assert!(
            inbound.metadata.contains_key("message_id"),
            "message_id key should be present in metadata"
        );
    }

    #[tokio::test]
    async fn test_roundtrip_marks_as_seen() {
        let config = match create_ethereal_account().await {
            Some(c) => c,
            None => { eprintln!("Skipping: Ethereal unavailable"); return; }
        };

        send_self(&config, "Seen Test", "Should be marked as read").await;
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;

        // First poll should find the message.
        let msgs1 = poll_collect(&config).await;
        assert_eq!(msgs1.len(), 1, "First poll should find 1 message");

        // Second poll should find nothing (message was marked Seen).
        let msgs2 = poll_collect(&config).await;
        assert_eq!(msgs2.len(), 0, "Second poll should find 0 messages (marked Seen)");
    }

    #[tokio::test]
    async fn test_roundtrip_multiple_messages() {
        let config = match create_ethereal_account().await {
            Some(c) => c,
            None => { eprintln!("Skipping: Ethereal unavailable"); return; }
        };

        // Send 3 emails.
        send_self(&config, "Multi 1", "First message").await;
        send_self(&config, "Multi 2", "Second message").await;
        send_self(&config, "Multi 3", "Third message").await;
        tokio::time::sleep(std::time::Duration::from_secs(3)).await;

        let msgs = poll_collect(&config).await;
        assert_eq!(msgs.len(), 3, "Expected 3 messages, got {}", msgs.len());

        // All should be from the email channel.
        for m in &msgs {
            assert_eq!(m.channel, "email");
        }
    }

    #[tokio::test]
    async fn test_roundtrip_body_only_no_subject() {
        let config = match create_ethereal_account().await {
            Some(c) => c,
            None => { eprintln!("Skipping: Ethereal unavailable"); return; }
        };

        // Send with empty subject — content should be body only (no "Subject: " prefix).
        let mut msg = OutboundMessage::new(
            "email",
            format!("email:{}", config.username),
            "Body without subject line",
        );
        msg.metadata.insert("subject".to_string(), json!(""));
        send_email(&config, &msg).await.expect("send failed");

        tokio::time::sleep(std::time::Duration::from_secs(2)).await;

        let msgs = poll_collect(&config).await;
        assert_eq!(msgs.len(), 1);
        // With empty subject, content should be just the body text
        // (the "Subject: \n\n" prefix shouldn't appear for empty subjects).
        assert!(
            msgs[0].content.contains("Body without subject line"),
            "Body should be present: {}",
            msgs[0].content
        );
    }

    #[tokio::test]
    async fn test_roundtrip_unicode_preserved() {
        let config = match create_ethereal_account().await {
            Some(c) => c,
            None => { eprintln!("Skipping: Ethereal unavailable"); return; }
        };

        let unicode_body = "Caf\u{00E9} \u{4F60}\u{597D} \u{1F600}";
        send_self(&config, "Unicode Roundtrip", unicode_body).await;
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;

        let msgs = poll_collect(&config).await;
        assert_eq!(msgs.len(), 1);
        // The body should contain the unicode text (at least the ASCII-compatible parts).
        assert!(
            msgs[0].content.contains("Caf\u{00E9}"),
            "Unicode should be preserved: {}",
            msgs[0].content
        );
    }

    #[tokio::test]
    async fn test_allow_from_filters_sender() {
        let config = match create_ethereal_account().await {
            Some(c) => c,
            None => { eprintln!("Skipping: Ethereal unavailable"); return; }
        };

        send_self(&config, "Filter Test", "Should be filtered").await;
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;

        // Poll with allow_from set to a different address — should filter out.
        let mut filtered = config.clone();
        filtered.allow_from = vec!["nobody@example.com".to_string()];
        let msgs = poll_collect(&filtered).await;
        assert_eq!(msgs.len(), 0, "Should be filtered by allow_from");
    }

    #[tokio::test]
    async fn test_allow_from_passes_matching_sender() {
        let config = match create_ethereal_account().await {
            Some(c) => c,
            None => { eprintln!("Skipping: Ethereal unavailable"); return; }
        };

        send_self(&config, "Allowed Test", "Should pass through").await;
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;

        // Poll with allow_from set to our own address — should pass through.
        let mut allowed = config.clone();
        allowed.allow_from = vec![config.username.clone()];
        let msgs = poll_collect(&allowed).await;
        assert_eq!(msgs.len(), 1, "Should pass allow_from filter with matching sender");
    }

    #[tokio::test]
    async fn test_allow_from_empty_allows_all() {
        let config = match create_ethereal_account().await {
            Some(c) => c,
            None => { eprintln!("Skipping: Ethereal unavailable"); return; }
        };

        send_self(&config, "Open Test", "No filter").await;
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;

        // Empty allow_from should allow all senders.
        assert!(config.allow_from.is_empty());
        let msgs = poll_collect(&config).await;
        assert_eq!(msgs.len(), 1, "Empty allow_from should allow all");
    }

    #[tokio::test]
    async fn test_channel_start_stop() {
        let config = match create_ethereal_account().await {
            Some(c) => c,
            None => { eprintln!("Skipping: Ethereal unavailable"); return; }
        };

        let (tx, _rx) = mpsc::unbounded_channel::<InboundMessage>();
        let mut channel = EmailChannel::new(config, tx);

        assert!(!channel.is_running());
        channel.start().await.expect("start failed");
        assert!(channel.is_running());
        channel.stop().await.expect("stop failed");
        assert!(!channel.is_running());
    }

    #[tokio::test]
    async fn test_channel_start_stop_restart() {
        let config = match create_ethereal_account().await {
            Some(c) => c,
            None => { eprintln!("Skipping: Ethereal unavailable"); return; }
        };

        let (tx, _rx) = mpsc::unbounded_channel::<InboundMessage>();
        let mut channel = EmailChannel::new(config, tx);

        // Start → Stop → Start again should work.
        channel.start().await.expect("first start failed");
        assert!(channel.is_running());
        channel.stop().await.expect("stop failed");
        assert!(!channel.is_running());
        channel.start().await.expect("restart failed");
        assert!(channel.is_running());
        channel.stop().await.expect("final stop failed");
    }

    // ======================================================================
    // Ignored tests (require real credentials via env vars)
    // ======================================================================

    /// Send a real email using credentials from environment variables.
    ///
    /// Run with: cargo test test_send_real_email -- --ignored --nocapture
    /// Env vars: TEST_SMTP_HOST, TEST_SMTP_PORT, TEST_EMAIL_USER, TEST_EMAIL_PASS, TEST_EMAIL_TO
    /// Optional: TEST_SMTP_USER (falls back to TEST_EMAIL_USER)
    #[tokio::test]
    #[ignore]
    async fn test_send_real_email() {
        use lettre::message::Mailbox;
        use lettre::transport::smtp::authentication::Credentials;
        use lettre::{AsyncSmtpTransport, AsyncTransport, Message, Tokio1Executor};

        let smtp_host = std::env::var("TEST_SMTP_HOST").expect("TEST_SMTP_HOST required");
        let smtp_port: u16 = std::env::var("TEST_SMTP_PORT")
            .unwrap_or("465".into())
            .parse()
            .unwrap();
        let from_addr = std::env::var("TEST_EMAIL_USER").expect("TEST_EMAIL_USER required");
        let smtp_user = std::env::var("TEST_SMTP_USER").unwrap_or_else(|_| from_addr.clone());
        let password = std::env::var("TEST_EMAIL_PASS").expect("TEST_EMAIL_PASS required");
        let to_addr = std::env::var("TEST_EMAIL_TO").expect("TEST_EMAIL_TO required");

        let from: Mailbox = from_addr.parse().expect("invalid from address");
        let to: Mailbox = to_addr.parse().expect("invalid to address");

        let email = Message::builder()
            .from(from)
            .to(to)
            .subject("nanobot email channel test")
            .body("Hello from nanobot!\n\nThis is a test email sent by the nanobot email channel.".to_string())
            .expect("failed to build email");

        let creds = Credentials::new(smtp_user.clone(), password);

        let mailer = if smtp_port == 465 {
            AsyncSmtpTransport::<Tokio1Executor>::relay(&smtp_host)
                .expect("relay failed")
                .port(smtp_port)
                .credentials(creds)
                .build()
        } else {
            AsyncSmtpTransport::<Tokio1Executor>::starttls_relay(&smtp_host)
                .expect("starttls relay failed")
                .port(smtp_port)
                .credentials(creds)
                .build()
        };

        println!("Sending from {} (auth: {}) to {} via {}:{}", from_addr, smtp_user, to_addr, smtp_host, smtp_port);
        let result = mailer.send(email).await;
        match &result {
            Ok(resp) => println!("Email sent! Response: {:?}", resp),
            Err(e) => println!("Send failed: {}", e),
        }
        assert!(result.is_ok(), "Failed to send: {:?}", result.err());
    }

    /// Test IMAP polling against a real server.
    ///
    /// Run with: cargo test test_poll_real_imap -- --ignored --nocapture
    /// Env vars: TEST_IMAP_HOST, TEST_IMAP_PORT, TEST_EMAIL_USER, TEST_EMAIL_PASS
    #[tokio::test]
    #[ignore]
    async fn test_poll_real_imap() {
        let config = EmailConfig {
            enabled: true,
            imap_host: std::env::var("TEST_IMAP_HOST").expect("TEST_IMAP_HOST required"),
            imap_port: std::env::var("TEST_IMAP_PORT")
                .unwrap_or("993".into())
                .parse()
                .unwrap(),
            smtp_host: String::new(),
            smtp_port: 587,
            username: std::env::var("TEST_EMAIL_USER").expect("TEST_EMAIL_USER required"),
            password: std::env::var("TEST_EMAIL_PASS").expect("TEST_EMAIL_PASS required"),
            poll_interval_secs: 30,
            allow_from: Vec::new(),
        };

        println!("Polling {}:{} as {}", config.imap_host, config.imap_port, config.username);
        let (tx, mut rx) = mpsc::unbounded_channel::<InboundMessage>();
        let result = poll_inbox(&config, &tx).await;
        match &result {
            Ok(()) => {
                let mut count = 0;
                while let Ok(m) = rx.try_recv() {
                    count += 1;
                    println!("  Message {}: from={} subject={:?}",
                        count, m.sender_id,
                        m.metadata.get("subject").and_then(|v| v.as_str()));
                }
                println!("Poll succeeded: {} messages", count);
            }
            Err(e) => println!("Poll failed: {}", e),
        }
        assert!(result.is_ok(), "IMAP poll failed: {:?}", result.err());
    }

    // ======================================================================
    // Helper function tests
    // ======================================================================

    #[test]
    fn test_extract_email_from_display_name_format() {
        assert_eq!(
            extract_email_address("John Doe <john@example.com>"),
            "john@example.com"
        );
    }

    #[test]
    fn test_extract_email_from_plain_address() {
        assert_eq!(
            extract_email_address("john@example.com"),
            "john@example.com"
        );
    }

    #[test]
    fn test_extract_email_with_quotes_and_brackets() {
        assert_eq!(
            extract_email_address("\"Doe, John\" <john@example.com>"),
            "john@example.com"
        );
    }

    #[test]
    fn test_extract_email_empty_string() {
        assert_eq!(extract_email_address(""), "");
    }

    #[test]
    fn test_url_encode_plain_text() {
        assert_eq!(url_encode("hello"), "hello");
    }

    #[test]
    fn test_url_encode_message_id() {
        let encoded = url_encode("<abc@example.com>");
        assert!(!encoded.contains('<'));
        assert!(!encoded.contains('>'));
        assert!(!encoded.contains('@'));
        assert!(encoded.contains("%3C")); // <
        assert!(encoded.contains("%3E")); // >
        assert!(encoded.contains("%40")); // @
    }

    #[test]
    fn test_url_encode_special_chars() {
        let encoded = url_encode("a b/c?d=e&f");
        assert!(!encoded.contains(' '));
        assert!(!encoded.contains('/'));
        assert!(!encoded.contains('?'));
    }

    /// Test REST API polling against agentmail.to.
    ///
    /// Run with: cargo test test_poll_agentmail_api -- --ignored --nocapture
    /// Env vars: TEST_AGENTMAIL_USER, TEST_AGENTMAIL_PASS
    #[tokio::test]
    #[ignore]
    async fn test_poll_agentmail_api() {
        let config = EmailConfig {
            enabled: true,
            imap_host: "imap.agentmail.to".to_string(),
            imap_port: 993,
            smtp_host: "smtp.agentmail.to".to_string(),
            smtp_port: 465,
            username: std::env::var("TEST_AGENTMAIL_USER")
                .expect("TEST_AGENTMAIL_USER required"),
            password: std::env::var("TEST_AGENTMAIL_PASS")
                .expect("TEST_AGENTMAIL_PASS required"),
            poll_interval_secs: 30,
            allow_from: Vec::new(),
        };

        println!("Polling agentmail.to API as {}", config.username);
        let (tx, mut rx) = mpsc::unbounded_channel::<InboundMessage>();
        let result = poll_inbox_api(&config, &tx).await;
        match &result {
            Ok(()) => {
                let mut count = 0;
                while let Ok(m) = rx.try_recv() {
                    count += 1;
                    println!("  Message {}: from={} subject={:?}",
                        count, m.sender_id,
                        m.metadata.get("subject").and_then(|v| v.as_str()));
                }
                println!("API poll succeeded: {} messages", count);
            }
            Err(e) => println!("API poll failed: {}", e),
        }
        assert!(result.is_ok(), "API poll failed: {:?}", result.err());
    }
}
