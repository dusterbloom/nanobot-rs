//! Telegram channel implementation using the Bot API directly via reqwest.
//!
//! Uses long polling (`getUpdates`) so no public IP or webhook is needed.

use std::error::Error as StdError;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use regex::Regex;
use serde_json::{json, Value};
use tokio::sync::mpsc::UnboundedSender;
use tracing::{debug, error, info, warn};

use crate::bus::events::{InboundMessage, OutboundMessage};
use crate::channels::base::Channel;
use crate::config::schema::TelegramConfig;

#[cfg(feature = "voice")]
use crate::voice_pipeline::VoicePipeline;

/// Telegram channel using long-polling.
pub struct TelegramChannel {
    config: TelegramConfig,
    bus_tx: UnboundedSender<InboundMessage>,
    groq_api_key: String,
    running: Arc<AtomicBool>,
    client: reqwest::Client,
    #[cfg(feature = "voice")]
    voice_pipeline: Option<Arc<VoicePipeline>>,
}

impl TelegramChannel {
    /// Create a new `TelegramChannel`.
    pub fn new(
        config: TelegramConfig,
        bus_tx: UnboundedSender<InboundMessage>,
        groq_api_key: String,
        #[cfg(feature = "voice")] voice_pipeline: Option<Arc<VoicePipeline>>,
    ) -> Self {
        Self {
            config,
            bus_tx,
            groq_api_key,
            running: Arc::new(AtomicBool::new(false)),
            client: reqwest::Client::new(),
            #[cfg(feature = "voice")]
            voice_pipeline,
        }
    }

    /// Process a single Telegram update.
    async fn _on_message(
        client: &reqwest::Client,
        token: &str,
        bus_tx: &UnboundedSender<InboundMessage>,
        allow_from: &[String],
        update: &Value,
        _groq_api_key: &str,
        #[cfg(feature = "voice")] voice_pipeline: &Option<Arc<VoicePipeline>>,
    ) {
        let message = match update.get("message") {
            Some(m) => m,
            None => return,
        };

        let chat_id = message
            .get("chat")
            .and_then(|c| c.get("id"))
            .and_then(|id| id.as_i64())
            .unwrap_or(0);

        let user = match message.get("from") {
            Some(u) => u,
            None => return,
        };

        let user_id = user.get("id").and_then(|v| v.as_i64()).unwrap_or(0);
        let username = user.get("username").and_then(|v| v.as_str()).unwrap_or("");

        // Build composite sender_id.
        let sender_id = if username.is_empty() {
            user_id.to_string()
        } else {
            format!("{}|{}", user_id, username)
        };

        // Check allow list.
        if !allow_from.is_empty() {
            let allowed = allow_from.contains(&sender_id)
                || allow_from.contains(&user_id.to_string())
                || (!username.is_empty() && allow_from.contains(&username.to_string()));
            if !allowed {
                debug!(
                    "Telegram: ignoring message from non-allowed sender {}",
                    sender_id
                );
                return;
            }
        }

        // Collect content from text, caption, and media.
        let mut content_parts: Vec<String> = Vec::new();

        if let Some(text) = message.get("text").and_then(|v| v.as_str()) {
            content_parts.push(text.to_string());
        }
        if let Some(caption) = message.get("caption").and_then(|v| v.as_str()) {
            content_parts.push(caption.to_string());
        }

        // Handle photo.
        if let Some(photos) = message.get("photo").and_then(|v| v.as_array()) {
            if !photos.is_empty() {
                // Take the largest photo (last in the array).
                if let Some(file_id) = photos
                    .last()
                    .and_then(|p| p.get("file_id"))
                    .and_then(|v| v.as_str())
                {
                    let media_path =
                        Self::_download_file(client, token, file_id, "image", ".jpg").await;
                    if let Some(path) = media_path {
                        content_parts.push(format!("[image: {}]", path));
                    } else {
                        content_parts.push("[image: download failed]".to_string());
                    }
                }
            }
        }

        // Handle voice.
        #[allow(unused_mut)]
        let mut is_voice_message = false;
        #[allow(unused_mut)]
        let mut voice_file_path: Option<String> = None;
        #[allow(unused_mut)]
        let mut detected_language: Option<String> = None;
        if let Some(voice) = message.get("voice") {
            if let Some(file_id) = voice.get("file_id").and_then(|v| v.as_str()) {
                let media_path =
                    Self::_download_file(client, token, file_id, "voice", ".ogg").await;
                if let Some(path) = media_path {
                    #[cfg(feature = "voice")]
                    {
                        if let Some(ref pipeline) = voice_pipeline {
                            match pipeline.transcribe_file(&path).await {
                                Ok((text, lang)) => {
                                    info!(
                                        "Transcribed Telegram voice: \"{}\" (lang: {})",
                                        &text[..text.len().min(60)],
                                        lang
                                    );
                                    content_parts.push(text);
                                    is_voice_message = true;
                                    voice_file_path = Some(path);
                                    detected_language = Some(lang);
                                }
                                Err(e) => {
                                    warn!("Voice transcription failed: {}", e);
                                    content_parts.push(format!("[voice: {}]", path));
                                }
                            }
                        } else {
                            content_parts.push(format!("[voice: {}]", path));
                        }
                    }
                    #[cfg(not(feature = "voice"))]
                    {
                        content_parts.push(format!("[voice: {}]", path));
                    }
                } else {
                    content_parts.push("[voice: download failed]".to_string());
                }
            }
        }

        // Handle document.
        if let Some(doc) = message.get("document") {
            if let Some(file_id) = doc.get("file_id").and_then(|v| v.as_str()) {
                let ext = doc
                    .get("file_name")
                    .and_then(|v| v.as_str())
                    .and_then(|name| name.rsplit('.').next())
                    .map(|e| format!(".{}", e))
                    .unwrap_or_default();
                let media_path = Self::_download_file(client, token, file_id, "file", &ext).await;
                if let Some(path) = media_path {
                    content_parts.push(format!("[file: {}]", path));
                } else {
                    content_parts.push("[file: download failed]".to_string());
                }
            }
        }

        let content = if content_parts.is_empty() {
            "[empty message]".to_string()
        } else {
            content_parts.join("\n")
        };

        debug!(
            "Telegram message from {}: {}",
            sender_id,
            &content[..content.len().min(50)]
        );

        let message_id = message
            .get("message_id")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);

        let is_group = message
            .get("chat")
            .and_then(|c| c.get("type"))
            .and_then(|v| v.as_str())
            .map(|t| t != "private")
            .unwrap_or(false);

        let mut msg = InboundMessage::new("telegram", &sender_id, &chat_id.to_string(), &content);
        msg.metadata
            .insert("message_id".to_string(), json!(message_id));
        msg.metadata.insert("user_id".to_string(), json!(user_id));
        msg.metadata.insert("username".to_string(), json!(username));
        msg.metadata.insert("is_group".to_string(), json!(is_group));

        if is_voice_message {
            msg.metadata
                .insert("voice_message".to_string(), json!(true));
            if let Some(ref vf) = voice_file_path {
                msg.metadata.insert("voice_file".to_string(), json!(vf));
            }
            if let Some(ref lang) = detected_language {
                msg.metadata
                    .insert("detected_language".to_string(), json!(lang));
            }
        }

        let _ = bus_tx.send(msg);
    }

    /// Download a file from Telegram using the getFile + download URL pattern.
    async fn _download_file(
        client: &reqwest::Client,
        token: &str,
        file_id: &str,
        media_type: &str,
        ext: &str,
    ) -> Option<String> {
        // Step 1: getFile
        let url = format!(
            "https://api.telegram.org/bot{}/getFile?file_id={}",
            token, file_id
        );
        let resp = client.get(&url).send().await.ok()?;
        let data: Value = resp.json().await.ok()?;
        let file_path = data
            .get("result")
            .and_then(|r| r.get("file_path"))
            .and_then(|v| v.as_str())?;

        // Step 2: download
        let download_url = format!("https://api.telegram.org/file/bot{}/{}", token, file_path);
        let bytes = client
            .get(&download_url)
            .send()
            .await
            .ok()?
            .bytes()
            .await
            .ok()?;

        // Step 3: save locally
        let home = dirs::home_dir().unwrap_or_else(|| std::path::PathBuf::from("."));
        let media_dir = home.join(".nanobot").join("media");
        let _ = std::fs::create_dir_all(&media_dir);

        let short_id = &file_id[..file_id.len().min(16)];
        let file_name = format!("{}{}", short_id, ext);
        let local_path = media_dir.join(&file_name);
        std::fs::write(&local_path, &bytes).ok()?;

        debug!(
            "Downloaded {} ({} bytes) to {}",
            media_type,
            bytes.len(),
            local_path.display()
        );

        Some(local_path.to_string_lossy().to_string())
    }
}

#[async_trait]
impl Channel for TelegramChannel {
    fn name(&self) -> &str {
        "telegram"
    }

    async fn start(&mut self) -> Result<()> {
        if self.config.token.is_empty() {
            return Err(anyhow::anyhow!("Telegram bot token not configured"));
        }

        self.running.store(true, Ordering::SeqCst);

        let token = self.config.token.clone();
        let bus_tx = self.bus_tx.clone();
        let running = self.running.clone();
        let client = self.client.clone();
        let allow_from = self.config.allow_from.clone();
        let groq_api_key = self.groq_api_key.clone();
        #[cfg(feature = "voice")]
        let voice_pipeline = self.voice_pipeline.clone();

        info!("Starting Telegram bot (long-polling mode)...");

        // Spawn the long-polling loop.
        tokio::spawn(async move {
            let mut offset: i64 = 0;
            let base_url = format!("https://api.telegram.org/bot{}/getUpdates", token);

            while running.load(Ordering::SeqCst) {
                let body = json!({
                    "offset": offset,
                    "timeout": 30,
                    "allowed_updates": ["message"],
                });

                match client
                    .post(&base_url)
                    .json(&body)
                    .timeout(std::time::Duration::from_secs(35))
                    .send()
                    .await
                {
                    Ok(resp) => {
                        if let Ok(data) = resp.json::<Value>().await {
                            if let Some(updates) = data.get("result").and_then(|v| v.as_array()) {
                                for update in updates {
                                    if let Some(update_id) =
                                        update.get("update_id").and_then(|v| v.as_i64())
                                    {
                                        offset = update_id + 1;
                                    }
                                    TelegramChannel::_on_message(
                                        &client,
                                        &token,
                                        &bus_tx,
                                        &allow_from,
                                        update,
                                        &groq_api_key,
                                        #[cfg(feature = "voice")]
                                        &voice_pipeline,
                                    )
                                    .await;
                                }
                            }
                        }
                    }
                    Err(e) => {
                        // Log full error chain for debugging.
                        let mut cause = String::new();
                        let mut src: Option<&dyn StdError> = StdError::source(&e);
                        while let Some(s) = src {
                            cause.push_str(&format!(" -> {}", s));
                            src = s.source();
                        }
                        warn!("Telegram polling error: {}{}", e, cause);
                        tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                    }
                }
            }
        });

        Ok(())
    }

    async fn stop(&mut self) -> Result<()> {
        self.running.store(false, Ordering::SeqCst);
        info!("Telegram bot stopped");
        Ok(())
    }

    async fn send(&self, msg: &OutboundMessage) -> Result<()> {
        if self.config.token.is_empty() {
            return Err(anyhow::anyhow!("Telegram bot token not configured"));
        }

        let chat_id: i64 = msg
            .chat_id
            .parse()
            .map_err(|_| anyhow::anyhow!("Invalid chat_id: {}", msg.chat_id))?;

        // If this is a reply to a voice message, try to send a voice note.
        #[cfg(feature = "voice")]
        {
            let is_voice = msg
                .metadata
                .get("voice_message")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            if is_voice {
                if let Some(ref pipeline) = self.voice_pipeline {
                    let tts_text = crate::tui::strip_markdown_for_tts(&msg.content);
                    let lang = msg
                        .metadata
                        .get("detected_language")
                        .and_then(|v| v.as_str())
                        .unwrap_or("en");
                    if !tts_text.is_empty() {
                        match pipeline.synthesize_to_file(&tts_text, lang).await {
                            Ok(ogg_path) => {
                                let caption = if msg.content.len() > 1024 {
                                    let end = crate::utils::helpers::floor_char_boundary(&msg.content, 1024);
                                    &msg.content[..end]
                                } else {
                                    &msg.content
                                };
                                match self._send_voice(chat_id, &ogg_path, caption).await {
                                    Ok(()) => {
                                        // Clean up temp file
                                        let _ = std::fs::remove_file(&ogg_path);
                                        return Ok(());
                                    }
                                    Err(e) => {
                                        warn!("Failed to send voice, falling back to text: {}", e);
                                        let _ = std::fs::remove_file(&ogg_path);
                                        // Fall through to text send
                                    }
                                }
                            }
                            Err(e) => {
                                warn!("TTS synthesis failed, sending text only: {}", e);
                                // Fall through to text send
                            }
                        }
                    }
                }
            }
        }

        let html_content = markdown_to_telegram_html(&msg.content);
        let url = format!(
            "https://api.telegram.org/bot{}/sendMessage",
            self.config.token
        );

        let resp = self
            .client
            .post(&url)
            .json(&json!({
                "chat_id": chat_id,
                "text": html_content,
                "parse_mode": "HTML",
            }))
            .send()
            .await;

        match resp {
            Ok(r) if r.status().is_success() => Ok(()),
            Ok(_) => {
                // Fallback to plain text if HTML fails.
                warn!("HTML parse failed, falling back to plain text");
                let _ = self
                    .client
                    .post(&url)
                    .json(&json!({
                        "chat_id": chat_id,
                        "text": msg.content,
                    }))
                    .send()
                    .await;
                Ok(())
            }
            Err(e) => Err(anyhow::anyhow!("Failed to send Telegram message: {}", e)),
        }
    }

    fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }
}

#[cfg(feature = "voice")]
impl TelegramChannel {
    /// Send a voice note via Telegram sendVoice API.
    async fn _send_voice(&self, chat_id: i64, ogg_path: &str, caption: &str) -> Result<()> {
        let url = format!(
            "https://api.telegram.org/bot{}/sendVoice",
            self.config.token
        );

        let file_bytes = std::fs::read(ogg_path)
            .map_err(|e| anyhow::anyhow!("Failed to read voice file: {}", e))?;

        let part = reqwest::multipart::Part::bytes(file_bytes)
            .file_name("voice.ogg")
            .mime_str("audio/ogg")?;

        let form = reqwest::multipart::Form::new()
            .text("chat_id", chat_id.to_string())
            .part("voice", part)
            .text("caption", caption.to_string());

        let resp = self.client.post(&url).multipart(form).send().await?;

        if resp.status().is_success() {
            Ok(())
        } else {
            let body = resp.text().await.unwrap_or_default();
            Err(anyhow::anyhow!("sendVoice failed: {}", body))
        }
    }
}

// ---------------------------------------------------------------------------
// Markdown -> Telegram HTML conversion
// ---------------------------------------------------------------------------

/// Convert markdown to Telegram-safe HTML.
///
/// This is a port of the Python `_markdown_to_telegram_html` function.
pub fn markdown_to_telegram_html(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }

    // 1. Extract and protect code blocks.
    let mut code_blocks: Vec<String> = Vec::new();
    let re_codeblock = Regex::new(r"```[\w]*\n?([\s\S]*?)```").unwrap();
    let text = re_codeblock
        .replace_all(text, |caps: &regex::Captures| {
            let idx = code_blocks.len();
            code_blocks.push(caps[1].to_string());
            format!("\x00CB{}\x00", idx)
        })
        .to_string();

    // 2. Extract and protect inline code.
    let mut inline_codes: Vec<String> = Vec::new();
    let re_inline = Regex::new(r"`([^`]+)`").unwrap();
    let text = re_inline
        .replace_all(&text, |caps: &regex::Captures| {
            let idx = inline_codes.len();
            inline_codes.push(caps[1].to_string());
            format!("\x00IC{}\x00", idx)
        })
        .to_string();

    // 3. Headers -> plain text.
    let re_header = Regex::new(r"(?m)^#{1,6}\s+(.+)$").unwrap();
    let text = re_header.replace_all(&text, "$1").to_string();

    // 4. Blockquotes -> plain text.
    let re_blockquote = Regex::new(r"(?m)^>\s*(.*)$").unwrap();
    let text = re_blockquote.replace_all(&text, "$1").to_string();

    // 5. Escape HTML special characters.
    let text = text
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;");

    // 6. Links [text](url).
    let re_link = Regex::new(r"\[([^\]]+)\]\(([^)]+)\)").unwrap();
    let text = re_link
        .replace_all(&text, r#"<a href="$2">$1</a>"#)
        .to_string();

    // 7. Bold **text** or __text__.
    let re_bold_star = Regex::new(r"\*\*(.+?)\*\*").unwrap();
    let text = re_bold_star.replace_all(&text, "<b>$1</b>").to_string();
    let re_bold_under = Regex::new(r"__(.+?)__").unwrap();
    let text = re_bold_under.replace_all(&text, "<b>$1</b>").to_string();

    // 8. Italic _text_ (avoid matching inside words).
    // The regex crate does not support look-around, so we capture the
    // preceding and following non-alphanumeric characters and restore them.
    let re_italic = Regex::new(r"(^|[^a-zA-Z0-9])_([^_]+)_($|[^a-zA-Z0-9])").unwrap();
    let text = re_italic.replace_all(&text, "$1<i>$2</i>$3").to_string();

    // 9. Strikethrough ~~text~~.
    let re_strike = Regex::new(r"~~(.+?)~~").unwrap();
    let text = re_strike.replace_all(&text, "<s>$1</s>").to_string();

    // 10. Bullet lists.
    let re_bullet = Regex::new(r"(?m)^[-*]\s+").unwrap();
    let mut text = re_bullet.replace_all(&text, "\u{2022} ").to_string();

    // 11. Restore inline code.
    for (i, code) in inline_codes.iter().enumerate() {
        let escaped = code
            .replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;");
        text = text.replace(
            &format!("\x00IC{}\x00", i),
            &format!("<code>{}</code>", escaped),
        );
    }

    // 12. Restore code blocks.
    for (i, code) in code_blocks.iter().enumerate() {
        let escaped = code
            .replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;");
        text = text.replace(
            &format!("\x00CB{}\x00", i),
            &format!("<pre><code>{}</code></pre>", escaped),
        );
    }

    text
}

#[cfg(test)]
mod tests {
    use super::*;

    // ----- empty / plain text -----

    #[test]
    fn test_empty_string() {
        assert_eq!(markdown_to_telegram_html(""), "");
    }

    #[test]
    fn test_plain_text_passthrough() {
        assert_eq!(markdown_to_telegram_html("hello world"), "hello world");
    }

    // ----- HTML escaping -----

    #[test]
    fn test_html_escaping_angle_brackets() {
        let result = markdown_to_telegram_html("<script>alert('xss')</script>");
        assert!(result.contains("&lt;script&gt;"));
        assert!(result.contains("&lt;/script&gt;"));
        assert!(!result.contains("<script>"));
    }

    #[test]
    fn test_html_escaping_ampersand() {
        let result = markdown_to_telegram_html("A & B");
        assert_eq!(result, "A &amp; B");
    }

    // ----- bold -----

    #[test]
    fn test_bold_double_asterisk() {
        let result = markdown_to_telegram_html("this is **bold** text");
        assert_eq!(result, "this is <b>bold</b> text");
    }

    #[test]
    fn test_bold_double_underscore() {
        let result = markdown_to_telegram_html("this is __bold__ text");
        assert_eq!(result, "this is <b>bold</b> text");
    }

    // ----- italic -----

    #[test]
    fn test_italic_underscore() {
        let result = markdown_to_telegram_html("this is _italic_ text");
        assert_eq!(result, "this is <i>italic</i> text");
    }

    // ----- strikethrough -----

    #[test]
    fn test_strikethrough() {
        let result = markdown_to_telegram_html("this is ~~deleted~~ text");
        assert_eq!(result, "this is <s>deleted</s> text");
    }

    // ----- inline code -----

    #[test]
    fn test_inline_code() {
        let result = markdown_to_telegram_html("use `println!` macro");
        assert_eq!(result, "use <code>println!</code> macro");
    }

    #[test]
    fn test_inline_code_escapes_html_inside() {
        let result = markdown_to_telegram_html("try `<div>`");
        assert!(result.contains("<code>&lt;div&gt;</code>"));
    }

    // ----- code blocks -----

    #[test]
    fn test_code_block() {
        let input = "```\nlet x = 1;\n```";
        let result = markdown_to_telegram_html(input);
        assert!(result.contains("<pre><code>"));
        assert!(result.contains("let x = 1;"));
        assert!(result.contains("</code></pre>"));
    }

    #[test]
    fn test_code_block_with_language() {
        let input = "```rust\nfn main() {}\n```";
        let result = markdown_to_telegram_html(input);
        assert!(result.contains("<pre><code>"));
        assert!(result.contains("fn main() {}"));
    }

    #[test]
    fn test_code_block_escapes_html_inside() {
        let input = "```\n<b>not bold</b>\n```";
        let result = markdown_to_telegram_html(input);
        assert!(result.contains("&lt;b&gt;not bold&lt;/b&gt;"));
    }

    // ----- links -----

    #[test]
    fn test_link() {
        let result = markdown_to_telegram_html("[click here](https://example.com)");
        assert_eq!(result, r#"<a href="https://example.com">click here</a>"#);
    }

    // ----- headers -----

    #[test]
    fn test_header_stripped_to_plain_text() {
        let result = markdown_to_telegram_html("# Header");
        assert_eq!(result, "Header");
    }

    #[test]
    fn test_header_h2() {
        let result = markdown_to_telegram_html("## Sub Header");
        assert_eq!(result, "Sub Header");
    }

    #[test]
    fn test_header_h3() {
        let result = markdown_to_telegram_html("### Deep Header");
        assert_eq!(result, "Deep Header");
    }

    // ----- bullet lists -----

    #[test]
    fn test_bullet_list_dash() {
        let result = markdown_to_telegram_html("- item one\n- item two");
        assert!(result.contains("\u{2022} item one"));
        assert!(result.contains("\u{2022} item two"));
    }

    #[test]
    fn test_bullet_list_asterisk() {
        let result = markdown_to_telegram_html("* first\n* second");
        assert!(result.contains("\u{2022} first"));
        assert!(result.contains("\u{2022} second"));
    }

    // ----- blockquotes -----

    #[test]
    fn test_blockquote_stripped() {
        let result = markdown_to_telegram_html("> quoted text");
        assert_eq!(result, "quoted text");
    }

    // ----- combined formatting -----

    #[test]
    fn test_combined_formatting() {
        let input = "# Title\n\nSome **bold** and _italic_ text.\n\n- item 1\n- item 2\n\n`code`";
        let result = markdown_to_telegram_html(input);
        assert!(result.contains("Title"));
        assert!(result.contains("<b>bold</b>"));
        assert!(result.contains("<i>italic</i>"));
        assert!(result.contains("\u{2022} item 1"));
        assert!(result.contains("<code>code</code>"));
    }

    #[test]
    fn test_bold_and_link_together() {
        let input = "Check **this** [link](https://example.com)";
        let result = markdown_to_telegram_html(input);
        assert!(result.contains("<b>this</b>"));
        assert!(result.contains(r#"<a href="https://example.com">link</a>"#));
    }

    #[test]
    fn test_code_block_not_affected_by_formatting() {
        // Bold markers inside code blocks should not be converted.
        let input = "```\n**not bold**\n```";
        let result = markdown_to_telegram_html(input);
        // The ** should pass through unmodified inside the code block.
        assert!(!result.contains("<b>"));
    }

    #[test]
    fn test_inline_code_not_affected_by_formatting() {
        // Bold markers inside inline code should not be converted.
        let input = "see `**raw**` here";
        let result = markdown_to_telegram_html(input);
        assert!(!result.contains("<b>"));
        assert!(result.contains("<code>"));
    }

    #[test]
    fn test_multiline_message() {
        let input = "\
# Status Report

Deployment was **successful**.

## Details

- Service A: _running_
- Service B: ~~stopped~~ _restarted_

See [docs](https://docs.example.com) for more info.

```bash
systemctl restart service-b
```";
        let result = markdown_to_telegram_html(input);
        assert!(result.contains("Status Report"));
        assert!(result.contains("<b>successful</b>"));
        assert!(result.contains("<i>running</i>"));
        assert!(result.contains("<s>stopped</s>"));
        assert!(result.contains(r#"<a href="https://docs.example.com">docs</a>"#));
        assert!(result.contains("<pre><code>"));
        assert!(result.contains("systemctl restart service-b"));
    }
}
