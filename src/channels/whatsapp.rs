//! WhatsApp channel implementation using a Node.js WebSocket bridge.
//!
//! The bridge uses `whatsapp-web.js` to handle the WhatsApp Web protocol.
//! Communication between Rust and Node.js is via WebSocket.
//!
//! On startup, the channel auto-spawns the Node.js bridge as a child process.
//! Bridge files are embedded at compile time and extracted to
//! `~/.nanobot/bridge/whatsapp/` on first run.

use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio::sync::mpsc::UnboundedSender;
use tokio::sync::Mutex as TokioMutex;
use tokio_tungstenite::tungstenite::Message as WsMessage;
use tracing::{debug, error, info, warn};

use crate::bus::events::{InboundMessage, OutboundMessage};
use crate::channels::base::Channel;
use crate::config::schema::WhatsAppConfig;

#[cfg(feature = "voice")]
use crate::voice_pipeline::VoicePipeline;

/// Embedded bridge files (baked in at compile time for `cargo install` support).
const BRIDGE_INDEX_JS: &str = include_str!("../../bridge/whatsapp/index.js");
const BRIDGE_PACKAGE_JSON: &str = include_str!("../../bridge/whatsapp/package.json");

/// WhatsApp channel that connects to a Node.js bridge via WebSocket.
pub struct WhatsAppChannel {
    config: WhatsAppConfig,
    bus_tx: UnboundedSender<InboundMessage>,
    running: Arc<AtomicBool>,
    /// Sender for outgoing WebSocket messages (set once connected).
    ws_tx: Arc<TokioMutex<Option<UnboundedSender<String>>>>,
    /// Child bridge process (auto-spawned).
    bridge_process: Option<Child>,
    #[cfg(feature = "voice")]
    voice_pipeline: Option<Arc<VoicePipeline>>,
}

impl Drop for WhatsAppChannel {
    fn drop(&mut self) {
        if let Some(ref mut child) = self.bridge_process {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl WhatsAppChannel {
    /// Create a new `WhatsAppChannel`.
    pub fn new(
        config: WhatsAppConfig,
        bus_tx: UnboundedSender<InboundMessage>,
        #[cfg(feature = "voice")] voice_pipeline: Option<Arc<VoicePipeline>>,
    ) -> Self {
        Self {
            config,
            bus_tx,
            running: Arc::new(AtomicBool::new(false)),
            ws_tx: Arc::new(TokioMutex::new(None)),
            bridge_process: None,
            #[cfg(feature = "voice")]
            voice_pipeline,
        }
    }

    /// Resolve the bridge directory. Checks dev path first, then installed path.
    /// If neither has files, extracts embedded files to the installed path.
    fn resolve_bridge_dir() -> Result<PathBuf> {
        // Dev path: ./bridge/whatsapp/ (when running from repo checkout).
        let dev_path = PathBuf::from("bridge/whatsapp");
        if dev_path.join("index.js").exists() && dev_path.join("package.json").exists() {
            info!("Using dev bridge at {}", dev_path.display());
            return Ok(dev_path);
        }

        // Installed path: ~/.nanobot/bridge/whatsapp/
        let home = dirs::home_dir().context("Cannot determine home directory")?;
        let installed_path = home.join(".nanobot").join("bridge").join("whatsapp");

        if !installed_path.join("index.js").exists() {
            info!(
                "Extracting embedded WhatsApp bridge to {}",
                installed_path.display()
            );
            std::fs::create_dir_all(&installed_path)?;
            std::fs::write(installed_path.join("index.js"), BRIDGE_INDEX_JS)?;
            std::fs::write(installed_path.join("package.json"), BRIDGE_PACKAGE_JSON)?;
        }

        Ok(installed_path)
    }

    /// Run `npm install` in the bridge directory if `node_modules/` doesn't exist.
    fn ensure_npm_deps(bridge_dir: &PathBuf) -> Result<()> {
        if bridge_dir.join("node_modules").exists() {
            return Ok(());
        }

        info!("Installing WhatsApp bridge dependencies...");
        let status = Command::new("npm")
            .arg("install")
            .arg("--no-audit")
            .arg("--no-fund")
            .current_dir(bridge_dir)
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .status()
            .context("Failed to run npm install — is Node.js installed?")?;

        if !status.success() {
            return Err(anyhow::anyhow!("npm install failed with status {}", status));
        }

        Ok(())
    }

    /// Spawn the Node.js bridge process.
    fn spawn_bridge(bridge_dir: &PathBuf, port: u16) -> Result<Child> {
        info!("Starting WhatsApp bridge on port {}...", port);
        let child = Command::new("node")
            .arg("index.js")
            .arg("--port")
            .arg(port.to_string())
            .current_dir(bridge_dir)
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .spawn()
            .context("Failed to spawn bridge process — is Node.js installed?")?;

        Ok(child)
    }

    /// Handle a JSON message from the bridge.
    async fn _handle_bridge_message(
        data: &Value,
        bus_tx: &UnboundedSender<InboundMessage>,
        allow_from: &[String],
        #[cfg(feature = "voice")] voice_pipeline: &Option<Arc<VoicePipeline>>,
    ) {
        let msg_type = data.get("type").and_then(|v| v.as_str()).unwrap_or("");

        match msg_type {
            "message" => {
                let sender = data.get("sender").and_then(|v| v.as_str()).unwrap_or("");
                let content = data.get("content").and_then(|v| v.as_str()).unwrap_or("");

                // Extract phone number from JID (phone@s.whatsapp.net).
                let chat_id = if sender.contains('@') {
                    sender.split('@').next().unwrap_or(sender)
                } else {
                    sender
                };

                // Check allow list.
                if !allow_from.is_empty()
                    && !allow_from.contains(&chat_id.to_string())
                    && !allow_from.contains(&sender.to_string())
                {
                    debug!(
                        "WhatsApp: ignoring message from non-allowed sender {}",
                        chat_id
                    );
                    return;
                }

                #[allow(unused_mut)]
                let mut is_voice_message = false;
                #[allow(unused_mut)]
                let mut voice_file_path: Option<String> = None;
                #[allow(unused_mut)]
                let mut detected_language: Option<String> = None;
                let voice_file = data.get("voiceFile").and_then(|v| v.as_str());

                let content = if let Some(vf) = voice_file {
                    // Voice message with downloaded file — try to transcribe.
                    #[cfg(feature = "voice")]
                    {
                        if let Some(ref pipeline) = voice_pipeline {
                            match pipeline.transcribe_file(vf).await {
                                Ok((text, lang)) => {
                                    info!(
                                        "Transcribed WhatsApp voice: \"{}\" (lang: {})",
                                        &text[..text.len().min(60)],
                                        lang
                                    );
                                    is_voice_message = true;
                                    voice_file_path = Some(vf.to_string());
                                    detected_language = Some(lang);
                                    text
                                }
                                Err(e) => {
                                    warn!("WhatsApp voice transcription failed: {}", e);
                                    format!("[voice: {}]", vf)
                                }
                            }
                        } else {
                            format!("[voice: {}]", vf)
                        }
                    }
                    #[cfg(not(feature = "voice"))]
                    {
                        format!("[voice: {}]", vf)
                    }
                } else if content == "[Voice Message]" {
                    "[Voice Message: Transcription not available for WhatsApp yet]".to_string()
                } else {
                    content.to_string()
                };

                let is_group = data
                    .get("isGroup")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);

                let mut msg = InboundMessage::new("whatsapp", chat_id, sender, &content);
                if let Some(id) = data.get("id").and_then(|v| v.as_str()) {
                    msg.metadata.insert("message_id".to_string(), json!(id));
                }
                if let Some(ts) = data.get("timestamp") {
                    msg.metadata.insert("timestamp".to_string(), ts.clone());
                }
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
            "status" => {
                let status = data
                    .get("status")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                info!("WhatsApp status: {}", status);
            }
            "qr" => {
                info!("Scan QR code in the bridge terminal to connect WhatsApp");
            }
            "error" => {
                let err = data
                    .get("error")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown error");
                error!("WhatsApp bridge error: {}", err);
            }
            other => {
                debug!("WhatsApp bridge: unknown message type '{}'", other);
            }
        }
    }
}

#[async_trait]
impl Channel for WhatsAppChannel {
    fn name(&self) -> &str {
        "whatsapp"
    }

    async fn start(&mut self) -> Result<()> {
        self.running.store(true, Ordering::SeqCst);

        // Auto-spawn the bridge if no explicit bridge_url is configured.
        if self.config.bridge_url.is_none() {
            let bridge_dir = Self::resolve_bridge_dir()?;
            Self::ensure_npm_deps(&bridge_dir)?;
            let child = Self::spawn_bridge(&bridge_dir, self.config.bridge_port)?;
            self.bridge_process = Some(child);
        }

        let bridge_url = self.config.effective_bridge_url();
        let bus_tx = self.bus_tx.clone();
        let running = self.running.clone();
        let ws_tx_slot = self.ws_tx.clone();
        let allow_from = self.config.allow_from.clone();
        #[cfg(feature = "voice")]
        let voice_pipeline = self.voice_pipeline.clone();

        info!("Connecting to WhatsApp bridge at {}...", bridge_url);

        tokio::spawn(async move {
            while running.load(Ordering::SeqCst) {
                match tokio_tungstenite::connect_async(&bridge_url).await {
                    Ok((ws_stream, _)) => {
                        info!("Connected to WhatsApp bridge");
                        let (write, mut read) = ws_stream.split();

                        // Create an mpsc channel to send messages to the WebSocket.
                        let (out_tx, mut out_rx) = tokio::sync::mpsc::unbounded_channel::<String>();

                        // Store the sender so send() can use it.
                        {
                            let mut slot = ws_tx_slot.lock().await;
                            *slot = Some(out_tx);
                        }

                        // Spawn writer task.
                        let write_arc = Arc::new(TokioMutex::new(write));
                        let write_arc_clone = write_arc.clone();
                        let writer_running = running.clone();
                        let writer_handle = tokio::spawn(async move {
                            while writer_running.load(Ordering::SeqCst) {
                                match out_rx.recv().await {
                                    Some(text) => {
                                        let mut w = write_arc_clone.lock().await;
                                        if w.send(WsMessage::Text(text)).await.is_err() {
                                            break;
                                        }
                                    }
                                    None => break,
                                }
                            }
                        });

                        // Read loop.
                        while let Some(msg_result) = read.next().await {
                            match msg_result {
                                Ok(WsMessage::Text(text)) => {
                                    match serde_json::from_str::<Value>(&text) {
                                        Ok(data) => {
                                            Self::_handle_bridge_message(
                                                &data,
                                                &bus_tx,
                                                &allow_from,
                                                #[cfg(feature = "voice")]
                                                &voice_pipeline,
                                            )
                                            .await;
                                        }
                                        Err(_) => {
                                            warn!(
                                                "Invalid JSON from bridge: {}",
                                                &text[..text.len().min(100)]
                                            );
                                        }
                                    }
                                }
                                Ok(WsMessage::Close(_)) => {
                                    info!("WhatsApp bridge closed connection");
                                    break;
                                }
                                Err(e) => {
                                    warn!("WhatsApp WebSocket error: {}", e);
                                    break;
                                }
                                _ => {}
                            }
                        }

                        // Clean up.
                        {
                            let mut slot = ws_tx_slot.lock().await;
                            *slot = None;
                        }
                        writer_handle.abort();
                    }
                    Err(e) => {
                        warn!("WhatsApp bridge connection error: {}", e);
                    }
                }

                if running.load(Ordering::SeqCst) {
                    info!("Reconnecting to WhatsApp bridge in 5 seconds...");
                    tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                }
            }
        });

        Ok(())
    }

    async fn stop(&mut self) -> Result<()> {
        self.running.store(false, Ordering::SeqCst);
        {
            let mut slot = self.ws_tx.lock().await;
            *slot = None;
        }
        // Kill the bridge child process if we spawned it.
        if let Some(ref mut child) = self.bridge_process {
            info!("Stopping WhatsApp bridge process...");
            let _ = child.kill();
            let _ = child.wait();
            self.bridge_process = None;
        }
        info!("WhatsApp channel stopped");
        Ok(())
    }

    async fn send(&self, msg: &OutboundMessage) -> Result<()> {
        let slot = self.ws_tx.lock().await;
        let tx = match slot.as_ref() {
            Some(tx) => tx.clone(),
            None => {
                warn!("WhatsApp bridge not connected");
                return Err(anyhow::anyhow!("WhatsApp bridge not connected"));
            }
        };
        drop(slot);

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
                                if let Ok(bytes) = std::fs::read(&ogg_path) {
                                    use base64::Engine;
                                    let b64 =
                                        base64::engine::general_purpose::STANDARD.encode(&bytes);
                                    let media_payload = json!({
                                        "type": "sendMedia",
                                        "to": msg.chat_id,
                                        "media": b64,
                                        "mimetype": "audio/ogg",
                                        "filename": "voice.ogg",
                                        "caption": msg.content,
                                    });
                                    let _ = tx.send(
                                        serde_json::to_string(&media_payload).unwrap_or_default(),
                                    );
                                    let _ = std::fs::remove_file(&ogg_path);
                                    return Ok(());
                                }
                                let _ = std::fs::remove_file(&ogg_path);
                            }
                            Err(e) => {
                                warn!("WhatsApp TTS synthesis failed, sending text only: {}", e);
                            }
                        }
                    }
                }
            }
        }

        let payload = json!({
            "type": "send",
            "to": msg.chat_id,
            "text": msg.content,
        });

        tx.send(serde_json::to_string(&payload).unwrap_or_default())
            .map_err(|e| anyhow::anyhow!("Failed to send WhatsApp message: {}", e))?;

        Ok(())
    }

    fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }
}
