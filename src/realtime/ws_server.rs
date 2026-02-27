//! WebSocket server for OpenAI-compatible realtime voice API.
//!
//! Provides a WebSocket endpoint that accepts audio input and streams
//! back audio output, compatible with OpenAI's Realtime API format.

use std::net::SocketAddr;
use std::sync::Arc;

use futures_util::{SinkExt, StreamExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;
use tokio_tungstenite::{tungstenite::Message, WebSocketStream};
use tracing::{error, info, warn};

use crate::config::schema::TtsEngineConfig;

/// Configuration for the realtime WebSocket server.
#[derive(Debug, Clone)]
pub struct RealtimeServerConfig {
    /// Port to listen on.
    pub port: u16,
    /// TTS engine to use.
    pub tts_engine: TtsEngineConfig,
    /// Voice name for TTS.
    pub voice: String,
    /// Host to bind to.
    pub host: String,
}

impl Default for RealtimeServerConfig {
    fn default() -> Self {
        Self {
            port: 8080,
            tts_engine: TtsEngineConfig::Pocket,
            voice: "ryan".to_string(),
            host: "127.0.0.1".to_string(),
        }
    }
}

/// WebSocket server for realtime voice API.
pub struct RealtimeServer {
    config: RealtimeServerConfig,
    shutdown: Arc<tokio::sync::Notify>,
}

impl RealtimeServer {
    /// Create a new realtime server.
    pub fn new(config: RealtimeServerConfig) -> Self {
        Self {
            config,
            shutdown: Arc::new(tokio::sync::Notify::new()),
        }
    }

    /// Start the server.
    pub async fn start(&self) -> Result<(), String> {
        let addr: SocketAddr = format!("{}:{}", self.config.host, self.config.port)
            .parse()
            .map_err(|e| format!("Invalid address: {}", e))?;

        let listener = TcpListener::bind(&addr)
            .await
            .map_err(|e| format!("Failed to bind to {}: {}", addr, e))?;

        info!("Realtime WebSocket server listening on ws://{}", addr);

        let shutdown = self.shutdown.clone();
        let tts_engine = self.config.tts_engine;
        let voice = self.config.voice.clone();

        tokio::spawn(async move {
            loop {
                tokio::select! {
                    accept_result = listener.accept() => {
                        match accept_result {
                            Ok((stream, peer_addr)) => {
                                info!("New WebSocket connection from {}", peer_addr);
                                let tts_engine = tts_engine;
                                let voice = voice.clone();
                                tokio::spawn(async move {
                                    if let Err(e) = handle_connection(stream, tts_engine, voice).await {
                                        error!("WebSocket error from {}: {}", peer_addr, e);
                                    }
                                });
                            }
                            Err(e) => {
                                error!("Failed to accept connection: {}", e);
                            }
                        }
                    }
                    _ = shutdown.notified() => {
                        info!("Realtime server shutting down");
                        break;
                    }
                }
            }
        });

        Ok(())
    }

    /// Stop the server.
    pub fn stop(&self) {
        self.shutdown.notify_waiters();
    }

    /// Get the server's listen address.
    pub fn addr(&self) -> String {
        format!("{}:{}", self.config.host, self.config.port)
    }
}

/// Handle a single WebSocket connection.
async fn handle_connection(
    stream: TcpStream,
    _tts_engine: TtsEngineConfig,
    _voice: String,
) -> Result<(), String> {
    let ws_stream = tokio_tungstenite::accept_async(stream)
        .await
        .map_err(|e| format!("WebSocket handshake failed: {}", e))?;

    info!("WebSocket connection established");

    let (mut ws_sender, mut ws_receiver) = ws_stream.split();

    // Send session.created event (OpenAI-compatible)
    let session_created = serde_json::json!({
        "type": "session.created",
        "session": {
            "id": uuid::Uuid::new_v4().to_string(),
            "model": "nanobot-realtime",
            "voice": _voice,
        }
    });

    ws_sender
        .send(Message::Text(session_created.to_string()))
        .await
        .map_err(|e| format!("Failed to send session.created: {}", e))?;

    // Handle incoming messages
    while let Some(msg_result) = ws_receiver.next().await {
        match msg_result {
            Ok(Message::Text(text)) => {
                // Parse the incoming event
                if let Ok(event) = serde_json::from_str::<serde_json::Value>(&text) {
                    if let Some(event_type) = event.get("type").and_then(|v| v.as_str()) {
                        match event_type {
                            "input_audio_buffer.append" => {
                                // Handle audio input
                                if let Some(_audio) = event.get("audio") {
                                    // TODO: Process audio through VAD/STT
                                    tracing::debug!("Received audio chunk");
                                }
                            }
                            "input_audio_buffer.commit" => {
                                // Commit the audio buffer for transcription
                                tracing::debug!("Audio buffer committed");
                            }
                            "session.update" => {
                                // Update session settings
                                tracing::debug!("Session update requested");
                            }
                            _ => {
                                warn!("Unknown event type: {}", event_type);
                            }
                        }
                    }
                }
            }
            Ok(Message::Binary(data)) => {
                // Handle binary audio data
                tracing::debug!("Received {} bytes of binary audio", data.len());
            }
            Ok(Message::Close(_)) => {
                info!("WebSocket connection closed by client");
                break;
            }
            Ok(Message::Ping(data)) => {
                ws_sender
                    .send(Message::Pong(data))
                    .await
                    .map_err(|e| format!("Failed to send pong: {}", e))?;
            }
            Err(e) => {
                error!("WebSocket error: {}", e);
                break;
            }
            _ => {}
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_realtime_server_config_default() {
        let config = RealtimeServerConfig::default();
        assert_eq!(config.port, 8080);
        assert_eq!(config.tts_engine, TtsEngineConfig::Pocket);
        assert_eq!(config.voice, "ryan");
        assert_eq!(config.host, "127.0.0.1");
    }

    #[test]
    fn test_realtime_server_config_custom() {
        let config = RealtimeServerConfig {
            port: 9000,
            tts_engine: TtsEngineConfig::Qwen,
            voice: "serena".to_string(),
            host: "0.0.0.0".to_string(),
        };
        assert_eq!(config.port, 9000);
        assert_eq!(config.tts_engine, TtsEngineConfig::Qwen);
        assert_eq!(config.voice, "serena");
        assert_eq!(config.host, "0.0.0.0");
    }

    #[test]
    fn test_realtime_server_new() {
        let config = RealtimeServerConfig::default();
        let server = RealtimeServer::new(config);
        assert_eq!(server.addr(), "127.0.0.1:8080");
    }

    #[tokio::test]
    async fn test_realtime_server_start_stop() {
        let config = RealtimeServerConfig {
            port: 18080, // Use non-standard port to avoid conflicts
            ..Default::default()
        };
        let server = RealtimeServer::new(config);
        
        let result = server.start().await;
        assert!(result.is_ok(), "Server should start: {:?}", result.err());
        
        // Give it a moment to start
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        
        server.stop();
        
        // Give it a moment to stop
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }

    #[tokio::test]
    async fn test_realtime_server_addr() {
        let config = RealtimeServerConfig {
            port: 9999,
            host: "localhost".to_string(),
            ..Default::default()
        };
        let server = RealtimeServer::new(config);
        assert_eq!(server.addr(), "localhost:9999");
    }
}
