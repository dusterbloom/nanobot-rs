//! Voice agent integration for realtime voice conversations with LLM.
//!
//! Combines RealtimeSession with AgentLoop for full voice-to-voice conversations:
//! 1. User speaks -> VAD detects -> STT transcribes
//! 2. Transcription -> LLM processes -> Streaming response
//! 3. LLM response -> TTS synthesizes -> Audio playback

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use tokio::sync::mpsc;

use crate::config::schema::TtsEngineConfig;

/// Configuration for the voice agent.
#[derive(Debug, Clone)]
pub struct VoiceAgentConfig {
    /// Realtime session config.
    pub realtime: super::RealtimeConfig,
    /// Session key for the LLM agent.
    pub session_key: String,
    /// Channel identifier for the agent.
    pub channel: String,
    /// Chat ID for the agent.
    pub chat_id: String,
    /// Use local LLM instead of cloud.
    pub local: bool,
    /// System prompt for voice mode.
    pub system_prompt: Option<String>,
}

impl Default for VoiceAgentConfig {
    fn default() -> Self {
        Self {
            realtime: super::RealtimeConfig::default(),
            session_key: "voice:default".to_string(),
            channel: "voice".to_string(),
            chat_id: "voice".to_string(),
            local: false,
            system_prompt: Some("You are a helpful voice assistant. Keep responses concise and conversational. Respond in the same language the user speaks.".to_string()),
        }
    }
}

/// Events emitted by the voice agent.
#[derive(Debug, Clone)]
pub enum VoiceAgentEvent {
    /// User started speaking.
    UserSpeechStart,
    /// User stopped speaking.
    UserSpeechEnd,
    /// User's transcribed text.
    UserText { text: String, language: String },
    /// LLM started responding.
    LlmResponseStart,
    /// LLM text delta (streaming).
    LlmTextDelta { delta: String },
    /// LLM response complete.
    LlmResponseComplete { full_text: String },
    /// TTS audio chunk ready.
    AudioChunk { samples: Vec<f32>, sample_rate: u32 },
    /// TTS finished playing.
    AudioComplete,
    /// Error occurred.
    Error(String),
    /// Agent is ready.
    Ready,
}

/// Voice agent that integrates realtime voice with LLM.
///
/// This is the main entry point for voice conversations with the AI agent.
/// It coordinates:
/// - Audio capture and VAD
/// - Speech-to-text transcription
/// - LLM processing with streaming responses
/// - Text-to-speech synthesis
/// - Audio playback
pub struct VoiceAgent {
    config: VoiceAgentConfig,
    running: Arc<AtomicBool>,
}

impl VoiceAgent {
    /// Create a new voice agent with the given configuration.
    pub fn new(config: VoiceAgentConfig) -> Self {
        Self {
            config,
            running: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Start the voice agent.
    ///
    /// Returns an event receiver for voice agent events.
    /// The agent will run until `stop()` is called.
    #[cfg(feature = "voice")]
    pub async fn start(&mut self) -> Result<mpsc::Receiver<VoiceAgentEvent>, String> {
        self.running.store(true, Ordering::SeqCst);
        let (event_tx, event_rx) = mpsc::channel(64);

        let realtime_config = self.config.realtime.clone();
        let running = self.running.clone();

        // Spawn the main voice agent loop in a dedicated thread
        // We create the RealtimeSession inside the thread to avoid Send issues
        std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("Failed to create tokio runtime");
            
            rt.block_on(async move {
                use super::{RealtimeConfig, RealtimeSession, RealtimeEvent};
                
                // Create session inside the dedicated runtime
                let session_result = RealtimeSession::new(realtime_config).await;
                
                let mut session = match session_result {
                    Ok(s) => s,
                    Err(e) => {
                        let _ = event_tx.send(VoiceAgentEvent::Error(format!("Failed to create session: {}", e))).await;
                        return;
                    }
                };
                
                let start_result = session.start();
                
                if let Err(e) = start_result {
                    let _ = event_tx.send(VoiceAgentEvent::Error(format!("Failed to start session: {}", e))).await;
                    return;
                }
                
                let (_audio_tx, mut realtime_rx) = start_result.unwrap();
                
                // Send Ready only after session is successfully started
                let _ = event_tx.send(VoiceAgentEvent::Ready).await;
                
                tracing::info!("Voice agent started");

                while running.load(Ordering::SeqCst) {
                    tokio::select! {
                        Some(event) = realtime_rx.recv() => {
                            match event {
                                RealtimeEvent::SpeechStart => {
                                    let _ = event_tx.send(VoiceAgentEvent::UserSpeechStart).await;
                                }
                                RealtimeEvent::SpeechEnd => {
                                    let _ = event_tx.send(VoiceAgentEvent::UserSpeechEnd).await;
                                }
                                RealtimeEvent::TurnComplete { text, language } => {
                                    let _ = event_tx.send(VoiceAgentEvent::UserText {
                                        text: text.clone(),
                                        language: language.clone(),
                                    }).await;

                                    // TODO: Process with LLM agent
                                    // For now, echo back via TTS
                                    let response = format!("You said: {}", text);
                                    
                                    let _ = event_tx.send(VoiceAgentEvent::LlmResponseStart).await;
                                    let _ = event_tx.send(VoiceAgentEvent::LlmTextDelta { delta: response.clone() }).await;
                                    let _ = event_tx.send(VoiceAgentEvent::LlmResponseComplete { full_text: response }).await;
                                }
                                RealtimeEvent::PartialTranscript { text } => {
                                    tracing::debug!("Partial: {}", text);
                                }
                                RealtimeEvent::AudioChunk { samples, sample_rate } => {
                                    let _ = event_tx.send(VoiceAgentEvent::AudioChunk { samples, sample_rate }).await;
                                }
                                RealtimeEvent::SynthesisComplete => {
                                    let _ = event_tx.send(VoiceAgentEvent::AudioComplete).await;
                                }
                                RealtimeEvent::Error(e) => {
                                    let _ = event_tx.send(VoiceAgentEvent::Error(e)).await;
                                }
                            }
                        }
                        _ = tokio::time::sleep(std::time::Duration::from_millis(100)) => {
                            // Periodic check
                        }
                    }
                }

                session.stop();
                tracing::info!("Voice agent stopped");
            });
        });

        Ok(event_rx)
    }

    /// Start the voice agent (stub for non-voice builds).
    #[cfg(not(feature = "voice"))]
    pub async fn start(&mut self) -> Result<mpsc::Receiver<VoiceAgentEvent>, String> {
        Err("Voice agent requires 'voice' feature".to_string())
    }

    /// Stop the voice agent.
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }

    /// Check if the voice agent is running.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }
}

#[cfg(all(test, feature = "voice"))]
mod tests {
    use super::*;

    #[test]
    fn test_voice_agent_config_default() {
        let config = VoiceAgentConfig::default();
        assert_eq!(config.session_key, "voice:default");
        assert_eq!(config.channel, "voice");
        assert_eq!(config.chat_id, "voice");
        assert!(!config.local);
        assert!(config.system_prompt.is_some());
    }

    #[test]
    fn test_voice_agent_config_custom() {
        let config = VoiceAgentConfig {
            realtime: super::super::RealtimeConfig {
                tts_engine: TtsEngineConfig::Qwen,
                qwen_voice: "serena".to_string(),
                ..Default::default()
            },
            session_key: "my-session".to_string(),
            channel: "telegram".to_string(),
            chat_id: "chat-123".to_string(),
            local: true,
            system_prompt: None,
        };
        assert_eq!(config.realtime.tts_engine, TtsEngineConfig::Qwen);
        assert_eq!(config.session_key, "my-session");
        assert!(config.local);
    }

    #[test]
    fn test_voice_agent_event_variants() {
        let _ = VoiceAgentEvent::UserSpeechStart;
        let _ = VoiceAgentEvent::UserSpeechEnd;
        let _ = VoiceAgentEvent::UserText { text: "hello".to_string(), language: "en".to_string() };
        let _ = VoiceAgentEvent::LlmResponseStart;
        let _ = VoiceAgentEvent::LlmTextDelta { delta: "Hi".to_string() };
        let _ = VoiceAgentEvent::LlmResponseComplete { full_text: "Hi there".to_string() };
        let _ = VoiceAgentEvent::AudioChunk { samples: vec![], sample_rate: 24000 };
        let _ = VoiceAgentEvent::AudioComplete;
        let _ = VoiceAgentEvent::Error("test".to_string());
        let _ = VoiceAgentEvent::Ready;
    }

    #[test]
    fn test_voice_agent_new() {
        let config = VoiceAgentConfig::default();
        let agent = VoiceAgent::new(config);
        assert!(!agent.is_running());
    }

    #[tokio::test]
    async fn test_voice_agent_start_stop() {
        let config = VoiceAgentConfig {
            realtime: super::super::RealtimeConfig {
                tts_engine: TtsEngineConfig::Pocket,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut agent = VoiceAgent::new(config);
        
        let mut event_rx = agent.start().await.expect("Should start agent");
        assert!(agent.is_running());
        
        // Wait for Ready event
        let ready = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            event_rx.recv()
        ).await;
        assert!(matches!(ready, Ok(Some(VoiceAgentEvent::Ready))), "Should receive Ready event");
        
        agent.stop();
        assert!(!agent.is_running());
    }

    #[tokio::test]
    async fn test_voice_agent_with_qwen_config() {
        let config = VoiceAgentConfig {
            realtime: super::super::RealtimeConfig {
                tts_engine: TtsEngineConfig::Qwen,
                qwen_voice: "ryan".to_string(),
                ..Default::default()
            },
            ..Default::default()
        };
        let mut agent = VoiceAgent::new(config);
        
        let result = agent.start().await;
        // Qwen requires GPU - may fail on systems without GPU
        if result.is_ok() {
            let _event_rx = result.unwrap();
            assert!(agent.is_running());
            agent.stop();
        } else {
            // Expected on systems without GPU
            let err = result.unwrap_err();
            assert!(err.contains("GPU") || err.contains("Qwen"));
        }
    }
}
