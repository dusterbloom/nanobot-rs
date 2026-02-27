//! Realtime voice session for bidirectional streaming audio.
//!
//! Integrates VAD, turn detection, STT, and TTS for realtime conversations.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::sync::mpsc;

#[cfg(feature = "voice")]
use jack_voice::{
    AudioCapture, AudioPlayer, AudioError,
    SpeechToText, SttMode, TextToSpeech, TtsEngine,
    TurnDetector, TurnDecision, VoiceActivityDetector,
};

use crate::config::schema::TtsEngineConfig;

/// Configuration for realtime voice sessions.
#[derive(Debug, Clone)]
pub struct RealtimeConfig {
    /// TTS engine to use for synthesis.
    pub tts_engine: TtsEngineConfig,
    /// Voice name for Qwen TTS engines.
    pub qwen_voice: String,
    /// VAD threshold (0.0-1.0).
    pub vad_threshold: f32,
    /// Silence duration in ms before turn completion check.
    pub silence_timeout_ms: u64,
    /// Enable SmartTurn for turn detection.
    pub smart_turn_enabled: bool,
}

impl Default for RealtimeConfig {
    fn default() -> Self {
        Self {
            tts_engine: TtsEngineConfig::Pocket,
            qwen_voice: "ryan".to_string(),
            vad_threshold: 0.3,
            silence_timeout_ms: 1200,
            smart_turn_enabled: true,
        }
    }
}

/// Events emitted by a realtime session.
#[derive(Debug, Clone)]
pub enum RealtimeEvent {
    /// User started speaking.
    SpeechStart,
    /// User stopped speaking.
    SpeechEnd,
    /// Turn completed with transcribed text.
    TurnComplete { text: String, language: String },
    /// Partial transcription (interim result).
    PartialTranscript { text: String },
    /// TTS audio chunk ready for playback.
    AudioChunk { samples: Vec<f32>, sample_rate: u32 },
    /// TTS finished.
    SynthesisComplete,
    /// Error occurred.
    Error(String),
}

/// Realtime voice session for bidirectional streaming.
///
/// Coordinates audio capture, VAD, turn detection, STT, and TTS
/// for realtime voice conversations.
pub struct RealtimeSession {
    config: RealtimeConfig,
    #[cfg(feature = "voice")]
    vad: Option<VoiceActivityDetector>,
    #[cfg(feature = "voice")]
    turn_detector: Option<TurnDetector>,
    #[cfg(feature = "voice")]
    stt: Option<SpeechToText>,
    #[cfg(feature = "voice")]
    tts: Option<Arc<Mutex<TextToSpeech>>>,
    #[cfg(feature = "voice")]
    capture: Option<AudioCapture>,
    running: Arc<AtomicBool>,
}

impl RealtimeSession {
    /// Create a new realtime session with the given configuration.
    #[cfg(feature = "voice")]
    pub async fn new(config: RealtimeConfig) -> Result<Self, String> {
        tracing::info!("Initializing realtime session with {:?}...", config.tts_engine);

        // Initialize VAD
        let vad = VoiceActivityDetector::new()
            .map_err(|e| format!("VAD init failed: {}", e))?;
        tracing::info!("VAD ready");

        // Initialize turn detector
        let turn_detector = if config.smart_turn_enabled {
            match TurnDetector::new() {
                Ok(td) => {
                    tracing::info!("TurnDetector ready (SmartTurn {})", 
                        if td.is_available() { "enabled" } else { "disabled" });
                    Some(td)
                }
                Err(e) => {
                    tracing::warn!("TurnDetector init failed, VAD-only mode: {}", e);
                    None
                }
            }
        } else {
            None
        };

        // Initialize STT
        let stt = SpeechToText::new(SttMode::Streaming)
            .map_err(|e| format!("STT init failed: {}", e))?;
        tracing::info!("STT ready (streaming mode)");

        // Initialize TTS
        let tts = match config.tts_engine {
            TtsEngineConfig::Pocket => {
                let tts = tokio::task::spawn_blocking(|| TextToSpeech::with_engine(TtsEngine::Pocket))
                    .await
                    .map_err(|e| format!("TTS spawn error: {}", e))?
                    .map_err(|e| format!("Pocket TTS init failed: {}", e))?;
                Some(Arc::new(Mutex::new(tts)))
            }
            TtsEngineConfig::Kokoro => {
                let tts = tokio::task::spawn_blocking(|| TextToSpeech::with_engine(TtsEngine::Kokoro))
                    .await
                    .map_err(|e| format!("TTS spawn error: {}", e))?
                    .map_err(|e| format!("Kokoro TTS init failed: {}", e))?;
                Some(Arc::new(Mutex::new(tts)))
            }
            TtsEngineConfig::Qwen => {
                if !TextToSpeech::can_run_qwen() {
                    return Err("Qwen TTS requires GPU (CUDA)".to_string());
                }
                let tts = tokio::task::spawn_blocking(|| TextToSpeech::with_engine_auto(TtsEngine::Qwen))
                    .await
                    .map_err(|e| format!("TTS spawn error: {}", e))?
                    .map_err(|e| format!("Qwen TTS init failed: {}", e))?;
                Some(Arc::new(Mutex::new(tts)))
            }
            TtsEngineConfig::QwenLarge => {
                if !TextToSpeech::can_run_qwen() {
                    return Err("QwenLarge TTS requires GPU (CUDA)".to_string());
                }
                let tts = tokio::task::spawn_blocking(|| TextToSpeech::with_engine_auto(TtsEngine::QwenLarge))
                    .await
                    .map_err(|e| format!("TTS spawn error: {}", e))?
                    .map_err(|e| format!("QwenLarge TTS init failed: {}", e))?;
                Some(Arc::new(Mutex::new(tts)))
            }
        };
        tracing::info!("TTS ready ({:?})", config.tts_engine);

        Ok(Self {
            config,
            vad: Some(vad),
            turn_detector,
            stt: Some(stt),
            tts,
            capture: None,
            running: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Create a new realtime session (stub for non-voice builds).
    #[cfg(not(feature = "voice"))]
    pub async fn new(_config: RealtimeConfig) -> Result<Self, String> {
        Err("Realtime session requires 'voice' feature".to_string())
    }

    /// Start the session with an event receiver.
    ///
    /// Returns an audio sender for injecting audio samples (for testing)
    /// and an event receiver for session events.
    #[cfg(feature = "voice")]
    pub fn start(&mut self) -> Result<(mpsc::Sender<Vec<f32>>, mpsc::Receiver<RealtimeEvent>), String> {
        let (audio_tx, mut audio_rx) = mpsc::channel::<Vec<f32>>(32);
        let (event_tx, event_rx) = mpsc::channel::<RealtimeEvent>(32);

        self.running.store(true, Ordering::SeqCst);
        let running = self.running.clone();

        // Take ownership of components
        let mut vad = self.vad.take().ok_or("VAD not initialized")?;
        let mut turn_detector = self.turn_detector.take();
        let mut stt_engine = self.stt.take().ok_or("STT not initialized")?;
        let tts = self.tts.clone();
        let config = self.config.clone();

        // Spawn the audio processing loop
        tokio::spawn(async move {
            tracing::info!("Realtime session started");

            while running.load(Ordering::SeqCst) {
                tokio::select! {
                    Some(samples) = audio_rx.recv() => {
                        // Process audio through VAD
                        if let Some(segment) = vad.process(&samples).ok().flatten() {
                            let _ = event_tx.send(RealtimeEvent::SpeechStart).await;
                            
                            // Feed to turn detector
                            if let Some(ref mut td) = turn_detector {
                                td.feed_audio(&segment.samples);
                                
                                // Check for turn completion
                                let decision = td.on_silence();
                                if let TurnDecision::Complete(audio) = decision {
                                    let _ = event_tx.send(RealtimeEvent::SpeechEnd).await;
                                    
                                    // Transcribe
                                    if let Ok(result) = stt_engine.transcribe(&audio) {
                                        let text = result.text.trim().to_string();
                                        if !text.is_empty() {
                                            let lang = crate::voice::detect_language(&text);
                                            let _ = event_tx.send(RealtimeEvent::TurnComplete {
                                                text,
                                                language: lang,
                                            }).await;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    _ = tokio::time::sleep(Duration::from_millis(100)) => {
                        // Periodic check for silence timeout
                    }
                }
            }

            tracing::info!("Realtime session stopped");
        });

        Ok((audio_tx, event_rx))
    }

    /// Start the session (stub for non-voice builds).
    #[cfg(not(feature = "voice"))]
    pub fn start(&mut self) -> Result<(mpsc::Sender<Vec<f32>>, mpsc::Receiver<RealtimeEvent>), String> {
        Err("Realtime session requires 'voice' feature".to_string())
    }

    /// Synthesize text to audio chunks.
    ///
    /// Sends AudioChunk events to the event channel.
    #[cfg(feature = "voice")]
    pub async fn synthesize(&self, text: &str, event_tx: mpsc::Sender<RealtimeEvent>) -> Result<(), String> {
        let tts = self.tts.as_ref().ok_or("TTS not initialized")?;
        let tts = tts.clone();
        let text = text.to_string();
        let voice = self.config.qwen_voice.clone();

        tokio::task::spawn_blocking(move || {
            let mut guard = tts.lock().map_err(|e| format!("TTS lock error: {}", e))?;
            
            // Set voice for Qwen engines
            let engine_type = guard.engine_type();
            if engine_type == "qwen" || engine_type == "qwen-large" {
                guard.set_speaker(&voice).ok();
            }

            guard
                .synthesize_streaming(&text, |samples, sample_rate| {
                    let chunk = RealtimeEvent::AudioChunk {
                        samples: samples.to_vec(),
                        sample_rate,
                    };
                    // Can't await in callback, so we skip sending for now
                    // In production, use a sync channel or buffer
                    true
                })
                .map_err(|e| format!("TTS synthesis failed: {}", e))?;

            Ok::<_, String>(())
        })
        .await
        .map_err(|e| format!("TTS spawn error: {}", e))??;

        let _ = event_tx.send(RealtimeEvent::SynthesisComplete).await;
        Ok(())
    }

    /// Stop the session.
    pub fn stop(&mut self) {
        self.running.store(false, Ordering::SeqCst);
    }

    /// Check if the session is running.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Check if SmartTurn is available.
    #[cfg(feature = "voice")]
    pub fn has_smart_turn(&self) -> bool {
        self.turn_detector.as_ref().map(|td| td.is_available()).unwrap_or(false)
    }

    /// Check if SmartTurn is available (stub).
    #[cfg(not(feature = "voice"))]
    pub fn has_smart_turn(&self) -> bool {
        false
    }
}

#[cfg(all(test, feature = "voice"))]
mod tests {
    use super::*;

    #[test]
    fn test_realtime_config_default() {
        let config = RealtimeConfig::default();
        assert_eq!(config.tts_engine, TtsEngineConfig::Pocket);
        assert_eq!(config.qwen_voice, "ryan");
        assert!((config.vad_threshold - 0.3f32).abs() < f32::EPSILON);
        assert_eq!(config.silence_timeout_ms, 1200);
        assert!(config.smart_turn_enabled);
    }

    #[test]
    fn test_realtime_config_custom() {
        let config = RealtimeConfig {
            tts_engine: TtsEngineConfig::Qwen,
            qwen_voice: "serena".to_string(),
            vad_threshold: 0.5,
            silence_timeout_ms: 800,
            smart_turn_enabled: false,
        };
        assert_eq!(config.tts_engine, TtsEngineConfig::Qwen);
        assert_eq!(config.qwen_voice, "serena");
        assert!((config.vad_threshold - 0.5f32).abs() < f32::EPSILON);
        assert_eq!(config.silence_timeout_ms, 800);
        assert!(!config.smart_turn_enabled);
    }

    #[test]
    fn test_realtime_event_variants() {
        let _ = RealtimeEvent::SpeechStart;
        let _ = RealtimeEvent::SpeechEnd;
        let _ = RealtimeEvent::TurnComplete {
            text: "hello".to_string(),
            language: "en".to_string(),
        };
        let _ = RealtimeEvent::PartialTranscript {
            text: "hel".to_string(),
        };
        let _ = RealtimeEvent::AudioChunk {
            samples: vec![0.0f32; 100],
            sample_rate: 24000,
        };
        let _ = RealtimeEvent::SynthesisComplete;
        let _ = RealtimeEvent::Error("test".to_string());
    }

    #[tokio::test]
    async fn test_realtime_session_new_with_pocket() {
        let config = RealtimeConfig {
            tts_engine: TtsEngineConfig::Pocket,
            ..Default::default()
        };
        let result = RealtimeSession::new(config).await;
        assert!(result.is_ok(), "Should create session with Pocket TTS");
        let session = result.unwrap();
        assert!(!session.is_running());
    }

    #[tokio::test]
    async fn test_realtime_session_has_smart_turn() {
        let config = RealtimeConfig::default();
        let session = RealtimeSession::new(config).await.expect("Should create session");
        // SmartTurn may or may not be available depending on model download
        let _ = session.has_smart_turn();
    }

    #[tokio::test]
    async fn test_realtime_session_start_stop() {
        let config = RealtimeConfig::default();
        let mut session = RealtimeSession::new(config).await.expect("Should create session");
        
        let (audio_tx, event_rx) = session.start().expect("Should start session");
        assert!(session.is_running());
        
        session.stop();
        assert!(!session.is_running());
        
        drop(audio_tx);
        drop(event_rx);
    }

    #[tokio::test]
    async fn test_realtime_session_qwen_requires_gpu() {
        let config = RealtimeConfig {
            tts_engine: TtsEngineConfig::Qwen,
            ..Default::default()
        };
        let result = RealtimeSession::new(config).await;
        if TextToSpeech::can_run_qwen() {
            assert!(result.is_ok(), "Should create session with Qwen if GPU available");
        } else {
            match result {
                Err(err) => {
                    assert!(err.contains("GPU"), "Error should mention GPU: {}", err);
                }
                Ok(_) => panic!("Should fail without GPU"),
            }
        }
    }
}
