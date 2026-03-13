//! Realtime voice session for bidirectional streaming audio.
//!
//! Integrates VAD, turn detection, STT, and TTS for realtime conversations.

use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc;

#[cfg(feature = "voice")]
use jack_voice::{
    AudioCapture, AudioError, AudioPlayer, SpeechToText, SttMode, TextToSpeech, TtsEngine,
    TurnDecision, TurnDetector, VoiceActivityDetector,
};

use crate::config::schema::TtsEngineConfig;

/// Audio input mode for the realtime session.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum InputMode {
    /// VAD-based continuous listening (hands-free).
    #[default]
    Continuous,
    /// Push-to-talk: audio flows only while key held.
    PushToTalk,
}

/// Configuration for realtime voice sessions.
#[derive(Debug, Clone)]
pub struct RealtimeConfig {
    /// TTS engine to use for synthesis.
    pub tts_engine: TtsEngineConfig,
    /// VAD threshold (0.0-1.0).
    pub vad_threshold: f32,
    /// Silence duration in ms before turn completion check.
    pub silence_timeout_ms: u64,
    /// Enable SmartTurn for turn detection.
    pub smart_turn_enabled: bool,
    /// Audio input mode (continuous VAD or push-to-talk).
    pub input_mode: InputMode,
}

impl Default for RealtimeConfig {
    fn default() -> Self {
        Self {
            tts_engine: TtsEngineConfig::Pocket,
            vad_threshold: 0.3,
            silence_timeout_ms: 1200,
            smart_turn_enabled: true,
            input_mode: InputMode::default(),
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
    /// TTS playback finished (all audio played to speaker).
    TtsPlaybackComplete,
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
    /// English TTS engine (Pocket — fast, English-only).
    #[cfg(feature = "voice")]
    tts_en: Option<Arc<Mutex<TextToSpeech>>>,
    /// Multilingual TTS engine (Kokoro — supports 8 languages).
    #[cfg(feature = "voice")]
    tts_multi: Option<Arc<Mutex<TextToSpeech>>>,
    #[cfg(feature = "voice")]
    #[cfg(feature = "voice")]
    capture: Option<AudioCapture>,
    running: Arc<AtomicBool>,
}

impl RealtimeSession {
    /// Create a new realtime session with the given configuration.
    #[cfg(feature = "voice")]
    pub async fn new(config: RealtimeConfig) -> Result<Self, String> {
        tracing::info!(
            "Initializing realtime session with {:?}...",
            config.tts_engine
        );

        // Ensure required models are downloaded (VAD, STT, turn detector)
        let progress = jack_voice::LogProgress;
        for bundle in jack_voice::models::MODEL_BUNDLES {
            let target = if bundle.extract_dir.is_empty() {
                bundle.name
            } else {
                bundle.extract_dir
            };
            if !jack_voice::models::model_exists(target) {
                tracing::info!(
                    "Downloading model: {} ({}MB)...",
                    bundle.name,
                    bundle.size_mb
                );
                jack_voice::models::download_model(bundle, &progress)
                    .await
                    .map_err(|e| format!("Model download failed ({}): {}", bundle.name, e))?;
            }
        }
        // Initialize VAD
        let vad = VoiceActivityDetector::new().map_err(|e| format!("VAD init failed: {}", e))?;
        tracing::info!("VAD ready");

        // Initialize turn detector
        let turn_detector = if config.smart_turn_enabled {
            match TurnDetector::new() {
                Ok(td) => {
                    tracing::info!(
                        "TurnDetector ready (SmartTurn {})",
                        if td.is_available() {
                            "enabled"
                        } else {
                            "disabled"
                        }
                    );
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

        // Ensure Parakeet TDT model is downloaded (multilingual, 25 langs, ~600MB)
        if !jack_voice::models::parakeet_tdt_ready() {
            tracing::info!("Downloading Parakeet TDT model (600MB)...");
            jack_voice::models::ensure_parakeet_models(&progress)
                .await
                .map_err(|e| format!("Parakeet TDT download failed: {}", e))?;
        }

        // Initialize STT — Parakeet TDT batch mode (multilingual, 25 langs, ~10x faster than Whisper)
        // Falls back to Whisper Turbo if Parakeet unavailable
        let stt = SpeechToText::with_language(SttMode::Batch, Some(String::new()), None)
            .map_err(|e| format!("STT init failed: {}", e))?;
        tracing::info!("STT ready (Parakeet TDT primary, Whisper fallback)");

        // Initialize TTS — always load both Pocket (English) and Kokoro (multilingual)
        let tts_en = match tokio::task::spawn_blocking(|| {
            TextToSpeech::with_engine(TtsEngine::Pocket)
        })
        .await
        {
            Ok(Ok(tts)) => {
                tracing::info!("Pocket TTS ready (English)");
                Some(Arc::new(Mutex::new(tts)))
            }
            Ok(Err(e)) => {
                tracing::warn!("Pocket TTS init failed: {}", e);
                None
            }
            Err(e) => {
                tracing::warn!("Pocket TTS spawn error: {}", e);
                None
            }
        };

        let tts_multi = match tokio::task::spawn_blocking(|| {
            TextToSpeech::with_engine(TtsEngine::Kokoro)
        })
        .await
        {
            Ok(Ok(tts)) => {
                tracing::info!("Kokoro TTS ready (multilingual)");
                Some(Arc::new(Mutex::new(tts)))
            }
            Ok(Err(e)) => {
                tracing::warn!("Kokoro TTS init failed: {}", e);
                None
            }
            Err(e) => {
                tracing::warn!("Kokoro TTS spawn error: {}", e);
                None
            }
        };
        // })
        // .await
        // {
        //     Ok(Ok(tts)) => {
        //         Some(Arc::new(Mutex::new(tts)))
        //     }
        //     Ok(Err(e)) => {
        //         None
        //     }
        //     Err(e) => {
        //         None
        //     }
        // };

        tracing::info!(
            "TTS ready (en: {}, multi: {})",
            if tts_en.is_some() { "Pocket" } else { "none" },
            if tts_multi.is_some() {
                "Kokoro"
            } else {
                "none"
            },
        );

        Ok(Self {
            config,
            vad: Some(vad),
            turn_detector,
            stt: Some(stt),
            tts_en,
            tts_multi,
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
    pub fn start(
        &mut self,
    ) -> Result<(mpsc::Sender<Vec<f32>>, mpsc::Receiver<RealtimeEvent>), String> {
        let (audio_tx, mut audio_rx) = mpsc::channel::<Vec<f32>>(32);
        let (event_tx, event_rx) = mpsc::channel::<RealtimeEvent>(32);

        self.running.store(true, Ordering::SeqCst);
        let running = self.running.clone();

        // Start audio capture in continuous mode
        if self.config.input_mode == InputMode::Continuous {
            self.start_continuous_capture(audio_tx.clone())?;
        }

        // Take ownership of components
        let mut vad = self.vad.take().ok_or("VAD not initialized")?;
        let mut turn_detector = self.turn_detector.take();
        let mut stt_engine = self.stt.take().ok_or("STT not initialized")?;
        let config = self.config.clone();

        // Spawn the audio processing loop
        tokio::spawn(async move {
            tracing::info!("Realtime session started");
            let mut was_speaking = false;

            while running.load(Ordering::SeqCst) {
                tokio::select! {
                    Some(samples) = audio_rx.recv() => {
                        // Feed ALL audio to turn detector continuously (speech + silence).
                        if let Some(ref mut td) = turn_detector {
                            td.feed_audio(&samples);
                        }

                        // Process audio through VAD
                        match vad.process(&samples) {
                            Ok(Some(segment)) => {
                                // VAD returned a complete speech segment (speech ended).
                                tracing::debug!("VAD segment: {} samples", segment.samples.len());
                                let _ = event_tx.send(RealtimeEvent::SpeechEnd).await;
                                was_speaking = false;

                                // Try smart turn detection first
                                let audio_to_transcribe = if let Some(ref mut td) = turn_detector {
                                    let decision = td.on_silence();
                                    match decision {
                                        TurnDecision::Complete(audio) => {
                                            tracing::debug!("TurnDetector: Complete ({} samples)", audio.len());
                                            audio
                                        }
                                        _ => {
                                            // Turn detector says incomplete — use VAD segment directly.
                                            tracing::debug!("TurnDetector: not complete, using VAD segment");
                                            td.clear();
                                            segment.samples
                                        }
                                    }
                                } else {
                                    // No turn detector — use VAD segment directly.
                                    segment.samples
                                };

                                // Transcribe
                                if let Ok(result) = stt_engine.transcribe(&audio_to_transcribe) {
                                    let text = result.text.trim().to_string();
                                    if !text.is_empty() {
                                        let lang = crate::voice_pipeline::detect_language(&text);
                                        tracing::info!("Transcribed: \"{}\" ({})", text, lang);
                                        let _ = event_tx.send(RealtimeEvent::TurnComplete {
                                            text,
                                            language: lang,
                                        }).await;
                                    }
                                }
                            }
                            Ok(None) => {
                                // No segment yet — check if VAD sees speech starting.
                                // Use energy-gated check to reject speaker echo / ambient noise.
                                if vad.is_speech_with_energy(&samples) && !was_speaking {
                                    was_speaking = true;
                                    tracing::debug!("Speech detected");
                                    let _ = event_tx.send(RealtimeEvent::SpeechStart).await;
                                    if let Some(ref mut td) = turn_detector {
                                        td.on_speech_start();
                                    }
                                }
                            }
                            Err(e) => {
                                tracing::warn!("VAD error: {}", e);
                            }
                        }
                    }
                    _ = tokio::time::sleep(Duration::from_millis(100)) => {
                        // Periodic check
                    }
                }
            }

            tracing::info!("Realtime session stopped");
        });

        Ok((audio_tx, event_rx))
    }

    /// Start the session (stub for non-voice builds).
    #[cfg(not(feature = "voice"))]
    pub fn start(
        &mut self,
    ) -> Result<(mpsc::Sender<Vec<f32>>, mpsc::Receiver<RealtimeEvent>), String> {
        Err("Realtime session requires 'voice' feature".to_string())
    }

    /// Synthesize text to audio chunks.
    ///
    /// Sends AudioChunk events to the event channel.
    #[cfg(feature = "voice")]
    pub async fn synthesize(
        &self,
        text: &str,
        event_tx: mpsc::Sender<RealtimeEvent>,
    ) -> Result<(), String> {
        let tts = self
            .tts_en
            .as_ref()
            .or(self.tts_multi.as_ref())
            .ok_or("TTS not initialized")?;
        let tts = tts.clone();
        let text = text.to_string();

        tokio::task::spawn_blocking(move || {
            let mut guard = tts.lock();

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

    /// Start continuous audio capture via microphone.
    ///
    /// Creates a std::sync::mpsc channel, passes the sender to AudioCapture::start(),
    /// then spawns a bridge task that forwards chunks from the sync receiver to the
    /// async mpsc sender. Stores the capture handle for cleanup on stop().
    #[cfg(feature = "voice")]
    fn start_continuous_capture(&mut self, audio_tx: mpsc::Sender<Vec<f32>>) -> Result<(), String> {
        // AudioCapture::start() takes a std::sync::mpsc::Sender and writes directly to it
        let (cap_tx, cap_rx) = std::sync::mpsc::channel::<Vec<f32>>();

        let capture =
            AudioCapture::start(cap_tx).map_err(|e| format!("AudioCapture start failed: {}", e))?;

        self.capture = Some(capture);

        // Bridge: sync cap_rx -> async audio_tx
        // Uses spawn_blocking because cap_rx.recv() is blocking
        let running = self.running.clone();
        tokio::task::spawn_blocking(move || {
            while running.load(Ordering::SeqCst) {
                match cap_rx.recv() {
                    Ok(samples) => {
                        // try_send drops samples on backpressure to maintain real-time
                        if audio_tx.try_send(samples).is_err() {
                            tracing::debug!("Audio bridge: dropped samples (backpressure)");
                        }
                    }
                    Err(_) => break, // Capture stopped
                }
            }
        });

        tracing::info!("Continuous audio capture started");
        Ok(())
    }

    /// Stop the session.
    pub fn stop(&mut self) {
        self.running.store(false, Ordering::SeqCst);
        // Drop capture handle to stop audio recording
        self.capture = None;
    }

    /// Check if the session is running.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Get clones of all TTS engine handles for external use (e.g., voice agent playback).
    #[cfg(feature = "voice")]
    pub fn tts_handles(
        &self,
    ) -> (
        Option<Arc<Mutex<TextToSpeech>>>,
        Option<Arc<Mutex<TextToSpeech>>>,
    ) {
        (self.tts_en.clone(), self.tts_multi.clone())
    }

    /// Check if SmartTurn is available.
    #[cfg(feature = "voice")]
    pub fn has_smart_turn(&self) -> bool {
        self.turn_detector
            .as_ref()
            .map(|td| td.is_available())
            .unwrap_or(false)
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
        assert!((config.vad_threshold - 0.3f32).abs() < f32::EPSILON);
        assert_eq!(config.silence_timeout_ms, 1200);
        assert!(config.smart_turn_enabled);
    }

    #[test]
    fn test_realtime_config_custom() {
        let config = RealtimeConfig {
            tts_engine: TtsEngineConfig::Kokoro,
            vad_threshold: 0.5,
            silence_timeout_ms: 800,
            smart_turn_enabled: false,
            input_mode: InputMode::Continuous,
        };
        assert_eq!(config.tts_engine, TtsEngineConfig::Kokoro);
        assert!((config.vad_threshold - 0.5f32).abs() < f32::EPSILON);
        assert_eq!(config.silence_timeout_ms, 800);
        assert!(!config.smart_turn_enabled);
    }

    #[test]
    fn test_input_mode_default_is_continuous() {
        assert_eq!(InputMode::default(), InputMode::Continuous);
    }

    #[test]
    fn test_realtime_config_has_input_mode() {
        let config = RealtimeConfig::default();
        assert_eq!(config.input_mode, InputMode::Continuous);
    }

    #[test]
    fn test_realtime_config_ptt_mode() {
        let config = RealtimeConfig {
            input_mode: InputMode::PushToTalk,
            ..Default::default()
        };
        assert_eq!(config.input_mode, InputMode::PushToTalk);
    }

    #[test]
    fn test_continuous_mode_config_default() {
        let config = RealtimeConfig::default();
        assert_eq!(config.input_mode, InputMode::Continuous);
    }

    #[test]
    fn test_ptt_mode_config() {
        let config = RealtimeConfig {
            input_mode: InputMode::PushToTalk,
            ..Default::default()
        };
        assert_eq!(config.input_mode, InputMode::PushToTalk);
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
        let session = RealtimeSession::new(config)
            .await
            .expect("Should create session");
        // SmartTurn may or may not be available depending on model download
        let _ = session.has_smart_turn();
    }

    #[tokio::test]
    async fn test_realtime_session_start_stop() {
        // Use PushToTalk mode to avoid opening audio hardware (no device in CI/WSL)
        let config = RealtimeConfig {
            input_mode: InputMode::PushToTalk,
            ..Default::default()
        };
        let mut session = RealtimeSession::new(config)
            .await
            .expect("Should create session");

        let (audio_tx, event_rx) = session.start().expect("Should start session");
        assert!(session.is_running());

        session.stop();
        assert!(!session.is_running());

        drop(audio_tx);
        drop(event_rx);
    }

    #[tokio::test]
    async fn test_realtime_session_start_continuous_no_device() {
        // Continuous mode attempts AudioCapture; on systems without a mic it should
        // return an error rather than panic.
        let config = RealtimeConfig {
            input_mode: InputMode::Continuous,
            ..Default::default()
        };
        let result = RealtimeSession::new(config).await;
        if let Ok(mut session) = result {
            let start_result = session.start();
            // Either succeeds (device present) or returns a descriptive error (no device)
            match start_result {
                Ok(_) => {
                    session.stop();
                }
                Err(e) => {
                    assert!(
                        e.contains("AudioCapture") || e.contains("device") || e.contains("audio"),
                        "Error should mention audio capture: {}",
                        e
                    );
                }
            }
        }
    }
}
