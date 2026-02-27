#![cfg(feature = "voice")]

//! Voice pipeline for channel voice message transcription and TTS synthesis.
//!
//! Provides `VoicePipeline` which wraps jack_voice STT/TTS models and uses
//! ffmpeg subprocesses for audio codec conversion. Designed to be shared
//! across channels via `Arc<VoicePipeline>`.

use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};

use crate::config::schema::TtsEngineConfig;
use crate::voice::{detect_language, split_tts_sentences};
use jack_voice::{
    models::{self, ModelProgressCallback},
    SpeechToText, SttMode, TextToSpeech, TtsEngine,
};
use tracing::{debug, info};

/// Shared voice pipeline for channel voice message processing.
///
/// Both STT and TTS are CPU-bound and use `std::sync::Mutex` so they can be
/// held across `spawn_blocking` boundaries. Wrapped in `Arc` so they can
/// be moved into the blocking closure.
///
/// Holds TTS engines based on configuration:
/// - Pocket (fast English-only, CPU inference)
/// - Kokoro (multilingual, CPU inference)
/// - Qwen/QwenLarge (multilingual with voice cloning, requires GPU)
pub struct VoicePipeline {
    stt: Arc<Mutex<SpeechToText>>,
    /// Pocket TTS engine (fast, English-only). `None` if not configured or init failed.
    tts_en: Option<Arc<Mutex<TextToSpeech>>>,
    /// Kokoro TTS engine (multilingual). `None` if not configured or init failed.
    tts_multi: Option<Arc<Mutex<TextToSpeech>>>,
    /// Qwen TTS engine (multilingual with GPU). `None` if not configured or init failed.
    tts_qwen: Option<Arc<Mutex<TextToSpeech>>>,
    /// Selected TTS engine config.
    engine_config: TtsEngineConfig,
    /// Qwen voice name (for Qwen/QwenLarge engines).
    qwen_voice: String,
}

impl VoicePipeline {
    /// Create a new voice pipeline with default engines (Pocket + Kokoro).
    ///
    /// Initializes both Pocket (English) and Kokoro (multilingual) TTS
    /// engines. If one fails, the other is used as fallback.
    pub async fn new() -> Result<Self, String> {
        Self::with_engine(TtsEngineConfig::Pocket).await
    }

    /// Create a voice pipeline with a specific TTS engine.
    ///
    /// This allows selecting Qwen/QwenLarge engines which require GPU.
    pub async fn with_engine(engine: TtsEngineConfig) -> Result<Self, String> {
        info!("Initializing voice pipeline for channels ({:?})...", engine);

        let progress = &LogProgress;
        for bundle in models::MODEL_BUNDLES {
            let target = if bundle.extract_dir.is_empty() {
                bundle.name
            } else {
                bundle.extract_dir
            };
            if !models::model_exists(target) {
                progress.on_download_start(bundle.name, bundle.size_mb);
                models::download_model(bundle, progress)
                    .await
                    .map_err(|e| format!("Model download failed: {e}"))?;
                progress.on_download_complete(bundle.name);
            }
        }

        let stt = SpeechToText::new(SttMode::Batch).map_err(|e| format!("STT init failed: {e}"))?;

        let (tts_en, tts_multi, tts_qwen) = match engine {
            TtsEngineConfig::Pocket => {
                // Also load Kokoro as fallback for non-English
                models::ensure_kokoro_model(progress)
                    .await
                    .map_err(|e| format!("Model download failed: {e}"))?;

                let tts_en = match tokio::task::spawn_blocking(|| TextToSpeech::with_engine(TtsEngine::Pocket))
                    .await
                    .map_err(|e| format!("spawn_blocking join error: {e}"))?
                {
                    Ok(tts) => {
                        info!("Pocket TTS ready (English)");
                        Some(Arc::new(Mutex::new(tts)))
                    }
                    Err(e) => {
                        info!("Pocket TTS not available: {e}");
                        None
                    }
                };

                let tts_multi = match tokio::task::spawn_blocking(|| TextToSpeech::with_engine(TtsEngine::Kokoro))
                    .await
                    .map_err(|e| format!("spawn_blocking join error: {e}"))?
                {
                    Ok(tts) => {
                        info!("Kokoro TTS ready (multilingual fallback)");
                        Some(Arc::new(Mutex::new(tts)))
                    }
                    Err(e) => {
                        info!("Kokoro TTS not available: {e}");
                        None
                    }
                };

                (tts_en, tts_multi, None)
            }
            TtsEngineConfig::Kokoro => {
                models::ensure_kokoro_model(progress)
                    .await
                    .map_err(|e| format!("Model download failed: {e}"))?;

                let tts = tokio::task::spawn_blocking(|| TextToSpeech::with_engine(TtsEngine::Kokoro))
                    .await
                    .map_err(|e| format!("spawn_blocking join error: {e}"))?
                    .map_err(|e| format!("Kokoro TTS init failed: {e}"))?;
                info!("Kokoro TTS ready (multilingual)");
                (None, Some(Arc::new(Mutex::new(tts))), None)
            }
            TtsEngineConfig::Qwen => {
                if !TextToSpeech::can_run_qwen() {
                    return Err("Qwen TTS requires GPU (CUDA). No GPU detected.".to_string());
                }
                let tts = tokio::task::spawn_blocking(|| TextToSpeech::with_engine_auto(TtsEngine::Qwen))
                    .await
                    .map_err(|e| format!("spawn_blocking join error: {e}"))?
                    .map_err(|e| format!("Qwen TTS init failed: {e}"))?;
                info!("Qwen TTS ready");
                (None, None, Some(Arc::new(Mutex::new(tts))))
            }
            TtsEngineConfig::QwenLarge => {
                if !TextToSpeech::can_run_qwen() {
                    return Err("QwenLarge TTS requires GPU (CUDA). No GPU detected.".to_string());
                }
                let tts = tokio::task::spawn_blocking(|| TextToSpeech::with_engine_auto(TtsEngine::QwenLarge))
                    .await
                    .map_err(|e| format!("spawn_blocking join error: {e}"))?
                    .map_err(|e| format!("QwenLarge TTS init failed: {e}"))?;
                info!("QwenLarge TTS ready (voice cloning enabled)");
                (None, None, Some(Arc::new(Mutex::new(tts))))
            }
            TtsEngineConfig::QwenOnnx => {
                models::ensure_qwen_onnx_model(false, progress)
                    .await
                    .map_err(|e| format!("ONNX model download failed: {e}"))?;
                let tts = tokio::task::spawn_blocking(|| TextToSpeech::with_engine(TtsEngine::QwenOnnx))
                    .await
                    .map_err(|e| format!("spawn_blocking join error: {e}"))?
                    .map_err(|e| format!("QwenOnnx TTS init failed: {e}"))?;
                info!("QwenOnnx TTS ready (ONNX Runtime)");
                (None, None, Some(Arc::new(Mutex::new(tts))))
            }
            TtsEngineConfig::QwenOnnxInt8 => {
                models::ensure_qwen_onnx_model(true, progress)
                    .await
                    .map_err(|e| format!("ONNX INT8 model download failed: {e}"))?;
                let tts = tokio::task::spawn_blocking(|| TextToSpeech::with_engine(TtsEngine::QwenOnnxInt8))
                    .await
                    .map_err(|e| format!("spawn_blocking join error: {e}"))?
                    .map_err(|e| format!("QwenOnnxInt8 TTS init failed: {e}"))?;
                info!("QwenOnnxInt8 TTS ready (ONNX Runtime, quantized)");
                (None, None, Some(Arc::new(Mutex::new(tts))))
            }
        };

        if tts_en.is_none() && tts_multi.is_none() && tts_qwen.is_none() {
            return Err("No TTS engine could be initialized".to_string());
        }

        info!("Voice pipeline ready");

        Ok(Self {
            stt: Arc::new(Mutex::new(stt)),
            tts_en,
            tts_multi,
            tts_qwen,
            engine_config: engine,
            qwen_voice: "ryan".to_string(),
        })
    }

    /// Set the Qwen voice name for Qwen/QwenLarge engines.
    pub fn set_qwen_voice(&mut self, voice: &str) {
        self.qwen_voice = voice.to_string();
    }

    /// Get the current TTS engine config.
    pub fn engine_config(&self) -> TtsEngineConfig {
        self.engine_config
    }

    /// Transcribe an audio file (e.g. `.ogg`) to text.
    ///
    /// Decodes via ffmpeg to f32le 16kHz mono, then runs STT.
    /// Returns `(text, detected_language_code)` where language is ISO 639-1.
    pub async fn transcribe_file(&self, path: &str) -> Result<(String, String), String> {
        let path = path.to_string();
        let samples = tokio::task::spawn_blocking(move || decode_audio_file(&path))
            .await
            .map_err(|e| format!("spawn_blocking join error: {e}"))??;

        if samples.is_empty() {
            return Err("Decoded audio is empty".to_string());
        }

        debug!("Decoded {} samples from audio file", samples.len());

        let stt = self.stt.clone();
        let text = tokio::task::spawn_blocking(move || {
            let mut guard = stt.lock().map_err(|e| format!("STT lock poisoned: {e}"))?;
            let result = guard
                .transcribe(&samples)
                .map_err(|e| format!("Transcription failed: {e}"))?;
            Ok::<String, String>(result.text.trim().to_string())
        })
        .await
        .map_err(|e| format!("spawn_blocking join error: {e}"))??;

        if text.is_empty() {
            return Err("Transcription produced empty text".to_string());
        }

        let lang = detect_language(&text);
        debug!(
            "Transcribed: \"{}\" (lang: {})",
            &text[..text.len().min(80)],
            lang
        );
        Ok((text, lang))
    }

    /// Synthesize text to an `.ogg` opus file.
    ///
    /// `lang` is an ISO 639-1 code (e.g. "en", "es", "fr") used to route
    /// to the appropriate TTS engine based on configuration.
    /// Returns the path to the generated file in `~/.nanobot/media/`.
    pub async fn synthesize_to_file(&self, text: &str, lang: &str) -> Result<String, String> {
        let text = text.to_string();
        let lang = lang.to_string();
        let lang_for_log = lang.clone();
        let qwen_voice = self.qwen_voice.clone();

        // If Qwen engine is configured and available, use it
        let tts = if let Some(ref tts) = self.tts_qwen {
            tts.clone()
        } else {
            // Route to appropriate engine based on language.
            let is_english = matches!(lang.as_str(), "en" | "en-us" | "en-gb");
            if is_english {
                self.tts_en.as_ref().or(self.tts_multi.as_ref())
            } else {
                self.tts_multi.as_ref().or(self.tts_en.as_ref())
            }
            .ok_or("No TTS engine available")?
            .clone()
        };

        let is_qwen = self.tts_qwen.is_some();

        let (all_samples, sample_rate) = tokio::task::spawn_blocking(move || {
            let mut guard = tts.lock().map_err(|e| format!("TTS lock poisoned: {e}"))?;

            let engine_type = guard.engine_type().to_string();

            if is_qwen || engine_type == "qwen" || engine_type == "qwen-large" {
                // Qwen engine - use configured voice
                guard
                    .set_speaker(&qwen_voice)
                    .map_err(|e| format!("Qwen voice switch failed: {e}"))?;
            } else if engine_type == "pocket" {
                // Pocket is English-only; already initialized with default voice
            } else {
                // Kokoro - switch to language-appropriate voice
                let (voice_id, _kokoro_lang) = language_to_kokoro_voice(&lang);
                guard
                    .set_speaker(voice_id)
                    .map_err(|e| format!("Voice switch failed: {e}"))?;
            }

            let sentences = split_tts_sentences(&text);
            if sentences.is_empty() {
                return Err("No text to synthesize".to_string());
            }

            let mut all_samples: Vec<f32> = Vec::new();
            let mut sample_rate = 0u32;

            for sentence in &sentences {
                sample_rate = guard
                    .synthesize_streaming(sentence, |samples, rate| {
                        sample_rate = rate;
                        all_samples.extend_from_slice(samples);
                        true
                    })
                    .map_err(|e| format!("TTS failed: {e}"))?;
            }

            Ok::<(Vec<f32>, u32), String>((all_samples, sample_rate))
        })
        .await
        .map_err(|e| format!("spawn_blocking join error: {e}"))??;

        let output_path = encode_samples_to_ogg(&all_samples, sample_rate)?;
        debug!(
            "Synthesized voice to {} (lang: {})",
            output_path, lang_for_log
        );
        Ok(output_path)
    }
}

/// Map ISO 639-1 language code to a default Kokoro voice ID and language code.
/// Returns `(voice_id_str, kokoro_lang_code)`.
fn language_to_kokoro_voice(lang: &str) -> (&'static str, &'static str) {
    match lang {
        "es" => ("28", "es"), // ef_dora (Spanish female)
        "fr" => ("30", "fr"), // ff_siwis (French female)
        "hi" => ("31", "hi"), // hf_alpha (Hindi female)
        "it" => ("35", "it"), // if_sara (Italian female)
        "ja" => ("37", "ja"), // jf_alpha (Japanese female)
        "pt" => ("42", "pt"), // pf_dora (Portuguese female)
        "zh" => ("45", "zh"), // zf_xiaobei (Mandarin female)
        _ => ("3", "en-us"),  // af_heart (American English, default)
    }
}

/// Decode an audio file to f32le 16kHz mono samples using ffmpeg.
fn decode_audio_file(path: &str) -> Result<Vec<f32>, String> {
    let output = Command::new("ffmpeg")
        .args([
            "-i", path, "-f", "f32le", "-ar", "16000", "-ac", "1", "pipe:1",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("ffmpeg decode failed: {e}\n  Install: sudo apt install ffmpeg"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "ffmpeg decode failed (exit {}): {}",
            output.status,
            &stderr[..stderr.len().min(200)]
        ));
    }

    let samples: Vec<f32> = output
        .stdout
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    Ok(samples)
}

/// Encode f32 samples to an ogg/opus file using ffmpeg.
///
/// Returns the path to the output file.
fn encode_samples_to_ogg(samples: &[f32], sample_rate: u32) -> Result<String, String> {
    let home = dirs::home_dir().ok_or("Cannot determine home directory")?;
    let media_dir = home.join(".nanobot").join("media");
    std::fs::create_dir_all(&media_dir).map_err(|e| format!("Failed to create media dir: {e}"))?;

    let filename = format!("tts_{}.ogg", uuid::Uuid::new_v4());
    let output_path = media_dir.join(&filename);
    let output_path_str = output_path.to_string_lossy().to_string();

    // Convert f32 samples to raw bytes
    let raw_bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();

    let mut child = Command::new("ffmpeg")
        .args([
            "-f",
            "f32le",
            "-ar",
            &sample_rate.to_string(),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-c:a",
            "libopus",
            "-b:a",
            "128k",
            "-y",
            &output_path_str,
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("ffmpeg encode failed: {e}"))?;

    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(&raw_bytes)
            .map_err(|e| format!("Failed to write to ffmpeg stdin: {e}"))?;
    }

    let output = child
        .wait_with_output()
        .map_err(|e| format!("ffmpeg encode wait failed: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "ffmpeg encode failed (exit {}): {}",
            output.status,
            &stderr[..stderr.len().min(200)]
        ));
    }

    Ok(output_path_str)
}

/// Logging-based model download progress (no terminal cursor control).
struct LogProgress;

impl ModelProgressCallback for LogProgress {
    fn on_download_start(&self, model: &str, size_mb: u64) {
        info!("Downloading voice model {} ({} MB)...", model, size_mb);
    }

    fn on_download_progress(&self, _model: &str, _progress_percent: u32, _downloaded_mb: u64) {
        // Suppress per-percent logging in gateway mode
    }

    fn on_download_complete(&self, model: &str) {
        info!("Voice model {} downloaded", model);
    }

    fn on_extracting(&self, model: &str) {
        info!("Extracting voice model {}...", model);
    }
}
