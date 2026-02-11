#![cfg(feature = "voice")]

//! Voice pipeline for channel voice message transcription and TTS synthesis.
//!
//! Provides `VoicePipeline` which wraps jack_voice STT/TTS models and uses
//! ffmpeg subprocesses for audio codec conversion. Designed to be shared
//! across channels via `Arc<VoicePipeline>`.

use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};

use jack_voice::{
    models::{self, ModelProgressCallback},
    SpeechToText, SttMode, TextToSpeech, TtsEngine,
};
use tracing::{debug, info};
use whatlang;

use crate::voice::split_tts_sentences;

/// Shared voice pipeline for channel voice message processing.
///
/// Both STT and TTS are CPU-bound and use `std::sync::Mutex` so they can be
/// held across `spawn_blocking` boundaries. Wrapped in `Arc` so they can
/// be moved into the blocking closure.
///
/// Holds two TTS engines: Supertonic (fast English) and Kokoro (multilingual).
/// Each synthesis call is routed to the appropriate engine based on the
/// detected language of the text.
pub struct VoicePipeline {
    stt: Arc<Mutex<SpeechToText>>,
    /// Supertonic TTS engine (fast, English-only). `None` if init failed.
    tts_en: Option<Arc<Mutex<TextToSpeech>>>,
    /// Kokoro TTS engine (multilingual). `None` if init failed.
    tts_multi: Option<Arc<Mutex<TextToSpeech>>>,
}

impl VoicePipeline {
    /// Create a new voice pipeline, downloading models if needed.
    ///
    /// Initializes both Supertonic (English) and Kokoro (multilingual) TTS
    /// engines. If one fails, the other is used as fallback.
    pub async fn new() -> Result<Self, String> {
        info!("Initializing voice pipeline for channels...");

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
        models::ensure_kokoro_model(progress)
            .await
            .map_err(|e| format!("Model download failed: {e}"))?;

        let stt = SpeechToText::new(SttMode::Batch).map_err(|e| format!("STT init failed: {e}"))?;

        // Init Supertonic (English TTS)
        let tts_en = match tokio::task::spawn_blocking(|| {
            TextToSpeech::with_engine(TtsEngine::Supertonic)
        })
        .await
        .map_err(|e| format!("spawn_blocking join error: {e}"))?
        {
            Ok(tts) => {
                info!("Supertonic TTS ready (English)");
                Some(tts)
            }
            Err(e) => {
                info!("Supertonic TTS not available, English will use Kokoro: {e}");
                None
            }
        };

        // Init Kokoro (multilingual TTS)
        let tts_multi = match tokio::task::spawn_blocking(|| {
            TextToSpeech::with_engine(TtsEngine::Kokoro)
        })
        .await
        .map_err(|e| format!("spawn_blocking join error: {e}"))?
        {
            Ok(tts) => {
                info!("Kokoro TTS ready (multilingual)");
                Some(tts)
            }
            Err(e) => {
                info!("Kokoro TTS not available: {e}");
                None
            }
        };

        if tts_en.is_none() && tts_multi.is_none() {
            return Err("No TTS engine could be initialized".to_string());
        }

        info!("Voice pipeline ready");

        Ok(Self {
            stt: Arc::new(Mutex::new(stt)),
            tts_en: tts_en.map(|t| Arc::new(Mutex::new(t))),
            tts_multi: tts_multi.map(|t| Arc::new(Mutex::new(t))),
        })
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
    /// to the appropriate TTS engine: Supertonic for English, Kokoro for others.
    /// Returns the path to the generated file in `~/.nanobot/media/`.
    pub async fn synthesize_to_file(&self, text: &str, lang: &str) -> Result<String, String> {
        let text = text.to_string();
        let lang = lang.to_string();
        let lang_for_log = lang.clone();

        // Route to appropriate engine based on language.
        let is_english = matches!(lang.as_str(), "en" | "en-us" | "en-gb");
        let tts = if is_english {
            self.tts_en.as_ref().or(self.tts_multi.as_ref())
        } else {
            self.tts_multi.as_ref().or(self.tts_en.as_ref())
        }
        .ok_or("No TTS engine available")?
        .clone();

        let (all_samples, sample_rate) = tokio::task::spawn_blocking(move || {
            let mut guard = tts.lock().map_err(|e| format!("TTS lock poisoned: {e}"))?;

            let is_supertonic = guard.engine_type() == "supertonic";
            if is_supertonic {
                // Supertonic is English-only; already initialized with default voice
            } else {
                // Switch to the language-appropriate Kokoro voice.
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
                let output = guard
                    .synthesize(sentence)
                    .map_err(|e| format!("TTS failed: {e}"))?;
                sample_rate = output.sample_rate;
                all_samples.extend_from_slice(&output.samples);
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

/// Detect language from text, returns ISO 639-1 code (e.g. "en", "es", "fr").
/// whatlang returns ISO 639-3 (3-letter) codes; we map to 2-letter for Kokoro.
fn detect_language(text: &str) -> String {
    whatlang::detect(text)
        .map(|info| iso639_3_to_1(info.lang().code()))
        .unwrap_or_else(|| "en".to_string())
}

/// Map ISO 639-3 (whatlang) codes to ISO 639-1 (Kokoro) codes.
fn iso639_3_to_1(code: &str) -> String {
    match code {
        "eng" => "en",
        "spa" => "es",
        "fra" => "fr",
        "hin" => "hi",
        "ita" => "it",
        "jpn" => "ja",
        "por" => "pt",
        "cmn" | "zho" => "zh",
        _ => code, // pass through unknown codes
    }
    .to_string()
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
