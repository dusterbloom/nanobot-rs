#![cfg(feature = "voice")]

//! Unified voice pipeline for nanobot.
//!
//! Single `VoicePipeline` service used by all voice consumers:
//! - REPL `/voice` toggle (mic + speaker)
//! - `nanobot realtime` CLI (mic + speaker + LLM loop)
//! - Channel adapters (file I/O, no audio hardware)
//!
//! Replaces the former `voice.rs` (VoiceSession) and channel-only `VoicePipeline`.
//! Uses cross-platform `AudioCapture`/`AudioPlayer` from jack-voice (cpal-based),
//! no `parec` dependency.

use parking_lot::Mutex;
use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc as std_mpsc;
use std::sync::Arc;
use std::time::Duration;

use crate::config::schema::TtsEngineConfig;
use jack_voice::{
    models::{self, ModelProgressCallback},
    AudioCapture, AudioError, AudioPlayer, SpeechToText, SttMode, TextToSpeech, TtsEngine,
};
use lingua::{Language, LanguageDetector, LanguageDetectorBuilder};
use once_cell::sync::Lazy;
use tracing::{debug, info};

// ============================================================================
// Language detection (shared)
// ============================================================================

/// Shared language detector for TTS routing. Restricted to the 8 languages
/// we have TTS voices for — keeps detection fast and accurate on short text.
/// lingua achieves 100% accuracy on English single words (vs whatlang's 17.9%).
static LANG_DETECTOR: Lazy<LanguageDetector> = Lazy::new(|| {
    LanguageDetectorBuilder::from_languages(&[
        Language::English,
        Language::Spanish,
        Language::French,
        Language::Hindi,
        Language::Italian,
        Language::Japanese,
        Language::Portuguese,
        Language::Chinese,
    ])
    .build()
});

/// Detect language from text, returns ISO 639-1 code (e.g. "en", "es").
pub(crate) fn detect_language(text: &str) -> String {
    LANG_DETECTOR
        .detect_language_of(text)
        .map(|lang| match lang {
            Language::English => "en",
            Language::Spanish => "es",
            Language::French => "fr",
            Language::Hindi => "hi",
            Language::Italian => "it",
            Language::Japanese => "ja",
            Language::Portuguese => "pt",
            Language::Chinese => "zh",
            _ => "en",
        })
        .unwrap_or("en")
        .to_string()
}

// ============================================================================
// Text processing (shared)
// ============================================================================

/// Max chunk size in characters for TTS batching.
const TTS_CHUNK_MAX_CHARS: usize = 250;

fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut start = 0;
    let bytes = text.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        if b == b'.' || b == b'!' || b == b'?' {
            let end = i + 1;
            let s = text[start..end].trim().to_string();
            if !s.is_empty() {
                sentences.push(s);
            }
            start = end;
        }
    }
    let remainder = text[start..].trim().to_string();
    if !remainder.is_empty() {
        sentences.push(remainder);
    }
    sentences
}

/// Split text into TTS chunks up to 250 chars, always ending on sentence punctuation.
/// Short responses (<=500 chars) are synthesized as a single chunk.
pub(crate) fn split_tts_sentences(text: &str) -> Vec<String> {
    let normalized: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let trimmed = normalized.trim();
    if trimmed.len() <= 500 {
        return if trimmed.is_empty() {
            vec![]
        } else {
            vec![trimmed.to_string()]
        };
    }

    let sentences = split_sentences(&normalized);
    let mut chunks = Vec::new();
    let mut current = String::new();

    for sentence in sentences {
        if current.is_empty() {
            current = sentence;
        } else if current.len() + 1 + sentence.len() <= TTS_CHUNK_MAX_CHARS {
            current.push(' ');
            current.push_str(&sentence);
        } else {
            chunks.push(current);
            current = sentence;
        }
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

/// Normalize samples to a target peak level so all sentences have consistent volume.
pub(crate) fn normalize_peak(samples: &mut [f32], target_peak: f32) {
    let peak = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if peak > 1e-6 {
        let gain = target_peak / peak;
        for s in samples.iter_mut() {
            *s *= gain;
        }
    }
}

/// Apply fade-in and fade-out envelopes to eliminate clicks at sentence boundaries.
pub(crate) fn apply_fade_envelope(samples: &mut [f32], fade_samples: usize) {
    let len = samples.len();
    let fade = fade_samples.min(len / 2);
    for i in 0..fade {
        samples[i] *= i as f32 / fade as f32;
    }
    for i in 0..fade {
        samples[len - 1 - i] *= i as f32 / fade as f32;
    }
}

/// Convert f32 samples to raw little-endian bytes.
fn samples_to_f32le_bytes(samples: &[f32]) -> Vec<u8> {
    samples.iter().flat_map(|s| s.to_le_bytes()).collect()
}

fn f32le_bytes_to_samples(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Map ISO 639-1 language code to a default Kokoro voice ID and language code.
fn language_to_kokoro_voice(lang: &str) -> (&'static str, &'static str) {
    match lang {
        "es" => ("28", "es"),
        "fr" => ("30", "fr"),
        "hi" => ("31", "hi"),
        "it" => ("35", "it"),
        "ja" => ("37", "ja"),
        "pt" => ("42", "pt"),
        "zh" => ("45", "zh"),
        _ => ("3", "en-us"),
    }
}

// ============================================================================
// Playback helpers (cross-platform)
// ============================================================================

/// A chunk of synthesized audio ready for playback.
struct AudioChunk {
    data: Vec<u8>, // f32le raw bytes
    sample_rate: u32,
}

#[cfg(target_os = "macos")]
fn play_chunks_native(
    audio_rx: std_mpsc::Receiver<AudioChunk>,
    cancel: Arc<AtomicBool>,
) -> Result<(), String> {
    let mut player = AudioPlayer::new()
        .map_err(|e| format!("Audio playback failed: {e}. Check macOS output device settings."))?;

    for chunk in audio_rx {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        let samples = f32le_bytes_to_samples(&chunk.data);
        if samples.is_empty() {
            continue;
        }
        player.play(samples, chunk.sample_rate);
    }

    if cancel.load(Ordering::Relaxed) {
        player.stop();
    } else {
        player.wait();
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn pulse_server() -> String {
    if std::path::Path::new("/mnt/wslg/PulseServer").exists() {
        "unix:/mnt/wslg/PulseServer".to_string()
    } else {
        std::env::var("PULSE_SERVER").unwrap_or_default()
    }
}

#[cfg(not(target_os = "macos"))]
fn play_chunks_paplay(
    audio_rx: std_mpsc::Receiver<AudioChunk>,
    cancel: Arc<AtomicBool>,
) -> Result<(), String> {
    let first_chunk = match audio_rx.recv() {
        Ok(c) => c,
        Err(_) => return Ok(()),
    };

    let mut child = Command::new("paplay")
        .args([
            "--raw",
            "--format=float32le",
            "--channels=1",
            &format!("--rate={}", first_chunk.sample_rate),
        ])
        .env("PULSE_SERVER", pulse_server())
        .env("PULSE_LATENCY_MSEC", "10")
        .stdin(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|e| format!("paplay failed: {e}\n  Install: sudo apt install pulseaudio-utils"))?;

    let mut stdin = child.stdin.take().unwrap();

    if stdin.write_all(&first_chunk.data).is_err() {
        let _ = child.kill();
        let _ = child.wait();
        return Ok(());
    }

    for chunk in audio_rx {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        if stdin.write_all(&chunk.data).is_err() {
            break;
        }
    }

    drop(stdin);
    if cancel.load(Ordering::Relaxed) {
        let _ = child.kill();
    }
    let _ = child.wait();
    Ok(())
}

/// Block SIGINT delivery in the current thread to prevent segfaults in C/C++ FFI.
#[cfg(unix)]
fn mask_sigint() {
    unsafe {
        let mut sigset: libc::sigset_t = std::mem::zeroed();
        libc::sigemptyset(&mut sigset);
        libc::sigaddset(&mut sigset, libc::SIGINT);
        libc::pthread_sigmask(libc::SIG_BLOCK, &sigset, std::ptr::null_mut());
    }
}

fn format_native_capture_error(error: AudioError) -> String {
    #[cfg(target_os = "macos")]
    {
        match error {
            AudioError::NoInputDevice => {
                "No microphone input device found. Connect/select an input in macOS Sound settings and retry /voice.".to_string()
            }
            AudioError::StreamError(e) | AudioError::ConfigError(e) => format!(
                "Microphone capture failed: {e}. Enable microphone access for your terminal in System Settings > Privacy & Security > Microphone, then restart the terminal and retry /voice."
            ),
            other => format!(
                "Microphone capture failed: {other}. Check microphone access in System Settings > Privacy & Security > Microphone and retry /voice."
            ),
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        format!(
            "Microphone capture failed: {error}. Verify your default audio input device and retry /voice."
        )
    }
}

fn start_native_capture(sample_tx: std_mpsc::Sender<Vec<f32>>) -> Result<AudioCapture, String> {
    AudioCapture::start(sample_tx).map_err(format_native_capture_error)
}

// ============================================================================
// TTS command & sentence accumulator (shared)
// ============================================================================

/// A command sent to the synthesis thread.
pub(crate) enum TtsCommand {
    /// Synthesize text with detected language (ISO code e.g. "en", "it", "es").
    Synthesize {
        text: String,
        language: String,
    },
    Finish,
}

/// Accumulates streaming text deltas and batches complete sentences into ~200-char
/// chunks before sending to TTS.
pub(crate) struct SentenceAccumulator {
    buffer: String,
    pending: String,
    in_code_block: bool,
    in_thinking_block: bool,
    sentence_tx: std_mpsc::Sender<TtsCommand>,
    eager: bool,
    first_buffered: Option<std::time::Instant>,
}

impl SentenceAccumulator {
    pub fn new(sentence_tx: std_mpsc::Sender<TtsCommand>) -> Self {
        Self {
            buffer: String::new(),
            pending: String::new(),
            in_code_block: false,
            in_thinking_block: false,
            sentence_tx,
            eager: false,
            first_buffered: None,
        }
    }

    /// Create an accumulator that sends each sentence immediately for low-latency
    /// streaming TTS.
    pub fn new_streaming(sentence_tx: std_mpsc::Sender<TtsCommand>) -> Self {
        Self {
            buffer: String::new(),
            pending: String::new(),
            in_code_block: false,
            in_thinking_block: false,
            sentence_tx,
            eager: true,
            first_buffered: None,
        }
    }

    pub fn push(&mut self, delta: &str) {
        self.buffer.push_str(delta);
        self.strip_thinking_from_buffer();

        if self.first_buffered.is_none() && !self.buffer.trim().is_empty() {
            self.first_buffered = Some(std::time::Instant::now());
        }
        self.extract_sentences();
        if self.eager && !self.in_code_block {
            self.try_timeout_flush();
        }
    }

    fn strip_thinking_from_buffer(&mut self) {
        loop {
            if self.in_thinking_block {
                if let Some(end) = self.buffer.find("</thinking>") {
                    self.buffer = self.buffer[end + "</thinking>".len()..].to_string();
                    self.in_thinking_block = false;
                } else {
                    self.buffer.clear();
                    return;
                }
            } else if let Some(start) = self.buffer.find("<thinking>") {
                let before = self.buffer[..start].to_string();
                let after_tag = self.buffer[start + "<thinking>".len()..].to_string();
                self.in_thinking_block = true;
                if let Some(end) = after_tag.find("</thinking>") {
                    let remaining = after_tag[end + "</thinking>".len()..].to_string();
                    self.buffer = format!("{}{}", before, remaining);
                    self.in_thinking_block = false;
                } else {
                    self.buffer = before;
                    return;
                }
            } else {
                return;
            }
        }
    }

    fn try_timeout_flush(&mut self) {
        if let Some(t) = self.first_buffered {
            if t.elapsed() >= std::time::Duration::from_millis(500) && self.buffer.trim().len() > 20
            {
                let text = std::mem::take(&mut self.buffer);
                let cleaned = strip_inline_markdown(text.trim());
                if !cleaned.is_empty() {
                    self.send_to_tts(cleaned);
                }
                self.first_buffered = None;
            }
        }
    }

    pub fn flush(self) {
        let mut pending = self.pending;
        let remainder = self.buffer.trim().to_string();
        if !remainder.is_empty() && !self.in_code_block {
            let cleaned = strip_inline_markdown(&remainder);
            if !cleaned.is_empty() {
                if !pending.is_empty() {
                    pending.push(' ');
                }
                pending.push_str(&cleaned);
            }
        }
        if !pending.is_empty() {
            let language = detect_language(&pending);
            let _ = self.sentence_tx.send(TtsCommand::Synthesize {
                text: pending,
                language,
            });
        }
        let _ = self.sentence_tx.send(TtsCommand::Finish);
    }

    /// Send text to TTS with auto-detected language.
    fn send_to_tts(&self, text: String) {
        let language = detect_language(&text);
        let _ = self
            .sentence_tx
            .send(TtsCommand::Synthesize { text, language });
    }

    /// Same as send_to_tts but returns the Result (for use with `let _ =`).
    fn send_to_tts_raw(&self, text: String) -> Result<(), std_mpsc::SendError<TtsCommand>> {
        let language = detect_language(&text);
        self.sentence_tx
            .send(TtsCommand::Synthesize { text, language })
    }

    fn enqueue_sentence(&mut self, sentence: &str) {
        // Always batch sentences — even in eager mode — to give TTS enough context
        // for consistent voice. Short isolated sentences cause voice drift in Qwen3-TTS.
        if self.pending.is_empty() {
            self.pending = sentence.to_string();
        } else if self.pending.len() + 1 + sentence.len() <= TTS_CHUNK_MAX_CHARS {
            self.pending.push(' ');
            self.pending.push_str(sentence);
        } else {
            let batch = std::mem::replace(&mut self.pending, sentence.to_string());
            let _ = self.send_to_tts_raw(batch);
        }
        // In eager mode, flush pending if we have enough text for a coherent TTS call
        if self.eager && self.pending.len() >= 80 {
            let batch = std::mem::take(&mut self.pending);
            let _ = self.send_to_tts_raw(batch);
            self.first_buffered = None;
        }
    }

    fn extract_sentences(&mut self) {
        loop {
            if let Some(pos) = self.buffer.find("```") {
                if !self.in_code_block {
                    let before = self.buffer[..pos].trim().to_string();
                    if !before.is_empty() {
                        let cleaned = strip_inline_markdown(&before);
                        if !cleaned.is_empty() {
                            self.enqueue_sentence(&cleaned);
                        }
                    }
                    if !self.pending.is_empty() {
                        let batch = std::mem::take(&mut self.pending);
                        let _ = self.send_to_tts_raw(batch);
                    }
                }
                self.in_code_block = !self.in_code_block;
                let after_marker = pos + 3;
                if let Some(nl) = self.buffer[after_marker..].find('\n') {
                    self.buffer = self.buffer[after_marker + nl + 1..].to_string();
                } else {
                    self.buffer = self.buffer[after_marker..].to_string();
                    return;
                }
                continue;
            }

            if self.in_code_block {
                return;
            }

            if let Some(pos) = find_sentence_boundary(&self.buffer) {
                let sentence = self.buffer[..=pos].trim().to_string();
                self.buffer = self.buffer[pos + 1..].to_string();
                if !sentence.is_empty() {
                    let cleaned = strip_inline_markdown(&sentence);
                    if !cleaned.is_empty() {
                        self.enqueue_sentence(&cleaned);
                    }
                }
            } else {
                return;
            }
        }
    }
}

fn find_sentence_boundary(text: &str) -> Option<usize> {
    let bytes = text.as_bytes();
    for i in 0..bytes.len().saturating_sub(1) {
        if matches!(bytes[i], b'.' | b'!' | b'?') {
            if bytes[i + 1].is_ascii_whitespace() {
                return Some(i);
            }
        }
    }
    None
}

/// Strip inline markdown syntax for cleaner TTS.
fn strip_inline_markdown(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '*' | '_' | '`' | '~' => {}
            '#' if out.is_empty() || out.ends_with('\n') => {
                while chars.peek() == Some(&'#') {
                    chars.next();
                }
                if chars.peek() == Some(&' ') {
                    chars.next();
                }
            }
            '[' => {
                let mut link_text = String::new();
                for lc in chars.by_ref() {
                    if lc == ']' {
                        break;
                    }
                    link_text.push(lc);
                }
                out.push_str(&link_text);
                if chars.peek() == Some(&'(') {
                    let mut depth = 1;
                    chars.next();
                    for lc in chars.by_ref() {
                        if lc == '(' {
                            depth += 1;
                        }
                        if lc == ')' {
                            depth -= 1;
                            if depth == 0 {
                                break;
                            }
                        }
                    }
                }
            }
            _ => {
                // Skip emoji and other non-speech Unicode symbols.
                // Covers Dingbats, Emoticons, Symbols, Flags, etc.
                let cp = c as u32;
                if cp >= 0x2600 && cp <= 0x27BF       // Misc symbols, Dingbats
                    || cp >= 0xFE00 && cp <= 0xFE0F    // Variation selectors
                    || cp >= 0x1F000 && cp <= 0x1FAFF  // All emoji blocks
                    || cp >= 0x200D && cp <= 0x200D     // Zero-width joiner
                    || cp >= 0xE0020 && cp <= 0xE007F
                // Tag sequences (flags)
                {
                    continue;
                }
                out.push(c);
            }
        }
    }
    out.trim().to_string()
}

// ============================================================================
// VoicePipeline — unified struct for all voice consumers
// ============================================================================

/// Unified voice pipeline for all nanobot voice consumers.
///
/// Holds STT and TTS engines behind `Arc<Mutex<>>` for thread-safe access.
/// Two construction modes:
/// - `with_engine()` / `with_lang()` — mic+speaker mode for TUI/realtime
/// - `for_channels()` — file I/O only, no audio hardware
pub struct VoicePipeline {
    stt: Arc<Mutex<SpeechToText>>,
    tts_en: Option<Arc<Mutex<TextToSpeech>>>,
    tts_multi: Option<Arc<Mutex<TextToSpeech>>>,
    engine_config: TtsEngineConfig,
    cancel: Arc<AtomicBool>,
}

impl VoicePipeline {
    // ----------------------------------------------------------------
    // Constructors
    // ----------------------------------------------------------------

    /// Create a pipeline for channel use (file I/O, no mic/speaker).
    pub async fn for_channels(engine: TtsEngineConfig) -> Result<Self, String> {
        Self::init_pipeline(engine, None, &LogProgress).await
    }

    /// Create a pipeline with optional language-based engine selection (mic mode).
    ///
    /// - `None` → load both Pocket (English) and Kokoro (multilingual)
    /// - `Some("en")` → load only Pocket
    /// - `Some(_)` → load only Kokoro
    pub async fn with_lang(lang: Option<&str>) -> Result<Self, String> {
        let load_pocket = lang.is_none() || lang == Some("en");
        let load_kokoro = lang.is_none() || lang != Some("en");

        if load_kokoro && std::env::var("PIPER_ESPEAKNG_DATA_DIRECTORY").is_err() {
            let home = dirs::home_dir().unwrap_or_default();
            let local_data = home.join(".local/share/espeak-ng-data");
            if local_data.exists() {
                std::env::set_var("PIPER_ESPEAKNG_DATA_DIRECTORY", home.join(".local/share"));
            }
        }

        let label = match lang {
            Some("en") => "Pocket only",
            Some(_) => "Kokoro only",
            None => "Pocket + Kokoro",
        };
        info!("Initializing voice pipeline ({label})...");

        let progress = &TerminalProgress;

        // Ensure base models (VAD, Whisper, SmartTurn)
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

        if load_kokoro {
            models::ensure_kokoro_model(progress)
                .await
                .map_err(|e| format!("Model download failed: {e}"))?;
        }

        let stt = SpeechToText::new(SttMode::Batch).map_err(|e| format!("STT init failed: {e}"))?;

        let tts_en = if load_pocket {
            match tokio::task::spawn_blocking(|| TextToSpeech::with_engine(TtsEngine::Pocket))
                .await
                .map_err(|e| format!("spawn_blocking join error: {e}"))?
            {
                Ok(tts) => {
                    info!(
                        "{} TTS ready (English) [engine: {}]",
                        if tts.engine_type() == "pocket" {
                            "Pocket"
                        } else {
                            tts.engine_type()
                        },
                        tts.engine_type()
                    );
                    Some(Arc::new(Mutex::new(tts)))
                }
                Err(e) => {
                    tracing::warn!("Pocket TTS init failed, English will use Kokoro: {e}");
                    None
                }
            }
        } else {
            None
        };

        let tts_multi = if load_kokoro {
            match tokio::task::spawn_blocking(|| TextToSpeech::with_engine(TtsEngine::Kokoro))
                .await
                .map_err(|e| format!("spawn_blocking join error: {e}"))?
            {
                Ok(tts) => {
                    info!("Kokoro TTS ready (multilingual)");
                    Some(Arc::new(Mutex::new(tts)))
                }
                Err(e) => {
                    tracing::warn!("Kokoro TTS init failed: {e}");
                    None
                }
            }
        } else {
            None
        };

        if tts_en.is_none() && tts_multi.is_none() {
            return Err("No TTS engine could be initialized".to_string());
        }

        Ok(Self {
            stt: Arc::new(Mutex::new(stt)),
            tts_en,
            tts_multi,
            engine_config: TtsEngineConfig::Pocket,
            cancel: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Create a pipeline with a specific TTS engine (mic mode).
    pub async fn with_engine(engine: TtsEngineConfig) -> Result<Self, String> {
        Self::init_pipeline(engine, None, &TerminalProgress).await
    }

    /// Internal: initialize the pipeline with the given engine and progress callback.
    async fn init_pipeline(
        engine: TtsEngineConfig,
        _lang: Option<&str>,
        progress: &(dyn ModelProgressCallback + Sync),
    ) -> Result<Self, String> {
        info!("Initializing voice pipeline ({:?})...", engine);

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

        let (tts_en, tts_multi) = match engine {
            TtsEngineConfig::Pocket => {
                models::ensure_kokoro_model(progress)
                    .await
                    .map_err(|e| format!("Model download failed: {e}"))?;

                let tts_en = match tokio::task::spawn_blocking(|| {
                    TextToSpeech::with_engine(TtsEngine::Pocket)
                })
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

                let tts_multi = match tokio::task::spawn_blocking(|| {
                    TextToSpeech::with_engine(TtsEngine::Kokoro)
                })
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

                (tts_en, tts_multi)
            }
            TtsEngineConfig::Kokoro => {
                models::ensure_kokoro_model(progress)
                    .await
                    .map_err(|e| format!("Model download failed: {e}"))?;
                let tts =
                    tokio::task::spawn_blocking(|| TextToSpeech::with_engine(TtsEngine::Kokoro))
                        .await
                        .map_err(|e| format!("spawn_blocking join error: {e}"))?
                        .map_err(|e| format!("Kokoro TTS init failed: {e}"))?;
                info!("Kokoro TTS ready (multilingual)");
                (None, Some(Arc::new(Mutex::new(tts))))
            }
        };

        if tts_en.is_none() && tts_multi.is_none() {
            return Err("No TTS engine could be initialized".to_string());
        }

        info!("Voice pipeline ready");

        Ok(Self {
            stt: Arc::new(Mutex::new(stt)),
            tts_en,
            tts_multi,
            engine_config: engine,
            cancel: Arc::new(AtomicBool::new(false)),
        })
    }

    // ----------------------------------------------------------------
    // Configuration
    // ----------------------------------------------------------------

    pub fn engine_config(&self) -> TtsEngineConfig {
        self.engine_config
    }

    // ----------------------------------------------------------------
    // Mic mode: record & transcribe
    // ----------------------------------------------------------------

    /// Record audio from mic and transcribe. Returns `(text, detected_language_code)`.
    pub fn record_and_transcribe(&mut self) -> Result<Option<(String, String)>, String> {
        use crossterm::event::{self, Event, KeyCode, KeyModifiers};

        print!("\x1b[2mrecording...\x1b[0m");
        std::io::stdout().flush().ok();

        let (sample_tx, sample_rx) = std_mpsc::channel::<Vec<f32>>();
        let capture = start_native_capture(sample_tx)?;

        let stop_flag = Arc::new(AtomicBool::new(false));
        let stop_flag_clone = stop_flag.clone();
        let collector = std::thread::spawn(move || {
            let mut all_samples = Vec::new();
            let mut buf = Vec::new();
            while !stop_flag_clone.load(Ordering::Relaxed) {
                match sample_rx.recv_timeout(Duration::from_millis(50)) {
                    Ok(samples) => {
                        buf.extend(samples);
                    }
                    Err(std_mpsc::RecvTimeoutError::Timeout) => {}
                    Err(_) => break,
                }
            }
            if !buf.is_empty() {
                all_samples.extend(buf.drain(..));
            }
            all_samples
        });

        loop {
            if let Ok(Event::Key(key)) = event::read() {
                let is_stop = key.code == KeyCode::Enter
                    || (key.code == KeyCode::Char(' ')
                        && key.modifiers.contains(KeyModifiers::CONTROL))
                    || key.code == KeyCode::Esc;
                if is_stop {
                    break;
                }
            }
        }

        stop_flag.store(true, Ordering::Relaxed);
        drop(capture); // stop AudioCapture

        let all_samples = collector.join().map_err(|_| "Audio collector panicked")?;

        if all_samples.is_empty() {
            return Ok(None);
        }

        print!("\x1b[12D\x1b[K");
        std::io::stdout().flush().ok();

        let stt = self.stt.clone();
        let result = {
            let mut guard = stt.lock();
            guard
                .transcribe(&all_samples)
                .map_err(|e| format!("Transcription failed: {e}"))?
        };

        let text = result.text.trim().to_string();
        if text.is_empty() {
            print!("\x1b[2m(no speech)\x1b[0m");
            println!();
            return Ok(None);
        }

        let lang = detect_language(&text);
        Ok(Some((text, lang)))
    }

    // ----------------------------------------------------------------
    // Mic mode: speak (blocking)
    // ----------------------------------------------------------------

    /// Select the appropriate TTS engine based on language and config.
    fn select_tts(&self, lang: &str) -> Result<(Arc<Mutex<TextToSpeech>>, String), String> {
        let is_english = matches!(lang, "en" | "en-us" | "en-gb");
        let tts = if is_english {
            self.tts_en.as_ref().or(self.tts_multi.as_ref())
        } else {
            self.tts_multi.as_ref().or(self.tts_en.as_ref())
        }
        .ok_or("No TTS engine available")?
        .clone();

        let is_pocket = tts.lock().engine_type() == "pocket";
        let voice_id = if is_pocket {
            "alba".to_string()
        } else {
            let (vid, _) = language_to_kokoro_voice(lang);
            vid.to_string()
        };

        Ok((tts, voice_id))
    }

    pub fn speak(&mut self, text: &str, lang: &str) -> Result<(), String> {
        let sentences = split_tts_sentences(text);
        if sentences.is_empty() {
            return Ok(());
        }

        let (tts, voice_id) = self.select_tts(lang)?;
        let cancel = self.cancel.clone();

        let (audio_tx, audio_rx) = std_mpsc::sync_channel::<AudioChunk>(2);

        let cancel_synth = cancel.clone();
        let cancel_play = cancel.clone();

        let synth_handle = std::thread::spawn(move || {
            #[cfg(unix)]
            mask_sigint();
            let mut guard = tts.lock();
            if let Err(e) = guard.set_speaker(&voice_id) {
                tracing::warn!("Voice switch to {} failed: {}", voice_id, e);
            }

            for (i, sentence) in sentences.iter().enumerate() {
                if cancel_synth.load(Ordering::Relaxed) {
                    break;
                }
                tracing::debug!("Synthesizing sentence {}/{}...", i + 1, sentences.len());
                let cancel_ref = &cancel_synth;
                let tx_ref = &audio_tx;
                match guard.synthesize_streaming(sentence, |samples, sample_rate| {
                    if cancel_ref.load(Ordering::Relaxed) {
                        return false;
                    }
                    let chunk = AudioChunk {
                        data: samples_to_f32le_bytes(samples),
                        sample_rate,
                    };
                    tx_ref.send(chunk).is_ok()
                }) {
                    Ok(_) => {}
                    Err(e) => {
                        tracing::error!("TTS synthesis failed: {}", e);
                        break;
                    }
                }
            }
        });

        let playback_handle = std::thread::spawn(move || -> Result<(), String> {
            #[cfg(unix)]
            mask_sigint();
            #[cfg(target_os = "macos")]
            {
                play_chunks_native(audio_rx, cancel_play)
            }
            #[cfg(not(target_os = "macos"))]
            {
                play_chunks_paplay(audio_rx, cancel_play)
            }
        });

        let _ = synth_handle.join();
        match playback_handle.join() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(e),
            Err(_) => return Err("Playback thread panicked".to_string()),
        }

        Ok(())
    }

    /// Start a streaming speak session driven by external `TtsCommand`s.
    pub fn start_streaming_speak(
        &mut self,
        lang: &str,
        display_tx: Option<tokio::sync::mpsc::UnboundedSender<String>>,
    ) -> Result<(std_mpsc::Sender<TtsCommand>, std::thread::JoinHandle<()>), String> {
        let (tts, voice_id) = self.select_tts(lang)?;
        let cancel = self.cancel.clone();

        let (sentence_tx, sentence_rx) = std_mpsc::channel::<TtsCommand>();
        let (audio_tx, audio_rx) = std_mpsc::sync_channel::<AudioChunk>(2);

        let cancel_synth = cancel.clone();
        let synth_handle = std::thread::spawn(move || {
            #[cfg(unix)]
            mask_sigint();
            let mut guard = tts.lock();
            if let Err(e) = guard.set_speaker(&voice_id) {
                tracing::warn!("Voice switch to {} failed: {}", voice_id, e);
            }

            for cmd in sentence_rx {
                match cmd {
                    TtsCommand::Finish => break,
                    TtsCommand::Synthesize { text: sentence, .. } => {
                        if cancel_synth.load(Ordering::Relaxed) {
                            break;
                        }
                        if let Some(ref dtx) = display_tx {
                            let _ = dtx.send(sentence.clone());
                        }
                        let cancel_ref = &cancel_synth;
                        let tx_ref = &audio_tx;
                        match guard.synthesize_streaming(&sentence, |samples, sample_rate| {
                            if cancel_ref.load(Ordering::Relaxed) {
                                return false;
                            }
                            let chunk = AudioChunk {
                                data: samples_to_f32le_bytes(samples),
                                sample_rate,
                            };
                            tx_ref.send(chunk).is_ok()
                        }) {
                            Ok(_) => {}
                            Err(e) => {
                                tracing::error!("Streaming TTS synthesis failed: {}", e);
                                break;
                            }
                        }
                    }
                }
            }
        });

        let cancel_play = cancel.clone();
        let playback_handle = std::thread::spawn(move || -> Result<(), String> {
            #[cfg(unix)]
            mask_sigint();
            #[cfg(target_os = "macos")]
            {
                play_chunks_native(audio_rx, cancel_play)
            }
            #[cfg(not(target_os = "macos"))]
            {
                play_chunks_paplay(audio_rx, cancel_play)
            }
        });

        let join_handle = std::thread::spawn(move || {
            let _ = synth_handle.join();
            match playback_handle.join() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => tracing::error!("Streaming playback failed: {}", e),
                Err(_) => tracing::error!("Streaming playback thread panicked"),
            }
        });

        Ok((sentence_tx, join_handle))
    }

    // ----------------------------------------------------------------
    // Cancel / lifecycle
    // ----------------------------------------------------------------

    pub fn cancel_flag(&self) -> Arc<AtomicBool> {
        self.cancel.clone()
    }

    pub fn clear_cancel(&self) {
        self.cancel.store(false, Ordering::Relaxed);
    }

    pub fn stop_playback(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    /// Clean shutdown: stop playback and leak the session so native FFI destructors
    /// never run. The process is exiting — the OS reclaims memory.
    pub fn shutdown(mut self) {
        self.stop_playback();
        std::mem::forget(self);
    }

    // ----------------------------------------------------------------
    // Channel mode: file-based transcription & synthesis
    // ----------------------------------------------------------------

    /// Transcribe an audio file (e.g. `.ogg`) to text.
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
            let mut guard = stt.lock();
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

    /// Synthesize text to an `.ogg` opus file. Returns path to output file.
    pub async fn synthesize_to_file(&self, text: &str, lang: &str) -> Result<String, String> {
        let text = text.to_string();
        let lang = lang.to_string();
        let lang_for_log = lang.clone();
        let tts = {
            let is_english = matches!(lang.as_str(), "en" | "en-us" | "en-gb");
            if is_english {
                self.tts_en.as_ref().or(self.tts_multi.as_ref())
            } else {
                self.tts_multi.as_ref().or(self.tts_en.as_ref())
            }
            .ok_or("No TTS engine available")?
            .clone()
        };

        let (all_samples, sample_rate) = tokio::task::spawn_blocking(move || {
            let mut guard = tts.lock();
            let engine_type = guard.engine_type().to_string();

            if engine_type == "pocket" {
                // default voice
            } else {
                let (voice_id, _) = language_to_kokoro_voice(&lang);
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

// ============================================================================
// VoiceSession — backward-compatible alias
// ============================================================================

/// Backward-compatible alias for `VoicePipeline` in mic+speaker mode.
pub type VoiceSession = VoicePipeline;

// ============================================================================
// File codec helpers (channel mode)
// ============================================================================

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

fn encode_samples_to_ogg(samples: &[f32], sample_rate: u32) -> Result<String, String> {
    let home = dirs::home_dir().ok_or("Cannot determine home directory")?;
    let media_dir = home.join(".nanobot").join("media");
    std::fs::create_dir_all(&media_dir).map_err(|e| format!("Failed to create media dir: {e}"))?;

    let filename = format!("tts_{}.ogg", uuid::Uuid::new_v4());
    let output_path = media_dir.join(&filename);
    let output_path_str = output_path.to_string_lossy().to_string();

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

// ============================================================================
// Progress callbacks
// ============================================================================

struct LogProgress;

impl ModelProgressCallback for LogProgress {
    fn on_download_start(&self, model: &str, size_mb: u64) {
        info!("Downloading voice model {} ({} MB)...", model, size_mb);
    }
    fn on_download_progress(&self, _model: &str, _progress_percent: u32, _downloaded_mb: u64) {}
    fn on_download_complete(&self, model: &str) {
        info!("Voice model {} downloaded", model);
    }
    fn on_extracting(&self, model: &str) {
        info!("Extracting voice model {}...", model);
    }
}

struct TerminalProgress;

impl ModelProgressCallback for TerminalProgress {
    fn on_download_start(&self, model: &str, size_mb: u64) {
        println!("Downloading {} ({} MB)...", model, size_mb);
    }
    fn on_download_progress(&self, model: &str, progress_percent: u32, downloaded_mb: u64) {
        print!("\r  {} {}% ({} MB)", model, progress_percent, downloaded_mb);
        std::io::Write::flush(&mut std::io::stdout()).ok();
    }
    fn on_download_complete(&self, model: &str) {
        println!("\r  {} done", model);
    }
    fn on_extracting(&self, model: &str) {
        println!("  Extracting {}...", model);
    }
}

// ============================================================================
// Standalone TTS playback (used by realtime voice agent)
// ============================================================================

/// Start a standalone TTS synthesis+playback pipeline from a raw TTS engine handle.
///
/// Returns a `TtsCommand` sender and a cancel flag. Send `TtsCommand::Synthesize(text)`
/// for each sentence, then `TtsCommand::Finish` when done. The cancel flag can be set
/// to interrupt playback (barge-in).
///
/// This is used by `VoiceAgent` which owns the TTS via `RealtimeSession` and needs
/// to pipe LLM streaming deltas to audio output.
pub(crate) fn start_tts_playback(
    tts_en: Option<Arc<Mutex<TextToSpeech>>>,
    tts_multi: Option<Arc<Mutex<TextToSpeech>>>,
    tts_playing: Arc<AtomicBool>,
) -> (std_mpsc::Sender<TtsCommand>, Arc<AtomicBool>) {
    let cancel = Arc::new(AtomicBool::new(false));

    let (sentence_tx, sentence_rx) = std_mpsc::channel::<TtsCommand>();
    let (audio_tx, audio_rx) = std_mpsc::sync_channel::<AudioChunk>(2);

    let cancel_synth = cancel.clone();
    std::thread::spawn(move || {
        #[cfg(unix)]
        mask_sigint();

        // Pre-warm: lock each engine once, set default voices, then release.
        // This forces model loading to happen NOW, not on first synthesis.
        if let Some(ref tts) = tts_en {
            let mut g = tts.lock();
            let _ = g.set_speaker("alba");
            tracing::debug!("[tts-synth] Pocket pre-warmed (English)");
        }
        if let Some(ref tts) = tts_multi {
            let mut g = tts.lock();
            // Set a default Kokoro voice to force model load
            let _ = g.set_speaker("35"); // Italian (if_sara)
            tracing::debug!("[tts-synth] Kokoro pre-warmed (multilingual)");
        }

        for cmd in sentence_rx {
            match cmd {
                TtsCommand::Finish => {
                    tracing::debug!("[tts-synth] received Finish, waiting for next turn");
                    continue;
                }
                TtsCommand::Synthesize {
                    text: sentence,
                    language,
                } => {
                    if cancel_synth.load(Ordering::Relaxed) {
                        continue;
                    }

                    // Route to the right engine based on language
                    let is_english = matches!(language.as_str(), "en" | "en-us" | "en-gb" | "");

                    // Pick the right Arc, lock it per-sentence, synthesize, then release
                    let engine: Option<&Arc<Mutex<TextToSpeech>>> = if is_english {
                        tts_en.as_ref().or(tts_multi.as_ref())
                    } else {
                        tts_multi.as_ref().or(tts_en.as_ref())
                    };

                    let engine = match engine {
                        Some(e) => e,
                        None => {
                            tracing::warn!("[tts-synth] no TTS engine available");
                            continue;
                        }
                    };

                    let mut guard = engine.lock();

                    // Set voice per language
                    if is_english {
                        let _ = guard.set_speaker("alba");
                    } else {
                        let (voice_id, _) = language_to_kokoro_voice(&language);
                        if let Err(e) = guard.set_speaker(voice_id) {
                            tracing::warn!(
                                "[tts-synth] voice switch to {} failed: {}",
                                voice_id,
                                e
                            );
                        }
                    }

                    let cancel_ref = &cancel_synth;
                    let mut all_samples: Vec<f32> = Vec::new();
                    let mut sr = 0u32;
                    let mut cancelled = false;
                    if let Err(e) = guard.synthesize_streaming(&sentence, |samples, sample_rate| {
                        if cancel_ref.load(Ordering::Relaxed) {
                            cancelled = true;
                            return false;
                        }
                        sr = sample_rate;
                        all_samples.extend_from_slice(samples);
                        true
                    }) {
                        tracing::error!("[tts-synth] synthesis failed: {}", e);
                    }
                    if !cancelled && !all_samples.is_empty() {
                        let chunk = AudioChunk {
                            data: samples_to_f32le_bytes(&all_samples),
                            sample_rate: sr,
                        };
                        let _ = audio_tx.send(chunk);
                    }
                }
            }
        }
    });

    // Playback thread — long-lived, reads audio chunks until sender drops
    let cancel_play = cancel.clone();
    let tts_playing_play = tts_playing;
    std::thread::spawn(move || {
        #[cfg(unix)]
        mask_sigint();

        let mut player: Option<AudioPlayer> = None;

        tracing::debug!("[tts-play] playback thread started, waiting for audio chunks");
        for chunk in audio_rx {
            if cancel_play.load(Ordering::Relaxed) {
                tracing::debug!("[tts-play] cancelled, stopping immediately");
                if let Some(ref mut p) = player {
                    p.stop();
                }
                tts_playing_play.store(false, Ordering::SeqCst);
                // Reset cancel flag so next turn can play
                cancel_play.store(false, Ordering::Relaxed);
                continue;
            }
            let samples = f32le_bytes_to_samples(&chunk.data);
            if samples.is_empty() {
                tracing::debug!("[tts-play] empty chunk, skipping");
                continue;
            }
            tracing::debug!(
                "[tts-play] received {} samples @ {}Hz",
                samples.len(),
                chunk.sample_rate
            );
            tts_playing_play.store(true, Ordering::SeqCst);
            let p = player.get_or_insert_with(|| {
                tracing::debug!("[tts-play] creating AudioPlayer");
                AudioPlayer::new().expect("Failed to create audio player")
            });
            p.play(samples, chunk.sample_rate);
            // Block until this chunk finishes playing, then clear the flag.
            // This ensures echo suppression only applies during actual playback.
            p.wait();
            tts_playing_play.store(false, Ordering::SeqCst);
        }
    });

    (sentence_tx, cancel)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lingua_detects_short_english() {
        let text = "You just tell me a joke testing uh your pocket";
        let detected = detect_language(text);
        assert_eq!(detected, "en");
    }

    #[test]
    fn test_lingua_detects_longer_english() {
        let text = "Hello, how are you doing today? I wanted to ask you about the weather forecast for this weekend.";
        assert_eq!(detect_language(text), "en");
    }

    #[test]
    fn test_lingua_detects_spanish() {
        assert_eq!(
            detect_language("Hola, cómo estás hoy? Quiero preguntarte sobre el clima."),
            "es"
        );
    }

    #[test]
    fn test_lingua_detects_japanese() {
        assert_eq!(detect_language("今日の天気はどうですか？"), "ja");
    }

    #[test]
    fn test_split_tts_sentences_empty() {
        assert!(split_tts_sentences("").is_empty());
        assert!(split_tts_sentences("   ").is_empty());
    }

    #[test]
    fn test_split_tts_sentences_short() {
        assert_eq!(split_tts_sentences("Hello world."), vec!["Hello world."]);
    }

    #[test]
    fn test_split_tts_sentences_no_split_under_500() {
        let text = "First sentence. Second sentence. Third sentence.";
        assert_eq!(split_tts_sentences(text).len(), 1);
    }

    #[test]
    fn test_strip_inline_markdown() {
        assert_eq!(strip_inline_markdown("**bold** text"), "bold text");
        assert_eq!(strip_inline_markdown("# Heading"), "Heading");
        assert_eq!(strip_inline_markdown("[link](url)"), "link");
    }

    #[test]
    fn test_sentence_accumulator_strips_thinking_block() {
        let (tx, rx) = std::sync::mpsc::channel();
        let mut acc = SentenceAccumulator::new(tx);
        acc.push("<thinking>\nLet me think...\n</thinking>\n\nThe answer is 42.");
        acc.flush();
        let mut sentences = Vec::new();
        while let Ok(cmd) = rx.try_recv() {
            if let TtsCommand::Synthesize { text: s, .. } = cmd {
                sentences.push(s);
            }
        }
        let combined = sentences.join(" ");
        assert!(!combined.contains("thinking"));
        assert!(!combined.contains("Let me think"));
        assert!(combined.contains("The answer is 42"));
    }

    #[test]
    fn test_sentence_accumulator_strips_thinking_across_pushes() {
        let (tx, rx) = std::sync::mpsc::channel();
        let mut acc = SentenceAccumulator::new(tx);
        acc.push("<thinking>");
        acc.push("\nInternal reasoning here.\n");
        acc.push("</thinking>");
        acc.push("\nHello world.");
        acc.flush();
        let mut sentences = Vec::new();
        while let Ok(cmd) = rx.try_recv() {
            if let TtsCommand::Synthesize { text: s, .. } = cmd {
                sentences.push(s);
            }
        }
        let combined = sentences.join(" ");
        assert!(!combined.contains("Internal reasoning"));
        assert!(combined.contains("Hello world"));
    }

    #[test]
    fn test_pocket_tts_synthesizes() {
        use jack_voice::{TextToSpeech, TtsEngine};
        let tts = TextToSpeech::with_engine(TtsEngine::Pocket);
        match tts {
            Ok(mut tts) => {
                assert_eq!(tts.engine_type(), "pocket");
                let result = tts.synthesize("Hello, this is a test.");
                match result {
                    Ok(output) => {
                        assert!(output.samples.len() > 100);
                        assert!(output.sample_rate > 0);
                    }
                    Err(e) => panic!("Pocket TTS synthesis failed: {}", e),
                }
            }
            Err(e) => panic!("Pocket TTS init failed: {}", e),
        }
    }

    #[test]
    fn test_pocket_tts_streaming() {
        use jack_voice::{TextToSpeech, TtsEngine};
        let mut tts = TextToSpeech::with_engine(TtsEngine::Pocket).expect("Pocket TTS init failed");

        let mut chunk_count = 0u32;
        let mut total_samples = 0usize;
        let mut rate = 0u32;

        let sr = tts
            .synthesize_streaming(
                "Hello, this is a streaming test. The audio should arrive in multiple chunks.",
                |samples, sample_rate| {
                    chunk_count += 1;
                    total_samples += samples.len();
                    rate = sample_rate;
                    true
                },
            )
            .expect("Streaming synthesis failed");

        assert!(sr > 0);
        assert_eq!(sr, rate);
        assert!(total_samples > 100);
        assert!(chunk_count > 1);
    }
}
