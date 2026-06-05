#![cfg(feature = "voice")]
#![allow(unsafe_code)]

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

use std::collections::HashMap;
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
use parking_lot::Mutex;
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
        Language::Vietnamese,
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
            Language::Vietnamese => "vi",
            _ => "en",
        })
        .unwrap_or("en")
        .to_string()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PocketLanguageSpec {
    language: &'static str,
    default_voice: &'static str,
}

fn pocket_tts_language(lang: &str) -> Option<PocketLanguageSpec> {
    let lang = lang.trim();
    if lang.is_empty() {
        return None;
    }
    let base = lang
        .split(['-', '_'])
        .next()
        .unwrap_or(lang)
        .to_ascii_lowercase();

    match base.as_str() {
        "en" | "english" => Some(PocketLanguageSpec {
            language: "en",
            default_voice: "alba",
        }),
        "fr" | "french" => Some(PocketLanguageSpec {
            language: "fr",
            default_voice: "estelle",
        }),
        "de" | "german" => Some(PocketLanguageSpec {
            language: "de",
            default_voice: "juergen",
        }),
        "it" | "italian" => Some(PocketLanguageSpec {
            language: "it",
            default_voice: "giovanni",
        }),
        "pt" | "portuguese" => Some(PocketLanguageSpec {
            language: "pt",
            default_voice: "rafael",
        }),
        "es" | "spanish" => Some(PocketLanguageSpec {
            language: "es",
            default_voice: "lola",
        }),
        _ => None,
    }
}

// ============================================================================
// Text processing (shared)
// ============================================================================

/// Max chunk size in characters for TTS batching.
const TTS_CHUNK_MAX_CHARS: usize = 250;
const STREAM_TTS_TIMEOUT: Duration = Duration::from_millis(800);
const STREAM_TTS_TIMEOUT_MIN_CHARS: usize = 80;
const STREAM_TTS_TIMEOUT_TARGET_CHARS: usize = 160;
const STREAM_TTS_EAGER_SENTENCE_MIN_CHARS: usize = 40;

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

fn send_audio_chunk(
    audio_tx: &std_mpsc::SyncSender<AudioChunk>,
    samples: &[f32],
    sample_rate: u32,
) -> bool {
    if samples.is_empty() {
        return true;
    }
    audio_tx
        .send(AudioChunk {
            data: samples_to_f32le_bytes(samples),
            sample_rate,
        })
        .is_ok()
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
        while player.is_playing() {
            if cancel.load(Ordering::Relaxed) {
                player.stop();
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
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
    /// Synthesize text with a stable TTS language (ISO code e.g. "en", "it", "es").
    Synthesize {
        text: String,
        language: String,
    },
    Finish,
}

/// Find the earliest occurrence of any `needles` substring in `hay`, returning
/// its byte index and the matched needle. Used to strip reasoning tags where a
/// model may emit either `<think>` or `<thinking>` variants.
fn earliest_tag<'a>(hay: &str, needles: &[&'a str]) -> Option<(usize, &'a str)> {
    needles
        .iter()
        .filter_map(|n| hay.find(n).map(|i| (i, *n)))
        .min_by_key(|(i, _)| *i)
}

/// Accumulates streaming text deltas and batches complete sentences into ~200-char
/// chunks before sending to TTS.
pub(crate) struct SentenceAccumulator {
    buffer: String,
    pending: String,
    in_code_block: bool,
    in_thinking_block: bool,
    tool_filter: ToolCallSpeechFilter,
    sentence_tx: std_mpsc::Sender<TtsCommand>,
    eager: bool,
    first_buffered: Option<std::time::Instant>,
    language_override: Option<String>,
}

impl SentenceAccumulator {
    pub fn new(sentence_tx: std_mpsc::Sender<TtsCommand>) -> Self {
        Self::with_mode(sentence_tx, false, None)
    }

    /// Create an accumulator that sends each sentence immediately for low-latency
    /// streaming TTS.
    pub fn new_streaming(sentence_tx: std_mpsc::Sender<TtsCommand>) -> Self {
        Self::with_mode(sentence_tx, true, None)
    }

    /// Create a streaming accumulator with a stable session language. Passing
    /// `None` or `"auto"` keeps full-text language detection for each emitted chunk.
    pub fn new_streaming_with_language(
        sentence_tx: std_mpsc::Sender<TtsCommand>,
        lang: Option<&str>,
    ) -> Self {
        Self::with_mode(sentence_tx, true, normalize_language_override(lang))
    }

    fn with_mode(
        sentence_tx: std_mpsc::Sender<TtsCommand>,
        eager: bool,
        language_override: Option<String>,
    ) -> Self {
        Self {
            buffer: String::new(),
            pending: String::new(),
            in_code_block: false,
            in_thinking_block: false,
            tool_filter: ToolCallSpeechFilter::new(),
            sentence_tx,
            eager,
            first_buffered: None,
            language_override,
        }
    }

    pub fn push(&mut self, delta: &str) {
        let delta = self.tool_filter.filter(delta);
        if delta.is_empty() {
            return;
        }
        self.buffer.push_str(&delta);
        self.strip_thinking_from_buffer();

        if self.first_buffered.is_none() && !self.buffer.trim().is_empty() {
            self.first_buffered = Some(std::time::Instant::now());
        }
        let before_extract_len = self.buffer.len();
        self.extract_sentences();
        if self.buffer.trim().is_empty() {
            self.first_buffered = None;
        } else if self.buffer.len() < before_extract_len {
            self.first_buffered = Some(std::time::Instant::now());
        }
        if self.eager && !self.in_code_block {
            self.try_timeout_flush();
        }
    }

    fn strip_thinking_from_buffer(&mut self) {
        // Reasoning models emit either <think>…</think> or <thinking>…</thinking>.
        // Strip whole blocks; also drop a stray closing tag whose opener arrived
        // in an earlier segment (common on the non-streaming continuation path,
        // where a bare "</think>" can lead the continuation text and otherwise
        // gets spoken aloud).
        const OPENS: [&str; 2] = ["<thinking>", "<think>"];
        const CLOSES: [&str; 2] = ["</thinking>", "</think>"];
        loop {
            if self.in_thinking_block {
                if let Some((end, close)) = earliest_tag(&self.buffer, &CLOSES) {
                    self.buffer = self.buffer[end + close.len()..].to_string();
                    self.in_thinking_block = false;
                } else {
                    self.buffer.clear();
                    return;
                }
            } else if let Some((start, open)) = earliest_tag(&self.buffer, &OPENS) {
                let before = self.buffer[..start].to_string();
                let after_tag = self.buffer[start + open.len()..].to_string();
                self.in_thinking_block = true;
                if let Some((end, close)) = earliest_tag(&after_tag, &CLOSES) {
                    let remaining = after_tag[end + close.len()..].to_string();
                    self.buffer = format!("{}{}", before, remaining);
                    self.in_thinking_block = false;
                } else {
                    self.buffer = before;
                    return;
                }
            } else if let Some((idx, close)) = earliest_tag(&self.buffer, &CLOSES) {
                // Orphan closing tag, no opener in view — drop just the tag.
                self.buffer.replace_range(idx..idx + close.len(), "");
            } else {
                return;
            }
        }
    }

    fn try_timeout_flush(&mut self) {
        if let Some(t) = self.first_buffered {
            if t.elapsed() >= STREAM_TTS_TIMEOUT {
                let Some(split_at) = find_streaming_timeout_boundary(&self.buffer) else {
                    return;
                };

                let text = self.buffer[..split_at].trim();
                let cleaned = strip_inline_markdown(text);
                self.buffer = self.buffer[split_at..].trim_start().to_string();

                if !cleaned.is_empty() {
                    if self.pending.is_empty() {
                        self.send_to_tts(cleaned);
                    } else if self.pending.len() + 1 + cleaned.len() <= TTS_CHUNK_MAX_CHARS {
                        self.pending.push(' ');
                        self.pending.push_str(&cleaned);
                        let batch = std::mem::take(&mut self.pending);
                        let _ = self.send_to_tts_raw(batch);
                    } else {
                        let batch = std::mem::take(&mut self.pending);
                        let _ = self.send_to_tts_raw(batch);
                        self.send_to_tts(cleaned);
                    }
                }
                self.first_buffered = if self.buffer.trim().is_empty() {
                    None
                } else {
                    Some(std::time::Instant::now())
                };
            }
        }
    }

    pub fn flush(self) {
        let SentenceAccumulator {
            buffer,
            mut pending,
            in_code_block,
            sentence_tx,
            language_override,
            ..
        } = self;

        let remainder = buffer.trim().to_string();
        if !remainder.is_empty() && !in_code_block {
            let cleaned = strip_inline_markdown(&remainder);
            if !cleaned.is_empty() {
                if !pending.is_empty() {
                    pending.push(' ');
                }
                pending.push_str(&cleaned);
            }
        }
        if !pending.is_empty() {
            let language = language_override.unwrap_or_else(|| detect_language(&pending));
            let _ = sentence_tx.send(TtsCommand::Synthesize {
                text: pending,
                language,
            });
        }
        let _ = sentence_tx.send(TtsCommand::Finish);
    }

    /// Send text to TTS with auto-detected language.
    fn send_to_tts(&self, text: String) {
        let language = self.language_for_tts(&text);
        let _ = self
            .sentence_tx
            .send(TtsCommand::Synthesize { text, language });
    }

    /// Same as send_to_tts but returns the Result (for use with `let _ =`).
    fn send_to_tts_raw(&self, text: String) -> Result<(), std_mpsc::SendError<TtsCommand>> {
        let language = self.language_for_tts(&text);
        self.sentence_tx
            .send(TtsCommand::Synthesize { text, language })
    }

    fn language_for_tts(&self, text: &str) -> String {
        self.language_override
            .clone()
            .unwrap_or_else(|| detect_language(text))
    }

    fn enqueue_sentence(&mut self, sentence: &str) {
        // Batch very short sentences to give TTS enough context, then flush
        // completed text in eager mode once it is long enough to be coherent.
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
        if self.eager && self.pending.len() >= STREAM_TTS_EAGER_SENTENCE_MIN_CHARS {
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

fn normalize_language_override(lang: Option<&str>) -> Option<String> {
    let lang = lang?.trim();
    if lang.is_empty() || lang.eq_ignore_ascii_case("auto") {
        None
    } else {
        Some(lang.to_string())
    }
}

fn find_streaming_timeout_boundary(text: &str) -> Option<usize> {
    if text.trim().len() < STREAM_TTS_TIMEOUT_MIN_CHARS {
        return None;
    }

    let mut best_before_target = None;
    let mut first_after_target = None;

    for (idx, ch) in text.char_indices() {
        if !is_safe_stream_boundary(ch) {
            continue;
        }

        let boundary = if ch.is_whitespace() {
            idx
        } else {
            idx + ch.len_utf8()
        };
        if boundary == 0 {
            continue;
        }

        let prefix_len = text[..boundary].trim().len();
        if prefix_len < STREAM_TTS_TIMEOUT_MIN_CHARS {
            continue;
        }

        if boundary <= STREAM_TTS_TIMEOUT_TARGET_CHARS {
            best_before_target = Some(boundary);
        } else if boundary <= TTS_CHUNK_MAX_CHARS {
            first_after_target = Some(boundary);
            break;
        } else {
            break;
        }
    }

    best_before_target.or(first_after_target)
}

fn is_safe_stream_boundary(ch: char) -> bool {
    ch.is_whitespace() || matches!(ch, ',' | ';' | ':' | ')' | ']')
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ToolCallSpeechState {
    Normal,
    DropUntil(&'static str),
}

/// Streaming text filter for model families that leak textual tool-call markup
/// before the structured tool call is parsed.
struct ToolCallSpeechFilter {
    state: ToolCallSpeechState,
    partial: String,
}

impl ToolCallSpeechFilter {
    fn new() -> Self {
        Self {
            state: ToolCallSpeechState::Normal,
            partial: String::new(),
        }
    }

    fn filter(&mut self, delta: &str) -> String {
        let mut text = if self.partial.is_empty() {
            delta.to_string()
        } else {
            let mut combined = std::mem::take(&mut self.partial);
            combined.push_str(delta);
            combined
        };

        let mut out = String::new();
        loop {
            match self.state {
                ToolCallSpeechState::DropUntil(end_tag) => {
                    if let Some(end) = text.find(end_tag) {
                        let after = end + end_tag.len();
                        text = text[after..].to_string();
                        self.state = ToolCallSpeechState::Normal;
                        continue;
                    }
                    return out;
                }
                ToolCallSpeechState::Normal => {
                    if let Some((start, end_tag)) = find_tool_call_start(&text) {
                        out.push_str(&text[..start]);
                        let rest = &text[start..];
                        if let Some(end) = rest.find(end_tag) {
                            let after = end + end_tag.len();
                            text = rest[after..].to_string();
                            continue;
                        }
                        self.state = ToolCallSpeechState::DropUntil(end_tag);
                        return out;
                    }

                    let safe_len = hold_partial_tool_marker(&text);
                    if safe_len < text.len() {
                        self.partial = text[safe_len..].to_string();
                        out.push_str(&text[..safe_len]);
                    } else {
                        out.push_str(&text);
                    }
                    return out;
                }
            }
        }
    }
}

const TOOL_CALL_MARKERS: [(&str, &str); 4] = [
    ("<tool_call", "</tool_call>"),
    ("<function=", "</function>"),
    ("<start_function_call>", "<end_function_call>"),
    ("<|python_tag|>", ")"),
];

fn find_tool_call_start(text: &str) -> Option<(usize, &'static str)> {
    TOOL_CALL_MARKERS
        .iter()
        .filter_map(|(start, end)| text.find(start).map(|idx| (idx, *end)))
        .min_by_key(|(idx, _)| *idx)
}

fn hold_partial_tool_marker(text: &str) -> usize {
    for (marker, _) in TOOL_CALL_MARKERS {
        let max_prefix = marker.len().saturating_sub(1).min(text.len());
        for len in (1..=max_prefix).rev() {
            if text.ends_with(&marker[..len]) {
                return text.len() - len;
            }
        }
    }
    text.len()
}

/// Remove textual tool-call markup from a complete response before TTS cleanup.
pub(crate) fn strip_tool_calls_for_tts(text: &str) -> String {
    let mut filter = ToolCallSpeechFilter::new();
    filter.filter(text)
}

/// Short spoken cue for a tool action, used to narrate tool execution in voice
/// mode. Deliberately speaks only the *action*, never the parameters or output
/// (those carry paths, commands, and data that shouldn't be read aloud).
pub(crate) fn tool_speech_cue(tool_name: &str) -> String {
    match tool_name {
        "exec" => "running a command".to_string(),
        "web_search" => "searching the web".to_string(),
        "web_fetch" | "read_url" => "reading a web page".to_string(),
        "read_file" => "reading a file".to_string(),
        "write_file" => "writing a file".to_string(),
        "edit_file" => "editing a file".to_string(),
        "recall" => "checking memory".to_string(),
        other => format!("using {}", other),
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
/// Per-sentence TTS routing: select engine + voice based on detected language.
///
/// Shared by `VoicePipeline::select_tts`, `start_streaming_speak`, and
/// `start_tts_playback` so language changes mid-session always switch voices.
fn route_tts(
    lang: &str,
    engine_config: &TtsEngineConfig,
    configured_voice: Option<&str>,
    tts_en: &Option<Arc<Mutex<TextToSpeech>>>,
    tts_multi: &Option<Arc<Mutex<TextToSpeech>>>,
    tts_pocket_by_lang: &Arc<Mutex<HashMap<String, Arc<Mutex<TextToSpeech>>>>>,
) -> Result<(Arc<Mutex<TextToSpeech>>, String), String> {
    if *engine_config == TtsEngineConfig::Supertonic {
        let tts = tts_en
            .as_ref()
            .or(tts_multi.as_ref())
            .ok_or("No TTS engine available")?
            .clone();
        let engine_type = {
            let mut guard = tts.lock();
            let engine_type = guard.engine_type().to_string();
            if engine_type == "supertonic" {
                // Per-message language switch: tell Supertonic to wrap the input
                // with the language tag (<it>…</it>, <es>…</es>, etc.). The model
                // is multilingual; voice stays the same across languages.
                if let Err(e) = guard.set_language(lang) {
                    tracing::warn!("Supertonic language switch to {} failed: {}", lang, e);
                }
            }
            engine_type
        };
        let voice_id = if engine_type == "supertonic" {
            // Voice selection — single persona across all languages:
            //   1. Explicit ttsVoice from config (e.g. "F5") — user's choice.
            //   2. Curated per-language pick from jack_voice when nothing set.
            configured_voice.map(|s| s.to_string()).unwrap_or_else(|| {
                jack_voice::tts::recommended_supertonic_voice(Some(lang)).to_string()
            })
        } else if engine_type == "pocket" {
            pocket_tts_language(lang)
                .map(|spec| spec.default_voice)
                .unwrap_or("alba")
                .to_string()
        } else {
            let (vid, _) = language_to_kokoro_voice(lang);
            vid.to_string()
        };
        return Ok((tts, voice_id));
    }

    if *engine_config != TtsEngineConfig::Kokoro {
        if let Some(spec) = pocket_tts_language(lang) {
            if let Some(tts) = tts_pocket_by_lang.lock().get(spec.language).cloned() {
                return Ok((tts, spec.default_voice.to_string()));
            }
            match TextToSpeech::new_pocket_with_language(spec.language) {
                Ok(tts) => {
                    let tts = Arc::new(Mutex::new(tts));
                    tts_pocket_by_lang
                        .lock()
                        .insert(spec.language.to_string(), tts.clone());
                    return Ok((tts, spec.default_voice.to_string()));
                }
                Err(e) => {
                    tracing::warn!(
                        "Pocket TTS init failed for {}: {e}; falling back",
                        spec.language
                    );
                }
            }
        }
    }

    let tts = tts_multi
        .as_ref()
        .or(tts_en.as_ref())
        .ok_or("No TTS engine available")?
        .clone();

    let voice_id = if tts.lock().engine_type() == "pocket" {
        pocket_tts_language(lang)
            .map(|spec| spec.default_voice)
            .unwrap_or("alba")
            .to_string()
    } else {
        let (vid, _) = language_to_kokoro_voice(lang);
        vid.to_string()
    };

    Ok((tts, voice_id))
}

/// Holds STT and TTS engines behind `Arc<Mutex<>>` for thread-safe access.
/// Two construction modes:
/// - `with_engine()` / `with_lang()` — mic+speaker mode for TUI/realtime
/// - `for_channels()` — file I/O only, no audio hardware
pub struct VoicePipeline {
    stt: Arc<Mutex<SpeechToText>>,
    tts_en: Option<Arc<Mutex<TextToSpeech>>>,
    tts_multi: Option<Arc<Mutex<TextToSpeech>>>,
    tts_pocket_by_lang: Arc<Mutex<HashMap<String, Arc<Mutex<TextToSpeech>>>>>,
    engine_config: TtsEngineConfig,
    /// User-configured voice ID from `~/.nanobot/config.json` (`voice.ttsVoice`).
    /// Applied to Supertonic across all languages — "single persona, multilingual".
    /// When `None`, the curated per-language voice picker takes over.
    configured_voice: Option<String>,
    cancel: Arc<AtomicBool>,
}

impl VoicePipeline {
    // ----------------------------------------------------------------
    // Constructors
    // ----------------------------------------------------------------

    /// Create a pipeline for channel use (file I/O, no mic/speaker).
    /// Uses default voice. For config-driven voice selection, see
    /// [`Self::for_channels_with_voice_config`].
    pub async fn for_channels(engine: TtsEngineConfig) -> Result<Self, String> {
        Self::init_pipeline(engine, None, None, &LogProgress).await
    }

    /// Create a channel-mode pipeline from a fully-resolved [`VoiceConfig`].
    pub async fn for_channels_with_voice_config(
        cfg: &crate::config::schema::VoiceConfig,
    ) -> Result<Self, String> {
        Self::init_pipeline(
            cfg.tts_engine,
            cfg.language.as_deref(),
            cfg.tts_voice.as_deref(),
            &LogProgress,
        )
        .await
    }

    /// Create a pipeline with optional language-based engine selection (mic mode).
    ///
    /// - `None` → load Pocket (English) and Kokoro fallback; other Pocket languages load lazily
    /// - `Some("en" | "fr" | "de" | "it" | "pt" | "es")` → load Pocket for that language
    /// - `Some(_)` → load only Kokoro
    pub async fn with_lang(lang: Option<&str>) -> Result<Self, String> {
        let pocket_lang = lang.and_then(pocket_tts_language);
        let load_pocket = lang.is_none() || pocket_lang.is_some();
        let load_kokoro = lang.is_none() || pocket_lang.is_none();
        let initial_pocket_lang = pocket_lang.map(|spec| spec.language).unwrap_or("en");

        if load_kokoro && std::env::var("PIPER_ESPEAKNG_DATA_DIRECTORY").is_err() {
            let home = dirs::home_dir().unwrap_or_default();
            let local_data = home.join(".local/share/espeak-ng-data");
            if local_data.exists() {
                std::env::set_var("PIPER_ESPEAKNG_DATA_DIRECTORY", home.join(".local/share"));
            }
        }

        let label = match lang {
            Some(lang) if pocket_tts_language(lang).is_some() => "Pocket multilingual",
            Some(_) => "Kokoro only",
            None => "Pocket multilingual + Kokoro fallback",
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
            match tokio::task::spawn_blocking(move || {
                TextToSpeech::new_pocket_with_language(initial_pocket_lang)
            })
            .await
            .map_err(|e| format!("spawn_blocking join error: {e}"))?
            {
                Ok(tts) => {
                    info!(
                        "{} TTS ready ({initial_pocket_lang}) [engine: {}]",
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
                    tracing::warn!(
                        "Pocket TTS init failed for {initial_pocket_lang}, fallback may be used: {e}"
                    );
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

        let tts_pocket_by_lang = Arc::new(Mutex::new(HashMap::new()));
        if let Some(tts) = &tts_en {
            tts_pocket_by_lang
                .lock()
                .insert(initial_pocket_lang.to_string(), tts.clone());
        }

        if tts_en.is_none() && tts_multi.is_none() {
            return Err("No TTS engine could be initialized".to_string());
        }

        Ok(Self {
            stt: Arc::new(Mutex::new(stt)),
            tts_en,
            tts_multi,
            tts_pocket_by_lang,
            engine_config: TtsEngineConfig::Pocket,
            configured_voice: None,
            cancel: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Create a pipeline with a specific TTS engine (mic mode).
    /// Uses default voice for the engine. For config-driven voice selection,
    /// see [`Self::with_voice_config`].
    pub async fn with_engine(engine: TtsEngineConfig) -> Result<Self, String> {
        Self::init_pipeline(engine, None, None, &TerminalProgress).await
    }

    /// Create a pipeline from a fully-resolved [`VoiceConfig`].
    ///
    /// Honors `tts_voice` and `language`:
    ///   - If `tts_voice` is set, it's applied to the constructed TTS via
    ///     `set_speaker` (e.g. `"F5"` for SuperTonic Italian female).
    ///   - If `tts_voice` is None and the engine is Supertonic, we look up
    ///     `jack_voice::tts::recommended_supertonic_voice(language)` to get
    ///     the curated per-language pick (M2 for Italian/English, etc.).
    ///   - For Kokoro/Pocket, None means "engine's default voice".
    pub async fn with_voice_config(
        cfg: &crate::config::schema::VoiceConfig,
    ) -> Result<Self, String> {
        Self::init_pipeline(
            cfg.tts_engine,
            cfg.language.as_deref(),
            cfg.tts_voice.as_deref(),
            &TerminalProgress,
        )
        .await
    }

    /// Internal: initialize the pipeline with the given engine and progress callback.
    async fn init_pipeline(
        engine: TtsEngineConfig,
        lang: Option<&str>,
        tts_voice: Option<&str>,
        progress: &(dyn ModelProgressCallback + Sync),
    ) -> Result<Self, String> {
        info!(
            "Initializing voice pipeline (engine={:?}, lang={:?}, voice={:?})...",
            engine, lang, tts_voice
        );

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
                    TextToSpeech::new_pocket_with_language("en")
                })
                .await
                .map_err(|e| format!("spawn_blocking join error: {e}"))?
                {
                    Ok(tts) => {
                        info!("Pocket TTS ready (en)");
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
            TtsEngineConfig::Supertonic => {
                models::ensure_supertonic_models(progress)
                    .await
                    .map_err(|e| format!("Model download failed: {e}"))?;

                let tts_en = match tokio::task::spawn_blocking(|| {
                    TextToSpeech::with_engine(TtsEngine::Supertonic)
                })
                .await
                .map_err(|e| format!("spawn_blocking join error: {e}"))?
                {
                    Ok(tts) => {
                        info!("Supertonic TTS ready (44.1kHz)");
                        Some(Arc::new(Mutex::new(tts)))
                    }
                    Err(e) => {
                        info!("Supertonic TTS not available: {e}");
                        None
                    }
                };

                let tts_multi = if tts_en.is_none() {
                    models::ensure_kokoro_model(progress)
                        .await
                        .map_err(|e| format!("Model download failed: {e}"))?;

                    match tokio::task::spawn_blocking(|| {
                        TextToSpeech::with_engine(TtsEngine::Kokoro)
                    })
                    .await
                    .map_err(|e| format!("spawn_blocking join error: {e}"))?
                    {
                        Ok(tts) => {
                            info!("Kokoro TTS ready (Supertonic fallback)");
                            Some(Arc::new(Mutex::new(tts)))
                        }
                        Err(e) => {
                            info!("Kokoro TTS fallback not available: {e}");
                            None
                        }
                    }
                } else {
                    None
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

        // Apply configured voice (or curated per-language default) to the
        // primary TTS handle. We do NOT touch tts_multi here — that's the
        // multilingual fallback (typically Kokoro) which uses its own voice
        // selection based on detected language at synth time.
        if let Some(tts) = &tts_en {
            let voice_to_apply: Option<String> = match (tts_voice, engine) {
                // Explicit voice from config always wins.
                (Some(v), _) => Some(v.to_string()),
                // Supertonic + no explicit voice → curated per-language pick.
                (None, TtsEngineConfig::Supertonic) => {
                    Some(jack_voice::tts::recommended_supertonic_voice(lang).to_string())
                }
                // Other engines: leave their own default in place.
                (None, _) => None,
            };
            if let Some(voice_id) = voice_to_apply {
                let mut guard = tts.lock();
                match guard.set_speaker(&voice_id) {
                    Ok(()) => {
                        info!(
                            "Applied configured TTS voice: engine={:?}, voice={}",
                            engine, voice_id
                        );
                    }
                    Err(e) => {
                        // Don't fail the whole pipeline — log and proceed with the engine's
                        // default voice. The user will hear *something* and see the warning.
                        tracing::warn!(
                            "Failed to apply configured voice {:?} for engine {:?}: {}. Falling back to engine default.",
                            voice_id, engine, e
                        );
                    }
                }
            }
        }

        let tts_pocket_by_lang = Arc::new(Mutex::new(HashMap::new()));
        if let Some(tts) = &tts_en {
            tts_pocket_by_lang
                .lock()
                .insert("en".to_string(), tts.clone());
        }

        if tts_en.is_none() && tts_multi.is_none() {
            return Err("No TTS engine could be initialized".to_string());
        }

        info!("Voice pipeline ready");

        Ok(Self {
            stt: Arc::new(Mutex::new(stt)),
            tts_en,
            tts_multi,
            tts_pocket_by_lang,
            engine_config: engine,
            // Stash the user's configured voice (None if they didn't set one).
            // route_tts() reads this on every synthesis so the persona stays
            // consistent across mid-session language switches.
            configured_voice: tts_voice.map(|s| s.to_string()),
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

    fn get_or_init_pocket_tts(
        &self,
        spec: PocketLanguageSpec,
    ) -> Result<Arc<Mutex<TextToSpeech>>, String> {
        if let Some(tts) = self.tts_pocket_by_lang.lock().get(spec.language).cloned() {
            return Ok(tts);
        }

        let tts = TextToSpeech::new_pocket_with_language(spec.language)
            .map_err(|e| format!("Pocket TTS init failed for {}: {e}", spec.language))?;
        let tts = Arc::new(Mutex::new(tts));
        self.tts_pocket_by_lang
            .lock()
            .insert(spec.language.to_string(), tts.clone());
        Ok(tts)
    }

    /// Select the appropriate TTS engine based on language and config.
    fn select_tts(&self, lang: &str) -> Result<(Arc<Mutex<TextToSpeech>>, String), String> {
        route_tts(
            lang,
            &self.engine_config,
            self.configured_voice.as_deref(),
            &self.tts_en,
            &self.tts_multi,
            &self.tts_pocket_by_lang,
        )
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
                match guard.synthesize_streaming(sentence, |samples, sample_rate| {
                    if cancel_ref.load(Ordering::Relaxed) {
                        return false;
                    }
                    send_audio_chunk(&audio_tx, samples, sample_rate)
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
        _lang: &str,
        display_tx: Option<tokio::sync::mpsc::UnboundedSender<String>>,
    ) -> Result<(std_mpsc::Sender<TtsCommand>, std::thread::JoinHandle<()>), String> {
        let tts_en = self.tts_en.clone();
        let tts_multi = self.tts_multi.clone();
        let tts_pocket_by_lang = self.tts_pocket_by_lang.clone();
        let engine_config = self.engine_config.clone();
        let configured_voice = self.configured_voice.clone();
        let cancel = self.cancel.clone();

        let (sentence_tx, sentence_rx) = std_mpsc::channel::<TtsCommand>();
        let (audio_tx, audio_rx) = std_mpsc::sync_channel::<AudioChunk>(2);

        let cancel_synth = cancel.clone();
        let synth_handle = std::thread::spawn(move || {
            #[cfg(unix)]
            mask_sigint();

            for cmd in sentence_rx {
                match cmd {
                    TtsCommand::Finish => break,
                    TtsCommand::Synthesize {
                        text: sentence,
                        language,
                    } => {
                        if cancel_synth.load(Ordering::Relaxed) {
                            break;
                        }
                        if let Some(ref dtx) = display_tx {
                            let _ = dtx.send(sentence.clone());
                        }
                        let (tts, voice_id) = match route_tts(
                            &language,
                            &engine_config,
                            configured_voice.as_deref(),
                            &tts_en,
                            &tts_multi,
                            &tts_pocket_by_lang,
                        ) {
                            Ok(r) => r,
                            Err(e) => {
                                tracing::error!("TTS routing failed: {}", e);
                                continue;
                            }
                        };
                        let mut guard = tts.lock();
                        if let Err(e) = guard.set_speaker(&voice_id) {
                            tracing::warn!("Voice switch to {} failed: {}", voice_id, e);
                        }
                        let cancel_ref = &cancel_synth;
                        match guard.synthesize_streaming(&sentence, |samples, sample_rate| {
                            if cancel_ref.load(Ordering::Relaxed) {
                                return false;
                            }
                            send_audio_chunk(&audio_tx, samples, sample_rate)
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
        let (tts, voice_id) = self.select_tts(&lang)?;

        let (all_samples, sample_rate) = tokio::task::spawn_blocking(move || {
            let mut guard = tts.lock();
            guard
                .set_speaker(&voice_id)
                .map_err(|e| format!("Voice switch failed: {e}"))?;

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

impl Drop for VoicePipeline {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);

        // Some native TTS backends (notably Kokoro in jack-voice) own a Tokio
        // runtime internally. Channel pipelines are dropped during async gateway
        // shutdown, and dropping that runtime from async context panics. This
        // mirrors `shutdown()`: leak the long-lived native TTS handles and let
        // the OS reclaim them when the process exits.
        if let Some(tts) = self.tts_en.take() {
            std::mem::forget(tts);
        }
        if let Some(tts) = self.tts_multi.take() {
            std::mem::forget(tts);
        }
        for (_, tts) in self.tts_pocket_by_lang.lock().drain() {
            std::mem::forget(tts);
        }
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
    let tts_pocket_by_lang = Arc::new(parking_lot::Mutex::new(HashMap::new()));
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
            let _ = g.set_speaker("35");
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

                    let (tts, voice_id) = match route_tts(
                        &language,
                        &TtsEngineConfig::Pocket,
                        None, // legacy path — no configured voice plumbed here
                        &tts_en,
                        &tts_multi,
                        &tts_pocket_by_lang,
                    ) {
                        Ok(r) => r,
                        Err(e) => {
                            tracing::warn!("[tts-synth] TTS routing failed: {}", e);
                            continue;
                        }
                    };

                    let mut guard = tts.lock();
                    if let Err(e) = guard.set_speaker(&voice_id) {
                        tracing::warn!("[tts-synth] voice switch to {} failed: {}", voice_id, e);
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
    fn test_tool_speech_cue_known_and_default() {
        assert_eq!(tool_speech_cue("exec"), "running a command");
        assert_eq!(tool_speech_cue("web_search"), "searching the web");
        assert_eq!(tool_speech_cue("read_file"), "reading a file");
        assert_eq!(tool_speech_cue("recall"), "checking memory");
        // Unknown tools fall back to a generic cue naming the tool.
        assert_eq!(tool_speech_cue("frobnicate"), "using frobnicate");
    }

    #[test]
    fn test_tool_speech_cue_passes_through_speech_filter() {
        // A cue must survive the tool-call speech filter (no markup to strip).
        let mut filter = ToolCallSpeechFilter::new();
        let cue = tool_speech_cue("exec");
        assert_eq!(filter.filter(&cue), cue);
    }

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
    fn test_pocket_tts_language_supports_multilingual_defaults() {
        let cases = [
            ("en", "en", "alba"),
            ("en-US", "en", "alba"),
            ("fr", "fr", "estelle"),
            ("de", "de", "juergen"),
            ("it", "it", "giovanni"),
            ("pt", "pt", "rafael"),
            ("es", "es", "lola"),
        ];

        for (input, expected_lang, expected_voice) in cases {
            let spec = pocket_tts_language(input).expect("language should be supported by Pocket");
            assert_eq!(spec.language, expected_lang);
            assert_eq!(spec.default_voice, expected_voice);
        }
    }

    #[test]
    fn test_pocket_tts_language_rejects_kokoro_fallback_languages() {
        for lang in ["ja", "zh", "hi", "vi", ""] {
            assert!(
                pocket_tts_language(lang).is_none(),
                "{lang:?} should use Kokoro fallback, not Pocket"
            );
        }
    }

    #[test]
    fn test_strip_thinking_handles_think_and_thinking_tags() {
        let (tx, _rx) = std_mpsc::channel();
        let mut acc = SentenceAccumulator::new(tx);

        // <think>…</think> (the variant local models emit) is removed.
        acc.buffer = "<think>secret reasoning</think>Hello".to_string();
        acc.strip_thinking_from_buffer();
        assert_eq!(acc.buffer, "Hello");
        assert!(!acc.in_thinking_block);

        // Legacy <thinking>…</thinking> still stripped.
        acc.buffer = "<thinking>hidden</thinking>World".to_string();
        acc.strip_thinking_from_buffer();
        assert_eq!(acc.buffer, "World");

        // Stray closing tag (opener arrived in an earlier segment) is dropped,
        // not spoken — this is the </think> leak seen on the continuation path.
        acc.in_thinking_block = false;
        acc.buffer = "tail of thought</think> the real answer".to_string();
        acc.strip_thinking_from_buffer();
        assert_eq!(acc.buffer, "tail of thought the real answer");
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

    fn collect_tts(rx: &std::sync::mpsc::Receiver<TtsCommand>) -> Vec<(String, String)> {
        let mut out = Vec::new();
        while let Ok(cmd) = rx.try_recv() {
            if let TtsCommand::Synthesize { text, language } = cmd {
                out.push((text, language));
            }
        }
        out
    }

    #[test]
    fn test_streaming_timeout_flush_splits_only_at_safe_boundaries() {
        let original = "Cambiare il TTS velocemente senza tagliare le parole dentro ai frammenti vocali finali mentre il testo continua";
        let (tx, rx) = std::sync::mpsc::channel();
        let mut acc = SentenceAccumulator::new_streaming(tx);
        acc.push(original);
        acc.first_buffered = Some(std::time::Instant::now() - STREAM_TTS_TIMEOUT);
        acc.try_timeout_flush();

        let chunks = collect_tts(&rx);
        assert_eq!(chunks.len(), 1);
        let emitted = &chunks[0].0;
        assert!(original.starts_with(emitted));
        assert_eq!(original.as_bytes().get(emitted.len()), Some(&b' '));
        assert!(!acc.buffer.trim().is_empty());
        assert!(original.ends_with(acc.buffer.trim()));
    }

    #[test]
    fn test_streaming_timeout_flush_preserves_pending_order() {
        let (tx, rx) = std::sync::mpsc::channel();
        let mut acc = SentenceAccumulator::new_streaming(tx);
        acc.pending = "Capisco, vuoi accelerare il TTS.".to_string();
        acc.buffer = "Cambiare il modello di sintesi vocale senza tagliare la prima parola mentre il testo continua".to_string();
        acc.first_buffered = Some(std::time::Instant::now() - STREAM_TTS_TIMEOUT);
        acc.try_timeout_flush();

        let chunks = collect_tts(&rx);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0]
            .0
            .starts_with("Capisco, vuoi accelerare il TTS. Cambiare"));
        assert!(!chunks[0].0.contains("TTS. mbiare"));
    }

    #[test]
    fn test_streaming_timeout_flush_does_not_split_single_word() {
        let original = "supercalifragilisticexpialidocioussupercalifragilisticexpialidocioussupercalifragilistic";
        let (tx, rx) = std::sync::mpsc::channel();
        let mut acc = SentenceAccumulator::new_streaming(tx);
        acc.push(original);
        acc.first_buffered = Some(std::time::Instant::now() - STREAM_TTS_TIMEOUT);
        acc.try_timeout_flush();

        assert!(collect_tts(&rx).is_empty());
        assert_eq!(acc.buffer, original);
    }

    #[test]
    fn test_streaming_accumulator_uses_language_override() {
        let (tx, rx) = std::sync::mpsc::channel();
        let mut acc = SentenceAccumulator::new_streaming_with_language(tx, Some("it"));
        acc.push("Hello world. This response should keep the session language for TTS.");
        acc.flush();

        let chunks = collect_tts(&rx);
        assert!(!chunks.is_empty());
        assert!(chunks.iter().all(|(_, lang)| lang == "it"));
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
    fn test_sentence_accumulator_strips_tool_call_across_pushes() {
        let (tx, rx) = std::sync::mpsc::channel();
        let mut acc = SentenceAccumulator::new(tx);
        acc.push("I will check. <tool_cal");
        acc.push("l>{\"name\":\"read_file\",\"arguments\":{\"path\":\"secret.txt\"}}</tool_call>");
        acc.push(" The result is ready.");
        acc.flush();

        let mut sentences = Vec::new();
        while let Ok(cmd) = rx.try_recv() {
            if let TtsCommand::Synthesize { text: s, .. } = cmd {
                sentences.push(s);
            }
        }
        let combined = sentences.join(" ");
        assert!(combined.contains("I will check."));
        assert!(combined.contains("The result is ready."));
        assert!(!combined.contains("tool_call"));
        assert!(!combined.contains("read_file"));
        assert!(!combined.contains("secret.txt"));
    }

    #[test]
    fn test_strip_tool_calls_for_tts_handles_textual_formats() {
        let text = concat!(
            "Before ",
            "<function=exec>{\"command\":\"ls\"}</function> ",
            "<start_function_call>{\"command\":\"pwd\"}<end_function_call> ",
            "<|python_tag|>web_search.call({\"query\":\"rust\"}) ",
            "after."
        );
        let stripped = strip_tool_calls_for_tts(text);

        assert!(stripped.contains("Before"));
        assert!(stripped.contains("after."));
        assert!(!stripped.contains("<function="));
        assert!(!stripped.contains("<start_function_call>"));
        assert!(!stripped.contains("<|python_tag|>"));
        assert!(!stripped.contains("web_search"));
        assert!(!stripped.contains("command"));
        assert!(!stripped.contains("query"));
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
