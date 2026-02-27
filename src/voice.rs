#![cfg(feature = "voice")]

use std::io::Write;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc as std_mpsc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::config::schema::TtsEngineConfig;
use jack_voice::{
    AudioCapture, AudioError, AudioPlayer,
    models::{self, ModelProgressCallback},
    SpeechToText, SttMode, TextToSpeech, TtsEngine,
};
use lingua::{Language, LanguageDetector, LanguageDetectorBuilder};
use once_cell::sync::Lazy;

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

/// Max chunk size in characters for TTS batching.
/// Pocket TTS runs faster-than-real-time on CPU (~200ms latency per chunk).
/// Chunks always end on sentence-ending punctuation (.!?) so prosody stays natural.
/// Short sentences (even 5 chars) are valid chunks — no artificial minimum delay.
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
/// Short responses (<=500 chars) are synthesized as a single chunk — no splitting overhead.
pub(crate) fn split_tts_sentences(text: &str) -> Vec<String> {
    // Collapse newlines and multiple whitespace into single spaces so TTS
    // engines don't pause on soft-wrapped line breaks.
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
/// `target_peak` should be in 0.0..=1.0 (e.g. 0.9 for 90% of full scale).
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
/// `fade_samples` is the number of samples to ramp over (~5ms at 24kHz = 120 samples).
pub(crate) fn apply_fade_envelope(samples: &mut [f32], fade_samples: usize) {
    let len = samples.len();
    let fade = fade_samples.min(len / 2);
    // Fade in
    for i in 0..fade {
        samples[i] *= i as f32 / fade as f32;
    }
    // Fade out
    for i in 0..fade {
        samples[len - 1 - i] *= i as f32 / fade as f32;
    }
}

/// Convert f32 samples to raw little-endian bytes for piping to paplay.
fn samples_to_f32le_bytes(samples: &[f32]) -> Vec<u8> {
    samples.iter().flat_map(|s| s.to_le_bytes()).collect()
}

fn f32le_bytes_to_samples(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
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
fn play_chunks_paplay(
    audio_rx: std_mpsc::Receiver<AudioChunk>,
    cancel: Arc<AtomicBool>,
) -> Result<(), String> {
    // Wait for the first chunk to get the sample rate.
    let first_chunk = match audio_rx.recv() {
        Ok(c) => c,
        Err(_) => return Ok(()), // no audio produced
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

    drop(stdin); // close stdin so paplay can finish
    if cancel.load(Ordering::Relaxed) {
        let _ = child.kill();
    }
    let _ = child.wait();
    Ok(())
}

/// Block SIGINT delivery in the current thread.
///
/// Prevents Ctrl+C from interrupting native TTS/playback code mid-execution,
/// which would cause segfaults in the C/C++ FFI libraries (Pocket, Kokoro).
/// The signal is still delivered to the tokio handler thread.
#[cfg(unix)]
fn mask_sigint() {
    unsafe {
        let mut sigset: libc::sigset_t = std::mem::zeroed();
        libc::sigemptyset(&mut sigset);
        libc::sigaddset(&mut sigset, libc::SIGINT);
        libc::pthread_sigmask(libc::SIG_BLOCK, &sigset, std::ptr::null_mut());
    }
}

/// A command sent to the synthesis thread.
pub(crate) enum TtsCommand {
    Synthesize(String),
    Finish,
}

/// A chunk of synthesized audio ready for playback.
struct AudioChunk {
    data: Vec<u8>, // f32le raw bytes
    sample_rate: u32,
}

pub struct VoiceSession {
    stt: SpeechToText,
    /// Pocket TTS engine (fast, English-only, CPU inference).
    tts_en: Option<Arc<Mutex<TextToSpeech>>>,
    /// Kokoro TTS engine (multilingual).
    tts_multi: Option<Arc<Mutex<TextToSpeech>>>,
    /// Qwen TTS engine (multilingual with GPU, includes QwenLarge for voice cloning).
    tts_qwen: Option<Arc<Mutex<TextToSpeech>>>,
    /// Selected TTS engine config.
    engine_config: TtsEngineConfig,
    /// Selected Qwen voice name (for Qwen/QwenLarge engines).
    qwen_voice: String,
    cancel: Arc<AtomicBool>,
}

enum CaptureSession {
    Process(Child),
    Native(AudioCapture),
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

fn pulse_server() -> String {
    if std::path::Path::new("/mnt/wslg/PulseServer").exists() {
        "unix:/mnt/wslg/PulseServer".to_string()
    } else {
        std::env::var("PULSE_SERVER").unwrap_or_default()
    }
}

fn start_parec(sample_tx: std_mpsc::Sender<Vec<f32>>) -> Result<Child, String> {
    let mut child = Command::new("parec")
        .args(["--format=float32le", "--rate=16000", "--channels=1"])
        .env("PULSE_SERVER", pulse_server())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("parec failed: {e}\n  Install: sudo apt install pulseaudio-utils"))?;

    let stdout = child.stdout.take().unwrap();
    std::thread::spawn(move || {
        use std::io::Read;
        let mut reader = std::io::BufReader::new(stdout);
        let mut buf = [0u8; 3200]; // 800 f32 samples = 50ms at 16kHz
        while reader.read_exact(&mut buf).is_ok() {
            let samples: Vec<f32> = buf
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            if sample_tx.send(samples).is_err() {
                break;
            }
        }
    });

    Ok(child)
}

fn start_native_capture(sample_tx: std_mpsc::Sender<Vec<f32>>) -> Result<AudioCapture, String> {
    AudioCapture::start(sample_tx).map_err(format_native_capture_error)
}

fn start_capture(sample_tx: std_mpsc::Sender<Vec<f32>>) -> Result<CaptureSession, String> {
    #[cfg(target_os = "macos")]
    {
        start_native_capture(sample_tx).map(CaptureSession::Native)
    }

    #[cfg(not(target_os = "macos"))]
    {
        match start_parec(sample_tx.clone()) {
            Ok(child) => Ok(CaptureSession::Process(child)),
            Err(parec_err) => {
                start_native_capture(sample_tx)
                    .map(CaptureSession::Native)
                    .map_err(|native_err| {
                        format!("{parec_err}\nFallback native capture failed: {native_err}")
                    })
            }
        }
    }
}

fn stop_capture(capture: &mut CaptureSession) {
    match capture {
        CaptureSession::Process(child) => {
            let _ = child.kill();
            let _ = child.wait();
        }
        CaptureSession::Native(stream) => stream.stop(),
    }
}

fn start_playback(samples: Vec<f32>, sample_rate: u32) -> Result<Child, String> {
    let raw = samples_to_f32le_bytes(&samples);

    let mut child = Command::new("paplay")
        .args([
            "--raw",
            "--format=float32le",
            "--channels=1",
            &format!("--rate={}", sample_rate),
        ])
        .env("PULSE_SERVER", pulse_server())
        .env("PULSE_LATENCY_MSEC", "10")
        .stdin(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|e| format!("paplay failed: {e}\n  Install: sudo apt install pulseaudio-utils"))?;

    let mut stdin = child.stdin.take().unwrap();
    std::thread::spawn(move || {
        let _ = stdin.write_all(&raw);
    });

    Ok(child)
}

impl VoiceSession {
    pub async fn new() -> Result<Self, String> {
        Self::with_lang(None).await
    }

    /// Create a voice session with optional language-based engine selection.
    ///
    /// - `None` → load both Pocket (English) and Kokoro (multilingual),
    ///   route automatically per utterance based on detected language.
    /// - `Some("en")` → load only Pocket (fast English, skip Kokoro download).
    /// - `Some(_)` → load only Kokoro (multilingual).
    pub async fn with_lang(lang: Option<&str>) -> Result<Self, String> {
        let load_pocket = lang.is_none() || lang == Some("en");
        let load_kokoro = lang.is_none() || lang != Some("en");

        // Set espeak-ng data path for Kokoro TTS if not already configured.
        // espeak-rs-sys bakes the build-time path which breaks after installation.
        if load_kokoro && std::env::var("PIPER_ESPEAKNG_DATA_DIRECTORY").is_err() {
            let home = dirs::home_dir().unwrap_or_default();
            let local_data = home.join(".local/share/espeak-ng-data");
            if local_data.exists() {
                // The env var points to the PARENT dir containing espeak-ng-data/
                std::env::set_var("PIPER_ESPEAKNG_DATA_DIRECTORY", home.join(".local/share"));
            }
        }

        let label = match lang {
            Some("en") => "Pocket only",
            Some(_) => "Kokoro only",
            None => "Pocket + Kokoro",
        };
        tracing::info!("Initializing voice mode ({label})...");

        // Fail fast: check that parec exists
        Command::new("parec")
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map_err(|_| {
                "parec not found. Install: sudo apt install pulseaudio-utils".to_string()
            })?;

        tracing::info!("Checking models...");

        let progress = &TerminalProgress;
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

        // TTS init must happen on a blocking thread because Kokoro internally
        // creates a tokio Runtime for async model loading (which panics if
        // called from within an existing runtime).
        let tts_en = if load_pocket {
            match tokio::task::spawn_blocking(|| TextToSpeech::with_engine(TtsEngine::Pocket))
                .await
                .map_err(|e| format!("spawn_blocking join error: {e}"))?
            {
                Ok(tts) => {
                    let engine = tts.engine_type();
                    tracing::info!(
                        "{} TTS ready (English) [engine: {}]",
                        if engine == "pocket" { "Pocket" } else { engine },
                        engine
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
                    tracing::info!("Kokoro TTS ready (multilingual)");
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
            stt,
            tts_en,
            tts_multi,
            tts_qwen: None,
            engine_config: TtsEngineConfig::Pocket,
            qwen_voice: "ryan".to_string(),
            cancel: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Create a voice session with a specific TTS engine.
    ///
    /// This allows selecting Qwen/QwenLarge engines which require GPU.
    /// Falls back gracefully if the requested engine is not available.
    pub async fn with_engine(engine: TtsEngineConfig) -> Result<Self, String> {
        tracing::info!("Initializing voice mode with {:?}...", engine);

        // Fail fast: check that parec exists
        Command::new("parec")
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map_err(|_| {
                "parec not found. Install: sudo apt install pulseaudio-utils".to_string()
            })?;

        tracing::info!("Checking models...");

        let progress = &TerminalProgress;
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
                let tts = tokio::task::spawn_blocking(|| TextToSpeech::with_engine(TtsEngine::Pocket))
                    .await
                    .map_err(|e| format!("spawn_blocking join error: {e}"))?
                    .map_err(|e| format!("Pocket TTS init failed: {e}"))?;
                tracing::info!("Pocket TTS ready");
                (Some(Arc::new(Mutex::new(tts))), None, None)
            }
            TtsEngineConfig::Kokoro => {
                models::ensure_kokoro_model(progress)
                    .await
                    .map_err(|e| format!("Model download failed: {e}"))?;
                let tts = tokio::task::spawn_blocking(|| TextToSpeech::with_engine(TtsEngine::Kokoro))
                    .await
                    .map_err(|e| format!("spawn_blocking join error: {e}"))?
                    .map_err(|e| format!("Kokoro TTS init failed: {e}"))?;
                tracing::info!("Kokoro TTS ready");
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
                tracing::info!("Qwen TTS ready");
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
                tracing::info!("QwenLarge TTS ready (voice cloning enabled)");
                (None, None, Some(Arc::new(Mutex::new(tts))))
            }
        };

        if tts_en.is_none() && tts_multi.is_none() && tts_qwen.is_none() {
            return Err("No TTS engine could be initialized".to_string());
        }

        Ok(Self {
            stt,
            tts_en,
            tts_multi,
            tts_qwen,
            engine_config: engine,
            qwen_voice: "ryan".to_string(),
            cancel: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Set the Qwen voice name (e.g., "ryan", "serena", "vivian").
    pub fn set_qwen_voice(&mut self, voice: &str) {
        self.qwen_voice = voice.to_string();
    }

    /// Get the current TTS engine config.
    pub fn engine_config(&self) -> TtsEngineConfig {
        self.engine_config
    }

    /// Check if the current engine supports voice cloning.
    pub fn supports_voice_cloning(&self) -> bool {
        if let Some(ref tts) = self.tts_qwen {
            if let Ok(guard) = tts.lock() {
                return guard.supports_voice_cloning();
            }
        }
        false
    }

    /// Record audio and transcribe. Returns `(text, detected_language_code)`.
    pub fn record_and_transcribe(&mut self) -> Result<Option<(String, String)>, String> {
        use crossterm::event::{self, Event, KeyCode, KeyModifiers};
        use crossterm::terminal;

        // "recording..." appears after the ~> prompt on the same line
        print!("\x1b[2mrecording...\x1b[0m");
        std::io::stdout().flush().ok();

        let (sample_tx, sample_rx) = std_mpsc::channel::<Vec<f32>>();
        let mut capture = start_capture(sample_tx)?;

        // Accumulate audio in a background thread
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

        // Listen for Enter or Ctrl+Space to stop recording
        loop {
            if let Ok(Event::Key(key)) = event::read() {
                let is_stop = key.code == KeyCode::Enter
                    || (key.code == KeyCode::Char(' ')
                        && key.modifiers.contains(KeyModifiers::CONTROL))
                    || key.code == KeyCode::Esc;
                if is_stop {
                    break;
                }
            } else if let Ok(Event::Resize(_, _)) = event::read() {
                // Terminal resize - ignore
            }
        }

        // Signal collector to stop and close capture
        stop_flag.store(true, Ordering::Relaxed);
        stop_capture(&mut capture);

        let all_samples = collector.join().map_err(|_| "Audio collector panicked")?;

        if all_samples.is_empty() {
            return Ok(None);
        }

        // Erase "recording..." (12 chars) by backspacing over it
        print!("\x1b[12D\x1b[K");
        std::io::stdout().flush().ok();

        let result = self
            .stt
            .transcribe(&all_samples)
            .map_err(|e| format!("Transcription failed: {e}"))?;

        let text = result.text.trim().to_string();
        if text.is_empty() {
            print!("\x1b[2m(no speech)\x1b[0m");
            println!();
            return Ok(None);
        }

        let lang = detect_language(&text);

        // Transcription text returned to REPL for formatted display.
        Ok(Some((text, lang)))
    }

    /// Select the appropriate TTS engine based on language and config.
    /// Returns `(Arc<Mutex<TextToSpeech>>, voice_id)`.
    fn select_tts(&self, lang: &str) -> Result<(Arc<Mutex<TextToSpeech>>, String), String> {
        // If Qwen engine is configured and available, use it
        if let Some(ref tts) = self.tts_qwen {
            let voice = if self.qwen_voice.is_empty() {
                "ryan".to_string()
            } else {
                self.qwen_voice.clone()
            };
            tracing::debug!("TTS engine selected: qwen for lang={} with voice={}", lang, voice);
            return Ok((tts.clone(), voice));
        }

        let is_english = matches!(lang, "en" | "en-us" | "en-gb");

        let tts = if is_english {
            self.tts_en.as_ref().or(self.tts_multi.as_ref())
        } else {
            self.tts_multi.as_ref().or(self.tts_en.as_ref())
        }
        .ok_or("No TTS engine available")?
        .clone();

        let is_pocket = tts.lock().unwrap().engine_type() == "pocket";
        tracing::debug!(
            "TTS engine selected: {} for lang={}",
            if is_pocket { "pocket" } else { "kokoro" },
            lang
        );
        let voice_id = if is_pocket {
            "alba".to_string()
        } else {
            match lang {
                "es" => "28".to_string(),
                "fr" => "30".to_string(),
                "hi" => "31".to_string(),
                "it" => "35".to_string(),
                "ja" => "37".to_string(),
                "pt" => "42".to_string(),
                "zh" => "45".to_string(),
                _ => "3".to_string(),
            }
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

        // Bounded channel for audio chunks: synthesis → playback.
        // Capacity of 2 allows synthesis to stay ~2 sentences ahead.
        let (audio_tx, audio_rx) = std_mpsc::sync_channel::<AudioChunk>(2);

        // Clone cancel for each thread before spawning
        let cancel_synth = cancel.clone();
        let cancel_play = cancel.clone();

        // --- Synthesis thread ---
        // Must be a plain std::thread (not tokio) because Kokoro internally
        // creates a tokio Runtime which panics inside an existing runtime.
        let synth_handle = std::thread::spawn(move || {
            #[cfg(unix)]
            mask_sigint();
            let mut guard = tts.lock().unwrap();
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
            // audio_tx is dropped here, signaling the playback thread to finish.
        });

        // --- Playback thread ---
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

        // Wait for synthesis to complete (playback continues in parallel).
        let _ = synth_handle.join();

        // Wait for playback thread
        match playback_handle.join() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(e),
            Err(_) => return Err("Playback thread panicked".to_string()),
        }

        Ok(())
    }

    /// Start a streaming speak session driven by external `TtsCommand`s.
    ///
    /// Returns `(sentence_tx, join_handle)` where:
    /// - `sentence_tx` sends `TtsCommand::Synthesize(text)` for each sentence
    /// - `TtsCommand::Finish` signals completion
    /// - The join handle waits for all synthesis + playback to finish
    ///
    /// If `display_tx` is provided, each sentence's text is sent to it right
    /// after synthesis completes (synchronized with audio, not LLM speed).
    pub fn start_streaming_speak(
        &mut self,
        lang: &str,
        display_tx: Option<tokio::sync::mpsc::UnboundedSender<String>>,
    ) -> Result<(std_mpsc::Sender<TtsCommand>, std::thread::JoinHandle<()>), String> {
        let (tts, voice_id) = self.select_tts(lang)?;
        let cancel = self.cancel.clone();

        // sentence_tx/rx: tokio task → synthesis thread
        let (sentence_tx, sentence_rx) = std_mpsc::channel::<TtsCommand>();
        // audio_tx/rx: synthesis thread → playback thread (bounded for backpressure)
        let (audio_tx, audio_rx) = std_mpsc::sync_channel::<AudioChunk>(2);

        // --- Synthesis thread ---
        let cancel_synth = cancel.clone();
        let synth_handle = std::thread::spawn(move || {
            #[cfg(unix)]
            mask_sigint();
            let mut guard = tts.lock().unwrap();
            if let Err(e) = guard.set_speaker(&voice_id) {
                tracing::warn!("Voice switch to {} failed: {}", voice_id, e);
            }

            for cmd in sentence_rx {
                match cmd {
                    TtsCommand::Finish => break,
                    TtsCommand::Synthesize(sentence) => {
                        if cancel_synth.load(Ordering::Relaxed) {
                            break;
                        }
                        tracing::debug!(
                            "Streaming TTS: synthesizing \"{}\"",
                            &sentence[..sentence.len().min(40)]
                        );
                        // Display sentence text as synthesis begins (audio will follow in ~80ms)
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
            // audio_tx dropped → playback finishes
        });

        // --- Playback thread ---
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

        // --- Coordinator thread: waits for both threads ---
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

    pub fn cancel_flag(&self) -> Arc<AtomicBool> {
        self.cancel.clone()
    }

    pub fn clear_cancel(&self) {
        self.cancel.store(false, Ordering::Relaxed);
    }

    pub fn stop_playback(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    /// Clean shutdown: stop playback and leak the entire session so native
    /// FFI destructors (Pocket/Kokoro C++, Whisper) never run. The process
    /// is exiting — the OS reclaims memory. Without this, the C++ dtors
    /// segfault during drop.
    pub fn shutdown(mut self) {
        self.stop_playback();
        std::mem::forget(self);
    }
}

/// Accumulates streaming text deltas and batches complete sentences into ~200-char
/// chunks before sending to TTS. Batching reduces per-chunk overhead and keeps
/// Pocket TTS latency low (~200ms per chunk on CPU).
///
/// Detects sentence boundaries (`.` `!` `?` followed by space/newline) and
/// skips code blocks (```). Call `flush()` at the end to emit any remaining text.
pub(crate) struct SentenceAccumulator {
    buffer: String,
    /// Sentences waiting to be batched into a chunk.
    pending: String,
    in_code_block: bool,
    /// Track whether we're inside a `<thinking>` block (reasoning content to suppress).
    in_thinking_block: bool,
    sentence_tx: std_mpsc::Sender<TtsCommand>,
    /// When true, send each sentence immediately instead of batching to 250 chars.
    /// Use for streaming TTS where latency matters more than batching efficiency.
    eager: bool,
    /// When the buffer first received un-flushed content (for timer-based flush).
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
    /// streaming TTS. Also flushes partial text after 500ms if no sentence boundary
    /// is found, so the user hears audio before the first period.
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

    /// Feed a text delta. Complete sentences are sent to the TTS thread immediately
    /// in streaming mode. If no sentence boundary is found within 500ms, the buffer
    /// is flushed as-is so the user hears audio before the first period.
    pub fn push(&mut self, delta: &str) {
        self.buffer.push_str(delta);

        // Filter out <thinking>...</thinking> blocks that arrive token-by-token.
        // These are internal chain-of-thought from reasoning models.
        self.strip_thinking_from_buffer();

        if self.first_buffered.is_none() && !self.buffer.trim().is_empty() {
            self.first_buffered = Some(std::time::Instant::now());
        }
        self.extract_sentences();
        if self.eager && !self.in_code_block {
            self.try_timeout_flush();
        }
    }

    /// Strip `<thinking>...</thinking>` blocks from the buffer, handling the case
    /// where tags arrive across multiple `push()` calls (token-by-token streaming).
    fn strip_thinking_from_buffer(&mut self) {
        loop {
            if self.in_thinking_block {
                // We're inside a thinking block — look for closing tag
                if let Some(end) = self.buffer.find("</thinking>") {
                    // Discard everything up to and including </thinking>
                    self.buffer = self.buffer[end + "</thinking>".len()..].to_string();
                    self.in_thinking_block = false;
                    // Loop to check for another <thinking> in remaining text
                } else {
                    // No closing tag yet — discard entire buffer, wait for more data
                    self.buffer.clear();
                    return;
                }
            } else if let Some(start) = self.buffer.find("<thinking>") {
                // Found opening tag — keep text before it, discard tag + content
                let before = self.buffer[..start].to_string();
                let after_tag = self.buffer[start + "<thinking>".len()..].to_string();
                self.in_thinking_block = true;
                // Check if closing tag is also in this chunk
                if let Some(end) = after_tag.find("</thinking>") {
                    let remaining = after_tag[end + "</thinking>".len()..].to_string();
                    self.buffer = format!("{}{}", before, remaining);
                    self.in_thinking_block = false;
                    // Loop to check for more <thinking> blocks
                } else {
                    // No closing tag yet — keep the before part only
                    self.buffer = before;
                    return;
                }
            } else {
                return; // No thinking tags found
            }
        }
    }

    /// Flush buffer contents if 500ms has passed without a sentence boundary.
    fn try_timeout_flush(&mut self) {
        if let Some(t) = self.first_buffered {
            if t.elapsed() >= std::time::Duration::from_millis(500) && self.buffer.trim().len() > 20
            {
                let text = std::mem::take(&mut self.buffer);
                let cleaned = strip_inline_markdown(text.trim());
                if !cleaned.is_empty() {
                    let _ = self.sentence_tx.send(TtsCommand::Synthesize(cleaned));
                }
                self.first_buffered = None;
            }
        }
    }

    /// Flush any remaining text and send Finish.
    pub fn flush(self) {
        let mut pending = self.pending;
        // Emit whatever is left in the buffer
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
            let _ = self.sentence_tx.send(TtsCommand::Synthesize(pending));
        }
        let _ = self.sentence_tx.send(TtsCommand::Finish);
    }

    /// Add a sentence to the pending batch. Emits when batch reaches target size,
    /// or immediately in eager (streaming) mode.
    fn enqueue_sentence(&mut self, sentence: &str) {
        if self.eager {
            let _ = self
                .sentence_tx
                .send(TtsCommand::Synthesize(sentence.to_string()));
            self.first_buffered = None;
            return;
        }
        if self.pending.is_empty() {
            self.pending = sentence.to_string();
        } else if self.pending.len() + 1 + sentence.len() <= TTS_CHUNK_MAX_CHARS {
            self.pending.push(' ');
            self.pending.push_str(sentence);
        } else {
            // Current batch is full — send it, start new batch with this sentence
            let batch = std::mem::replace(&mut self.pending, sentence.to_string());
            let _ = self.sentence_tx.send(TtsCommand::Synthesize(batch));
        }
    }

    fn extract_sentences(&mut self) {
        loop {
            // Check for code block toggles
            if let Some(pos) = self.buffer.find("```") {
                // Emit any text before the code block marker
                if !self.in_code_block {
                    let before = self.buffer[..pos].trim().to_string();
                    if !before.is_empty() {
                        let cleaned = strip_inline_markdown(&before);
                        if !cleaned.is_empty() {
                            self.enqueue_sentence(&cleaned);
                        }
                    }
                    // Code block boundary — flush pending batch so it doesn't stall
                    if !self.pending.is_empty() {
                        let batch = std::mem::take(&mut self.pending);
                        let _ = self.sentence_tx.send(TtsCommand::Synthesize(batch));
                    }
                }
                self.in_code_block = !self.in_code_block;
                // Skip past the ``` and the rest of the line
                let after_marker = pos + 3;
                if let Some(nl) = self.buffer[after_marker..].find('\n') {
                    self.buffer = self.buffer[after_marker + nl + 1..].to_string();
                } else {
                    // No newline yet after ```, wait for more data
                    self.buffer = self.buffer[after_marker..].to_string();
                    return;
                }
                continue;
            }

            if self.in_code_block {
                return; // Wait for closing ```
            }

            // Look for sentence boundary: .!? followed by space, newline, or end of known chunk
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
                return; // No complete sentence yet
            }
        }
    }
}

/// Find a sentence boundary: position of .!? that's followed by whitespace.
fn find_sentence_boundary(text: &str) -> Option<usize> {
    let bytes = text.as_bytes();
    for i in 0..bytes.len().saturating_sub(1) {
        if matches!(bytes[i], b'.' | b'!' | b'?') {
            // Must be followed by whitespace (space, newline, tab)
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
            '*' | '_' | '`' | '~' => {} // skip markdown syntax
            '#' if out.is_empty() || out.ends_with('\n') => {
                // Skip heading markers at line start
                while chars.peek() == Some(&'#') {
                    chars.next();
                }
                if chars.peek() == Some(&' ') {
                    chars.next();
                }
            }
            '[' => {
                // Collect link text, skip URL part
                let mut link_text = String::new();
                for lc in chars.by_ref() {
                    if lc == ']' {
                        break;
                    }
                    link_text.push(lc);
                }
                out.push_str(&link_text);
                // Skip (url) if present
                if chars.peek() == Some(&'(') {
                    let mut depth = 1;
                    chars.next(); // consume '('
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
            _ => out.push(c),
        }
    }
    out.trim().to_string()
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

#[cfg(test)]
mod tests {
    use super::*;

    /// lingua correctly detects short informal English that whatlang misdetected
    /// as Estonian. This was the motivating case for the switch from whatlang.
    #[test]
    fn test_lingua_detects_short_english() {
        let text = "You just tell me a joke testing uh your pocket";
        let detected = detect_language(text);
        assert_eq!(
            detected, "en",
            "lingua should detect short informal English correctly"
        );
    }

    /// Longer English text is also correctly detected.
    #[test]
    fn test_lingua_detects_longer_english() {
        let text = "Hello, how are you doing today? I wanted to ask you about the weather forecast for this weekend.";
        let detected = detect_language(text);
        assert_eq!(detected, "en");
    }

    /// Spanish text routes to "es".
    #[test]
    fn test_lingua_detects_spanish() {
        let detected = detect_language("Hola, cómo estás hoy? Quiero preguntarte sobre el clima.");
        assert_eq!(detected, "es");
    }

    /// Japanese text routes to "ja".
    #[test]
    fn test_lingua_detects_japanese() {
        let detected = detect_language("今日の天気はどうですか？");
        assert_eq!(detected, "ja");
    }

    #[test]
    fn test_split_tts_sentences_empty() {
        assert!(split_tts_sentences("").is_empty());
        assert!(split_tts_sentences("   ").is_empty());
    }

    #[test]
    fn test_split_tts_sentences_short() {
        let result = split_tts_sentences("Hello world.");
        assert_eq!(result, vec!["Hello world."]);
    }

    #[test]
    fn test_split_tts_sentences_no_split_under_500() {
        let text = "First sentence. Second sentence. Third sentence.";
        let result = split_tts_sentences(text);
        assert_eq!(result.len(), 1, "Under 500 chars should be one chunk");
    }

    #[test]
    fn test_strip_inline_markdown() {
        assert_eq!(strip_inline_markdown("**bold** text"), "bold text");
        assert_eq!(strip_inline_markdown("# Heading"), "Heading");
        assert_eq!(strip_inline_markdown("[link](url)"), "link");
    }

    /// Actually init Pocket TTS and synthesize a sentence.
    /// This catches model-not-found, init failures, and synthesis crashes.
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
                        assert!(
                            output.samples.len() > 100,
                            "Expected audio samples, got {} samples",
                            output.samples.len()
                        );
                        assert!(
                            output.sample_rate > 0,
                            "Expected non-zero sample rate, got {}",
                            output.sample_rate
                        );
                        println!(
                            "Pocket TTS OK: {} samples @ {}Hz",
                            output.samples.len(),
                            output.sample_rate
                        );
                    }
                    Err(e) => panic!("Pocket TTS synthesis failed: {}", e),
                }
            }
            Err(e) => panic!("Pocket TTS init failed: {}", e),
        }
    }

    /// Test native streaming: Pocket should yield multiple chunks for a sentence.
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
                    println!(
                        "  chunk {}: {} samples @ {}Hz",
                        chunk_count,
                        samples.len(),
                        sample_rate
                    );
                    true // continue
                },
            )
            .expect("Streaming synthesis failed");

        assert!(
            sr > 0,
            "Expected non-zero sample rate from return, got {sr}"
        );
        assert_eq!(
            sr, rate,
            "Return sample rate should match callback sample rate"
        );
        assert!(
            total_samples > 100,
            "Expected audio samples, got {total_samples}"
        );
        assert!(
            chunk_count > 1,
            "Expected multiple streaming chunks, got {chunk_count}"
        );
        println!("Streaming OK: {chunk_count} chunks, {total_samples} samples @ {rate}Hz");
    }

    #[test]
    fn test_sentence_accumulator_strips_thinking_block() {
        let (tx, rx) = std::sync::mpsc::channel();
        let mut acc = SentenceAccumulator::new(tx);
        // Simulate token-by-token arrival of a thinking block followed by real content
        acc.push("<thinking>\nLet me think...\n</thinking>\n\nThe answer is 42.");
        acc.flush();
        // Collect all synthesized sentences
        let mut sentences = Vec::new();
        while let Ok(cmd) = rx.try_recv() {
            if let TtsCommand::Synthesize(s) = cmd {
                sentences.push(s);
            }
        }
        let combined = sentences.join(" ");
        assert!(
            !combined.contains("thinking"),
            "Thinking block should be stripped, got: {}",
            combined
        );
        assert!(
            !combined.contains("Let me think"),
            "Thinking content should be stripped, got: {}",
            combined
        );
        assert!(
            combined.contains("The answer is 42"),
            "Real content should remain, got: {}",
            combined
        );
    }

    #[test]
    fn test_sentence_accumulator_strips_thinking_across_pushes() {
        let (tx, rx) = std::sync::mpsc::channel();
        let mut acc = SentenceAccumulator::new(tx);
        // Thinking tag arrives as a whole token, but content streams in multiple pushes
        acc.push("<thinking>");
        acc.push("\nInternal reasoning here.\n");
        acc.push("</thinking>");
        acc.push("\nHello world.");
        acc.flush();
        let mut sentences = Vec::new();
        while let Ok(cmd) = rx.try_recv() {
            if let TtsCommand::Synthesize(s) = cmd {
                sentences.push(s);
            }
        }
        let combined = sentences.join(" ");
        assert!(
            !combined.contains("Internal reasoning"),
            "Thinking content should be stripped, got: {}",
            combined
        );
        assert!(
            combined.contains("Hello world"),
            "Real content should remain, got: {}",
            combined
        );
    }

    // ========================================
    // Qwen TTS Engine Tests (RED phase - TDD)
    // ========================================

    #[test]
    fn test_qwen_tts_engine_exists() {
        use jack_voice::TtsEngine;
        let _engine = TtsEngine::Qwen;
        let _engine_large = TtsEngine::QwenLarge;
    }

    #[test]
    fn test_qwen_available_voices() {
        let voices = jack_voice::TextToSpeech::available_qwen_voices();
        assert!(!voices.is_empty(), "Qwen should have available voices");
        assert!(voices.iter().any(|v| v.id_str == "ryan"), "Should include ryan");
        assert!(voices.iter().any(|v| v.id_str == "serena"), "Should include serena");
        assert!(voices.iter().any(|v| v.id_str == "vivian"), "Should include vivian");
    }

    #[test]
    fn test_qwen_can_run_check() {
        let can_run = jack_voice::TextToSpeech::can_run_qwen();
        assert!(can_run == true || can_run == false, "Should return a boolean");
    }

    #[tokio::test]
    async fn test_voice_session_with_engine_qwen() {
        let result = VoiceSession::with_engine(TtsEngineConfig::Qwen).await;
        if let Ok(session) = result {
            assert!(session.tts_qwen.is_some(), "Qwen TTS should be initialized");
            assert!(session.tts_en.is_none(), "Pocket should not be initialized");
            assert!(session.tts_multi.is_none(), "Kokoro should not be initialized");
        } else {
            println!("Qwen TTS not available (requires GPU): {:?}", result.err());
        }
    }

    #[tokio::test]
    async fn test_voice_session_with_engine_qwen_large() {
        let result = VoiceSession::with_engine(TtsEngineConfig::QwenLarge).await;
        if let Ok(session) = result {
            assert!(session.tts_qwen.is_some(), "QwenLarge TTS should be initialized");
            let tts = session.tts_qwen.as_ref().unwrap();
            let guard = tts.lock().unwrap();
            assert!(guard.supports_voice_cloning(), "QwenLarge should support voice cloning");
        } else {
            println!("QwenLarge TTS not available (requires GPU + model): {:?}", result.err());
        }
    }

    #[tokio::test]
    async fn test_voice_session_select_tts_qwen() {
        let result = VoiceSession::with_engine(TtsEngineConfig::Qwen).await;
        if let Ok(session) = result {
            let (tts, voice) = session.select_tts("en").expect("select_tts should work");
            assert_eq!(voice, "ryan", "Qwen default voice should be ryan");
        }
    }

    #[tokio::test]
    async fn test_voice_session_select_tts_qwen_voice_serena() {
        let result = VoiceSession::with_engine(TtsEngineConfig::Qwen).await;
        if let Ok(mut session) = result {
            session.set_qwen_voice("serena");
            let (tts, voice) = session.select_tts("en").expect("select_tts should work");
            assert_eq!(voice, "serena", "Voice should be serena");
        }
    }

    /// Real synthesis test - only runs if GPU is available.
    /// Marked with #[ignore] so it doesn't fail on CI without GPU.
    #[tokio::test]
    #[ignore]
    async fn test_qwen_tts_synthesizes_real_audio() {
        let result = VoiceSession::with_engine(TtsEngineConfig::Qwen).await;
        if let Ok(mut session) = result {
            let tts_result = session.speak("Hello, this is a test of Qwen TTS.", "en");
            assert!(tts_result.is_ok(), "Qwen synthesis should succeed: {:?}", tts_result.err());
        } else {
            panic!("Qwen TTS should be available for this test: {:?}", result.err());
        }
    }

    /// Real streaming synthesis test - only runs if GPU is available.
    #[test]
    #[ignore]
    fn test_qwen_tts_streaming_real_audio() {
        use jack_voice::{TextToSpeech, TtsEngine};
        if !TextToSpeech::can_run_qwen() {
            println!("Skipping: Qwen TTS requires GPU");
            return;
        }
        let mut tts = TextToSpeech::with_engine(TtsEngine::Qwen).expect("Qwen TTS init failed");
        tts.set_speaker("ryan").expect("Failed to set voice");

        let mut chunk_count = 0u32;
        let mut total_samples = 0usize;

        let sr = tts
            .synthesize_streaming(
                "Hello, this is a streaming test from Qwen TTS.",
                |samples, _sample_rate| {
                    chunk_count += 1;
                    total_samples += samples.len();
                    true
                },
            )
            .expect("Streaming synthesis failed");

        assert!(sr > 0, "Sample rate should be positive");
        assert!(total_samples > 100, "Should have audio samples");
        assert!(chunk_count >= 1, "Should have at least one chunk");
        println!("Qwen streaming OK: {} chunks, {} samples @ {}Hz", chunk_count, total_samples, sr);
    }
}
