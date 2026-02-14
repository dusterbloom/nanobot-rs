#![cfg(feature = "voice")]

use std::io::Write;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc as std_mpsc;
use std::sync::{Arc, Mutex};

use jack_voice::{
    models::{self, ModelProgressCallback},
    SpeechToText, SttMode, TextToSpeech, TtsEngine,
};
use whatlang;

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
    let trimmed = text.trim();
    if trimmed.len() <= 500 {
        return if trimmed.is_empty() { vec![] } else { vec![trimmed.to_string()] };
    }

    let sentences = split_sentences(text);
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

/// A command sent to the synthesis thread.
pub(crate) enum TtsCommand {
    Synthesize(String),
    Finish,
}

/// A chunk of synthesized audio ready for playback.
struct AudioChunk {
    data: Vec<u8>,    // f32le raw bytes
    sample_rate: u32,
}

pub struct VoiceSession {
    stt: SpeechToText,
    /// Pocket TTS engine (fast, English-only, CPU inference).
    tts_en: Option<Arc<Mutex<TextToSpeech>>>,
    /// Kokoro TTS engine (multilingual).
    tts_multi: Option<Arc<Mutex<TextToSpeech>>>,
    playback: Option<Child>,
    cancel: Arc<AtomicBool>,
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
        println!("Initializing voice mode ({label})... [build:v8]");

        // Fail fast: check that parec exists
        Command::new("parec")
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map_err(|_| {
                "parec not found. Install: sudo apt install pulseaudio-utils".to_string()
            })?;

        println!("Checking models...");

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
            match tokio::task::spawn_blocking(|| {
                TextToSpeech::with_engine(TtsEngine::Pocket)
            })
            .await
            .map_err(|e| format!("spawn_blocking join error: {e}"))?
            {
                Ok(tts) => {
                    let engine = tts.engine_type();
                    println!("  {} TTS ready (English) [engine: {}]",
                        if engine == "pocket" { "Pocket" } else { engine }, engine);
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
            match tokio::task::spawn_blocking(|| {
                TextToSpeech::with_engine(TtsEngine::Kokoro)
            })
            .await
            .map_err(|e| format!("spawn_blocking join error: {e}"))?
            {
                Ok(tts) => {
                    println!("  Kokoro TTS ready (multilingual)");
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
            playback: None,
            cancel: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Record audio and transcribe. Returns `(text, detected_language_code)`.
    pub fn record_and_transcribe(&mut self) -> Result<Option<(String, String)>, String> {
        use crossterm::event::{self, Event, KeyCode, KeyModifiers};
        use crossterm::terminal;

        // "recording..." appears after the ~> prompt on the same line
        print!("\x1b[2mrecording...\x1b[0m");
        std::io::stdout().flush().ok();

        let (sample_tx, sample_rx) = std_mpsc::channel::<Vec<f32>>();
        let mut parec = start_parec(sample_tx)?;

        // Accumulate audio in a background thread
        let stop_flag = Arc::new(AtomicBool::new(false));
        let stop_clone = stop_flag.clone();

        let collector = std::thread::spawn(move || {
            let mut all_samples: Vec<f32> = Vec::new();
            while !stop_clone.load(Ordering::Relaxed) {
                match sample_rx.recv_timeout(std::time::Duration::from_millis(50)) {
                    Ok(samples) => all_samples.extend_from_slice(&samples),
                    Err(std_mpsc::RecvTimeoutError::Timeout) => continue,
                    Err(std_mpsc::RecvTimeoutError::Disconnected) => break,
                }
            }
            // Drain any remaining samples in the channel
            while let Ok(samples) = sample_rx.try_recv() {
                all_samples.extend_from_slice(&samples);
            }
            all_samples
        });

        // Wait for stop keypress
        tokio::task::block_in_place(|| {
            if terminal::enable_raw_mode().is_ok() {
                loop {
                    if let Ok(Event::Key(key)) = event::read() {
                        let is_stop = key.code == KeyCode::Enter
                            || (key.code == KeyCode::Char(' ')
                                && key.modifiers.contains(KeyModifiers::CONTROL))
                            || (key.code == KeyCode::Char('c')
                                && key.modifiers.contains(KeyModifiers::CONTROL));
                        if is_stop {
                            break;
                        }
                    }
                }
                terminal::disable_raw_mode().ok();
            } else {
                // Fallback: wait for Enter
                let mut line = String::new();
                let _ = std::io::stdin().read_line(&mut line);
            }
        });

        // Signal collector to stop and kill parec
        stop_flag.store(true, Ordering::Relaxed);
        let _ = parec.kill();
        let _ = parec.wait();

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

        let lang = whatlang::detect(&text)
            .map(|info| {
                match info.lang().code() {
                    "eng" => "en",
                    "spa" => "es",
                    "fra" => "fr",
                    "hin" => "hi",
                    "ita" => "it",
                    "jpn" => "ja",
                    "por" => "pt",
                    "cmn" | "zho" => "zh",
                    other => other,
                }
                .to_string()
            })
            .unwrap_or_else(|| "en".to_string());

        // Transcription text returned to REPL for formatted display.
        Ok(Some((text, lang)))
    }

    /// Select the appropriate TTS engine based on language.
    /// Returns `(Arc<Mutex<TextToSpeech>>, voice_id)`.
    fn select_tts(&self, lang: &str) -> Result<(Arc<Mutex<TextToSpeech>>, &'static str), String> {
        let is_english = matches!(lang, "en" | "en-us" | "en-gb");

        let tts = if is_english {
            self.tts_en.as_ref().or(self.tts_multi.as_ref())
        } else {
            self.tts_multi.as_ref().or(self.tts_en.as_ref())
        }
        .ok_or("No TTS engine available")?
        .clone();

        let is_pocket = tts.lock().unwrap().engine_type() == "pocket";
        tracing::debug!("TTS engine selected: {} for lang={}", if is_pocket { "pocket" } else { "kokoro" }, lang);
        let voice_id = if is_pocket {
            "alba"
        } else {
            match lang {
                "es" => "28",
                "fr" => "30",
                "hi" => "31",
                "it" => "35",
                "ja" => "37",
                "pt" => "42",
                "zh" => "45",
                _ => "3",
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

        // --- Synthesis thread ---
        // Must be a plain std::thread (not tokio) because Kokoro internally
        // creates a tokio Runtime which panics inside an existing runtime.
        let synth_handle = std::thread::spawn(move || {
            let mut guard = tts.lock().unwrap();
            if let Err(e) = guard.set_speaker(voice_id) {
                tracing::warn!("Voice switch to {} failed: {}", voice_id, e);
            }

            for (i, sentence) in sentences.iter().enumerate() {
                if cancel.load(Ordering::Relaxed) {
                    break;
                }
                tracing::debug!("Synthesizing sentence {}/{}...", i + 1, sentences.len());
                match guard.synthesize(sentence) {
                    Ok(output) => {
                        if cancel.load(Ordering::Relaxed) {
                            break;
                        }
                        let chunk = AudioChunk {
                            data: samples_to_f32le_bytes(&output.samples),
                            sample_rate: output.sample_rate,
                        };
                        if audio_tx.send(chunk).is_err() {
                            break; // playback thread gone
                        }
                    }
                    Err(e) => {
                        tracing::error!("TTS synthesis failed: {}", e);
                        break;
                    }
                }
            }
            // audio_tx is dropped here, signaling the playback thread to finish.
        });

        // --- Playback thread ---
        // Spawns a single paplay process and writes audio chunks as they arrive.
        let playback_handle = std::thread::spawn(move || -> Result<Option<Child>, String> {
            // Wait for the first chunk to get the sample rate.
            let first_chunk = match audio_rx.recv() {
                Ok(c) => c,
                Err(_) => return Ok(None), // no audio produced
            };

            let mut child = Command::new("paplay")
                .args([
                    "--raw",
                    "--format=float32le",
                    "--channels=1",
                    &format!("--rate={}", first_chunk.sample_rate),
                ])
                .env("PULSE_SERVER", pulse_server())
                .stdin(Stdio::piped())
                .stderr(Stdio::inherit())
                .spawn()
                .map_err(|e| {
                    format!("paplay failed: {e}\n  Install: sudo apt install pulseaudio-utils")
                })?;

            let mut stdin = child.stdin.take().unwrap();

            // Write first chunk
            if stdin.write_all(&first_chunk.data).is_err() {
                return Ok(Some(child));
            }

            // Write remaining chunks as they arrive
            for chunk in audio_rx {
                if stdin.write_all(&chunk.data).is_err() {
                    break;
                }
            }

            drop(stdin); // close stdin so paplay finishes
            Ok(Some(child))
        });

        // Wait for synthesis to complete (playback continues in parallel).
        let _ = synth_handle.join();

        // Wait for playback thread and store the child process for stop_playback().
        match playback_handle.join() {
            Ok(Ok(Some(child))) => {
                self.playback = Some(child);
            }
            Ok(Ok(None)) => {} // no audio
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
            let mut guard = tts.lock().unwrap();
            if let Err(e) = guard.set_speaker(voice_id) {
                tracing::warn!("Voice switch to {} failed: {}", voice_id, e);
            }

            for cmd in sentence_rx {
                match cmd {
                    TtsCommand::Finish => break,
                    TtsCommand::Synthesize(sentence) => {
                        if cancel_synth.load(Ordering::Relaxed) {
                            break;
                        }
                        tracing::debug!("Streaming TTS: synthesizing \"{}\"", &sentence[..sentence.len().min(40)]);
                        match guard.synthesize(&sentence) {
                            Ok(output) => {
                                if cancel_synth.load(Ordering::Relaxed) {
                                    break;
                                }
                                // Display sentence text synchronized with audio readiness
                                if let Some(ref dtx) = display_tx {
                                    let _ = dtx.send(sentence.clone());
                                }
                                let chunk = AudioChunk {
                                    data: samples_to_f32le_bytes(&output.samples),
                                    sample_rate: output.sample_rate,
                                };
                                if audio_tx.send(chunk).is_err() {
                                    break;
                                }
                            }
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
        let cancel_play = cancel;
        let playback_handle = std::thread::spawn(move || -> Option<Child> {
            let first_chunk = match audio_rx.recv() {
                Ok(c) => c,
                Err(_) => return None,
            };

            let mut child = match Command::new("paplay")
                .args([
                    "--raw",
                    "--format=float32le",
                    "--channels=1",
                    &format!("--rate={}", first_chunk.sample_rate),
                ])
                .env("PULSE_SERVER", pulse_server())
                .stdin(Stdio::piped())
                .stderr(Stdio::inherit())
                .spawn()
            {
                Ok(c) => c,
                Err(e) => {
                    tracing::error!("paplay failed: {}", e);
                    return None;
                }
            };

            let mut stdin = child.stdin.take().unwrap();

            if stdin.write_all(&first_chunk.data).is_err() {
                return Some(child);
            }

            for chunk in audio_rx {
                if cancel_play.load(Ordering::Relaxed) {
                    break;
                }
                if stdin.write_all(&chunk.data).is_err() {
                    break;
                }
            }

            drop(stdin);
            Some(child)
        });

        // --- Coordinator thread: waits for both threads, stores playback child ---
        // We need a mutable reference to self.playback, but we can't move it into a thread.
        // Instead, return the join handle and let the caller wait.
        let join_handle = std::thread::spawn(move || {
            let _ = synth_handle.join();
            let child = playback_handle.join().ok().flatten();
            // We can't store child into self.playback from a thread, so we just
            // wait for playback to finish here.
            if let Some(mut c) = child {
                let _ = c.wait();
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
        if let Some(ref mut child) = self.playback {
            let _ = child.kill();
            let _ = child.wait();
        }
        self.playback = None;
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
    sentence_tx: std_mpsc::Sender<TtsCommand>,
}

impl SentenceAccumulator {
    pub fn new(sentence_tx: std_mpsc::Sender<TtsCommand>) -> Self {
        Self {
            buffer: String::new(),
            pending: String::new(),
            in_code_block: false,
            sentence_tx,
        }
    }

    /// Feed a text delta. Batched chunks are sent to the TTS thread when
    /// accumulated sentences reach ~200 chars.
    pub fn push(&mut self, delta: &str) {
        self.buffer.push_str(delta);
        self.extract_sentences();
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

    /// Add a sentence to the pending batch. Emits when batch reaches target size.
    fn enqueue_sentence(&mut self, sentence: &str) {
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

    /// whatlang is unreliable for short/informal speech — it misdetects English
    /// as Estonian, etc. This test documents the problem so we don't rely on
    /// per-utterance detection for TTS routing. The REPL uses the session-level
    /// language setting (from config/CLI) instead.
    #[test]
    fn test_whatlang_misdetects_short_english() {
        // This phrase gets detected as Estonian ("est"), not English.
        // This is why we use session-level lang for TTS routing, not per-utterance.
        let text = "You just tell me a joke testing uh your pocket";
        let detected = whatlang::detect(text).map(|i| i.lang().code().to_string());
        assert_ne!(detected.as_deref(), Some("eng"),
            "If whatlang starts detecting this correctly, the session-level override is still correct but not strictly necessary");
    }

    /// Longer English text IS correctly detected.
    #[test]
    fn test_whatlang_detects_longer_english() {
        let text = "Hello, how are you doing today? I wanted to ask you about the weather forecast for this weekend.";
        let detected = whatlang::detect(text).map(|info| {
            match info.lang().code() {
                "eng" => "en",
                other => other,
            }.to_string()
        }).unwrap_or_else(|| "en".to_string());
        assert_eq!(detected, "en");
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
                        assert!(output.samples.len() > 100,
                            "Expected audio samples, got {} samples", output.samples.len());
                        assert!(output.sample_rate > 0,
                            "Expected non-zero sample rate, got {}", output.sample_rate);
                        println!("Pocket TTS OK: {} samples @ {}Hz",
                            output.samples.len(), output.sample_rate);
                    }
                    Err(e) => panic!("Pocket TTS synthesis failed: {}", e),
                }
            }
            Err(e) => panic!("Pocket TTS init failed: {}", e),
        }
    }
}
