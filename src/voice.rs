#![cfg(feature = "voice")]

use std::io::Write;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc as std_mpsc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use jack_voice::{
    SpeechToText, TextToSpeech, SttMode,
    models::{self, ModelProgressCallback},
};

fn split_tts_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut start = 0;
    let bytes = text.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        if b == b'.' || b == b'!' || b == b'?' {
            // Include the punctuation, skip trailing whitespace for the boundary
            let end = i + 1;
            let s = text[start..end].trim().to_string();
            if !s.is_empty() {
                sentences.push(s);
            }
            start = end;
        }
    }
    // Remainder (text after last punctuation)
    let remainder = text[start..].trim().to_string();
    if !remainder.is_empty() {
        sentences.push(remainder);
    }
    sentences
}

/// Apply fade-in and fade-out envelopes to eliminate clicks at sentence boundaries.
/// `fade_samples` is the number of samples to ramp over (~5ms at 44.1kHz = 220 samples).
fn apply_fade_envelope(samples: &mut [f32], fade_samples: usize) {
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

fn samples_to_s16le_stereo(samples: &[f32]) -> Vec<u8> {
    samples.iter().flat_map(|&s| {
        let clamped = s.clamp(-1.0, 1.0);
        let i16_val = (clamped * 32767.0) as i16;
        let bytes = i16_val.to_le_bytes();
        // Duplicate mono → stereo (left = right)
        [bytes[0], bytes[1], bytes[0], bytes[1]]
    }).collect()
}

pub struct VoiceSession {
    stt: SpeechToText,
    tts: TextToSpeech,
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
    let pcm = samples_to_s16le_stereo(&samples);

    let mut child = Command::new("paplay")
        .args([
            "--raw",
            "--format=s16le",
            "--channels=2",
            &format!("--rate={}", sample_rate),
        ])
        .env("PULSE_SERVER", pulse_server())
        .stdin(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|e| format!("paplay failed: {e}\n  Install: sudo apt install pulseaudio-utils"))?;

    let mut stdin = child.stdin.take().unwrap();
    std::thread::spawn(move || {
        let _ = stdin.write_all(&pcm);
    });

    Ok(child)
}

impl VoiceSession {
    pub async fn new() -> Result<Self, String> {
        println!("Initializing voice mode...");

        // Fail fast: check that parec exists
        Command::new("parec")
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map_err(|_| "parec not found. Install: sudo apt install pulseaudio-utils".to_string())?;

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
        models::ensure_supertonic_models(progress)
            .await
            .map_err(|e| format!("Model download failed: {e}"))?;

        let stt = SpeechToText::new(SttMode::Batch)
            .map_err(|e| format!("STT init failed: {e}"))?;
        let tts = TextToSpeech::new()
            .map_err(|e| format!("TTS init failed: {e}"))?;

        Ok(Self {
            stt,
            tts,
            playback: None,
            cancel: Arc::new(AtomicBool::new(false)),
        })
    }

    pub fn record_and_transcribe(&mut self) -> Result<Option<String>, String> {
        use crossterm::event::{self, Event, KeyCode, KeyModifiers};
        use crossterm::terminal;

        println!("\x1b[33m\u{1f3a4} Recording... (press Enter or Ctrl+Space to stop)\x1b[0m");

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

        println!("\x1b[33mTranscribing...\x1b[0m");

        let result = self
            .stt
            .transcribe(&all_samples)
            .map_err(|e| format!("Transcription failed: {e}"))?;

        let text = result.text.trim().to_string();
        if text.is_empty() {
            return Ok(None);
        }

        println!("\x1b[36mYou said: \"{}\"\x1b[0m", text);
        Ok(Some(text))
    }

    pub fn speak(&mut self, text: &str) -> Result<(), String> {
        let sentences = split_tts_sentences(text);
        if sentences.is_empty() {
            return Ok(());
        }

        // Synthesize first sentence to get sample_rate, then spawn paplay
        let mut first = self.tts.synthesize(&sentences[0])
            .map_err(|e| format!("TTS failed: {e}"))?;
        let sample_rate = first.sample_rate;
        // ~5ms fade at 44.1kHz
        let fade_samples = (sample_rate as usize * 5) / 1000;
        apply_fade_envelope(&mut first.samples, fade_samples);

        let mut child = Command::new("paplay")
            .args([
                "--raw",
                "--format=s16le",
                "--channels=2",
                &format!("--rate={}", sample_rate),
            ])
            .env("PULSE_SERVER", pulse_server())
            .stdin(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| format!("paplay failed: {e}\n  Install: sudo apt install pulseaudio-utils"))?;

        let mut stdin = child.stdin.take().unwrap();

        // Write first sentence immediately
        let pcm = samples_to_s16le_stereo(&first.samples);
        stdin.write_all(&pcm).map_err(|e| format!("Write to paplay failed: {e}"))?;

        // Stream remaining sentences, checking cancel between each
        for sentence in &sentences[1..] {
            if self.cancel.load(Ordering::Relaxed) {
                break;
            }
            let mut output = self.tts.synthesize(sentence)
                .map_err(|e| format!("TTS failed: {e}"))?;
            if self.cancel.load(Ordering::Relaxed) {
                break;
            }
            apply_fade_envelope(&mut output.samples, fade_samples);
            let pcm = samples_to_s16le_stereo(&output.samples);
            stdin.write_all(&pcm).map_err(|e| format!("Write to paplay failed: {e}"))?;
        }

        // Drop stdin to signal EOF — paplay plays remaining buffer then exits
        drop(stdin);
        self.playback = Some(child);
        Ok(())
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
