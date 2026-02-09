#![cfg(feature = "voice")]

use std::process::{Child, Command, Stdio};
use std::sync::mpsc as std_mpsc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use jack_voice::{
    SpeechToText, TextToSpeech, SttMode,
    models::{self, ModelProgressCallback},
};

pub struct VoiceSession {
    stt: SpeechToText,
    tts: TextToSpeech,
    playback: Option<Child>,
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
    // Convert float32 mono → s16le stereo to match the PulseAudio sink exactly
    // (RDPSink: s16le 2ch 44100Hz). This avoids any PulseAudio format conversion.
    let pcm: Vec<u8> = samples.iter().flat_map(|&s| {
        let clamped = s.clamp(-1.0, 1.0);
        let i16_val = (clamped * 32767.0) as i16;
        let bytes = i16_val.to_le_bytes();
        // Duplicate mono → stereo (left = right)
        [bytes[0], bytes[1], bytes[0], bytes[1]]
    }).collect();

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
        use std::io::Write;
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
        let output = self
            .tts
            .synthesize(text)
            .map_err(|e| format!("TTS failed: {e}"))?;
        self.playback = Some(start_playback(output.samples, output.sample_rate)?);
        Ok(())
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
