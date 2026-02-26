// Audio input diagnostic test for macOS
// Run with: cargo test --features voice audio_diagnostic -- --nocapture

#![cfg(feature = "voice")]

use std::sync::mpsc::channel;
use std::time::{Duration, Instant};

#[test]
fn audio_diagnostic() {
    use jack_voice::AudioCapture;

    println!("\n=== Audio Input Diagnostic ===\n");
    println!("Recording 3 seconds of audio from default input device...");
    println!("Speak at NORMAL conversational volume.\n");

    let (tx, rx) = channel::<Vec<f32>>();
    let capture = match AudioCapture::start(tx) {
        Ok(c) => c,
        Err(e) => {
            println!("ERROR: Failed to start audio capture: {e}");
            println!("  - Check System Settings > Privacy & Security > Microphone");
            return;
        }
    };

    let start = Instant::now();
    let mut all_samples = Vec::new();
    let mut chunk_count = 0;

    // Collect 3 seconds of audio
    while start.elapsed() < Duration::from_secs(3) {
        if let Ok(samples) = rx.recv_timeout(Duration::from_millis(100)) {
            chunk_count += 1;
            all_samples.extend(samples);
        }
    }

    capture.stop();

    if all_samples.is_empty() {
        println!("ERROR: No audio samples received!");
        return;
    }

    // Calculate energy metrics (same as STT does)
    let max_val = all_samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    let rms = (all_samples.iter().map(|s| s * s).sum::<f32>() / all_samples.len() as f32).sqrt();

    // STT thresholds
    const STT_MIN_RMS: f32 = 0.01;
    const STT_MIN_AMP: f32 = 0.03;

    println!("Collected {} samples in {} chunks", all_samples.len(), chunk_count);
    println!("Device sample rate: {}Hz", capture.device_sample_rate);
    println!();
    println!("=== Energy Metrics ===");
    println!("  Peak amplitude:  {:.6}  (threshold: {:.3})", max_val, STT_MIN_AMP);
    println!("  RMS energy:      {:.6}  (threshold: {:.3})", rms, STT_MIN_RMS);
    println!();

    let would_reject = rms < STT_MIN_RMS || max_val < STT_MIN_AMP;
    if would_reject {
        println!("⚠️  WARNING: This audio would be REJECTED by STT!");
        println!();
        if rms < STT_MIN_RMS {
            println!("  - RMS ({:.6}) is below threshold ({:.3})", rms, STT_MIN_RMS);
            println!("    → Need {:.1}x more volume to pass RMS check", STT_MIN_RMS / rms);
        }
        if max_val < STT_MIN_AMP {
            println!("  - Amplitude ({:.6}) is below threshold ({:.3})", max_val, STT_MIN_AMP);
            println!("    → Need {:.1}x more volume to pass amplitude check", STT_MIN_AMP / max_val);
        }
        println!();
        println!("SOLUTIONS:");
        println!("  1. Increase system input volume in Sound settings");
        println!("  2. Use headphones with better mic");
        println!("  3. Move closer to the microphone");
        println!("  4. Lower STT thresholds in jack-voice/src/stt.rs");
    } else {
        println!("✓ Audio levels are GOOD - STT should work!");
    }
    println!();

    // Print histogram of amplitudes
    println!("=== Amplitude Distribution ===");
    let mut buckets = [0u64; 10];
    for s in &all_samples {
        let idx = ((s.abs() * 10.0) as usize).min(9);
        buckets[idx] += 1;
    }
    for (i, count) in buckets.iter().enumerate() {
        let range_start = i as f32 / 10.0;
        let range_end = (i + 1) as f32 / 10.0;
        let bar_len = (*count as f64 / all_samples.len() as f64 * 50.0) as usize;
        println!(
            "  {:.1}-{:.1}: {} {}",
            range_start,
            range_end,
            "#".repeat(bar_len),
            count
        );
    }
}
