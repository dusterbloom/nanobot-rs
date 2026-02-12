# Research Report: Supertonic/Kokoro TTS Engines - Audio Quality and Configuration

Generated: 2026-02-11

## Executive Summary

Supertonic is a fast, on-device TTS engine by Supertone Inc. that uses flow-matching diffusion with ONNX Runtime, outputting 44.1kHz audio. Kokoro is a separate 82M-parameter TTS model with multiple Rust implementations (kokoro-tiny, Kokoros/kokorox), outputting 24kHz audio. Both are wrapped locally via the `jack-voice` crate with a unified `TextToSpeech` API. The nanobot project already implements fade envelopes (5ms) and peak normalization to eliminate clicks at sentence boundaries. Neither engine natively supports true streaming/incremental audio generation -- both synthesize complete utterances then return samples.

## Research Question

Search for Supertonic and Kokoro TTS engines: GitHub repositories, audio quality best practices (click/pop elimination), configuration options, known issues with audio artifacts at sentence boundaries, and streaming/continuous audio output capabilities.

## Key Findings

### Finding 1: Supertonic TTS Engine

**Repository:** [supertone-inc/supertonic](https://github.com/supertone-inc/supertonic)

Supertonic is a lightning-fast, on-device TTS system with:
- **Architecture:** Three-component pipeline: speech autoencoder (continuous latent representation), text-to-latent module (flow-matching), and utterance-level duration predictor. Uses ConvNeXt blocks for lightweight processing.
- **Model size:** 44M-66M parameters
- **Performance:** Up to 167x faster than real-time on M4 Pro; 912-1,263 characters/second on consumer hardware
- **Languages:** English, Korean, Spanish, Portuguese, French (Supertonic 2)
- **Output format:** 16-bit WAV, **sample rate: 44,100 Hz** (confirmed from local crate source: `pub const SAMPLE_RATE: u32 = 44100`)
- **License:** MIT

**Local implementation** (`/home/peppi/Dev/jack-voice/supertonic/src/lib.rs`):
- Four ONNX model pipeline: `duration_predictor.onnx`, `text_encoder.onnx`, `vector_estimator.onnx`, `vocoder.onnx`
- Configurable inference steps (`num_inference_steps`, default 5) -- higher = better quality, slower
- Speed control (`speed`, default 1.05)
- Voice styles: F1, F2, M1, M2 (loaded from JSON files)
- **Automatic text chunking** at ~300 characters with 0.3s silence between chunks
- **Mid-sentence gap detection and retry** -- the flow-matching denoiser starts from random noise, so occasionally drops words creating silent gaps. The engine detects these (via sliding window energy analysis) and re-synthesizes up to 2 times.
- **CUDA support** with automatic fallback to CPU
- **Vocoder output truncation** to predicted duration to avoid "doubling" artifacts from padding noise

**Rust API (local crate `supertonic`):**
```rust
use supertonic::{TextToSpeech, VoiceStyleData, SAMPLE_RATE};

let mut tts = TextToSpeech::new("path/to/models")?;
let voice = VoiceStyleData::from_json_file("F1.json", "F1", "Female 1")?;
tts.set_voice_style(&voice);
tts.set_speed(1.0);
tts.set_inference_steps(5);
let audio = tts.synthesize("Hello world")?;
// audio.samples: Vec<f32>, audio.sample_rate: 44100
```

- Source: Local crate at `/home/peppi/Dev/jack-voice/supertonic/src/lib.rs`

### Finding 2: Kokoro TTS Engine

**Main Rust implementations:**
- [lucasjinreal/Kokoros](https://github.com/lucasjinreal/Kokoros) -- 690+ stars, streaming support, OpenAI-compatible API
- [WismutHansen/kokorox](https://github.com/WismutHansen/kokorox) -- another Rust implementation
- [thewh1teagle/kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) -- ONNX-based
- [kokoro-tts on crates.io](https://crates.io/crates/kokoro-tts) -- 9,294 downloads

**Local implementation** (`/home/peppi/Dev/jack-voice/jack-voice/src/kokoro_tts.rs`):
- Uses `kokoro-tiny` crate (version 0.1.0) as the base engine
- **Sample rate: 24,000 Hz** (fixed)
- **53 voices** across 10 languages: American English (0-10), British English (20-27), Spanish (28-29), French (30), Hindi (31-34), Italian (35-36), Japanese (37-41), Portuguese (42-44), Mandarin Chinese (45-52)
- **Dual pipeline:** All voices use a "DirectPipeline" that runs espeak-ng phonemization then ONNX inference
- Italian has a dedicated rule-based G2P; all others use espeak-ng with `--ipa`
- **Text chunking:** Splits at ~300 chars, merges short sentences, with fallback splitting at comma/semicolon
- **Token padding:** Wraps tokens with pad token (0) at start and end for proper sequence boundaries
- Supports int8 quantized model (~88MB, 2-3x faster on CPU) and f32 model (~310MB)
- **CUDA support** via feature flag
- Model files stored in `~/.cache/kokoros/`

- Source: Local file at `/home/peppi/Dev/jack-voice/jack-voice/src/kokoro_tts.rs`

### Finding 3: Audio Quality -- Click/Pop Prevention at Sentence Boundaries

The nanobot project already implements two techniques in `/home/peppi/Dev/nanobot/src/voice.rs`:

**1. Peak normalization (`normalize_peak`):**
```rust
pub(crate) fn normalize_peak(samples: &mut [f32], target_peak: f32) {
    let peak = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if peak > 1e-6 {
        let gain = target_peak / peak;
        for s in samples.iter_mut() { *s *= gain; }
    }
}
```
Normalizes each sentence's audio to 0.85 peak, ensuring consistent volume between sentences.

**2. Fade envelope (`apply_fade_envelope`):**
```rust
pub(crate) fn apply_fade_envelope(samples: &mut [f32], fade_samples: usize) {
    let fade = fade_samples.min(len / 2);
    // Fade in: linear ramp 0.0 -> 1.0
    for i in 0..fade { samples[i] *= i as f32 / fade as f32; }
    // Fade out: linear ramp 1.0 -> 0.0
    for i in 0..fade { samples[len - 1 - i] *= i as f32 / fade as f32; }
}
```
Applied with `fade_samples = (sample_rate * 5) / 1000` = ~5ms fade, which at 24kHz = 120 samples, at 44.1kHz = 220 samples.

**These are applied in `voice_pipeline.rs` (line 202-204) and `voice.rs` (lines 434-436):**
```rust
let fade_samples = (output.sample_rate as usize * 5) / 1000;
normalize_peak(&mut output.samples, 0.85);
apply_fade_envelope(&mut output.samples, fade_samples);
```

**Note:** In `voice.rs` `start_streaming_speak()` (line 559), the streaming path applies `normalize_peak` but does NOT apply `apply_fade_envelope`. This is a discrepancy -- the streaming path may still produce clicks.

- Source: `/home/peppi/Dev/nanobot/src/voice.rs` lines 40-63, 197-206, 551-565

### Finding 4: Streaming / Continuous Audio Output

**Neither engine supports true incremental/streaming synthesis** -- both synthesize complete text chunks and return all samples at once.

**Supertonic:** The `synthesize()` method runs the full 4-stage pipeline (duration prediction -> text encoding -> iterative denoising -> vocoding) for the entire input. There is no callback or chunked output mechanism. Issue #57 on GitHub asks about real-time playback from memory (vs. saving to WAV), but no streaming API exists.

**Kokoro (via kokoro-tiny):** Similarly synthesizes complete utterances. The Kokoros project (lucasjinreal) advertises "streaming" but this means streaming text input line-by-line, not incremental audio generation within a single utterance.

**jack-voice wrapper** (`/home/peppi/Dev/jack-voice/jack-voice/src/tts.rs`):
```rust
pub fn synthesize_streaming<F>(&mut self, text: &str, mut on_chunk: F) -> Result<u32, TtsError>
where F: FnMut(&[f32], u32) -> bool,
{
    // NOT true streaming -- just calls synthesize() and invokes on_chunk once
    let audio = self.synthesize(text)?;
    let _ = on_chunk(&audio.samples, audio.sample_rate);
    Ok(audio.sample_rate)
}
```

**nanobot's workaround:** The project achieves streaming-like behavior at the sentence level:
1. `SentenceAccumulator` in `voice.rs` detects sentence boundaries in streaming LLM output
2. Each complete sentence is sent via `TtsCommand::Synthesize(text)` to a synthesis thread
3. Synthesis thread processes sentences sequentially, sending audio chunks to playback thread via bounded channel (capacity 2)
4. Playback thread writes raw audio to a single `paplay` process via stdin pipe
5. This gives ~1 sentence of lookahead while maintaining continuous audio output

- Source: `/home/peppi/Dev/jack-voice/jack-voice/src/tts.rs` lines 227-239; `/home/peppi/Dev/nanobot/src/voice.rs` lines 415-577

### Finding 5: Configuration Options Summary

| Parameter | Supertonic | Kokoro |
|-----------|-----------|--------|
| Sample rate | 44,100 Hz (fixed) | 24,000 Hz (fixed) |
| Voice styles | F1, F2, M1, M2 (from JSON) | 53 voices (0-52, numeric IDs) |
| Speed | 0.25-4.0 (default 1.05) | 0.25-4.0 (default 1.0) |
| Inference steps | 1-20 (default 5) | N/A (single pass) |
| Languages | English only | 10 languages |
| Max chunk chars | 300 | 300 |
| Inter-chunk silence | 0.3s | None (chunks concatenated directly) |
| GPU support | CUDA (auto-fallback to CPU) | CUDA via feature flag |
| Model size | ~4 ONNX files | ~88MB (int8) or ~310MB (f32) |
| Gap detection | Yes (with retry) | No |

### Finding 6: Known Issues and Artifacts

**Supertonic:**
- **Mid-sentence word dropping:** The flow-matching denoiser occasionally drops words, creating silent gaps. Mitigated by detection + retry (up to 2 attempts). This is inherent to the stochastic denoising process.
- **Vocoder padding noise:** The vocoder produces more samples than needed; without truncation to predicted duration, "doubling" artifacts occur. Already handled in the local implementation.
- **ONNX Runtime mutex cleanup warning** on macOS (cosmetic, uses `libc::_exit()` workaround).
- **Pronunciation issues:** GitHub issues report problems with Spanish, French numerals, single letters, and mixed-language text.
- **No pause control:** Issue #71 asks for custom pause duration between segments -- currently not supported.
- **GPU mode untested.**

**Kokoro:**
- **Token limit:** 510 tokens max per inference call. Exceeding this causes errors or degraded output.
- **espeak-ng dependency:** Non-English phonemization requires `espeak-ng` installed (`sudo apt install espeak-ng`).
- **opus dependency:** Requires `libopus-dev` on Linux.
- **No crossfade between chunks:** The DirectPipeline in `kokoro_tts.rs` simply concatenates chunk audio (`audio.extend_from_slice(&chunk_audio)`) without any crossfading or silence insertion. This can cause audible discontinuities between chunks.

**Both engines at the sentence concatenation level (in nanobot):**
- The streaming speak path (`start_streaming_speak` in `voice.rs`) applies `normalize_peak` but skips `apply_fade_envelope`, meaning clicks may still occur in the streaming/CLI path.
- The channel voice pipeline path (`voice_pipeline.rs`) correctly applies both normalize and fade.

## Sources

- [supertone-inc/supertonic GitHub](https://github.com/supertone-inc/supertonic) -- Main Supertonic repository
- [supertonic/rust](https://github.com/supertone-inc/supertonic/tree/main/rust) -- Rust implementation examples
- [lucasjinreal/Kokoros](https://github.com/lucasjinreal/Kokoros) -- Primary Kokoro Rust implementation
- [kokoro-tts crate](https://crates.io/crates/kokoro-tts) -- Published Kokoro TTS crate on crates.io
- [SupertonicTTS paper (arXiv:2503.23108)](https://arxiv.org/abs/2503.23108) -- Academic paper on architecture
- [Supertonic GitHub Issue #57](https://github.com/supertone-inc/supertonic/issues/57) -- Real-time playback discussion
- [Supertonic GitHub Issue #71](https://github.com/supertone-inc/supertonic/issues/71) -- Pause control request
- [thewh1teagle/kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) -- ONNX-based Kokoro implementation
- Local codebase: `/home/peppi/Dev/jack-voice/supertonic/src/lib.rs`
- Local codebase: `/home/peppi/Dev/jack-voice/jack-voice/src/kokoro_tts.rs`
- Local codebase: `/home/peppi/Dev/jack-voice/jack-voice/src/tts.rs`
- Local codebase: `/home/peppi/Dev/nanobot/src/voice.rs`
- Local codebase: `/home/peppi/Dev/nanobot/src/voice_pipeline.rs`

## Recommendations

1. **Fix the streaming path click issue:** The `start_streaming_speak` function in `voice.rs` skips `apply_fade_envelope`. Adding it there would eliminate remaining clicks in the CLI voice mode.

2. **Consider crossfade instead of hard concatenation for Kokoro chunks:** The DirectPipeline in `kokoro_tts.rs` concatenates chunks with no processing. A short crossfade (5-10ms overlap-add) between chunks would smooth transitions.

3. **Add inter-chunk silence for Kokoro:** Supertonic adds 0.3s silence between its internal chunks; Kokoro does not. Adding even a small silence gap (50-100ms) or crossfade region would improve naturalness.

4. **For lower latency:** Consider reducing Supertonic inference steps from 5 to 2-3 (RTF improves significantly with minimal quality loss according to benchmarks).

5. **For better streaming:** The current sentence-level streaming (SentenceAccumulator -> synthesis thread -> playback thread via bounded channel) is already well-designed. True sub-sentence streaming would require model-level changes that neither engine supports.

## Open Questions

- The Supertonic sample rate (44.1kHz) vs Kokoro sample rate (24kHz) mismatch is handled by the playback layer but could cause issues if audio from both engines is ever mixed or compared. Is resampling needed for any downstream use case?
- The Supertonic mid-sentence gap retry mechanism adds latency (up to 3x synthesis time in worst case). Is this acceptable for real-time voice, or should it be configurable/disableable?
- kokoro-tiny vs the DirectPipeline: the code shows `needs_direct_pipeline()` returns true for ALL voices (0-52), meaning the kokoro-tiny engine path is never actually used. Is this intentional?
