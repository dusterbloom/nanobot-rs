---
date: 2026-02-08T23:09:50+0100
session_name: general
researcher: claude
git_commit: 1a7d617ac26be4f2bffe3ee2ed720248931558ab
branch: main
repository: nanobot
topic: "PulseAudio CLI Voice I/O - Bypass cpal for WSL2"
tags: [implementation, voice, pulseaudio, wsl2, audio]
status: complete
last_updated: 2026-02-08
last_updated_by: claude
type: implementation_strategy
root_span_id: ""
turn_span_id: ""
---

# Handoff: Replace cpal/rodio with PulseAudio CLI (parec/paplay) for voice I/O

## Task(s)
- **[COMPLETED]** Rewrite `src/voice.rs` to use `parec`/`paplay` subprocess pipes instead of `cpal::AudioCapture` and `rodio::AudioPlayer` from jack-voice. This was needed because cpal's ALSA backend fails on WSL2 (device enumeration broken), while PulseAudio CLI tools work perfectly via WSLg socket.
- **[COMPLETED]** Build verification with `cargo build --features voice` — compiles cleanly (warnings only, no errors).
- **[NOT TESTED]** Runtime verification — requires actual WSL2 audio environment to test `/voice` toggle, mic capture, and TTS playback.

## Critical References
- Implementation plan was provided inline in user message (not a separate file)
- `src/voice.rs` — the only file that changed
- jack-voice library at `../jack-voice/jack-voice/src/lib.rs` — still used for STT, TTS, VAD, and model downloads

## Recent changes
- `src/voice.rs:1-199` — Complete rewrite of voice module

Key changes:
- Removed imports: `AudioCapture`, `AudioPlayer` from jack-voice
- Added `pulse_server()` (line ~19): detects WSLg socket at `/mnt/wslg/PulseServer`, falls back to `$PULSE_SERVER` env var
- Added `start_parec()` (line ~27): spawns `parec --format=float32le --rate=16000 --channels=1`, reads stdout pipe into mpsc channel (same interface VAD expects — `Vec<f32>` chunks)
- Added `start_paplay()` (line ~56): spawns `paplay --raw --format=float32le`, writes f32le bytes to stdin in background thread
- `VoiceSession::new()`: removed `AudioPlayer::new()`, added `parec --version` fail-fast check
- `VoiceSession::record_and_transcribe()`: replaced `AudioCapture::start(sample_tx)` with `start_parec(sample_tx)`, replaced `capture.stop()` with `child.kill()`
- `VoiceSession::speak()`: replaced `self.player.play(samples, rate)` with `self.playback = Some(start_paplay(samples, rate)?)`
- `VoiceSession::stop_playback()`: replaced `self.player.stop()` with `child.kill()` + `child.wait()`
- Struct field changed: `player: AudioPlayer` → `playback: Option<Child>`

## Learnings
- cpal's ALSA backend on WSL2 is fundamentally broken for device enumeration. Configuring `.asoundrc` and `PULSE_SERVER` for ALSA-over-PulseAudio does not help cpal.
- PulseAudio CLI tools (`parec`, `paplay`) work perfectly on WSL2 via the WSLg PulseAudio socket at `/mnt/wslg/PulseServer`.
- The `parec` stdout outputs raw PCM in the exact format specified by `--format`, making it trivial to pipe into the existing VAD pipeline.
- `paplay --raw` accepts raw PCM on stdin, so TTS output can be written directly without encoding.
- The jack-voice library is still required for STT (`SpeechToText`), TTS (`TextToSpeech`), VAD (`VoiceActivityDetector`), and model management. Only the audio capture/playback layer was replaced.
- `Cargo.toml` still has `jack-voice` as an optional dependency under the `voice` feature — this is correct since we still use its STT/TTS/VAD.

## Post-Mortem (Required for Artifact Index)

### What Worked
- Subprocess approach: spawning `parec`/`paplay` as child processes with piped stdio is clean, ~50 lines total, and maintains the exact same `VoiceSession` API surface.
- The mpsc channel interface between `start_parec()` and the VAD thread is identical to what `AudioCapture::start()` provided, so no changes were needed in the VAD/STT pipeline.
- Fail-fast pattern: checking `parec --version` in `VoiceSession::new()` gives a clear error message if `pulseaudio-utils` isn't installed.

### What Failed
- No failures during implementation — the plan was well-scoped and straightforward.
- Note: runtime testing not yet performed (need WSL2 audio environment).

### Key Decisions
- Decision: Use PulseAudio CLI tools instead of fixing cpal or using WebSocket/browser audio
  - Alternatives considered: (1) Fix cpal ALSA config (impossible — device enumeration broken), (2) WebSocket browser audio (~130 lines, UX change, more latency)
  - Reason: Zero UX change, ~50 lines of code, same latency as native, works on Linux/WSL today
- Decision: Keep jack-voice dependency for STT/TTS/VAD
  - Reason: Only the audio I/O layer was broken; STT/TTS/VAD work fine and would be complex to replace

## Artifacts
- `src/voice.rs` — complete rewritten voice module

## Action Items & Next Steps
1. **Runtime test on WSL2**: Run `cargo run --features voice -- agent`, toggle `/voice`, test mic capture and TTS playback
2. **Verify PulseAudio prerequisites**: Ensure `pulseaudio-utils` is installed (`sudo apt install pulseaudio-utils`)
3. **Test interrupt behavior**: Press Enter during TTS playback to verify `stop_playback()` kills `paplay` correctly
4. **Consider**: Add WebSocket/browser audio as a future transport option for non-PulseAudio environments
5. **Consider**: The changes are uncommitted on `main` — commit when satisfied with testing

## Other Notes
- The `Cargo.toml` still lists `jack-voice` as optional dep — this is intentional and correct
- `main.rs` voice integration code (lines 287-461) is unchanged — the `/voice` toggle, Enter-to-speak loop, and TTS-after-response all work through the same `VoiceSession` API
- `build.rs` and `Makefile` exist as untracked files but are unrelated to this change
- All existing warnings in the build output are pre-existing (dead code in various modules) and unrelated to the voice changes
