# Continuity Ledger: Voice Mode & Local LLM

## Goal
Fully functional voice assistant with local LLM support: low-latency TTS, interruptible playback, model switching, natural conversation flow.

## Constraints
- `TextToSpeech` is NOT Send - cannot move to background thread
- Supertonic v2 diffusion model has known upstream issues (word dropping, silent segments) - no fix available
- paplay via PulseAudio for audio output (WSL2 uses /mnt/wslg/PulseServer)
- llama.cpp server for local inference

## Key Decisions
- **Sentence-by-sentence TTS streaming**: Pipe PCM to single paplay stdin per response, not one paplay per sentence. Natural pipelining, zero infrastructure. [2026-02-09]
- **Cancel flag between sentences**: Since TTS is not Send, use AtomicBool checked between synthesize() calls. Max interrupt delay ~500ms (one sentence). [2026-02-09]
- **Keypress watcher thread for TTS interrupt**: Spawns crossterm raw-mode listener alongside speak(). Sets cancel flag on Enter/Ctrl+Space. [2026-02-09]
- **pkill for stale server cleanup**: kill_stale_llama_servers() runs pkill -f llama-server before spawning. Simple, safe since we own all instances. [2026-02-09]
- **Stronger voice prompt**: STRICT RULES format with explicit "1-3 sentences" and "NEVER use emoji/markdown". Cloud models were ignoring softer instructions. [2026-02-09]

## State
- Done:
  - [x] Voice mode: STT (Whisper) + TTS (Supertonic) + paplay
  - [x] Streaming TTS: sentence-by-sentence pipelining (~300-500ms time-to-first-audio)
  - [x] TTS interrupt: Enter/Ctrl+Space during playback kills paplay, loops to recording
  - [x] /model command: list ~/models/*.gguf, switch live, restart server
  - [x] Stale server cleanup: pkill orphaned llama-server before spawn
  - [x] Voice prompt hardening: no emoji, no markdown, 1-3 sentences
  - [x] /local auto-spawn: llama.cpp server with --ctx-size 16384
- Now: Idle - no active work
- Remaining:
  - [ ] Investigate Supertonic word-dropping (upstream diffusion issue, no fix yet)
  - [ ] Consider multi-sample generation (generate N, pick best) for quality vs latency tradeoff
  - [ ] Voice activity detection (VAD) for hands-free recording start/stop
  - [ ] Streaming LLM response + TTS (start speaking before full LLM response)

## Open Questions
- UNCONFIRMED: Can supertonic's diffusion steps parameter reduce word dropping? (issue #39 suggests picking 1st sample instead of 4th helps)
- UNCONFIRMED: Would pre-processing text (expand abbreviations, normalize numbers) reduce TTS failures? (issue #60)

## Known Upstream Issues (supertone-inc/supertonic)
- [#48](https://github.com/supertone-inc/supertonic/issues/48): Silent words randomly - diffusion is non-deterministic
- [#39](https://github.com/supertone-inc/supertonic/issues/39): Word drops on multi-sample generation - model handles ~2-3 sentences max
- [#30](https://github.com/supertone-inc/supertonic/issues/30): Specific words ("else") silently disappear
- [#60](https://github.com/supertone-inc/supertonic/issues/60): Long sentences + numbers cause drops

## Working Set
- Files: `src/voice.rs`, `src/main.rs`, `src/agent/context.rs`
- Branch: main
- Build: `cargo build --features voice`
- Test: `cargo test`
- Local models: `~/models/*.gguf`
- Upstream TTS: `supertone-inc/supertonic` (ONNX diffusion)
