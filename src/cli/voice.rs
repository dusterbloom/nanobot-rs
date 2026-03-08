//! Voice and realtime commands (feature-gated behind "voice").

use std::io::{self, Write};
use std::sync::Arc;

use crate::config::loader::load_config;

use super::core_builder::{build_core_handle, create_agent_loop};

#[cfg(feature = "voice")]
pub(crate) fn parse_input_mode(s: &str) -> crate::realtime::InputMode {
    match s.to_lowercase().as_str() {
        "continuous" | "c" => crate::realtime::InputMode::Continuous,
        "ptt" | "push-to-talk" | "p" => crate::realtime::InputMode::PushToTalk,
        _ => crate::realtime::InputMode::Continuous,
    }
}

#[cfg(feature = "voice")]
pub(crate) fn cmd_realtime(
    engine: String,
    voice: String,
    session: String,
    local: bool,
    mode: String,
) {
    use crate::config::schema::TtsEngineConfig;
    use crate::realtime::{
        InputMode, RealtimeConfig, VoiceAgent, VoiceAgentConfig, VoiceAgentEvent,
    };

    println!("{} Realtime Voice Mode\n", crate::LOGO);

    let tts_engine = match engine.to_lowercase().as_str() {
        "pocket" => TtsEngineConfig::Pocket,
        "kokoro" => TtsEngineConfig::Kokoro,
        _ => {
            eprintln!("Unknown TTS engine: {}. Using pocket.", engine);
            TtsEngineConfig::Pocket
        }
    };

    let input_mode = parse_input_mode(&mode);

    // Load nanobot config for LLM provider
    let nanobot_config = load_config(None);
    let is_local = local || !nanobot_config.agents.defaults.local_api_base.is_empty();

    let va_config = VoiceAgentConfig {
        realtime: RealtimeConfig {
            tts_engine,
            input_mode: input_mode.clone(),
            ..Default::default()
        },
        session_key: session.clone(),
        local: is_local,
        ..Default::default()
    };

    println!("  TTS Engine: {:?}", va_config.realtime.tts_engine);
    println!("  TTS: {:?}", va_config.realtime.tts_engine);
    println!("  Session: {}", va_config.session_key);
    println!("  Local: {}", va_config.local);
    if is_local {
        println!("  Model: {}", nanobot_config.agents.defaults.local_model);
    } else {
        println!("  Model: {}", nanobot_config.agents.defaults.model);
    }
    println!();
    if input_mode == InputMode::Continuous {
        println!("  Mode: Continuous (hands-free, VAD-based)");
        println!("  Just speak — no keys needed");
    } else {
        println!("  Mode: Push-to-Talk");
        println!("  Hold Space to speak, release to process");
    }
    println!();
    println!("  Hotkeys:");
    println!("    Q = Quit");
    println!("    I = Interrupt TTS");
    println!();

    // Build LLM provider + agent loop
    let local_model = if nanobot_config.agents.defaults.local_model.is_empty() {
        None
    } else {
        Some(nanobot_config.agents.defaults.local_model.as_str())
    };

    #[cfg(feature = "mlx")]
    let mlx_handle: Option<super::MlxHandle> =
        if nanobot_config.agents.defaults.inference_engine == "mlx" {
            match super::start_mlx_provider(&nanobot_config) {
                Ok(h) => Some(h),
                Err(e) => {
                    eprintln!("⚠ MLX provider failed to start: {e}");
                    None
                }
            }
        } else {
            None
        };

    #[cfg(feature = "mlx")]
    let core_handle = if let Some(ref mlx) = mlx_handle {
        super::build_core_handle_mlx(&nanobot_config, mlx)
    } else {
        build_core_handle(
            &nanobot_config,
            "8080",
            local_model,
            None,
            None,
            None,
            is_local,
        )
    };
    #[cfg(not(feature = "mlx"))]
    let core_handle = build_core_handle(
        &nanobot_config,
        "8080",
        local_model,
        None,
        None,
        None,
        is_local,
    );

    #[cfg(feature = "mlx")]
    let agent_loop = if let Some(ref mlx) = mlx_handle {
        super::create_agent_loop_mlx(core_handle, &nanobot_config, None, None, None, None, mlx)
    } else {
        create_agent_loop(core_handle, &nanobot_config, None, None, None, None)
    };
    #[cfg(not(feature = "mlx"))]
    let agent_loop = create_agent_loop(core_handle, &nanobot_config, None, None, None, None);
    let agent_loop = Arc::new(agent_loop);

    let rt = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");

    rt.block_on(async {
        let mut agent = VoiceAgent::with_agent_loop(va_config, agent_loop);

        match agent.start().await {
            Ok(mut event_rx) => {
                if input_mode == InputMode::PushToTalk {
                    println!("Voice agent initialized. Hold Space to speak.\n");
                } else {
                    println!("Voice agent initialized. Just speak — audio flows automatically.\n");
                }

                while agent.is_running() {
                    tokio::select! {
                        Some(event) = event_rx.recv() => {
                            match event {
                                VoiceAgentEvent::Ready => {
                                    println!("\x1b[2m[Ready]\x1b[0m");
                                }
                                VoiceAgentEvent::UserSpeechStart => {
                                    print!("\x1b[2m[Listening...]\x1b[0m ");
                                    let _ = io::stdout().flush();
                                }
                                VoiceAgentEvent::UserSpeechEnd => {
                                    println!("\x1b[2m[Processing]\x1b[0m");
                                }
                                VoiceAgentEvent::UserText { text, language } => {
                                    println!("\x1b[1m> {}\x1b[0m (lang: {})", text, language);
                                }
                                VoiceAgentEvent::LlmResponseStart => {
                                    print!("\x1b[36m");
                                    let _ = io::stdout().flush();
                                }
                                VoiceAgentEvent::LlmTextDelta { delta } => {
                                    print!("{}", delta);
                                    let _ = io::stdout().flush();
                                }
                                VoiceAgentEvent::LlmResponseComplete { full_text } => {
                                    println!("\x1b[0m");
                                    let _ = full_text;
                                }
                                VoiceAgentEvent::AudioChunk { samples, sample_rate } => {
                                    let _ = (samples, sample_rate);
                                }
                                VoiceAgentEvent::AudioComplete => {
                                    println!("\x1b[2m[Done speaking]\x1b[0m");
                                }
                                VoiceAgentEvent::Error(e) => {
                                    eprintln!("\x1b[31mError: {}\x1b[0m", e);
                                }
                            }
                        }
                        _ = tokio::time::sleep(std::time::Duration::from_millis(100)) => {
                            use crossterm::event::{self, Event, KeyCode};
                            if event::poll(std::time::Duration::from_millis(10)).unwrap_or(false) {
                                if let Ok(Event::Key(key)) = event::read() {
                                    match key.code {
                                        KeyCode::Char('q') | KeyCode::Esc => {
                                            agent.stop();
                                        }
                                        KeyCode::Char('i') => {
                                            println!("\x1b[2m[Interrupted]\x1b[0m");
                                        }
                                        KeyCode::Char(' ') => {
                                            // Space key is only used in PTT mode.
                                            // In continuous mode, audio flows automatically via AudioCapture.
                                            if input_mode == InputMode::PushToTalk {
                                                // TODO: Toggle recording state for PTT.
                                                // This will be connected to audio_tx gating in the future.
                                                eprintln!("[PTT] Space pressed");
                                            }
                                        }
                                        _ => {}
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Err(e) => {
                eprintln!("Failed to start voice agent: {}", e);
                eprintln!("\nTip: If using Qwen TTS, ensure GPU is available.");
            }
        }
    });
}

#[cfg(feature = "voice")]
pub(crate) fn cmd_realtime_server(port: u16, engine: String, voice: String, host: String) {
    use crate::config::schema::TtsEngineConfig;
    use crate::realtime::{RealtimeServer, RealtimeServerConfig};

    println!("{} Realtime WebSocket Server\n", crate::LOGO);

    let tts_engine = match engine.to_lowercase().as_str() {
        "pocket" => TtsEngineConfig::Pocket,
        "kokoro" => TtsEngineConfig::Kokoro,
        _ => {
            eprintln!("Unknown TTS engine: {}. Using pocket.", engine);
            TtsEngineConfig::Pocket
        }
    };

    let config = RealtimeServerConfig {
        port,
        tts_engine,
        voice,
        host,
    };

    println!("  Listen: ws://{}", config.host);
    println!("  Port: {}", config.port);
    println!("  TTS Engine: {:?}", config.tts_engine);
    println!("  Voice: {}", config.voice);
    println!();
    println!(
        "  OpenAI-compatible endpoint: ws://{}:{}/v1/realtime",
        config.host, config.port
    );
    println!();
    println!("  Press Ctrl+C to stop");
    println!();

    let rt = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");

    rt.block_on(async {
        let server = RealtimeServer::new(config);

        if let Err(e) = server.start().await {
            eprintln!("Failed to start server: {}", e);
            std::process::exit(1);
        }

        // Wait for Ctrl+C
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to listen for Ctrl+C");

        println!("\nShutting down...");
        server.stop();
    });
}

#[cfg(feature = "voice")]
pub(crate) fn cmd_voice_list(engine: String) {
    println!("{} Available Voices\n", crate::LOGO);

    match engine.to_lowercase().as_str() {
        "kokoro" => {
            println!("Kokoro TTS Voices (use with --engine kokoro):\n");
            println!("  Voice IDs: 0-10 (numeric)");
            println!("  Use: nanobot realtime --engine kokoro --voice 3");
        }
        "pocket" => {
            println!("Pocket TTS Voices (use with --engine pocket):\n");
            println!("  alba, marius, javert (default)");
        }
        _ => {
            eprintln!("Unknown TTS engine: {}", engine);
            eprintln!("Valid engines: pocket, kokoro, qwen, qwenLarge, qwenOnnx, qwenOnnxInt8");
        }
    }
}

#[cfg(feature = "voice")]
pub(crate) fn cmd_voice_clone(name: String, audio: String, transcript: Option<String>) {
    println!("{} Voice Cloning\n", crate::LOGO);

    let audio_path = std::path::PathBuf::from(&audio);
    if !audio_path.exists() {
        eprintln!("Error: Audio file not found: {}", audio);
        std::process::exit(1);
    }

    // Get workspace path for storing voice profiles
    let workspace = crate::utils::helpers::get_workspace_path(None);
    let voices_dir = workspace.join("voices");
    if let Err(e) = std::fs::create_dir_all(&voices_dir) {
        eprintln!("Error: Failed to create voices directory: {}", e);
        std::process::exit(1);
    }

    // Copy audio file to workspace
    let dest_path = voices_dir.join(format!("{}.wav", name));
    if let Err(e) = std::fs::copy(&audio, &dest_path) {
        eprintln!("Error: Failed to copy audio file: {}", e);
        std::process::exit(1);
    }

    // Create metadata file
    let metadata = serde_json::json!({
        "name": name,
        "audio_path": dest_path.display().to_string(),
        "transcript": transcript,
        "created_at": chrono::Utc::now().to_rfc3339(),
    });

    let meta_path = voices_dir.join(format!("{}.json", name));
    if let Err(e) = std::fs::write(&meta_path, serde_json::to_string_pretty(&metadata).unwrap()) {
        eprintln!("Error: Failed to write metadata: {}", e);
        std::process::exit(1);
    }

    println!("Voice profile '{}' created successfully!\n", name);
    println!("  Audio: {}", dest_path.display());
    println!("  Config: {}", meta_path.display());
    println!();
    println!("To use this voice, add to ~/.nanobot/config.json:");
    println!();
    println!(r#"  "voice": {{"#);
    println!(r#"    "ttsEngine": "qwenLarge","#);
    println!(r#"    "voiceCloneRef": {{"#);
    println!(r#"      "audioPath": "{}","#, dest_path.display());
    if let Some(ref t) = transcript {
        println!(r#"      "transcript": "{}"#, t);
    }
    println!(r#"    }}"#);
    println!(r#"  }}"#);
}

#[cfg(feature = "voice")]
pub(crate) fn cmd_voice_config() {
    println!("{} Voice Configuration\n", crate::LOGO);
    println!("Add a 'voice' section to ~/.nanobot/config.json:\n");
    println!(r#"{{"#);
    println!(r#"  "voice": {{"#);
    println!(r#"    "ttsEngine": "qwen",          // pocket, kokoro, qwen, qwenLarge"#);
    println!(r#"    "ttsVoice": "ryan",           // Voice ID or name"#);
    println!(r#"    "voiceCloneRef": {{           // Optional, for qwenLarge only"#);
    println!(r#"      "audioPath": "~/.nanobot/workspace/voices/myvoice.wav","#);
    println!(r#"      "transcript": "Optional transcript""#);
    println!(r#"    }}"#);
    println!(r#"  }}"#);
    println!(r#"}}"#);
    println!();
    println!("Commands:");
    println!("  nanobot voice list --engine qwen     List available voices");
    println!("  nanobot voice clone myvoice audio.wav Clone a voice");
    println!("  nanobot realtime --engine qwen       Start realtime session");
}
