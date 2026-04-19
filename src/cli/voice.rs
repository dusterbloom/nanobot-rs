//! Voice commands (feature-gated behind "voice").
//!
//! The realtime subsystem (cmd_realtime, cmd_realtime_server, parse_input_mode)
//! was removed in commit 1f9e1d5 along with src/realtime/*. Only voice profile
//! management commands remain here.

use std::io::{self, Write};

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

    let _ = io::stdout().flush();
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
