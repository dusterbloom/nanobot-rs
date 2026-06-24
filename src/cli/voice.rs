//! Voice commands (feature-gated behind "voice").
//!
//! The realtime subsystem (cmd_realtime, cmd_realtime_server, parse_input_mode)
//! was removed in commit 1f9e1d5 along with src/realtime/*. Only voice listing
//! and config helpers remain here.

#[cfg(feature = "voice")]
pub(crate) fn cmd_voice_list(engine: String) {
    println!("{} Available Voices\n", crate::LOGO);

    match engine.to_lowercase().as_str() {
        "supertonic" => {
            println!("Supertonic TTS Voices (use with --engine supertonic):\n");
            println!("  F1, F2, F3, F4, F5");
            println!("  M1, M2, M3, M4, M5");
            println!("  Use: nanobot voice config with ttsEngine \"supertonic\"");
        }
        "say" => {
            println!("Apple say Voices (use with ttsEngine \"say\" on macOS):\n");
            println!("  Use a macOS voice name such as Samantha, Ava, or Alice");
        }
        _ => {
            eprintln!("Unknown TTS engine: {}", engine);
            eprintln!("Valid engines: supertonic, say");
        }
    }
}

#[cfg(feature = "voice")]
pub(crate) fn cmd_voice_config() {
    println!("{} Voice Configuration\n", crate::LOGO);
    println!("Add a 'voice' section to ~/.nanobot/config.json:\n");
    println!(r#"{{"#);
    println!(r#"  "voice": {{"#);
    println!(r#"    "ttsEngine": "supertonic",    // supertonic or say"#);
    println!(r#"    "ttsVoice": "M2",             // Supertonic ID or macOS say voice"#);
    println!(r#"    "language": "auto""#);
    println!(r#"  }}"#);
    println!(r#"}}"#);
    println!();
    println!("Commands:");
    println!("  nanobot voice list --engine supertonic  List available voices");
    println!("  nanobot voice list --engine say         List Apple say support");
}
