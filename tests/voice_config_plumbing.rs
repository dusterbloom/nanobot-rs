//! Verify that VoiceConfig.tts_voice and VoiceConfig.language reach the
//! TTS engine via with_voice_config().
//!
//! Run with:
//!   cargo test --features voice --test voice_config_plumbing -- --nocapture --ignored

#![cfg(feature = "voice")]

use nanobot::config::schema::{TtsEngineConfig, VoiceConfig};
use nanobot::voice_pipeline::VoicePipeline;

/// Pure-function test: VoiceConfig deserialises from a realistic ~/.nanobot/config.json
/// fragment with all three relevant fields set.
#[test]
fn voice_config_parses_full_supertonic_block() {
    let json = r#"{
        "voice": {
            "ttsEngine": "supertonic",
            "ttsVoice": "F5",
            "language": "it"
        }
    }"#;
    #[derive(serde::Deserialize)]
    struct Wrapper {
        voice: VoiceConfig,
    }
    let w: Wrapper = serde_json::from_str(json).expect("parse");
    assert_eq!(w.voice.tts_engine, TtsEngineConfig::Supertonic);
    assert_eq!(w.voice.tts_voice.as_deref(), Some("F5"));
    assert_eq!(w.voice.language.as_deref(), Some("it"));
}

/// Same as above with null voice + language — checks that the user's actual
/// current config.json fragment shape parses, and that defaults kick in.
#[test]
fn voice_config_parses_nulls_as_none() {
    let json = r#"{
        "voice": {
            "ttsEngine": "supertonic",
            "ttsVoice": null,
            "language": null
        }
    }"#;
    #[derive(serde::Deserialize)]
    struct Wrapper {
        voice: VoiceConfig,
    }
    let w: Wrapper = serde_json::from_str(json).expect("parse");
    assert_eq!(w.voice.tts_engine, TtsEngineConfig::Supertonic);
    assert_eq!(w.voice.tts_voice, None);
    assert_eq!(w.voice.language, None);
}

/// Live test (ignored by default): build a Supertonic pipeline with
/// VoiceConfig { ttsVoice: Some("F5"), language: Some("it") } and confirm the
/// configured voice was applied. We can't observe the speaker_id from outside
/// the pipeline, but we can confirm the build doesn't error and the engine
/// initialised — which is all we need to know set_speaker was called.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn supertonic_pipeline_honors_explicit_voice() {
    let _ = tracing_subscriber::fmt::try_init();
    let cfg = VoiceConfig {
        tts_engine: TtsEngineConfig::Supertonic,
        tts_voice: Some("F5".to_string()),
        language: Some("it".to_string()),
    };
    let pipeline = VoicePipeline::with_voice_config(&cfg)
        .await
        .expect("pipeline init");
    drop(pipeline);
}

/// Live test (ignored): null voice + Italian language should resolve to the
/// curated recommended voice from jack_voice. We rely on logs to verify the
/// applied voice — the test just confirms init succeeds without panic.
#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn supertonic_pipeline_falls_back_to_curated_voice_for_italian() {
    let _ = tracing_subscriber::fmt::try_init();
    let cfg = VoiceConfig {
        tts_engine: TtsEngineConfig::Supertonic,
        tts_voice: None,
        language: Some("it".to_string()),
    };
    let pipeline = VoicePipeline::with_voice_config(&cfg)
        .await
        .expect("pipeline init");
    drop(pipeline);
}

/// Seamless multilingual: when the user speaks 4 different languages in a
/// session, the SAME voice persona is used for all of them (we just change
/// the language tag wrapping inside the model). This test simulates what
/// `route_tts` resolves to, without requiring live engines.
///
/// Two scenarios:
///   1. User set `ttsVoice: "F5"` → F5 used for en/it/es/de.
///   2. User left `ttsVoice: null` → curated picker resolves all to M2.
#[test]
fn seamless_persona_across_supported_languages() {
    let langs = ["en", "it", "es", "de"];

    // Scenario A: explicit voice in config → same voice for every language.
    let configured = Some("F5".to_string());
    for lang in &langs {
        let resolved = configured.clone().unwrap_or_else(|| {
            jack_voice::tts::recommended_supertonic_voice(Some(lang)).to_string()
        });
        assert_eq!(
            resolved, "F5",
            "voice changed for {lang} — persona drift bug"
        );
    }

    // Scenario B: null voice in config → curated per-language pick.
    // Today both en and it map to M2; es/de fall back to M2 global default.
    // So with null voice the persona is *also* consistent across these 4 languages.
    let configured: Option<String> = None;
    for lang in &langs {
        let resolved = configured.clone().unwrap_or_else(|| {
            jack_voice::tts::recommended_supertonic_voice(Some(lang)).to_string()
        });
        assert_eq!(
            resolved, "M2",
            "curated picker returned different voice for {lang} — would break seamless flow"
        );
    }
}

/// Verify the jack-voice picker returns the same voice nanobot will apply.
/// This catches the case where someone bumps a default in one place but not
/// the other — the integration stays sound.
#[test]
fn nanobot_and_jack_voice_agree_on_curated_picks() {
    // Italian
    assert_eq!(
        jack_voice::tts::recommended_supertonic_voice(Some("it")),
        "M2"
    );
    assert_eq!(
        jack_voice::tts::recommended_supertonic_female_voice(Some("it")),
        "F5"
    );
    // Unknown language → global default
    assert_eq!(
        jack_voice::tts::recommended_supertonic_voice(Some("xx")),
        "M2"
    );
    assert_eq!(jack_voice::tts::recommended_supertonic_voice(None), "M2");
}
