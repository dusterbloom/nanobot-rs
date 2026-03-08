//! Voice agent integration for realtime voice conversations with LLM.
//!
//! Combines RealtimeSession with AgentLoop for full voice-to-voice conversations:
//! 1. User speaks -> VAD detects -> STT transcribes
//! 2. Transcription -> LLM processes -> Streaming response
//! 3. LLM response -> TTS synthesizes -> Audio playback

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::mpsc;
use tokio::sync::mpsc::UnboundedSender;
use tokio_util::sync::CancellationToken;

use crate::config::schema::TtsEngineConfig;

/// Configuration for the voice agent.
#[derive(Debug, Clone)]
pub struct VoiceAgentConfig {
    /// Realtime session config.
    pub realtime: super::RealtimeConfig,
    /// Session key for the LLM agent.
    pub session_key: String,
    /// Channel identifier for the agent.
    pub channel: String,
    /// Chat ID for the agent.
    pub chat_id: String,
    /// Use local LLM instead of cloud.
    pub local: bool,
    /// System prompt for voice mode.
    pub system_prompt: Option<String>,
}

impl Default for VoiceAgentConfig {
    fn default() -> Self {
        Self {
            realtime: super::RealtimeConfig::default(),
            session_key: "voice:default".to_string(),
            channel: "voice".to_string(),
            chat_id: "voice".to_string(),
            local: false,
            system_prompt: Some("You are a helpful voice assistant. Keep responses concise and conversational. Respond in the same language the user speaks. Never use emoji, markdown formatting, or special characters — your output will be spoken aloud.".to_string()),
        }
    }
}

/// Events emitted by the voice agent.
#[derive(Debug, Clone, PartialEq)]
pub enum VoiceAgentEvent {
    /// User started speaking.
    UserSpeechStart,
    /// User stopped speaking.
    UserSpeechEnd,
    /// User's transcribed text.
    UserText { text: String, language: String },
    /// LLM started responding.
    LlmResponseStart,
    /// LLM text delta (streaming).
    LlmTextDelta { delta: String },
    /// LLM response complete.
    LlmResponseComplete { full_text: String },
    /// TTS audio chunk ready.
    AudioChunk { samples: Vec<f32>, sample_rate: u32 },
    /// TTS finished playing.
    AudioComplete,
    /// Error occurred.
    Error(String),
    /// Agent is ready.
    Ready,
}

/// Voice agent state for the continuous voice pipeline.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum VoiceAgentState {
    Listening,
    Processing,
    Speaking,
}

/// Actions to execute after a state transition.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum VoiceAction {
    EmitEvent(VoiceAgentEvent),
    StartLlm {
        text: String,
        language: String,
    },
    /// Cancel in-flight LLM generation AND stop TTS playback (barge-in).
    CancelAll,
}

/// Pure state transition function for the voice agent.
///
/// Given the current state and an incoming event, returns the new state
/// and a list of actions to execute. This is the core of the voice pipeline
/// state machine — fully testable with synthetic inputs.
pub(crate) fn next_state(
    state: &VoiceAgentState,
    event: &super::RealtimeEvent,
) -> (VoiceAgentState, Vec<VoiceAction>) {
    use super::RealtimeEvent;
    use VoiceAgentState::*;

    match (state, event) {
        // Listening: user starts speaking
        (Listening, RealtimeEvent::SpeechStart) => (
            Listening,
            vec![VoiceAction::EmitEvent(VoiceAgentEvent::UserSpeechStart)],
        ),

        // Listening: user stops speaking
        (Listening, RealtimeEvent::SpeechEnd) => (
            Listening,
            vec![VoiceAction::EmitEvent(VoiceAgentEvent::UserSpeechEnd)],
        ),

        // Listening: turn complete -> start LLM processing
        (Listening, RealtimeEvent::TurnComplete { text, language }) => (
            Processing,
            vec![
                VoiceAction::EmitEvent(VoiceAgentEvent::UserText {
                    text: text.clone(),
                    language: language.clone(),
                }),
                VoiceAction::StartLlm {
                    text: text.clone(),
                    language: language.clone(),
                },
            ],
        ),

        // Processing or Speaking: user starts speaking -> barge-in
        (Processing | Speaking, RealtimeEvent::SpeechStart) => (
            Listening,
            vec![
                VoiceAction::CancelAll,
                VoiceAction::EmitEvent(VoiceAgentEvent::UserSpeechStart),
            ],
        ),

        // Speaking: TTS playback complete -> back to listening
        (Speaking, RealtimeEvent::TtsPlaybackComplete) => (
            Listening,
            vec![VoiceAction::EmitEvent(VoiceAgentEvent::AudioComplete)],
        ),

        // Any state: error
        (_, RealtimeEvent::Error(msg)) => (
            Listening,
            vec![VoiceAction::EmitEvent(VoiceAgentEvent::Error(msg.clone()))],
        ),

        // All other combinations: no state change, no actions
        (current, _) => (current.clone(), vec![]),
    }
}

/// Abstraction over LLM processing for the voice pipeline.
///
/// Allows injecting mock LLM processors for testing without
/// requiring a full AgentLoop with config, API keys, etc.
#[async_trait]
pub trait LlmProcessor: Send + Sync + 'static {
    /// Process user text and stream response deltas.
    ///
    /// Returns the full response text. Each delta should be sent
    /// to `text_delta_tx` as it arrives from the LLM.
    async fn process_text(
        &self,
        text: &str,
        session_key: &str,
        language: Option<&str>,
        text_delta_tx: UnboundedSender<String>,
        cancellation_token: Option<CancellationToken>,
    ) -> String;
}

/// Production LLM processor wrapping AgentLoop::process_direct_streaming.
pub struct AgentLoopProcessor {
    agent_loop: std::sync::Arc<crate::agent::agent_loop::AgentLoop>,
    channel: String,
    chat_id: String,
}

impl AgentLoopProcessor {
    pub fn new(
        agent_loop: std::sync::Arc<crate::agent::agent_loop::AgentLoop>,
        channel: String,
        chat_id: String,
    ) -> Self {
        Self {
            agent_loop,
            channel,
            chat_id,
        }
    }
}

#[async_trait]
impl LlmProcessor for AgentLoopProcessor {
    async fn process_text(
        &self,
        text: &str,
        session_key: &str,
        language: Option<&str>,
        text_delta_tx: UnboundedSender<String>,
        cancellation_token: Option<CancellationToken>,
    ) -> String {
        self.agent_loop
            .process_direct_streaming(
                text,
                session_key,
                &self.channel,
                &self.chat_id,
                language,
                text_delta_tx,
                None, // tool_event_tx
                cancellation_token,
                None, // priority_rx
            )
            .await
    }
}

/// Run the voice agent event loop.
///
/// This is the core event processing loop, extracted as a standalone async function
/// for testability. It:
/// 1. Receives RealtimeEvents (from audio pipeline or test harness)
/// 2. Runs them through the `next_state()` pure state machine
/// 3. Executes resulting actions (emit events, start LLM, cancel)
///
/// The LLM is invoked via the `LlmProcessor` trait, allowing mock injection in tests.
pub(crate) async fn run_voice_event_loop(
    config: VoiceAgentConfig,
    llm: Box<dyn LlmProcessor>,
    mut event_rx: mpsc::Receiver<super::RealtimeEvent>,
    event_tx: mpsc::Sender<VoiceAgentEvent>,
) {
    let llm = std::sync::Arc::new(llm);
    let mut state = VoiceAgentState::Listening;
    let mut current_cancel: Option<CancellationToken> = None;

    while let Some(rt_event) = event_rx.recv().await {
        let (new_state, actions) = next_state(&state, &rt_event);
        state = new_state;

        for action in actions {
            match action {
                VoiceAction::EmitEvent(evt) => {
                    let _ = event_tx.send(evt).await;
                }
                VoiceAction::StartLlm { text, language } => {
                    // Cancel any in-flight LLM
                    if let Some(ref cancel) = current_cancel {
                        cancel.cancel();
                    }

                    let cancel_token = CancellationToken::new();
                    current_cancel = Some(cancel_token.clone());

                    let llm = llm.clone();
                    let event_tx = event_tx.clone();
                    let session_key = config.session_key.clone();

                    tokio::spawn(async move {
                        let _ = event_tx.send(VoiceAgentEvent::LlmResponseStart).await;

                        let (delta_tx, mut delta_rx) =
                            tokio::sync::mpsc::unbounded_channel::<String>();

                        // Spawn delta forwarder
                        let event_tx_deltas = event_tx.clone();
                        let delta_forwarder = tokio::spawn(async move {
                            while let Some(delta) = delta_rx.recv().await {
                                let _ = event_tx_deltas
                                    .send(VoiceAgentEvent::LlmTextDelta { delta })
                                    .await;
                            }
                        });

                        let full_text = llm
                            .process_text(
                                &text,
                                &session_key,
                                Some(&language),
                                delta_tx,
                                Some(cancel_token),
                            )
                            .await;

                        // Wait for delta forwarder to finish
                        let _ = delta_forwarder.await;

                        let _ = event_tx
                            .send(VoiceAgentEvent::LlmResponseComplete { full_text })
                            .await;
                    });
                }
                VoiceAction::CancelAll => {
                    if let Some(ref cancel) = current_cancel {
                        cancel.cancel();
                    }
                    current_cancel = None;
                    // Note: TTS cancellation will be added in Cycle 6
                }
            }
        }
    }
}

/// Drive LLM text deltas through SentenceAccumulator to TTS.
///
/// Reads from `delta_rx`, pushes each delta into the accumulator (which extracts
/// sentences and sends TtsCommand::Synthesize), and flushes on completion.
/// Also forwards deltas as VoiceAgentEvent::LlmTextDelta to the event channel.
///
/// This is the production wiring used by the `start()` method to connect the
/// LLM streaming output to the TTS pipeline with sentence-level batching.
#[cfg(feature = "voice")]
pub(crate) async fn drive_llm_to_tts(
    mut delta_rx: tokio::sync::mpsc::UnboundedReceiver<String>,
    tts_tx: std::sync::mpsc::Sender<crate::voice_pipeline::TtsCommand>,
    event_tx: mpsc::Sender<VoiceAgentEvent>,
) {
    let mut accumulator = crate::voice_pipeline::SentenceAccumulator::new_streaming(tts_tx);
    let mut delta_count = 0u32;
    while let Some(delta) = delta_rx.recv().await {
        delta_count += 1;
        let _ = event_tx
            .send(VoiceAgentEvent::LlmTextDelta {
                delta: delta.clone(),
            })
            .await;

        // Filter out protocol metadata (NUL-prefixed messages from agent_loop)
        if delta.starts_with('\x00') {
            tracing::debug!("[drive-tts] skipping metadata: {:?}", delta);
            continue;
        }
        // Skip error messages — display them but don't speak them
        if is_error_message(&delta) {
            tracing::debug!("[drive-tts] skipping error message from TTS");
            continue;
        }
        // Strip ANSI escape sequences (thinking token decorations)
        let cleaned = strip_ansi(&delta);
        if !cleaned.is_empty() {
            accumulator.push(&cleaned);
        }
    }
    tracing::debug!(
        "[drive-tts] LLM stream ended after {} deltas, flushing accumulator",
        delta_count
    );
    accumulator.flush();
}

/// Check if a delta looks like an error message that should not be spoken.
fn is_error_message(text: &str) -> bool {
    text.contains("I encountered an error")
        || text.starts_with("[LLM Error]")
        || text.starts_with("[nanobot]")
        || text.contains("error sending request for url")
}

/// Check if a full LLM response indicates a transient error worth retrying.
fn is_error_response(text: &str) -> bool {
    text.contains("I encountered an error")
        || text.starts_with("[LLM Error]")
        || text.contains("error sending request for url")
        || text.contains("HTTP request failed")
}

/// Strip ANSI escape sequences from a string (CSI sequences like \x1b[...m).
fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            // Skip ESC [ ... <final byte>
            if chars.next() == Some('[') {
                for c2 in chars.by_ref() {
                    if c2.is_ascii_alphabetic() {
                        break;
                    }
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Voice agent that integrates realtime voice with LLM.
///
/// This is the main entry point for voice conversations with the AI agent.
/// It coordinates:
/// - Audio capture and VAD
/// - Speech-to-text transcription
/// - LLM processing with streaming responses
/// - Text-to-speech synthesis
/// - Audio playback
pub struct VoiceAgent {
    config: VoiceAgentConfig,
    running: Arc<AtomicBool>,
    agent_loop: Option<Arc<crate::agent::agent_loop::AgentLoop>>,
}

impl VoiceAgent {
    /// Create a new voice agent with the given configuration.
    pub fn new(config: VoiceAgentConfig) -> Self {
        Self {
            config,
            running: Arc::new(AtomicBool::new(false)),
            agent_loop: None,
        }
    }

    /// Create a new voice agent with an AgentLoop for real LLM processing.
    pub fn with_agent_loop(
        config: VoiceAgentConfig,
        agent_loop: Arc<crate::agent::agent_loop::AgentLoop>,
    ) -> Self {
        Self {
            config,
            running: Arc::new(AtomicBool::new(false)),
            agent_loop: Some(agent_loop),
        }
    }

    /// Start the voice agent.
    ///
    /// Returns an event receiver for voice agent events.
    /// The agent will run until `stop()` is called.
    #[cfg(feature = "voice")]
    pub async fn start(&mut self) -> Result<mpsc::Receiver<VoiceAgentEvent>, String> {
        self.running.store(true, Ordering::SeqCst);
        let (event_tx, event_rx) = mpsc::channel(64);

        let realtime_config = self.config.realtime.clone();
        let running = self.running.clone();
        let agent_loop = self.agent_loop.clone();
        let session_key = self.config.session_key.clone();
        let channel = self.config.channel.clone();
        let chat_id = self.config.chat_id.clone();

        // Spawn the main voice agent loop in a dedicated thread
        // We create the RealtimeSession inside the thread to avoid Send issues
        std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("Failed to create tokio runtime");

            rt.block_on(async move {
                use super::{RealtimeSession, RealtimeEvent};

                // Create session inside the dedicated runtime
                let session_result = RealtimeSession::new(realtime_config).await;

                let mut session = match session_result {
                    Ok(s) => s,
                    Err(e) => {
                        let _ = event_tx.send(VoiceAgentEvent::Error(format!("Failed to create session: {}", e))).await;
                        return;
                    }
                };

                let start_result = session.start();

                if let Err(e) = start_result {
                    let _ = event_tx.send(VoiceAgentEvent::Error(format!("Failed to start session: {}", e))).await;
                    return;
                }

                let (_audio_tx, mut realtime_rx) = start_result.unwrap();

                // Set up TTS playback pipeline with echo suppression
                let (tts_en, tts_multi) = session.tts_handles();
                let tts_playing = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
                let tts_state: Option<(
                    std::sync::mpsc::Sender<crate::voice_pipeline::TtsCommand>,
                    std::sync::Arc<std::sync::atomic::AtomicBool>,
                )> = if tts_en.is_some() || tts_multi.is_some() {
                    Some(crate::voice_pipeline::start_tts_playback(
                        tts_en, tts_multi, tts_playing.clone(),
                    ))
                } else {
                    None
                };

                // Send Ready only after session is successfully started
                let _ = event_tx.send(VoiceAgentEvent::Ready).await;

                tracing::info!("Voice agent started (LLM: {}, TTS: {})",
                    if agent_loop.is_some() { "enabled" } else { "stub" },
                    if tts_state.is_some() { "enabled" } else { "none" });

                // Track active LLM cancellation token for barge-in
                let mut current_cancel: Option<CancellationToken> = None;
                // Grace period: ignore SpeechStart for 500ms after entering
                // Processing state, to avoid residual audio triggering false barge-in.
                let mut processing_started_at: Option<std::time::Instant> = None;
                const BARGEIN_GRACE_MS: u128 = 500;

                while running.load(Ordering::SeqCst) {
                    tokio::select! {
                        Some(event) = realtime_rx.recv() => {
                            match event {
                                RealtimeEvent::SpeechStart => {
                                    // Suppress false barge-in from TTS speaker echo
                                    if tts_playing.load(std::sync::atomic::Ordering::SeqCst) {
                                        tracing::debug!("Barge-in: suppressed (TTS playing, echo rejection)");
                                        continue;
                                    }
                                    // Skip barge-in during grace period after processing starts
                                    if let Some(started) = processing_started_at {
                                        if started.elapsed().as_millis() < BARGEIN_GRACE_MS {
                                            tracing::debug!("Barge-in: suppressed ({}ms grace period)", started.elapsed().as_millis());
                                            continue;
                                        }
                                    }
                                    // Barge-in: cancel in-flight LLM + TTS
                                    if let Some(ref cancel) = current_cancel {
                                        cancel.cancel();
                                        current_cancel = None;
                                        tracing::debug!("Barge-in: cancelled LLM");
                                    }
                                    if let Some((_, ref tts_cancel)) = tts_state {
                                        tts_cancel.store(true, std::sync::atomic::Ordering::Relaxed);
                                    }
                                    // Clear tts_playing so echo suppression doesn't block further input
                                    tts_playing.store(false, std::sync::atomic::Ordering::SeqCst);
                                    processing_started_at = None;
                                    let _ = event_tx.send(VoiceAgentEvent::UserSpeechStart).await;
                                }
                                RealtimeEvent::SpeechEnd => {
                                    let _ = event_tx.send(VoiceAgentEvent::UserSpeechEnd).await;
                                }
                                RealtimeEvent::TurnComplete { text, language } => {
                                    // Start grace period to suppress false barge-in
                                    // from residual audio after STT completes
                                    processing_started_at = Some(std::time::Instant::now());

                                    // Cancel any in-flight LLM task from a previous turn
                                    if let Some(ref cancel) = current_cancel {
                                        cancel.cancel();
                                    }

                                    let _ = event_tx.send(VoiceAgentEvent::UserText {
                                        text: text.clone(),
                                        language: language.clone(),
                                    }).await;

                                    if let Some(ref al) = agent_loop {
                                        // Reset TTS cancel flag for new turn
                                        if let Some((_, ref tts_cancel)) = tts_state {
                                            tts_cancel.store(false, std::sync::atomic::Ordering::Relaxed);
                                        }

                                        let cancel_token = CancellationToken::new();
                                        current_cancel = Some(cancel_token.clone());

                                        let al = al.clone();
                                        let event_tx = event_tx.clone();
                                        let sk = session_key.clone();
                                        let ch = channel.clone();
                                        let ci = chat_id.clone();
                                        let tts_tx = tts_state.as_ref().map(|(tx, _)| tx.clone());

                                        tokio::spawn(async move {
                                            const MAX_RETRIES: u32 = 2;
                                            let mut attempt = 0u32;

                                            loop {
                                                let _ = event_tx.send(VoiceAgentEvent::LlmResponseStart).await;

                                                let (delta_tx, delta_rx) = tokio::sync::mpsc::unbounded_channel::<String>();

                                                // Forward deltas to event channel AND TTS
                                                let event_tx_deltas = event_tx.clone();
                                                let delta_forwarder = if let Some(ref tts_tx) = tts_tx {
                                                    let tts_tx = tts_tx.clone();
                                                    tokio::spawn(drive_llm_to_tts(delta_rx, tts_tx, event_tx_deltas))
                                                } else {
                                                    tokio::spawn(async move {
                                                        let mut delta_rx = delta_rx;
                                                        while let Some(delta) = delta_rx.recv().await {
                                                            let _ = event_tx_deltas
                                                                .send(VoiceAgentEvent::LlmTextDelta { delta })
                                                                .await;
                                                        }
                                                    })
                                                };

                                                let cancel_for_call = cancel_token.clone();
                                                let full_text = al
                                                    .process_direct_streaming(
                                                        &text,
                                                        &sk,
                                                        &ch,
                                                        &ci,
                                                        Some(&language),
                                                        delta_tx,
                                                        None,
                                                        Some(cancel_for_call),
                                                        None,
                                                    )
                                                    .await;

                                                let _ = delta_forwarder.await;

                                                // Retry on transient LLM errors
                                                if is_error_response(&full_text) && attempt < MAX_RETRIES && !cancel_token.is_cancelled() {
                                                    attempt += 1;
                                                    tracing::warn!("[voice-agent] LLM error (attempt {}/{}), retrying in 2s", attempt, MAX_RETRIES);
                                                    tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                                                    continue;
                                                }

                                                let _ = event_tx
                                                    .send(VoiceAgentEvent::LlmResponseComplete { full_text })
                                                    .await;
                                                break;
                                            }
                                        });
                                    } else {
                                        // No AgentLoop — echo stub
                                        let response = format!("You said: {}", text);
                                        let _ = event_tx.send(VoiceAgentEvent::LlmResponseStart).await;
                                        let _ = event_tx.send(VoiceAgentEvent::LlmTextDelta { delta: response.clone() }).await;
                                        let _ = event_tx.send(VoiceAgentEvent::LlmResponseComplete { full_text: response }).await;
                                    }
                                }
                                RealtimeEvent::PartialTranscript { text } => {
                                    tracing::debug!("Partial: {}", text);
                                }
                                RealtimeEvent::AudioChunk { samples, sample_rate } => {
                                    let _ = event_tx.send(VoiceAgentEvent::AudioChunk { samples, sample_rate }).await;
                                }
                                RealtimeEvent::SynthesisComplete => {
                                    let _ = event_tx.send(VoiceAgentEvent::AudioComplete).await;
                                }
                                RealtimeEvent::TtsPlaybackComplete => {
                                    let _ = event_tx.send(VoiceAgentEvent::AudioComplete).await;
                                }
                                RealtimeEvent::Error(e) => {
                                    let _ = event_tx.send(VoiceAgentEvent::Error(e)).await;
                                }
                            }
                        }
                        _ = tokio::time::sleep(std::time::Duration::from_millis(100)) => {
                            // Periodic check
                        }
                    }
                }

                session.stop();
                tracing::info!("Voice agent stopped");
            });
        });

        Ok(event_rx)
    }

    /// Start the voice agent (stub for non-voice builds).
    #[cfg(not(feature = "voice"))]
    pub async fn start(&mut self) -> Result<mpsc::Receiver<VoiceAgentEvent>, String> {
        Err("Voice agent requires 'voice' feature".to_string())
    }

    /// Stop the voice agent.
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }

    /// Check if the voice agent is running.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::realtime::RealtimeEvent;

    /// Mock LLM processor that returns a fixed response and sends deltas.
    struct MockLlmProcessor {
        response: String,
    }

    impl MockLlmProcessor {
        fn new(response: &str) -> Self {
            Self {
                response: response.to_string(),
            }
        }
    }

    #[async_trait::async_trait]
    impl super::LlmProcessor for MockLlmProcessor {
        async fn process_text(
            &self,
            _text: &str,
            _session_key: &str,
            _language: Option<&str>,
            text_delta_tx: tokio::sync::mpsc::UnboundedSender<String>,
            _cancellation_token: Option<tokio_util::sync::CancellationToken>,
        ) -> String {
            // Send response as word-by-word deltas
            for word in self.response.split_whitespace() {
                let delta = format!("{} ", word);
                let _ = text_delta_tx.send(delta);
            }
            self.response.clone()
        }
    }

    /// Slow mock that respects cancellation, for barge-in tests.
    struct SlowMockLlmProcessor {
        response: String,
        delay_per_word: std::time::Duration,
    }

    impl SlowMockLlmProcessor {
        fn new(response: &str, delay_ms: u64) -> Self {
            Self {
                response: response.to_string(),
                delay_per_word: std::time::Duration::from_millis(delay_ms),
            }
        }
    }

    #[async_trait::async_trait]
    impl super::LlmProcessor for SlowMockLlmProcessor {
        async fn process_text(
            &self,
            _text: &str,
            _session_key: &str,
            _language: Option<&str>,
            text_delta_tx: tokio::sync::mpsc::UnboundedSender<String>,
            cancellation_token: Option<tokio_util::sync::CancellationToken>,
        ) -> String {
            let mut output = String::new();
            for word in self.response.split_whitespace() {
                if let Some(ref token) = cancellation_token {
                    if token.is_cancelled() {
                        break;
                    }
                }
                tokio::time::sleep(self.delay_per_word).await;
                let delta = format!("{} ", word);
                output.push_str(&delta);
                let _ = text_delta_tx.send(delta);
            }
            output.trim().to_string()
        }
    }

    #[tokio::test]
    async fn test_mock_llm_processor_sends_deltas() {
        let mock = MockLlmProcessor::new("Hello world from mock");
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();

        let result = mock
            .process_text("test input", "session1", Some("en"), tx, None)
            .await;

        assert_eq!(result, "Hello world from mock");

        // Collect all deltas
        let mut deltas = Vec::new();
        while let Ok(delta) = rx.try_recv() {
            deltas.push(delta);
        }
        assert_eq!(deltas.len(), 4); // "Hello ", "world ", "from ", "mock "
    }

    #[tokio::test]
    async fn test_slow_mock_respects_cancellation() {
        let mock = SlowMockLlmProcessor::new("word1 word2 word3 word4 word5", 50);
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let cancel = tokio_util::sync::CancellationToken::new();

        let cancel_clone = cancel.clone();
        // Cancel after 80ms (should get ~1-2 words)
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(80)).await;
            cancel_clone.cancel();
        });

        let result = mock
            .process_text("test", "s1", None, tx, Some(cancel))
            .await;

        // Should have been cancelled before completing all 5 words
        let word_count = result.split_whitespace().count();
        assert!(
            word_count < 5,
            "Expected cancellation before all 5 words, got {}",
            word_count
        );
        drop(rx);
    }

    #[test]
    fn test_listening_on_speech_start_emits_event() {
        let (state, actions) = next_state(&VoiceAgentState::Listening, &RealtimeEvent::SpeechStart);
        assert_eq!(state, VoiceAgentState::Listening);
        assert_eq!(
            actions,
            vec![VoiceAction::EmitEvent(VoiceAgentEvent::UserSpeechStart)]
        );
    }

    #[test]
    fn test_listening_on_turn_complete_transitions_to_processing() {
        let (state, actions) = next_state(
            &VoiceAgentState::Listening,
            &RealtimeEvent::TurnComplete {
                text: "hello".to_string(),
                language: "en".to_string(),
            },
        );
        assert_eq!(state, VoiceAgentState::Processing);
        assert!(
            actions.contains(&VoiceAction::EmitEvent(VoiceAgentEvent::UserText {
                text: "hello".to_string(),
                language: "en".to_string(),
            }))
        );
        assert!(actions.contains(&VoiceAction::StartLlm {
            text: "hello".to_string(),
            language: "en".to_string(),
        }));
    }

    #[test]
    fn test_speaking_on_speech_start_triggers_bargein() {
        let (state, actions) = next_state(&VoiceAgentState::Speaking, &RealtimeEvent::SpeechStart);
        assert_eq!(state, VoiceAgentState::Listening);
        assert!(actions.contains(&VoiceAction::CancelAll));
        assert!(actions.contains(&VoiceAction::EmitEvent(VoiceAgentEvent::UserSpeechStart)));
    }

    #[test]
    fn test_processing_on_speech_start_triggers_bargein() {
        let (state, actions) =
            next_state(&VoiceAgentState::Processing, &RealtimeEvent::SpeechStart);
        assert_eq!(state, VoiceAgentState::Listening);
        assert!(actions.contains(&VoiceAction::CancelAll));
    }

    #[test]
    fn test_listening_on_speech_end_emits_event() {
        let (state, actions) = next_state(&VoiceAgentState::Listening, &RealtimeEvent::SpeechEnd);
        assert_eq!(state, VoiceAgentState::Listening);
        assert_eq!(
            actions,
            vec![VoiceAction::EmitEvent(VoiceAgentEvent::UserSpeechEnd)]
        );
    }

    #[test]
    fn test_any_state_on_error_emits_error() {
        for state in [
            VoiceAgentState::Listening,
            VoiceAgentState::Processing,
            VoiceAgentState::Speaking,
        ] {
            let (new_state, actions) =
                next_state(&state, &RealtimeEvent::Error("test error".to_string()));
            assert_eq!(new_state, VoiceAgentState::Listening);
            assert_eq!(
                actions,
                vec![VoiceAction::EmitEvent(VoiceAgentEvent::Error(
                    "test error".to_string()
                ))]
            );
        }
    }

    #[test]
    fn test_speaking_on_tts_complete_returns_to_listening() {
        let (state, actions) = next_state(
            &VoiceAgentState::Speaking,
            &RealtimeEvent::TtsPlaybackComplete,
        );
        assert_eq!(state, VoiceAgentState::Listening);
        assert_eq!(
            actions,
            vec![VoiceAction::EmitEvent(VoiceAgentEvent::AudioComplete)]
        );
    }

    #[test]
    fn test_any_state_on_tts_error_returns_to_listening() {
        let (state, actions) = next_state(
            &VoiceAgentState::Speaking,
            &RealtimeEvent::Error("TTS synthesis failed".to_string()),
        );
        assert_eq!(state, VoiceAgentState::Listening);
        assert!(matches!(
            &actions[0],
            VoiceAction::EmitEvent(VoiceAgentEvent::Error(_))
        ));
    }

    #[tokio::test]
    async fn test_bargein_cancels_llm_during_processing() {
        use tokio::sync::mpsc;

        // Use SlowMockLlmProcessor so we have time to interrupt
        let slow_llm = SlowMockLlmProcessor::new(
            "This is a very long response that should get cancelled before completion",
            100, // 100ms per word
        );
        let (event_in_tx, event_in_rx) = mpsc::channel::<crate::realtime::RealtimeEvent>(32);
        let (event_out_tx, mut event_out_rx) = mpsc::channel::<VoiceAgentEvent>(64);

        let config = VoiceAgentConfig::default();

        // Start event loop
        let handle = tokio::spawn(super::run_voice_event_loop(
            config,
            Box::new(slow_llm),
            event_in_rx,
            event_out_tx,
        ));

        // Send first turn - this starts the slow LLM
        event_in_tx
            .send(crate::realtime::RealtimeEvent::TurnComplete {
                text: "first question".to_string(),
                language: "en".to_string(),
            })
            .await
            .unwrap();

        // Wait a bit for LLM to start producing deltas
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;

        // BARGE-IN: user starts speaking while LLM is still generating
        event_in_tx
            .send(crate::realtime::RealtimeEvent::SpeechStart)
            .await
            .unwrap();

        // Small delay for cancellation to propagate
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // Send a new turn (the barge-in speech)
        event_in_tx
            .send(crate::realtime::RealtimeEvent::TurnComplete {
                text: "actually nevermind".to_string(),
                language: "en".to_string(),
            })
            .await
            .unwrap();

        // Collect events with timeout
        let mut events = Vec::new();
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);
        let mut got_second_response = false;
        loop {
            tokio::select! {
                Some(evt) = event_out_rx.recv() => {
                    events.push(evt.clone());
                    // We're done when we see the second LlmResponseComplete
                    if let VoiceAgentEvent::LlmResponseComplete { .. } = evt {
                        if events.iter().filter(|e| matches!(e, VoiceAgentEvent::LlmResponseComplete { .. })).count() >= 2 {
                            got_second_response = true;
                            break;
                        }
                    }
                }
                _ = tokio::time::sleep_until(deadline) => {
                    break;
                }
            }
        }

        drop(event_in_tx);
        let _ = handle.await;

        // Verify: first LLM response was cut short (cancelled)
        let first_complete = events
            .iter()
            .find(|e| matches!(e, VoiceAgentEvent::LlmResponseComplete { .. }));
        assert!(
            first_complete.is_some(),
            "Should have at least one LlmResponseComplete"
        );

        if let Some(VoiceAgentEvent::LlmResponseComplete { full_text }) = first_complete {
            let word_count = full_text.split_whitespace().count();
            // The original response has 12 words, at 100ms/word it should be cancelled
            // well before all 12 are generated (250ms wait => ~2 words emitted)
            assert!(
                word_count < 12,
                "First response should have been cancelled early, got {} words: '{}'",
                word_count,
                full_text
            );
        }

        // Verify: barge-in event was emitted
        assert!(
            events
                .iter()
                .any(|e| matches!(e, VoiceAgentEvent::UserSpeechStart)),
            "Should have received UserSpeechStart from barge-in"
        );

        // Verify: second turn was processed
        assert!(events.iter().any(|e| matches!(e, VoiceAgentEvent::UserText { text, .. } if text == "actually nevermind")),
            "Should have received UserText for the barge-in turn");
    }

    #[tokio::test]
    async fn test_voice_agent_streams_tts_from_llm_deltas() {
        use tokio::sync::mpsc;

        let mock_llm = MockLlmProcessor::new("Hello there. How are you doing today?");
        let (event_in_tx, event_in_rx) = mpsc::channel::<crate::realtime::RealtimeEvent>(32);
        let (event_out_tx, mut event_out_rx) = mpsc::channel::<VoiceAgentEvent>(64);

        let config = VoiceAgentConfig::default();

        let handle = tokio::spawn(super::run_voice_event_loop(
            config,
            Box::new(mock_llm),
            event_in_rx,
            event_out_tx,
        ));

        // Inject turn
        event_in_tx
            .send(crate::realtime::RealtimeEvent::TurnComplete {
                text: "say something".to_string(),
                language: "en".to_string(),
            })
            .await
            .unwrap();

        // Collect events
        let mut events = Vec::new();
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(2);
        loop {
            tokio::select! {
                Some(evt) = event_out_rx.recv() => {
                    let is_complete = matches!(evt, VoiceAgentEvent::LlmResponseComplete { .. });
                    events.push(evt);
                    if is_complete { break; }
                }
                _ = tokio::time::sleep_until(deadline) => break,
            }
        }

        drop(event_in_tx);
        let _ = handle.await;

        // Verify multiple LlmTextDelta events were emitted
        let delta_count = events
            .iter()
            .filter(|e| matches!(e, VoiceAgentEvent::LlmTextDelta { .. }))
            .count();
        assert!(
            delta_count >= 2,
            "Expected multiple text deltas, got {}",
            delta_count
        );

        // Verify the sequence: UserText -> LlmResponseStart -> LlmTextDelta(s) -> LlmResponseComplete
        let event_types: Vec<&str> = events
            .iter()
            .map(|e| match e {
                VoiceAgentEvent::UserText { .. } => "UserText",
                VoiceAgentEvent::LlmResponseStart => "LlmResponseStart",
                VoiceAgentEvent::LlmTextDelta { .. } => "LlmTextDelta",
                VoiceAgentEvent::LlmResponseComplete { .. } => "LlmResponseComplete",
                _ => "Other",
            })
            .collect();

        // Find indices
        let user_text_idx = event_types.iter().position(|&t| t == "UserText");
        let response_start_idx = event_types.iter().position(|&t| t == "LlmResponseStart");
        let first_delta_idx = event_types.iter().position(|&t| t == "LlmTextDelta");
        let response_complete_idx = event_types
            .iter()
            .rposition(|&t| t == "LlmResponseComplete");

        assert!(
            user_text_idx < response_start_idx,
            "UserText should come before LlmResponseStart"
        );
        assert!(
            response_start_idx < first_delta_idx,
            "LlmResponseStart should come before first delta"
        );
        assert!(
            first_delta_idx < response_complete_idx,
            "First delta should come before LlmResponseComplete"
        );
    }

    #[tokio::test]
    async fn test_voice_agent_processes_turn_with_mock_llm() {
        use tokio::sync::mpsc;

        let mock_llm = MockLlmProcessor::new("I heard you clearly");
        let (event_in_tx, event_in_rx) = mpsc::channel::<crate::realtime::RealtimeEvent>(32);
        let (event_out_tx, mut event_out_rx) = mpsc::channel::<VoiceAgentEvent>(64);

        let config = VoiceAgentConfig::default();

        // Start event loop in background
        let handle = tokio::spawn(super::run_voice_event_loop(
            config,
            Box::new(mock_llm),
            event_in_rx,
            event_out_tx,
        ));

        // Inject a TurnComplete event
        event_in_tx
            .send(crate::realtime::RealtimeEvent::TurnComplete {
                text: "hello world".to_string(),
                language: "en".to_string(),
            })
            .await
            .unwrap();

        // Collect events with timeout
        let mut events = Vec::new();
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(2);
        loop {
            tokio::select! {
                Some(evt) = event_out_rx.recv() => {
                    events.push(evt);
                    // Check if we got LlmResponseComplete
                    if events.iter().any(|e| matches!(e, VoiceAgentEvent::LlmResponseComplete { .. })) {
                        break;
                    }
                }
                _ = tokio::time::sleep_until(deadline) => {
                    break;
                }
            }
        }

        // Drop sender to stop the loop
        drop(event_in_tx);
        let _ = handle.await;

        // Verify event sequence
        assert!(
            events.iter().any(
                |e| matches!(e, VoiceAgentEvent::UserText { text, .. } if text == "hello world")
            ),
            "Expected UserText with 'hello world', got: {:?}",
            events
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, VoiceAgentEvent::LlmResponseStart)),
            "Expected LlmResponseStart, got: {:?}",
            events
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, VoiceAgentEvent::LlmTextDelta { .. })),
            "Expected LlmTextDelta, got: {:?}",
            events
        );
        assert!(
            events.iter().any(|e| matches!(e, VoiceAgentEvent::LlmResponseComplete { full_text } if full_text == "I heard you clearly")),
            "Expected LlmResponseComplete with 'I heard you clearly', got: {:?}", events
        );
    }
}

#[cfg(all(test, feature = "voice"))]
mod voice_feature_tests {
    use super::*;

    #[test]
    fn test_voice_agent_config_default() {
        let config = VoiceAgentConfig::default();
        assert_eq!(config.session_key, "voice:default");
        assert_eq!(config.channel, "voice");
        assert_eq!(config.chat_id, "voice");
        assert!(!config.local);
        assert!(config.system_prompt.is_some());
    }

    #[test]
    fn test_voice_agent_config_custom() {
        let config = VoiceAgentConfig {
            realtime: super::super::RealtimeConfig {
                tts_engine: TtsEngineConfig::Kokoro,
                ..Default::default()
            },
            session_key: "my-session".to_string(),
            channel: "telegram".to_string(),
            chat_id: "chat-123".to_string(),
            local: true,
            system_prompt: None,
        };
        assert_eq!(config.realtime.tts_engine, TtsEngineConfig::Kokoro);
        assert_eq!(config.session_key, "my-session");
        assert!(config.local);
    }

    #[test]
    fn test_voice_agent_event_variants() {
        let _ = VoiceAgentEvent::UserSpeechStart;
        let _ = VoiceAgentEvent::UserSpeechEnd;
        let _ = VoiceAgentEvent::UserText {
            text: "hello".to_string(),
            language: "en".to_string(),
        };
        let _ = VoiceAgentEvent::LlmResponseStart;
        let _ = VoiceAgentEvent::LlmTextDelta {
            delta: "Hi".to_string(),
        };
        let _ = VoiceAgentEvent::LlmResponseComplete {
            full_text: "Hi there".to_string(),
        };
        let _ = VoiceAgentEvent::AudioChunk {
            samples: vec![],
            sample_rate: 24000,
        };
        let _ = VoiceAgentEvent::AudioComplete;
        let _ = VoiceAgentEvent::Error("test".to_string());
        let _ = VoiceAgentEvent::Ready;
    }

    #[test]
    fn test_voice_agent_new() {
        let config = VoiceAgentConfig::default();
        let agent = VoiceAgent::new(config);
        assert!(!agent.is_running());
    }

    #[tokio::test]
    async fn test_voice_agent_start_stop() {
        let config = VoiceAgentConfig {
            realtime: super::super::RealtimeConfig {
                tts_engine: TtsEngineConfig::Pocket,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut agent = VoiceAgent::new(config);

        let mut event_rx = agent.start().await.expect("Should start agent");
        assert!(agent.is_running());

        // Wait for Ready event
        let ready = tokio::time::timeout(std::time::Duration::from_secs(30), event_rx.recv()).await;
        assert!(
            matches!(ready, Ok(Some(VoiceAgentEvent::Ready))),
            "Should receive Ready event"
        );

        agent.stop();
        assert!(!agent.is_running());
    }
}
