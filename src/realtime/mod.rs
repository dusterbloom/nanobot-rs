//! Realtime voice session support for nanobot.
//!
//! Provides bidirectional streaming audio with VAD-based turn detection,
//! STT transcription, and TTS synthesis for realtime voice conversations.
//!
//! Uses jack-voice's VAD (Silero), TurnDetector (SmartTurn v3), and
//! audio capture/playback directly.

mod session;
mod voice_agent;
mod ws_server;

pub use session::{RealtimeConfig, RealtimeSession, RealtimeEvent};
pub use voice_agent::{VoiceAgent, VoiceAgentConfig, VoiceAgentEvent};
pub use ws_server::{RealtimeServer, RealtimeServerConfig};
