//! nanobot library — exposes internal modules for the trio_bench binary.

pub mod agent;
#[cfg(feature = "cluster")]
pub mod cluster;
pub mod bus;
pub mod channels;
pub mod config;
pub mod cron;
pub mod errors;
pub mod heartbeat;
pub mod lms;
#[cfg(feature = "voice")]
pub mod realtime;
pub mod providers;
pub mod server;
pub mod session;
pub mod tui;
pub mod utils;
#[cfg(feature = "voice")]
pub mod voice;
#[cfg(feature = "voice")]
pub mod voice_pipeline;
