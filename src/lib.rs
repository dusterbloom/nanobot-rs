//! nanobot library — exposes internal modules for the trio_bench binary.

pub mod agent;
pub mod bus;
pub mod channels;
#[cfg(feature = "cluster")]
pub mod cluster;
pub mod config;
pub mod cron;
pub mod errors;
pub mod heartbeat;
pub mod lms;
pub mod providers;
#[cfg(feature = "voice")]
pub mod realtime;
pub mod searxng;
pub mod server;
pub mod session;
pub mod tui;
pub mod utils;
#[cfg(feature = "voice")]
pub mod voice_pipeline;
