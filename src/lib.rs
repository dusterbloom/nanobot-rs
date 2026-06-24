#![deny(unsafe_code)]
//! nanobot library — exposes internal modules for the trio_bench binary.

pub(crate) const VERSION: &str = "0.1.0";
pub(crate) const LOGO: &str = "*";

pub mod agent;
pub mod bus;
pub mod channels;
pub(crate) mod cli;
#[cfg(feature = "cluster")]
pub mod cluster;
pub mod config;
pub mod cron;
pub mod errors;
pub mod heartbeat;
pub(crate) mod higgs;
pub mod lms;
pub mod providers;
pub(crate) mod repl;
pub mod searxng;
pub mod server;
pub mod session;
pub(crate) mod sessions_cmd;
pub(crate) mod syntax;
pub mod tui;
pub(crate) mod tui_app;
pub mod utils;
#[cfg(feature = "voice")]
pub mod voice_pipeline;

mod app;

pub use app::run;
