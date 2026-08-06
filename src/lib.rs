#![deny(unsafe_code)]
// Test builds get the sanctioned escape hatch (research doc §3.6): the
// production deny regime (unwrap/expect/panic/indexing/as_conversions/...)
// applies to `cargo clippy` lib/bin builds; test modules may keep their
// pragmatic unwraps without blocking the flip.
#![cfg_attr(test, allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::todo,
    clippy::unimplemented,
    clippy::dbg_macro,
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::shadow_reuse,
    clippy::shadow_unrelated,
    clippy::shadow_same,
    clippy::string_add,
    clippy::format_push_string,
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::pedantic,
    clippy::nursery,
))]
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
pub mod crw;
pub mod errors;
pub mod heartbeat;
pub(crate) mod higgs;
pub mod lms;
pub(crate) mod local_discovery;
pub mod providers;
pub(crate) mod repl;
pub mod searxng;
pub mod server;
pub mod session;
pub(crate) mod sessions_cmd;
pub(crate) mod syntax;
pub mod tui;
pub(crate) mod tui_app;
pub(crate) mod turn_stream;
pub mod utils;
#[cfg(feature = "voice")]
pub mod voice_pipeline;

mod app;

pub use app::run;
