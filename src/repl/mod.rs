//! REPL loop and interactive command dispatch for `nanobot agent`.
//!
//! Contains the main agent REPL, slash-command handlers, voice recording
//! pipeline, and background channel management.

pub(crate) mod commands;
mod incremental;

pub(crate) use commands::{should_auto_activate_trio, trio_enable};

use std::collections::BTreeSet;
use std::io::{self, IsTerminal, Write as _};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use rustyline::error::ReadlineError;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use crate::agent::agent_loop::SharedCoreHandle;
use crate::agent::audit::{AuditLog, ToolEvent};
use crate::agent::provenance::{ClaimStatus, ClaimVerifier};
use crate::agent::reflector::Reflector;
use crate::cli;
use crate::config::loader::{get_data_dir, load_config, save_config};
use crate::config::schema::Config;
use crate::cron::service::CronService;
use crate::heartbeat::service::{
    HeartbeatService, DEFAULT_HEARTBEAT_INTERVAL_S, DEFAULT_MAINTENANCE_COMMANDS,
};
use crate::syntax;
use crate::tui;
use crate::turn_stream::{Completion, TurnEvent, TurnStream};

// ============================================================================
// Streaming TTS type (feature-gated)
// ============================================================================

#[cfg(feature = "voice")]
type TtsSentenceSender = Option<std::sync::mpsc::Sender<crate::voice_pipeline::TtsCommand>>;
#[cfg(not(feature = "voice"))]
type TtsSentenceSender = Option<()>;

// ============================================================================
// Helpers (testable, pure-ish)
// ============================================================================

// Only used by the truncation unit tests below now that tool output is collapsed.
#[cfg(test)]
use crate::utils::helpers::truncate_lines_chars as truncate_output;

/// ANSI escape to rewind `n` rows and clear everything below the cursor,
/// without emitting newlines. Used to overwrite a previously-rendered block
/// in place — e.g. rewriting raw user input as a styled box, or replacing a
/// stale tool-call status when the same tool emits a new `CallEnd`.
///
/// An earlier implementation looped `"\x1b[2K\r\n"` per row, which scrolls
/// the terminal when the cursor sits on the last row — migrating the old
/// block into scrollback instead of overwriting it in place.
fn rewind_and_clear_below(n: usize) -> String {
    if n == 0 {
        return String::new();
    }
    // CSI n A  — cursor up n rows
    // CSI   J  — erase from cursor to end of screen
    format!("\x1b[{}A\x1b[J", n)
}

/// Prefill spinner: the otherwise-silent wait before a model's first token.
///
/// `Disabled` when stdout is not a terminal. `Idle` between prefills —
/// cleared by the first streamed output, re-armed by the next LLM call's
/// progress markers (tool loops prefill once per call). While `Active`, a
/// 100ms ticker redraws elapsed time, upgraded to a true percentage when the
/// server streams `prompt_progress` (higgs/llama.cpp `return_progress`).
enum PrefillSpinner {
    Disabled,
    Idle,
    Active {
        started: std::time::Instant,
        progress: Option<(u64, u64)>,
    },
}

impl PrefillSpinner {
    /// Terminal-gated constructor: the spinner renders only on real TTYs.
    fn new() -> Self {
        if std::io::stdout().is_terminal() {
            Self::Idle
        } else {
            Self::Disabled
        }
    }

    fn is_active(&self) -> bool {
        matches!(self, Self::Active { .. })
    }

    /// Show the spinner at the start of the blind wait. No-op unless Idle.
    fn start(&mut self) {
        if matches!(self, Self::Idle) {
            *self = Self::Active {
                started: std::time::Instant::now(),
                progress: None,
            };
            self.redraw();
        }
    }

    /// Record server-reported prefill progress and redraw. Re-arms an Idle
    /// spinner so later calls' prefill waits become visible too.
    fn on_progress(&mut self, processed: u64, total: u64) {
        match self {
            Self::Disabled => {}
            Self::Idle => {
                *self = Self::Active {
                    started: std::time::Instant::now(),
                    progress: Some((processed, total)),
                };
                self.redraw();
            }
            Self::Active { progress, .. } => {
                *progress = Some((processed, total));
                self.redraw();
            }
        }
    }

    /// Repaint the spinner line in place. No-op unless Active.
    /// `prefill_line` styles itself (animated mark + dim time), so no dim wrap.
    fn redraw(&self) {
        if let Self::Active { started, progress } = self {
            print!(
                "\r\x1b[K{}",
                prefill_line(started.elapsed().as_secs_f32(), *progress)
            );
            std::io::stdout().flush().ok();
        }
    }

    /// Erase the spinner the moment real output arrives. Active → Idle;
    /// no-op otherwise, so callers can invoke it unconditionally.
    fn clear(&mut self) {
        if self.is_active() {
            print!("\r\x1b[K");
            std::io::stdout().flush().ok();
            *self = Self::Idle;
        }
    }
}

/// Control-marker wire protocol — owned by `turn_stream` (single home for
/// both encode and parse); re-exported here for the many existing users.
pub(crate) use crate::turn_stream::{
    parse_control_marker, CacheResetReason, CacheStatus, ControlMarker,
};

#[cfg(any(test, feature = "voice"))]
fn thinking_delta_should_skip_tts(delta: &str, in_thinking: &mut bool) -> bool {
    if delta.starts_with("\x1b[90m\x1b[2m") {
        *in_thinking = true;
        return true;
    }
    if *in_thinking {
        if delta.starts_with("\x1b[0m") {
            *in_thinking = false;
        }
        return true;
    }
    false
}

#[cfg(test)]
mod thinking_tts_tests {
    use super::*;

    #[test]
    fn tts_filter_skips_ansi_delimited_thinking_only() {
        let mut in_thinking = false;

        assert!(thinking_delta_should_skip_tts(
            "\x1b[90m\x1b[2m",
            &mut in_thinking
        ));
        assert!(in_thinking);
        assert!(thinking_delta_should_skip_tts(
            "private thought",
            &mut in_thinking
        ));
        assert!(thinking_delta_should_skip_tts(
            "\x1b[0m\n\n",
            &mut in_thinking
        ));
        assert!(!in_thinking);
        assert!(!thinking_delta_should_skip_tts(
            "visible answer",
            &mut in_thinking
        ));
    }
}

/// Animate the brand mark `▞▞▞` as the loading indicator: the lit block sweeps
/// across the three cells. The same mark prefixes the reply, so the animation
/// "settles" into the answer. Tied to the prefill progress we track: shows `%`
/// when the server reports it, otherwise just elapsed.
fn prefill_line(elapsed_secs: f32, progress: Option<(u64, u64)>) -> String {
    const GLYPH: &str = "\u{259e}"; // ▞
    let active = ((elapsed_secs * 6.0) as usize) % 3;
    let mut mark = String::new();
    for i in 0..3 {
        if i == active {
            mark.push_str(&format!("\x1b[1m\x1b[36m{GLYPH}\x1b[0m")); // bright
        } else {
            mark.push_str(&format!("\x1b[2m\x1b[36m{GLYPH}\x1b[0m")); // dim
        }
    }
    match progress {
        Some((processed, total)) if total > 0 => format!(
            "{mark} \x1b[2m{}% \u{b7} {:.1}s\x1b[0m",
            processed * 100 / total,
            elapsed_secs
        ),
        _ => format!("{mark} \x1b[2m{:.1}s\x1b[0m", elapsed_secs),
    }
}

/// Extract a string field value from a (possibly truncated) JSON object string.
///
/// `arguments_preview` is clipped to a fixed width, so the JSON may not parse.
/// Try a real parse first; on failure, scan for `"field"` and read the quoted
/// value that follows, honoring `\"`/`\n`/`\t` escapes.
fn extract_json_string_field(json_ish: &str, field: &str) -> Option<String> {
    if let Ok(serde_json::Value::Object(map)) = serde_json::from_str::<serde_json::Value>(json_ish)
    {
        if let Some(v) = map.get(field).and_then(|v| v.as_str()) {
            return Some(v.to_string());
        }
    }
    let key = format!("\"{}\"", field);
    let start = json_ish.find(&key)? + key.len();
    let after_colon = json_ish[start..].trim_start().strip_prefix(':')?;
    let body = after_colon.trim_start().strip_prefix('"')?;
    let mut out = String::new();
    let mut chars = body.chars();
    while let Some(c) = chars.next() {
        match c {
            '\\' => match chars.next() {
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some(other) => out.push(other),
                None => break,
            },
            '"' => break,
            other => out.push(other),
        }
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

/// Extract a short context label for a tool's persistent status line.
///
/// `args_preview` is the JSON arguments captured at `CallStart` (the result
/// data alone doesn't contain the command/path). For `read_file`/`edit_file`
/// the path comes from `result_data`; for `exec` the command comes from
/// `args_preview`. Returns "" when no useful label can be extracted.
fn extract_tool_context(tool_name: &str, result_data: &str, args_preview: &str) -> String {
    match tool_name {
        "read_file" => {
            // Result starts with "# /path/to/file.rs (lines N-M of T)"
            if let Some(line) = result_data.lines().next() {
                if let Some(rest) = line.strip_prefix("# ") {
                    // Extract just the filename, not the full path
                    if let Some(paren) = rest.find(" (") {
                        let path = &rest[..paren];
                        // Show just filename or last 2 components
                        let short: String = path
                            .rsplit('/')
                            .take(2)
                            .collect::<Vec<_>>()
                            .into_iter()
                            .rev()
                            .collect::<Vec<_>>()
                            .join("/");
                        return short;
                    }
                }
            }
            // Fallback: check for error messages
            if result_data.starts_with("Error:") {
                let preview: String = result_data.chars().take(60).collect();
                return preview;
            }
            String::new()
        }
        "exec" => {
            // The command lives in the arguments, not the output. Show it
            // (first line, clipped) so the persistent line reads `$ <command>`.
            let Some(cmd) = extract_json_string_field(args_preview, "command") else {
                return String::new();
            };
            let one_line = cmd.trim().lines().next().unwrap_or("").trim();
            if one_line.is_empty() {
                return String::new();
            }
            let short: String = one_line.chars().take(80).collect();
            if short.chars().count() < one_line.chars().count() {
                format!("$ {}…", short)
            } else {
                format!("$ {}", short)
            }
        }
        "edit_file" => {
            if result_data.contains("Successfully edited") {
                if let Some(path) = result_data.strip_prefix("Successfully edited ") {
                    let path = path.trim();
                    let short: String = path
                        .rsplit('/')
                        .take(2)
                        .collect::<Vec<_>>()
                        .into_iter()
                        .rev()
                        .collect::<Vec<_>>()
                        .join("/");
                    return short;
                }
            }
            String::new()
        }
        "web_search" | "session_search" | "recall" => {
            clip_param(extract_json_string_field(args_preview, "query"))
        }
        "web_fetch" => clip_param(extract_json_string_field(args_preview, "url")),
        "read_skill" | "spawn" => clip_param(extract_json_string_field(args_preview, "name")),
        "list_dir" | "write_file" => clip_param(extract_json_string_field(args_preview, "path")),
        _ => {
            // Generic: surface the first recognisable string argument so the tool
            // input is always visible on the line.
            for key in ["query", "path", "url", "name", "command", "input", "text"] {
                if let Some(v) = extract_json_string_field(args_preview, key) {
                    return clip_param(Some(v));
                }
            }
            String::new()
        }
    }
}

/// Clip a parameter value to one short line for the tool status display.
fn clip_param(value: Option<String>) -> String {
    let Some(v) = value else { return String::new() };
    let one_line = v.trim().lines().next().unwrap_or("").trim();
    let short: String = one_line.chars().take(64).collect();
    if short.chars().count() < one_line.chars().count() {
        format!("{short}…")
    } else {
        short
    }
}

/// Extract the host from an endpoint URL like `"http://192.168.1.22:1234/v1"`.
///
/// Returns an empty string on parse failure (caller should fall back to
/// `api_host()` or `"127.0.0.1"`).
pub(crate) fn extract_url_host(url: &str) -> String {
    // Strip scheme
    let without_scheme = url
        .trim_start_matches("https://")
        .trim_start_matches("http://");
    // Host is everything before the first ':' (port separator)
    let host = without_scheme.split(':').next().unwrap_or("").trim();
    if host.is_empty() || host == "localhost" {
        // Treat "localhost" as empty so callers fall back to api_host()
        // which resolves the WSL2 Windows host IP when needed.
        String::new()
    } else {
        host.to_string()
    }
}

/// INFO-log one interactive-startup phase: ms since the previous phase and
/// since process entry. Greppable as `startup_phase` in ~/.nanobot/logs.
fn log_startup_phase(phase: &str, t0: std::time::Instant, last: &mut std::time::Instant) {
    let now = std::time::Instant::now();
    info!(
        phase,
        phase_ms = now.duration_since(*last).as_millis() as u64,
        total_ms = now.duration_since(t0).as_millis() as u64,
        "startup_phase"
    );
    *last = now;
}

async fn prewarm_remote_lms_models(config: &Config, main_model: &str) {
    let base = config.agents.defaults.local_api_base.trim();
    if base.is_empty() {
        return;
    }
    let native = base.trim_end_matches('/').trim_end_matches("/v1");
    let url = format!("{}/api/v1/models/load", native);

    let mut models: Vec<(String, Option<usize>)> = Vec::new();
    if !main_model.trim().is_empty() {
        models.push((
            main_model.trim().to_string(),
            Some(config.agents.defaults.local_max_context_tokens),
        ));
    }

    let role_models_enabled = config.trio.enabled;

    if role_models_enabled {
        if !config.trio.router_model.trim().is_empty() {
            models.push((
                config.trio.router_model.trim().to_string(),
                Some(config.trio.router_ctx_tokens),
            ));
        }
        if !config.trio.specialist_model.trim().is_empty() {
            models.push((
                config.trio.specialist_model.trim().to_string(),
                Some(config.trio.specialist_ctx_tokens),
            ));
        }
    }

    if config.lcm.is_enabled() {
        if let Some(ref ep) = config.lcm.compaction_endpoint {
            if !ep.model.trim().is_empty() {
                models.push((
                    ep.model.trim().to_string(),
                    Some(config.lcm.compaction_context_size),
                ));
            }
        }
    }

    // Query already-loaded models so we can skip redundant loads
    let models_url = format!("{}/api/v1/models", native);
    let client = reqwest::Client::new();
    let api_key = &config.agents.defaults.local_api_key;
    let loaded_map: std::collections::HashMap<String, Option<usize>> = match client
        .get(&models_url)
        .header("Authorization", format!("Bearer {}", api_key))
        .timeout(std::time::Duration::from_secs(5))
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => {
            let json: serde_json::Value = resp.json().await.unwrap_or_default();
            json.get("models")
                .and_then(|m| m.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|m| {
                            let key = m.get("key")?.as_str()?.to_string();
                            let instances = m.get("loaded_instances")?.as_array()?;
                            if instances.is_empty() {
                                return None;
                            }
                            let ctx = instances
                                .first()
                                .and_then(|inst| inst.get("config"))
                                .and_then(|c| c.get("context_length"))
                                .and_then(|v| v.as_u64())
                                .map(|n| n as usize);
                            Some((key, ctx))
                        })
                        .collect()
                })
                .unwrap_or_default()
        }
        _ => std::collections::HashMap::new(),
    };

    let mut seen = BTreeSet::new();
    for (model, ctx) in models {
        if !seen.insert(model.clone()) {
            continue;
        }
        // Skip if model is already loaded on the remote — don't force context reload
        if loaded_map
            .keys()
            .any(|k| crate::lms::model_matches(k, &model))
        {
            info!(model = %model, "remote_lms_prewarm_already_loaded");
            continue;
        }
        let mut body = serde_json::json!({ "model": model });
        if let Some(c) = ctx {
            body["context_length"] = serde_json::json!(c);
        }
        match client.post(&url).json(&body).send().await {
            Ok(resp) if resp.status().is_success() => {
                info!(model = %body["model"].as_str().unwrap_or(""), "remote_lms_prewarm_ok");
            }
            Ok(resp) => {
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                warn!(model = %body["model"].as_str().unwrap_or(""), %status, body = %text, "remote_lms_prewarm_failed");
            }
            Err(e) => {
                warn!(model = %body["model"].as_str().unwrap_or(""), error = %e, "remote_lms_prewarm_error");
            }
        }
    }
}

/// Default Higgs keepalive interval (seconds). Short enough to keep the 35B
/// resident across read/think pauses, long enough to be negligible compute.
const DEFAULT_HIGGS_KEEPALIVE_SECS: u64 = 45;

/// Decide whether (and how often) to run the Higgs warm-keep ping.
///
/// Returns `Some(secs)` only for *our own* local Higgs sidecar
/// (localhost/127.0.0.1) — we never keep a remote peer warm. The env var
/// `NANOBOT_HIGGS_KEEPALIVE_SECS` overrides the interval; `0` disables it.
/// A non-numeric value falls back to the default.
pub(crate) fn higgs_keepalive_secs(
    backend: &str,
    api_base: &str,
    env: Option<&str>,
) -> Option<u64> {
    if !crate::config::schema::is_higgs_backend(backend) {
        return None;
    }
    let lower = api_base.trim().to_lowercase();
    let host = lower
        .strip_prefix("http://")
        .or_else(|| lower.strip_prefix("https://"))
        .unwrap_or(&lower);
    let is_local = host.starts_with("127.0.0.1:") || host.starts_with("localhost:");
    if !is_local {
        return None;
    }
    match env {
        Some(v) => match v.trim().parse::<u64>() {
            Ok(0) => None,
            Ok(n) => Some(n),
            Err(_) => Some(DEFAULT_HIGGS_KEEPALIVE_SECS),
        },
        None => Some(DEFAULT_HIGGS_KEEPALIVE_SECS),
    }
}

/// Keep a local Higgs model resident by sending a 1-token completion whenever
/// the REPL is idle.
///
/// Why: an idle gap lets the OS evict the 35B's weights, so the next real turn
/// pays a multi-second cold reload (observed 37–67 s TTFT vs ~3 s warm). A tiny
/// periodic inference touches the weights so they stay hot, and surfaces a
/// crash early via a WARN. The 1-token prompt cannot evict the user's cached
/// prefix from Higgs's radix cache (it's a separate, negligible branch).
///
/// Skips a tick while a real request is in flight (`inference_active`) — that
/// request is already keeping the model warm.
fn spawn_higgs_keepalive(
    api_base: String,
    api_key: String,
    model: String,
    interval_s: u64,
    inference_active: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let client = reqwest::Client::new();
        let url = format!("{}/chat/completions", api_base.trim_end_matches('/'));
        let body = serde_json::json!({
            "model": model,
            "messages": [{"role": "user", "content": "."}],
            "max_tokens": 1,
            "temperature": 0.0,
            "stream": false,
        });
        loop {
            tokio::time::sleep(Duration::from_secs(interval_s)).await;
            if stop.load(Ordering::Relaxed) {
                break;
            }
            if inference_active.load(Ordering::Relaxed) {
                continue; // a real request is already keeping it warm
            }
            let started = Instant::now();
            match client
                .post(&url)
                .header("Authorization", format!("Bearer {}", api_key))
                .json(&body)
                .timeout(Duration::from_secs(60))
                .send()
                .await
            {
                Ok(r) if r.status().is_success() => {
                    debug!(
                        ms = started.elapsed().as_millis() as u64,
                        "higgs_keepalive_ok"
                    );
                }
                Ok(r) => warn!(status = %r.status(), "higgs_keepalive_unexpected_status"),
                Err(e) => warn!(error = %e, "higgs_keepalive_failed (Higgs may be down)"),
            }
        }
        debug!("higgs_keepalive stopped");
    })
}

/// Parse a `/ctx` argument into a byte count.
///
/// Accepts:
/// - `""` (empty) → None (means auto-detect)
/// - `"32768"` → Some(32768)
/// - `"32K"` or `"32k"` → Some(32768)
/// - Values < 2048 → Err
/// - Non-numeric → Err
pub(crate) fn parse_ctx_arg(arg: &str) -> Result<Option<usize>, &'static str> {
    let s = arg.trim();
    if s.is_empty() {
        return Ok(None);
    }
    let lower = s.to_lowercase();
    let n = if let Some(prefix) = lower.strip_suffix('k') {
        prefix
            .parse::<usize>()
            .map(|n| n * 1024)
            .map_err(|_| "invalid number")?
    } else {
        lower.parse::<usize>().map_err(|_| "invalid number")?
    };
    if n < 2048 {
        return Err("minimum context size is 2048");
    }
    // Round down to nearest 1024
    Ok(Some((n / 1024) * 1024))
}

/// Shorten channel names for status display.
pub(crate) fn short_channel_name(name: &str) -> &str {
    match name {
        "whatsapp" => "wa",
        "telegram" => "tg",
        other => other,
    }
}

/// Build the REPL prompt string based on current mode.
pub(crate) fn build_prompt(_is_local: bool, _voice_on: bool, _thinking_on: bool) -> String {
    use crate::tui::{BOLD, GREEN, RESET};
    // One consistent prompt across all modes — mode/model already shows in the
    // footer, so the prompt stays minimal (a single green caret). `\x1b[48;5;238m`
    // is the input-box background, kept on after the caret so typed text sits
    // inside the box.
    const BG: &str = "\x1b[48;5;238m";
    format!("{BG} {BOLD}{GREEN}\u{276f}{RESET}{BG} ")
}

/// Print the /help text.
pub(crate) fn print_help() {
    println!("\nCommands:");
    println!("  /local, /l      - Toggle between local and cloud mode");
    println!("  /model, /m [q]  - Pick model from all sources (LMS, cluster, ~/models/)");
    println!("  /lane           - Toggle lane (answer/action) or /lane answer|action");
    println!("  /trio           - Toggle trio mode (router + specialist)");
    println!("  /trio budget    - Show VRAM budget breakdown");
    println!("  /trio cap <GB>  - Set VRAM cap (e.g. /trio cap 12)");
    println!("  /ctx [size]     - Set context size (e.g. /ctx 32K) or auto-detect");
    println!("  /think, /t      - Toggle extended thinking (/thinking on|off|N)");
    println!("  /nothink, /nt   - Disable extended thinking");
    println!("  /long           - Set large output budget (/long on|off|N)");
    println!("  /voice, /v      - Toggle voice mode (Ctrl+Space or Enter to speak)");
    println!("  /whatsapp, /wa  - Start WhatsApp channel (runs alongside chat)");
    println!("  /telegram, /tg  - Start Telegram channel (runs alongside chat)");
    println!("  /email          - Start Email channel (runs alongside chat)");

    println!("  /stop           - Stop all running channels");
    println!("  /agents, /a     - List running background agents");
    println!("  /kill <id>      - Cancel a background agent");
    println!("  /status, /s     - Show current mode, model, and channel info");
    println!("  /context        - Show context breakdown (tokens, messages, memory)");
    println!("  /memory         - Show working memory for current session");
    println!("  /learn          - Distill accumulated sessions into MEMORY.md now");
    println!("  /clear, /c      - Clear working memory for current session");
    println!("  /replay         - Show session message history (/replay full | /replay N)");
    println!("  /restart, /rd   - Restart local servers (or delegation in cloud mode)");
    println!("  /sessions, /ss  - Session management (list, export, purge, archive, index)");
    println!("  /audit          - Show audit log for current session");
    println!("  /verify         - Re-verify claims in last response");
    println!("  /provenance     - Toggle provenance display on/off");
    println!("  /cluster, /cl   - Show cluster peers, models, and routing status");
    println!("  /skill, /sk     - Manage skills (list, find, add, remove)");
    println!("  /help, /h       - Show this help");
    println!("  Ctrl+C          - Exit\n");
}

// ============================================================================
// Input Watcher (Full-Duplex REPL)
// ============================================================================

/// Spawn a key watcher thread that runs during agent streaming/tool execution.
///
/// Handles:
/// - **Enter**: cancel via `cancel_token` + set `enter_interrupted` flag.
///   In voice mode this signals "start recording"; in text mode it's a fast cancel.
/// - **ESC+ESC** (within 500ms): instant cancel via `cancel_token`
/// - **Ctrl+C**: backup cancel via `cancel_token`
/// - **Backtick (`)**: temporarily exits raw mode, reads an injection line
///   from stdin, sends it through `inject_tx`, re-enters raw mode
///
/// The thread exits when `done` is set to `true`.
///
/// Modeled on `tui::spawn_interrupt_watcher()` (voice mode pattern).
pub(crate) fn spawn_input_watcher(
    cancel_token: tokio_util::sync::CancellationToken,
    inject_tx: tokio::sync::mpsc::UnboundedSender<String>,
    done: Arc<AtomicBool>,
    enter_interrupted: Arc<AtomicBool>,
) -> std::thread::JoinHandle<()> {
    use termimad::crossterm::event::{self, Event, KeyCode, KeyModifiers};

    std::thread::spawn(move || {
        let owned = tui::enter_raw_mode();
        debug!("input_watcher: started, raw_mode_owned={}", owned);
        let mut last_esc: Option<Instant> = None;
        let mut poll_cycles = 0u32;

        while !done.load(Ordering::Relaxed) {
            poll_cycles += 1;
            if poll_cycles % 50 == 0 {
                debug!("input_watcher: alive, poll_cycles={}", poll_cycles);
            }
            if event::poll(Duration::from_millis(100)).unwrap_or(false) {
                if let Ok(Event::Key(key)) = event::read() {
                    // Enter → cancel + signal "user wants to input/record"
                    if key.code == KeyCode::Enter {
                        enter_interrupted.store(true, Ordering::Relaxed);
                        debug!("input_watcher: key=Enter, cancelling");
                        cancel_token.cancel();
                        break;
                    }

                    // Ctrl+C → cancel
                    if key.code == KeyCode::Char('c')
                        && key.modifiers.contains(KeyModifiers::CONTROL)
                    {
                        debug!("input_watcher: key=Ctrl+C, cancelling");
                        cancel_token.cancel();
                        break;
                    }

                    // ESC double-tap → cancel
                    if key.code == KeyCode::Esc {
                        if let Some(prev) = last_esc {
                            if prev.elapsed() < Duration::from_millis(2000) {
                                debug!("input_watcher: key=Esc+Esc, cancelling");
                                cancel_token.cancel();
                                break;
                            }
                        }
                        last_esc = Some(Instant::now());
                        continue;
                    }

                    // Backtick → inject prompt
                    if key.code == KeyCode::Char('`')
                        && !key.modifiers.contains(KeyModifiers::CONTROL)
                        && !key.modifiers.contains(KeyModifiers::ALT)
                    {
                        // Exit raw mode so the user gets normal line editing.
                        tui::exit_raw_mode(owned);
                        print!("\n\x1b[33minject>\x1b[0m ");
                        io::stdout().flush().ok();

                        let mut line = String::new();
                        if io::stdin().read_line(&mut line).is_ok() {
                            let trimmed = line.trim().to_string();
                            if !trimmed.is_empty() {
                                let _ = inject_tx.send(trimmed);
                            }
                        }

                        // Re-enter raw mode for continued watching.
                        // Note: we don't update `owned` here because we already own the mode.
                        tui::enter_raw_mode();
                        continue;
                    }

                    // Any other key clears the ESC state.
                    last_esc = None;
                }
            }
        }

        debug!(
            "input_watcher: exiting, done={}",
            done.load(Ordering::Relaxed)
        );
        tui::exit_raw_mode(owned);
    })
}

/// Stream an LLM response with live delta printing, then erase and re-render with syntax highlighting.
///
/// This replaces the 3x copy-pasted pattern:
///   create delta channel → spawn print task → stream → await → erase raw → re-render
///
/// When provenance is enabled, tool call events are displayed during streaming
/// and claim verification is applied to the final render.
///
/// Returns the full response text.
pub(crate) async fn stream_and_render(
    agent_loop: &mut crate::agent::agent_loop::AgentLoop,
    input: &str,
    session_id: &str,
    channel: &str,
    lang: Option<&str>,
    core_handle: &SharedCoreHandle,
) -> String {
    stream_and_render_inner(
        agent_loop,
        input,
        session_id,
        channel,
        lang,
        core_handle,
        false,
        None,
    )
    .await
    .0
}

/// Like `stream_and_render` but skips the user text erase-and-reprint.
/// Use when the caller has already rendered the user turn (e.g. voice recording).
///
/// Returns `(response_text, enter_interrupted)`. When `enter_interrupted` is true,
/// the user pressed Enter to cancel — the voice loop should skip TTS and start recording.
#[cfg(feature = "voice")]
pub(crate) async fn stream_and_render_voice(
    agent_loop: &mut crate::agent::agent_loop::AgentLoop,
    input: &str,
    session_id: &str,
    channel: &str,
    lang: Option<&str>,
    core_handle: &SharedCoreHandle,
    tts_sentence_tx: Option<std::sync::mpsc::Sender<crate::voice_pipeline::TtsCommand>>,
) -> (String, bool) {
    stream_and_render_inner(
        agent_loop,
        input,
        session_id,
        channel,
        lang,
        core_handle,
        true,
        tts_sentence_tx,
    )
    .await
}

async fn stream_and_render_inner(
    agent_loop: &mut crate::agent::agent_loop::AgentLoop,
    input: &str,
    session_id: &str,
    channel: &str,
    lang: Option<&str>,
    core_handle: &SharedCoreHandle,
    user_already_rendered: bool,
    tts_tx: TtsSentenceSender,
) -> (String, bool) {
    // Render the user turn into the conversation area, ABOVE the pinned input
    // box. The readline echo on the (pinned) prompt row was already cleared by
    // the caller; we drop to the bottom of the scroll region so the user box and
    // the reply scroll up into history while the input box stays put.
    if !user_already_rendered && std::io::stdout().is_terminal() {
        use std::io::Write as _;
        print!("\x1b[{};1H", tui::conversation_bottom_row());
        std::io::stdout().flush().ok();
        print!("{}", syntax::render_turn(input, syntax::TurnRole::User));
    }

    // Check provenance config for tool event display and claim verification.
    let (show_tool_calls, verify_claims, strict_mode, workspace) = {
        let core = core_handle.swappable();
        (
            core.provenance_config.enabled && core.provenance_config.show_tool_calls,
            core.provenance_config.enabled && core.provenance_config.verify_claims,
            core.provenance_config.strict_mode,
            core.workspace.clone(),
        )
    };

    let (delta_tx, delta_rx) = tokio::sync::mpsc::unbounded_channel::<String>();

    // Create the tool-event channel when provenance display is on (text REPL)
    // OR when streaming to TTS — voice mode needs CallStart events to narrate
    // tool actions aloud. In non-voice builds `tts_tx` is always None, so this
    // reduces to the provenance condition.
    let want_tool_events = show_tool_calls || tts_tx.is_some();
    let (tool_rx_opt, tool_event_tx) = if want_tool_events {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<ToolEvent>();
        (Some(rx), Some(tx))
    } else {
        (None, None)
    };

    #[cfg(feature = "voice")]
    let tts_lang_override = lang
        .map(str::trim)
        .filter(|s| !s.is_empty() && !s.eq_ignore_ascii_case("auto"))
        .map(str::to_string);

    // Breathing room before the streamed answer. Printed here (cooked mode,
    // before the input watcher enters raw mode and before the print task is
    // scheduled) so it can't race with the task's prefill spinner.
    println!();

    // Spawn unified print task: incremental renderer for text deltas,
    // tool events interleaved via clear_partial/restore_partial.
    let print_task = tokio::spawn(async move {
        use std::io::Write as _;
        #[cfg(feature = "voice")]
        let mut tts_acc = tts_tx.map(|tx| {
            crate::voice_pipeline::SentenceAccumulator::new_streaming_with_language(
                tx,
                tts_lang_override.as_deref(),
            )
        });
        #[cfg(feature = "voice")]
        let mut tts_in_thinking = false;
        #[cfg(not(feature = "voice"))]
        let _ = tts_tx;

        let mut renderer = incremental::IncrementalRenderer::new();
        let mut full_text = String::new();
        let mut tool_lines = 0usize;
        let mut collected: Vec<String> = Vec::new();
        // Track previous CallEnd to coalesce repeated status-check calls.
        let mut prev_call_end: Option<(String, usize)> = None;
        // Arguments captured at CallStart, keyed by tool_call_id, so the
        // persistent CallEnd line can show the command/path that ran.
        let mut args_by_id: std::collections::HashMap<String, String> =
            std::collections::HashMap::new();

        // Prefill spinner: the blind wait before the first token is otherwise
        // silent. Ticks elapsed time every 100ms; shows a true percentage when
        // the server streams prefill progress. Cleared the moment any output
        // arrives; re-armed by later calls' progress markers.
        let mut prefill = PrefillSpinner::new();
        prefill.start();

        // Shared engine, REPL completion policy: the agent future is awaited
        // by the caller (`process_direct_streaming` below) rather than owned
        // here, so the turn finishes when both channels close
        // (`Completion::ChannelClose`). The TUI passes its agent JoinHandle
        // instead (`Completion::AgentHandle`).
        let mut stream = TurnStream::new(delta_rx, tool_rx_opt, Completion::ChannelClose, None);
        loop {
            tokio::select! {
                biased;
                event = stream.next() => match event {
                    TurnEvent::Finished(_) => break,
                    TurnEvent::Delta(d) => {
                            // Control markers are consumed here; only real
                            // text falls through to the renderer/TTS.
                            if let Some(marker) = parse_control_marker(&d) {
                                match marker {
                                    ControlMarker::RetractReply => {
                                        prefill.clear();
                                        renderer.clear_partial();
                                        full_text.clear();
                                        renderer = incremental::IncrementalRenderer::new();
                                    }
                                    ControlMarker::FinishReason(fr) => {
                                        renderer.finish_reason = Some(fr);
                                    }
                                    ControlMarker::Tokens(n) => renderer.add_tokens(n),
                                    ControlMarker::DecodeMs(ms) => renderer.add_decode_ms(ms),
                                    ControlMarker::PromptTokens(_) => {}
                                    ControlMarker::PrefillEstimate(_) => {}
                                    ControlMarker::PrefillProgress { processed, total } => {
                                        // Never redraw over restored partial text.
                                        if prefill.is_active() || !renderer.has_partial_text() {
                                            prefill.on_progress(processed, total);
                                        }
                                    }
                                    ControlMarker::CacheStatus(_) => {}
                                }
                            } else {
                                prefill.clear();
                                full_text.push_str(&d);
                                renderer.push(&d);
                                #[cfg(feature = "voice")]
                                {
                                    let skip_tts =
                                        thinking_delta_should_skip_tts(&d, &mut tts_in_thinking);
                                    if !skip_tts {
                                        if let Some(ref mut acc) = tts_acc {
                                            acc.push(&d);
                                        }
                                    }
                                }
                            }
                    }
                    TurnEvent::Tool(event) => match event {
                        ToolEvent::CallStart { ref tool_name, ref tool_call_id, ref arguments_preview } => {
                            prefill.clear();
                            // Stash args so the CallEnd line can show the command/path.
                            args_by_id.insert(tool_call_id.clone(), arguments_preview.clone());
                            // Voice: narrate the action (no params/output spoken).
                            #[cfg(feature = "voice")]
                            if let Some(ref mut acc) = tts_acc {
                                acc.push(&format!(
                                    "{}. ",
                                    crate::voice_pipeline::tool_speech_cue(tool_name)
                                ));
                            }
                            renderer.flush_pending();
                            renderer.clear_partial();
                            renderer.emit_marker();
                            let line = format!(
                                "\x1b[36m  \u{25b6} {}({})\x1b[0m",
                                tool_name, arguments_preview
                            );
                            print!("\r{}\x1b[K", line);
                            std::io::stdout().flush().ok();
                            renderer.restore_partial();
                        }
                        ToolEvent::Progress { ref tool_name, elapsed_ms, ref output_preview, .. } => {
                            prefill.clear();
                            renderer.flush_pending();
                            renderer.clear_partial();
                            renderer.emit_marker();
                            let preview_str = output_preview.as_deref().unwrap_or("");
                            let line = format!(
                                "\x1b[36m  \u{25b6} {}\x1b[0m  \x1b[2m{}s{}\x1b[0m",
                                tool_name,
                                elapsed_ms / 1000,
                                if preview_str.is_empty() {
                                    String::new()
                                } else {
                                    format!(" {}", preview_str)
                                }
                            );
                            print!("\r\x1b[K{}", line);
                            std::io::stdout().flush().ok();
                            renderer.restore_partial();
                        }
                        ToolEvent::CallEnd { ref tool_name, ref tool_call_id, ok, duration_ms, ref result_data } => {
                            prefill.clear();
                            renderer.flush_pending();
                            renderer.clear_partial();
                            renderer.emit_marker();
                            // Coalesce repeated CallEnd for the same tool.
                            if let Some((ref prev_name, prev_lines)) = prev_call_end {
                                if prev_name == tool_name && prev_lines > 0 {
                                    print!("{}", rewind_and_clear_below(prev_lines));
                                    std::io::stdout().flush().ok();
                                    tool_lines = tool_lines.saturating_sub(prev_lines);
                                    let keep = collected.len().saturating_sub(prev_lines);
                                    collected.truncate(keep);
                                }
                            }

                            let marker = if ok { "\x1b[32m\u{2713}\x1b[0m" } else { "\x1b[31m\u{2717}\x1b[0m" };
                            // Extract a short label (command/path) from the args
                            // captured at CallStart plus the result data.
                            let args_preview = args_by_id
                                .get(tool_call_id.as_str())
                                .map(String::as_str)
                                .unwrap_or("");
                            let context = extract_tool_context(tool_name, result_data, args_preview);
                            let status_line = if context.is_empty() {
                                format!(
                                    "\x1b[36m  \u{25b6} {}\x1b[0m  {} \x1b[2m{}ms\x1b[0m",
                                    tool_name, marker, duration_ms
                                )
                            } else {
                                format!(
                                    "\x1b[36m  \u{25b6} {}\x1b[0m \x1b[2m{}\x1b[0m  {} \x1b[2m{}ms\x1b[0m",
                                    tool_name, context, marker, duration_ms
                                )
                            };
                            println!("\r\x1b[K{}", status_line);
                            let mut this_box_lines = 1usize;
                            collected.push(status_line);

                            if ok && !result_data.is_empty() {
                                // Collapsed by default: a one-line summary instead
                                // of the full output box (full output is in /audit).
                                let n = result_data.lines().filter(|l| !l.trim().is_empty()).count();
                                let first = result_data
                                    .lines()
                                    .find(|l| !l.trim().is_empty())
                                    .unwrap_or("")
                                    .trim();
                                let preview: String = first.chars().take(72).collect();
                                let more = if preview.chars().count() < first.chars().count() {
                                    "…"
                                } else {
                                    ""
                                };
                                let summary = format!(
                                    "    \x1b[2m\u{21b3} {}{}  \u{b7} {} line{}\x1b[0m",
                                    preview,
                                    more,
                                    n,
                                    if n == 1 { "" } else { "s" }
                                );
                                println!("\r\x1b[K{}", summary);
                                collected.push(summary);
                                this_box_lines += 1;
                            } else if !ok && !result_data.is_empty() {
                                let preview: String = result_data.chars().take(80).collect();
                                let err_line = format!("    \x1b[31m{}\x1b[0m", preview);
                                println!("\r\x1b[K{}", err_line);
                                collected.push(err_line);
                                this_box_lines += 1;
                            }
                            tool_lines += this_box_lines;
                            prev_call_end = Some((tool_name.clone(), this_box_lines));
                            std::io::stdout().flush().ok();
                            renderer.restore_partial();
                        }
                    },
                },
                _ = tokio::time::sleep(Duration::from_millis(100)) => {
                    renderer.tick();
                    // Keep the prefill spinner's elapsed time live (no-op
                    // unless the spinner is showing).
                    prefill.redraw();
                    if crate::tui::take_resize_pending() {
                        crate::tui::reset_scroll_region();
                        renderer.notify_resize();
                    }
                }
            }
        }
        // Turn over (both channels closed): finalize the streamed answer.
        // This ran inside the delta-close select arm before the TurnStream
        // extraction; the engine now surfaces closure as `Finished`.
        renderer.finish();
        #[cfg(feature = "voice")]
        if let Some(acc) = tts_acc.take() {
            acc.flush();
        }
        // Use \r\n — raw mode (input watcher) makes \n LF-only.
        print!("\r\n");
        std::io::stdout().flush().ok();
        (tool_lines, collected)
    });

    // Full-duplex input watcher: handles Enter (cancel + record), ESC+ESC (cancel),
    // Ctrl+C (cancel), and backtick (priority injection) during streaming/tool execution.
    let cancel_token = tokio_util::sync::CancellationToken::new();
    let (inject_tx, inject_rx) = tokio::sync::mpsc::unbounded_channel();
    let watcher_done = Arc::new(AtomicBool::new(false));
    let enter_interrupted = Arc::new(AtomicBool::new(false));

    let watcher = spawn_input_watcher(
        cancel_token.clone(),
        inject_tx,
        watcher_done.clone(),
        enter_interrupted.clone(),
    );

    let response = agent_loop
        .process_direct_streaming(
            input,
            session_id,
            channel,
            "direct",
            lang,
            delta_tx,
            tool_event_tx,
            Some(cancel_token.clone()),
            Some(inject_rx),
            None,
        )
        .await;

    // Signal watcher thread to stop and wait for it.
    watcher_done.store(true, Ordering::Relaxed);
    watcher.join().ok();
    // Defensive: ensure raw mode is off even if watcher thread panicked.
    tui::force_exit_raw_mode();
    // Flush any leftover keystrokes (e.g. rapid Esc presses) so they don't
    // leak into rustyline as partial ANSI escape sequences, which would hang.
    tui::drain_stdin();

    let cancelled = cancel_token.is_cancelled();
    let (_tool_lines, _tool_event_lines) = print_task.await.unwrap_or((0, Vec::new()));

    // Response was already rendered incrementally by IncrementalRenderer.
    // Only clean up trailing blank line and optionally append provenance footer.
    if !response.is_empty() && std::io::stdout().is_terminal() {
        use std::io::Write as _;
        // The print task leaves the cursor on a fresh blank row after the
        // streamed answer/footer. Clear only that current row; moving upward
        // here can erase visible assistant output on terminals that handle
        // raw-mode newlines differently.
        print!("\r\x1b[2K");
        std::io::stdout().flush().ok();

        // Show redaction warning if strict mode removed fabricated claims.
        let redaction_count = response.matches("[unverified claim removed]").count();
        if redaction_count > 0 {
            println!(
                "\x1b[33m\x1b[1m  \u{26a0} {} claim(s) could not be verified against tool outputs and were redacted.\x1b[0m\n",
                redaction_count
            );
        }

        // Provenance: append claim summary footer (no full re-render — text
        // was already printed incrementally by IncrementalRenderer).
        if verify_claims {
            let audit = AuditLog::new(&workspace, session_id);
            let entries = audit.get_entries();
            let verifier = ClaimVerifier::new(&entries);
            let annotated = verifier.verify(&response);
            let claims: Vec<(usize, usize, u8, String)> = annotated
                .iter()
                .map(|c| {
                    let status = match c.status {
                        ClaimStatus::Observed => 0u8,
                        ClaimStatus::Derived => 1,
                        ClaimStatus::Claimed => 2,
                        ClaimStatus::Recalled => 3,
                    };
                    (c.span.0, c.span.1, status, c.text.clone())
                })
                .collect();
            print!("{}", syntax::render_provenance_footer(&claims, strict_mode));
        }
    }

    let was_enter = enter_interrupted.load(Ordering::Relaxed);
    if cancelled {
        if was_enter {
            // Enter-interrupt: user wants to take over. Brief marker.
            println!("\n  \x1b[2mInterrupted.\x1b[0m");
        } else {
            println!("\n  \x1b[33mCancelled.\x1b[0m");
        }
    }

    (response, was_enter)
}

// ============================================================================
// Server Lifecycle (DRY: replaces 6x copy-pasted spawn+wait+rebuild patterns)
// ============================================================================

/// Which inference engine backend is managing the local server.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InferenceEngine {
    /// No local engine active.
    None,
    /// LM Studio via `lms` CLI (daemon mode).
    Lms,
    /// Higgs Rust MLX server (managed sidecar).
    Higgs,
}

pub(crate) struct ServerState {
    pub local_port: String,
    /// True when LM Studio's `lms` CLI manages the server lifecycle.
    pub lms_managed: bool,
    /// Path to the `lms` binary (set when lms_managed is true).
    pub lms_binary: Option<std::path::PathBuf>,
    /// Which inference engine is currently active.
    pub engine: InferenceEngine,
}

/// What [`ServerState::shutdown`] will tear down. Pure, so the teardown policy
/// is unit-testable without spawning real servers.
#[derive(Debug, Default, PartialEq, Eq)]
struct ShutdownPlan {
    stop_lms: bool,
    stop_higgs: bool,
}

impl ServerState {
    pub fn new(port: String) -> Self {
        Self {
            local_port: port,
            lms_managed: false,
            lms_binary: None,
            engine: InferenceEngine::None,
        }
    }

    /// Unload models from the current LMS-managed server.
    #[cfg(test)]
    pub async fn kill_current(&mut self, lms_port: u16, unload_timeout_secs: u64) {
        if self.lms_managed {
            crate::lms::unload_all("", lms_port, unload_timeout_secs)
                .await
                .ok();
        }
        self.engine = InferenceEngine::None;
    }

    /// What [`Self::shutdown`] tears down. LM Studio is stopped only when nanobot
    /// manages it. Higgs is a resident sidecar — kept warm by the keepalive ping
    /// and reused across launches — so it is NEVER stopped on exit, neither the
    /// instance nanobot spawned nor one the user started externally. (Was:
    /// `self.engine == Higgs`, which SIGKILLed it every exit and reloaded the
    /// 35B on the next launch.)
    fn shutdown_plan(&self) -> ShutdownPlan {
        ShutdownPlan {
            stop_lms: self.lms_managed,
            stop_higgs: false,
        }
    }

    /// Full shutdown: stop only the servers [`Self::shutdown_plan`] selects.
    pub fn shutdown(&mut self) {
        let plan = self.shutdown_plan();
        if plan.stop_lms {
            if let Some(ref bin) = self.lms_binary {
                println!("Stopping LM Studio server...");
                crate::lms::server_stop(bin).ok();
            }
            self.lms_managed = false;
        }
        if plan.stop_higgs {
            println!("Stopping Higgs server...");
            crate::higgs::server_stop().ok();
        }
        self.engine = InferenceEngine::None;
    }
}

/// Resolve which inference engine to use based on config preference.
///
/// Returns `(engine_kind, binary_path)` for the first available engine.
/// Currently only resolves LM Studio. Higgs is handled separately via
/// `use_higgs` / `is_higgs_backend` in the startup and `/l` paths.
pub(crate) fn resolve_inference_engine() -> Option<(InferenceEngine, std::path::PathBuf)> {
    crate::lms::find_lms_binary().map(|b| (InferenceEngine::Lms, b))
}

/// Rebuild the agent core and agent loop after a server change.
///
/// Call this after a mode switch or config update.
pub(crate) fn apply_server_change(
    state: &ServerState,
    model_path: &std::path::Path,
    core_handle: &SharedCoreHandle,
    config: &Config,
    is_local: bool,
) {
    // Prefer lms_main_model which preserves namespace prefixes like "qwen/qwen3-vl-8b".
    // PathBuf::file_name() strips parent components, breaking namespaced model identifiers.
    let model_name = if !config.agents.defaults.lms_main_model.is_empty() {
        Some(config.agents.defaults.lms_main_model.as_str())
    } else {
        model_path.file_name().and_then(|n| n.to_str())
    };
    cli::rebuild_core(
        core_handle,
        config,
        &state.local_port,
        model_name,
        None,
        None,
        None,
        is_local,
    );
}

// ============================================================================
// cmd_agent - Main REPL entry point
// ============================================================================

// Background channel state
pub(crate) struct ActiveChannel {
    pub name: String,
    pub stop: Arc<AtomicBool>,
    pub handle: tokio::task::JoinHandle<()>,
}

pub(crate) fn cmd_agent(
    message: Option<String>,
    session_id: String,
    local_flag: bool,
    lang: Option<String>,
    resume: Option<String>,
    continue_session: bool,
) {
    // Singleton guard: kill any stale agent process from a previous crashed run.
    crate::agent::pid_file::acquire_agent_singleton();

    let mut config = load_config(None);

    // Resolve voice language: CLI --lang > config voice.language > None (auto)
    let lang = lang.or_else(|| config.voice.language.clone());

    // Check environment variable for local mode
    let local_env = std::env::var("NANOBOT_LOCAL")
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false);

    // Set initial local mode from flag, environment, or config (localApiBase).
    let has_remote_local = !config.agents.defaults.local_api_base.is_empty();
    let is_local = local_flag || local_env || has_remote_local;

    // Fallback local port when no localApiBase is set. Defaults to the LM Studio
    // port (the only remaining JIT backend), never a hardcoded :8080 that nothing
    // listens on. NANOBOT_LOCAL_PORT still overrides.
    let local_port = std::env::var("NANOBOT_LOCAL_PORT")
        .unwrap_or_else(|_| config.agents.defaults.lms_port.to_string());
    if !is_local {
        let api_key = config.get_api_key();
        let model = &config.agents.defaults.model;
        let has_prefix = config.resolve_provider_for_model(model).is_some();
        let has_oauth = dirs::home_dir()
            .map(|h| h.join(".claude").join(".credentials.json").exists())
            .unwrap_or(false);
        if api_key.is_none()
            && !has_prefix
            && !model.starts_with("bedrock/")
            && !model.starts_with("claude-max")
            && !has_oauth
        {
            eprintln!("Error: No API key configured.");
            eprintln!("Set one in ~/.nanobot/config.json under providers.openrouter.apiKey");
            eprintln!("Or authenticate with Claude CLI: claude login");
            eprintln!("Or use --local flag to use a local LLM server.");
            std::process::exit(1);
        }
    }

    let runtime = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");

    runtime.block_on(async {
        // Startup phase timings land in ~/.nanobot/logs so first-paint
        // regressions are visible: grep for "startup_phase".
        let startup_t0 = std::time::Instant::now();
        let mut startup_last = startup_t0;

        // Auto-start SearXNG / crw-server in the background — these used to be
        // awaited here and could stall first paint for ~a minute when Docker
        // Desktop was down. Web tools self-heal mid-session if they come up late.
        cli::spawn_web_service_autostart(&config);
        log_startup_phase("web_autostart_spawned", startup_t0, &mut startup_last);

        // Create shared core and initial agent loop.
        // Use persisted local model name if available, else hardcoded default.
        // Prefer lms_main_model (clean identifier like "nanbeige4.1-3b") over
        // local_model which may hold a GGUF filename (e.g. "GLM-4.7-Flash-Q4_K_S.gguf").
        let mut local_model_name: String = if !config.agents.defaults.lms_main_model.is_empty() {
            config.agents.defaults.lms_main_model.clone()
        } else {
            config.agents.defaults.local_model.clone()
        };

        // --- Discovery-first local inference ---
        // Probe candidates in parallel (configured localApiBase, higgs port,
        // LM Studio port, cluster peers; ≤1s each) BEFORE any spawn decision.
        // Endpoint and model are adopted as a PAIR from one discovery result,
        // never resolved independently (the ":1234 vs adopted model" bug).
        let startup_action: Option<crate::local_discovery::StartupAction> = if is_local {
            let candidates = crate::local_discovery::discover_endpoints(&config).await;
            let selected = crate::local_discovery::select_endpoint(&candidates, &local_model_name);
            Some(crate::local_discovery::decide_startup(
                selected,
                config.agents.defaults.local_autostart,
            ))
        } else {
            None
        };
        log_startup_phase("local_discovery_done", startup_t0, &mut startup_last);

        let discovered_local = match &startup_action {
            Some(crate::local_discovery::StartupAction::UseDiscovered(pair)) => Some(pair.clone()),
            _ => None,
        };
        let no_local_server_note = matches!(
            startup_action,
            Some(crate::local_discovery::StartupAction::NoServerNote)
        );
        // Spawn intents — ONLY set when discovery found nothing AND the user
        // opted in via `localAutostart` (explicit ≠ autonomous).
        let use_higgs = matches!(
            startup_action,
            Some(crate::local_discovery::StartupAction::SpawnHiggs)
        );
        let needs_lms = matches!(
            startup_action,
            Some(crate::local_discovery::StartupAction::SpawnLmStudio)
        );
        let mut higgs_sidecar_port: Option<u16> = None;

        // Adopt endpoint + model together from the discovery result.
        if let Some(pair) = &discovered_local {
            config.agents.defaults.local_api_base = pair.base_url.clone();
            config.agents.defaults.lms_main_model = pair.model.clone();
            local_model_name = pair.model.clone();
            match pair.source {
                crate::local_discovery::EndpointSource::Higgs => {
                    config.agents.defaults.local_backend = "higgs".to_string();
                    config.agents.defaults.skip_jit_gate = true;
                    higgs_sidecar_port = Some(config.agents.defaults.higgs_port);
                }
                crate::local_discovery::EndpointSource::LmStudio => {
                    config.agents.defaults.local_backend = "lmstudio".to_string();
                }
                _ => {}
            }
            info!(
                endpoint = %pair.base_url,
                model = %pair.model,
                "local inference: using {} serving {} (discovered)",
                pair.base_url,
                pair.model
            );
        }
        // Plain REPL / single-message: print the note now. TUI: delivered via
        // display_tx below (stdout is swallowed by the alternate screen).
        if no_local_server_note && !crate::tui_app::enabled() {
            eprintln!(
                "  {}{}{}{}",
                tui::BOLD,
                tui::YELLOW,
                crate::local_discovery::NO_SERVER_NOTE,
                tui::RESET
            );
        }
        // Keep the backend tag consistent with the spawn decision so the
        // provider layer (session cache, JIT gate) sees the right backend
        // even when a stale tag survived in config.
        if use_higgs {
            config.agents.defaults.local_backend = "higgs".to_string();
        } else if needs_lms {
            config.agents.defaults.local_backend = "lmstudio".to_string();
        }
        // Non-higgs local endpoints keep the LM Studio feature set (remote
        // probe, prewarm, trio auto-activation, JIT warmup) whether spawned or
        // discovered — only SPAWNING is gated by `localAutostart`.
        let lms_features = is_local
            && !crate::config::schema::is_higgs_backend(&config.agents.defaults.local_backend);

        // Higgs sidecar: auto-start when backend is "higgs" (single-message or interactive).
        // Start even when localApiBase is set — it may point to the managed Higgs port
        // from a previous run whose PID file survived or whose server is still loading.
        //
        // Exception: when localApiBase points at a REMOTE host (cluster peer, not
        // localhost), respect the user's explicit endpoint and skip Higgs entirely.
        // The higgs backend tag is sticky across /m switches (see cmd_lifecycle.rs:589),
        // so a stale "higgs" tag must not clobber a deliberate remote URL.
        let api_base_is_remote_host = {
            let base = config.agents.defaults.local_api_base.trim().to_lowercase();
            let host_port = base
                .strip_prefix("http://")
                .or_else(|| base.strip_prefix("https://"))
                .unwrap_or(&base);
            !base.is_empty()
                && !host_port.starts_with("localhost:")
                && !host_port.starts_with("127.0.0.1:")
        };
        // A pre-configured remote base is not a nanobot-managed JIT server, so don't
        // serialise its requests behind the JIT gate (what the removed "omlx" backend
        // used to signal). Resident servers stall under a single-permit gate.
        if api_base_is_remote_host {
            config.agents.defaults.skip_jit_gate = true;
        }
        if use_higgs && api_base_is_remote_host {
            info!(
                api_base = %config.agents.defaults.local_api_base,
                "skipping higgs auto-start: localApiBase points at a remote host"
            );
        } else if use_higgs {
            let higgs_port = config.agents.defaults.higgs_port;
            match crate::higgs::resolve_model_dir(&config) {
                Ok(model_dir) => {
                    if let Some(bin) = crate::higgs::find_binary() {
                        match crate::higgs::server_start(&bin, higgs_port, &model_dir, &config.agents.defaults.local_model).await {
                            Ok(crate::higgs::StartResult::Ready) => {
                                higgs_sidecar_port = Some(higgs_port);
                                config.agents.defaults.local_api_base =
                                    format!("http://127.0.0.1:{higgs_port}/v1");
                                config.agents.defaults.skip_jit_gate = true;
                                // Resolve the served id that matches the configured model.
                                // An externally-managed Higgs may serve several models; taking
                                // the first one (or a stale name) makes nanobot request a model
                                // Higgs hasn't loaded → 404.
                                if let Some(name) =
                                    crate::higgs::resolve_served_model(higgs_port, &local_model_name)
                                        .await
                                {
                                    config.agents.defaults.lms_main_model = name.clone();
                                    local_model_name = name;
                                }
                            }
                            Ok(crate::higgs::StartResult::Loading { pid, port }) => {
                                higgs_sidecar_port = Some(port);
                                eprintln!(
                                    "Warning: Higgs (pid {}) still loading model on port {} — requests will retry until ready",
                                    pid, port
                                );
                                config.agents.defaults.local_api_base =
                                    format!("http://127.0.0.1:{port}/v1");
                                config.agents.defaults.skip_jit_gate = true;
                            }
                            Err(e) => {
                                // Don't clear a user-set localApiBase on failure — falling back
                                // to a default port silently routes to a dead server (the :8080
                                // bug). Surface the error; let requests fail against the real URL.
                                eprintln!("Warning: failed to start Higgs: {e}");
                            }
                        }
                    } else {
                        eprintln!(
                            "Error: Higgs binary not found. Install with: cargo install higgs"
                        );
                        std::process::exit(1);
                    }
                }
                Err(e) => {
                    eprintln!("Error: {e}");
                    std::process::exit(1);
                }
            }
        }

        // In local mode with single-message (-m), just start main server.
        // Trio is for interactive sessions - single messages use inline tools.
        // When localApiBase is set, skip all local server spawning — use remote server.
        let mut trio_state: Option<ServerState> = None;
        // Recompute has_remote_local — discovery adoption or Higgs auto-start
        // may have filled local_api_base.
        let has_remote_local = !config.agents.defaults.local_api_base.is_empty();
        if needs_lms && !has_remote_local && message.is_some() {
            // Single-message local mode: start LMS if available.
            let mut srv = ServerState::new(local_port.clone());
            if let Some((InferenceEngine::Lms, bin)) = resolve_inference_engine() {
                let lms_port = config.agents.defaults.lms_port;
                match crate::lms::server_start(&bin, lms_port).await {
                    Ok(()) => {
                        let available = crate::lms::list_available("", lms_port).await;
                        let main_model = if !config.agents.defaults.lms_main_model.is_empty() {
                            config.agents.defaults.lms_main_model.clone()
                        } else {
                            let hint = cli::strip_gguf_suffix(&local_model_name);
                            crate::lms::resolve_model_name(&available, hint)
                        };
                        let main_ctx = Some(config.agents.defaults.local_max_context_tokens);
                        if let Err(e) = crate::lms::load_model("", lms_port, &main_model, main_ctx, config.timeouts.lms_load_secs).await {
                            eprintln!("Warning: lms load failed: {}", e);
                        } else {
                            local_model_name = main_model;
                            srv.lms_managed = true;
                            srv.lms_binary = Some(bin);
                            srv.local_port = lms_port.to_string();
                            if config.agents.defaults.local_api_base.is_empty() {
                                let lms_host = crate::lms::api_host();
                                config.agents.defaults.local_api_base =
                                    format!("http://{}:{}/v1", lms_host, lms_port);
                            }
                            config.agents.defaults.skip_jit_gate = true;
                        }

                        // Trio model loading for single-message mode.
                        let (auto_router, auto_specialist) =
                            commands::pick_trio_models(&available, &local_model_name);
                        if config.trio.router_model.is_empty() {
                            if let Some(r) = auto_router {
                                config.trio.router_model = r;
                            }
                        }
                        if config.trio.specialist_model.is_empty() {
                            if let Some(s) = auto_specialist {
                                config.trio.specialist_model = s;
                            }
                        }
                        if config.trio.enabled
                            || commands::should_auto_activate_trio(
                                is_local,
                                &config.trio.router_model,
                                &config.trio.specialist_model,
                                config.trio.router_endpoint.is_some(),
                                config.trio.specialist_endpoint.is_some(),
                                &config.tool_delegation.mode,
                            )
                        {
                            // Load router model if configured.
                            if !config.trio.router_model.is_empty() {
                                let _ = crate::lms::load_model(
                                    "",
                                    lms_port,
                                    &config.trio.router_model,
                                    Some(config.trio.router_ctx_tokens),
                                    config.timeouts.lms_load_secs,
                                )
                                .await;
                            }
                            // Load specialist model if configured.
                            if !config.trio.specialist_model.is_empty() {
                                let _ = crate::lms::load_model(
                                    "",
                                    lms_port,
                                    &config.trio.specialist_model,
                                    Some(config.trio.specialist_ctx_tokens),
                                    config.timeouts.lms_load_secs,
                                )
                                .await;
                            }
                            commands::trio_enable(&mut config);
                        }
                    }
                    Err(e) => {
                        eprintln!("Warning: lms server start failed: {}", e);
                    }
                }
            } else {
                eprintln!("Error: No local inference engine found. Install LM Studio (lms CLI) or Higgs (cargo install higgs).");
                std::process::exit(1);
            }
            trio_state = Some(srv);
        }

        // Interactive REPL: detect LMS and set config BEFORE building core,
        // so the initial core handle and SubagentManager get the right URL.
        let mut srv = ServerState::new(local_port.clone());
        if let Some(port) = higgs_sidecar_port {
            srv.engine = InferenceEngine::Higgs;
            srv.local_port = port.to_string();
        }
        let mut config = config; // shadow to allow mutation
        let is_interactive = message.is_none();
        // The full-screen TUI renders its own intro; suppress the legacy stdout
        // splash (CLEAR_SCREEN + logo) so the old screen never flashes before
        // ratatui takes the alternate screen.
        let show_splash = !crate::tui_app::enabled();
        // Higgs splash for both a spawned sidecar and a discovered instance.
        if is_interactive && (use_higgs || higgs_sidecar_port.is_some()) && show_splash {
            tui::register_resize_handler();
            let higgs_port = config.agents.defaults.higgs_port;
            tui::print_higgs_splash(&local_model_name, higgs_port);
        }
        if is_interactive && needs_lms && !has_remote_local {
            if show_splash {
                tui::register_resize_handler();
                tui::print_startup_splash(&local_port, is_local);
            }

            if let Some((InferenceEngine::Lms, bin)) = resolve_inference_engine() {
                let lms_port = config.agents.defaults.lms_port;
                println!(
                    "  {}{}LM Studio{} detected, starting server on port {}...",
                    tui::BOLD, tui::YELLOW, tui::RESET, lms_port
                );

                match crate::lms::server_start(&bin, lms_port).await {
                    Ok(()) => {
                        let available = crate::lms::list_available("", lms_port).await;
                        let main_model = if !config.agents.defaults.lms_main_model.is_empty() {
                            config.agents.defaults.lms_main_model.clone()
                        } else {
                            let hint = cli::strip_gguf_suffix(&local_model_name);
                            crate::lms::resolve_model_name(&available, hint)
                        };
                        let main_ctx = Some(config.agents.defaults.local_max_context_tokens);
                        print!("  Loading {}... ", main_model);
                        io::stdout().flush().ok();
                        match crate::lms::load_model("", lms_port, &main_model, main_ctx, config.timeouts.lms_load_secs).await {
                            Ok(()) => println!("{}OK{}", tui::GREEN, tui::RESET),
                            Err(e) => println!("{}FAILED: {}{}", tui::RED, e, tui::RESET),
                        }

                        // Auto-detect trio roles from available models if not configured.
                        let (auto_router, auto_specialist) =
                            commands::pick_trio_models(&available, &main_model);
                        if config.trio.router_model.is_empty() {
                            if let Some(r) = auto_router {
                                config.trio.router_model = r;
                                info!(router = %config.trio.router_model, "trio_router_auto_detected");
                            }
                        }
                        if config.trio.specialist_model.is_empty() {
                            if let Some(s) = auto_specialist {
                                config.trio.specialist_model = s;
                                info!(specialist = %config.trio.specialist_model, "trio_specialist_auto_detected");
                            }
                        }

                        if config.trio.enabled {
                            if !config.trio.router_model.is_empty() {
                                print!("  Loading {}... ", config.trio.router_model);
                                io::stdout().flush().ok();
                                match crate::lms::load_model("", lms_port, &config.trio.router_model, Some(config.trio.router_ctx_tokens), config.timeouts.lms_load_secs).await {
                                    Ok(()) => println!("{}OK{}", tui::GREEN, tui::RESET),
                                    Err(e) => println!("{}FAILED: {}{}", tui::RED, e, tui::RESET),
                                }
                            }
                            if !config.trio.specialist_model.is_empty() {
                                print!("  Loading {}... ", config.trio.specialist_model);
                                io::stdout().flush().ok();
                                match crate::lms::load_model("", lms_port, &config.trio.specialist_model, Some(config.trio.specialist_ctx_tokens), config.timeouts.lms_load_secs).await {
                                    Ok(()) => println!("{}OK{}", tui::GREEN, tui::RESET),
                                    Err(e) => println!("{}FAILED: {}{}", tui::RED, e, tui::RESET),
                                }
                            }
                        }

                        // Load LCM compaction model when configured.
                        if config.lcm.is_enabled() {
                            if let Some(ref ep) = config.lcm.compaction_endpoint {
                                print!("  Loading {} (LCM compactor)... ", ep.model);
                                io::stdout().flush().ok();
                                match crate::lms::load_model("", lms_port, &ep.model, Some(config.lcm.compaction_context_size), config.timeouts.lms_load_secs).await {
                                    Ok(()) => println!("{}OK{}", tui::GREEN, tui::RESET),
                                    Err(e) => println!("{}FAILED: {}{}", tui::RED, e, tui::RESET),
                                }
                            }
                        }

                        local_model_name = main_model;
                        srv.lms_managed = true;
                        srv.lms_binary = Some(bin);
                        srv.local_port = lms_port.to_string();
                        if config.agents.defaults.local_api_base.is_empty() {
                            let lms_host = crate::lms::api_host();
                            config.agents.defaults.local_api_base =
                                format!("http://{}:{}/v1", lms_host, lms_port);
                        }
                        config.agents.defaults.skip_jit_gate = true;
                    }
                    Err(e) => {
                        println!(
                            "  {}{}lms server start failed:{} {}",
                            tui::BOLD, tui::YELLOW, tui::RESET, e
                        );
                    }
                }
            }
        }

        log_startup_phase("local_backend_ready", startup_t0, &mut startup_last);

        // Recompute has_remote_local after potential lms setup
        let mut has_remote_local = !config.agents.defaults.local_api_base.is_empty();

        // --- Remote peer probe ---
        // If the saved endpoint is a remote peer we didn't start, probe it.
        // When the user has explicitly configured localApiBase we NEVER fall back
        // to a local llama.cpp server: requests would go to the configured remote
        // while the local server receives nothing.  Instead, warn and clear the
        // dead endpoint so the user knows what happened.
        // Skip entirely when using MLX or oMLX local backend — no LM Studio involved.
        // Also skip when discovery just validated this endpoint (<1s ago) and
        // adopted its served model — re-probing would only add latency.
        if lms_features && has_remote_local && !srv.lms_managed && discovered_local.is_none() {
            let peer_url = config.agents.defaults.local_api_base.clone();
            let peer_host = extract_url_host(&peer_url);
            let peer_port = peer_url
                .split(':')
                .last()
                .and_then(|p| p.split('/').next())
                .and_then(|p| p.parse::<u16>().ok())
                .unwrap_or(18080);

            let mut probe = crate::lms::list_available(&peer_host, peer_port).await;
            if !probe.is_empty() && config.agents.defaults.lms_main_model.is_empty() {
                // Remote is alive but lms_main_model is not configured: use the
                // first loaded model reported by the remote instead of the stale
                // local_model config value (which may be a GGUF filename like
                // "GLM-4.7-Flash-Q4_K_S.gguf" that refers to a different model).
                local_model_name = probe[0].clone();
            }
            if probe.is_empty() {
                // Not an LM Studio native API — the endpoint may still be a
                // healthy OpenAI-compatible resident server (e.g. Higgs left at
                // localApiBase after a `/model` pick re-tagged the backend as
                // "lmstudio"). Ask /v1/models directly; if it answers, adopt the
                // model id the server ACTUALLY serves so the first request can't
                // 404 on a stale configured name (this is what previously forced
                // a manual /model before anything worked).
                let served = crate::higgs::list_served_models_at(
                    &peer_url,
                    &config.agents.defaults.local_api_key,
                )
                .await;
                if let Some(adopted) =
                    crate::higgs::adopt_served_model(&local_model_name, &served)
                {
                    info!(
                        configured = %local_model_name,
                        adopted = %adopted,
                        "startup: adopted served model from local endpoint"
                    );
                    local_model_name = adopted.clone();
                    config.agents.defaults.lms_main_model = adopted;
                }
                probe = served;
            }
            if probe.is_empty() {
                // Remote is unreachable.  Because localApiBase is explicitly
                // configured, do NOT start a local llama.cpp server — requests
                // still go to the configured URL, so a local server would be
                // ignored.  Warn the user and clear the dead endpoint instead.
                println!(
                    "  {}{}Remote LM Studio at {} is unreachable.{} Check your localApiBase config.",
                    tui::BOLD,
                    tui::YELLOW,
                    peer_url,
                    tui::RESET,
                );
                config.agents.defaults.local_api_base.clear();
                let mut disk_cfg = load_config(None);
                disk_cfg.agents.defaults.local_api_base.clear();
                save_config(&disk_cfg, None);
                has_remote_local = false;
                println!(
                    "  Cleared dead endpoint. Use {}/m{} to pick a model when the remote comes online.",
                    tui::BOLD, tui::RESET,
                );
            }
        }

        // Remote LM Studio base: proactively prewarm main/router/specialist models
        // to avoid first-turn latency spikes from JIT loading.
        // Skip for oMLX — it uses LRU auto-eviction, not JIT loading.
        if lms_features && has_remote_local && !srv.lms_managed {
            // Background: prewarming is a latency optimization for the FIRST
            // turn; it must not delay first paint (JIT loads can take 30s+).
            let cfg = config.clone();
            let model = local_model_name.clone();
            tokio::spawn(async move {
                prewarm_remote_lms_models(&cfg, &model).await;
                info!("remote_lms_prewarm_done");
            });
        }

        // Auto-activate trio mode for local sessions when both router and
        // specialist models are configured.  The downgrade block below will
        // revert strict flags if the router turns out to be unreachable.
        // Skip for MLX/oMLX local — no LM Studio trio support.
        if lms_features && commands::should_auto_activate_trio(
            is_local,
            &config.trio.router_model,
            &config.trio.specialist_model,
            config.trio.router_endpoint.is_some(),
            config.trio.specialist_endpoint.is_some(),
            &config.tool_delegation.mode,
        ) {
            commands::trio_enable(&mut config);
            info!(
                delegation_mode = ?config.tool_delegation.mode,
                router_model = %config.trio.router_model,
                specialist_model = %config.trio.specialist_model,
                "trio_auto_activated"
            );
        }

        // When no trio router is available, disable strict mode so the single model
        // can handle tools directly. Must happen BEFORE build_core_handle so the core
        // gets the updated tool_delegation_config.
        if lms_features
            && config.tool_delegation.strict_no_tools_main
            && config.tool_delegation.strict_router_schema
        {
            let router_available = if srv.lms_managed || has_remote_local {
                // For both managed (started by nanobot) and remote LM Studio,
                // verify the model is actually loaded via list_available()
                let (lms_host, lms_port) = if srv.lms_managed {
                    (String::new(), config.agents.defaults.lms_port)
                } else {
                    // Extract host and port from local_api_base
                    // (e.g. "http://192.168.1.22:18080/v1")
                    let base = &config.agents.defaults.local_api_base;
                    let port = base
                        .split(':')
                        .last()
                        .and_then(|p| p.split('/').next())
                        .and_then(|p| p.parse::<u16>().ok())
                        .unwrap_or(18080);
                    (extract_url_host(base), port)
                };
                let available = crate::lms::list_available(&lms_host, lms_port).await;
                crate::lms::is_model_available(&available, &config.trio.router_model)
            } else {
                false
            };

            if !router_available {
                info!("trio_downgrade: router not available, clearing strict flags");
                config.tool_delegation.strict_no_tools_main = false;
                config.tool_delegation.strict_router_schema = false;
            }
        }

        info!(
            delegation_mode = ?config.tool_delegation.mode,
            strict_no_tools_main = config.tool_delegation.strict_no_tools_main,
            strict_router_schema = config.tool_delegation.strict_router_schema,
            is_local,
            "delegation_config_at_core_build"
        );

        log_startup_phase("endpoint_probe_done", startup_t0, &mut startup_last);

        let core_handle = cli::build_core_handle(
            &config,
            &srv.local_port,
            Some(&local_model_name),
            None,
            None,
            None,
            is_local,
        );
        // Resolve --resume / --continue to a real session key.
        let session_id = if let Some(ref id) = resume {
            // --resume <id>: look up session by ID and use its session_key
            let core = core_handle.swappable();
            if let Some(meta) = core.sessions.get_session(id).await {
                info!(session_id = %meta.id, session_key = %meta.session_key, "resuming session by ID");
                meta.session_key
            } else {
                eprintln!("Warning: session '{}' not found, starting new session", id);
                session_id
            }
        } else if continue_session {
            // --continue: use latest session for the given key (default behavior
            // of get_or_resume, so session_id as-is is correct)
            info!(session_key = %session_id, "continuing latest session");
            session_id
        } else if session_id == "cli:default" {
            // No explicit --session / --resume / --continue: mint a fresh
            // per-invocation key so we don't silently resume the latest
            // cli:default session (which accretes history across runs and
            // blows up TTFT on cold local models).
            let fresh = format!(
                "cli:oneshot-{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis())
                    .unwrap_or(0)
            );
            info!(session_key = %fresh, "starting fresh ephemeral session (no --continue/--resume)");
            fresh
        } else {
            session_id
        };

        let cron_store_path = get_data_dir().join("cron").join("jobs.json");
        let cron_service = Arc::new(CronService::new(cron_store_path));

        // Provide email config to the REPL agent when credentials are configured.
        let email_config = {
            let ec = &config.channels.email;
            if !ec.imap_host.is_empty() && !ec.username.is_empty() && !ec.password.is_empty() {
                Some(ec.clone())
            } else {
                None
            }
        };

        // Channel for subagents/background gateways to send display lines to the REPL.
        let (display_tx, display_rx) = mpsc::unbounded_channel::<String>();

        // TUI swallows pre-alt-screen stdout, so deliver the no-server note
        // as a display line where it lands in the transcript.
        if no_local_server_note && crate::tui_app::enabled() {
            let _ = display_tx.send(crate::local_discovery::NO_SERVER_NOTE.to_string());
        }

        let health_registry = std::sync::Arc::new(crate::heartbeat::health::build_registry(&config));

        // Must be `mut` for the cluster-feature code path below, which passes
        // `&mut agent_loop` to `setup_cluster_for_repl`. Under non-cluster
        // builds the binding is still reassigned/shadowed later, so no
        // unused_mut warning is emitted regardless.
        #[cfg_attr(not(feature = "cluster"), allow(unused_mut))]
        let mut agent_loop = cli::create_agent_loop(
            core_handle.clone(),
            &config,
            Some(cron_service.clone()),
            email_config.clone(),
            Some(display_tx.clone()),
            Some(health_registry.clone()),
        );

        // Set up cluster discovery for the REPL path (feature-gated).
        // Returns the ClusterState so /cluster commands can query it.
        #[cfg(feature = "cluster")]
        let cluster_state = cli::setup_cluster_for_repl(&mut agent_loop, &config);

        log_startup_phase("core_and_agent_built", startup_t0, &mut startup_last);

        if let Some(msg) = message {
            // Single-message mode: process and exit.
            // Keep trio servers alive during processing (they're dropped at end of scope).
            let _servers = &trio_state;
            let mut agent_loop = agent_loop;
            stream_and_render(
                &mut agent_loop,
                &msg,
                &session_id,
                "cli",
                None,
                &core_handle,
            )
            .await;
            index_sessions_background();
            srv.shutdown();
        } else {
            // Interactive REPL mode.
            // Catch-up session indexing at startup — same reconciliation the
            // exit path runs, so sessions from crashed runs become searchable
            // via recall. Already-indexed sessions are skipped (idempotent);
            // spawn_blocking keeps the fs work off the first-prompt path.
            tokio::task::spawn_blocking(index_sessions_background);

            // Splash and LMS detection already happened above (before core build).
            // Skip for MLX/oMLX local — already printed banner above.
            if (lms_features || !is_local) && (!is_local || has_remote_local) && show_splash {
                tui::register_resize_handler();
                tui::print_startup_splash(&local_port, is_local);
            }

            // Load persisted local model preference.
            let default_model = {
                let models_dir = dirs::home_dir().unwrap().join("models");
                let saved = &config.agents.defaults.local_model;
                let saved_path = models_dir.join(saved);
                if saved_path.exists() {
                    saved_path
                } else {
                    models_dir.join(saved)
                }
            };

            // Readline editor with history
            let history_path = get_data_dir().join("history.txt");
            let mut rl = rustyline::DefaultEditor::new().expect("Failed to create line editor");

            // Alt+Enter inserts a newline (multi-line editing) instead of submitting.
            rl.bind_sequence(
                rustyline::KeyEvent(rustyline::KeyCode::Enter, rustyline::Modifiers::ALT),
                rustyline::Cmd::Newline,
            );

            // Build ReplContext — all mutable REPL state in one struct.
            let (restart_tx, restart_rx) = tokio::sync::mpsc::unbounded_channel();
            let mut ctx = commands::ReplContext {
                config,
                core_handle,
                agent_loop,
                session_id,
                lang,
                srv,
                current_model_path: default_model,
                active_channels: vec![],
                display_tx,
                display_rx,
                cron_service,
                email_config,
                rl: Some(rl),
                watchdog_handle: None,
                restart_tx: restart_tx.clone(),
                restart_rx,
                health_registry: Some(health_registry),
                #[cfg(feature = "voice")]
                voice_session: None,
                #[cfg(feature = "cluster")]
                cluster_state,
            };

            let _ = ctx.rl.as_mut().unwrap().load_history(&history_path);

            // JIT warmup: pre-load models on the remote JIT server (e.g. LM Studio).
            // This forces each model to load sequentially before any real requests,
            // avoiding concurrent model-switch crashes and cold-start latency.
            // Fires for any JIT server (localApiBase set), not just trio mode.
            // Skip for MLX/oMLX local — no remote JIT server involved.
            if lms_features && has_remote_local && !ctx.srv.lms_managed {
                use crate::providers::jit_gate::warmup_jit_models;

                let base = ctx.config.agents.defaults.local_api_base.clone();
                let mut models_to_warm: Vec<String> = Vec::new();

                // Main model — always warm so first message is fast. Prefer the
                // resolved/adopted id (lms_main_model) over the raw localModel,
                // which may be a stale name the server never loaded.
                let main_model_ref = if !ctx.config.agents.defaults.lms_main_model.is_empty() {
                    &ctx.config.agents.defaults.lms_main_model
                } else {
                    &ctx.config.agents.defaults.local_model
                };
                models_to_warm.push(cli::strip_gguf_suffix(main_model_ref).to_string());

                // Trio models (router + specialist) when enabled.
                if ctx.config.trio.enabled {
                    // Router: prefer explicit endpoint model, fall back to trio config.
                    if let Some(ref ep) = ctx.config.trio.router_endpoint {
                        models_to_warm.push(ep.model.clone());
                    } else if !ctx.config.trio.router_model.is_empty() {
                        models_to_warm.push(ctx.config.trio.router_model.clone());
                    }
                    // Specialist: prefer explicit endpoint model, fall back to trio config.
                    if let Some(ref ep) = ctx.config.trio.specialist_endpoint {
                        models_to_warm.push(ep.model.clone());
                    } else if !ctx.config.trio.specialist_model.is_empty() {
                        models_to_warm.push(ctx.config.trio.specialist_model.clone());
                    }
                }

                // LCM compaction model when configured.
                if let Some(ref ep) = ctx.config.lcm.compaction_endpoint {
                    if ctx.config.lcm.is_enabled() {
                        models_to_warm.push(ep.model.clone());
                    }
                }

                // Background: warming can take 30s per model — if it hasn't
                // finished by the first message, that request JIT-loads anyway.
                tokio::spawn(async move {
                    let refs: Vec<&str> = models_to_warm.iter().map(String::as_str).collect();
                    warmup_jit_models(&base, "local", &refs).await;
                    info!("jit_warmup_done");
                });
            }

            // Start heartbeat: maintenance commands run on every tick (no LLM).
            let maintenance_cmds: Vec<String> = DEFAULT_MAINTENANCE_COMMANDS
                .iter()
                .map(|s| s.to_string())
                .collect();
            let heartbeat = HeartbeatService::new(
                ctx.config.workspace_path(),
                None, // No LLM callback — maintenance only
                maintenance_cmds,
                DEFAULT_HEARTBEAT_INTERVAL_S,
                true,
                ctx.health_registry.clone(),
            );
            heartbeat.start().await;

            // Start health watchdog for local servers (if any are running).
            // Skip when using a remote local server (e.g. oMLX) — there is
            // no local server to monitor and the watchdog would spam the remote.
            // Higgs is a managed sidecar so it DOES need the watchdog despite
            // setting local_api_base.
            if is_local
                && (!has_remote_local
                    || crate::config::schema::is_higgs_backend(
                        &ctx.config.agents.defaults.local_backend,
                    ))
            {
                ctx.restart_watchdog();
            }

            // Keep the local Higgs model warm so idle gaps don't trigger a cold
            // reload on the next turn (the watchdog only checks liveness; it
            // doesn't touch the weights). Gated to our own localhost Higgs.
            let keepalive_stop = Arc::new(AtomicBool::new(false));
            let keepalive_handle = higgs_keepalive_secs(
                &ctx.config.agents.defaults.local_backend,
                &ctx.config.agents.defaults.local_api_base,
                std::env::var("NANOBOT_HIGGS_KEEPALIVE_SECS").ok().as_deref(),
            )
            .map(|secs| {
                let model = if !ctx.config.agents.defaults.lms_main_model.is_empty() {
                    ctx.config.agents.defaults.lms_main_model.clone()
                } else {
                    local_model_name.clone()
                };
                info!(interval_s = secs, model = %model, "higgs_keepalive_enabled");
                spawn_higgs_keepalive(
                    ctx.config.agents.defaults.local_api_base.clone(),
                    ctx.config.agents.defaults.local_api_key.clone(),
                    model,
                    secs,
                    Arc::clone(&ctx.core_handle.counters.inference_active),
                    Arc::clone(&keepalive_stop),
                )
            });

            log_startup_phase("interactive_ready", startup_t0, &mut startup_last);

            // First bar render pushes content up to make room; all subsequent
            // renders refresh in place so we never get a duplicate bar.
            let mut bar_needs_push = true;

            loop {
                // Full-screen ratatui UI (opt-in via NANOBOT_TUI). Runs one
                // interactive session, then breaks to the shared cleanup below.
                if crate::tui_app::enabled() {
                    if let Err(e) = crate::tui_app::run(&mut ctx).await {
                        eprintln!("nanobot: TUI error: {e}");
                    }
                    break;
                }

                // Drain any pending display messages from background channels.
                ctx.drain_display();

                // Handle auto-restart requests from watchdog.
                ctx.handle_restart_requests().await;

                // migrated from swappable().is_local — phase 09-03
                let is_local = ctx.core_handle.swappable().mode().is_local();
                let voice_on = ctx.voice_on();
                let thinking_on = ctx
                    .core_handle
                    .counters
                    .thinking_budget
                    .load(Ordering::SeqCst)
                    > 0;
                let prompt = build_prompt(is_local, voice_on, thinking_on);

                // Render Claude Code-style input bar below the prompt line.
                let sa_count = ctx.agent_loop.subagent_manager().get_running_count().await;
                ctx.active_channels.retain(|ch| !ch.handle.is_finished());
                let ch_names: Vec<&str> = ctx
                    .active_channels
                    .iter()
                    .map(|c| short_channel_name(&c.name))
                    .collect();
                if tui::take_resize_pending() {
                    tui::reset_scroll_region();
                }
                tui::render_input_bar(&ctx.core_handle, &ch_names, sa_count, bar_needs_push);
                bar_needs_push = false;

                // === GET INPUT ===
                let input_text: String;
                // `do_record` is only read in voice builds (recording trigger).
                #[allow(unused_mut, unused_variables)]
                let mut do_record = false;

                #[cfg(feature = "voice")]
                if voice_on {
                    print!("{}", prompt);
                    io::stdout().flush().ok();
                    match tui::voice_read_input() {
                        tui::VoiceAction::Record => {
                            do_record = true;
                            input_text = String::new();
                        }
                        tui::VoiceAction::Text(t) => {
                            input_text = t;
                        }
                        tui::VoiceAction::Exit => break,
                    }
                } else {
                    match ctx.readline_async(&prompt).await {
                        Ok(line) => {
                            let _ = ctx.rl.as_mut().unwrap().add_history_entry(&line);
                            input_text = line;
                        }
                        Err(ReadlineError::Interrupted | ReadlineError::Eof) => break,
                        Err(_) => break,
                    }
                }

                #[cfg(not(feature = "voice"))]
                {
                    match ctx.readline_async(&prompt).await {
                        Ok(line) => {
                            let _ = ctx.rl.as_mut().unwrap().add_history_entry(&line);
                            input_text = line;
                        }
                        Err(ReadlineError::Interrupted | ReadlineError::Eof) => break,
                        Err(_) => break,
                    }
                }

                // Clear the submitted text from the pinned input box (keeping its
                // grey background) so the box stays visible and empty while the
                // reply streams above it.
                tui::clear_input_row();

                // === VOICE RECORDING ===
                // Uses the same stream_and_render pipeline as text mode for
                // identical UI quality (syntax highlighting, provenance, status bar).
                // TTS plays after rendering completes.
                #[cfg(feature = "voice")]
                if do_record {
                    let mut keep_recording = true;
                    // Stop any ongoing playback.
                    if let Some(ref mut vs) = ctx.voice_session {
                        vs.stop_playback();
                    }
                    while keep_recording {
                        keep_recording = false;

                        // Phase 1: Record and transcribe (borrows vs briefly).
                        let transcription = ctx
                            .voice_session
                            .as_mut()
                            .and_then(|vs| vs.record_and_transcribe().transpose())
                            .transpose();

                        match transcription {
                            Ok(Some((text, detected_lang))) => {
                                let tts_lang_owned = ctx.lang.clone().unwrap_or(detected_lang);

                                // Render user text with purple ● marker.
                                print!(
                                    "{}",
                                    syntax::render_turn(&text, syntax::TurnRole::VoiceUser)
                                );

                                // Start streaming TTS pipeline BEFORE LLM call.
                                let tts_parts = ctx.voice_session.as_mut().and_then(|vs| {
                                    vs.clear_cancel();
                                    vs.start_streaming_speak(&tts_lang_owned, None).ok()
                                });
                                let (sentence_tx, join_handle) = match tts_parts {
                                    Some((tx, jh)) => (Some(tx), Some(jh)),
                                    None => (None, None),
                                };

                                // Phase 2: LLM call with parallel TTS feeding.
                                let (_response, enter_pressed) = stream_and_render_voice(
                                    &mut ctx.agent_loop,
                                    &text,
                                    &ctx.session_id,
                                    "voice",
                                    Some(&tts_lang_owned),
                                    &ctx.core_handle,
                                    sentence_tx,
                                )
                                .await;

                                ctx.drain_display();
                                println!();
                                ctx.print_status_bar().await;
                                println!(); // breathing room before next input bar

                                // Phase 3: Wait for TTS playback to finish.
                                if enter_pressed {
                                    // Enter-interrupt: skip TTS, start recording immediately.
                                    if let Some(ref mut vs) = ctx.voice_session {
                                        vs.stop_playback();
                                    }
                                    keep_recording = true;
                                } else if let Some(jh) = join_handle {
                                    let cancel = ctx
                                        .voice_session
                                        .as_ref()
                                        .map(|vs| vs.cancel_flag())
                                        .unwrap_or_else(|| Arc::new(AtomicBool::new(false)));
                                    let done = Arc::new(AtomicBool::new(false));
                                    let done2 = done.clone();
                                    let watcher =
                                        tui::spawn_interrupt_watcher(cancel.clone(), done2);
                                    let _ = jh.join(); // blocks until all audio played
                                    done.store(true, Ordering::Relaxed);
                                    let interrupted = watcher.join().unwrap_or(false);
                                    if interrupted {
                                        if let Some(ref mut vs) = ctx.voice_session {
                                            vs.stop_playback();
                                        }
                                        keep_recording = true;
                                    }
                                }
                            }
                            Ok(None) => println!("\x1b[2m(no speech detected)\x1b[0m"),
                            Err(e) => eprintln!("\x1b[31m{}\x1b[0m", e),
                        }
                    }
                    tui::drain_stdin();
                    continue;
                }

                // === TEXT INPUT ===
                let input = input_text.trim();
                if input.is_empty() {
                    continue;
                }

                // Dispatch slash commands.
                if input.starts_with('/') && ctx.dispatch(input).await {
                    // Push content up on next bar render so short command
                    // output (e.g. /cluster, /status) isn't overwritten.
                    bar_needs_push = true;
                    continue;
                }

                // Process message (streaming)
                let channel = if voice_on { "voice" } else { "cli" };
                #[allow(unused_variables)]
                let response = stream_and_render(
                    &mut ctx.agent_loop,
                    input,
                    &ctx.session_id,
                    channel,
                    None,
                    &ctx.core_handle,
                )
                .await;
                ctx.drain_display();
                println!();
                // The input bar is re-rendered at the top of the loop; rendering
                // it again here drew a second, stale bar each turn (two stacked
                // bars + cursor/scroll-region artifacts at the bottom). The
                // loop-top render is the single source of truth.

                #[cfg(feature = "voice")]
                if let Some(ref mut vs) = ctx.voice_session {
                    let tts_text = tui::strip_markdown_for_tts(&response);
                    if !tts_text.is_empty() {
                        let tts_lang = ctx
                            .lang
                            .as_deref()
                            .map(str::trim)
                            .filter(|s| !s.is_empty() && !s.eq_ignore_ascii_case("auto"))
                            .map(str::to_string)
                            .unwrap_or_else(|| crate::voice_pipeline::detect_language(&tts_text));
                        tui::speak_interruptible(vs, &tts_text, &tts_lang);
                    }
                }
            }

            // Stop any active background channels
            for ch in &ctx.active_channels {
                ch.stop.store(true, Ordering::Relaxed);
            }
            if !ctx.active_channels.is_empty() {
                tokio::time::sleep(Duration::from_millis(500)).await;
                for ch in &ctx.active_channels {
                    ch.handle.abort();
                }
            }

            // Shutdown voice session first — leaks native TTS engines to
            // avoid C++ destructor segfault on exit.
            #[cfg(feature = "voice")]
            if let Some(vs) = ctx.voice_session.take() {
                vs.shutdown();
            }

            // Reset terminal: clear the pinned input bar and restore full scroll region
            // so the shell prompt returns to a clean state.
            {
                use std::io::Write as _;
                print!("\x1b[r"); // reset scroll region to full terminal
                let h = tui::terminal_height();
                print!("\x1b[{};1H", h); // move to last row
                print!("\x1b[J"); // clear from cursor to end of screen
                std::io::stdout().flush().ok();
            }

            // Cleanup: stop heartbeat, save readline history, kill servers
            heartbeat.stop().await;
            if let Some(h) = keepalive_handle {
                keepalive_stop.store(true, Ordering::Relaxed);
                h.abort();
            }
            let _ = ctx.rl.as_mut().unwrap().save_history(&history_path);

            // Unload trio models so LM Studio returns to just the main model.
            // Run when trio is enabled and we have any LMS connection (managed or
            // user-started via localApiBase) — not only when nanobot started the server.
            if ctx.config.trio.enabled && (ctx.srv.lms_managed || has_remote_local) {
                let (lms_host, lms_port) = if ctx.srv.lms_managed {
                    (String::new(), ctx.config.agents.defaults.lms_port)
                } else {
                    // Extract host and port from localApiBase
                    // (e.g. "http://192.168.1.22:18080/v1")
                    let base = &ctx.config.agents.defaults.local_api_base;
                    let port = base
                        .split(':')
                        .last()
                        .and_then(|p| p.split('/').next())
                        .and_then(|p| p.parse::<u16>().ok())
                        .unwrap_or(ctx.config.agents.defaults.lms_port);
                    (extract_url_host(base), port)
                };
                if !ctx.config.trio.router_model.is_empty() {
                    let _ = crate::lms::unload_model(&lms_host, lms_port, &ctx.config.trio.router_model, ctx.config.timeouts.lms_unload_secs).await;
                }
                if !ctx.config.trio.specialist_model.is_empty() {
                    let _ = crate::lms::unload_model(&lms_host, lms_port, &ctx.config.trio.specialist_model, ctx.config.timeouts.lms_unload_secs).await;
                }
            }

            ctx.srv.shutdown();

            // Shut down auxiliary mlx-lm server if we spawned one.

            // Run skill cleanup commands (e.g. stop background audio).
            {
                let ws = ctx.config.workspace_path();
                let loader = crate::agent::skills::SkillsLoader::new(&ws, None);
                for (name, cmd) in loader.get_cleanup_commands() {
                    debug!(skill = %name, cmd = %cmd, "running skill cleanup");
                    let expanded = cmd.replace('~', &dirs::home_dir().unwrap_or_default().to_string_lossy());
                    let _ = std::process::Command::new("sh")
                        .arg("-c")
                        .arg(&expanded)
                        .stdout(std::process::Stdio::null())
                        .stderr(std::process::Stdio::null())
                        .status();
                }
            }

            // Safety net: kill any managed child processes whose Drop may not
            // have fired (e.g. Arc still held elsewhere).
            crate::agent::pid_file::cleanup_stale_pids();
            crate::agent::pid_file::release_agent_singleton();

            // On exit, force a reflection pass if any working memory has
            // accumulated — threshold=0 means "reflect if there is anything".
            // This ensures facts from the just-completed session are distilled
            // into MEMORY.md before the process exits.
            {
                let core = ctx.core_handle.swappable();
                if core.memory_enabled {
                    let reflector = Reflector::new(
                        core.memory_provider.clone(),
                        core.memory_model.clone(),
                        &core.workspace,
                        0, // threshold=0: reflect whenever there is any content
                    );
                    if reflector.should_reflect() {
                        info!("Exit: reflecting on accumulated working memory (background)...");
                        let reflection_handle = tokio::spawn(async move {
                            match reflector.reflect().await {
                                Ok(()) => info!("Exit reflection complete — MEMORY.md updated"),
                                Err(e) => warn!("Exit reflection failed: {}", e),
                            }
                        });
                        // Wait up to 5s for reflection to complete; don't block exit indefinitely
                        let _ = tokio::time::timeout(
                            std::time::Duration::from_secs(5),
                            reflection_handle,
                        ).await;
                    }
                }
            }

            // Re-index sessions in-process so the latest conversation is
            // immediately searchable via recall in the next session.
            index_sessions_background();

            println!("Goodbye!");
            // Print session resume hint so the user can pick up where they left off.
            // Try get_latest_session first (searches by session_key), then fall back to
            // get_session (searches by ID) so resumed-by-ID sessions also print the hint.
            {
                let core = ctx.core_handle.swappable();
                let meta = core.sessions.get_latest_session(&ctx.session_id).await
                    .or(core.sessions.get_session(&ctx.session_id).await);
                if let Some(meta) = meta {
                    eprintln!("Resume this session with: nanobot sessions resume {}", meta.id);
                }
            }
        }
    });
}

// ============================================================================
// Post-session indexing
// ============================================================================

/// Re-index sessions so the latest conversation is searchable via the recall
/// tool in the next session. Runs the in-process session indexer (JSONL →
/// SESSION_*.md + knowledge store ingestion). Fire-and-forget: errors are
/// logged but never block shutdown.
fn index_sessions_background() {
    let sessions_dir = match dirs::home_dir() {
        Some(h) => h.join(".nanobot/sessions"),
        None => {
            warn!("Cannot determine home directory for session indexing");
            return;
        }
    };
    let memory_sessions_dir = match dirs::home_dir() {
        Some(h) => h.join(".nanobot/workspace/memory/sessions"),
        None => return,
    };

    let (indexed, skipped, errors) =
        crate::agent::session_indexer::index_sessions(&sessions_dir, &memory_sessions_dir);

    if indexed > 0 || errors > 0 {
        debug!(
            "Session indexing complete: {} indexed, {} skipped, {} errors",
            indexed, skipped, errors
        );
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shutdown_never_stops_resident_higgs() {
        // Higgs is a resident sidecar — kept warm by the keepalive ping and
        // reused across launches. A nanobot exit must NEVER stop it: not the
        // instance nanobot spawned, not one the user started externally.
        // Regression: shutdown SIGKILLed Higgs on every exit, reloading the 35B
        // each launch and killing user-started servers.
        let mut higgs = ServerState::new("8000".to_string());
        higgs.engine = InferenceEngine::Higgs;
        assert!(
            !higgs.shutdown_plan().stop_higgs,
            "resident Higgs must survive a nanobot shutdown"
        );

        // LM Studio policy is unchanged: stop only when nanobot manages it,
        // never an externally-run instance.
        let mut managed_lms = ServerState::new("1234".to_string());
        managed_lms.lms_managed = true;
        assert!(managed_lms.shutdown_plan().stop_lms);
        let external_lms = ServerState::new("1234".to_string());
        assert!(!external_lms.shutdown_plan().stop_lms);
    }

    /// The `decode_ms` wire marker round-trips through the parser, and the
    /// leading NUL is required (bare text must never be mistaken for a marker).
    #[test]
    fn test_parse_decode_ms_marker() {
        assert!(matches!(
            parse_control_marker("\x00decode_ms:227300"),
            Some(ControlMarker::DecodeMs(227300))
        ));
        assert!(parse_control_marker("decode_ms:5").is_none());
        assert!(parse_control_marker("\x00decode_ms:notanum").is_none());
    }

    // --- prefill_line ---

    /// The loading indicator is the animated brand mark `▞▞▞` plus elapsed time,
    /// showing the prefill `%` when the server reports it. The lit block sweeps,
    /// so the active cell index changes with elapsed time.
    #[test]
    fn test_prefill_line_formats() {
        let glyphs = |s: &str| s.chars().filter(|&c| c == '\u{259e}').count();

        // No server progress → animated mark (three ▞) + elapsed, no %.
        let p = prefill_line(3.24, None);
        assert_eq!(glyphs(&p), 3, "three-cell mark: {p}");
        assert!(p.contains("3.2s"), "{p}");
        assert!(!p.contains('%'), "{p}");

        // Server progress → mark + percentage + elapsed (linked to prefill).
        let p = prefill_line(3.24, Some((1400, 3391)));
        assert_eq!(glyphs(&p), 3, "{p}");
        assert!(p.contains("41%") && p.contains("3.2s"), "{p}");

        // The lit block sweeps: the bright (bold) span differs between phases.
        let a = prefill_line(0.0, None);
        let b = prefill_line(0.2, None); // ~1 step later at 6 cells/s
        assert_ne!(a, b, "animation must advance over time: {a:?} vs {b:?}");

        // total==0 → no percentage, just the animated mark + elapsed.
        let p = prefill_line(1.0, Some((0, 0)));
        assert_eq!(glyphs(&p), 3, "{p}");
        assert!(!p.contains('%'), "{p}");
    }

    // --- parse_ctx_arg ---

    #[test]
    fn test_parse_ctx_arg_empty_returns_none() {
        assert_eq!(parse_ctx_arg("").unwrap(), None);
        assert_eq!(parse_ctx_arg("  ").unwrap(), None);
    }

    #[test]
    fn test_parse_ctx_arg_numeric() {
        assert_eq!(parse_ctx_arg("32768").unwrap(), Some(32768));
        assert_eq!(parse_ctx_arg("4096").unwrap(), Some(4096));
        assert_eq!(parse_ctx_arg("2048").unwrap(), Some(2048));
    }

    #[test]
    fn test_parse_ctx_arg_k_suffix() {
        assert_eq!(parse_ctx_arg("32K").unwrap(), Some(32768));
        assert_eq!(parse_ctx_arg("32k").unwrap(), Some(32768));
        assert_eq!(parse_ctx_arg("4K").unwrap(), Some(4096));
        assert_eq!(parse_ctx_arg("128k").unwrap(), Some(131072));
    }

    #[test]
    fn test_parse_ctx_arg_rounds_down() {
        // 5000 → rounds to 4096 (4 * 1024)
        assert_eq!(parse_ctx_arg("5000").unwrap(), Some(4096));
        // 33000 → rounds to 32768 (32 * 1024)
        assert_eq!(parse_ctx_arg("33000").unwrap(), Some(32768));
    }

    #[test]
    fn test_parse_ctx_arg_too_small() {
        assert!(parse_ctx_arg("1024").is_err());
        assert!(parse_ctx_arg("1K").is_err());
        assert!(parse_ctx_arg("100").is_err());
    }

    #[test]
    fn test_parse_ctx_arg_invalid() {
        assert!(parse_ctx_arg("abc").is_err());
        assert!(parse_ctx_arg("32M").is_err());
        assert!(parse_ctx_arg("--").is_err());
    }

    // --- short_channel_name ---

    #[test]
    fn test_short_channel_name() {
        assert_eq!(short_channel_name("whatsapp"), "wa");
        assert_eq!(short_channel_name("telegram"), "tg");
        assert_eq!(short_channel_name("email"), "email");
        assert_eq!(short_channel_name("other"), "other");
    }

    // --- build_prompt ---

    /// The prompt is now a single consistent green caret regardless of mode
    /// (local / cloud / voice / thinking) — mode is shown in the footer instead.
    #[test]
    fn test_build_prompt_is_consistent_green_caret() {
        let caret = "\u{276f}";
        let cloud = build_prompt(false, false, false);
        assert!(cloud.contains(caret), "{cloud:?}");
        assert!(cloud.contains(crate::tui::GREEN), "{cloud:?}");
        // No legacy mode-specific markers or colours.
        assert!(!cloud.contains("L>") && !cloud.contains("~>"), "{cloud:?}");
        assert!(!cloud.contains(crate::tui::YELLOW), "{cloud:?}");
        // Local / voice / thinking all render the identical prompt.
        assert_eq!(cloud, build_prompt(true, false, false));
        assert_eq!(cloud, build_prompt(false, true, false));
        assert_eq!(cloud, build_prompt(false, false, true));
    }

    // --- ServerState ---

    #[test]
    fn test_server_state_new() {
        let state = ServerState::new("8080".to_string());
        assert_eq!(state.local_port, "8080");
        assert!(!state.lms_managed);
        assert!(state.lms_binary.is_none());
        assert_eq!(state.engine, InferenceEngine::None);
    }

    #[tokio::test]
    async fn test_server_state_kill_current_when_empty() {
        // Should not panic when there's no process to kill
        let mut state = ServerState::new("8080".to_string());
        state.kill_current(1234, 30).await;
        assert_eq!(state.engine, InferenceEngine::None);
    }

    #[test]
    fn test_server_state_shutdown_when_empty() {
        // Should not panic when there's nothing to shut down
        let mut state = ServerState::new("8080".to_string());
        state.shutdown();
        assert!(!state.lms_managed);
        assert_eq!(state.engine, InferenceEngine::None);
    }

    // --- rewind_and_clear_below (render scroll regression) ---

    #[test]
    fn rewind_escape_is_empty_when_no_prior_block() {
        assert_eq!(rewind_and_clear_below(0), "");
    }

    #[test]
    fn rewind_escape_does_not_emit_newlines() {
        // The prior implementation used `\x1b[2K\r\n` in a loop, which scrolls
        // the terminal on the bottom row. Guard against regressing to that.
        for n in 1..=10 {
            let esc = rewind_and_clear_below(n);
            assert!(!esc.contains('\n'), "n={n}: {esc:?} contains newline");
            assert!(!esc.contains('\r'), "n={n}: {esc:?} contains CR");
        }
    }

    #[test]
    fn rewind_escape_moves_up_and_clears_to_end() {
        // CSI n A = cursor up n rows; CSI J = erase cursor-to-end-of-screen.
        assert_eq!(rewind_and_clear_below(3), "\x1b[3A\x1b[J");
        assert_eq!(rewind_and_clear_below(1), "\x1b[1A\x1b[J");
    }

    // --- truncate_output ---

    #[test]
    fn test_truncate_output_short() {
        let result = truncate_output("hello\nworld", 40, 2000);
        assert_eq!(result, "hello\nworld");
    }

    #[test]
    fn test_truncate_output_max_lines() {
        let data = (0..50)
            .map(|i| format!("line {}", i))
            .collect::<Vec<_>>()
            .join("\n");
        let result = truncate_output(&data, 5, 10000);
        assert!(result.lines().count() <= 6); // 5 lines + truncated marker
        assert!(result.contains("...[truncated]"));
    }

    #[test]
    fn test_truncate_output_max_chars() {
        let data = "x".repeat(5000);
        let result = truncate_output(&data, 100, 100);
        assert!(result.len() < 200);
        assert!(result.contains("...[truncated]"));
    }

    #[test]
    fn test_truncate_output_empty() {
        let result = truncate_output("", 40, 2000);
        assert_eq!(result, "");
    }

    // --- extract_json_string_field ---

    #[test]
    fn test_extract_json_field_wellformed() {
        let got = extract_json_string_field(r#"{"command":"ls -la"}"#, "command");
        assert_eq!(got.as_deref(), Some("ls -la"));
    }

    #[test]
    fn test_extract_json_field_truncated() {
        // Clipped preview: no closing quote/brace, but the value is still readable.
        let got = extract_json_string_field(r#"{"command":"echo hello world"#, "command");
        assert_eq!(got.as_deref(), Some("echo hello world"));
    }

    #[test]
    fn test_extract_json_field_escapes() {
        let got = extract_json_string_field(r#"{"command":"grep \"foo\" ."}"#, "command");
        assert_eq!(got.as_deref(), Some(r#"grep "foo" ."#));
    }

    #[test]
    fn test_extract_json_field_missing() {
        assert!(extract_json_string_field(r#"{"path":"/tmp"}"#, "command").is_none());
    }

    // --- extract_tool_context (exec shows the command) ---

    #[test]
    fn test_extract_tool_context_exec_shows_command() {
        let ctx = extract_tool_context("exec", "some stdout output", r#"{"command":"ls -la"}"#);
        assert_eq!(ctx, "$ ls -la");
    }

    #[test]
    fn test_extract_tool_context_exec_first_line_only() {
        // Multi-line commands collapse to the first line for the status line.
        let ctx = extract_tool_context("exec", "out", "{\"command\":\"echo a\\nrm -rf b\"}");
        assert_eq!(ctx, "$ echo a");
    }

    #[test]
    fn test_extract_tool_context_exec_clips_long_command() {
        let long = "x".repeat(200);
        let args = format!(r#"{{"command":"{}"}}"#, long);
        let ctx = extract_tool_context("exec", "out", &args);
        assert!(ctx.starts_with("$ "));
        assert!(ctx.ends_with('\u{2026}'), "expected ellipsis, got: {ctx}");
    }

    #[test]
    fn test_extract_tool_context_exec_no_args() {
        assert_eq!(extract_tool_context("exec", "out", ""), "");
    }

    #[test]
    fn test_extract_tool_context_surfaces_input_params() {
        // The tool input (not output) is what's shown on the status line.
        assert_eq!(
            extract_tool_context(
                "web_search",
                "lots of results",
                r#"{"query":"space news 2024"}"#
            ),
            "space news 2024"
        );
        assert_eq!(
            extract_tool_context("web_fetch", "html", r#"{"url":"https://example.com"}"#),
            "https://example.com"
        );
        // Unknown tools fall back to the first recognisable string argument.
        assert_eq!(
            extract_tool_context("some_new_tool", "out", r#"{"input":"hello there"}"#),
            "hello there"
        );
        // Long params are clipped with an ellipsis.
        let long = "q".repeat(200);
        let ctx = extract_tool_context("web_search", "out", &format!(r#"{{"query":"{long}"}}"#));
        assert!(ctx.ends_with('\u{2026}'), "expected ellipsis: {ctx}");
    }

    // --- higgs_keepalive_secs (warm-keep decision) ---

    #[test]
    fn test_keepalive_local_higgs_default() {
        let s = higgs_keepalive_secs("higgs", "http://127.0.0.1:8000/v1", None);
        assert_eq!(s, Some(DEFAULT_HIGGS_KEEPALIVE_SECS));
        // localhost host form also counts.
        assert_eq!(
            higgs_keepalive_secs("higgs", "http://localhost:8000/v1", None),
            Some(DEFAULT_HIGGS_KEEPALIVE_SECS)
        );
    }

    #[test]
    fn test_keepalive_env_override_and_disable() {
        assert_eq!(
            higgs_keepalive_secs("higgs", "http://127.0.0.1:8000/v1", Some("30")),
            Some(30)
        );
        // "0" disables.
        assert_eq!(
            higgs_keepalive_secs("higgs", "http://127.0.0.1:8000/v1", Some("0")),
            None
        );
        // Non-numeric → default.
        assert_eq!(
            higgs_keepalive_secs("higgs", "http://127.0.0.1:8000/v1", Some("nope")),
            Some(DEFAULT_HIGGS_KEEPALIVE_SECS)
        );
    }

    #[test]
    fn test_keepalive_skips_remote_and_non_higgs() {
        // Remote peer → never kept warm by us.
        assert_eq!(
            higgs_keepalive_secs("higgs", "http://192.168.1.22:8000/v1", None),
            None
        );
        // Non-higgs backend → no keepalive (LMS/oMLX manage themselves).
        assert_eq!(
            higgs_keepalive_secs("lms", "http://127.0.0.1:8000/v1", None),
            None
        );
    }
}
