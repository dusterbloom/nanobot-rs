//! TUI-related functions: ANSI constants, status bars, banners, and voice helpers.

use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::agent::agent_loop::SharedCoreHandle;
use crate::config::loader::load_config;

// ============================================================================
// ANSI Escape Sequences
// ============================================================================

pub const RESET: &str = "\x1b[0m";
pub const BOLD: &str = "\x1b[1m";
pub const DIM: &str = "\x1b[2m";
pub const CYAN: &str = "\x1b[36m";
pub const GREEN: &str = "\x1b[32m";
pub const YELLOW: &str = "\x1b[33m";
pub const RED: &str = "\x1b[31m";
pub const MAGENTA: &str = "\x1b[35m";
pub const WHITE: &str = "\x1b[97m";
pub const CLEAR_SCREEN: &str = "\x1b[2J\x1b[H";
pub const HIDE_CURSOR: &str = "\x1b[?25l";
pub const SHOW_CURSOR: &str = "\x1b[?25h";

/// Print the nanobot demoscene-style ASCII logo.
pub fn print_logo() {
    println!("  {BOLD}{CYAN} _____             _       _   {RESET}");
    println!("  {BOLD}{WHITE}|   | |___ ___ ___| |_ ___| |_ {RESET}");
    println!("  {BOLD}{WHITE}| | | | .'|   | . | . | . |  _|{RESET}");
    println!("  {BOLD}{CYAN}|_|___|__,|_|_|___|___|___|_|  {RESET}");
}

/// Animated loading sequence.
pub fn loading_animation(message: &str) {
    let frames = ["   ", ".  ", ".. ", "..."];
    print!("{HIDE_CURSOR}");
    for i in 0..8 {
        print!("\r  {DIM}{}{}{RESET}  ", message, frames[i % frames.len()]);
        std::io::stdout().flush().ok();
        std::thread::sleep(std::time::Duration::from_millis(150));
    }
    print!("\r{}\r", " ".repeat(60)); // clear the line
    print!("{SHOW_CURSOR}");
    std::io::stdout().flush().ok();
}

// ============================================================================
// Terminal Utilities
// ============================================================================

/// Count terminal rows that `text` occupies when printed raw, accounting for line wrapping.
///
/// Each `\n`-delimited line takes `ceil(len / width)` rows (minimum 1).
/// Adds `extra` for surrounding blank lines / println calls.
pub(crate) fn terminal_rows(text: &str, extra: usize) -> usize {
    let width = termimad::crossterm::terminal::size().map(|(w, _)| w as usize).unwrap_or(80);
    let rows: usize = text.split('\n').map(|line| {
        let len = line.len();
        if len == 0 { 1 } else { (len + width - 1) / width }
    }).sum();
    rows + extra
}

/// Format a number with thousands separators (e.g. 12430 -> "12,430").
pub(crate) fn format_thousands(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

// ============================================================================
// Status Bar & Banners
// ============================================================================

/// Format token count as compact string (e.g. 12430 -> "12.4K", 1200000 -> "1.2M").
fn format_tokens(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

/// Print a compact status bar after each agent response.
///
/// Shows: ctx tokens/max | msgs | tools called | working memory | channels | agents | turn
pub(crate) fn print_status_bar(core_handle: &SharedCoreHandle, channel_names: &[&str], subagent_count: usize) {
    // Read counters directly — no RwLock needed.
    let counters = &core_handle.counters;
    let used = counters.last_context_used.load(Ordering::Relaxed) as usize;
    let max = counters.last_context_max.load(Ordering::Relaxed) as usize;
    let turn = counters.learning_turn_counter.load(Ordering::Relaxed);
    let msg_count = counters.last_message_count.load(Ordering::Relaxed) as usize;
    let wm_tokens = counters.last_working_memory_tokens.load(Ordering::Relaxed) as usize;

    let pct = if max > 0 { (used * 100) / max } else { 0 };
    let ctx_color = match pct {
        0..=49 => GREEN,
        50..=79 => YELLOW,
        _ => RED,
    };

    let mut parts: Vec<String> = Vec::new();

    // Context: 12.4K/1M (colored by usage)
    parts.push(format!(
        "ctx {}{}{}/{}{}",
        ctx_color,
        BOLD,
        format_tokens(used),
        format_tokens(max),
        RESET
    ));

    // Message count
    parts.push(format!("msgs:{}", msg_count));

    // Tools called this turn
    let tools_called: Vec<String> = counters.last_tools_called.lock()
        .map(|g| g.clone())
        .unwrap_or_default();
    if !tools_called.is_empty() {
        let mut sorted = tools_called.clone();
        sorted.sort();
        parts.push(format!("tools:{} ({})", sorted.len(), sorted.join(", ")));
    }

    // Working memory tokens
    if wm_tokens > 0 {
        parts.push(format!("wm:{}tok", format_tokens(wm_tokens)));
    }

    if !channel_names.is_empty() {
        parts.push(format!(
            "{}{}{}",
            CYAN,
            channel_names.join(" "),
            RESET
        ));
    }

    if subagent_count > 0 {
        parts.push(format!(
            "{} agent{}",
            subagent_count,
            if subagent_count > 1 { "s" } else { "" }
        ));
    }

    parts.push(format!("t:{}", turn));

    println!("  {}{}{}", DIM, parts.join(" | "), RESET);
}

/// Print the current mode banner (compact, for mode switches mid-session).
pub(crate) fn print_mode_banner(local_port: &str) {
    let is_local = crate::LOCAL_MODE.load(Ordering::SeqCst);
    println!();
    if is_local {
        println!("  {BOLD}{YELLOW}LOCAL MODE{RESET} {DIM}llama.cpp on port {local_port}{RESET}");
        let props_url = format!("http://localhost:{}/props", local_port);
        if let Ok(resp) = reqwest::blocking::get(&props_url) {
            if let Ok(json) = resp.json::<serde_json::Value>() {
                if let Some(model) = json
                    .get("default_generation_settings")
                    .and_then(|s| s.get("model"))
                    .and_then(|m| m.as_str())
                {
                    let model_name = std::path::Path::new(model)
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or(model);
                    println!("  {DIM}Model: {RESET}{GREEN}{model_name}{RESET}");
                }
                if let Some(n_ctx) = json
                    .get("default_generation_settings")
                    .and_then(|s| s.get("n_ctx"))
                    .and_then(|n| n.as_u64())
                {
                    let n_parallel = json
                        .get("default_generation_settings")
                        .and_then(|s| s.get("n_parallel"))
                        .and_then(|n| n.as_u64())
                        .or_else(|| json.get("n_parallel").and_then(|n| n.as_u64()))
                        .unwrap_or(1)
                        .max(1);
                    let per_request = (n_ctx / n_parallel).max(1);
                    if n_parallel > 1 {
                        println!(
                            "  {DIM}Context: {RESET}{GREEN}{}K{RESET}{DIM} ({}K total / parallel {}){RESET}",
                            per_request / 1024,
                            n_ctx / 1024,
                            n_parallel
                        );
                    } else {
                        println!("  {DIM}Context: {RESET}{GREEN}{}K{RESET}", n_ctx / 1024);
                    }
                }
            }
        }
    } else {
        let config = load_config(None);
        println!(
            "  {BOLD}{CYAN}CLOUD MODE{RESET} {DIM}{}{RESET}",
            config.agents.defaults.model
        );
    }
    println!();
}

/// Full startup splash: clear screen, ASCII logo, mode info, hints.
pub(crate) fn print_startup_splash(local_port: &str) {
    // Clear the terminal for a fresh start.
    print!("{CLEAR_SCREEN}");
    std::io::Write::flush(&mut std::io::stdout()).ok();

    print_logo();

    let is_local = crate::LOCAL_MODE.load(Ordering::SeqCst);
    if is_local {
        println!("  {BOLD}{YELLOW}LOCAL{RESET} {DIM}llama.cpp :{local_port}{RESET}");
    } else {
        let config = load_config(None);
        println!(
            "  {BOLD}{CYAN}CLOUD{RESET} {DIM}{}{RESET}",
            config.agents.defaults.model
        );
    }
    println!("  {DIM}v{}  |  /local  /model  /voice  Ctrl+C quit{RESET}", crate::VERSION);
    println!();

    // Brief loading animation
    loading_animation("Initializing agent");
}

// ============================================================================
// Voice Mode Helpers
// ============================================================================

/// Strip markdown formatting, code blocks, emojis, and special characters
/// so that TTS receives only clean natural language text.
#[cfg(feature = "voice")]
pub(crate) fn strip_markdown_for_tts(text: &str) -> String {
    let mut out = String::new();
    let mut in_code_block = false;

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("```") {
            in_code_block = !in_code_block;
            continue;
        }
        if in_code_block {
            continue;
        }

        let line = trimmed.trim_start_matches('#').trim();
        if line.is_empty() {
            continue;
        }

        for c in line.chars() {
            match c {
                'a'..='z' | 'A'..='Z' | '0'..='9' => out.push(c),
                ' ' | '.' | ',' | '!' | '?' | ';' | ':' | '\'' | '"' | '-' | '(' | ')' => {
                    out.push(c)
                }
                '*' | '_' | '`' | '~' | '[' | ']' | '|' | '#' => {} // strip markdown syntax
                _ if c.is_alphabetic() => out.push(c),              // keep non-English letters
                _ => {}                                             // strip emojis, arrows, etc.
            }
        }
        out.push(' ');
    }

    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Flush any buffered terminal input (e.g. extra Enter keypresses during recording).
#[cfg(feature = "voice")]
pub(crate) fn drain_stdin() {
    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd;
        let fd = std::io::stdin().as_raw_fd();
        unsafe {
            libc::tcflush(fd, libc::TCIFLUSH);
        }
    }
}

/// Spawn a watcher thread for interrupt detection (Enter or Ctrl+Space).
/// Returns the thread handle that resolves to `true` if interrupted.
#[cfg(feature = "voice")]
pub(crate) fn spawn_interrupt_watcher(
    cancel: Arc<AtomicBool>,
    done: Arc<AtomicBool>,
) -> std::thread::JoinHandle<bool> {
    use crossterm::event::{self, Event, KeyCode, KeyModifiers};
    use crossterm::terminal;

    std::thread::spawn(move || {
        terminal::enable_raw_mode().ok();
        let mut interrupted = false;
        while !done.load(Ordering::Relaxed) {
            if event::poll(std::time::Duration::from_millis(100)).unwrap_or(false) {
                if let Ok(Event::Key(key)) = event::read() {
                    let is_interrupt = key.code == KeyCode::Enter
                        || (key.code == KeyCode::Char(' ')
                            && key.modifiers.contains(KeyModifiers::CONTROL));
                    if is_interrupt {
                        cancel.store(true, Ordering::Relaxed);
                        interrupted = true;
                        break;
                    }
                }
            }
        }
        terminal::disable_raw_mode().ok();
        interrupted
    })
}

/// Speak with TTS while watching for user interrupt (Enter or Ctrl+Space).
/// Returns true if the user interrupted (wants to speak next).
#[cfg(feature = "voice")]
pub(crate) fn speak_interruptible(vs: &mut crate::voice::VoiceSession, text: &str, lang: &str) -> bool {
    vs.clear_cancel();
    let cancel = vs.cancel_flag();
    let done = Arc::new(AtomicBool::new(false));
    let done2 = done.clone();

    // Spawn thread to watch for keypress during TTS
    let watcher = spawn_interrupt_watcher(cancel, done2);

    if let Err(e) = vs.speak(text, lang) {
        eprintln!("TTS error: {}", e);
    }

    // Signal watcher to stop and collect result
    done.store(true, Ordering::Relaxed);
    let interrupted = watcher.join().unwrap_or(false);

    if interrupted {
        vs.stop_playback();
    }

    interrupted
}

#[cfg(feature = "voice")]
pub(crate) enum VoiceAction {
    Record,
    Text(String),
    Exit,
}

/// Read input in voice mode using crossterm raw terminal.
/// Ctrl+Space or Enter (empty) → Record, typed text + Enter → Text, Ctrl+C → Exit.
#[cfg(feature = "voice")]
pub(crate) fn voice_read_input() -> VoiceAction {
    use crossterm::event::{self, Event, KeyCode, KeyModifiers};
    use crossterm::terminal;

    if terminal::enable_raw_mode().is_err() {
        // Fallback: just use regular read_line
        let mut line = String::new();
        return match io::stdin().read_line(&mut line) {
            Ok(0) | Err(_) => VoiceAction::Exit,
            _ => {
                let trimmed = line.trim().to_string();
                if trimmed.is_empty() {
                    VoiceAction::Record
                } else {
                    VoiceAction::Text(trimmed)
                }
            }
        };
    }

    let mut buffer = String::new();

    let result = loop {
        match event::read() {
            Ok(Event::Key(key)) => {
                // Ctrl+Space → record
                if (key.code == KeyCode::Char(' ') && key.modifiers.contains(KeyModifiers::CONTROL))
                    || (key.code == KeyCode::Char('\0'))
                {
                    print!("\r\n");
                    io::stdout().flush().ok();
                    break VoiceAction::Record;
                }
                // Ctrl+C → exit
                if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
                    print!("\r\n");
                    io::stdout().flush().ok();
                    break VoiceAction::Exit;
                }
                // Enter
                if key.code == KeyCode::Enter {
                    print!("\r\n");
                    io::stdout().flush().ok();
                    if buffer.is_empty() {
                        break VoiceAction::Record;
                    }
                    break VoiceAction::Text(buffer);
                }
                // Backspace
                if key.code == KeyCode::Backspace {
                    if buffer.pop().is_some() {
                        print!("\x08 \x08");
                        io::stdout().flush().ok();
                    }
                    continue;
                }
                // Regular character (no ctrl/alt modifier)
                if let KeyCode::Char(c) = key.code {
                    if !key.modifiers.contains(KeyModifiers::CONTROL)
                        && !key.modifiers.contains(KeyModifiers::ALT)
                    {
                        buffer.push(c);
                        print!("{}", c);
                        io::stdout().flush().ok();
                    }
                }
            }
            Ok(_) => {} // ignore mouse, resize, etc.
            Err(_) => break VoiceAction::Exit,
        }
    };

    terminal::disable_raw_mode().ok();
    result
}
