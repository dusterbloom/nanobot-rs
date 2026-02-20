//! TUI-related functions: ANSI constants, status bars, banners, and voice helpers.

use std::io::{self, BufWriter, Write};
use std::sync::atomic::{AtomicBool, AtomicU16, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use unicode_width::UnicodeWidthStr;

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
pub const GREY: &str = "\x1b[90m";
pub const CLEAR_SCREEN: &str = "\x1b[2J\x1b[H";
pub const HIDE_CURSOR: &str = "\x1b[?25l";
pub const SHOW_CURSOR: &str = "\x1b[?25h";

// ============================================================================
// Terminal Writer (synchronized stdout)
// ============================================================================

/// Global terminal writer — all TUI output should go through this to prevent
/// interleaved writes from concurrent tasks (streaming, tool events, channels).
///
/// Uses DEC private mode 2026 (synchronized output) brackets when available
/// to batch writes and prevent flicker.
pub struct TerminalWriter {
    inner: Mutex<BufWriter<io::Stdout>>,
}

impl TerminalWriter {
    fn new() -> Self {
        Self {
            inner: Mutex::new(BufWriter::new(io::stdout())),
        }
    }

    /// Write a string to stdout under the lock, then flush.
    pub fn write_str(&self, s: &str) {
        if let Ok(mut w) = self.inner.lock() {
            let _ = w.write_all(s.as_bytes());
            let _ = w.flush();
        }
    }

    /// Write a string followed by newline.
    pub fn writeln(&self, s: &str) {
        if let Ok(mut w) = self.inner.lock() {
            let _ = w.write_all(s.as_bytes());
            let _ = w.write_all(b"\n");
            let _ = w.flush();
        }
    }

    /// Execute a closure with exclusive access to the buffered writer.
    /// The writer is flushed after the closure returns.
    pub fn with_writer<F>(&self, f: F)
    where
        F: FnOnce(&mut BufWriter<io::Stdout>),
    {
        if let Ok(mut w) = self.inner.lock() {
            f(&mut w);
            let _ = w.flush();
        }
    }
}

/// Get the global terminal writer singleton.
pub fn terminal_writer() -> &'static TerminalWriter {
    static WRITER: OnceLock<TerminalWriter> = OnceLock::new();
    WRITER.get_or_init(TerminalWriter::new)
}

// ============================================================================
// Raw Mode Guard
// ============================================================================

/// Global flag tracking whether raw mode is currently active.
/// Prevents double-enter races across the 4 raw mode entry points
/// (spawn_input_watcher, spawn_interrupt_watcher, voice_read_input, voice recording).
pub(crate) static RAW_MODE_ACTIVE: AtomicBool = AtomicBool::new(false);

/// Enter raw mode if not already active. Returns true if this call entered raw mode.
pub fn enter_raw_mode() -> bool {
    if RAW_MODE_ACTIVE
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_ok()
    {
        termimad::crossterm::terminal::enable_raw_mode().ok();
        true
    } else {
        false
    }
}

/// Exit raw mode if this caller originally entered it (pass the return value of `enter_raw_mode`).
pub fn exit_raw_mode(owned: bool) {
    if owned
        && RAW_MODE_ACTIVE
            .compare_exchange(true, false, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
    {
        termimad::crossterm::terminal::disable_raw_mode().ok();
    }
}

/// Force-exit raw mode regardless of ownership (for panic/cleanup paths).
pub fn force_exit_raw_mode() {
    if RAW_MODE_ACTIVE.swap(false, Ordering::SeqCst) {
        termimad::crossterm::terminal::disable_raw_mode().ok();
    }
}

// ============================================================================
// SIGWINCH / Terminal Resize Handling
// ============================================================================

/// Cached terminal dimensions. Invalidated on SIGWINCH.
static CACHED_WIDTH: AtomicU16 = AtomicU16::new(0);
static CACHED_HEIGHT: AtomicU16 = AtomicU16::new(0);

/// Register the SIGWINCH signal handler that invalidates cached terminal dimensions.
/// Call once at REPL startup.
pub fn register_resize_handler() {
    #[cfg(unix)]
    {
        // SIGWINCH handler: clear cached dimensions so next query re-reads from OS.
        unsafe {
            libc::signal(libc::SIGWINCH, sigwinch_handler as libc::sighandler_t);
        }
    }
}

#[cfg(unix)]
extern "C" fn sigwinch_handler(_sig: libc::c_int) {
    CACHED_WIDTH.store(0, Ordering::Relaxed);
    CACHED_HEIGHT.store(0, Ordering::Relaxed);
}

/// Invalidate cached terminal dimensions (e.g. after scroll region changes).
pub fn invalidate_terminal_size() {
    CACHED_WIDTH.store(0, Ordering::Relaxed);
    CACHED_HEIGHT.store(0, Ordering::Relaxed);
}

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
    let width = terminal_width();
    let rows: usize = text
        .split('\n')
        .map(|line| {
            let len = line.width();
            if len == 0 {
                1
            } else {
                (len + width - 1) / width
            }
        })
        .sum();
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
// Input Bar (Claude Code-style)
// ============================================================================

/// Get terminal width, defaulting to 80 if unavailable.
/// Uses cached value that is invalidated on SIGWINCH.
pub(crate) fn terminal_width() -> usize {
    let cached = CACHED_WIDTH.load(Ordering::Relaxed);
    if cached > 0 {
        return cached as usize;
    }
    let (w, h) = termimad::crossterm::terminal::size().unwrap_or((80, 24));
    CACHED_WIDTH.store(w, Ordering::Relaxed);
    CACHED_HEIGHT.store(h, Ordering::Relaxed);
    w as usize
}

/// Get terminal height, defaulting to 24 if unavailable.
/// Uses cached value that is invalidated on SIGWINCH.
pub(crate) fn terminal_height() -> usize {
    let cached = CACHED_HEIGHT.load(Ordering::Relaxed);
    if cached > 0 {
        return cached as usize;
    }
    let (w, h) = termimad::crossterm::terminal::size().unwrap_or((80, 24));
    CACHED_WIDTH.store(w, Ordering::Relaxed);
    CACHED_HEIGHT.store(h, Ordering::Relaxed);
    h as usize
}

/// Render a Claude Code-style input context bar pinned to the bottom of the terminal.
///
/// Layout (at terminal bottom):
/// ```text
/// ──────────────────────────────────────────────
///   ~/Dev/nanobot · opus-4-6 · 🧠 thinking
///   ⏵⏵ /t think · /l local · /v voice · ctx 12K/1M
/// ```
///
/// The bar is rendered at the bottom 3 rows of the terminal. The cursor is
/// then repositioned so readline draws the prompt above the bar. The area
/// between the current output and the bar is left empty (scroll region).
///
/// Returns the number of bar lines (for cleanup/erase after input).
pub(crate) fn render_input_bar(
    core_handle: &crate::agent::agent_loop::SharedCoreHandle,
    channel_names: &[&str],
    subagent_count: usize,
    push_content: bool,
) -> usize {
    use std::io::Write as _;
    use std::sync::atomic::Ordering;

    let counters = &core_handle.counters;
    let width = terminal_width().min(100);
    let height = terminal_height();
    let separator = "─".repeat(width);

    // Gather info
    let thinking_on = counters.thinking_budget.load(Ordering::Relaxed) > 0;
    let used = counters.last_context_used.load(Ordering::Relaxed) as usize;
    let max = counters.last_context_max.load(Ordering::Relaxed) as usize;

    let cwd = std::env::current_dir()
        .map(|p| {
            let home = std::env::var("HOME").unwrap_or_default();
            let s = p.display().to_string();
            if !home.is_empty() && s.starts_with(&home) {
                format!("~{}", &s[home.len()..])
            } else {
                s
            }
        })
        .unwrap_or_else(|_| "?".into());

    let model_name = {
        let core = core_handle.swappable();
        core.model.clone()
    };

    let think_str = if thinking_on {
        format!(" · {GREY}\u{1f9e0} thinking{RESET}")
    } else {
        String::new()
    };

    let mut hints: Vec<String> = Vec::new();
    hints.push(format!("{DIM}⏵⏵ /t think · /l local · /v voice{RESET}"));

    if subagent_count > 0 {
        hints.push(format!(
            "{CYAN}{} agent{}{RESET}",
            subagent_count,
            if subagent_count > 1 { "s" } else { "" }
        ));
    }

    if !channel_names.is_empty() {
        hints.push(format!("{CYAN}{}{RESET}", channel_names.join(" ")));
    }

    if max > 0 {
        let pct = (used * 100) / max;
        let ctx_color = match pct {
            0..=49 => GREEN,
            50..=79 => YELLOW,
            _ => RED,
        };
        hints.push(format!(
            "ctx {ctx_color}{}{RESET}/{DIM}{}{RESET}",
            format_tokens(used),
            format_tokens(max)
        ));
    }

    let bar_lines = 3usize;
    // bar_row is the terminal row where the separator starts (1-indexed)
    let bar_row = height.saturating_sub(bar_lines) + 1;
    // prompt_row is where the user types — just above the bar
    let prompt_row = bar_row.saturating_sub(1);

    if push_content {
        // First render: push existing content up to make room for the bar.
        print!("\x1b[r"); // reset scroll region to full screen
        print!("\x1b[{};1H", height); // move to last row
        for _ in 0..bar_lines + 1 {
            println!(); // push content up by scrolling
        }
    } else {
        // Refresh: save cursor, update bar content in place, restore cursor.
        print!("\x1b[s");
    }

    // Render the bar at its fixed position (outside scroll region)
    print!("\x1b[{};1H", bar_row);
    print!("\x1b[J"); // clear from bar_row to end of screen
    println!("{DIM}{separator}{RESET}");
    println!("  {DIM}{cwd} · {RESET}{GREEN}{model_name}{RESET}{think_str}");
    print!("  {}", hints.join(" · "));

    // Set scroll region to rows 1..prompt_row so text never overwrites the bar
    print!("\x1b[1;{}r", prompt_row);

    if push_content {
        // Position cursor at prompt_row for initial input
        print!("\x1b[{};1H\x1b[2K", prompt_row);
    } else {
        // Restore cursor to where it was (inside scroll region)
        print!("\x1b[u");
    }
    std::io::stdout().flush().ok();

    bar_lines
}

// Status Bar & Banners
// ============================================================================

/// Reset the terminal scroll region to the full screen.
/// Call this before AI output starts streaming so text can use all rows.
pub(crate) fn reset_scroll_region() {
    print!("\x1b[r"); // reset scroll region to full terminal
    std::io::stdout().flush().ok();
}

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
pub(crate) fn print_status_bar(
    core_handle: &SharedCoreHandle,
    channel_names: &[&str],
    subagent_count: usize,
) {
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
    let tools_called: Vec<String> = counters
        .last_tools_called
        .lock()
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
        parts.push(format!("{}{}{}", CYAN, channel_names.join(" "), RESET));
    }

    if subagent_count > 0 {
        parts.push(format!(
            "{} agent{}",
            subagent_count,
            if subagent_count > 1 { "s" } else { "" }
        ));
    }

    // Thinking mode indicator
    if core_handle.counters.thinking_budget.load(Ordering::Relaxed) > 0 {
        parts.push(format!("{GREY}\u{1f9e0}{RESET}"));
    }

    parts.push(format!("t:{}", turn));

    println!("  {}{}{}", DIM, parts.join(" | "), RESET);
}

/// Print the current mode banner (compact, for mode switches mid-session).
pub(crate) fn print_mode_banner(local_port: &str) {
    let is_local = crate::LOCAL_MODE.load(Ordering::SeqCst);
    println!();
    if is_local {
        let props_url = format!("http://localhost:{}/props", local_port);
        println!("  {BOLD}{YELLOW}LOCAL MODE{RESET} {DIM}LM Studio on port {local_port}{RESET}");
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
        let config = load_config(None);
        let base = &config.agents.defaults.local_api_base;
        if !base.is_empty() {
            println!("  {BOLD}{YELLOW}LOCAL{RESET} {DIM}{base}{RESET}");
        } else {
            println!("  {BOLD}{YELLOW}LOCAL{RESET} {DIM}LM Studio :{local_port}{RESET}");
        }
    } else {
        let config = load_config(None);
        println!(
            "  {BOLD}{CYAN}CLOUD{RESET} {DIM}{}{RESET}",
            config.agents.defaults.model
        );
    }
    println!(
        "  {DIM}v{}  |  /local  /model  /voice  Ctrl+C quit{RESET}",
        crate::VERSION
    );
    println!();

    // Brief loading animation
    loading_animation("Initializing agent");
}

// ============================================================================
// Voice Mode Helpers
// ============================================================================

/// Strip `<thinking>...</thinking>` blocks from text. These contain internal
/// chain-of-thought from reasoning models and should never be spoken or displayed.
/// Works across single-line and multi-line blocks.
#[cfg(feature = "voice")]
fn strip_thinking_blocks(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut remaining = text;
    while let Some(start) = remaining.find("<thinking>") {
        result.push_str(&remaining[..start]);
        if let Some(end) = remaining[start..].find("</thinking>") {
            let after_tag = start + end + "</thinking>".len();
            remaining = &remaining[after_tag..];
            // Skip leading whitespace/newlines after the closing tag
            remaining = remaining.trim_start_matches(|c: char| c == '\n' || c == '\r');
        } else {
            // Unclosed tag — discard everything from <thinking> onward
            remaining = "";
            break;
        }
    }
    result.push_str(remaining);
    result
}

/// Strip markdown formatting, code blocks, emojis, and special characters
/// so that TTS receives only clean natural language text.
#[cfg(feature = "voice")]
pub(crate) fn strip_markdown_for_tts(text: &str) -> String {
    // First: strip any <thinking>...</thinking> blocks (safety net — these
    // should already be removed at the provider level, but defense in depth).
    let cleaned_text = strip_thinking_blocks(text);
    let mut out = String::new();
    let mut in_code_block = false;

    for line in cleaned_text.lines() {
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

    std::thread::spawn(move || {
        let owned = enter_raw_mode();
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
        exit_raw_mode(owned);
        interrupted
    })
}

/// Speak with TTS while watching for user interrupt (Enter or Ctrl+Space).
/// Returns true if the user interrupted (wants to speak next).
#[cfg(feature = "voice")]
pub(crate) fn speak_interruptible(
    vs: &mut crate::voice::VoiceSession,
    text: &str,
    lang: &str,
) -> bool {
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

    let owned = enter_raw_mode();
    if !owned && !RAW_MODE_ACTIVE.load(Ordering::SeqCst) {
        // Couldn't enter raw mode at all — fallback to line read.
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

    exit_raw_mode(owned);
    result
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- terminal_rows (unicode-width) ---

    #[test]
    fn test_terminal_rows_ascii() {
        // "hello" is 5 display chars — fits in one row on an 80-col terminal.
        let rows = terminal_rows("hello", 0);
        assert_eq!(rows, 1);
    }

    #[test]
    fn test_terminal_rows_empty() {
        let rows = terminal_rows("", 0);
        assert_eq!(rows, 1); // empty line still occupies 1 row
    }

    #[test]
    fn test_terminal_rows_multiline() {
        let rows = terminal_rows("line1\nline2\nline3", 0);
        assert_eq!(rows, 3);
    }

    #[test]
    fn test_terminal_rows_extra() {
        let rows = terminal_rows("hello", 2);
        assert_eq!(rows, 3); // 1 row + 2 extra
    }

    #[test]
    fn test_terminal_rows_cjk_double_width() {
        // CJK characters are 2 columns wide. 3 chars = 6 display columns.
        // This should still fit in 1 row on an 80-col terminal.
        let rows = terminal_rows("你好世", 0);
        assert_eq!(rows, 1);
    }

    #[test]
    fn test_terminal_rows_emoji() {
        // Emoji vary — but the key test is that we DON'T use byte length.
        // "🧠" is 4 bytes but typically 2 display columns.
        let text = "🧠";
        let rows = terminal_rows(text, 0);
        // Should be 1 row regardless (display width <= 80)
        assert_eq!(rows, 1);
        // Verify we're not using byte length: byte len is 4, but width is 2.
        assert_ne!(text.len(), text.width());
    }

    // --- raw mode guard ---

    #[test]
    fn test_raw_mode_guard_initial_state() {
        // RAW_MODE_ACTIVE should be false by default in tests
        // (we can't actually test enter/exit because they affect the real terminal)
        assert!(!RAW_MODE_ACTIVE.load(Ordering::SeqCst));
    }

    // --- SIGWINCH cache ---

    #[test]
    fn test_invalidate_terminal_size() {
        // Set some cached values
        CACHED_WIDTH.store(120, Ordering::Relaxed);
        CACHED_HEIGHT.store(40, Ordering::Relaxed);

        invalidate_terminal_size();

        assert_eq!(CACHED_WIDTH.load(Ordering::Relaxed), 0);
        assert_eq!(CACHED_HEIGHT.load(Ordering::Relaxed), 0);
    }

    // --- TerminalWriter ---

    #[test]
    fn test_terminal_writer_singleton() {
        let w1 = terminal_writer();
        let w2 = terminal_writer();
        // Same address — singleton
        assert!(std::ptr::eq(w1, w2));
    }
}
