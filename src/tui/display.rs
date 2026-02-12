/// Display utilities for terminal UI elements.

use std::sync::atomic::Ordering;

use super::ansi::*;
use crate::agent::agent_loop::SharedCoreHandle;
use crate::config::loader::load_config;

/// Print the nanobot ASCII logo -- all white.
pub fn print_logo() {
    println!("  {BOLD}{WHITE} _____             _       _   {RESET}");
    println!("  {BOLD}{WHITE}|   | |___ ___ ___| |_ ___| |_ {RESET}");
    println!("  {BOLD}{WHITE}| | | | .'|   | . | . | . |  _|{RESET}");
    println!("  {BOLD}{WHITE}|_|___|__,|_|_|___|___|___|_|  {RESET}");
}

/// Print a turn separator: --- Role ---------------------
pub fn print_turn_header(role: &str, is_user: bool) {
    let width = terminal_width().saturating_sub(2); // 2-space left margin
    let prefix_len = 4; // "--- "
    let label_len = role.len() + 2; // " Role "
    let rule_len = width.saturating_sub(prefix_len + label_len);
    let color = if is_user { WHITE } else { GREEN };
    println!(
        "  {DIM}───{RESET} {}{BOLD}{}{RESET} {DIM}{}{RESET}",
        color,
        role,
        "─".repeat(rule_len),
    );
}

/// Print user message text, word-wrapped with indent.
pub fn print_user_text(text: &str) {
    let inner = terminal_width().saturating_sub(4);
    for line in wrap_text(text, inner) {
        println!("  {}", line);
    }
}

/// Wrap text to fit within a given width.
pub fn wrap_text(text: &str, width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    for paragraph in text.split('\n') {
        if paragraph.is_empty() {
            lines.push(String::new());
            continue;
        }
        let mut current_line = String::new();
        let mut current_width = 0;
        for word in paragraph.split_whitespace() {
            let word_width = word.chars().count();
            if current_width + word_width + 1 > width && !current_line.is_empty() {
                lines.push(current_line);
                current_line = String::new();
                current_width = 0;
            }
            if !current_line.is_empty() {
                current_line.push(' ');
                current_width += 1;
            }
            current_line.push_str(word);
            current_width += word_width;
        }
        if !current_line.is_empty() {
            lines.push(current_line);
        }
    }
    if lines.is_empty() {
        lines.push(String::new());
    }
    lines
}

/// Animated loading sequence with retro block characters.
pub fn loading_animation(message: &str) {
    use std::io::Write;
    let frames = ["░  ", "▒░ ", "▓▒░", "█▓▒", "▓▒░", "▒░ "];
    print!("{HIDE_CURSOR}");
    for i in 0..12 {
        print!(
            "\r  {DIM}{} {}{RESET} ",
            message,
            frames[i % frames.len()]
        );
        std::io::stdout().flush().ok();
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    print!("\r{}\r", " ".repeat(60)); // clear the line
    print!("{SHOW_CURSOR}");
    std::io::stdout().flush().ok();
}

/// Format a number with thousands separators.
pub fn format_thousands(n: usize) -> String {
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

/// Print a compact status bar: ctx N% . model . t:N
pub fn print_info_bar(core_handle: &SharedCoreHandle, model: &str) {
    let core = core_handle.read().unwrap().clone();
    let used = core.last_context_used.load(Ordering::Relaxed) as usize;
    let max = core.last_context_max.load(Ordering::Relaxed) as usize;
    let turn = core.learning_turn_counter.load(Ordering::Relaxed);

    let pct = if max > 0 { (used * 100) / max } else { 0 };
    let ctx_color = match pct {
        0..=49 => GREEN,
        50..=79 => YELLOW,
        _ => RED,
    };

    println!(
        "  {}ctx {}{}{}%{}{} · {} · t:{}{}",
        DIM, ctx_color, BOLD, pct, RESET,
        DIM, model, turn, RESET,
    );
}

/// Print the current mode banner (compact, for mode switches mid-session).
pub fn print_mode_banner(local_port: &str) {
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
            "  {BOLD}{WHITE}CLOUD MODE{RESET} {DIM}{}{RESET}",
            config.agents.defaults.model
        );
    }
    println!();
}

/// Full startup splash: clear screen, ASCII logo, mode info, hints.
pub fn print_startup_splash(local_port: &str) {
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
            "  {BOLD}{WHITE}CLOUD{RESET} {DIM}{}{RESET}",
            config.agents.defaults.model
        );
    }
    println!("  {DIM}v{}  ░  /local  /model  /voice  Ctrl+C quit{RESET}", crate::VERSION);
    println!();

    // Brief loading animation
    loading_animation("Initializing agent");
}
