//! CLI subcommands for session management.
//!
//! Extracted from `cli.rs` to keep that module focused on agent/config
//! commands. All `nanobot sessions *` subcommands live here.

use std::io::{self, Write};

use crate::utils::helpers::get_workspace_path;

// ---------------------------------------------------------------------------
// Public commands
// ---------------------------------------------------------------------------

/// List all sessions with date, size, and message count.
pub fn cmd_sessions_list() {
    let workspace = get_workspace_path(None);
    let mgr = crate::session::manager::SessionManager::new(&workspace);
    let sessions = mgr.list_sessions();

    if sessions.is_empty() {
        println!("No sessions found.");
        return;
    }

    println!("{:<40} {:<24} {:>6} {:>10}", "SESSION", "UPDATED", "MSGS", "SIZE");
    println!("{}", "-".repeat(82));

    for s in &sessions {
        let key = s.get("key").and_then(|v| v.as_str()).unwrap_or("?");
        let updated = s
            .get("updated_at")
            .and_then(|v| v.as_str())
            .unwrap_or("?");
        let path_str = s.get("path").and_then(|v| v.as_str()).unwrap_or("");
        let size = std::fs::metadata(path_str)
            .map(|m| format_bytes(m.len()))
            .unwrap_or_else(|_| "?".to_string());
        let msg_count = std::fs::read_to_string(path_str)
            .map(|c| c.lines().filter(|l| !l.contains("\"_type\":\"metadata\"")).count())
            .unwrap_or(0);
        println!(
            "{:<40} {:<24} {:>6} {:>10}",
            truncate(key, 38),
            truncate(updated, 22),
            msg_count,
            size,
        );
    }
    println!("\n{} session(s) total.", sessions.len());
}

/// Generate a session key from an optional user-provided name.
pub fn make_session_key(name: Option<&str>) -> String {
    match name {
        Some(n) => format!("cli:{}", n),
        None => format!("cli:{}", &uuid::Uuid::new_v4().to_string()[..8]),
    }
}

/// Export a session to stdout in markdown or JSONL format.
pub fn cmd_sessions_export(key: &str, format: &str) {
    let workspace = get_workspace_path(None);
    let mgr = crate::session::manager::SessionManager::new(&workspace);
    let sessions = mgr.list_sessions();

    // Find matching session by key.
    let matched = sessions.iter().find(|s| {
        s.get("key").and_then(|v| v.as_str()) == Some(key)
    });

    let path_str = match matched {
        Some(s) => s.get("path").and_then(|v| v.as_str()).unwrap_or("").to_string(),
        None => {
            eprintln!("Session '{}' not found.", key);
            eprintln!("Use `nanobot sessions list` to see available sessions.");
            std::process::exit(1);
        }
    };

    let content = match std::fs::read_to_string(&path_str) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error reading session file: {}", e);
            std::process::exit(1);
        }
    };

    match format {
        "jsonl" => {
            print!("{}", content);
        }
        "md" | _ => {
            println!("# Session: {}\n", key);
            for line in content.lines() {
                let parsed: serde_json::Value = match serde_json::from_str(line) {
                    Ok(v) => v,
                    Err(_) => continue,
                };

                // Skip metadata lines.
                if parsed.get("_type").and_then(|v| v.as_str()) == Some("metadata") {
                    continue;
                }

                let role = parsed.get("role").and_then(|v| v.as_str()).unwrap_or("unknown");
                let timestamp = parsed.get("timestamp").and_then(|v| v.as_str()).unwrap_or("");

                // Extract just the time portion if it's an ISO timestamp.
                let time_display = if timestamp.len() >= 19 {
                    &timestamp[11..19]
                } else {
                    timestamp
                };

                match role {
                    "user" => {
                        let text = parsed.get("content").and_then(|v| v.as_str()).unwrap_or("");
                        println!("## User ({})\n\n{}\n", time_display, text);
                    }
                    "assistant" => {
                        let text = parsed.get("content").and_then(|v| v.as_str()).unwrap_or("");
                        if !text.is_empty() {
                            println!("## Assistant ({})\n\n{}\n", time_display, text);
                        }
                    }
                    "tool" => {
                        let tool_name = parsed.get("name").and_then(|v| v.as_str()).unwrap_or("tool");
                        let result = parsed.get("content").and_then(|v| v.as_str()).unwrap_or("");
                        let abbreviated = truncate(result, 200);
                        println!("## Tool: {} ({})\n\n{}\n", tool_name, time_display, abbreviated);
                    }
                    _ => {
                        let text = parsed.get("content").and_then(|v| v.as_str()).unwrap_or("");
                        if !text.is_empty() {
                            println!("## {} ({})\n\n{}\n", role, time_display, text);
                        }
                    }
                }
            }
        }
    }
}

/// Wipe all sessions, logs, and metrics.
pub fn cmd_sessions_nuke(force: bool) {
    let home = dirs::home_dir().unwrap_or_default().join(".nanobot");
    let sessions_dir = home.join("sessions");
    let logs_dir = home.join("logs");
    let metrics_path = home.join("metrics.jsonl");

    // Count files.
    let session_count = count_files(&sessions_dir);
    let log_count = count_files(&logs_dir);
    let has_metrics = metrics_path.exists();
    let total = session_count + log_count + if has_metrics { 1 } else { 0 };

    if total == 0 {
        println!("Nothing to nuke. Already clean.");
        return;
    }

    println!(
        "This will delete {} session file(s), {} log file(s){}. ({} total)",
        session_count,
        log_count,
        if has_metrics { ", and metrics.jsonl" } else { "" },
        total,
    );

    if !force {
        print!("Are you sure? [y/N] ");
        io::stdout().flush().ok();
        let mut input = String::new();
        io::stdin().read_line(&mut input).ok();
        if !input.trim().eq_ignore_ascii_case("y") {
            println!("Cancelled.");
            return;
        }
    }

    // Remove and recreate directories.
    if sessions_dir.exists() {
        let _ = std::fs::remove_dir_all(&sessions_dir);
        let _ = std::fs::create_dir_all(&sessions_dir);
    }
    if logs_dir.exists() {
        let _ = std::fs::remove_dir_all(&logs_dir);
        let _ = std::fs::create_dir_all(&logs_dir);
    }
    if has_metrics {
        let _ = std::fs::remove_file(&metrics_path);
    }

    println!("Nuked. All sessions, logs, and metrics removed.");
}

/// Purge session and log files older than the given duration string (e.g. "7d", "24h").
pub fn cmd_sessions_purge(older_than: &str) {
    let seconds = match parse_duration_str(older_than) {
        Some(s) => s,
        None => {
            eprintln!("Invalid duration: '{}'. Use format like '7d', '24h', '30d'.", older_than);
            return;
        }
    };

    let cutoff = chrono::Utc::now() - chrono::Duration::seconds(seconds as i64);
    let mut removed = 0u32;

    // Purge sessions
    let sessions_dir = dirs::home_dir()
        .unwrap_or_default()
        .join(".nanobot")
        .join("sessions");
    removed += purge_old_files(&sessions_dir, &cutoff, "jsonl");

    // Purge rotated log files
    let logs_dir = dirs::home_dir()
        .unwrap_or_default()
        .join(".nanobot")
        .join("logs");
    removed += purge_old_files(&logs_dir, &cutoff, "log");

    // Purge metrics (only if older than threshold)
    let metrics_path = dirs::home_dir()
        .unwrap_or_default()
        .join(".nanobot")
        .join("metrics.jsonl");
    if let Ok(meta) = std::fs::metadata(&metrics_path) {
        if let Ok(modified) = meta.modified() {
            let modified_dt: chrono::DateTime<chrono::Utc> = modified.into();
            if modified_dt < cutoff {
                let _ = std::fs::remove_file(&metrics_path);
                removed += 1;
            }
        }
    }

    println!("Purged {} file(s) older than {}.", removed, older_than);
}

/// Show summary of session and log disk usage, with counts by age.
pub fn cmd_sessions_archive() {
    let home = dirs::home_dir().unwrap_or_default().join(".nanobot");

    let sessions_dir = home.join("sessions");
    let logs_dir = home.join("logs");
    let metrics_path = home.join("metrics.jsonl");

    let session_size = dir_total_size(&sessions_dir);
    let log_size = dir_total_size(&logs_dir);
    let metrics_size = std::fs::metadata(&metrics_path)
        .map(|m| m.len())
        .unwrap_or(0);

    println!("Disk usage:");
    println!("  Sessions:  {}", format_bytes(session_size));
    println!("  Logs:      {}", format_bytes(log_size));
    println!("  Metrics:   {}", format_bytes(metrics_size));
    println!("  Total:     {}", format_bytes(session_size + log_size + metrics_size));
    println!();
    println!("To free space: nanobot sessions purge --older-than 7d");
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn count_files(dir: &std::path::Path) -> usize {
    std::fs::read_dir(dir)
        .map(|entries| entries.flatten().filter(|e| e.path().is_file()).count())
        .unwrap_or(0)
}

fn dir_total_size(dir: &std::path::Path) -> u64 {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return 0,
    };
    entries
        .flatten()
        .filter_map(|e| e.metadata().ok())
        .map(|m| m.len())
        .sum()
}

fn parse_duration_str(s: &str) -> Option<u64> {
    let s = s.trim();
    if s.len() < 2 {
        return None;
    }
    let (num_str, unit) = s.split_at(s.len() - 1);
    let num: u64 = num_str.parse().ok()?;
    match unit {
        "s" => Some(num),
        "m" => Some(num * 60),
        "h" => Some(num * 3600),
        "d" => Some(num * 86400),
        _ => None,
    }
}

fn purge_old_files(dir: &std::path::Path, cutoff: &chrono::DateTime<chrono::Utc>, ext: &str) -> u32 {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return 0,
    };
    let mut count = 0u32;
    for entry in entries.flatten() {
        let path = entry.path();
        let matches_ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e == ext)
            .unwrap_or(false);
        // Also match .gz files
        let matches_gz = path.to_string_lossy().contains(&format!(".{}.", ext))
            || path.to_string_lossy().ends_with(&format!(".{}.gz", ext));
        if !matches_ext && !matches_gz {
            continue;
        }
        let modified = match std::fs::metadata(&path)
            .ok()
            .and_then(|m| m.modified().ok())
        {
            Some(t) => t,
            None => continue,
        };
        let modified_dt: chrono::DateTime<chrono::Utc> = modified.into();
        if modified_dt < *cutoff {
            let _ = std::fs::remove_file(&path);
            count += 1;
        }
    }
    count
}

fn format_bytes(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{}B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1}KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1}MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max.saturating_sub(3)])
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_make_session_key_with_name() {
        let key = make_session_key(Some("my-session"));
        assert_eq!(key, "cli:my-session");
    }

    #[test]
    fn test_make_session_key_without_name() {
        let key = make_session_key(None);
        assert!(key.starts_with("cli:"));
        // UUID8 portion should be 8 chars.
        assert_eq!(key.len(), "cli:".len() + 8);
    }

    #[test]
    fn test_count_files_empty_dir() {
        let dir = tempdir().unwrap();
        assert_eq!(count_files(dir.path()), 0);
    }

    #[test]
    fn test_count_files_with_files() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "").unwrap();
        std::fs::write(dir.path().join("b.txt"), "").unwrap();
        assert_eq!(count_files(dir.path()), 2);
    }

    #[test]
    fn test_count_files_nonexistent_dir() {
        let dir = std::path::Path::new("/tmp/nanobot_test_nonexistent_dir_xyz");
        assert_eq!(count_files(dir), 0);
    }
}
