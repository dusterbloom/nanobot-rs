//! CLI subcommands for the canonical SQLite session store.

use std::io::{self, Write};
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::session::db::{LegacyImportOutcome, SessionDb};

fn default_db_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_default()
        .join(".nanobot")
        .join("sessions.db")
}

pub async fn cmd_sessions_list() {
    let db = SessionDb::new(&default_db_path());
    let sessions = db.list_sessions(None, 100).await;

    if sessions.is_empty() {
        println!("No sessions found.");
        return;
    }

    println!(
        "{:<28} {:<32} {:<20} {:>6}",
        "SESSION ID", "SESSION KEY", "UPDATED", "MSGS"
    );
    println!("{}", "-".repeat(90));
    for session in &sessions {
        let updated = session.updated_at.format("%Y-%m-%d %H:%M UTC").to_string();
        println!(
            "{:<28} {:<32} {:<20} {:>6}",
            truncate(&session.id, 26),
            truncate(&session.session_key, 30),
            truncate(&updated, 18),
            session.message_count,
        );
    }
    println!("\n{} session(s) total.", sessions.len());
}

/// Delete one concrete SQLite session and all of its owned rows.
pub async fn cmd_sessions_delete(id: &str, force: bool) {
    let db = SessionDb::new(&default_db_path());
    let Some(session) = db.get_session(id).await else {
        eprintln!("Session '{id}' not found.");
        eprintln!("Use `nanobot sessions list` to see session IDs.");
        return;
    };

    println!(
        "Delete session {} ({}, {} message(s))?",
        session.id, session.session_key, session.message_count
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

    match db.delete_session(&session.id).await {
        Ok(true) => println!("Deleted session {}.", session.id),
        Ok(false) => eprintln!("Session '{}' disappeared before deletion.", session.id),
        Err(error) => eprintln!("Failed to delete session transactionally: {error}"),
    }
}

pub fn make_session_key(name: Option<&str>) -> String {
    match name {
        Some(name) => format!("cli:{name}"),
        None => format!("cli:{}", &uuid::Uuid::new_v4().to_string()[..8]),
    }
}

/// Export remains available in both Markdown and line-delimited JSON.
pub async fn cmd_sessions_export(key: &str, format: &str) {
    let db = SessionDb::new(&default_db_path());
    let session_id = if let Some(meta) = db.get_latest_session(key).await {
        meta.id
    } else if let Some(meta) = db.get_session(key).await {
        meta.id
    } else {
        eprintln!("Session '{key}' not found.");
        eprintln!("Use `nanobot sessions list` to see available sessions.");
        return;
    };

    let messages = db.get_all_messages(&session_id).await;
    print!("{}", render_session_export(key, format, &messages));
}

/// One-shot migration entry point for legacy JSONL transcripts.
///
/// With a path, imports exactly that file. Without one, scans the legacy
/// `~/.nanobot/sessions/*.jsonl` directory once. Embedded metadata owns the
/// session key; the filename-derived key is only a safe fallback.
pub async fn cmd_sessions_import(path: Option<&Path>) {
    let db = SessionDb::new(&default_db_path());
    let sources = match path {
        Some(path) => vec![path.to_path_buf()],
        None => discover_legacy_jsonl(
            &dirs::home_dir()
                .unwrap_or_default()
                .join(".nanobot")
                .join("sessions"),
        ),
    };
    if sources.is_empty() {
        println!("No legacy JSONL session files found.");
        return;
    }

    let mut imported = 0;
    let mut unchanged = 0;
    let mut rejected = 0;
    for source in sources {
        let fallback_key = fallback_session_key(&source);
        match db.import_legacy_jsonl(&source, &fallback_key).await {
            Ok(LegacyImportOutcome::Imported {
                session_id,
                message_count,
            }) => {
                imported += 1;
                println!(
                    "Imported {}: {message_count} message(s) into session {session_id}.",
                    source.display()
                );
            }
            Ok(LegacyImportOutcome::AlreadyImported {
                session_id,
                message_count,
            }) => {
                unchanged += 1;
                println!(
                    "Already imported {}: {message_count} message(s) in session {session_id}.",
                    source.display()
                );
            }
            Err(error) => {
                rejected += 1;
                eprintln!("Rejected {}: {error}", source.display());
            }
        }
    }
    println!("Import complete: {imported} new, {unchanged} unchanged, {rejected} rejected.");
}

/// Delete all SQLite sessions transactionally, then remove unrelated logs,
/// metrics, and explicitly-confirmed legacy JSONL migration inputs.
pub async fn cmd_sessions_nuke(force: bool) {
    let home = dirs::home_dir().unwrap_or_default().join(".nanobot");
    let db_path = home.join("sessions.db");
    let db = SessionDb::new(&db_path);
    let session_count = db.list_sessions(None, usize::MAX).await.len();
    let legacy_dir = home.join("sessions");
    let legacy_count = count_files(&legacy_dir);
    let logs_dir = home.join("logs");
    let log_count = count_files(&logs_dir);
    let metrics_path = home.join("metrics.jsonl");
    let has_metrics = metrics_path.exists();
    let total = session_count + legacy_count + log_count + usize::from(has_metrics);

    if total == 0 {
        println!("Nothing to nuke. Already clean.");
        return;
    }

    println!(
        "This will delete {session_count} SQLite session(s), {legacy_count} legacy session file(s), \
         {log_count} log file(s){}.",
        if has_metrics { ", and metrics.jsonl" } else { "" }
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

    let deleted = match db.nuke_sessions().await {
        Ok(count) => count,
        Err(error) => {
            eprintln!("Failed to delete sessions transactionally: {error}");
            return;
        }
    };
    if legacy_dir.exists() {
        let _ = std::fs::remove_dir_all(&legacy_dir);
        let _ = std::fs::create_dir_all(&legacy_dir);
    }
    if logs_dir.exists() {
        let _ = std::fs::remove_dir_all(&logs_dir);
        let _ = std::fs::create_dir_all(&logs_dir);
    }
    if has_metrics {
        let _ = std::fs::remove_file(&metrics_path);
    }
    println!("Nuked {deleted} SQLite session(s), legacy files, logs, and metrics.");
}

/// Purge old SQLite sessions transactionally. Legacy JSONL files are retained
/// until explicitly imported or removed with `nuke` so migration inputs are
/// never destroyed by routine retention.
pub async fn cmd_sessions_purge(older_than: &str) {
    let seconds = match parse_duration_str(older_than) {
        Some(seconds) => seconds,
        None => {
            eprintln!("Invalid duration: '{older_than}'. Use format like '7d', '24h', '30d'.");
            return;
        }
    };
    let cutoff = chrono::Utc::now() - chrono::Duration::seconds(seconds as i64);
    let db = SessionDb::new(&default_db_path());
    let sessions_removed = match db.purge_sessions_before(cutoff).await {
        Ok(count) => count,
        Err(error) => {
            eprintln!("Failed to purge sessions transactionally: {error}");
            return;
        }
    };

    let home = dirs::home_dir().unwrap_or_default().join(".nanobot");
    let logs_removed = purge_old_files(&home.join("logs"), &cutoff, "log");
    let metrics_path = home.join("metrics.jsonl");
    let mut metrics_removed = 0;
    if let Ok(metadata) = std::fs::metadata(&metrics_path) {
        if let Ok(modified) = metadata.modified() {
            let modified: chrono::DateTime<chrono::Utc> = modified.into();
            if modified < cutoff && std::fs::remove_file(&metrics_path).is_ok() {
                metrics_removed = 1;
            }
        }
    }
    println!(
        "Purged {sessions_removed} session(s) and {} log/metric file(s) older than {older_than}.",
        logs_removed + metrics_removed
    );
}

pub async fn cmd_sessions_archive() {
    let home = dirs::home_dir().unwrap_or_default().join(".nanobot");
    let db_path = default_db_path();
    let db = SessionDb::new(&db_path);
    let session_count = db.list_sessions(None, usize::MAX).await.len();
    let session_size = std::fs::metadata(&db_path)
        .map(|meta| meta.len())
        .unwrap_or(0);
    let legacy_size = dir_total_size(&home.join("sessions"));
    let log_size = dir_total_size(&home.join("logs"));
    let metrics_size = std::fs::metadata(home.join("metrics.jsonl"))
        .map(|meta| meta.len())
        .unwrap_or(0);

    println!("Disk usage:");
    println!(
        "  SQLite sessions ({session_count}): {}",
        format_bytes(session_size)
    );
    println!("  Legacy JSONL: {}", format_bytes(legacy_size));
    println!("  Logs:         {}", format_bytes(log_size));
    println!("  Metrics:      {}", format_bytes(metrics_size));
    println!(
        "  Total:        {}",
        format_bytes(session_size + legacy_size + log_size + metrics_size)
    );
    println!("\nTo free space: nanobot sessions purge --older-than 7d");
}

fn render_session_export(key: &str, format: &str, messages: &[Value]) -> String {
    if format == "jsonl" {
        let mut output = messages
            .iter()
            .map(|message| serde_json::to_string(message).unwrap_or_default())
            .collect::<Vec<_>>()
            .join("\n");
        if !output.is_empty() {
            output.push('\n');
        }
        return output;
    }

    let mut output = format!("# Session: {key}\n\n");
    for message in messages {
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        let timestamp = message
            .get("timestamp")
            .and_then(Value::as_str)
            .unwrap_or("");
        let time = if timestamp.len() >= 19 {
            &timestamp[11..19]
        } else {
            timestamp
        };
        let content = message.get("content").and_then(Value::as_str).unwrap_or("");
        match role {
            "user" => output.push_str(&format!("## User ({time})\n\n{content}\n\n")),
            "assistant" if !content.is_empty() => {
                output.push_str(&format!("## Assistant ({time})\n\n{content}\n\n"));
            }
            "tool" => {
                let name = message
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("tool");
                output.push_str(&format!(
                    "## Tool: {name} ({time})\n\n{}\n\n",
                    truncate(content, 200)
                ));
            }
            _ if !content.is_empty() => {
                output.push_str(&format!("## {role} ({time})\n\n{content}\n\n"));
            }
            _ => {}
        }
    }
    output
}

fn count_files(dir: &Path) -> usize {
    std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .flatten()
                .filter(|entry| entry.path().is_file())
                .count()
        })
        .unwrap_or(0)
}

fn discover_legacy_jsonl(dir: &Path) -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("jsonl"))
        .collect();
    paths.sort();
    paths
}

fn fallback_session_key(path: &Path) -> String {
    let stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("imported");
    let stem = strip_date_suffix(stem);
    let safe: String = stem
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.') {
                character
            } else {
                '_'
            }
        })
        .collect();
    let safe = safe.trim_matches('_');
    format!("legacy:{}", if safe.is_empty() { "imported" } else { safe })
}

fn strip_date_suffix(stem: &str) -> &str {
    let Some((prefix, suffix)) = stem.rsplit_once('_') else {
        return stem;
    };
    let bytes = suffix.as_bytes();
    let is_date = bytes.len() == 10
        && bytes[4] == b'-'
        && bytes[7] == b'-'
        && bytes
            .iter()
            .enumerate()
            .all(|(index, byte)| matches!(index, 4 | 7) || byte.is_ascii_digit());
    if is_date && !prefix.is_empty() {
        prefix
    } else {
        stem
    }
}

fn dir_total_size(dir: &Path) -> u64 {
    std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .flatten()
                .filter_map(|entry| entry.metadata().ok())
                .map(|metadata| metadata.len())
                .sum()
        })
        .unwrap_or(0)
}

fn parse_duration_str(value: &str) -> Option<u64> {
    let value = value.trim();
    if value.len() < 2 {
        return None;
    }
    let (number, unit) = value.split_at(value.len() - 1);
    let number: u64 = number.parse().ok()?;
    match unit {
        "s" => Some(number),
        "m" => Some(number * 60),
        "h" => Some(number * 3600),
        "d" => Some(number * 86400),
        _ => None,
    }
}

fn purge_old_files(dir: &Path, cutoff: &chrono::DateTime<chrono::Utc>, extension: &str) -> usize {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    entries
        .flatten()
        .filter(|entry| {
            let path = entry.path();
            let matches_extension =
                path.extension().and_then(|ext| ext.to_str()) == Some(extension);
            let text = path.to_string_lossy();
            let matches_rotated = text.contains(&format!(".{extension}."))
                || text.ends_with(&format!(".{extension}.gz"));
            matches_extension || matches_rotated
        })
        .filter(|entry| {
            entry
                .metadata()
                .ok()
                .and_then(|metadata| metadata.modified().ok())
                .map(|modified| chrono::DateTime::<chrono::Utc>::from(modified) < *cutoff)
                .unwrap_or(false)
        })
        .filter(|entry| std::fs::remove_file(entry.path()).is_ok())
        .count()
}

fn format_bytes(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{bytes}B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1}KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1}MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

use crate::utils::helpers::truncate_string as truncate;

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::tempdir;

    #[test]
    fn make_session_keys_are_stable_or_unique_as_requested() {
        assert_eq!(make_session_key(Some("my-session")), "cli:my-session");
        let generated = make_session_key(None);
        assert!(generated.starts_with("cli:"));
        assert_eq!(generated.len(), "cli:".len() + 8);
    }

    #[test]
    fn jsonl_export_preserves_one_complete_json_value_per_line() {
        let messages = vec![
            json!({"role": "user", "content": "hello", "_db_id": 1}),
            json!({"role": "assistant", "content": "hi", "_db_id": 2}),
        ];
        let output = render_session_export("cli:test", "jsonl", &messages);
        let decoded: Vec<Value> = output
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(decoded, messages);
        assert!(output.ends_with('\n'));
    }

    #[test]
    fn markdown_export_keeps_tool_output_bounded() {
        let output = render_session_export(
            "cli:test",
            "md",
            &[json!({
                "role": "tool",
                "name": "read_file",
                "content": "x".repeat(500),
                "timestamp": "2026-01-01T12:34:56Z"
            })],
        );
        assert!(output.contains("## Tool: read_file (12:34:56)"));
        assert!(output.len() < 400);
    }

    #[test]
    fn duration_parser_and_file_count_are_deterministic() {
        assert_eq!(parse_duration_str("7d"), Some(7 * 86_400));
        assert_eq!(parse_duration_str("24h"), Some(24 * 3_600));
        assert_eq!(parse_duration_str("bad"), None);
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("one"), "").unwrap();
        std::fs::write(dir.path().join("two"), "").unwrap();
        assert_eq!(count_files(dir.path()), 2);
    }

    #[test]
    fn jsonl_scan_and_fallback_key_are_safe_and_deterministic() {
        let dir = tempdir().unwrap();
        let first = dir.path().join("cli_project_2026-07-14.jsonl");
        let second = dir.path().join("telegram user.jsonl");
        std::fs::write(&second, "").unwrap();
        std::fs::write(&first, "").unwrap();
        std::fs::write(dir.path().join("ignore.txt"), "").unwrap();

        assert_eq!(
            discover_legacy_jsonl(dir.path()),
            vec![first.clone(), second.clone()]
        );
        assert_eq!(fallback_session_key(&first), "legacy:cli_project");
        assert_eq!(fallback_session_key(&second), "legacy:telegram_user");
    }
}
