//! Local CLI smoke test for `nanobot agent -l`.
//!
//! This test is ignored by default because it requires a running local
//! OpenAI-compatible endpoint (for example LM Studio on 127.0.0.1:1234).

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::Duration;

use chrono::Local;
use serde_json::Value;

use nanobot::config::schema::Config;
use nanobot::utils::helpers::safe_filename;

fn resolve_nanobot_bin() -> PathBuf {
    if let Ok(path) = std::env::var("CARGO_BIN_EXE_nanobot") {
        return PathBuf::from(path);
    }
    if let Ok(path) = std::env::var("NANOBOT_BIN") {
        return PathBuf::from(path);
    }

    // Fallback for environments where cargo does not expose CARGO_BIN_EXE_*.
    // current_exe() => target/debug/deps/<test-binary>; bin => target/debug/nanobot
    let exe = std::env::current_exe().expect("failed to resolve current_exe");
    let debug_dir = exe
        .parent()
        .and_then(|p| p.parent())
        .expect("failed to resolve target/debug directory");
    let candidate = debug_dir.join("nanobot");
    assert!(
        candidate.exists(),
        "nanobot binary not found at {}; set NANOBOT_BIN",
        candidate.display()
    );
    candidate
}

fn write_isolated_config(home: &Path, local_api_base: &str, local_model: &str) {
    let nanobot_dir = home.join(".nanobot");
    let workspace = nanobot_dir.join("workspace");
    fs::create_dir_all(&workspace).expect("failed to create workspace");
    fs::create_dir_all(nanobot_dir.join("sessions")).expect("failed to create sessions dir");
    fs::create_dir_all(nanobot_dir.join("logs")).expect("failed to create logs dir");

    let mut cfg = Config::default();
    cfg.agents.defaults.workspace = workspace.to_string_lossy().to_string();
    cfg.agents.defaults.local_api_base = local_api_base.to_string();
    cfg.agents.defaults.local_model = local_model.to_string();
    cfg.agents.defaults.skip_jit_gate = true;

    let cfg_path = nanobot_dir.join("config.json");
    let cfg_json = serde_json::to_string_pretty(&cfg).expect("failed to serialize config");
    fs::write(cfg_path, cfg_json).expect("failed to write config");
}

fn expected_session_path(home: &Path, session_key: &str) -> PathBuf {
    let safe = safe_filename(&session_key.replace(':', "_"));
    let date = Local::now().format("%Y-%m-%d");
    home.join(".nanobot")
        .join("sessions")
        .join(format!("{}_{}.jsonl", safe, date))
}

fn read_session_jsonl(path: &Path) -> Vec<Value> {
    let data = fs::read_to_string(path).expect("failed to read session file");
    data.lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                return None;
            }
            serde_json::from_str::<Value>(trimmed).ok()
        })
        .collect()
}

fn log_contains_token(logs_dir: &Path, needle: &str) -> bool {
    if let Ok(entries) = fs::read_dir(logs_dir) {
        for entry in entries.flatten() {
            if let Ok(content) = fs::read_to_string(entry.path()) {
                if content.contains(needle) {
                    return true;
                }
            }
        }
    }
    false
}

fn run_local_agent_once(bin: &Path, home: &Path, session: &str, prompt: &str) -> std::process::Output {
    Command::new(bin)
        .env("HOME", home)
        .arg("agent")
        .arg("-l")
        .arg("-s")
        .arg(session)
        .arg("-m")
        .arg(prompt)
        .output()
        .expect("failed to run nanobot agent -l")
}

fn run_local_agent_with_transient_retry(
    bin: &Path,
    home: &Path,
    session: &str,
    prompt: &str,
    attempts: usize,
) -> std::process::Output {
    let mut last = run_local_agent_once(bin, home, session, prompt);
    for _ in 1..attempts {
        if last.status.success() {
            let stderr = String::from_utf8_lossy(&last.stderr);
            let stdout = String::from_utf8_lossy(&last.stdout);
            let transient_transport = stderr.contains("error sending request for url")
                || stdout.contains("error sending request for url");
            if !transient_transport {
                return last;
            }
        }

        let stderr = String::from_utf8_lossy(&last.stderr);
        let stdout = String::from_utf8_lossy(&last.stdout);
        let transient_transport = stderr.contains("error sending request for url")
            || stdout.contains("error sending request for url");
        if !transient_transport {
            return last;
        }

        thread::sleep(Duration::from_millis(500));
        last = run_local_agent_once(bin, home, session, prompt);
    }
    last
}

#[test]
#[ignore = "requires running local OpenAI-compatible endpoint (e.g. LM Studio)"]
fn agent_local_single_turn_smoke() {
    let bin = resolve_nanobot_bin();

    let temp_home = tempfile::tempdir().expect("failed to create temp home");
    let home = temp_home.path();

    let local_api_base = std::env::var("NANOBOT_TEST_LOCAL_API_BASE")
        .unwrap_or_else(|_| "http://127.0.0.1:1234/v1".to_string());
    let local_model = std::env::var("NANOBOT_TEST_LOCAL_MODEL")
        .unwrap_or_else(|_| "qwen/qwen3-4b-thinking-2507".to_string());

    write_isolated_config(home, &local_api_base, &local_model);

    let session = format!("cli:smoke_local_{}", uuid::Uuid::new_v4());
    let prompt = "Briefly reply with local smoke acknowledgement.";

    let output = run_local_agent_with_transient_retry(&bin, home, &session, prompt, 3);

    assert!(
        output.status.success(),
        "agent command failed: status={:?} stderr={} stdout={}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout)
    );

    let session_path = expected_session_path(home, &session);
    assert!(
        session_path.exists(),
        "session file not created: {}",
        session_path.display()
    );

    let session_turns = read_session_jsonl(&session_path);
    assert!(session_turns.iter().any(|m| m.get("role") == Some(&Value::String("user".to_string()))));
    assert!(
        session_turns
            .iter()
            .any(|m| m.get("role") == Some(&Value::String("assistant".to_string())))
    );

    let logs_dir = home.join(".nanobot").join("logs");
    assert!(
        !log_contains_token(&logs_dir, "ClaimedButNotExecuted"),
        "found ClaimedButNotExecuted in isolated logs"
    );
}

#[test]
#[ignore = "requires running local OpenAI-compatible endpoint (e.g. LM Studio)"]
fn agent_local_tool_call_smoke() {
    let bin = resolve_nanobot_bin();
    let temp_home = tempfile::tempdir().expect("failed to create temp home");
    let home = temp_home.path();

    let local_api_base = std::env::var("NANOBOT_TEST_LOCAL_API_BASE")
        .unwrap_or_else(|_| "http://127.0.0.1:1234/v1".to_string());
    let local_model = std::env::var("NANOBOT_TEST_LOCAL_MODEL")
        .unwrap_or_else(|_| "qwen/qwen3-4b-thinking-2507".to_string());
    write_isolated_config(home, &local_api_base, &local_model);

    let tool_file = home
        .join(".nanobot")
        .join("workspace")
        .join("smoke_tool_input.txt");
    fs::write(&tool_file, "SMOKE_TOOL_SENTINEL\nline2").expect("failed to write tool input file");

    let session = format!("cli:smoke_tool_{}", uuid::Uuid::new_v4());
    let prompt = format!(
        "Use the read_file tool exactly once to read this file: {}. Then reply with the first line only.",
        tool_file.display()
    );

    let output = run_local_agent_with_transient_retry(&bin, home, &session, &prompt, 3);
    assert!(
        output.status.success(),
        "agent command failed: status={:?} stderr={} stdout={}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout)
    );

    let session_path = expected_session_path(home, &session);
    assert!(
        session_path.exists(),
        "session file not created: {}",
        session_path.display()
    );
    let session_turns = read_session_jsonl(&session_path);

    let has_assistant_tool_call = session_turns.iter().any(|m| {
        m.get("role") == Some(&Value::String("assistant".to_string()))
            && m
                .get("tool_calls")
                .and_then(|v| v.as_array())
                .map(|a| !a.is_empty())
                .unwrap_or(false)
    });
    let has_tool_result = session_turns.iter().any(|m| {
        m.get("role") == Some(&Value::String("tool".to_string()))
            || m.get("kind") == Some(&Value::String("tool_result".to_string()))
    });

    assert!(
        has_assistant_tool_call,
        "expected assistant tool_calls in session JSONL; got {:?}",
        session_turns
    );
    assert!(
        has_tool_result,
        "expected tool result turn in session JSONL; got {:?}",
        session_turns
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("SMOKE_TOOL_SENTINEL") || stdout.to_lowercase().contains("smoke_tool_sentinel"),
        "expected final output to include sentinel line; stdout={} stderr={}",
        stdout,
        String::from_utf8_lossy(&output.stderr)
    );

    let logs_dir = home.join(".nanobot").join("logs");
    assert!(
        !log_contains_token(&logs_dir, "ClaimedButNotExecuted"),
        "found ClaimedButNotExecuted in isolated logs"
    );
}
