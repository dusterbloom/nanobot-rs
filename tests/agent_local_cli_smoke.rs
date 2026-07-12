//! Local CLI smoke test for `nanobot agent -l`.
//!
//! This test is ignored by default because it requires a running local
//! OpenAI-compatible endpoint on the default local port, 127.0.0.1:8000.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::Duration;

use serde_json::{json, Value};

use nanobot::config::schema::Config;

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

fn expected_sessions_db(home: &Path) -> PathBuf {
    home.join(".nanobot").join("sessions.db")
}

fn read_session_messages(path: &Path, session_key: &str) -> Vec<Value> {
    let connection = rusqlite::Connection::open(path).expect("failed to open sessions.db");
    let mut statement = connection
        .prepare(
            "SELECT m.role, m.content, m.tool_calls, m.tool_call_id, m.tool_name
             FROM messages m
             JOIN sessions s ON s.id = m.session_id
             WHERE s.session_key = ?1
             ORDER BY m.id",
        )
        .expect("failed to prepare session query");
    statement
        .query_map([session_key], |row| {
            let role: String = row.get(0)?;
            let content: Option<String> = row.get(1)?;
            let tool_calls: Option<String> = row.get(2)?;
            let tool_call_id: Option<String> = row.get(3)?;
            let tool_name: Option<String> = row.get(4)?;
            let mut message = json!({"role": role, "content": content});
            if let Some(tool_calls) = tool_calls {
                message["tool_calls"] =
                    serde_json::from_str(&tool_calls).unwrap_or(Value::Null);
            }
            if let Some(tool_call_id) = tool_call_id {
                message["tool_call_id"] = Value::String(tool_call_id);
            }
            if let Some(tool_name) = tool_name {
                message["name"] = Value::String(tool_name);
            }
            Ok(message)
        })
        .expect("failed to query session messages")
        .map(|row| row.expect("failed to decode session row"))
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

fn run_local_agent_once(
    bin: &Path,
    home: &Path,
    session: &str,
    prompt: &str,
) -> std::process::Output {
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
        .unwrap_or_else(|_| "http://127.0.0.1:8000/v1".to_string());
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

    let session_path = expected_sessions_db(home);
    assert!(
        session_path.exists(),
        "sessions.db not created: {}",
        session_path.display()
    );

    let session_turns = read_session_messages(&session_path, &session);
    assert!(session_turns
        .iter()
        .any(|m| m.get("role") == Some(&Value::String("user".to_string()))));
    assert!(session_turns
        .iter()
        .any(|m| m.get("role") == Some(&Value::String("assistant".to_string()))));

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
        .unwrap_or_else(|_| "http://127.0.0.1:8000/v1".to_string());
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

    let session_path = expected_sessions_db(home);
    assert!(
        session_path.exists(),
        "sessions.db not created: {}",
        session_path.display()
    );
    let session_turns = read_session_messages(&session_path, &session);

    let has_assistant_tool_call = session_turns.iter().any(|m| {
        m.get("role") == Some(&Value::String("assistant".to_string()))
            && m.get("tool_calls")
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
        "expected assistant tool_calls in sessions.db; got {:?}",
        session_turns
    );
    assert!(
        has_tool_result,
        "expected tool result turn in sessions.db; got {:?}",
        session_turns
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("SMOKE_TOOL_SENTINEL")
            || stdout.to_lowercase().contains("smoke_tool_sentinel"),
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

#[test]
#[ignore = "requires running local OpenAI-compatible endpoint (e.g. Higgs)"]
fn agent_local_bounded_tool_result_stays_verbatim() {
    let bin = resolve_nanobot_bin();
    let temp_home = tempfile::tempdir().expect("failed to create temp home");
    let home = temp_home.path();

    let local_api_base = std::env::var("NANOBOT_TEST_LOCAL_API_BASE")
        .unwrap_or_else(|_| "http://127.0.0.1:8000/v1".to_string());
    let local_model = std::env::var("NANOBOT_TEST_LOCAL_MODEL")
        .unwrap_or_else(|_| "qwen36-35b-a3b".to_string());
    write_isolated_config(home, &local_api_base, &local_model);

    let session = format!("cli:bounded_tool_{}", uuid::Uuid::new_v4());
    let prompt = "Use exec exactly once to run `seq 1 400`. After the result arrives, reply only with LAST=400.";
    let output = run_local_agent_with_transient_retry(&bin, home, &session, &prompt, 3);
    assert!(
        output.status.success(),
        "bounded tool agent command failed: status={:?} stderr={} stdout={}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout)
    );

    let turns = read_session_messages(&expected_sessions_db(home), &session);
    let tool_content = turns
        .iter()
        .find(|message| {
            message.get("role") == Some(&Value::String("tool".to_string()))
                && message.get("name") == Some(&Value::String("exec".to_string()))
        })
        .and_then(|message| message.get("content"))
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("expected persisted exec tool result; turns={turns:?}"));
    assert!(tool_content.contains("\n250\n"), "middle was lost: {tool_content}");
    assert!(
        !tool_content.contains("[truncated:") && !tool_content.contains("# Content Summary"),
        "configured-bounded result was eagerly compressed: {tool_content}"
    );
}

#[test]
#[ignore = "requires running local OpenAI-compatible endpoint (e.g. Higgs)"]
fn agent_local_parallel_tool_call_smoke() {
    let bin = resolve_nanobot_bin();
    let temp_home = tempfile::tempdir().expect("failed to create temp home");
    let home = temp_home.path();

    let local_api_base = std::env::var("NANOBOT_TEST_LOCAL_API_BASE")
        .unwrap_or_else(|_| "http://127.0.0.1:8000/v1".to_string());
    let local_model = std::env::var("NANOBOT_TEST_LOCAL_MODEL")
        .unwrap_or_else(|_| "qwen36-35b-a3b".to_string());
    write_isolated_config(home, &local_api_base, &local_model);

    let workspace = home.join(".nanobot").join("workspace");
    let first = workspace.join("parallel_a.txt");
    let second = workspace.join("parallel_b.txt");
    fs::write(&first, "PARALLEL_A_SENTINEL").expect("failed to write first input");
    fs::write(&second, "PARALLEL_B_SENTINEL").expect("failed to write second input");

    let session = format!("cli:parallel_tool_{}", uuid::Uuid::new_v4());
    let prompt = format!(
        "In your next assistant response, issue exactly two read_file tool calls together: one for {} and one for {}. Do not use a batch tool and do not read them in separate rounds. After both results arrive, reply with both sentinel lines.",
        first.display(),
        second.display()
    );
    let output = run_local_agent_with_transient_retry(&bin, home, &session, &prompt, 3);
    assert!(
        output.status.success(),
        "parallel agent command failed: status={:?} stderr={} stdout={}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout)
    );

    let turns = read_session_messages(&expected_sessions_db(home), &session);
    let parallel_carrier = turns.iter().find(|message| {
        message.get("role") == Some(&Value::String("assistant".to_string()))
            && message
                .get("tool_calls")
                .and_then(Value::as_array)
                .is_some_and(|calls| calls.len() == 2)
    });
    assert!(
        parallel_carrier.is_some(),
        "expected one assistant carrier with exactly two tool calls; got {turns:?}"
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("PARALLEL_A_SENTINEL"), "stdout={stdout}");
    assert!(stdout.contains("PARALLEL_B_SENTINEL"), "stdout={stdout}");
}
