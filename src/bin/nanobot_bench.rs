//! nanobot-bench — speed regression track for the agent loop.
//!
//! Models the ds4-bench shape: a fixed task corpus, instantaneous per-task
//! metrics, CSV output ready for before/after comparison. The single headline
//! metric is `total_ms` summed across the run.
//!
//! Usage:
//!     cargo run --release --bin nanobot-bench -- \
//!         --tasks t01,t02,t05 \
//!         --csv /tmp/nanobot-speed.csv
//!
//!     # full sweep
//!     cargo run --release --bin nanobot-bench -- --tasks all --csv /tmp/sweep.csv
//!
//!     # diff against committed baseline
//!     python3 scripts/bench_diff.py benches/baseline.csv /tmp/nanobot-speed.csv
//!
//! Determinism: bench mode forces temperature=0, suppresses timestamp/uuid
//! injection in the system prompt, and refuses to run if the provider is not
//! pinned. Same machine, provider, model — same expectations as ds4-bench.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;

/// Fixed task corpus. Identifiers are stable — don't renumber them, that would
/// invalidate every prior baseline.csv row.
const TASKS: &[BenchTask] = &[
    BenchTask {
        id: "t01_noop",
        kind: TaskKind::NoTools,
    },
    BenchTask {
        id: "t02_read",
        kind: TaskKind::SingleTool,
    },
    BenchTask {
        id: "t03_search",
        kind: TaskKind::MultiTool,
    },
    BenchTask {
        id: "t04_long_in",
        kind: TaskKind::LongInput,
    },
    BenchTask {
        id: "t05_chain",
        kind: TaskKind::ChainedTools,
    },
    BenchTask {
        id: "t06_resume",
        kind: TaskKind::SessionResume,
    },
    BenchTask {
        id: "t07_skill",
        kind: TaskKind::Skill,
    },
    BenchTask {
        id: "t08_router",
        kind: TaskKind::Router,
    },
];

#[derive(Debug)]
struct BenchTask {
    id: &'static str,
    kind: TaskKind,
}

#[derive(Debug, Clone, Copy)]
enum TaskKind {
    NoTools,
    SingleTool,
    MultiTool,
    LongInput,
    ChainedTools,
    SessionResume,
    Skill,
    Router,
}

/// One CSV row per task per run. Don't reorder columns — `bench_diff.py`
/// matches by header name, but committed baselines are easier to diff visually
/// when columns are stable.
#[derive(Debug)]
struct BenchRow {
    commit_sha: String,
    task_id: &'static str,
    model: String,
    cold_start_ms: u64,
    context_ms: u64,
    ttfb_ms: u64,
    total_ms: u64,
    tokens_in: u32,
    tokens_out: u32,
    tool_calls: u32,
    turn_rounds: u32,
    success: bool,
}

impl BenchRow {
    fn csv_header() -> &'static str {
        "commit_sha,task_id,model,cold_start_ms,context_ms,ttfb_ms,total_ms,tokens_in,tokens_out,tool_calls,turn_rounds,success"
    }
    fn to_csv(&self) -> String {
        format!(
            "{},{},{},{},{},{},{},{},{},{},{},{}",
            self.commit_sha,
            self.task_id,
            self.model,
            self.cold_start_ms,
            self.context_ms,
            self.ttfb_ms,
            self.total_ms,
            self.tokens_in,
            self.tokens_out,
            self.tool_calls,
            self.turn_rounds,
            self.success
        )
    }
}

#[derive(Parser, Debug)]
#[command(version, about = "speed regression bench for the nanobot agent loop")]
struct Cli {
    /// Comma-separated task ids, or `all`. Defaults to the quick smoke set.
    #[arg(long, default_value = "t01_noop,t02_read,t05_chain")]
    tasks: String,

    /// Output CSV path. Use `-` for stdout.
    #[arg(long, default_value = "-")]
    csv: String,

    /// Pinned model name (must match a configured provider).
    #[arg(long, default_value = "")]
    model: String,

    /// Force-disable timestamps / uuids / cwd in the system prompt (default true
    /// in bench mode; expose as a flag for diagnostics only).
    #[arg(long, default_value_t = true)]
    deterministic_prompt: bool,
}

fn select_tasks(spec: &str) -> Vec<&'static BenchTask> {
    if spec == "all" {
        return TASKS.iter().collect();
    }
    let wanted: std::collections::HashSet<&str> = spec.split(',').map(str::trim).collect();
    TASKS.iter().filter(|t| wanted.contains(t.id)).collect()
}

fn commit_sha() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

/// Run one task against the agent loop and return measured timings.
///
/// TODO: wire this into the real `AgentLoop::process_direct()` path.
/// Currently a stub so the CSV shape and the diff workflow can be
/// established and reviewed before the runner is filled in. When wired up:
/// - Instantiate AgentLoop with bench config (deterministic prompt, temp=0)
/// - Start a fresh process timer at this fn entry → cold_start_ms
/// - Time context build → context_ms
/// - Time first-byte from provider → ttfb_ms
/// - Time full reply → total_ms
/// - Count tokens via the provider's usage object
/// - Count tool_calls and turn_rounds from the loop's iteration counter
fn run_task(task: &BenchTask, model: &str, sha: &str) -> BenchRow {
    let _ = (task.kind, model); // suppress unused warnings until wired up

    let start = Instant::now();
    // ... stub body ...
    let total = start.elapsed().as_millis() as u64;

    BenchRow {
        commit_sha: sha.to_string(),
        task_id: task.id,
        model: model.to_string(),
        cold_start_ms: 0,
        context_ms: 0,
        ttfb_ms: 0,
        total_ms: total,
        tokens_in: 0,
        tokens_out: 0,
        tool_calls: 0,
        turn_rounds: 0,
        success: false, // false until the runner is wired up
    }
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let sha = commit_sha();
    let tasks = select_tasks(&cli.tasks);

    if tasks.is_empty() {
        anyhow::bail!("no tasks selected; spec={}", cli.tasks);
    }

    let mut writer: Box<dyn Write> = if cli.csv == "-" {
        Box::new(std::io::stdout())
    } else {
        Box::new(BufWriter::new(File::create(PathBuf::from(&cli.csv))?))
    };

    writeln!(writer, "{}", BenchRow::csv_header())?;

    let mut total_ms_sum: u64 = 0;
    for task in &tasks {
        let row = run_task(task, &cli.model, &sha);
        total_ms_sum += row.total_ms;
        writeln!(writer, "{}", row.to_csv())?;
    }
    writer.flush()?;

    // Headline metric on stderr so it doesn't pollute the CSV when streamed.
    eprintln!("headline total_ms sum: {}", total_ms_sum);
    eprintln!("tasks run: {}", tasks.len());
    eprintln!("NOTE: task runner is a stub — populate `run_task()` to record real timings.");
    Ok(())
}
