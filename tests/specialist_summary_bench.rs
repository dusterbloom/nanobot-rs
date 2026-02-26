//! Benchmark specialist summarization quality on long tool outputs.
//!
//! This benchmark is ignored by default because it requires a running
//! OpenAI-compatible local endpoint (for example LM Studio on 127.0.0.1:1234)
//! with the target models downloaded.
//!
//! Run:
//!   cargo test --test specialist_summary_bench -- --ignored --nocapture
//!
//! Optional env vars:
//!   NANOBOT_SPECIALIST_BENCH_BASE=http://127.0.0.1:1234/v1
//!   NANOBOT_SPECIALIST_REPEATS=3
//!   NANOBOT_SPECIALIST_TARGET_TOKENS=420
//!   NANOBOT_SPECIALIST_MODELS=model_a,model_b,...

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Instant;

use serde_json::json;

use nanobot::agent::compaction::ContextCompactor;
use nanobot::agent::context_gate::ContentGate;
use nanobot::agent::token_budget::TokenBudget;
use nanobot::providers::base::LLMProvider;
use nanobot::providers::openai_compat::OpenAICompatProvider;

#[derive(Clone)]
struct Fixture {
    name: &'static str,
    content: String,
    must_include: Vec<&'static str>,
    must_exclude: Vec<&'static str>,
    safety_critical: bool,
    qa_checks: Vec<(&'static str, &'static str)>,
}

#[derive(Clone)]
struct BenchProfile {
    name: &'static str,
    temperature: f64,
    top_p: Option<f64>,
    thinking_budget: Option<u32>,
    strict_no_reasoning: bool,
}

#[derive(Clone)]
struct RunRow {
    model: String,
    profile: String,
    fixture: String,
    repeat: usize,
    latency_ms: u128,
    input_tokens: usize,
    output_tokens: usize,
    compression_ratio: f64,
    fact_recall: f64,
    qa_accuracy: f64,
    hallucinations: usize,
    thought_ratio: f64,
    verbosity_overrun: f64,
    fallback_like: bool,
    summary: String,
    context_retries: usize,
}

#[derive(Default)]
struct Aggregate {
    runs: usize,
    latencies: Vec<u128>,
    fact_recall_sum: f64,
    qa_sum: f64,
    comp_ratio_sum: f64,
    thought_ratio_sum: f64,
    verbosity_overrun_sum: f64,
    hallucinations_sum: usize,
    safety_hallucinations: usize,
    fallback_count: usize,
}

#[derive(Clone, Copy)]
enum BenchMode {
    Specialist,
    Compactor,
    Deterministic,
    Hybrid,
}

impl BenchMode {
    fn as_str(self) -> &'static str {
        match self {
            BenchMode::Specialist => "specialist",
            BenchMode::Compactor => "compactor",
            BenchMode::Deterministic => "deterministic",
            BenchMode::Hybrid => "hybrid",
        }
    }
}

fn long_fixtures() -> Vec<Fixture> {
    let rustc_error_trace = {
        let mut s = String::new();
        s.push_str("error[E0502]: cannot borrow `state` as mutable because it is also borrowed as immutable\n");
        s.push_str("  --> src/agent/router.rs:412:21\n");
        s.push_str("   |\n");
        s.push_str("398|     let config_ref = &state.config;\n");
        s.push_str("   |                      ------------ immutable borrow occurs here\n");
        s.push_str("412|     state.metrics.bump_retry();\n");
        s.push_str("   |     ^^^^^^^^^^^^^^^^^^^^^^^^^^ mutable borrow occurs here\n");
        s.push_str("\n");
        s.push_str("error[E0597]: `payload` does not live long enough\n");
        s.push_str("  --> src/agent/tool_engine.rs:288:36\n");
        for i in 0..80 {
            s.push_str(&format!(
                "note[{}]: while checking drop order in src/agent/tool_engine.rs:{}\n",
                i,
                280 + (i % 17)
            ));
        }
        s.push_str("error: aborting due to 2 previous errors\n");
        s
    };

    let cargo_test_failure = {
        let mut s = String::new();
        s.push_str("running 134 tests\n");
        s.push_str("test protocol::tests::local_assistant_preserves_tool_calls_for_lm_studio ... FAILED\n");
        s.push_str("test lcm::tests::test_second_compaction_summarizes_after_first_summary ... FAILED\n");
        for i in 0..120 {
            s.push_str(&format!("retry {}: waiting for local endpoint http://127.0.0.1:1234/v1\n", i + 1));
        }
        s.push_str("failures:\n");
        s.push_str("---- protocol::tests::local_assistant_preserves_tool_calls_for_lm_studio stdout ----\n");
        s.push_str("assertion failed: expected assistant tool_calls in session JSONL\n");
        s.push_str("---- lcm::tests::test_second_compaction_summarizes_after_first_summary stdout ----\n");
        s.push_str("assertion failed: Second compaction should not include messages from first summary\n");
        s.push_str("\ntest result: FAILED. 132 passed; 2 failed; 0 ignored\n");
        s
    };

    let json_api_dump = {
        let mut entries = Vec::new();
        for i in 0..240 {
            entries.push(json!({
                "id": i,
                "service": "billing",
                "status": if i == 173 { "error" } else { "ok" },
                "latencyMs": if i == 173 { 9821 } else { 140 + (i % 19) },
                "invoiceId": format!("INV-2026-{:04}", i),
                "customer": format!("CUST-{:05}", i * 7),
                "path": if i == 173 { "/v1/invoices/reconcile" } else { "/v1/invoices/post" },
                "error": if i == 173 { "checksum mismatch for invoice hash" } else { "" },
            }));
        }
        serde_json::to_string_pretty(&json!({
            "requestId": "req-7f3d9ab2",
            "region": "us-east-1",
            "entries": entries,
            "summary": {
                "ok": 239,
                "error": 1,
                "p95": 205,
                "max": 9821
            }
        }))
        .unwrap_or_default()
    };

    let ci_pipeline_log = {
        let mut s = String::new();
        s.push_str("[stage:build] cargo build --release\n");
        s.push_str("[stage:build] done in 121.3s\n");
        s.push_str("[stage:test] cargo test --all\n");
        for i in 0..140 {
            s.push_str(&format!("[stage:test] running shard {} of 140\n", i + 1));
        }
        s.push_str("[stage:test] FAIL: tests/protocol_tests.rs::local_assistant_preserves_tool_calls_for_lm_studio\n");
        s.push_str("[stage:deploy] skipped due to previous failure\n");
        s.push_str("final_status=failed\n");
        s
    };

    vec![
        Fixture {
            name: "rustc_error_trace",
            content: rustc_error_trace,
            must_include: vec![
                "E0502",
                "E0597",
                "src/agent/router.rs:412:21",
                "src/agent/tool_engine.rs:288:36",
                "aborting due to 2 previous errors",
            ],
            must_exclude: vec![
                "E9999",
                "src/nonexistent/file.rs",
                "network timeout to openai.com",
            ],
            safety_critical: true,
            qa_checks: vec![
                ("How many compiler errors are reported?", "2 previous errors"),
                ("Name one error code", "E0502"),
            ],
        },
        Fixture {
            name: "cargo_test_failure",
            content: cargo_test_failure,
            must_include: vec![
                "2 failed",
                "132 passed",
                "local_assistant_preserves_tool_calls_for_lm_studio",
                "test_second_compaction_summarizes_after_first_summary",
                "http://127.0.0.1:1234/v1",
            ],
            must_exclude: vec!["all tests passed", "0 failed", "segmentation fault"],
            safety_critical: true,
            qa_checks: vec![
                ("Did deploy run?", "skipped"),
                ("How many tests failed?", "2 failed"),
            ],
        },
        Fixture {
            name: "json_api_dump",
            content: json_api_dump,
            must_include: vec![
                "requestId",
                "req-7f3d9ab2",
                "INV-2026-0173",
                "checksum mismatch for invoice hash",
                "9821",
                "/v1/invoices/reconcile",
            ],
            must_exclude: vec!["payment gateway down", "INV-2026-9999", "eu-west-9"],
            safety_critical: true,
            qa_checks: vec![
                ("Which invoice had the error?", "INV-2026-0173"),
                ("What was max latency?", "9821"),
            ],
        },
        Fixture {
            name: "ci_pipeline_log",
            content: ci_pipeline_log,
            must_include: vec![
                "stage:test",
                "final_status=failed",
                "deploy] skipped",
                "protocol_tests.rs::local_assistant_preserves_tool_calls_for_lm_studio",
            ],
            must_exclude: vec!["final_status=success", "deploy completed", "artifact published"],
            safety_critical: false,
            qa_checks: vec![
                ("Which stage failed first?", "stage:test"),
                ("What happened to deploy?", "skipped"),
            ],
        },
    ]
}

fn default_models() -> Vec<String> {
    vec![
        "functiongemma-270m-it-mlx".to_string(),
        "lfm2-350m-mlx".to_string(),
        "qwen3-0.6b".to_string(),
        "llama-3.2-1b-instruct".to_string(),
        "qwen/qwen3-1.7b".to_string(),
        "nanbeige4.1-3b".to_string(),
        "qwen3-4b-mlx".to_string(),
    ]
}

fn profiles_for(model: &str) -> Vec<BenchProfile> {
    let lower = model.to_ascii_lowercase();
    if lower.contains("nanbeige") {
        vec![
            BenchProfile {
                name: "quality",
                temperature: 0.6,
                top_p: Some(0.95),
                thinking_budget: None,
                strict_no_reasoning: false,
            },
            BenchProfile {
                name: "concise",
                temperature: 0.2,
                top_p: Some(0.9),
                thinking_budget: None,
                strict_no_reasoning: true,
            },
            BenchProfile {
                name: "balanced",
                temperature: 0.35,
                top_p: Some(0.92),
                thinking_budget: None,
                strict_no_reasoning: true,
            },
        ]
    } else {
        vec![BenchProfile {
            name: "default",
            temperature: 0.2,
            top_p: Some(0.95),
            thinking_budget: None,
            strict_no_reasoning: true,
        }]
    }
}

fn env_models() -> Vec<String> {
    std::env::var("NANOBOT_SPECIALIST_MODELS")
        .ok()
        .map(|v| {
            v.split(',')
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
        })
        .filter(|v| !v.is_empty())
        .unwrap_or_else(default_models)
}

fn model_size_rank(model: &str) -> usize {
    let m = model.to_ascii_lowercase();
    if m.contains("270m") {
        270
    } else if m.contains("350m") {
        350
    } else if m.contains("0.5b") {
        500
    } else if m.contains("0.6b") {
        600
    } else if m.contains("1b") {
        1000
    } else if m.contains("1.2b") {
        1200
    } else if m.contains("1.7b") {
        1700
    } else if m.contains("3b") {
        3000
    } else if m.contains("4b") {
        4000
    } else {
        9999
    }
}

fn contains_case_insensitive(haystack: &str, needle: &str) -> bool {
    haystack
        .to_ascii_lowercase()
        .contains(&needle.to_ascii_lowercase())
}

fn recall_score(summary: &str, must_include: &[&str]) -> f64 {
    if must_include.is_empty() {
        return 1.0;
    }
    let hits = must_include
        .iter()
        .filter(|item| contains_case_insensitive(summary, item))
        .count();
    hits as f64 / must_include.len() as f64
}

fn qa_accuracy(summary: &str, checks: &[(&str, &str)]) -> f64 {
    if checks.is_empty() {
        return 1.0;
    }
    let hits = checks
        .iter()
        .filter(|(_, expected)| contains_case_insensitive(summary, expected))
        .count();
    hits as f64 / checks.len() as f64
}

fn count_forbidden(summary: &str, must_exclude: &[&str]) -> usize {
    must_exclude
        .iter()
        .filter(|item| contains_case_insensitive(summary, item))
        .count()
}

fn token_has_digit(t: &str) -> bool {
    t.chars().any(|c| c.is_ascii_digit())
}

fn hallucination_proxy(summary: &str, source: &str, must_exclude: &[&str]) -> usize {
    let mut count = count_forbidden(summary, must_exclude);
    let source_lower = source.to_ascii_lowercase();
    let mut seen = BTreeSet::new();
    for raw in summary.split_whitespace() {
        let t = raw
            .trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '/' && c != '_' && c != '.' && c != '-')
            .to_ascii_lowercase();
        if t.len() < 4 {
            continue;
        }
        // Ignore deterministic JSON pointer-style paths emitted by the gate,
        // they are synthetic anchors, not factual claims from source text.
        if t.starts_with("$.") || t.contains('[') || t.contains(']') {
            continue;
        }
        if !token_has_digit(&t) && !t.contains('/') {
            continue;
        }
        if !seen.insert(t.clone()) {
            continue;
        }
        if !source_lower.contains(&t) {
            count += 1;
        }
    }
    count
}

fn thought_ratio(summary: &str) -> f64 {
    let lower = summary.to_ascii_lowercase();
    let markers = [
        "<think>",
        "</think>",
        "let me think",
        "i should",
        "i need to",
        "analysis:",
        "reasoning:",
    ];
    let mut marker_hits = 0usize;
    for marker in &markers {
        marker_hits += lower.matches(marker).count();
    }
    let words = summary.split_whitespace().count().max(1);
    marker_hits as f64 / words as f64
}

fn is_fallback_like(summary: &str) -> bool {
    contains_case_insensitive(summary, "# Content Summary")
        || contains_case_insensitive(summary, "To inspect a section, use: read_file")
}

fn truncate_to_estimated_tokens(input: &str, max_tokens: usize) -> String {
    if max_tokens == 0 {
        return String::new();
    }
    let est = TokenBudget::estimate_str_tokens(input);
    if est <= max_tokens {
        return input.to_string();
    }
    let ratio = (max_tokens as f64 / est as f64).clamp(0.05, 1.0);
    let target_chars = ((input.chars().count() as f64) * ratio) as usize;
    if target_chars < 80 {
        return input.chars().take(target_chars).collect();
    }

    let marker = "\n... [truncated middle] ...\n";
    let head_chars = (target_chars as f64 * 0.6) as usize;
    let tail_chars = target_chars.saturating_sub(head_chars + marker.chars().count());

    let mut out = String::new();
    out.extend(input.chars().take(head_chars));
    out.push_str(marker);

    let tail: String = input
        .chars()
        .rev()
        .take(tail_chars)
        .collect::<String>()
        .chars()
        .rev()
        .collect();
    out.push_str(&tail);
    out
}

async fn list_available_models(base: &str) -> anyhow::Result<Vec<String>> {
    let url = format!("{}/models", base.trim_end_matches('/'));
    let value: serde_json::Value = reqwest::Client::new().get(&url).send().await?.json().await?;
    let data = value
        .get("data")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let out = data
        .iter()
        .filter_map(|m| m.get("id").and_then(|v| v.as_str()).map(|s| s.to_string()))
        .collect::<Vec<_>>();
    Ok(out)
}

async fn prewarm_model(base: &str, model: &str) -> anyhow::Result<()> {
    // LM Studio native endpoint (strip /v1 if present).
    let native = base.trim_end_matches('/').trim_end_matches("/v1");
    let already = loaded_instance_ids_for_model(base, model).await?;
    if !already.is_empty() {
        return Ok(());
    }
    let url = format!("{}/api/v1/models/load", native);
    let body = json!({ "model": model });
    let _ = reqwest::Client::new().post(&url).json(&body).send().await?;
    Ok(())
}

async fn list_loaded_instances(base: &str) -> anyhow::Result<Vec<(String, String)>> {
    let native = base.trim_end_matches('/').trim_end_matches("/v1");
    let url = format!("{}/api/v1/models", native);
    let value: serde_json::Value = reqwest::Client::new().get(&url).send().await?.json().await?;
    let mut out = Vec::new();
    if let Some(models) = value.get("models").and_then(|v| v.as_array()) {
        for model in models {
            let key = model
                .get("key")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            if let Some(instances) = model.get("loaded_instances").and_then(|v| v.as_array()) {
                for instance in instances {
                    if let Some(id) = instance.get("id").and_then(|v| v.as_str()) {
                        out.push((key.clone(), id.to_string()));
                    }
                }
            }
        }
    }
    Ok(out)
}

async fn loaded_instance_ids_for_model(base: &str, model: &str) -> anyhow::Result<Vec<String>> {
    let needle = model.to_ascii_lowercase();
    let instances = list_loaded_instances(base).await?;
    let out = instances
        .into_iter()
        .filter_map(|(key, id)| {
            let key_lower = key.to_ascii_lowercase();
            if key_lower == needle || key_lower.contains(&needle) || needle.contains(&key_lower) {
                Some(id)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    Ok(out)
}

async fn unload_instance(base: &str, instance_id: &str) -> anyhow::Result<()> {
    let native = base.trim_end_matches('/').trim_end_matches("/v1");
    let url = format!("{}/api/v1/models/unload", native);
    let body = json!({ "instance_id": instance_id });
    let _ = reqwest::Client::new().post(&url).json(&body).send().await?;
    Ok(())
}

async fn unload_all_loaded_instances(base: &str) -> anyhow::Result<usize> {
    let instances = list_loaded_instances(base).await?;
    let mut unloaded = 0usize;
    for (_, id) in instances {
        if unload_instance(base, &id).await.is_ok() {
            unloaded += 1;
        }
    }
    Ok(unloaded)
}

fn model_available(available: &[String], wanted: &str) -> bool {
    let needle = wanted.to_ascii_lowercase();
    available.iter().any(|m| {
        let a = m.to_ascii_lowercase();
        a == needle || a.contains(&needle) || needle.contains(&a)
    })
}

async fn summarize_once(
    provider: Arc<dyn LLMProvider>,
    model: &str,
    source_content: &str,
    target_tokens: usize,
    _profile: &BenchProfile,
    mode: BenchMode,
) -> anyhow::Result<(String, usize, usize)> {
    let input_tokens = TokenBudget::estimate_str_tokens(source_content);
    // We benchmark the real production path: tool output -> ContentGate -> specialist.
    // Gate target is approximately available/2, so choose max_tokens ~= 2.5 * target.
    let mut gate_max_tokens = ((target_tokens as f64) * 2.5).ceil() as usize;
    gate_max_tokens = gate_max_tokens.max(64);

    for attempt in 0..3 {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let cache_dir = std::env::temp_dir().join(format!(
            "nanobot_specialist_gate_bench_{}_{}_{}",
            std::process::id(),
            stamp,
            attempt
        ));
        let mut gate = ContentGate::new(gate_max_tokens, 0.20, cache_dir);
        let out = match mode {
            BenchMode::Specialist => {
                gate.admit_with_specialist(source_content, provider.as_ref(), model)
                    .await
            }
            BenchMode::Compactor => {
                let compactor = ContextCompactor::new(provider.clone(), model.to_string(), 4096);
                gate.admit_with_compactor(source_content, &compactor).await
            }
            BenchMode::Deterministic => gate.admit_with_deterministic(source_content),
            BenchMode::Hybrid => {
                gate.admit_with_hybrid(source_content, provider.as_ref(), model)
                    .await
            }
        };
        match out {
            nanobot::agent::context_gate::GateResult::Briefing { summary, .. } => {
                return Ok((summary, attempt, input_tokens));
            }
            nanobot::agent::context_gate::GateResult::Raw(raw) => {
                if attempt == 2 {
                    return Ok((raw, attempt, input_tokens));
                }
                gate_max_tokens = gate_max_tokens.saturating_div(2).max(32);
            }
        }
    }

    Ok((String::new(), 0, input_tokens))
}

fn p95(mut latencies: Vec<u128>) -> u128 {
    if latencies.is_empty() {
        return 0;
    }
    latencies.sort_unstable();
    let idx = ((latencies.len() as f64) * 0.95).ceil() as usize;
    let idx = idx.saturating_sub(1).min(latencies.len() - 1);
    latencies[idx]
}

#[tokio::test]
#[ignore = "requires local OpenAI-compatible endpoint with benchmark models"]
async fn bench_specialist_summaries_small_to_3b() {
    let base = std::env::var("NANOBOT_SPECIALIST_BENCH_BASE")
        .unwrap_or_else(|_| "http://127.0.0.1:1234/v1".to_string());
    let repeats: usize = std::env::var("NANOBOT_SPECIALIST_REPEATS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(3);
    let target_tokens: usize = std::env::var("NANOBOT_SPECIALIST_TARGET_TOKENS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(420);
    let fixture_max_tokens: usize = std::env::var("NANOBOT_SPECIALIST_FIXTURE_MAX_TOKENS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1400);
    let prewarm = std::env::var("NANOBOT_SPECIALIST_PREWARM")
        .ok()
        .map(|v| {
            let lower = v.to_ascii_lowercase();
            lower == "1" || lower == "true" || lower == "yes"
        })
        .unwrap_or(true);
    let isolate_models = std::env::var("NANOBOT_SPECIALIST_ISOLATE_MODELS")
        .ok()
        .map(|v| {
            let lower = v.to_ascii_lowercase();
            lower == "1" || lower == "true" || lower == "yes"
        })
        .unwrap_or(true);
    let mode = std::env::var("NANOBOT_SPECIALIST_BENCH_MODE")
        .ok()
        .map(|v| match v.to_ascii_lowercase().as_str() {
            "compactor" => BenchMode::Compactor,
            "deterministic" => BenchMode::Deterministic,
            "hybrid" => BenchMode::Hybrid,
            _ => BenchMode::Specialist,
        })
        .unwrap_or(BenchMode::Specialist);

    let requested_models = env_models();
    let available = list_available_models(&base)
        .await
        .expect("failed to fetch /v1/models from local endpoint");

    eprintln!("\n{}", "=".repeat(96));
    eprintln!("SPECIALIST SUMMARY BENCHMARK (long tool outputs)");
    eprintln!("base: {}", base);
    eprintln!(
        "repeats: {}  target_tokens: {}  fixture_max_tokens: {}  prewarm: {}  mode: {}",
        repeats,
        target_tokens,
        fixture_max_tokens,
        prewarm,
        mode.as_str()
    );
    eprintln!("available models: {}", available.len());
    eprintln!("isolate_models: {}", isolate_models);
    eprintln!("{}\n", "=".repeat(96));

    if isolate_models {
        match unload_all_loaded_instances(&base).await {
            Ok(n) => eprintln!("unloaded stale instances at start: {}", n),
            Err(e) => eprintln!("[WARN] failed to unload stale instances: {}", e),
        }
    }

    let fixtures = long_fixtures();
    let mut rows: Vec<RunRow> = Vec::new();
    let mut skipped: Vec<String> = Vec::new();

    for model in &requested_models {
        if isolate_models {
            match unload_all_loaded_instances(&base).await {
                Ok(n) if n > 0 => eprintln!("unloaded instances before model {}: {}", model, n),
                Ok(_) => {}
                Err(e) => eprintln!("[WARN] pre-model unload failed for {}: {}", model, e),
            }
        }

        if !model_available(&available, model) {
            skipped.push(model.clone());
            continue;
        }

        if prewarm {
            if let Err(e) = prewarm_model(&base, model).await {
                eprintln!("[WARN] prewarm failed for {}: {}", model, e);
            }
        }

        let provider: Arc<dyn LLMProvider> =
            Arc::new(OpenAICompatProvider::new("local", Some(&base), Some(model)));
        let profiles = profiles_for(model);
        for profile in &profiles {
            for fixture in &fixtures {
                let trimmed_fixture = if fixture.name.contains("json") {
                    fixture.content.clone()
                } else {
                    truncate_to_estimated_tokens(&fixture.content, fixture_max_tokens)
                };
                for repeat in 1..=repeats {
                    let start = Instant::now();
                    let (summary, context_retries, input_tokens) = match summarize_once(
                        provider.clone(),
                        model,
                        &trimmed_fixture,
                        target_tokens,
                        profile,
                        mode,
                    )
                    .await
                    {
                        Ok(v) => v,
                        Err(e) => {
                            let msg = format!(
                                "[ERROR] model={} profile={} fixture={} repeat={}: {}",
                                model, profile.name, fixture.name, repeat, e
                            );
                            eprintln!("{}", msg);
                            (String::new(), 0usize, TokenBudget::estimate_str_tokens(&trimmed_fixture))
                        }
                    };

                    let latency_ms = start.elapsed().as_millis();
                    let output_tokens = TokenBudget::estimate_str_tokens(&summary);
                    let compression_ratio = if input_tokens > 0 {
                        output_tokens as f64 / input_tokens as f64
                    } else {
                        1.0
                    };
                    let fact_recall = recall_score(&summary, &fixture.must_include);
                    let qa = qa_accuracy(&summary, &fixture.qa_checks);
                    let hall = hallucination_proxy(&summary, &trimmed_fixture, &fixture.must_exclude);
                    let think = thought_ratio(&summary);
                    let overrun = output_tokens as f64 / target_tokens.max(1) as f64;
                    let fallback_like = is_fallback_like(&summary);

                    rows.push(RunRow {
                        model: model.clone(),
                        profile: profile.name.to_string(),
                        fixture: fixture.name.to_string(),
                        repeat,
                        latency_ms,
                        input_tokens,
                        output_tokens,
                        compression_ratio,
                        fact_recall,
                        qa_accuracy: qa,
                        hallucinations: hall,
                        thought_ratio: think,
                        verbosity_overrun: overrun,
                        fallback_like,
                        summary,
                        context_retries,
                    });
                }
            }
        }

        if isolate_models {
            match unload_all_loaded_instances(&base).await {
                Ok(n) if n > 0 => eprintln!("unloaded instances after model {}: {}", model, n),
                Ok(_) => {}
                Err(e) => eprintln!("[WARN] post-model unload failed for {}: {}", model, e),
            }
        }
    }

    if !skipped.is_empty() {
        eprintln!("Skipped models (not available): {}", skipped.join(", "));
    }

    assert!(
        !rows.is_empty(),
        "No benchmark rows were produced. Ensure endpoint and models are available."
    );

    let mut agg: BTreeMap<(String, String), Aggregate> = BTreeMap::new();
    for row in &rows {
        let key = (row.model.clone(), row.profile.clone());
        let entry = agg.entry(key).or_default();
        entry.runs += 1;
        entry.latencies.push(row.latency_ms);
        entry.fact_recall_sum += row.fact_recall;
        entry.qa_sum += row.qa_accuracy;
        entry.comp_ratio_sum += row.compression_ratio;
        entry.thought_ratio_sum += row.thought_ratio;
        entry.verbosity_overrun_sum += row.verbosity_overrun;
        entry.hallucinations_sum += row.hallucinations;
        entry.fallback_count += usize::from(row.fallback_like);
        let fix = fixtures.iter().find(|f| f.name == row.fixture).unwrap();
        if fix.safety_critical {
            entry.safety_hallucinations += row.hallucinations;
        }
    }

    eprintln!("\n{:-<96}", "");
    eprintln!(
        "{:<30} {:<10} {:>6} {:>7} {:>7} {:>9} {:>9} {:>8} {:>8} {:>8}",
        "model",
        "profile",
        "runs",
        "recall",
        "qa",
        "hall",
        "p95ms",
        "compr%",
        "overrun",
        "fbk%"
    );
    eprintln!("{:-<96}", "");

    let mut ranked: Vec<(String, String, f64, bool)> = Vec::new();

    for ((model, profile), a) in &agg {
        let n = a.runs.max(1) as f64;
        let avg_recall = a.fact_recall_sum / n;
        let avg_qa = a.qa_sum / n;
        let avg_comp = a.comp_ratio_sum / n;
        let avg_overrun = a.verbosity_overrun_sum / n;
        let hall_rate = a.hallucinations_sum as f64 / n;
        let fbk_rate = a.fallback_count as f64 / n;
        let p95_ms = p95(a.latencies.clone());
        let quality = 0.45 * avg_recall + 0.30 * avg_qa + 0.25 * (1.0 / (1.0 + hall_rate));
        let pass = avg_recall >= 0.90
            && avg_qa >= 0.85
            && a.safety_hallucinations == 0
            && p95_ms <= 2500
            && fbk_rate <= 0.20;

        eprintln!(
            "{:<30} {:<10} {:>6} {:>7.3} {:>7.3} {:>9.3} {:>9} {:>7.1}% {:>8.2} {:>7.1}%{}",
            model,
            profile,
            a.runs,
            avg_recall,
            avg_qa,
            hall_rate,
            p95_ms,
            (1.0 - avg_comp).max(0.0) * 100.0,
            avg_overrun,
            fbk_rate * 100.0,
            if pass { "  PASS" } else { "  FAIL" }
        );
        ranked.push((model.clone(), profile.clone(), quality, pass));
    }

    ranked.sort_by(|a, b| {
        let ra = model_size_rank(&a.0);
        let rb = model_size_rank(&b.0);
        ra.cmp(&rb)
            .then_with(|| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal))
    });

    let smallest_passing = ranked.iter().find(|(_, _, _, pass)| *pass).cloned();
    let best_overall = ranked
        .iter()
        .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
        .cloned();

    eprintln!("\n{:-<96}", "");
    if let Some((m, p, score, _)) = smallest_passing {
        eprintln!(
            "Recommended specialist (smallest passing): {} [{}]  score={:.3}",
            m, p, score
        );
    } else {
        eprintln!("No profile passed strict gate. Keep current specialist and inspect failures.");
    }
    if let Some((m, p, score, _)) = best_overall {
        eprintln!("Best overall quality score: {} [{}]  score={:.3}", m, p, score);
    }

    // Persist compact JSON report.
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let out_dir = std::env::temp_dir().join("nanobot_specialist_bench");
    std::fs::create_dir_all(&out_dir).expect("failed to create benchmark output dir");
    let out_path = out_dir.join(format!("specialist_summary_bench_{}.json", ts));

    let rows_json = rows
        .iter()
        .map(|r| {
            json!({
                "model": r.model,
                "profile": r.profile,
                "fixture": r.fixture,
                "repeat": r.repeat,
                "latencyMs": r.latency_ms,
                "inputTokens": r.input_tokens,
                "outputTokens": r.output_tokens,
                "compressionRatio": r.compression_ratio,
                "factRecall": r.fact_recall,
                "qaAccuracy": r.qa_accuracy,
                "hallucinations": r.hallucinations,
                "thoughtRatio": r.thought_ratio,
                "verbosityOverrun": r.verbosity_overrun,
                "fallbackLike": r.fallback_like,
                "contextRetries": r.context_retries,
                "summary": r.summary,
                "summaryPreview": r.summary.chars().take(240).collect::<String>(),
            })
        })
        .collect::<Vec<_>>();

    let agg_json = agg
        .iter()
        .map(|((m, p), a)| {
            let n = a.runs.max(1) as f64;
            json!({
                "model": m,
                "profile": p,
                "runs": a.runs,
                "avgFactRecall": a.fact_recall_sum / n,
                "avgQaAccuracy": a.qa_sum / n,
                "avgCompressionRatio": a.comp_ratio_sum / n,
                "avgThoughtRatio": a.thought_ratio_sum / n,
                "avgVerbosityOverrun": a.verbosity_overrun_sum / n,
                "avgHallucinations": a.hallucinations_sum as f64 / n,
                "safetyHallucinations": a.safety_hallucinations,
                "fallbackRate": a.fallback_count as f64 / n,
                "p95LatencyMs": p95(a.latencies.clone()),
            })
        })
        .collect::<Vec<_>>();

    let payload = json!({
        "base": base,
        "repeats": repeats,
        "targetTokens": target_tokens,
        "modelsRequested": requested_models,
        "modelsSkipped": skipped,
        "fixtures": fixtures.iter().map(|f| f.name).collect::<Vec<_>>(),
        "rows": rows_json,
        "aggregates": agg_json,
    });

    std::fs::write(&out_path, serde_json::to_string_pretty(&payload).unwrap_or_default())
        .expect("failed to write benchmark report json");
    eprintln!("Saved benchmark JSON: {}", out_path.display());

    assert!(
        !agg.is_empty(),
        "Benchmark finished without aggregates; this indicates a harness issue."
    );
}
