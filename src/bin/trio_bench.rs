//! trio_bench — benchmark that compares solo vs trio routing configs.
//!
//! Runs 20 prompts against real LM Studio endpoints and writes a markdown
//! report to `~/.nanobot/workspace/experiments/trio_bench_YYYY-MM-DD.md`.
//!
//! Usage:
//!   cargo run --bin trio_bench
//!   cargo run --bin trio_bench -- --config trio_current --quick
//!   cargo run --bin trio_bench -- --out /tmp/bench.md

use std::fs;
use std::io::Write as _;
use std::time::Instant;

use chrono::Local;
use serde_json::{json, Value};

use nanobot::agent::router::{build_conversation_tail, request_strict_router_decision};
use nanobot::config::loader::load_config;
use nanobot::providers::base::LLMProvider;
use nanobot::providers::openai_compat::OpenAICompatProvider;

// ---------------------------------------------------------------------------
// Sweep configurations
// ---------------------------------------------------------------------------

struct SweepConfig {
    label: &'static str,
    trio_enabled: bool,
    router_temp: f64,
    specialist_temp: f64,
    router_ctx: usize,
}

const SWEEPS: &[SweepConfig] = &[
    SweepConfig {
        label: "solo",
        trio_enabled: false,
        router_temp: 0.0,
        specialist_temp: 0.0,
        router_ctx: 0,
    },
    SweepConfig {
        label: "trio_conservative",
        trio_enabled: true,
        router_temp: 0.1,
        specialist_temp: 0.3,
        router_ctx: 4096,
    },
    SweepConfig {
        label: "trio_current",
        trio_enabled: true,
        router_temp: 0.6,
        specialist_temp: 0.7,
        router_ctx: 8192,
    },
    SweepConfig {
        label: "trio_warm",
        trio_enabled: true,
        router_temp: 0.3,
        specialist_temp: 0.7,
        router_ctx: 8192,
    },
    SweepConfig {
        label: "trio_exploratory",
        trio_enabled: true,
        router_temp: 0.4,
        specialist_temp: 0.9,
        router_ctx: 8192,
    },
];

// ---------------------------------------------------------------------------
// Scenarios
// ---------------------------------------------------------------------------

struct Scenario {
    idx: usize,
    prompt: &'static str,
    expected: &'static str, // "respond", "tool", "specialist"
    max_tokens: u32,
}

fn build_scenarios() -> Vec<Scenario> {
    vec![
        // 1-5: simple factual / conversational — expected: respond
        Scenario { idx: 1,  prompt: "What is 7 × 8?",                          expected: "respond", max_tokens: 64 },
        Scenario { idx: 2,  prompt: "Say hello in a friendly way.",             expected: "respond", max_tokens: 64 },
        Scenario { idx: 3,  prompt: "What year did World War II end?",          expected: "respond", max_tokens: 64 },
        Scenario { idx: 4,  prompt: "Translate 'cat' to French.",               expected: "respond", max_tokens: 64 },
        Scenario { idx: 5,  prompt: "What is the capital of Japan?",            expected: "respond", max_tokens: 64 },
        // 6-10: file / shell / tool tasks — expected: tool
        Scenario { idx: 6,  prompt: "Read the file README.md and summarize it.", expected: "tool", max_tokens: 128 },
        Scenario { idx: 7,  prompt: "List all files in the /tmp directory.",    expected: "tool", max_tokens: 128 },
        Scenario { idx: 8,  prompt: "Run the command `echo hello world` and show me the output.", expected: "tool", max_tokens: 128 },
        Scenario { idx: 9,  prompt: "What is the content of Cargo.toml?",      expected: "tool", max_tokens: 128 },
        Scenario { idx: 10, prompt: "Show me the last 5 git commits.",          expected: "tool", max_tokens: 128 },
        // 11-15: deep coding / reasoning — expected: specialist
        Scenario { idx: 11, prompt: "Write a bubble sort implementation in Rust.", expected: "specialist", max_tokens: 256 },
        Scenario { idx: 12, prompt: "Explain how async/await works in Rust with examples.", expected: "specialist", max_tokens: 256 },
        Scenario { idx: 13, prompt: "Debug this error: `cannot borrow as mutable because it is also borrowed as immutable`", expected: "specialist", max_tokens: 256 },
        Scenario { idx: 14, prompt: "Write a regex pattern that matches email addresses.", expected: "specialist", max_tokens: 256 },
        Scenario { idx: 15, prompt: "Explain the difference between HashMap and BTreeMap in Rust.", expected: "specialist", max_tokens: 256 },
        // 16-20: context-stress — mark as "respond" for accuracy calculation
        Scenario { idx: 16, prompt: "How are you doing today?",                 expected: "respond", max_tokens: 64 },
        Scenario { idx: 17, prompt: "Based on our conversation, what can you tell me?", expected: "respond", max_tokens: 64 },
        Scenario { idx: 18, prompt: "Can you explain what we discussed earlier?", expected: "respond", max_tokens: 64 },
        Scenario { idx: 19, prompt: "What would you recommend given everything so far?", expected: "respond", max_tokens: 64 },
        Scenario { idx: 20, prompt: "Summarize what we have talked about.",     expected: "respond", max_tokens: 64 },
    ]
}

/// Build synthetic message history for context-stress scenarios 16-20.
fn build_context_history(scenario_idx: usize) -> Vec<Value> {
    match scenario_idx {
        16 => vec![
            json!({"role": "user",      "content": "Hi there!"}),
            json!({"role": "assistant", "content": "Hello! How can I help you today?"}),
            json!({"role": "user",      "content": "How are you doing today?"}),
        ],
        17 => {
            let context = "We have been discussing the Rust programming language and its ownership model. \
                           The borrow checker ensures memory safety at compile time, preventing data races \
                           and use-after-free bugs. This is one of Rust's core value propositions.";
            vec![
                json!({"role": "user",      "content": "Hi there!"}),
                json!({"role": "assistant", "content": "Hello! How can I help you today?"}),
                json!({"role": "user",      "content": context}),
                json!({"role": "assistant", "content": "That's a great summary of Rust's ownership model!"}),
                json!({"role": "user",      "content": "Based on our conversation, what can you tell me?"}),
            ]
        }
        18 => {
            let code = "```rust\nfn main() {\n    let v = vec![1, 2, 3];\n    let r = &v[0];\n    v.push(4); // error: cannot borrow `v` as mutable\n    println!(\"{}\", r);\n}\n```";
            vec![
                json!({"role": "user",      "content": "I have a Rust question."}),
                json!({"role": "assistant", "content": "Sure, I'd be happy to help with Rust!"}),
                json!({"role": "user",      "content": code}),
                json!({"role": "assistant", "content": "This is a classic borrow checker error. The reference `r` borrows `v` immutably, then `v.push(4)` tries to borrow it mutably simultaneously."}),
                json!({"role": "user",      "content": "Can you explain what we discussed earlier?"}),
            ]
        }
        19 => {
            let context = "A".repeat(500);
            vec![
                json!({"role": "user",      "content": "Starting a new conversation."}),
                json!({"role": "assistant", "content": "Hello! Ready to chat."}),
                json!({"role": "user",      "content": context}),
                json!({"role": "assistant", "content": "I've read your message."}),
                json!({"role": "user",      "content": "What would you recommend given everything so far?"}),
            ]
        }
        20 => {
            let context = "B".repeat(1000);
            vec![
                json!({"role": "user",      "content": "Here is some background context."}),
                json!({"role": "assistant", "content": "Got it, I'll keep that in mind."}),
                json!({"role": "user",      "content": context}),
                json!({"role": "assistant", "content": "Thank you for the context."}),
                json!({"role": "user",      "content": "Summarize what we have talked about."}),
            ]
        }
        _ => vec![],
    }
}

// ---------------------------------------------------------------------------
// Result struct
// ---------------------------------------------------------------------------

struct ScenarioResult {
    idx: usize,
    prompt_short: String,
    expected: &'static str,
    router_action: Option<String>,
    router_confidence: Option<f64>,
    specialist_dispatched: bool,
    routing_correct: bool,
    router_latency_ms: Option<u64>,
    total_latency_ms: u64,
    response_preview: String,
    #[allow(dead_code)]
    context_summary_found: bool,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn build_router_pack(prompt: &str, tool_names: &str) -> String {
    format!("User request: {}\nAvailable tools: {}", prompt, tool_names)
}

fn build_solo_messages(prompt: &str) -> Vec<Value> {
    vec![json!({"role": "user", "content": prompt})]
}

fn build_provider_messages(scenario: &Scenario) -> Vec<Value> {
    // For context-stress scenarios (16-20), use pre-built history
    if scenario.idx >= 16 {
        build_context_history(scenario.idx)
    } else {
        vec![json!({"role": "user", "content": scenario.prompt})]
    }
}

fn truncate_to(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..FloorCharBoundary::floor_char_boundary(s, max)])
    }
}

/// Extension trait to get floor char boundary for clean UTF-8 truncation.
trait FloorCharBoundary {
    fn floor_char_boundary(&self, index: usize) -> usize;
}

impl FloorCharBoundary for str {
    fn floor_char_boundary(&self, index: usize) -> usize {
        if index >= self.len() {
            return self.len();
        }
        let mut i = index;
        while i > 0 && !self.is_char_boundary(i) {
            i -= 1;
        }
        i
    }
}

// ---------------------------------------------------------------------------
// CLI arg parsing
// ---------------------------------------------------------------------------

struct CliArgs {
    config_filter: Option<String>,
    quick: bool,
    out: Option<String>,
}

fn parse_args() -> CliArgs {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut config_filter = None;
    let mut quick = false;
    let mut out = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--config" => {
                i += 1;
                if i < args.len() {
                    config_filter = Some(args[i].clone());
                }
            }
            "--quick" => {
                quick = true;
            }
            "--out" => {
                i += 1;
                if i < args.len() {
                    out = Some(args[i].clone());
                }
            }
            other => {
                eprintln!("Warning: unknown argument: {}", other);
            }
        }
        i += 1;
    }

    CliArgs { config_filter, quick, out }
}

// ---------------------------------------------------------------------------
// Endpoint extraction
// ---------------------------------------------------------------------------

struct EndpointInfo {
    url: String,
    model: String,
}

fn extract_endpoints(config: &nanobot::config::schema::Config) -> (EndpointInfo, EndpointInfo, EndpointInfo) {
    // Router endpoint
    let router = if let Some(ep) = &config.trio.router_endpoint {
        EndpointInfo { url: ep.url.clone(), model: ep.model.clone() }
    } else {
        let port = config.trio.router_port;
        let model = config.trio.router_model.clone();
        EndpointInfo {
            url: format!("http://localhost:{}/v1", port),
            model: if model.is_empty() { "router-model".to_string() } else { model },
        }
    };

    // Specialist endpoint
    let specialist = if let Some(ep) = &config.trio.specialist_endpoint {
        EndpointInfo { url: ep.url.clone(), model: ep.model.clone() }
    } else {
        let port = config.trio.specialist_port;
        let model = config.trio.specialist_model.clone();
        EndpointInfo {
            url: format!("http://localhost:{}/v1", port),
            model: if model.is_empty() { "specialist-model".to_string() } else { model },
        }
    };

    // Main (solo) endpoint — use local_api_base if set, then fall back to the
    // router/specialist endpoint URL (all three roles share one LM Studio server),
    // and only construct http://localhost:{port}/v1 as a last resort.
    let main = {
        let base = &config.agents.defaults.local_api_base;
        let url = if !base.is_empty() {
            base.clone()
        } else if let Some(ep) = &config.trio.router_endpoint {
            ep.url.clone()
        } else if let Some(ep) = &config.trio.specialist_endpoint {
            ep.url.clone()
        } else {
            format!("http://localhost:{}/v1", config.agents.defaults.lms_port)
        };
        let model = {
            let lms = &config.agents.defaults.lms_main_model;
            if !lms.is_empty() {
                lms.clone()
            } else {
                let local = &config.agents.defaults.local_model;
                if !local.is_empty() {
                    // Strip .gguf suffix for LM Studio model identifier
                    local
                        .trim_end_matches(".gguf")
                        .split('/')
                        .last()
                        .unwrap_or(local.as_str())
                        .to_string()
                } else {
                    config.agents.defaults.model.clone()
                }
            }
        };
        EndpointInfo { url, model }
    };

    (router, specialist, main)
}

// ---------------------------------------------------------------------------
// Report generation
// ---------------------------------------------------------------------------

fn generate_markdown_report(
    all_results: &[(String, Vec<ScenarioResult>)],
    date_str: &str,
) -> String {
    let mut md = String::new();

    md.push_str(&format!("# Trio Bench — {}\n\n", date_str));

    // Summary table
    md.push_str("## Summary: Routing Accuracy by Config\n\n");
    md.push_str("| Config | Accuracy | Avg Router Latency | Avg Total Latency |\n");
    md.push_str("|--------|----------|--------------------|-------------------|\n");

    for (label, results) in all_results {
        if results.is_empty() {
            md.push_str(&format!("| {} | N/A | N/A | N/A |\n", label));
            continue;
        }
        let correct = results.iter().filter(|r| r.routing_correct).count();
        let accuracy = correct as f64 / results.len() as f64 * 100.0;

        let router_ms: Vec<u64> = results.iter().filter_map(|r| r.router_latency_ms).collect();
        let avg_router_ms = if router_ms.is_empty() {
            "N/A".to_string()
        } else {
            format!("{:.0}ms", router_ms.iter().sum::<u64>() as f64 / router_ms.len() as f64)
        };

        let avg_total_ms = results.iter().map(|r| r.total_latency_ms).sum::<u64>() as f64
            / results.len() as f64;

        md.push_str(&format!(
            "| {} | {:.1}% ({}/{}) | {} | {:.0}ms |\n",
            label, accuracy, correct, results.len(), avg_router_ms, avg_total_ms
        ));
    }
    md.push('\n');

    // Per-sweep detail tables
    for (label, results) in all_results {
        // Find sweep config for label
        let sweep = SWEEPS.iter().find(|s| s.label == label.as_str());
        let (rt, st) = sweep.map(|s| (s.router_temp, s.specialist_temp)).unwrap_or((0.0, 0.0));

        md.push_str(&format!(
            "## Config: {} (router_temp={:.1}, spec_temp={:.1})\n\n",
            label, rt, st
        ));

        if results.is_empty() {
            md.push_str("_No results recorded._\n\n");
            continue;
        }

        md.push_str("| # | Prompt (short) | Expected | Got | Conf | Spec? | R-ms | T-ms |\n");
        md.push_str("|---|----------------|----------|-----|------|-------|------|------|\n");

        for r in results {
            let got = r.router_action.as_deref().unwrap_or("—");
            let conf = r
                .router_confidence
                .map(|c| format!("{:.2}", c))
                .unwrap_or_else(|| "—".to_string());
            let spec = if r.specialist_dispatched { "yes" } else { "no" };
            let r_ms = r
                .router_latency_ms
                .map(|m| m.to_string())
                .unwrap_or_else(|| "—".to_string());
            let correct_marker = if r.routing_correct { "" } else { " **X**" };
            md.push_str(&format!(
                "| {} | {} | {} | {}{} | {} | {} | {} | {} |\n",
                r.idx,
                r.prompt_short,
                r.expected,
                got,
                correct_marker,
                conf,
                spec,
                r_ms,
                r.total_latency_ms,
            ));
        }
        md.push('\n');

        // Show response previews
        md.push_str("### Response Previews\n\n");
        for r in results {
            if !r.response_preview.is_empty() {
                md.push_str(&format!("**[{}] {}**\n\n", r.idx, r.prompt_short));
                md.push_str(&format!("_{}_\n\n", r.response_preview.replace('\n', " ")));
            }
        }
        md.push('\n');
    }

    // Solo vs Trio comparison for selected prompts
    md.push_str("## Solo vs Trio Response Comparison — Selected Prompts\n\n");
    let selected_indices = [1, 5, 11, 15];

    for idx in &selected_indices {
        md.push_str(&format!("### Scenario {}\n\n", idx));
        for (label, results) in all_results {
            if let Some(r) = results.iter().find(|r| r.idx == *idx) {
                md.push_str(&format!("**{}**: {}\n\n", label, r.response_preview.replace('\n', " ")));
            }
        }
        md.push('\n');
    }

    md
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    // Initialize minimal tracing to stderr so output doesn't pollute stdout
    use tracing_subscriber::EnvFilter;
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();

    let args = parse_args();
    let date_str = Local::now().format("%Y-%m-%d").to_string();

    // Load config
    let config = load_config(None);

    // Extract endpoints
    let (router_ep, spec_ep, main_ep) = extract_endpoints(&config);

    println!("Trio Bench — {}", date_str);
    println!("Router:     {} / {}", router_ep.url, router_ep.model);
    println!("Specialist: {} / {}", spec_ep.url, spec_ep.model);
    println!("Solo/Main:  {} / {}", main_ep.url, main_ep.model);
    println!();

    // Build providers
    let router_provider = OpenAICompatProvider::new("", Some(&router_ep.url), Some(&router_ep.model));
    let spec_provider   = OpenAICompatProvider::new("", Some(&spec_ep.url),   Some(&spec_ep.model));
    let solo_provider   = OpenAICompatProvider::new("", Some(&main_ep.url),   Some(&main_ep.model));

    let tool_names = "read_file,list_directory,run_bash,fetch_url,spawn_agent";

    // Build scenarios list
    let all_scenarios = build_scenarios();

    // Determine which sweeps to run
    let sweeps_to_run: Vec<&SweepConfig> = SWEEPS
        .iter()
        .filter(|s| {
            if let Some(ref filter) = args.config_filter {
                s.label == filter.as_str()
            } else {
                true
            }
        })
        .collect();

    if sweeps_to_run.is_empty() {
        eprintln!(
            "Error: no sweep matches --config filter '{}'",
            args.config_filter.as_deref().unwrap_or("")
        );
        std::process::exit(1);
    }

    // Determine scenario slice
    let scenario_limit = if args.quick { 5 } else { all_scenarios.len() };
    let scenarios = &all_scenarios[..scenario_limit];

    let mut all_results: Vec<(String, Vec<ScenarioResult>)> = vec![];

    for sweep in &sweeps_to_run {
        println!("=== Running sweep: {} ===", sweep.label);
        let mut results: Vec<ScenarioResult> = vec![];

        for (si, scenario) in scenarios.iter().enumerate() {
            let prompt_short = truncate_to(scenario.prompt, 40);
            print!(
                "  [{}/{}] {} ...",
                si + 1,
                scenarios.len(),
                prompt_short
            );
            let _ = std::io::stdout().flush();

            let t_start = Instant::now();

            if sweep.trio_enabled {
                // Build history messages for context-tail (mainly used for scenarios 16-20)
                let history_msgs = build_context_history(scenario.idx);
                let tail = if !history_msgs.is_empty() {
                    build_conversation_tail(&history_msgs, 4, 500, sweep.router_ctx.max(1))
                } else {
                    String::new()
                };
                let context_summary_found =
                    nanobot::agent::router::find_scratch_pad_summary_in_messages(&history_msgs)
                        .is_some();

                let router_pack = if tail.is_empty() {
                    build_router_pack(scenario.prompt, tool_names)
                } else {
                    format!(
                        "Recent context:\n{}\n\nUser request: {}\nAvailable tools: {}",
                        tail, scenario.prompt, tool_names
                    )
                };

                let t_router = Instant::now();
                let decision_result = request_strict_router_decision(
                    &router_provider as &dyn LLMProvider,
                    &router_ep.model,
                    &router_pack,
                    config.trio.router_no_think,
                    sweep.router_temp,
                    config.trio.router_top_p,
                    tool_names,
                )
                .await;
                let router_ms = t_router.elapsed().as_millis() as u64;

                match decision_result {
                    Ok(decision) => {
                        let specialist_dispatched = decision.action == "specialist";

                        // Build messages for the downstream model call
                        let msgs = build_provider_messages(scenario);

                        let (provider_ref, model_str): (&dyn LLMProvider, &str) =
                            if specialist_dispatched {
                                (&spec_provider, &spec_ep.model)
                            } else {
                                (&solo_provider, &main_ep.model)
                            };

                        let response = provider_ref
                            .chat(
                                &msgs,
                                None,
                                Some(model_str),
                                scenario.max_tokens,
                                sweep.specialist_temp,
                                None,
                                None,
                            )
                            .await;

                        let response_preview = match &response {
                            Ok(r) => truncate_to(
                                r.content.as_deref().unwrap_or("[no content]"),
                                300,
                            ),
                            Err(e) => format!("[error: {}]", e),
                        };

                        let routing_correct = decision.action == scenario.expected
                            || (scenario.expected == "respond" && decision.action == "respond")
                            || (scenario.expected == "tool" && decision.action == "tool")
                            || (scenario.expected == "specialist" && decision.action == "specialist");

                        let total_ms = t_start.elapsed().as_millis() as u64;
                        println!(" {} ({:.2}) {}ms", decision.action, decision.confidence, total_ms);

                        results.push(ScenarioResult {
                            idx: scenario.idx,
                            prompt_short,
                            expected: scenario.expected,
                            router_action: Some(decision.action),
                            router_confidence: Some(decision.confidence),
                            specialist_dispatched,
                            routing_correct,
                            router_latency_ms: Some(router_ms),
                            total_latency_ms: total_ms,
                            response_preview,
                            context_summary_found,
                        });
                    }
                    Err(e) => {
                        let total_ms = t_start.elapsed().as_millis() as u64;
                        println!(" [router error] {}ms", total_ms);
                        eprintln!("    Router error: {}", e);

                        results.push(ScenarioResult {
                            idx: scenario.idx,
                            prompt_short,
                            expected: scenario.expected,
                            router_action: Some(format!("error: {}", e)),
                            router_confidence: None,
                            specialist_dispatched: false,
                            routing_correct: false,
                            router_latency_ms: Some(router_ms),
                            total_latency_ms: total_ms,
                            response_preview: String::new(),
                            context_summary_found: false,
                        });
                    }
                }
            } else {
                // Solo mode: direct call to solo provider, no router
                let msgs = build_solo_messages(scenario.prompt);
                let response = solo_provider
                    .chat(&msgs, None, Some(&main_ep.model), scenario.max_tokens, 0.7, None, None)
                    .await;

                let response_preview = match &response {
                    Ok(r) => truncate_to(r.content.as_deref().unwrap_or("[no content]"), 300),
                    Err(e) => format!("[error: {}]", e),
                };

                let total_ms = t_start.elapsed().as_millis() as u64;
                let success = response.is_ok();
                println!(" solo {}ms", total_ms);

                // For solo mode, routing is always "correct" (no routing decision made)
                // We mark it correct if the call succeeded
                results.push(ScenarioResult {
                    idx: scenario.idx,
                    prompt_short,
                    expected: scenario.expected,
                    router_action: None,
                    router_confidence: None,
                    specialist_dispatched: false,
                    routing_correct: success,
                    router_latency_ms: None,
                    total_latency_ms: total_ms,
                    response_preview,
                    context_summary_found: false,
                });
            }
        }

        all_results.push((sweep.label.to_string(), results));
    }

    // Generate report
    let report = generate_markdown_report(&all_results, &date_str);

    // Determine output path
    let out_path = if let Some(ref p) = args.out {
        p.clone()
    } else {
        let home = dirs::home_dir().unwrap_or_else(|| std::path::PathBuf::from("."));
        home.join(".nanobot")
            .join("workspace")
            .join("experiments")
            .join(format!("trio_bench_{}.md", date_str))
            .to_string_lossy()
            .to_string()
    };

    // Expand leading tilde manually if needed
    let out_path = if out_path.starts_with("~/") {
        let home = dirs::home_dir().unwrap_or_else(|| std::path::PathBuf::from("."));
        home.join(&out_path[2..]).to_string_lossy().to_string()
    } else {
        out_path
    };

    let out_path_buf = std::path::Path::new(&out_path);
    if let Some(parent) = out_path_buf.parent() {
        if let Err(e) = fs::create_dir_all(parent) {
            eprintln!("Warning: could not create output directory: {}", e);
        }
    }

    match fs::write(&out_path, &report) {
        Ok(_) => println!("\nReport saved to: {}", out_path),
        Err(e) => {
            eprintln!("Error writing report to {}: {}", out_path, e);
            // Print to stdout as fallback
            println!("\n--- REPORT ---\n{}", report);
        }
    }
}
