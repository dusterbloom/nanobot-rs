//! Evaluation benchmark commands (hanoi, haystack, learning, sprint, report).

use std::sync::Arc;

use crate::config::loader::load_config;
use crate::providers::base::LLMProvider;
use crate::providers::factory;

use super::provider::{check_api_key, create_provider};

/// Build an LLM provider for eval: local LM Studio or cloud API.
pub(crate) fn make_eval_provider(local: bool, port: u16) -> Arc<dyn LLMProvider> {
    if local {
        factory::create_openai_compat(factory::ProviderSpec::local(
            &format!("http://localhost:{}/v1", port),
            Some("local-model"),
        ))
    } else {
        let config = load_config(None);
        check_api_key(&config);
        create_provider(&config)
    }
}

/// Detect the model name for result labelling.
pub(crate) fn eval_model_name(local: bool, port: u16) -> String {
    if local {
        format!("local:{}", port)
    } else {
        let config = load_config(None);
        config.agents.defaults.model.clone()
    }
}

/// Wrap an LLM provider into the `Fn(String) -> Future<Result<String, String>>`
/// closure that runner.rs expects.
pub(super) fn make_llm_caller(
    provider: Arc<dyn LLMProvider>,
) -> impl Fn(String) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<String, String>>>>
{
    move |prompt: String| {
        let p = provider.clone();
        Box::pin(async move {
            let messages = vec![serde_json::json!({"role": "user", "content": prompt})];
            p.chat(&messages, None, None, 512, 0.3, None, None)
                .await
                .map(|r| r.content.unwrap_or_default())
                .map_err(|e| e.to_string())
        })
    }
}

pub(crate) fn cmd_eval_hanoi(
    disks: u8,
    calibrate: bool,
    samples: usize,
    solve: bool,
    catts: bool,
    k: usize,
    local: bool,
    port: u16,
) {
    use crate::agent::eval::hanoi;
    use crate::agent::eval::results;
    use crate::agent::eval::runner;
    use crate::agent::step_voter::estimate_voters_needed;

    if calibrate {
        let model_name = eval_model_name(local, port);
        println!(
            "{} Hanoi Calibration: {} disks, {} samples (model: {})\n",
            crate::LOGO,
            disks,
            samples,
            model_name
        );

        let total_steps = (1usize << disks as usize) - 1;
        println!("  Total optimal steps: {}", total_steps);
        println!("  Sampling up to {} steps for calibration...\n", samples);

        let provider = make_eval_provider(local, port);
        let llm_call = make_llm_caller(provider);

        let cal_config = runner::HanoiCalibrationConfig {
            num_disks: disks,
            num_samples: samples,
            target_reliability: 0.999,
        };

        let runtime = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");
        let mut cal = runtime.block_on(runner::calibrate_hanoi(&cal_config, llm_call));
        cal.model = model_name.clone();

        println!("  Results:");
        println!("  ─────────────────────────");
        println!(
            "    Accuracy:       {:.1}% ({}/{})",
            cal.accuracy * 100.0,
            (cal.accuracy * cal.num_samples as f64).round() as usize,
            cal.num_samples
        );
        println!("    Red Flag Rate:  {:.1}%", cal.red_flag_rate * 100.0);
        println!("    Median Latency: {:.0}ms", cal.median_latency_ms);

        // Voters needed
        match estimate_voters_needed(cal.accuracy, 0.999, 15) {
            Some(v) => println!("    Voters Needed:  {} (for 99.9% reliability)", v),
            None => println!("    Voters Needed:  impossible at this accuracy"),
        }

        // Save result
        let eval_result = runner::calibration_to_eval_result(&cal, disks, 0.999, &model_name);
        let dir = results::default_results_dir();
        match results::save_result(&eval_result, &dir) {
            Ok(path) => println!("\n  Saved to {}", path.display()),
            Err(e) => eprintln!("\n  Failed to save: {}", e),
        }
    } else if solve {
        println!(
            "{} Hanoi Solve: {} disks, k={}{}\n",
            crate::LOGO,
            disks,
            k,
            if catts { " (CATTS enabled)" } else { "" }
        );

        let solution = hanoi::optimal_solution(disks);
        println!("  Optimal solution: {} steps", solution.len());
        println!("  (Full MAKER voting solve not yet implemented.)");
    } else {
        println!("Usage: nanobot eval hanoi --calibrate|--solve [options]");
        println!("  --calibrate --samples N    Measure model accuracy on N sampled steps");
        println!("  --solve --catts -k K       Full solve with MAKER voting");
    }
}

pub(crate) fn cmd_eval_haystack(
    facts: usize,
    length: usize,
    aggregate: bool,
    local: bool,
    port: u16,
) {
    use crate::agent::eval::haystack;
    use crate::agent::eval::results;
    use crate::agent::eval::runner;
    use crate::agent::knowledge_store::KnowledgeStore;

    println!(
        "{} Aggregation Haystack: {} facts, {} chars{}\n",
        crate::LOGO,
        facts,
        length,
        if aggregate {
            " + aggregation"
        } else {
            " (retrieval only)"
        }
    );

    // Generate synthetic data
    let fact_list = haystack::generate_facts(facts, 42);
    let document = haystack::assemble_document(&fact_list, length, 42);

    println!("  Generated {} facts", fact_list.len());
    println!("  Document length: {} chars", document.len());

    // Ingest into a temporary knowledge store
    let tmp = std::env::temp_dir().join(format!("nanobot_eval_haystack_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).ok();
    let db_path = tmp.join("eval_haystack.db");
    let store = KnowledgeStore::open(&db_path).unwrap_or_else(|e| {
        eprintln!("Failed to open knowledge store: {}", e);
        std::process::exit(1);
    });

    let ingest_result = store
        .ingest("eval_haystack", None, &document, 4096, 256)
        .unwrap_or_else(|e| {
            eprintln!("Failed to ingest document: {}", e);
            std::process::exit(1);
        });

    println!("  Ingested: {} chunks\n", ingest_result.chunks_created);

    // Tier 1: Pure FTS5 retrieval benchmark
    let metrics = haystack::evaluate_retrieval(&fact_list, |query| {
        store
            .search(query, 10)
            .unwrap_or_default()
            .into_iter()
            .map(|h| h.snippet)
            .collect()
    });

    println!("  Tier 1: FTS5 Retrieval");
    println!("  ─────────────────────────");
    println!("    Precision: {:.3}", metrics.precision);
    println!("    Recall:    {:.3}", metrics.recall);
    println!("    MRR:       {:.3}", metrics.mrr);
    println!(
        "    Found:     {}/{}\n",
        metrics.facts_found, metrics.facts_total
    );

    // Save tier 1 result
    let model_name = eval_model_name(local, port);
    let retrieval_result = results::EvalResult {
        benchmark: results::BenchmarkType::Haystack,
        started_at: results::now_timestamp(),
        completed_at: results::now_timestamp(),
        model: "fts5".to_string(),
        data: results::BenchmarkData::HaystackRetrieval {
            num_facts: facts,
            total_length: length,
            precision: metrics.precision,
            recall: metrics.recall,
            mrr: metrics.mrr,
        },
        metadata: Default::default(),
    };
    let dir = results::default_results_dir();
    match results::save_result(&retrieval_result, &dir) {
        Ok(path) => println!("  Saved retrieval result to {}\n", path.display()),
        Err(e) => eprintln!("  Failed to save: {}\n", e),
    }

    if aggregate {
        let tasks = haystack::generate_aggregation_tasks(&fact_list);
        println!("  Tier 2: Aggregation Tasks ({})", model_name);
        println!("  ─────────────────────────");
        println!(
            "  Running {} aggregation tasks against LLM...\n",
            tasks.len()
        );

        let provider = make_eval_provider(local, port);
        let llm_call = make_llm_caller(provider);

        let runtime = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");
        let agg_results = runtime.block_on(runner::run_haystack_aggregation(&tasks, llm_call));

        let correct = agg_results.iter().filter(|r| r.correct).count();
        let total = agg_results.len();
        let accuracy = if total > 0 {
            correct as f64 / total as f64
        } else {
            0.0
        };
        let mean_search = agg_results
            .iter()
            .map(|r| r.search_calls as f64)
            .sum::<f64>()
            / total.max(1) as f64;

        for (i, res) in agg_results.iter().enumerate() {
            let status = if res.correct { "PASS" } else { "FAIL" };
            let prompt = haystack::build_aggregation_prompt(&res.task);
            println!(
                "    [{}] Task {}: {}",
                status,
                i + 1,
                prompt.lines().next().unwrap_or("")
            );
        }

        println!(
            "\n    Accuracy: {}/{} ({:.1}%)",
            correct,
            total,
            accuracy * 100.0
        );

        // Save tier 2 result
        let agg_eval_result = results::EvalResult {
            benchmark: results::BenchmarkType::Haystack,
            started_at: results::now_timestamp(),
            completed_at: results::now_timestamp(),
            model: model_name,
            data: results::BenchmarkData::HaystackAggregation {
                num_facts: facts,
                total_length: length,
                tasks_correct: correct,
                tasks_total: total,
                accuracy,
                mean_search_calls: mean_search,
            },
            metadata: Default::default(),
        };
        match results::save_result(&agg_eval_result, &dir) {
            Ok(path) => println!("  Saved aggregation result to {}", path.display()),
            Err(e) => eprintln!("  Failed to save: {}", e),
        }
    }
}

pub(crate) fn cmd_eval_learn(family: String, tasks: usize, depth: usize, local: bool, port: u16) {
    use crate::agent::eval::learning;
    use crate::agent::eval::results;
    use crate::agent::eval::runner;

    let task_family = match family.as_str() {
        "arithmetic" => learning::TaskFamily::ArithmeticChain { depth },
        "fact-retrieval" => learning::TaskFamily::FactRetrieval { num_facts: depth },
        "tool-chain" => learning::TaskFamily::ToolChain { num_steps: depth },
        _ => {
            eprintln!(
                "Unknown task family: '{}'. Use: arithmetic, fact-retrieval, tool-chain",
                family
            );
            std::process::exit(1);
        }
    };

    let model_name = eval_model_name(local, port);
    println!(
        "{} Learning Curve: {} tasks, family={:?} (model: {})\n",
        crate::LOGO,
        tasks,
        task_family,
        model_name
    );

    let curriculum = learning::generate_curriculum(&task_family, tasks, 42);
    println!("  Generated {} tasks", curriculum.len());
    println!("  Running against LLM...\n");

    let provider = make_eval_provider(local, port);
    let llm_call = make_llm_caller(provider);

    let runtime = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");
    let executions = runtime.block_on(runner::run_learning_eval(&curriculum, llm_call));

    // Compute metrics
    let curve = learning::compute_learning_curve(&task_family, &executions, 5.min(tasks));
    let correct = executions.iter().filter(|e| e.success).count();
    let total = executions.len();
    let final_accuracy = if total > 0 {
        correct as f64 / total as f64
    } else {
        0.0
    };

    // Print per-task results (compact)
    for exec in &executions {
        let status = if exec.success { "PASS" } else { "FAIL" };
        let task = &curriculum[exec.task_index];
        println!(
            "    [{}] Task {} (d{}): got \"{}\" expected \"{}\" ({:.0}ms)",
            status,
            exec.task_index,
            task.difficulty,
            exec.agent_answer
                .lines()
                .next()
                .unwrap_or("")
                .chars()
                .take(40)
                .collect::<String>(),
            task.expected_answer,
            exec.duration_ms
        );
    }

    println!("\n  Results:");
    println!("  ─────────────────────────");
    println!(
        "    Accuracy:         {}/{} ({:.1}%)",
        correct,
        total,
        final_accuracy * 100.0
    );
    println!("    Forward Transfer: {:.3}", curve.forward_transfer);
    println!("    Surprise Rate:    {:.1}%", curve.surprise_rate * 100.0);

    // Save result
    let eval_result = results::EvalResult {
        benchmark: results::BenchmarkType::Learning,
        started_at: results::now_timestamp(),
        completed_at: results::now_timestamp(),
        model: model_name,
        data: results::BenchmarkData::Learning {
            family,
            total_tasks: total,
            completed: correct,
            final_accuracy,
            forward_transfer: curve.forward_transfer,
            surprise_rate: curve.surprise_rate,
        },
        metadata: Default::default(),
    };
    let dir = results::default_results_dir();
    match results::save_result(&eval_result, &dir) {
        Ok(path) => println!("\n  Saved to {}", path.display()),
        Err(e) => eprintln!("\n  Failed to save: {}", e),
    }
}

pub(crate) fn cmd_eval_sprint(corpus_size: usize, questions: usize, local: bool, port: u16) {
    use crate::agent::eval::results;
    use crate::agent::eval::runner;
    use crate::agent::eval::sprint;

    let model_name = eval_model_name(local, port);
    let config = sprint::SprintConfig {
        corpus_size,
        num_questions: questions,
        ..Default::default()
    };

    println!(
        "{} Research Sprint: {} chars corpus, {} questions (model: {})\n",
        crate::LOGO,
        corpus_size,
        questions,
        model_name
    );

    let (domains, _document) = sprint::generate_corpus(&config);
    let all_facts = sprint::all_facts(&domains);
    let question_list = sprint::generate_questions(&domains, questions, config.seed);

    println!("  Domains: {}", domains.len());
    println!("  Total facts: {}", all_facts.len());
    println!("  Questions: {}", question_list.len());
    println!("  Running against LLM...\n");

    let provider = make_eval_provider(local, port);
    let llm_call = make_llm_caller(provider);

    let runtime = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");
    let (scorecard, executions) = runtime.block_on(runner::run_sprint(&config, llm_call));

    // Print per-question results
    for exec in &executions {
        let q = &question_list[exec.index];
        let status = if exec.correct { "PASS" } else { "FAIL" };
        println!(
            "    [{}] Q{} [{}]: got \"{}\"",
            status,
            exec.index + 1,
            q.difficulty.label(),
            exec.agent_answer
                .lines()
                .next()
                .unwrap_or("")
                .chars()
                .take(50)
                .collect::<String>()
        );
    }

    println!();
    print!("{}", sprint::format_scorecard(&scorecard));

    // Save result
    let catts_trend: Vec<f64> = executions
        .iter()
        .map(|e| if e.catts_accepted_pilot { 1.0 } else { 0.0 })
        .collect();
    let time_per_q: Vec<f64> = executions.iter().map(|e| e.duration_ms as f64).collect();

    let eval_result = results::EvalResult {
        benchmark: results::BenchmarkType::Sprint,
        started_at: results::now_timestamp(),
        completed_at: results::now_timestamp(),
        model: model_name,
        data: results::BenchmarkData::Sprint {
            corpus_size,
            questions_total: scorecard.questions_total,
            questions_correct: scorecard.questions_correct,
            accuracy: scorecard.accuracy,
            compound_score: scorecard.compound_score,
            catts_savings_trend: catts_trend,
            time_per_question: time_per_q,
        },
        metadata: Default::default(),
    };
    let dir = results::default_results_dir();
    match results::save_result(&eval_result, &dir) {
        Ok(path) => println!("\n  Saved to {}", path.display()),
        Err(e) => eprintln!("\n  Failed to save: {}", e),
    }
}

pub(crate) fn cmd_eval_report() {
    use crate::agent::eval::results;

    let dir = results::default_results_dir();
    println!("{} Evaluation Results\n", crate::LOGO);
    println!("  Results directory: {}\n", dir.display());

    match results::load_all_results(&dir) {
        Ok(results_list) if results_list.is_empty() => {
            println!("  No saved results found.");
            println!("  Run a benchmark first: nanobot eval hanoi --calibrate");
        }
        Ok(results_list) => {
            println!("{}", results::format_summary(&results_list));
        }
        Err(e) => {
            println!("  No results found ({})", e);
        }
    }
}
