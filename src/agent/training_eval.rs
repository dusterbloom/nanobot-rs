//! End-to-end training eval: does ANE LoRA training improve oMLX responses?
//!
//! This module contains a single `#[ignore]` integration test that measures
//! model quality before and after training using the exact production pipeline.
//!
//! ```bash
//! # Requires: oMLX running with 35B loaded, ANE available
//! cargo test --features ane,mlx --release --lib -- "eval_training_e2e" --nocapture --ignored --test-threads=1
//! ```

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::path::PathBuf;
    use std::sync::Arc;

    const EVAL_PROMPTS: &[(&str, &str)] = &[
        ("What is the capital of France?", "Paris"),
        ("What planet is closest to the Sun?", "Mercury"),
        ("What element has atomic number 6?", "Carbon"),
        ("Who wrote Romeo and Juliet?", "Shakespeare"),
        ("What is the boiling point of water in Celsius?", "100"),
        ("What is the largest organ in the human body?", "Skin"),
        ("How many sides does a hexagon have?", "6"),
        (
            "What gas do plants absorb from the atmosphere?",
            "carbon dioxide",
        ),
        (
            "What is the speed of light in m/s approximately?",
            "300",
        ),
        (
            "What language is primarily spoken in Brazil?",
            "Portuguese",
        ),
        ("What is the chemical formula for water?", "H2O"),
        ("What year did World War II end?", "1945"),
        ("What is the square root of 144?", "12"),
        ("Who painted the Mona Lisa?", "Vinci"),
        ("What is the smallest prime number?", "2"),
        ("What continent is Egypt in?", "Africa"),
        (
            "What is the powerhouse of the cell?",
            "mitochondri",
        ),
        (
            "How many bones are in the adult human body?",
            "206",
        ),
        ("What force keeps planets in orbit?", "gravity"),
        ("What is the pH of pure water?", "7"),
    ];

    /// Send a prompt to oMLX and check if the response contains the expected answer.
    fn grade_prompt(client: &reqwest::blocking::Client, base_url: &str, prompt: &str, expected: &str) -> (bool, String) {
        let url = format!("{}/v1/chat/completions", base_url);

        // Detect model name from oMLX /v1/models endpoint
        let model_name = client
            .get(&format!("{}/v1/models", base_url))
            .header("Authorization", "Bearer omlx")
            .send()
            .ok()
            .and_then(|r| r.json::<serde_json::Value>().ok())
            .and_then(|j| {
                j["data"]
                    .as_array()
                    .and_then(|a| a.first())
                    .and_then(|m| m["id"].as_str().map(String::from))
            })
            .unwrap_or_else(|| "Qwen3.5-35B-A3B-3bit".to_string());

        let body = serde_json::json!({
            "model": model_name,
            "messages": [
                {"role": "user", "content": format!("Answer in one word or number only. No explanation. {}", prompt)}
            ],
            "max_tokens": 30,
            "temperature": 0.0,
        });

        let resp = match client.post(&url)
            .header("Authorization", "Bearer omlx")
            .json(&body)
            .send() {
            Ok(r) => r,
            Err(e) => return (false, format!("HTTP error: {e}")),
        };

        let json: serde_json::Value = match resp.json() {
            Ok(j) => j,
            Err(e) => return (false, format!("JSON parse error: {e}")),
        };

        let text = json["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();

        let hit = text.to_lowercase().contains(&expected.to_lowercase());
        (hit, text.chars().take(60).collect())
    }

    /// Run all 20 eval prompts and return (correct_count, total, details).
    fn run_eval(client: &reqwest::blocking::Client, base_url: &str) -> (usize, usize, Vec<(String, String, bool)>) {
        let mut correct = 0;
        let mut details = Vec::new();

        for &(prompt, expected) in EVAL_PROMPTS {
            let (hit, response) = grade_prompt(client, base_url, prompt, expected);
            if hit {
                correct += 1;
            }
            details.push((prompt.to_string(), response, hit));
        }

        (correct, EVAL_PROMPTS.len(), details)
    }

    /// Training prompts that generate diverse experiences.
    const TRAINING_PROMPTS: &[&str] = &[
        "Explain quantum entanglement in simple terms",
        "Write a Python function to find prime numbers using a sieve",
        "What are the main differences between TCP and UDP protocols?",
        "Describe how photosynthesis works at the molecular level",
        "Explain the concept of entropy in thermodynamics",
        "Write a Rust function that implements binary search",
        "What causes the seasons on Earth?",
        "Explain how neural networks learn through backpropagation",
        "Describe the structure of DNA and its role in genetics",
        "What is the difference between stack and heap memory?",
        "Explain the Doppler effect with real-world examples",
        "Write a SQL query to find duplicate records in a table",
        "How does the immune system fight viral infections?",
        "Explain the concept of recursion with a practical example",
        "What are the fundamental forces in physics?",
        "Describe how a compiler transforms source code into machine code",
        "Explain the water cycle and its importance for life",
        "Write a function to detect cycles in a linked list",
        "What is the significance of the Turing test?",
        "Explain how public key cryptography works",
        "Describe the process of protein folding",
        "What are design patterns in software engineering?",
        "Explain the concept of natural selection in evolution",
        "Write a regular expression to validate email addresses",
        "How do vaccines work at the cellular level?",
        "Explain the difference between supervised and unsupervised learning",
        "What is the greenhouse effect and its impact on climate?",
        "Describe the architecture of a modern CPU",
        "Explain how CRISPR gene editing works",
        "What is the halting problem and why is it important?",
        "Describe the layers of the OSI networking model",
        "Explain how batteries store and release energy",
        "What is the significance of Euler's identity in mathematics?",
        "Write a function to find the longest common subsequence",
        "How does GPS determine your location?",
        "Explain the concept of database normalization",
        "What is dark matter and why do scientists believe it exists?",
        "Describe how garbage collection works in programming languages",
        "Explain the theory of plate tectonics",
        "What is the difference between REST and GraphQL APIs?",
        "How do magnets work at the atomic level?",
        "Explain the concept of Big O notation",
        "What is the Standard Model of particle physics?",
        "Describe how blockchain technology achieves consensus",
        "Explain the principles of special relativity",
        "What are monads in functional programming?",
        "How does the human brain process language?",
        "Explain the concept of information entropy",
        "What is quantum computing and how does it differ from classical?",
        "Describe the process of nuclear fusion in stars",
    ];

    /// Populate the experience buffer with training data.
    fn populate_experiences(db_path: &std::path::Path, prompts: &[&str]) -> usize {
        let eb = super::super::lora_bridge::ExperienceBuffer::open(db_path)
            .expect("failed to open experience buffer");

        let tool_entries = vec![super::super::audit::TurnToolEntry {
            name: "explain".into(),
            id: "call_1".into(),
            ok: true,
            duration_ms: 200,
            result_chars: 500,
        }];

        let quality = super::super::lora_bridge::compute_quality(&tool_entries, 2, 10, true, 300);

        for prompt in prompts {
            let trace = serde_json::json!([{
                "name": "explain",
                "arguments": {"topic": prompt},
                "result": format!("Detailed explanation of: {}", prompt)
            }])
            .to_string();

            let response = format!(
                "Here is a detailed explanation of the topic: {}. \
                 This involves several key concepts that are fundamental to understanding the subject.",
                prompt
            );

            eb.record_with_surprise(
                prompt,
                &trace,
                &response,
                true,
                quality,
                "eval-model",
                0.5, // above default threshold
            )
            .expect("failed to record experience");
        }

        prompts.len()
    }

    /// End-to-end eval: does ANE LoRA training improve oMLX responses?
    ///
    /// Requires: oMLX running with a model loaded, ANE available.
    ///
    /// ```bash
    /// cargo test --features ane,mlx --release --lib -- "eval_training_e2e" --nocapture --ignored --test-threads=1
    /// ```
    #[test]
    #[ignore]
    fn eval_training_e2e() {
        // ── Setup ──
        let base_url = std::env::var("NANOBOT_EVAL_URL")
            .unwrap_or_else(|_| "http://127.0.0.1:8080/v1".to_string());

        // Strip trailing /v1 if present — grade_prompt adds it
        let base_url = base_url.trim_end_matches("/v1").to_string();

        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .expect("failed to build HTTP client");

        // Use a temporary experience DB so we don't pollute production data
        let eval_dir = tempfile::tempdir().expect("failed to create temp dir");
        let db_path = eval_dir.path().join("eval_experience.db");

        eprintln!("\n{}", "=".repeat(60));
        eprintln!("  TRAINING EVAL — Does LoRA Training Improve the Model?");
        eprintln!("  oMLX: {base_url}");
        eprintln!("{}", "=".repeat(60));

        // ── 1. Baseline ──
        eprintln!("\n── Baseline (before training) ──");
        let (pre_correct, pre_total, pre_details) = run_eval(&client, &base_url);
        eprintln!("  Score: {pre_correct}/{pre_total} ({:.0}%)", 100.0 * pre_correct as f64 / pre_total as f64);
        for (prompt, response, hit) in &pre_details {
            let mark = if *hit { "+" } else { "-" };
            eprintln!("  [{mark}] {:<45} → {response}", &prompt[..prompt.len().min(45)]);
        }

        // ── 2. Populate experiences ──
        eprintln!("\n── Populating experience buffer ──");
        let n_experiences = populate_experiences(&db_path, TRAINING_PROMPTS);
        eprintln!("  Recorded {n_experiences} experiences");

        // ── 3. Train ──
        eprintln!("\n── Training ──");

        // Find model directory
        let home = std::env::var("HOME").unwrap();
        // Train the 35B directly. ANE compiles FFN kernels (shared expert fits
        // in 32 MB SRAM). Attention kernels may fail (head_dim=256 too large) —
        // strict_ane=false falls back to CPU for those, training still works.
        let model_dirs = [
            format!("{home}/.cache/lm-studio/models/NexVeridian/Qwen3.5-35B-A3B-3bit"),
            format!("{home}/.cache/lm-studio/models/mlx-community/Qwen3.5-35B-A3B-4bit"),
            format!("{home}/.cache/lm-studio/models/mlx-community/Qwen3.5-0.8B-8bit"),
        ];

        let model_dir = model_dirs
            .iter()
            .find(|d| std::path::Path::new(d).exists())
            .expect("No model found for training. Need Qwen3.5 in ~/.cache/lm-studio/models/");

        eprintln!("  Model: {model_dir}");

        #[cfg(all(feature = "ane", feature = "mlx"))]
        {
            use super::super::ane_mlx_bridge::{AneTrainingConfig, AneTrainingOptimizer, PersistentAneTrainer};
            use super::super::mlx_lora::ModelConfig;

            let model_path = std::path::PathBuf::from(model_dir);
            let model_cfg = ModelConfig::from_config_json(&model_path)
                .expect("failed to parse model config");

            // Build MilConfig from ModelConfig
            let mil_cfg = super::super::ane_mil::MilConfig {
                dim: model_cfg.dim,
                hidden_dim: model_cfg.hidden_dim,
                n_heads: model_cfg.n_heads,
                seq_len: 1,
                n_kv_heads: model_cfg.n_kv_heads,
                rope_theta: model_cfg.rope_theta as f64,
                rms_eps: model_cfg.rms_eps,
                has_lm_head: false,
                head_dim_explicit: model_cfg.head_dim,
                linear_attn_indices: model_cfg.linear_attn_indices.clone(),
                linear_n_heads: model_cfg.linear_n_heads,
                linear_head_dim: model_cfg.linear_head_dim,
                linear_n_value_heads: model_cfg.linear_n_value_heads,
                linear_value_head_dim: model_cfg.linear_value_head_dim,
                conv_kernel_size: model_cfg.conv_kernel_size,
                attn_output_gate: model_cfg.attn_output_gate,
            };

            let training_cfg = AneTrainingConfig {
                model_dir: model_path.clone(),
                training_model_dir: None,
                mil_config: mil_cfg,
                epochs: 3,
                lr: 1e-4,
                linear_attn_indices: model_cfg.linear_attn_indices.clone(),
                kv_dim: model_cfg.n_kv_heads * model_cfg.head_dim,
                softcap: 15.0,
                loss_scale: 256.0,
                lr_scale_attn: 0.05,
                lr_scale_ffn: 1.0,
                residual_scale: 1.0,
                optimizer: AneTrainingOptimizer::AdamW,
                strict_ane: false,
                accum_steps: 1,
                adaptive_layer_drop: false,
                dense_cache_budget_bytes: None,
            };

            // Load experiences from our temp DB
            let eb = super::super::lora_bridge::ExperienceBuffer::open(&db_path)
                .expect("failed to open experience buffer");
            let exps = eb.all_for_training(50).expect("failed to fetch experiences");
            eprintln!("  Fetched {} experiences for training", exps.len());

            if exps.is_empty() {
                eprintln!("  SKIP: No experiences to train on");
                return;
            }

            // Tokenize
            let tokenizer_path = model_path.join("tokenizer.json");
            let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
                .expect("failed to load tokenizer");

            let samples: Vec<(Vec<i32>, Vec<i32>, f32)> = exps
                .iter()
                .filter_map(|exp| {
                    let text = format!("{}\n{}", exp.prompt, exp.response);
                    let encoded = tokenizer.encode(text, false).ok()?;
                    let ids: Vec<i32> = encoded.get_ids().iter().map(|&id| id as i32).collect();
                    if ids.len() < 4 {
                        return None;
                    }
                    let input = ids[..ids.len() - 1].to_vec();
                    let target = ids[1..].to_vec();
                    Some((input, target, exp.quality as f32))
                })
                .collect();

            eprintln!("  Tokenized {} samples", samples.len());

            if samples.is_empty() {
                eprintln!("  SKIP: No tokenizable samples");
                return;
            }

            // Create trainer and run
            let trainer = PersistentAneTrainer::default();
            let t0 = std::time::Instant::now();

            let handle = trainer.spawn_training_with_progress(
                training_cfg,
                samples,
                None, // no mlx_tx — standalone
                None, // no runtime counters
                None, // no draft reload
            );

            let success = handle.join().unwrap_or(false);
            let train_secs = t0.elapsed().as_secs_f64();

            if success {
                eprintln!("  Training completed in {train_secs:.0}s");

                // Trigger oMLX reload
                eprintln!("  Reloading oMLX...");
                super::super::learn_loop::omlx_try_reload_from_config();
                std::thread::sleep(std::time::Duration::from_secs(5)); // give oMLX time to reload
            } else {
                eprintln!("  Training FAILED after {train_secs:.0}s");
                eprintln!("  Running post-eval anyway to check for regression...");
            }
        }

        #[cfg(not(all(feature = "ane", feature = "mlx")))]
        {
            eprintln!("  SKIP: requires --features ane,mlx");
            return;
        }

        // ── 4. Post-training eval ──
        eprintln!("\n── Post-training ──");
        let (post_correct, post_total, post_details) = run_eval(&client, &base_url);
        eprintln!("  Score: {post_correct}/{post_total} ({:.0}%)", 100.0 * post_correct as f64 / post_total as f64);
        for (prompt, response, hit) in &post_details {
            let mark = if *hit { "+" } else { "-" };
            eprintln!("  [{mark}] {:<45} → {response}", &prompt[..prompt.len().min(45)]);
        }

        // ── 5. Comparison ──
        let pre_pct = 100.0 * pre_correct as f64 / pre_total as f64;
        let post_pct = 100.0 * post_correct as f64 / post_total as f64;
        let delta = post_pct - pre_pct;

        eprintln!("\n{}", "=".repeat(60));
        eprintln!("  RESULT");
        eprintln!("{}", "=".repeat(60));
        eprintln!("  Before: {pre_correct}/{pre_total} ({pre_pct:.0}%)");
        eprintln!("  After:  {post_correct}/{post_total} ({post_pct:.0}%)");
        eprintln!("  Delta:  {delta:+.0}%");
        eprintln!();

        if delta > 0.0 {
            eprintln!("  VERDICT: Training IMPROVED the model (+{delta:.0}%)");
        } else if delta == 0.0 {
            eprintln!("  VERDICT: Training had NO EFFECT (0%)");
        } else {
            eprintln!("  VERDICT: Training DEGRADED the model ({delta:.0}%)");
        }

        // Don't assert — we want to see the data either way
        // The test "passes" as long as it runs without crashing
    }
}
