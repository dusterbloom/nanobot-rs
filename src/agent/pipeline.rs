//! Pipeline runner: Rust-level orchestration for multi-step pipelines.
//!
//! The key insight: the pipeline loop is Rust, not LLM. A 3B model doing
//! one step at a time is just as reliable as Opus, because it gets a fresh
//! context every step. The event log (events.jsonl) is below the model layer.
//!
//! Supports MAKER-style first-to-ahead-by-k voting, crash resume from
//! the event log, tool-equipped steps, and context chaining between steps.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tracing::{debug, error, info, warn};

use crate::agent::context::ContextBuilder;
use crate::agent::thread_repair;
use crate::agent::tools::{
    EditFileTool, ExecTool, ListDirTool, ReadFileTool, ToolRegistry, WebFetchTool,
    WebSearchTool, WriteFileTool,
};
use crate::providers::base::LLMProvider;

/// Default max iterations per tool-equipped step.
const DEFAULT_STEP_MAX_ITERATIONS: u32 = 5;

/// A single step in a pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStep {
    /// Step index (0-based).
    pub index: usize,
    /// The prompt to send to the LLM for this step.
    pub prompt: String,
    /// Optional expected answer (for voting verification).
    pub expected: Option<String>,
    /// Optional tool names for this step (None = no tools, text-only).
    #[serde(default)]
    pub tools: Option<Vec<String>>,
    /// Iteration budget per step when tools are enabled (default: 5).
    #[serde(default)]
    pub max_iterations: Option<u32>,
}

/// Result of executing one pipeline step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepResult {
    pub index: usize,
    pub answer: String,
    pub correct: Option<bool>,
    pub voters_used: usize,
    pub duration_ms: u64,
    /// Full output text to chain to the next step.
    #[serde(default)]
    pub context: String,
}

/// Pipeline configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Pipeline ID for resume tracking.
    pub pipeline_id: String,
    /// Steps to execute.
    pub steps: Vec<PipelineStep>,
    /// Ahead-by-k margin for MAKER voting. 0 = no voting (single call).
    pub ahead_by_k: usize,
    /// Maximum voters per step.
    pub max_voters: usize,
    /// Model name (for logging).
    pub model: String,
}

/// Full pipeline execution result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineResult {
    pub pipeline_id: String,
    pub steps_completed: usize,
    pub steps_total: usize,
    pub results: Vec<StepResult>,
    pub total_duration_ms: u64,
}

/// Run a pipeline: execute each step with optional MAKER voting.
///
/// Resumes from the last completed step found in the event log.
/// Each step result is appended to events.jsonl immediately.
/// Steps with `tools` set run a mini agent loop; otherwise plain LLM call.
/// Each step receives accumulated context from all previous steps.
pub async fn run_pipeline(
    config: &PipelineConfig,
    provider: &dyn LLMProvider,
    workspace: &Path,
) -> PipelineResult {
    let start = Instant::now();
    let mut results = Vec::new();

    // Resume: scan event log for completed steps of this pipeline.
    let completed = load_completed_steps(workspace, &config.pipeline_id);
    let resume_from = completed.len();
    if resume_from > 0 {
        info!(
            "Pipeline {}: resuming from step {} (found {} completed)",
            config.pipeline_id, resume_from, resume_from
        );
        results.extend(completed);
    }

    // Build accumulated context from resumed steps.
    let mut context_so_far = String::new();
    for sr in &results {
        if !sr.context.is_empty() {
            context_so_far.push_str(&format!("## Step {} output\n{}\n\n", sr.index, sr.context));
        } else if !sr.answer.is_empty() {
            context_so_far.push_str(&format!("## Step {} output\n{}\n\n", sr.index, sr.answer));
        }
    }

    for step in config.steps.iter().skip(resume_from) {
        debug!(
            "Pipeline {} step {}/{} (tools: {:?})",
            config.pipeline_id,
            step.index + 1,
            config.steps.len(),
            step.tools,
        );

        let step_start = Instant::now();

        let (answer, voters_used) = if step.tools.is_some() {
            // Tool-equipped step: run a mini agent loop.
            let ans = execute_step_with_tools(
                provider,
                &config.model,
                step,
                &context_so_far,
                workspace,
            )
            .await;
            (ans, 1) // No voting for tool-equipped steps.
        } else if config.ahead_by_k > 0 {
            // Plain step with voting.
            let prompt_with_context = build_step_prompt(&step.prompt, &context_so_far);
            vote_on_step(
                provider,
                &config.model,
                &prompt_with_context,
                config.ahead_by_k,
                config.max_voters,
            )
            .await
        } else {
            // Plain single-call step.
            let prompt_with_context = build_step_prompt(&step.prompt, &context_so_far);
            let ans = call_llm(provider, &config.model, &prompt_with_context).await;
            (ans, 1)
        };

        let correct = step.expected.as_ref().map(|exp| {
            exp.trim().to_lowercase() == answer.trim().to_lowercase()
        });

        // The context for chaining is the full answer.
        let step_context = answer.clone();

        let step_result = StepResult {
            index: step.index,
            answer: answer.clone(),
            correct,
            voters_used,
            duration_ms: step_start.elapsed().as_millis() as u64,
            context: step_context.clone(),
        };

        // Persist immediately to event log.
        append_pipeline_event(workspace, &config.pipeline_id, &step_result);

        // Accumulate context for subsequent steps.
        context_so_far.push_str(&format!("## Step {} output\n{}\n\n", step.index, step_context));

        results.push(step_result);
    }

    PipelineResult {
        pipeline_id: config.pipeline_id.clone(),
        steps_completed: results.len(),
        steps_total: config.steps.len(),
        results,
        total_duration_ms: start.elapsed().as_millis() as u64,
    }
}

/// Build a step prompt that includes accumulated context from previous steps.
fn build_step_prompt(prompt: &str, context_so_far: &str) -> String {
    if context_so_far.is_empty() {
        prompt.to_string()
    } else {
        format!(
            "## Previous steps\n{}\n## Your task\n{}",
            context_so_far, prompt
        )
    }
}

/// Execute a tool-equipped pipeline step as a mini agent loop.
///
/// Builds a ToolRegistry with the requested tools, then runs an
/// iterative LLM → tool execution → LLM loop until the model
/// produces a text-only response or hits the iteration limit.
async fn execute_step_with_tools(
    provider: &dyn LLMProvider,
    model: &str,
    step: &PipelineStep,
    context_so_far: &str,
    workspace: &Path,
) -> String {
    let max_iter = step.max_iterations.unwrap_or(DEFAULT_STEP_MAX_ITERATIONS);
    let tool_names = step.tools.as_deref().unwrap_or(&[]);

    // Build a tool registry with only the requested tools.
    let mut tools = ToolRegistry::new();
    let should_include = |name: &str| -> bool {
        tool_names.iter().any(|t| t == name)
    };

    if should_include("read_file") {
        tools.register(Box::new(ReadFileTool));
    }
    if should_include("write_file") {
        tools.register(Box::new(WriteFileTool));
    }
    if should_include("edit_file") {
        tools.register(Box::new(EditFileTool));
    }
    if should_include("list_dir") {
        tools.register(Box::new(ListDirTool));
    }
    if should_include("exec") {
        tools.register(Box::new(ExecTool::new(
            30, // timeout
            Some(workspace.to_string_lossy().to_string()),
            None,
            None,
            false, // restrict_to_workspace
            30000, // max_result_chars
        )));
    }
    if should_include("web_search") {
        tools.register(Box::new(WebSearchTool::new(None, 5)));
    }
    if should_include("web_fetch") {
        tools.register(Box::new(WebFetchTool::new(50_000)));
    }

    let tool_defs = tools.get_definitions();
    let tool_defs_opt: Option<&[Value]> = if tool_defs.is_empty() {
        None
    } else {
        Some(&tool_defs)
    };

    // Build initial messages with context from previous steps.
    let system_prompt = format!(
        "You are a pipeline step executor. Complete the task using the available tools.\n\
         When done, provide your final answer as plain text (no tool calls)."
    );

    let user_prompt = build_step_prompt(&step.prompt, context_so_far);

    let mut messages: Vec<Value> = vec![
        json!({"role": "system", "content": system_prompt}),
        json!({"role": "user", "content": user_prompt}),
    ];

    let mut final_content = String::new();

    // Detect local models for strict alternation repair.
    let is_local = model.starts_with("local:") || model.starts_with("local/");

    for iteration in 0..max_iter {
        debug!(
            "Pipeline step {} tool iteration {}/{}",
            step.index,
            iteration + 1,
            max_iter
        );

        // Local models with Jinja templates require strict user/assistant alternation.
        // Repair tool messages before each LLM call.
        if is_local && iteration > 0 {
            thread_repair::repair_for_strict_alternation(&mut messages);
        }

        let response = match provider
            .chat(&messages, tool_defs_opt, Some(model), 4096, 0.7, None)
            .await
        {
            Ok(r) => r,
            Err(e) => {
                error!("Pipeline step {} LLM call failed: {}", step.index, e);
                return format!("Error: {}", e);
            }
        };

        if response.finish_reason == "error" {
            let err = response.content.as_deref().unwrap_or("Unknown error");
            error!("Pipeline step {} LLM error: {}", step.index, err);
            return format!("Error: {}", err);
        }

        if response.has_tool_calls() {
            // Build assistant message with tool calls.
            let tc_json: Vec<Value> = response
                .tool_calls
                .iter()
                .map(|tc| {
                    json!({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": serde_json::to_string(&tc.arguments)
                                .unwrap_or_else(|_| "{}".to_string()),
                        }
                    })
                })
                .collect();

            ContextBuilder::add_assistant_message(
                &mut messages,
                response.content.as_deref(),
                Some(&tc_json),
            );

            // Execute each tool call.
            for tc in &response.tool_calls {
                debug!("Pipeline step {} calling tool: {}", step.index, tc.name);
                let result = tools.execute(&tc.name, tc.arguments.clone()).await;
                ContextBuilder::add_tool_result(&mut messages, &tc.id, &tc.name, &result.data);
            }
        } else {
            // No tool calls — step is done.
            final_content = response.content.unwrap_or_default();
            break;
        }
    }

    if final_content.is_empty() {
        final_content = "Pipeline step completed but produced no final text.".to_string();
    }

    final_content
}

/// MAKER-style first-to-ahead-by-k voting.
///
/// Calls the LLM up to `max_voters` times. Tallies answers. The first
/// answer to lead by `k` votes wins. If no answer reaches the margin,
/// returns the plurality answer.
async fn vote_on_step(
    provider: &dyn LLMProvider,
    model: &str,
    prompt: &str,
    ahead_by_k: usize,
    max_voters: usize,
) -> (String, usize) {
    use std::collections::HashMap;

    let mut tallies: HashMap<String, usize> = HashMap::new();
    let mut voters_used = 0;

    for _ in 0..max_voters {
        let answer = call_llm(provider, model, prompt).await;
        let normalized = answer.trim().to_lowercase();
        voters_used += 1;

        let count = tallies.entry(normalized.clone()).or_insert(0);
        *count += 1;

        // Check ahead-by-k: does this answer lead the second-place by k?
        let max_count = *count;
        let second_max = tallies
            .values()
            .filter(|&&v| v != max_count)
            .max()
            .copied()
            .unwrap_or(0);

        if max_count >= second_max + ahead_by_k {
            debug!("Vote converged after {} voters: '{}'", voters_used, normalized);
            return (answer, voters_used);
        }
    }

    // No convergence — return plurality.
    let best = tallies
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(answer, _)| answer)
        .unwrap_or_default();
    warn!("Vote did not converge after {} voters, using plurality: '{}'", max_voters, best);
    (best, voters_used)
}

/// Single LLM call for pipeline steps.
async fn call_llm(provider: &dyn LLMProvider, model: &str, prompt: &str) -> String {
    let messages = vec![serde_json::json!({"role": "user", "content": prompt})];
    match provider.chat(&messages, None, Some(model), 512, 0.3, None).await {
        Ok(resp) => resp.content.unwrap_or_default(),
        Err(e) => {
            error!("Pipeline LLM call failed: {}", e);
            format!("Error: {}", e)
        }
    }
}

/// Append a pipeline step result to the event log.
fn append_pipeline_event(workspace: &Path, pipeline_id: &str, result: &StepResult) {
    use std::io::Write;
    let event_path = workspace.join("events.jsonl");
    let event = serde_json::json!({
        "ts": chrono::Utc::now().to_rfc3339(),
        "kind": "pipeline_step",
        "pipeline_id": pipeline_id,
        "step_index": result.index,
        "answer": result.answer,
        "correct": result.correct,
        "voters_used": result.voters_used,
        "duration_ms": result.duration_ms,
    });
    let line = format!("{}\n", event);
    match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&event_path)
    {
        Ok(mut f) => { let _ = f.write_all(line.as_bytes()); }
        Err(e) => warn!("Failed to append pipeline event: {}", e),
    }
}

/// Load completed step results for a pipeline from the event log (for resume).
fn load_completed_steps(workspace: &Path, pipeline_id: &str) -> Vec<StepResult> {
    let event_path = workspace.join("events.jsonl");
    let content = match std::fs::read_to_string(&event_path) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

    let mut results: Vec<StepResult> = Vec::new();
    for line in content.lines() {
        if let Ok(ev) = serde_json::from_str::<serde_json::Value>(line) {
            if ev["kind"] == "pipeline_step"
                && ev["pipeline_id"].as_str() == Some(pipeline_id)
            {
                results.push(StepResult {
                    index: ev["step_index"].as_u64().unwrap_or(0) as usize,
                    answer: ev["answer"].as_str().unwrap_or("").to_string(),
                    correct: ev["correct"].as_bool(),
                    voters_used: ev["voters_used"].as_u64().unwrap_or(1) as usize,
                    duration_ms: ev["duration_ms"].as_u64().unwrap_or(0),
                    context: ev["answer"].as_str().unwrap_or("").to_string(),
                });
            }
        }
    }

    // Sort by step index and deduplicate (keep last result for each step).
    results.sort_by_key(|r| r.index);
    results.dedup_by_key(|r| r.index);
    results
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use async_trait::async_trait;
    use crate::providers::base::LLMResponse;

    struct MockPipelineProvider {
        answers: Vec<String>,
        call_count: std::sync::atomic::AtomicUsize,
    }

    impl MockPipelineProvider {
        fn new(answers: Vec<&str>) -> Self {
            Self {
                answers: answers.into_iter().map(|s| s.to_string()).collect(),
                call_count: std::sync::atomic::AtomicUsize::new(0),
            }
        }
    }

    #[async_trait]
    impl LLMProvider for MockPipelineProvider {
        async fn chat(
            &self,
            _messages: &[serde_json::Value],
            _tools: Option<&[serde_json::Value]>,
            _model: Option<&str>,
            _max_tokens: u32,
            _temperature: f64,
            _thinking_budget: Option<u32>,
        ) -> anyhow::Result<LLMResponse> {
            let idx = self.call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let answer = self.answers.get(idx % self.answers.len())
                .cloned()
                .unwrap_or_else(|| "default".to_string());
            Ok(LLMResponse {
                content: Some(answer),
                tool_calls: vec![],
                finish_reason: "stop".to_string(),
                usage: HashMap::new(),
            })
        }
        fn get_default_model(&self) -> &str { "mock" }
    }

    #[tokio::test]
    async fn test_pipeline_basic() {
        let dir = tempfile::tempdir().unwrap();
        let provider = MockPipelineProvider::new(vec!["42", "hello", "done"]);
        let config = PipelineConfig {
            pipeline_id: "test-1".to_string(),
            steps: vec![
                PipelineStep { index: 0, prompt: "Q1".into(), expected: Some("42".into()), tools: None, max_iterations: None },
                PipelineStep { index: 1, prompt: "Q2".into(), expected: None, tools: None, max_iterations: None },
                PipelineStep { index: 2, prompt: "Q3".into(), expected: Some("done".into()), tools: None, max_iterations: None },
            ],
            ahead_by_k: 0,
            max_voters: 1,
            model: "mock".to_string(),
        };

        let result = run_pipeline(&config, &provider, dir.path()).await;
        assert_eq!(result.steps_completed, 3);
        assert_eq!(result.results[0].correct, Some(true));
        assert_eq!(result.results[1].correct, None);
        assert_eq!(result.results[2].correct, Some(true));

        // Verify events were written.
        let events = std::fs::read_to_string(dir.path().join("events.jsonl")).unwrap();
        assert_eq!(events.lines().count(), 3);
    }

    #[tokio::test]
    async fn test_pipeline_resume() {
        let dir = tempfile::tempdir().unwrap();

        // Pre-write a completed step to event log.
        let event = serde_json::json!({
            "ts": "2026-01-01T00:00:00Z",
            "kind": "pipeline_step",
            "pipeline_id": "resume-test",
            "step_index": 0,
            "answer": "first",
            "correct": true,
            "voters_used": 1,
            "duration_ms": 100,
        });
        std::fs::write(
            dir.path().join("events.jsonl"),
            format!("{}\n", event),
        ).unwrap();

        let provider = MockPipelineProvider::new(vec!["second"]);
        let config = PipelineConfig {
            pipeline_id: "resume-test".to_string(),
            steps: vec![
                PipelineStep { index: 0, prompt: "Q1".into(), expected: None, tools: None, max_iterations: None },
                PipelineStep { index: 1, prompt: "Q2".into(), expected: None, tools: None, max_iterations: None },
            ],
            ahead_by_k: 0,
            max_voters: 1,
            model: "mock".to_string(),
        };

        let result = run_pipeline(&config, &provider, dir.path()).await;
        assert_eq!(result.steps_completed, 2);
        // First step was resumed from log.
        assert_eq!(result.results[0].answer, "first");
        // Second step was actually executed.
        assert_eq!(result.results[1].answer, "second");
    }

    #[tokio::test]
    async fn test_vote_convergence() {
        // Provider returns: "A", "B", "A" → A leads by 1 (ahead_by_k=1 should converge).
        let provider = MockPipelineProvider::new(vec!["A", "B", "A"]);
        let (answer, voters) = vote_on_step(&provider, "mock", "test", 1, 5).await;
        assert_eq!(answer.trim().to_lowercase(), "a");
        assert!(voters <= 3);
    }

    #[tokio::test]
    async fn test_context_chaining() {
        let dir = tempfile::tempdir().unwrap();
        // Each answer builds on context from previous steps.
        let provider = MockPipelineProvider::new(vec!["first_result", "second_result"]);
        let config = PipelineConfig {
            pipeline_id: "chain-test".to_string(),
            steps: vec![
                PipelineStep { index: 0, prompt: "Do step 1".into(), expected: None, tools: None, max_iterations: None },
                PipelineStep { index: 1, prompt: "Do step 2".into(), expected: None, tools: None, max_iterations: None },
            ],
            ahead_by_k: 0,
            max_voters: 1,
            model: "mock".to_string(),
        };

        let result = run_pipeline(&config, &provider, dir.path()).await;
        assert_eq!(result.steps_completed, 2);
        assert_eq!(result.results[0].context, "first_result");
        assert_eq!(result.results[1].context, "second_result");
    }

    #[tokio::test]
    async fn test_build_step_prompt_no_context() {
        let prompt = build_step_prompt("What is 2+2?", "");
        assert_eq!(prompt, "What is 2+2?");
    }

    #[tokio::test]
    async fn test_build_step_prompt_with_context() {
        let prompt = build_step_prompt("What is 2+2?", "## Step 0 output\n4\n\n");
        assert!(prompt.contains("Previous steps"));
        assert!(prompt.contains("Step 0 output"));
        assert!(prompt.contains("What is 2+2?"));
    }
}
