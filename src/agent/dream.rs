//! Dream consolidation (v0.5 E2): a scheduled pass that distills completed
//! working sessions into durable memory and skill proposals.
//!
//! Differs from the reflector on purpose: the reflector REWRITES MEMORY.md
//! from a full-context prompt and consumes (marks) its source rows; the
//! dream APPENDS deduplicated facts (idempotent across repeats, cannot lose
//! concurrent curation) and files non-memory output as human-reviewable
//! proposals in `workspace/DREAM_PROPOSALS.md`. Skills are proposed, never
//! auto-created — dream output lands on disk for review, not in the loop.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use chrono::Local;
use serde_json::json;
use tracing::{info, warn};

use crate::agent::memory::MemoryStore;
use crate::providers::base::LLMProvider;

const DREAM_PROMPT_TEMPLATE: &str = "\
You are consolidating an AI assistant's session history overnight.

Current MEMORY.md (do not repeat these facts):
{current_memory}

Completed session summaries:
{session_summaries}

Identify:
1. facts: durable user preferences, context, or rules worth remembering (max 7, one line each)
2. skill_candidates: repeated procedures worth turning into a skill (max 3, name + one-line why)
3. flags: friction or failures a human should know about (max 3, one line each)

Reply with ONLY a JSON object: {\"facts\": [...], \"skill_candidates\": [...], \"flags\": [...]}";

/// Parsed dream output.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct DreamOutput {
    pub facts: Vec<String>,
    pub skill_candidates: Vec<String>,
    pub flags: Vec<String>,
}

/// Extract the first JSON object from a possibly fenced/wordy reply.
fn extract_json_object(text: &str) -> Option<&str> {
    let start = text.find('{')?;
    // Matching close brace from the end keeps nested objects intact.
    let end = text.rfind('}')?;
    (end > start).then(|| &text[start..=end])
}

pub fn parse_dream_output(raw: &str) -> DreamOutput {
    let Some(slice) = extract_json_object(raw) else {
        return DreamOutput::default();
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(slice) else {
        return DreamOutput::default();
    };
    let strings = |key: &str| -> Vec<String> {
        value
            .get(key)
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .map(str::to_string)
                    .filter(|s| !s.trim().is_empty())
                    .take(7)
                    .collect()
            })
            .unwrap_or_default()
    };
    DreamOutput {
        facts: strings("facts"),
        // Keep proposal lists tighter than facts.
        skill_candidates: {
            let mut s = strings("skill_candidates");
            s.truncate(3);
            s
        },
        flags: {
            let mut s = strings("flags");
            s.truncate(3);
            s
        },
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct DreamReport {
    pub appended_facts: usize,
    pub proposals_written: usize,
}

/// Append proposals to `workspace/DREAM_PROPOSALS.md` as a dated entry.
fn write_proposals(workspace: &Path, output: &DreamOutput) -> usize {
    let n = output.skill_candidates.len() + output.flags.len();
    if n == 0 {
        return 0;
    }
    let path = workspace.join("DREAM_PROPOSALS.md");
    let mut entry = format!(
        "\n## Dream {} (Local)\n\n",
        Local::now().format("%Y-%m-%d %H:%M")
    );
    for candidate in &output.skill_candidates {
        entry.push_str(&format!("- Skill proposal: {candidate}\n"));
    }
    for flag in &output.flags {
        entry.push_str(&format!("- Flag: {flag}\n"));
    }
    entry.push('\n');
    if let Err(error) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .and_then(|mut f| {
            use std::io::Write;
            f.write_all(entry.as_bytes())
        })
    {
        warn!(%error, "Dream: failed to append DREAM_PROPOSALS.md");
        return 0;
    }
    n
}

/// One dream pass. `summaries` mirrors the reflector's completed-session
/// rendering ("**Session: key** (date)\ncontent"). Locking: callers hold the
/// process-level memory transaction lock (same as the reflector) so a dream
/// and a reflection never interleave MEMORY.md writes.
pub async fn run_dream(
    provider: &Arc<dyn LLMProvider>,
    model: &str,
    workspace: &Path,
    summaries: &[String],
) -> Result<DreamReport> {
    if summaries.is_empty() {
        info!("Dream: no completed sessions — skipped");
        return Ok(DreamReport::default());
    }
    let memory_store = MemoryStore::new(workspace);
    let current_memory = memory_store.read_long_term();
    let prompt = DREAM_PROMPT_TEMPLATE
        .replace("{current_memory}", &current_memory)
        .replace("{session_summaries}", &summaries.join("\n\n"));

    let messages = vec![
        json!({"role": "system", "content": "You are a memory consolidation assistant. Reply with only the requested JSON object."}),
        json!({"role": "user", "content": prompt}),
    ];
    let response = provider
        .chat(&messages, None, Some(model), 2048, 0.2, None, None)
        .await
        .context("dream LLM call failed")?;
    let raw = response
        .content
        .ok_or_else(|| anyhow::anyhow!("dream returned no content"))?;

    let output = parse_dream_output(&raw);
    let appended = memory_store.append_long_term_facts(&output.facts);
    let proposals = write_proposals(workspace, &output);
    info!(
        appended,
        proposals,
        "Dream: consolidated {} summaries",
        summaries.len()
    );
    Ok(DreamReport {
        appended_facts: appended,
        proposals_written: proposals,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use serde_json::Value;

    struct DreamMockLLM {
        body: String,
    }

    #[async_trait]
    impl LLMProvider for DreamMockLLM {
        async fn chat(
            &self,
            _messages: &[Value],
            _tools: Option<&[Value]>,
            _model: Option<&str>,
            _max_tokens: u32,
            _temperature: f64,
            _thinking_budget: Option<u32>,
            _top_p: Option<f64>,
        ) -> anyhow::Result<crate::providers::base::LLMResponse> {
            Ok(crate::providers::base::LLMResponse {
                content: Some(self.body.clone()),
                tool_calls: vec![],
                finish_reason: crate::providers::base::FinishReason::Stop,
                usage: std::collections::HashMap::new(),
            })
        }

        fn get_default_model(&self) -> &str {
            "dream-mock"
        }
    }

    #[test]
    fn parses_fenced_and_wordy_json() {
        let out = parse_dream_output(
            "Here is the JSON:\n```json\n{\"facts\":[\"likes tea\",\" \"],\"skill_candidates\":[\"nightly-report\"],\"flags\":[\"rss flaky\"]}\n```",
        );
        assert_eq!(out.facts, vec!["likes tea".to_string()], "blank dropped");
        assert_eq!(out.skill_candidates.len(), 1);
        assert_eq!(out.flags.len(), 1);

        let garbage = parse_dream_output("no json at all");
        assert_eq!(garbage, DreamOutput::default());
    }

    #[test]
    fn append_dedup_is_case_insensitive_and_idempotent() {
        let tmp = tempfile::tempdir().unwrap();
        let store = MemoryStore::new(tmp.path());
        store.write_long_term("- User prefers dark mode\n");
        assert_eq!(
            store.append_long_term_facts(&["USER PREFERS DARK MODE".into()]),
            0
        );
        assert_eq!(store.append_long_term_facts(&["Likes Rust".into()]), 1);
        assert_eq!(
            store.append_long_term_facts(&["Likes Rust".into()]),
            0,
            "idempotent"
        );
        let content = store.read_long_term();
        assert!(content.contains("- User prefers dark mode"));
        assert!(content.contains("- Likes Rust"));
    }

    #[tokio::test]
    async fn dream_appends_facts_and_writes_proposals() {
        let tmp = tempfile::tempdir().unwrap();
        // Seed existing memory to exercise dedup.
        MemoryStore::new(tmp.path()).write_long_term("- Already known\n");

        let provider: Arc<dyn LLMProvider> = Arc::new(DreamMockLLM {
            body: "{\"facts\":[\"already known\",\"fresh fact\"],\"skill_candidates\":[\"morning-brief\"],\"flags\":[\"searxng flaky\"]}".to_string(),
        });
        let report = run_dream(
            &provider,
            "dream-mock",
            tmp.path(),
            &["**Session: x**\nstuff".to_string()],
        )
        .await
        .unwrap();

        assert_eq!(report.appended_facts, 1, "duplicate fact skipped");
        assert_eq!(report.proposals_written, 2);
        let memory = MemoryStore::new(tmp.path()).read_long_term();
        assert!(memory.contains("- fresh fact"));
        let proposals = std::fs::read_to_string(tmp.path().join("DREAM_PROPOSALS.md")).unwrap();
        assert!(proposals.contains("morning-brief"));
        assert!(proposals.contains("searxng flaky"));
    }

    #[tokio::test]
    async fn empty_summaries_is_a_clean_skip() {
        let tmp = tempfile::tempdir().unwrap();
        let provider: Arc<dyn LLMProvider> = Arc::new(DreamMockLLM {
            body: "{}".to_string(),
        });
        let report = run_dream(&provider, "m", tmp.path(), &[]).await.unwrap();
        assert_eq!(report, DreamReport::default());
        assert!(!tmp.path().join("DREAM_PROPOSALS.md").exists());
    }
}
