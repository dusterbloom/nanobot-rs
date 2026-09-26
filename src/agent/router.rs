// Error-protocol layer-3 backlog (docs/research/2026-08-06-error-conventions-and-host-bridge.md §3.6):
// the deny regime in Cargo.toml is live; this module still carries pre-existing
// violations of the lints below. Remove this allow as the module migrates onto
// the regime.
// Tracking: docs/error-protocol-backlog.md
#![allow(
    clippy::as_conversions,
    clippy::format_push_string,
    clippy::indexing_slicing,
    clippy::shadow_reuse
)]
//! Tool-call routing: canonicalize proxy calls, fill execution defaults and
//! apply the tool guard before any call executes.

use serde_json::Value;
use tracing::warn;

use crate::agent::agent_loop::TurnContext;
use crate::agent::tool_guard::{ToolGuard, ToolGuardDecision};
use crate::agent::tools::registry::ToolRegistry;
use crate::providers::base::ToolCallRequest;

/// Distinguishes a provider/router failure from loss of replay durability.
/// Callers may recover from the former, but must fail closed on the latter.
#[derive(Debug, thiserror::Error)]
pub enum AuxiliaryCallError {
    #[error("{0}")]
    Persistence(String),
    #[error("{0}")]
    Call(String),
}

/// Result of post-tool routing.
pub(crate) enum RouteResult {
    /// Handled entirely (injected message) — continue main loop.
    Continue,
    /// Break with final_content.
    Break(String),
    /// One ordered carrier plus the execution/rejection disposition of each
    /// call. The carrier is persisted before any disposition is acted on.
    Execute(RoutedToolBatch),
}

pub(crate) enum RoutedToolDisposition {
    Execute,
    /// A prior successful call is replayed as a receipt. This is deliberately
    /// distinct from rejection: duplicate protection must not turn known data
    /// into a dead end or consume the turn execution budget.
    Replay {
        receipt: String,
        source_tool_call_id: String,
        result_digest: String,
    },
    Reject {
        reason: String,
        receipt: String,
    },
}

pub(crate) struct RoutedToolCall {
    pub(crate) call: ToolCallRequest,
    pub(crate) disposition: RoutedToolDisposition,
}

pub(crate) struct RoutedToolBatch {
    pub(crate) calls: Vec<RoutedToolCall>,
    /// Applies only after every rejection receipt in this batch is durable.
    pub(crate) return_after_rejections: bool,
    /// User-role recovery instruction appended after the assistant carrier and
    /// its matching rejection receipts, preserving the retained wire prefix.
    pub(crate) post_rejection_instruction: Option<String>,
}

/// Receipt for a blocked duplicate tool call. Pure function — extracted for
/// testability.
///
/// Escalation: the first duplicate replays the cached data (the model asked
/// for this content and already ran the command; blocking with a rejection
/// turns a helpable loop into a dead end). From the second duplicate on, the
/// data is already in context twice — re-dumping it just feeds the loop
/// (observed live: the model re-rolled identical recall calls until the
/// breaker fired). Later duplicates get a directive plus the lease progress
/// signal so the model sees budget state instead of the same bytes.
pub(crate) fn duplicate_receipt(
    name: &str,
    hits: u32,
    cached: Option<&str>,
    source_tool_call_id: Option<&str>,
    cached_chars: usize,
    progress: &str,
) -> String {
    let source = source_tool_call_id
        .map(|id| format!(" source_tool_call_id={}", serde_json::json!(id)))
        .unwrap_or_default();
    // A handle excerpt is not a page: its rendered length is never a source
    // cursor. Recover the original immutable artifact ID (including legacy
    // handles), and always start the first inspection at source offset zero.
    let recovery = cached
        .and_then(|data| data.strip_prefix(crate::agent::tool_engine::TOOL_RESULT_HANDLE_MARKER))
        .and_then(|fields| fields.trim_start().strip_prefix("id:"))
        .and_then(|fields| {
            serde_json::Deserializer::from_str(fields)
                .into_iter::<String>()
                .next()
                .and_then(Result::ok)
        })
        .filter(|id| !id.is_empty())
        .map(|id| {
            let args = serde_json::json!({"tool_call_id": id, "start_char": 0});
            format!(
                "\nRead the original saved output instead of repeating this call:\n\
                 inspect_tool_result({args})\n\
                 Then follow the inspection page's continuation instruction; \
                 do not infer a cursor from the excerpt length."
            )
        })
        .unwrap_or_default();
    match cached {
        Some(data) if hits <= 1 => {
            format!(
                "{} [cached result from earlier identical call{source} — {cached_chars} chars; no new execution occurred]\n{data}{recovery}\n{progress}",
                crate::agent::tool_engine::TOOL_CACHED_REPLAY_MARKER
            )
        }
        _ => format!(
            "{} [duplicate {name} call #{hits} this turn{source} — no new execution occurred. Synthesize \
             what the earlier result establishes and state any unresolved gap. If more evidence \
             is needed, change the query, range, cursor, or target meaningfully.]{recovery}\n{progress}"
            , crate::agent::tool_engine::TOOL_CACHED_REPLAY_MARKER
        ),
    }
}

fn canonicalize_proxy_execution(
    registry: &ToolRegistry,
    mut tc: ToolCallRequest,
) -> ToolCallRequest {
    let Some((name, arguments)) = registry.canonical_proxy_dispatch(&tc.name, &tc.arguments) else {
        return tc;
    };
    tc.name = name;
    tc.arguments = arguments;
    tc
}

/// Fill execution defaults before guard classification so the lookup key,
/// durable decision, and executor all see the same semantic arguments. The
/// command string itself is never rewritten.
fn materialize_execution_defaults(calls: &mut [ToolCallRequest]) {
    let Ok(cwd) = std::env::current_dir() else {
        return;
    };
    let cwd = Value::String(cwd.to_string_lossy().into_owned());
    for call in calls {
        if call.name == "exec" {
            call.arguments
                .entry("working_dir".to_string())
                .or_insert_with(|| cwd.clone());
        }
    }
}

/// Prepare the model's tool calls for execution: canonicalize proxy calls,
/// fill execution defaults and apply the tool guard (same-batch duplicates,
/// cached replays, rejections), then return a control flow signal.
pub(crate) fn route_tool_calls(
    ctx: &mut TurnContext,
    response_content: Option<&str>,
    mut routed_tool_calls: Vec<ToolCallRequest>,
) -> RouteResult {
    routed_tool_calls = routed_tool_calls
        .into_iter()
        .map(|tc| canonicalize_proxy_execution(&ctx.tools, tc))
        .collect();
    materialize_execution_defaults(&mut routed_tool_calls);

    // Preserve every call in the carrier, including same-batch duplicates and
    // guard rejections. A model-visible tool call must always receive exactly
    // one result in its own id slot, even when policy refuses execution.
    let original_count = routed_tool_calls.len();
    if original_count == 0 {
        if let Some(text) = response_content.filter(|s| !s.trim().is_empty()) {
            return RouteResult::Break(text.to_string());
        }
        return RouteResult::Continue;
    }

    let mut seen_in_batch = std::collections::HashSet::new();
    let mut calls = Vec::with_capacity(original_count);
    let mut allowed_count = 0usize;
    let mut blocked_count = 0usize;
    let mut all_blocked_uncached = true;
    for tc in routed_tool_calls {
        let key = crate::agent::tool_runner::normalize_call_key(&tc.name, &tc.arguments);
        let decision = if !seen_in_batch.insert(key) {
            ctx.flow.tool_guard.had_blocked_calls = true;
            ToolGuardDecision::Reject(format!(
                "duplicate tool call blocked for '{}': repeated in one routed batch",
                tc.name
            ))
        } else {
            ctx.flow.tool_guard.decide(&tc.name, &tc.arguments)
        };
        match decision {
            ToolGuardDecision::Replay(cached) => {
                let guard_key = ToolGuard::key(&tc.name, &tc.arguments);
                let receipt = duplicate_receipt(
                    &tc.name,
                    ctx.flow.tool_guard.cache_hits(&guard_key),
                    Some(&cached.result),
                    Some(&cached.source_tool_call_id),
                    cached.result.chars().count(),
                    &ctx.flow.lease.progress_signal(),
                );
                calls.push(RoutedToolCall {
                    call: tc,
                    disposition: RoutedToolDisposition::Replay {
                        receipt,
                        source_tool_call_id: cached.source_tool_call_id,
                        result_digest: cached.result_digest,
                    },
                });
                blocked_count += 1;
                all_blocked_uncached = false;
            }
            ToolGuardDecision::Reject(reason) => {
                warn!("{}", reason);
                blocked_count += 1;
                let guard_key = ToolGuard::key(&tc.name, &tc.arguments);
                let cached = ctx.flow.tool_guard.get_cached_result_entry(&guard_key);
                all_blocked_uncached &= cached.is_none();
                let receipt = if let Some(cached) = cached {
                    duplicate_receipt(
                        &tc.name,
                        ctx.flow.tool_guard.cache_hits(&guard_key),
                        Some(&cached.result),
                        Some(&cached.source_tool_call_id),
                        cached.result.chars().count(),
                        &ctx.flow.lease.progress_signal(),
                    )
                } else {
                    format!(
                        "tool guard rejected {} without execution: {}",
                        tc.name, reason
                    )
                };
                calls.push(RoutedToolCall {
                    call: tc,
                    disposition: RoutedToolDisposition::Reject { reason, receipt },
                });
            }
            ToolGuardDecision::Execute => {
                allowed_count += 1;
                calls.push(RoutedToolCall {
                    call: tc,
                    disposition: RoutedToolDisposition::Execute,
                });
            }
        }
    }

    if allowed_count == 0 {
        // All tool calls were blocked.
        if blocked_count == original_count {
            ctx.flow.consecutive_all_blocked += 1;
            let post_rejection_instruction =
                (all_blocked_uncached && ctx.flow.consecutive_all_blocked == 2).then(|| {
                    "[system] Your last several tool calls were duplicates or blocked. \
                     State what the available evidence establishes, identify unresolved gaps, \
                     and give an honest partial answer. Do not claim that an unexecuted call \
                     succeeded."
                        .to_string()
                });
            // A cached duplicate produces only a compact protocol receipt; it
            // does not execute a tool or add evidence. Count every all-blocked
            // round as zero progress so cached receipts cannot livelock the
            // agent loop while also bypassing its iteration budget.
            ctx.flow.round_executed_no_tools = true;
            return RouteResult::Execute(RoutedToolBatch {
                calls,
                return_after_rejections: true,
                post_rejection_instruction,
            });
        }
    }

    // Reset the consecutive blocked counter when tool calls succeed.
    ctx.flow.consecutive_all_blocked = 0;
    RouteResult::Execute(RoutedToolBatch {
        calls,
        return_after_rejections: false,
        post_rejection_instruction: None,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Arc;

    use crate::agent::agent_core::{build_swappable_core, RuntimeCounters, SwappableCoreConfig};
    use crate::agent::agent_loop::{
        CompactionHandle, FlowControl, MessageLog, ProviderCallMode, ProviderRequestState,
        RetentionEligibility, TurnOutcome,
    };
    use crate::agent::lane::Lane;
    use crate::agent::protocol::CloudProtocol;
    use crate::agent::reasoning::ReasoningEngine;
    use crate::agent::tools::reasoning_tools::SharedEngine;
    use crate::config::schema::{
        AdaptiveTokenConfig, CodeExecutionConfig, CuaToolConfig, MemoryConfig, ProvenanceConfig,
        PythonKernelConfig, ReasoningConfig, ToolDelegationConfig, TrioConfig,
    };
    use crate::providers::base::{FinishReason, LLMProvider, LLMResponse};
    use async_trait::async_trait;
    use serde_json::json;

    struct TestProvider;

    #[async_trait]
    impl LLMProvider for TestProvider {
        async fn chat(
            &self,
            _messages: &[Value],
            _tools: Option<&[Value]>,
            _model: Option<&str>,
            _max_tokens: u32,
            _temperature: f64,
            _thinking_budget: Option<u32>,
            _top_p: Option<f64>,
        ) -> anyhow::Result<LLMResponse> {
            Ok(LLMResponse {
                content: None,
                tool_calls: Vec::new(),
                finish_reason: FinishReason::Stop,
                usage: HashMap::new(),
            })
        }

        fn get_default_model(&self) -> &str {
            "router-test"
        }
    }

    fn test_turn_context() -> TurnContext {
        let workspace = tempfile::tempdir().unwrap().keep();
        let core = Arc::new(build_swappable_core(SwappableCoreConfig {
            provider: Arc::new(TestProvider),
            workspace: workspace.clone(),
            model: "router-test".to_string(),
            max_iterations: 5,
            max_continuations: 2,
            max_tokens: 512,
            temperature: 0.3,
            max_context_tokens: 4096,
            brave_api_key: None,
            search_provider: "searxng".to_string(),
            searxng_url: "http://localhost:8888".to_string(),
            crw_url: String::new(),
            search_max_results: 5,
            exec_timeout: 30,
            restrict_to_workspace: true,
            memory_config: MemoryConfig::default(),
            is_local: false,
            lane: Lane::default(),
            tool_delegation: ToolDelegationConfig::default(),
            provenance: ProvenanceConfig::default(),
            max_tool_result_chars: 2000,
            trio_config: TrioConfig::default(),
            model_capabilities_overrides: HashMap::new(),
            reasoning_config: ReasoningConfig::default(),
            tool_heartbeat_secs: 2,
            health_check_timeout_secs: 2,
            code_execution: CodeExecutionConfig::default(),
            python_kernel: PythonKernelConfig::default(),
            cua: CuaToolConfig::default(),
            adaptive_tokens: AdaptiveTokenConfig::default(),
            sessions_db_path: Some(workspace.join("sessions.db")),
        }));
        let counters = Arc::new(RuntimeCounters::new(4096));
        let reasoning: SharedEngine = Arc::new(parking_lot::Mutex::new(ReasoningEngine::new()));

        TurnContext {
            core,
            request_id: "router-test-request".to_string(),
            session_key: "router-test-session".to_string(),
            session_id: "router-test-session".to_string(),
            turn_count: 1,
            streaming: false,
            audit: None,
            tools: ToolRegistry::new(),
            user_content: "run it".to_string(),
            channel: "test".to_string(),
            chat_id: "test".to_string(),
            is_voice_message: false,
            detected_language: None,
            text_delta_tx: None,
            tool_event_tx: None,
            cancellation_token: None,
            priority_rx: None,
            messages: MessageLog::committed(vec![json!({"role": "user", "content": "run it"})]),
            new_start: 1,
            rendered_messages: Vec::new(),
            protocol: Arc::new(CloudProtocol),
            advertised_tool_names: None,
            used_tools: Default::default(),
            final_content: String::new(),
            turn_outcome: TurnOutcome::LimitExhausted,
            turn_tool_entries: Vec::new(),
            iterations_used: 0,
            compaction: CompactionHandle::new(),
            soft_compaction_requested: false,
            staged_auto_expansion: None,
            content_gate: crate::agent::context_gate::ContentGate::new(4096, 0.25),
            counters,
            capacity: Arc::new(crate::agent::capacity::CapacityRuntime::default()),
            effective_budget: crate::agent::token_budget::TokenBudget::new(4096, 512),
            flow: FlowControl {
                tool_guard: ToolGuard::new(1),
                iterations_since_compaction: 0,
                content_was_streamed: false,
                consecutive_all_blocked: 0,
                consecutive_no_progress_rounds: 0,
                round_executed_no_tools: false,
                lease: crate::agent::lease::Lease::new(
                    crate::agent::lease::DEFAULT_TOOLS_PER_LEASE,
                    crate::agent::lease::DEFAULT_MAX_LEASES_PER_TURN,
                ),
                llm_call_start: None,
                ttft_ms: None,
                provider_prompt_estimate: None,
                retries: crate::agent::agent_loop::RetryState::new(),
                restore_thinking_budget: None,
                provider_request: ProviderRequestState::default(),
                tool_rounds_completed: 0,
                pending_request_metrics: None,
                last_round_keys: Vec::new(),
                prev_round_keys: Vec::new(),
                consecutive_repeat_rounds: 0,
                provider_call_mode: ProviderCallMode::Normal,
                retention: RetentionEligibility::Contracted,
                terminal_attempted: false,
                infra_error: None,
            },
            taint_state: crate::agent::taint::TaintState::new(),
            reasoning,
        }
    }

    #[test]
    fn blocked_calls_do_not_rewrite_routed_prefix() {
        let mut ctx = test_turn_context();
        let arguments = HashMap::from([
            ("command".to_string(), json!("pwd")),
            (
                "working_dir".to_string(),
                json!(std::env::current_dir().unwrap().to_string_lossy()),
            ),
        ]);
        assert!(ctx.flow.tool_guard.allow("exec", &arguments).is_ok());
        let original_messages: Vec<Value> = ctx.messages.iter().cloned().collect();

        for id in ["blocked-1", "blocked-2"] {
            let result = route_tool_calls(
                &mut ctx,
                None,
                vec![ToolCallRequest {
                    id: id.to_string(),
                    name: "exec".to_string(),
                    arguments: arguments.clone(),
                }],
            );
            let RouteResult::Execute(batch) = result else {
                panic!("blocked tool call should return a routed batch")
            };
            if id == "blocked-2" {
                assert!(batch.post_rejection_instruction.is_some_and(|instruction| {
                    instruction.contains("Your last several tool calls were duplicates or blocked")
                        && instruction.contains("give an honest partial answer")
                        && instruction.contains("identify unresolved gaps")
                        && !instruction.contains("already have the data you need")
                }));
            }
        }

        assert_eq!(
            &*ctx.messages,
            original_messages.as_slice(),
            "routing must not insert recovery bytes before the assistant carrier"
        );
    }

    // Duplicate receipts escalate: hit 1 replays the cached bytes, later
    // hits switch to a directive + progress signal.
    #[test]
    fn test_duplicate_receipt_replays_once_then_directs() {
        let progress = "[Tool call 5 of 8 this lease — 2 leases remaining]";

        let first = duplicate_receipt(
            "recall",
            1,
            Some("found: PHASEONE data"),
            Some("source-1"),
            22,
            progress,
        );
        assert!(first.contains("[cached result from earlier identical call"));
        assert!(first.contains("source_tool_call_id=\"source-1\""));
        assert!(first.contains("22 chars; no new execution occurred"));
        assert!(first.contains("found: PHASEONE data"));
        assert!(first.contains(progress));

        let repeat = duplicate_receipt(
            "recall",
            2,
            Some("found: PHASEONE data"),
            Some("source-1"),
            22,
            progress,
        );
        assert!(
            !repeat.contains("found: PHASEONE data"),
            "repeat must not re-dump bytes"
        );
        assert!(repeat.contains("duplicate recall call #2"));
        assert!(repeat.contains("no new execution occurred"));
        assert!(repeat.contains("state any unresolved gap"));
        assert!(!repeat.contains("already have the data you need"));
        assert!(repeat.contains("source_tool_call_id=\"source-1\""));
        assert!(repeat.contains(progress));

        // Defensive arm: classified blocked-with-result but cache vanished.
        let none = duplicate_receipt("recall", 1, None, None, 0, progress);
        assert!(none.contains("duplicate recall call #1"));
    }

    #[test]
    fn duplicate_receipt_recovers_original_handle_from_first_page() {
        // v1 persisted receipts may predate first_read and use chars. ID must
        // survive JSON escaping, and the preview size must never become offset.
        let id = "original_\"雪|call";
        let handle = format!(
            "{} id:{} | tool:\"exec\" | ok:true | chars:14355 | excerpt:\"828 chars\"",
            crate::agent::tool_engine::TOOL_RESULT_HANDLE_MARKER,
            serde_json::to_string(id).unwrap()
        );
        for hit in [1, 2, 4] {
            let receipt = duplicate_receipt(
                "exec",
                hit,
                Some(&handle),
                Some("source-exec"),
                handle.len(),
                "budget",
            );
            let call = receipt.split("inspect_tool_result(").nth(1).unwrap();
            let args: Value = serde_json::from_str(call.split(")\n").next().unwrap()).unwrap();
            assert_eq!(args["tool_call_id"], id);
            assert_eq!(args["start_char"], 0);
        }
    }

    #[test]
    fn duplicate_receipt_does_not_invent_artifact_for_plain_or_bad_output() {
        for data in [
            "plain output",
            "TOOL_RESULT_HANDLE v1 | id:broken",
            "TOOL_RESULT_HANDLE v1 | id:\"\"",
        ] {
            let receipt = duplicate_receipt(
                "exec",
                1,
                Some(data),
                Some("source-exec"),
                data.len(),
                "budget",
            );
            assert!(!receipt.contains("inspect_tool_result("), "{receipt}");
        }
    }

    #[test]
    fn test_canonicalize_proxy_execution_nested_args() {
        let registry = ToolRegistry::new();
        let mut arguments = HashMap::new();
        arguments.insert("name".to_string(), json!("edit_file"));
        arguments.insert(
            "args".to_string(),
            json!({"path": "a.txt", "old_text": "old", "new_text": "new"}),
        );

        let tc = canonicalize_proxy_execution(
            &registry,
            ToolCallRequest {
                id: "tc_proxy_edit".to_string(),
                name: "tool".to_string(),
                arguments,
            },
        );

        assert_eq!(tc.name, "edit_file");
        assert_eq!(tc.arguments.get("path"), Some(&json!("a.txt")));
        assert_eq!(tc.arguments.get("old_text"), Some(&json!("old")));
    }

    #[test]
    fn test_canonicalize_proxy_execution_current_envelope() {
        let registry = ToolRegistry::new();
        let tc = canonicalize_proxy_execution(
            &registry,
            ToolCallRequest {
                id: "tc_proxy_web".to_string(),
                name: "tool".to_string(),
                arguments: HashMap::from([
                    ("tool_name".to_string(), json!("web_fetch")),
                    (
                        "tool_args".to_string(),
                        json!({"url": "https://example.com"}),
                    ),
                ]),
            },
        );

        assert_eq!(tc.name, "web_fetch");
        assert_eq!(tc.arguments.get("url"), Some(&json!("https://example.com")));
    }

    #[test]
    fn test_canonicalize_proxy_execution_flattened_args() {
        let registry = ToolRegistry::with_standard_tools(
            &crate::agent::tools::registry::ToolConfig::new(std::path::Path::new(".")),
        );
        let mut arguments = HashMap::new();
        arguments.insert("name".to_string(), json!("recall"));
        arguments.insert("mode".to_string(), json!("latest"));

        let tc = canonicalize_proxy_execution(
            &registry,
            ToolCallRequest {
                id: "tc_proxy_recall".to_string(),
                name: "tool".to_string(),
                arguments,
            },
        );

        assert_eq!(tc.name, "recall");
        assert_eq!(tc.arguments.get("mode"), Some(&json!("latest")));
    }

    #[test]
    fn test_canonicalize_proxy_execution_preserves_inspect() {
        let registry = ToolRegistry::new();
        let mut arguments = HashMap::new();
        arguments.insert("name".to_string(), json!("session_search"));

        let tc = canonicalize_proxy_execution(
            &registry,
            ToolCallRequest {
                id: "tc_proxy_inspect".to_string(),
                name: "tool".to_string(),
                arguments,
            },
        );

        assert_eq!(tc.name, "tool");
        assert_eq!(tc.arguments.get("name"), Some(&json!("session_search")));
    }

    #[test]
    fn test_canonicalize_proxy_args_as_json_string() {
        let registry = ToolRegistry::new();
        let mut arguments = HashMap::new();
        arguments.insert("name".to_string(), json!("web_search"));
        arguments.insert("args".to_string(), json!(r#"{"query":"news"}"#));

        let tc = canonicalize_proxy_execution(
            &registry,
            ToolCallRequest {
                id: "tc_proxy_web_search".to_string(),
                name: "tool".to_string(),
                arguments,
            },
        );

        assert_eq!(tc.name, "web_search");
        assert_eq!(tc.arguments.get("query"), Some(&json!("news")));
    }

    #[test]
    fn tool_guard_cached_result_matches_reordered_args() {
        let mut first_args = HashMap::new();
        first_args.insert("limit".to_string(), json!(50));
        first_args.insert("query".to_string(), json!("compaction"));
        first_args.insert("path".to_string(), json!("~/Dev/nanobot-rs/src"));

        let mut duplicate_args = HashMap::new();
        duplicate_args.insert("query".to_string(), json!("compaction"));
        duplicate_args.insert("path".to_string(), json!("~/Dev/nanobot-rs/src"));
        duplicate_args.insert("limit".to_string(), json!(50));
        let result = "Searched 4 file(s) under ~/Dev/nanobot-rs/src";

        let mut guard = ToolGuard::new(1);
        guard.record_result("search_files", &first_args, result.to_string());
        let key = ToolGuard::key("search_files", &duplicate_args);
        assert_eq!(
            guard
                .get_cached_result(&key)
                .map(str::chars)
                .map(Iterator::count),
            Some(result.chars().count())
        );
        assert!(guard.allow("search_files", &duplicate_args).is_err());

        let fresh_guard = ToolGuard::new(1);
        assert_eq!(
            fresh_guard.get_cached_result(&key),
            None,
            "ToolGuard lifetime scopes duplicate cache to one turn"
        );
    }
}
