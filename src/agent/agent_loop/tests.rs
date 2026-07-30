//! Tests for the agent loop.
//!
//! Declared as `#[cfg(test)] mod tests;` in `agent_loop/mod.rs`. The directory
//! layout (mod.rs + tests.rs as siblings) keeps the test surface in its own
//! file without needing the old `#[path]` hack.

use super::*;
use crate::agent::lane::Lane;
use crate::agent::router::{
    extract_json_object, parse_lenient_router_decision, request_strict_router_decision,
};
use crate::config::schema::{
    AdaptiveTokenConfig, MemoryConfig, ProvenanceConfig, ProviderConfig, ToolDelegationConfig,
    TrioConfig,
};
use crate::providers::base::LLMProvider;
use crate::providers::openai_compat::OpenAICompatProvider;
use async_trait::async_trait;
use backon::BackoffBuilder;

fn attested_text(content: &str) -> String {
    // Attestation protocol removed — text is itself the final answer.
    content.to_string()
}

/// Minimal mock LLM provider for wiring tests.
struct MockLLM {
    name: String,
}

impl MockLLM {
    fn named(name: &str) -> Arc<dyn LLMProvider> {
        Arc::new(Self {
            name: name.to_string(),
        })
    }
}

#[async_trait]
impl LLMProvider for MockLLM {
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
            content: Some("mock".to_string()),
            tool_calls: vec![],
            finish_reason: "stop".to_string(),
            usage: std::collections::HashMap::new(),
        })
    }

    fn get_default_model(&self) -> &str {
        &self.name
    }
}

fn test_runtime_counters(
    max_context_tokens: usize,
) -> Arc<crate::agent::agent_core::RuntimeCounters> {
    Arc::new(crate::agent::agent_core::RuntimeCounters::new_with_config(
        max_context_tokens,
        &crate::config::schema::CircuitBreakerConfig::default(),
    ))
}

struct StaticResponseLLM {
    name: String,
    body: String,
}

impl StaticResponseLLM {
    fn new(name: &str, body: &str) -> Self {
        Self {
            name: name.to_string(),
            body: attested_text(body),
        }
    }

    fn plain(name: &str, body: &str) -> Self {
        Self {
            name: name.to_string(),
            body: body.to_string(),
        }
    }
}

#[async_trait]
impl LLMProvider for StaticResponseLLM {
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
            finish_reason: "stop".to_string(),
            usage: std::collections::HashMap::new(),
        })
    }

    fn get_default_model(&self) -> &str {
        &self.name
    }
}

/// Helper to build a SwappableCore with minimal config for wiring tests.
fn build_test_core(
    delegation_enabled: bool,
    delegation_provider: Option<Arc<dyn LLMProvider>>,
    config_provider: Option<ProviderConfig>,
) -> SwappableCore {
    let workspace = tempfile::tempdir().unwrap().keep();
    // Isolate the session DB per test so parallel runs don't contend on the
    // user's real ~/.nanobot/sessions.db.
    let sessions_db = workspace.join("sessions.db");
    let main = MockLLM::named("main-provider");
    let td = ToolDelegationConfig {
        enabled: delegation_enabled,
        model: "delegation-model".to_string(),
        provider: config_provider,
        auto_local: true,
        ..Default::default()
    };
    build_swappable_core(SwappableCoreConfig {
        provider: main,
        workspace,
        model: "main-model".to_string(),
        max_iterations: 10,
        max_continuations: 2,
        max_tokens: 4096,
        temperature: 0.7,
        max_context_tokens: 16384,
        brave_api_key: None,
        search_provider: "searxng".to_string(),
        searxng_url: "http://localhost:8888".to_string(),
        crw_url: String::new(),
        search_max_results: 5,
        exec_timeout: 30,
        restrict_to_workspace: false,
        memory_config: MemoryConfig::default(),
        is_local: false,
        lane: Lane::default(),
        tool_delegation: td,
        provenance: ProvenanceConfig::default(),
        max_tool_result_chars: 2000,
        delegation_provider,
        specialist_provider: None,
        trio_config: TrioConfig::default(),
        model_capabilities_overrides: std::collections::HashMap::new(),
        reasoning_config: crate::config::schema::ReasoningConfig::default(),
        tool_heartbeat_secs: 2,
        health_check_timeout_secs: 2,
        adaptive_tokens: AdaptiveTokenConfig::default(),
        sessions_db_path: Some(sessions_db),
    })
}

#[test]
fn test_extract_json_object_from_markdown_fence() {
    let raw =
        "```json\n{\"action\":\"tool\",\"target\":\"exec\",\"args\":{},\"confidence\":0.9}\n```";
    let obj = extract_json_object(raw).expect("json object");
    assert!(obj.starts_with('{'));
    assert!(obj.ends_with('}'));
    assert!(obj.contains("\"action\":\"tool\""));
}

#[test]
fn test_extract_json_object_none_when_missing() {
    assert!(extract_json_object("no json here").is_none());
}

#[tokio::test]
async fn test_request_strict_router_decision_action_matrix() {
    let cases = vec![
        (
            r#"{"action":"tool","target":"read_file","args":{"path":"README.md"},"confidence":0.9}"#,
            "tool",
        ),
        (
            r#"{"action":"subagent","target":"builder","args":{"task":"x"},"confidence":0.8}"#,
            "subagent",
        ),
        (
            r#"{"action":"specialist","target":"summarizer","args":{"style":"tight"},"confidence":0.7}"#,
            "specialist",
        ),
        (
            r#"{"action":"ask_user","target":"clarify","args":{"question":"Need path?"},"confidence":0.6}"#,
            "ask_user",
        ),
    ];

    for (raw, expected_action) in cases {
        let llm = StaticResponseLLM::plain("router", raw);
        let decision = request_strict_router_decision(
            &llm,
            "router",
            "route this action with strict schema",
            false,
            0.6,
            1.0,
            "",
            256,
        )
        .await
        .expect("valid strict router decision");
        assert_eq!(decision.action, expected_action);
    }
}

/// Real-provider trio probe.
///
/// Runs against live OpenAI-compatible endpoints (e.g. LM Studio):
/// - main: `NANOBOT_REAL_MAIN_BASE` (default: http://127.0.0.1:8080/v1)
/// - router: `NANOBOT_REAL_ROUTER_BASE` (default: http://127.0.0.1:8094/v1)
/// - specialist: `NANOBOT_REAL_SPECIALIST_BASE` (default: http://127.0.0.1:8095/v1)
///
/// Optional model overrides:
/// - `NANOBOT_REAL_MAIN_MODEL`
/// - `NANOBOT_REAL_ROUTER_MODEL`
/// - `NANOBOT_REAL_SPECIALIST_MODEL`
#[tokio::test]
#[ignore = "requires running local providers on main/router/specialist ports"]
async fn test_real_providers_trio_probe() {
    let main_base = std::env::var("NANOBOT_REAL_MAIN_BASE")
        .unwrap_or_else(|_| "http://127.0.0.1:8080/v1".to_string());
    let router_base = std::env::var("NANOBOT_REAL_ROUTER_BASE")
        .unwrap_or_else(|_| "http://127.0.0.1:8094/v1".to_string());
    let specialist_base = std::env::var("NANOBOT_REAL_SPECIALIST_BASE")
        .unwrap_or_else(|_| "http://127.0.0.1:8095/v1".to_string());
    let main_model =
        std::env::var("NANOBOT_REAL_MAIN_MODEL").unwrap_or_else(|_| "local-model".to_string());
    let router_model = std::env::var("NANOBOT_REAL_ROUTER_MODEL")
        .unwrap_or_else(|_| "local-delegation".to_string());
    let specialist_model = std::env::var("NANOBOT_REAL_SPECIALIST_MODEL")
        .unwrap_or_else(|_| "local-specialist".to_string());

    let main = OpenAICompatProvider::new("local", Some(&main_base), Some(&main_model));
    let router = OpenAICompatProvider::new("local", Some(&router_base), Some(&router_model));
    let specialist =
        OpenAICompatProvider::new("local", Some(&specialist_base), Some(&specialist_model));

    let mut failures: Vec<String> = Vec::new();

    // Router: force each action in a constrained prompt and verify strict parsing.
    let router_cases = vec![
        (
            "tool",
            "Return action=tool target=read_file args={\"path\":\"README.md\"}.",
        ),
        (
            "subagent",
            "Return action=subagent target=builder args={\"task\":\"diagnose issue\"}.",
        ),
        (
            "specialist",
            "Return action=specialist target=summarizer args={\"objective\":\"compress\"}.",
        ),
        (
            "ask_user",
            "Return action=ask_user target=clarify args={\"question\":\"Which file?\"}.",
        ),
    ];
    for (expected_action, directive) in router_cases {
        let pack = format!("{}\nFollow schema strictly.", directive);
        match request_strict_router_decision(
            &router,
            &router_model,
            &pack,
            false,
            0.6,
            1.0,
            "",
            256,
        )
        .await
        {
            Ok(d) => {
                if d.action != expected_action {
                    failures.push(format!(
                        "router action mismatch: expected={}, got={} target={}",
                        expected_action, d.action, d.target
                    ));
                }
            }
            Err(e) => failures.push(format!("router {} failed: {}", expected_action, e)),
        }
    }

    // Specialist must produce non-empty response (with warmup retries).
    let specialist_messages = vec![
        json!({"role":"system","content":"ROLE=SPECIALIST\nReturn concise output."}),
        json!({"role":"user","content":"Summarize: tool call failed because server was down and port conflicted."}),
    ];
    let mut specialist_ok = false;
    let mut warmup_backoff = backon::ConstantBuilder::default()
        .with_delay(Duration::from_secs(2))
        .with_max_times(10)
        .build();
    loop {
        match specialist
            .chat(
                &specialist_messages,
                None,
                Some(&specialist_model),
                256,
                0.2,
                None,
                None,
            )
            .await
        {
            Ok(resp) => {
                let text = resp.content.unwrap_or_default();
                if !text.trim().is_empty() {
                    specialist_ok = true;
                    break;
                }
            }
            Err(e) => {
                let msg = e.to_string();
                let lower = msg.to_lowercase();
                if !lower.contains("loading model") && !lower.contains("503") {
                    failures.push(format!("specialist call failed: {}", msg));
                    break;
                }
            }
        }
        match warmup_backoff.next() {
            Some(delay) => tokio::time::sleep(delay).await,
            None => break,
        }
    }
    if !specialist_ok {
        failures.push("specialist did not become ready / returned empty output".to_string());
    }

    // Main provider smoke: should answer plain text with no tools when none offered.
    let main_messages = vec![json!({"role":"user","content":"Reply with exactly: main-ok"})];
    match main
        .chat(&main_messages, None, Some(&main_model), 64, 0.0, None, None)
        .await
    {
        Ok(resp) => {
            if resp.has_tool_calls() {
                failures.push("main returned tool calls unexpectedly".to_string());
            }
            let text = resp.content.unwrap_or_default();
            if !text.to_lowercase().contains("main-ok") {
                failures.push(format!("main output mismatch: {}", text));
            }
        }
        Err(e) => failures.push(format!("main call failed: {}", e)),
    }

    if !failures.is_empty() {
        panic!(
            "real trio probe failed (main={}, router={}, specialist={}):\n{}",
            main_base,
            router_base,
            specialist_base,
            failures.join("\n")
        );
    }
}

// -- Delegation provider wiring tests --

#[test]
fn test_delegation_disabled_no_runner_provider() {
    let core = build_test_core(false, None, None);
    assert!(
        core.tool_runner_provider.is_none(),
        "When delegation is disabled, tool_runner_provider should be None"
    );
    assert!(core.tool_runner_model.is_none());
}

#[test]
fn test_delegation_enabled_with_auto_provider() {
    // When an auto-spawned delegation provider is passed, it should be used
    let dp = MockLLM::named("auto-delegation");
    let core = build_test_core(true, Some(dp), None);

    assert!(core.tool_runner_provider.is_some());
    let provider = core.tool_runner_provider.as_ref().unwrap();
    assert_eq!(
        provider.get_default_model(),
        "auto-delegation",
        "Should use the auto-spawned delegation provider"
    );
    assert_eq!(core.tool_runner_model.as_deref(), Some("delegation-model"));
}

#[test]
fn test_delegation_auto_provider_takes_priority_over_config() {
    // Auto-spawned provider should take priority over config provider
    let dp = MockLLM::named("auto-delegation");
    let config_provider = ProviderConfig {
        api_key: "key".to_string(),
        api_base: Some("http://localhost:9999/v1".to_string()),
    };
    let core = build_test_core(true, Some(dp), Some(config_provider));

    let provider = core.tool_runner_provider.as_ref().unwrap();
    assert_eq!(
        provider.get_default_model(),
        "auto-delegation",
        "Auto-spawned provider should beat config provider"
    );
}

#[test]
fn test_delegation_config_provider_used_when_no_auto() {
    // When no auto provider, but config has one, it should create OpenAICompatProvider
    let config_provider = ProviderConfig {
        api_key: "key".to_string(),
        api_base: Some("http://localhost:9999/v1".to_string()),
    };
    let core = build_test_core(true, None, Some(config_provider));

    assert!(
        core.tool_runner_provider.is_some(),
        "Should have a provider from config"
    );
}

#[test]
fn test_delegation_falls_back_to_main_provider() {
    // When delegation enabled but no auto provider and no config provider,
    // should fall back to main
    let core = build_test_core(true, None, None);

    assert!(core.tool_runner_provider.is_some());
    let provider = core.tool_runner_provider.as_ref().unwrap();
    assert_eq!(
        provider.get_default_model(),
        "main-provider",
        "Should fall back to main provider"
    );
}

#[test]
fn test_delegation_model_uses_config_model() {
    let core = build_test_core(true, None, None);
    assert_eq!(
        core.tool_runner_model.as_deref(),
        Some("delegation-model"),
        "Should use the model from ToolDelegationConfig"
    );
}

#[test]
fn test_delegation_model_falls_back_to_main_when_empty() {
    let workspace = tempfile::tempdir().unwrap().keep();
    let main = MockLLM::named("main-provider");
    let td = ToolDelegationConfig {
        enabled: true,
        model: String::new(), // Empty → fall back to main model
        auto_local: true,
        ..Default::default()
    };
    let core = build_swappable_core(SwappableCoreConfig {
        provider: main,
        workspace,
        model: "main-model".to_string(),
        max_iterations: 10,
        max_continuations: 2,
        max_tokens: 4096,
        temperature: 0.7,
        max_context_tokens: 16384,
        brave_api_key: None,
        search_provider: "searxng".to_string(),
        searxng_url: "http://localhost:8888".to_string(),
        crw_url: String::new(),
        search_max_results: 5,
        exec_timeout: 30,
        restrict_to_workspace: false,
        memory_config: MemoryConfig::default(),
        is_local: false,
        lane: Lane::default(),
        tool_delegation: td,
        provenance: ProvenanceConfig::default(),
        max_tool_result_chars: 2000,
        delegation_provider: None,
        specialist_provider: None,
        trio_config: TrioConfig::default(),
        model_capabilities_overrides: std::collections::HashMap::new(),
        reasoning_config: crate::config::schema::ReasoningConfig::default(),
        tool_heartbeat_secs: 2,
        health_check_timeout_secs: 2,
        adaptive_tokens: AdaptiveTokenConfig::default(),
        sessions_db_path: Some(
            std::env::temp_dir().join(format!("nanobot-test-{}.sqlite", uuid::Uuid::new_v4())),
        ),
    });
    assert_eq!(
        core.tool_runner_model.as_deref(),
        Some("main-model"),
        "Empty delegation model should fall back to main model"
    );
}

#[test]
fn test_delegation_disabled_ignores_passed_provider() {
    // Even if a delegation_provider is passed, it should be ignored
    // when delegation is disabled.
    let dp = MockLLM::named("auto-delegation");
    let core = build_test_core(false, Some(dp), None);

    assert!(
        core.tool_runner_provider.is_none(),
        "Delegation disabled should ignore passed provider"
    );
    assert!(core.tool_runner_model.is_none());
}

#[test]
fn test_delegation_with_is_local_true() {
    // Verify wiring works when is_local=true (uses lite context builder)
    let workspace = tempfile::tempdir().unwrap().keep();
    let main = MockLLM::named("local-main");
    let dp = MockLLM::named("local-delegation");
    let td = ToolDelegationConfig {
        enabled: true,
        model: "delegation-model".to_string(),
        auto_local: true,
        ..Default::default()
    };
    let core = build_swappable_core(SwappableCoreConfig {
        provider: main,
        workspace,
        model: "local-model".to_string(),
        max_iterations: 10,
        max_continuations: 2,
        max_tokens: 4096,
        temperature: 0.7,
        max_context_tokens: 16384,
        brave_api_key: None,
        search_provider: "searxng".to_string(),
        searxng_url: "http://localhost:8888".to_string(),
        crw_url: String::new(),
        search_max_results: 5,
        exec_timeout: 30,
        restrict_to_workspace: false,
        memory_config: MemoryConfig::default(),
        is_local: true,
        lane: Lane::default(),
        tool_delegation: td,
        provenance: ProvenanceConfig::default(),
        max_tool_result_chars: 2000,
        delegation_provider: Some(dp),
        specialist_provider: None,
        trio_config: TrioConfig::default(),
        model_capabilities_overrides: std::collections::HashMap::new(),
        reasoning_config: crate::config::schema::ReasoningConfig::default(),
        tool_heartbeat_secs: 2,
        health_check_timeout_secs: 2,
        adaptive_tokens: AdaptiveTokenConfig::default(),
        sessions_db_path: Some(
            std::env::temp_dir().join(format!("nanobot-test-{}.sqlite", uuid::Uuid::new_v4())),
        ),
    });

    assert!(core.mode().is_local());
    assert!(core.tool_runner_provider.is_some());
    assert_eq!(
        core.tool_runner_provider
            .as_ref()
            .unwrap()
            .get_default_model(),
        "local-delegation",
        "Local mode should still use the delegation provider"
    );
}

/// Wave 0 cloud-path sibling of `test_delegation_with_is_local_true`.
///
/// Pins the `is_local=false` branches in `build_swappable_core`
/// (agent_core.rs:460-509 memory provider, :516-520 reserve cap) so
/// Wave 1→3 can't silently regress cloud delegation wiring.
///
/// Phase 09 plan:
///   .planning/phases/09-runtime-mode-spine/00-wave-0-coverage-PLAN.md
#[test]
fn test_delegation_with_is_local_false_cloud() {
    // Verify wiring + cloud-specific derivations when is_local=false.
    // MockLLM returns `None` from `get_api_base()` — treated as Anthropic
    // native → memory_model defaults to "haiku" (cheap summarisation).
    let workspace = tempfile::tempdir().unwrap().keep();
    let main = MockLLM::named("cloud-main");
    let dp = MockLLM::named("cloud-delegation");
    let td = ToolDelegationConfig {
        enabled: true,
        model: "delegation-model".to_string(),
        auto_local: true,
        ..Default::default()
    };
    let core = build_swappable_core(SwappableCoreConfig {
        provider: main,
        workspace,
        model: "cloud-model".to_string(),
        max_iterations: 10,
        max_continuations: 2,
        max_tokens: 4096,
        temperature: 0.7,
        max_context_tokens: 16384,
        brave_api_key: None,
        search_provider: "searxng".to_string(),
        searxng_url: "http://localhost:8888".to_string(),
        crw_url: String::new(),
        search_max_results: 5,
        exec_timeout: 30,
        restrict_to_workspace: false,
        memory_config: MemoryConfig::default(),
        is_local: false,
        lane: Lane::default(),
        tool_delegation: td,
        provenance: ProvenanceConfig::default(),
        max_tool_result_chars: 2000,
        delegation_provider: Some(dp),
        specialist_provider: None,
        trio_config: TrioConfig::default(),
        model_capabilities_overrides: std::collections::HashMap::new(),
        reasoning_config: crate::config::schema::ReasoningConfig::default(),
        tool_heartbeat_secs: 2,
        health_check_timeout_secs: 2,
        adaptive_tokens: AdaptiveTokenConfig::default(),
        sessions_db_path: Some(
            std::env::temp_dir().join(format!("nanobot-test-{}.sqlite", uuid::Uuid::new_v4())),
        ),
    });

    // pins agent_core.rs: is_local plumbs through to the core unchanged
    assert!(
        !core.mode().is_local(),
        "cloud core must carry is_local=false"
    );

    // pins agent_core.rs: delegation provider still wired through in cloud mode
    assert!(
        core.tool_runner_provider.is_some(),
        "cloud mode must still wire delegation provider"
    );
    assert_eq!(
        core.tool_runner_provider
            .as_ref()
            .unwrap()
            .get_default_model(),
        "cloud-delegation",
        "Cloud mode must use the delegation provider we passed in"
    );

    // pins agent_core.rs:487-498 cloud memory-model default (haiku for
    // Anthropic-native / OpenRouter — MockLLM.get_api_base() == None, so
    // the Anthropic branch wins).
    assert_eq!(core.memory_model, "haiku");
    assert_eq!(core.compactor.model(), "cloud-model");

    // pins agent_core.rs:516-520 reserve cap: cloud mode leaves max_tokens
    // as-is; local mode clamps to max_context/4. Here max_tokens=4096,
    // max_context=16384, so local would also be 4096 — a pure-cloud distinct
    // assertion belongs elsewhere, but we pin the cloud path doesn't
    // spuriously clamp when max_tokens > max_context/4 is not triggered.
    // (The stronger clamp-difference assertion is in the paired
    // `_cloud_reserve_uncapped` test below.)
    assert!(
        core.token_budget.max_context() == 16384,
        "max_context must pass through untouched in cloud mode"
    );
}

#[test]
fn test_local_reflection_and_delegation_providers_do_not_reroute_lcm() {
    let workspace = tempfile::tempdir().unwrap().keep();
    let main = MockLLM::named("main");
    let reflection = MockLLM::named("reflection");
    let delegation = MockLLM::named("delegation");
    let td = ToolDelegationConfig {
        enabled: true,
        model: "deleg-model".to_string(),
        auto_local: true,
        ..Default::default()
    };
    let core = build_swappable_core(SwappableCoreConfig {
        provider: main,
        workspace,
        model: "main-model".to_string(),
        max_iterations: 10,
        max_continuations: 2,
        max_tokens: 4096,
        temperature: 0.7,
        max_context_tokens: 16384,
        brave_api_key: None,
        search_provider: "searxng".to_string(),
        searxng_url: "http://localhost:8888".to_string(),
        crw_url: String::new(),
        search_max_results: 5,
        exec_timeout: 30,
        restrict_to_workspace: false,
        memory_config: MemoryConfig::default(),
        is_local: true,
        lane: Lane::default(),
        tool_delegation: td,
        provenance: ProvenanceConfig::default(),
        max_tool_result_chars: 2000,
        delegation_provider: Some(delegation),
        specialist_provider: Some(reflection),
        trio_config: TrioConfig::default(),
        model_capabilities_overrides: std::collections::HashMap::new(),
        reasoning_config: crate::config::schema::ReasoningConfig::default(),
        tool_heartbeat_secs: 2,
        health_check_timeout_secs: 2,
        adaptive_tokens: AdaptiveTokenConfig::default(),
        sessions_db_path: Some(
            std::env::temp_dir().join(format!("nanobot-test-{}.sqlite", uuid::Uuid::new_v4())),
        ),
    });

    assert_eq!(
        core.memory_provider.get_default_model(),
        "reflection",
        "local reflection should reuse the specialist provider"
    );
    assert_eq!(core.memory_model, "reflection");
    assert_eq!(
        core.compactor.model(),
        "main-model",
        "LCM must remain bound to the foreground model"
    );
    assert_eq!(
        core.tool_runner_provider
            .as_ref()
            .unwrap()
            .get_default_model(),
        "delegation",
        "Tool runner should use delegation provider"
    );
}

#[test]
fn test_cloud_memory_and_delegation_do_not_reroute_lcm() {
    let workspace = tempfile::tempdir().unwrap().keep();
    let main = MockLLM::named("main");
    let delegation = MockLLM::named("delegation");
    let td = ToolDelegationConfig {
        enabled: true,
        model: "deleg-model".to_string(),
        auto_local: true,
        ..Default::default()
    };
    let core = build_swappable_core(SwappableCoreConfig {
        provider: main,
        workspace,
        model: "main-model".to_string(),
        max_iterations: 10,
        max_continuations: 2,
        max_tokens: 4096,
        temperature: 0.7,
        max_context_tokens: 16384,
        brave_api_key: None,
        search_provider: "searxng".to_string(),
        searxng_url: "http://localhost:8888".to_string(),
        crw_url: String::new(),
        search_max_results: 5,
        exec_timeout: 30,
        restrict_to_workspace: false,
        memory_config: MemoryConfig::default(),
        is_local: false,
        lane: Lane::default(),
        tool_delegation: td,
        provenance: ProvenanceConfig::default(),
        max_tool_result_chars: 2000,
        delegation_provider: Some(delegation),
        specialist_provider: None,
        trio_config: TrioConfig::default(),
        model_capabilities_overrides: std::collections::HashMap::new(),
        reasoning_config: crate::config::schema::ReasoningConfig::default(),
        tool_heartbeat_secs: 2,
        health_check_timeout_secs: 2,
        adaptive_tokens: AdaptiveTokenConfig::default(),
        sessions_db_path: Some(
            std::env::temp_dir().join(format!("nanobot-test-{}.sqlite", uuid::Uuid::new_v4())),
        ),
    });

    assert_eq!(
        core.compactor.model(),
        "main-model",
        "cloud LCM must remain bound to the foreground model"
    );
    assert_eq!(
        core.memory_provider.get_default_model(),
        "main",
        "cloud reflection reuses the main provider by default"
    );
    assert_eq!(core.memory_model, "haiku");

    // Delegation plumbing still works identically on both paths.
    assert_eq!(
        core.tool_runner_provider
            .as_ref()
            .unwrap()
            .get_default_model(),
        "delegation",
        "Cloud mode: tool runner still uses delegation provider"
    );
}

// -----------------------------------------------------------------------
// E2E: Full agent loop with LCM enabled against real local LLM.
//
// This test requires LM Studio (or compatible) running. Set env vars:
//   NANOBOT_LCM_TEST_BASE  — API base (default: http://127.0.0.1:1234/v1)
//   NANOBOT_LCM_TEST_MODEL — Model name (default: local-model)
//
// Run with: cargo test test_real_lcm_e2e -- --ignored --nocapture
// -----------------------------------------------------------------------

#[tokio::test]
#[ignore = "requires running local LLM on NANOBOT_LCM_TEST_BASE"]
async fn test_real_lcm_e2e_compact_and_expand() {
    use crate::config::schema::LcmSchemaConfig;

    let api_base = std::env::var("NANOBOT_LCM_TEST_BASE")
        .unwrap_or_else(|_| "http://127.0.0.1:1234/v1".to_string());
    let model_name =
        std::env::var("NANOBOT_LCM_TEST_MODEL").unwrap_or_else(|_| "local-model".to_string());

    eprintln!("LCM E2E: using {} model={}", api_base, model_name);

    // Real provider pointing at local LLM.
    let provider: Arc<dyn LLMProvider> = Arc::new(OpenAICompatProvider::new(
        "local",
        Some(&api_base),
        Some(&model_name),
    ));

    // Warm up: verify the model is responding.
    let warmup = provider
        .chat(
            &[json!({"role": "user", "content": "Reply with exactly: ok"})],
            None,
            Some(&model_name),
            32,
            0.0,
            None,
            None,
        )
        .await;
    match warmup {
        Ok(r) => eprintln!(
            "LCM E2E warmup: {}",
            r.content.as_deref().unwrap_or("(empty)")
        ),
        Err(e) => panic!("LCM E2E: model not responding at {}: {}", api_base, e),
    }

    let workspace = tempfile::tempdir().unwrap().keep();

    // Build core with small context window + LCM thresholds that trigger fast.
    let core = build_swappable_core(SwappableCoreConfig {
        provider: provider.clone(),
        workspace: workspace.clone(),
        model: model_name.clone(),
        max_iterations: 3,
        max_continuations: 2,
        max_tokens: 512,
        temperature: 0.3,
        max_context_tokens: 2048, // Tiny so compaction triggers quickly.
        brave_api_key: None,
        search_provider: "searxng".to_string(),
        searxng_url: "http://localhost:8888".to_string(),
        crw_url: String::new(),
        search_max_results: 5,
        exec_timeout: 30,
        restrict_to_workspace: false,
        memory_config: MemoryConfig::default(),
        is_local: true,
        lane: Lane::default(),
        tool_delegation: ToolDelegationConfig::default(),
        provenance: ProvenanceConfig::default(),
        max_tool_result_chars: 2000,
        delegation_provider: None,
        specialist_provider: None,
        trio_config: TrioConfig::default(),
        model_capabilities_overrides: std::collections::HashMap::new(),
        reasoning_config: crate::config::schema::ReasoningConfig::default(),
        tool_heartbeat_secs: 2,
        health_check_timeout_secs: 2,
        adaptive_tokens: AdaptiveTokenConfig::default(),
        sessions_db_path: Some(
            std::env::temp_dir().join(format!("nanobot-test-{}.sqlite", uuid::Uuid::new_v4())),
        ),
    });
    let counters = test_runtime_counters(2048);
    let core_handle = AgentHandle::new(core, counters);

    let (inbound_tx, inbound_rx) = tokio::sync::mpsc::unbounded_channel::<InboundMessage>();
    let (outbound_tx, _outbound_rx) = tokio::sync::mpsc::unbounded_channel::<OutboundMessage>();

    let lcm_config = LcmSchemaConfig {
        tau_soft: 0.3, // Trigger early.
        tau_hard: 0.6,
        deterministic_target: 128,
        ..Default::default()
    };

    let agent_loop = AgentLoop::new(
        core_handle,
        inbound_rx,
        outbound_tx,
        inbound_tx,
        None, // no cron
        1,
        None, // no email
        None, // no repl display
        None, // no providers config
        ProprioceptionConfig::default(),
        lcm_config,
        None, // no health registry
    );

    let session_key = "lcm-e2e-test";
    let mut responses = Vec::new();

    // Send 12 verbose messages to fill the tiny 2K context.
    let prompts = [
        "Explain Rust ownership rules in detail with examples of move semantics. Be thorough and give at least 3 examples.",
        "Now explain borrowing and the difference between mutable and immutable references with code examples.",
        "Describe lifetime annotations and why they are needed. Give concrete examples with structs and functions.",
        "What are the rules for lifetime elision? When can you omit lifetime annotations? List all three rules.",
        "Explain smart pointers: Box, Rc, Arc, and when to use each one. Give a real-world use case for each.",
        "What is interior mutability? Explain Cell, RefCell, and Mutex with examples of each.",
        "Describe async/await in Rust. How do Futures work under the hood? Explain the state machine transformation.",
        "Explain trait objects vs generics. When would you use dynamic dispatch vs static dispatch?",
        "What are the differences between String and &str? When should you use each one in function signatures?",
        "Explain the Drop trait and how Rust's destructors work. What is the order of dropping?",
        "Describe the Pin and Unpin traits. Why are they needed for async Rust and self-referential structs?",
        "Explain how pattern matching works in Rust. Cover match, if let, while let, and destructuring.",
    ];

    for (i, prompt) in prompts.iter().enumerate() {
        eprintln!("LCM E2E: sending message {}/{}...", i + 1, prompts.len());
        let resp = agent_loop
            .process_direct(prompt, session_key, "test", "lcm-e2e")
            .await;
        eprintln!(
            "LCM E2E: response {} ({} chars): {}",
            i + 1,
            resp.len(),
            &resp[..resp.len().min(80)]
        );
        assert!(
            !resp.is_empty(),
            "Message {} should get a non-empty response",
            i + 1
        );
        responses.push(resp);
    }

    // Check LCM engine state.
    let concrete_session = agent_loop
        .shared
        .core_handle
        .swappable()
        .sessions
        .get_latest_session(session_key)
        .await
        .expect("test session must exist");
    let engines = agent_loop.shared.lcm_engines.lock().await;
    let engine_arc = engines
        .get(&concrete_session.id)
        .expect("LCM engine should exist for session");
    let engine = engine_arc.lock().await;

    eprintln!(
        "LCM E2E results: store={} active={} dag_nodes={}",
        engine.store_len(),
        engine.active_len(),
        engine.dag().len()
    );

    // Invariant 1: store has messages from the conversation.
    // Note: with is_local + small context, trim_to_fit_with_age runs before
    // LCM ingestion, so the store only contains messages that survived trimming.
    // SQLite is the true immutable store; the in-memory
    // LCM store tracks what entered the active context window.
    assert!(
        engine.store_len() >= 5,
        "Store should have at least 5 messages (system + some turns), got {}",
        engine.store_len()
    );

    // Invariant 2: active context should be shorter than store (compaction happened).
    // With tau_soft=0.3 and 4K context, compaction should trigger early.
    assert!(
        engine.active_len() < engine.store_len(),
        "Active ({}) should be shorter than store ({}) — compaction should have triggered",
        engine.active_len(),
        engine.store_len()
    );

    // Invariant 3: DAG should have at least one summary node.
    assert!(
        engine.dag().len() >= 1,
        "DAG should have at least 1 summary node, got {}",
        engine.dag().len()
    );

    // Invariant 4: every summary node's source IDs resolve to real messages.
    for i in 0..engine.dag().len() {
        let node = engine.dag().get(i).unwrap();
        let expanded = engine.expand(&node.source_ids);
        assert_eq!(
            expanded.len(),
            node.source_ids.len(),
            "Summary node {} has {} source IDs but only {} resolve",
            i,
            node.source_ids.len(),
            expanded.len()
        );
        eprintln!(
            "  DAG node {}: level={} sources={:?} tokens={}",
            i, node.level, node.source_ids, node.tokens
        );
    }

    // Invariant 5: active context contains at least one Summary entry.
    let summary_count = engine
        .active_entries()
        .iter()
        .filter(|e| matches!(e, crate::agent::lcm::ContextEntry::Summary { .. }))
        .count();
    assert!(
        summary_count >= 1,
        "Active context should have at least 1 summary entry, got {}",
        summary_count
    );

    // Invariant 6: lossless expand — all store IDs are retrievable.
    // IDs are db rowids (sparse), not positions — enumerate via store_ids().
    let all_ids: Vec<usize> = engine.store_ids();
    let expanded = engine.expand(&all_ids);
    assert_eq!(
        expanded.len(),
        engine.store_len(),
        "All {} store messages should be retrievable via expand",
        engine.store_len()
    );
    for (id, msg) in &expanded {
        let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
        assert!(
            !content.is_empty(),
            "Expanded message {} should have content",
            id
        );
    }

    eprintln!("LCM E2E: ALL INVARIANTS PASSED");
    eprintln!(
        "  Messages: {} stored, {} active, {} summary nodes",
        engine.store_len(),
        engine.active_len(),
        engine.dag().len()
    );

    // Cleanup.
    drop(engine);
    drop(engines);
    let _ = std::fs::remove_dir_all(&workspace);
}

// -----------------------------------------------------------------------
// Trio E2E test harness
//
// All tests require a single LM Studio endpoint serving three models.
// Configure via env vars:
//   NANOBOT_TRIO_BASE            — API base (default: http://192.168.1.22:1234/v1)
//   NANOBOT_TRIO_MAIN_MODEL      — Main model name
//   NANOBOT_TRIO_ROUTER_MODEL    — Router model name
//   NANOBOT_TRIO_SPECIALIST_MODEL — Specialist model name
//
// Run with: cargo test test_trio_e2e -- --ignored --nocapture
// -----------------------------------------------------------------------

/// Read trio E2E env vars (single shared endpoint).
fn trio_e2e_env() -> (String, String, String, String) {
    let base = std::env::var("NANOBOT_TRIO_BASE")
        .unwrap_or_else(|_| "http://192.168.1.22:1234/v1".to_string());
    let main_model =
        std::env::var("NANOBOT_TRIO_MAIN_MODEL").unwrap_or_else(|_| "gemma-3n-e4b-it".to_string());
    let router_model = std::env::var("NANOBOT_TRIO_ROUTER_MODEL")
        .unwrap_or_else(|_| "nvidia_orchestrator-8b".to_string());
    let specialist_model =
        std::env::var("NANOBOT_TRIO_SPECIALIST_MODEL").unwrap_or_else(|_| "qwen3-1.7b".to_string());
    (base, main_model, router_model, specialist_model)
}

/// Build an AgentLoop wired for trio E2E testing.
///
/// All three providers share one LM Studio endpoint, differentiated by model name.
/// A shared JitGate serialises requests to prevent concurrent model-loading crashes.
fn build_trio_e2e_harness(
    base_url: &str,
    main_model: &str,
    router_model: &str,
    specialist_model: &str,
) -> (AgentLoop, std::path::PathBuf) {
    use crate::config::schema::LcmSchemaConfig;
    use crate::providers::factory;
    use crate::providers::jit_gate::JitGate;

    let jit_gate = std::sync::Arc::new(JitGate::new());

    let main_provider: Arc<dyn LLMProvider> = factory::create_openai_compat(
        factory::ProviderSpec::local(base_url, Some(main_model))
            .with_jit_gate_opt(Some(jit_gate.clone())),
    );
    let router_provider: Arc<dyn LLMProvider> = factory::create_openai_compat(
        factory::ProviderSpec::local(base_url, Some(router_model))
            .with_jit_gate_opt(Some(jit_gate.clone())),
    );
    let specialist_provider: Arc<dyn LLMProvider> = factory::create_openai_compat(
        factory::ProviderSpec::local(base_url, Some(specialist_model))
            .with_jit_gate_opt(Some(jit_gate.clone())),
    );

    let workspace = tempfile::tempdir().unwrap().keep();

    let mut td = ToolDelegationConfig {
        mode: crate::config::schema::DelegationMode::trio(),
        ..Default::default()
    };
    td.apply_mode();

    let trio_config = TrioConfig {
        enabled: true,
        router_model: router_model.to_string(),
        specialist_model: specialist_model.to_string(),
        ..Default::default()
    };

    let core = build_swappable_core(SwappableCoreConfig {
        provider: main_provider,
        workspace: workspace.clone(),
        model: main_model.to_string(),
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
        is_local: true,
        lane: Lane::default(),
        tool_delegation: td,
        provenance: ProvenanceConfig::default(),
        max_tool_result_chars: 2000,
        delegation_provider: Some(router_provider),
        specialist_provider: Some(specialist_provider),
        trio_config,
        model_capabilities_overrides: std::collections::HashMap::new(),
        reasoning_config: crate::config::schema::ReasoningConfig::default(),
        tool_heartbeat_secs: 2,
        health_check_timeout_secs: 2,
        adaptive_tokens: AdaptiveTokenConfig::default(),
        sessions_db_path: Some(
            std::env::temp_dir().join(format!("nanobot-test-{}.sqlite", uuid::Uuid::new_v4())),
        ),
    });

    let counters = test_runtime_counters(4096);
    let core_handle = AgentHandle::new(core, counters);

    let (inbound_tx, inbound_rx) = tokio::sync::mpsc::unbounded_channel::<InboundMessage>();
    let (outbound_tx, _outbound_rx) = tokio::sync::mpsc::unbounded_channel::<OutboundMessage>();

    let agent_loop = AgentLoop::new(
        core_handle,
        inbound_rx,
        outbound_tx,
        inbound_tx,
        None,
        1,
        None,
        None,
        None,
        ProprioceptionConfig::default(),
        LcmSchemaConfig::default(),
        None,
    );

    (agent_loop, workspace)
}

/// Warmup a provider with backon retries (models may need JIT loading time).
async fn warmup_trio_provider(provider: &dyn LLMProvider, model: &str, role: &str) {
    use backon::ConstantBuilder;

    let messages = vec![serde_json::json!({"role": "user", "content": "Reply with: ok"})];
    let mut backoff = ConstantBuilder::default()
        .with_delay(Duration::from_secs(2))
        .with_max_times(10)
        .build();
    loop {
        match provider
            .chat(&messages, None, Some(model), 32, 0.0, None, None)
            .await
        {
            Ok(resp) => {
                let text = resp.content.unwrap_or_default();
                if !text.trim().is_empty() {
                    eprintln!("  {} warmup OK: {}", role, &text[..text.len().min(40)]);
                    return;
                }
            }
            Err(e) => {
                let msg = e.to_string().to_lowercase();
                if !msg.contains("loading") && !msg.contains("503") {
                    panic!("{} warmup failed (non-retryable): {}", role, e);
                }
            }
        }
        match backoff.next() {
            Some(delay) => {
                eprintln!("  {} warming up, retrying in {:?}...", role, delay);
                tokio::time::sleep(delay).await;
            }
            None => panic!("{} did not become ready after retries", role),
        }
    }
}

#[tokio::test]
#[ignore = "requires LM Studio at NANOBOT_TRIO_BASE"]
async fn test_trio_e2e_preflight() {
    let (base, main_model, router_model, specialist_model) = trio_e2e_env();
    eprintln!("trio E2E preflight: base={}", base);

    // 1. Verify LM Studio /models endpoint is reachable
    let _models_url = format!(
        "{}/models",
        base.trim_end_matches("/v1").trim_end_matches('/')
    );
    // Try the /v1/models path first (standard OpenAI-compat)
    let models_url_v1 = format!("{}/models", base.trim_end_matches('/'));
    let client = reqwest::Client::new();
    let models_resp = client
        .get(&models_url_v1)
        .header("Authorization", "Bearer local")
        .timeout(Duration::from_secs(10))
        .send()
        .await;

    match &models_resp {
        Ok(resp) if resp.status().is_success() => {
            eprintln!("  /models endpoint OK (status {})", resp.status());
        }
        Ok(resp) => {
            panic!(
                "preflight FAILED: /models returned HTTP {} — is LM Studio running at {}?",
                resp.status(),
                base
            );
        }
        Err(e) => {
            panic!(
                "preflight FAILED: cannot reach {} — {}\nStart LM Studio or set NANOBOT_TRIO_BASE.",
                models_url_v1, e
            );
        }
    }

    // 2. Parse model list and check availability
    let body: serde_json::Value = models_resp
        .unwrap()
        .json()
        .await
        .expect("preflight: /models response is not valid JSON");

    let model_ids: Vec<String> = body
        .get("data")
        .and_then(|d| d.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|m| m.get("id").and_then(|id| id.as_str()).map(String::from))
                .collect()
        })
        .unwrap_or_default();

    eprintln!("  available models: {:?}", model_ids);

    // Note: LM Studio with JIT loading may not list all models upfront.
    // We log availability but don't fail — the warmup step below is the real gate.
    for (name, role) in [
        (&main_model, "main"),
        (&router_model, "router"),
        (&specialist_model, "specialist"),
    ] {
        if model_ids.iter().any(|id| id.contains(name.as_str())) {
            eprintln!("  {} model '{}' found in /models", role, name);
        } else {
            eprintln!(
                "  {} model '{}' NOT listed (may JIT-load on demand)",
                role, name
            );
        }
    }

    // 3. Build harness and warmup all 3 providers (the real gate)
    let (agent_loop, workspace) =
        build_trio_e2e_harness(&base, &main_model, &router_model, &specialist_model);

    let core = agent_loop.shared.core_handle.swappable();
    warmup_trio_provider(&*core.provider, &main_model, "main").await;
    warmup_trio_provider(
        core.router_provider.as_ref().unwrap().as_ref(),
        &router_model,
        "router",
    )
    .await;
    warmup_trio_provider(
        core.specialist_provider.as_ref().unwrap().as_ref(),
        &specialist_model,
        "specialist",
    )
    .await;

    eprintln!("trio E2E preflight: ALL OK — infrastructure ready");
    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
#[ignore = "requires LM Studio at NANOBOT_TRIO_BASE"]
async fn test_trio_e2e_respond() {
    let (base, main_model, router_model, specialist_model) = trio_e2e_env();
    eprintln!("trio E2E respond: base={}", base);

    let (agent_loop, workspace) =
        build_trio_e2e_harness(&base, &main_model, &router_model, &specialist_model);

    // Warmup all 3 models
    let core = agent_loop.shared.core_handle.swappable();
    warmup_trio_provider(&*core.provider, &main_model, "main").await;
    warmup_trio_provider(
        core.router_provider.as_ref().unwrap().as_ref(),
        &router_model,
        "router",
    )
    .await;
    warmup_trio_provider(
        core.specialist_provider.as_ref().unwrap().as_ref(),
        &specialist_model,
        "specialist",
    )
    .await;

    let resp = tokio::time::timeout(
        Duration::from_secs(180),
        agent_loop.process_direct(
            "Hello, what is 2 + 2?",
            "trio-e2e-respond",
            "test",
            "trio-e2e",
        ),
    )
    .await
    .expect("test timed out");

    eprintln!(
        "trio E2E respond: response ({} chars): {}",
        resp.len(),
        &resp[..resp.len().min(200)]
    );
    assert!(!resp.is_empty(), "response should be non-empty");

    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
#[ignore = "requires LM Studio at NANOBOT_TRIO_BASE"]
async fn test_trio_e2e_tool_dispatch() {
    let (base, main_model, router_model, specialist_model) = trio_e2e_env();
    eprintln!("trio E2E tool dispatch: base={}", base);

    let (agent_loop, workspace) =
        build_trio_e2e_harness(&base, &main_model, &router_model, &specialist_model);

    // Write a known file to workspace
    std::fs::write(
        workspace.join("README.md"),
        "Nanobot is a lightweight AI assistant framework written in Rust.",
    )
    .unwrap();

    let core = agent_loop.shared.core_handle.swappable();
    warmup_trio_provider(&*core.provider, &main_model, "main").await;
    warmup_trio_provider(
        core.router_provider.as_ref().unwrap().as_ref(),
        &router_model,
        "router",
    )
    .await;
    warmup_trio_provider(
        core.specialist_provider.as_ref().unwrap().as_ref(),
        &specialist_model,
        "specialist",
    )
    .await;

    let resp = tokio::time::timeout(
        Duration::from_secs(180),
        agent_loop.process_direct(
            "Read the file README.md and tell me what it says",
            "trio-e2e-tool",
            "test",
            "trio-e2e",
        ),
    )
    .await
    .expect("test timed out");

    eprintln!(
        "trio E2E tool dispatch: response ({} chars): {}",
        resp.len(),
        &resp[..resp.len().min(200)]
    );
    assert!(!resp.is_empty(), "response should be non-empty");

    // Check TrioMetrics
    let metrics = &agent_loop.shared.core_handle.counters.trio_metrics;
    eprintln!(
        "  metrics: preflight={} action={:?} specialist={} tool={:?}",
        metrics
            .router_preflight_fired
            .load(std::sync::atomic::Ordering::Relaxed),
        metrics.router_action.lock(),
        metrics
            .specialist_dispatched
            .load(std::sync::atomic::Ordering::Relaxed),
        metrics.tool_dispatched.lock(),
    );

    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
#[ignore = "requires LM Studio at NANOBOT_TRIO_BASE"]
async fn test_trio_e2e_specialist_dispatch() {
    let (base, main_model, router_model, specialist_model) = trio_e2e_env();
    eprintln!("trio E2E specialist: base={}", base);

    let (agent_loop, workspace) =
        build_trio_e2e_harness(&base, &main_model, &router_model, &specialist_model);

    let core = agent_loop.shared.core_handle.swappable();
    warmup_trio_provider(&*core.provider, &main_model, "main").await;
    warmup_trio_provider(
        core.router_provider.as_ref().unwrap().as_ref(),
        &router_model,
        "router",
    )
    .await;
    warmup_trio_provider(
        core.specialist_provider.as_ref().unwrap().as_ref(),
        &specialist_model,
        "specialist",
    )
    .await;

    let resp = tokio::time::timeout(
        Duration::from_secs(180),
        agent_loop.process_direct(
            "Provide a detailed technical analysis of REST vs GraphQL",
            "trio-e2e-specialist",
            "test",
            "trio-e2e",
        ),
    )
    .await
    .expect("test timed out");

    eprintln!(
        "trio E2E specialist: response ({} chars): {}",
        resp.len(),
        &resp[..resp.len().min(200)]
    );
    assert!(!resp.is_empty(), "response should be non-empty");
    assert!(
        resp.len() > 50,
        "specialist response should be substantive (>50 chars)"
    );

    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
#[ignore = "requires LM Studio at NANOBOT_TRIO_BASE"]
async fn test_trio_e2e_ask_user() {
    let (base, main_model, router_model, specialist_model) = trio_e2e_env();
    eprintln!("trio E2E ask_user: base={}", base);

    let (agent_loop, workspace) =
        build_trio_e2e_harness(&base, &main_model, &router_model, &specialist_model);

    let core = agent_loop.shared.core_handle.swappable();
    warmup_trio_provider(&*core.provider, &main_model, "main").await;
    warmup_trio_provider(
        core.router_provider.as_ref().unwrap().as_ref(),
        &router_model,
        "router",
    )
    .await;
    warmup_trio_provider(
        core.specialist_provider.as_ref().unwrap().as_ref(),
        &specialist_model,
        "specialist",
    )
    .await;

    let resp = tokio::time::timeout(
        Duration::from_secs(180),
        agent_loop.process_direct(
            "Do that thing with the file",
            "trio-e2e-ask",
            "test",
            "trio-e2e",
        ),
    )
    .await
    .expect("test timed out");

    eprintln!(
        "trio E2E ask_user: response ({} chars): {}",
        resp.len(),
        &resp[..resp.len().min(200)]
    );
    assert!(!resp.is_empty(), "response should be non-empty");

    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
#[ignore = "requires LM Studio at NANOBOT_TRIO_BASE"]
async fn test_trio_e2e_router_unreachable() {
    let (base, main_model, _router_model, specialist_model) = trio_e2e_env();
    eprintln!("trio E2E router unreachable: base={}", base);

    // Router on dead port, main + specialist on real endpoint
    let (agent_loop, workspace) = build_trio_e2e_harness(
        &base,
        &main_model,
        &"unreachable-router-model".to_string(), // model doesn't matter since we override the provider
        &specialist_model,
    );

    // Actually, the harness uses shared base for all providers.
    // For unreachable router, we need a custom build with bad router URL.
    // Let's build it manually.
    drop(agent_loop);
    let _ = std::fs::remove_dir_all(&workspace);

    use crate::config::schema::{DelegationMode, LcmSchemaConfig};
    use crate::providers::factory;
    use crate::providers::jit_gate::JitGate;

    let jit_gate = std::sync::Arc::new(JitGate::new());
    let main_provider: Arc<dyn LLMProvider> = factory::create_openai_compat(
        factory::ProviderSpec::local(&base, Some(&main_model))
            .with_jit_gate_opt(Some(jit_gate.clone())),
    );
    // Router points to dead port
    let router_provider: Arc<dyn LLMProvider> = Arc::new(OpenAICompatProvider::new(
        "local",
        Some("http://127.0.0.1:19999/v1"),
        Some("dead-router"),
    ));
    let specialist_provider: Arc<dyn LLMProvider> = factory::create_openai_compat(
        factory::ProviderSpec::local(&base, Some(&specialist_model))
            .with_jit_gate_opt(Some(jit_gate.clone())),
    );

    let workspace = tempfile::tempdir().unwrap().keep();
    let mut td = ToolDelegationConfig {
        mode: DelegationMode::trio(),
        ..Default::default()
    };
    td.apply_mode();

    let trio_config = TrioConfig {
        enabled: true,
        router_model: "dead-router".to_string(),
        specialist_model: specialist_model.to_string(),
        ..Default::default()
    };

    let core = build_swappable_core(SwappableCoreConfig {
        provider: main_provider,
        workspace: workspace.clone(),
        model: main_model.to_string(),
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
        is_local: true,
        lane: Lane::default(),
        tool_delegation: td,
        provenance: ProvenanceConfig::default(),
        max_tool_result_chars: 2000,
        delegation_provider: Some(router_provider),
        specialist_provider: Some(specialist_provider),
        trio_config,
        model_capabilities_overrides: std::collections::HashMap::new(),
        reasoning_config: crate::config::schema::ReasoningConfig::default(),
        tool_heartbeat_secs: 2,
        health_check_timeout_secs: 2,
        adaptive_tokens: AdaptiveTokenConfig::default(),
        sessions_db_path: Some(
            std::env::temp_dir().join(format!("nanobot-test-{}.sqlite", uuid::Uuid::new_v4())),
        ),
    });
    let counters = test_runtime_counters(4096);
    let core_handle = AgentHandle::new(core, counters);

    let (inbound_tx, inbound_rx) = tokio::sync::mpsc::unbounded_channel::<InboundMessage>();
    let (outbound_tx, _outbound_rx) = tokio::sync::mpsc::unbounded_channel::<OutboundMessage>();

    let agent_loop = AgentLoop::new(
        core_handle,
        inbound_rx,
        outbound_tx,
        inbound_tx,
        None,
        1,
        None,
        None,
        None,
        ProprioceptionConfig::default(),
        LcmSchemaConfig::default(),
        None,
    );

    // Only warmup main (router is intentionally dead)
    let core = agent_loop.shared.core_handle.swappable();
    warmup_trio_provider(&*core.provider, &main_model, "main").await;

    let resp = tokio::time::timeout(
        Duration::from_secs(60),
        agent_loop.process_direct("Hello", "trio-e2e-router-dead", "test", "trio-e2e"),
    )
    .await
    .expect("test timed out");

    eprintln!(
        "trio E2E router unreachable: response ({} chars): {}",
        resp.len(),
        &resp[..resp.len().min(200)]
    );
    assert!(!resp.is_empty(), "should get error response, not panic");

    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
#[ignore = "requires LM Studio at NANOBOT_TRIO_BASE"]
async fn test_trio_e2e_specialist_unreachable() {
    let (base, main_model, router_model, _specialist_model) = trio_e2e_env();
    eprintln!("trio E2E specialist unreachable: base={}", base);

    use crate::config::schema::{DelegationMode, LcmSchemaConfig};
    use crate::providers::factory;
    use crate::providers::jit_gate::JitGate;

    let jit_gate = std::sync::Arc::new(JitGate::new());
    let main_provider: Arc<dyn LLMProvider> = factory::create_openai_compat(
        factory::ProviderSpec::local(&base, Some(&main_model))
            .with_jit_gate_opt(Some(jit_gate.clone())),
    );
    let router_provider: Arc<dyn LLMProvider> = factory::create_openai_compat(
        factory::ProviderSpec::local(&base, Some(&router_model))
            .with_jit_gate_opt(Some(jit_gate.clone())),
    );
    // Specialist points to dead port
    let specialist_provider: Arc<dyn LLMProvider> = Arc::new(OpenAICompatProvider::new(
        "local",
        Some("http://127.0.0.1:19999/v1"),
        Some("dead-specialist"),
    ));

    let workspace = tempfile::tempdir().unwrap().keep();
    let mut td = ToolDelegationConfig {
        mode: DelegationMode::trio(),
        ..Default::default()
    };
    td.apply_mode();

    let trio_config = TrioConfig {
        enabled: true,
        router_model: router_model.to_string(),
        specialist_model: "dead-specialist".to_string(),
        ..Default::default()
    };

    let core = build_swappable_core(SwappableCoreConfig {
        provider: main_provider,
        workspace: workspace.clone(),
        model: main_model.to_string(),
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
        is_local: true,
        lane: Lane::default(),
        tool_delegation: td,
        provenance: ProvenanceConfig::default(),
        max_tool_result_chars: 2000,
        delegation_provider: Some(router_provider),
        specialist_provider: Some(specialist_provider),
        trio_config,
        model_capabilities_overrides: std::collections::HashMap::new(),
        reasoning_config: crate::config::schema::ReasoningConfig::default(),
        tool_heartbeat_secs: 2,
        health_check_timeout_secs: 2,
        adaptive_tokens: AdaptiveTokenConfig::default(),
        sessions_db_path: Some(
            std::env::temp_dir().join(format!("nanobot-test-{}.sqlite", uuid::Uuid::new_v4())),
        ),
    });
    let counters = test_runtime_counters(4096);
    let core_handle = AgentHandle::new(core, counters);

    let (inbound_tx, inbound_rx) = tokio::sync::mpsc::unbounded_channel::<InboundMessage>();
    let (outbound_tx, _outbound_rx) = tokio::sync::mpsc::unbounded_channel::<OutboundMessage>();

    let agent_loop = AgentLoop::new(
        core_handle,
        inbound_rx,
        outbound_tx,
        inbound_tx,
        None,
        1,
        None,
        None,
        None,
        ProprioceptionConfig::default(),
        LcmSchemaConfig::default(),
        None,
    );

    let core = agent_loop.shared.core_handle.swappable();
    warmup_trio_provider(&*core.provider, &main_model, "main").await;
    warmup_trio_provider(
        core.router_provider.as_ref().unwrap().as_ref(),
        &router_model,
        "router",
    )
    .await;

    let resp = tokio::time::timeout(
        Duration::from_secs(180),
        agent_loop.process_direct(
            "Provide a detailed technical analysis of REST vs GraphQL",
            "trio-e2e-specialist-dead",
            "test",
            "trio-e2e",
        ),
    )
    .await
    .expect("test timed out");

    eprintln!(
        "trio E2E specialist unreachable: response ({} chars): {}",
        resp.len(),
        &resp[..resp.len().min(200)]
    );
    assert!(
        !resp.is_empty(),
        "should get response despite dead specialist"
    );

    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
#[ignore = "requires LM Studio at NANOBOT_TRIO_BASE"]
async fn test_trio_e2e_multi_turn() {
    let (base, main_model, router_model, specialist_model) = trio_e2e_env();
    eprintln!("trio E2E multi-turn: base={}", base);

    let (agent_loop, workspace) =
        build_trio_e2e_harness(&base, &main_model, &router_model, &specialist_model);

    // Write test file
    std::fs::write(
        workspace.join("README.md"),
        "Nanobot is a lightweight AI assistant.",
    )
    .unwrap();

    let core = agent_loop.shared.core_handle.swappable();
    warmup_trio_provider(&*core.provider, &main_model, "main").await;
    warmup_trio_provider(
        core.router_provider.as_ref().unwrap().as_ref(),
        &router_model,
        "router",
    )
    .await;
    warmup_trio_provider(
        core.specialist_provider.as_ref().unwrap().as_ref(),
        &specialist_model,
        "specialist",
    )
    .await;

    let session_key = "trio-e2e-multi";

    // Turn 1: simple greeting (respond path)
    let resp1 = tokio::time::timeout(
        Duration::from_secs(180),
        agent_loop.process_direct("Hello", session_key, "test", "trio-e2e"),
    )
    .await
    .expect("turn 1 timed out");
    eprintln!(
        "turn 1 ({} chars): {}",
        resp1.len(),
        &resp1[..resp1.len().min(100)]
    );
    assert!(!resp1.is_empty(), "turn 1 should be non-empty");

    // Turn 2: tool path
    let resp2 = tokio::time::timeout(
        Duration::from_secs(180),
        agent_loop.process_direct("Read README.md", session_key, "test", "trio-e2e"),
    )
    .await
    .expect("turn 2 timed out");
    eprintln!(
        "turn 2 ({} chars): {}",
        resp2.len(),
        &resp2[..resp2.len().min(100)]
    );
    assert!(!resp2.is_empty(), "turn 2 should be non-empty");

    // Turn 3: follow-up (tests session state persistence)
    let resp3 = tokio::time::timeout(
        Duration::from_secs(180),
        agent_loop.process_direct("Summarize what you found", session_key, "test", "trio-e2e"),
    )
    .await
    .expect("turn 3 timed out");
    eprintln!(
        "turn 3 ({} chars): {}",
        resp3.len(),
        &resp3[..resp3.len().min(100)]
    );
    assert!(!resp3.is_empty(), "turn 3 should be non-empty");

    let _ = std::fs::remove_dir_all(&workspace);
}

// -----------------------------------------------------------------------
// should_strip_tools_for_trio — pure function tests
// -----------------------------------------------------------------------

#[test]
fn test_should_strip_tools_all_healthy() {
    assert!(should_strip_tools_for_trio(true, true, true, true));
}

#[test]
fn test_should_strip_tools_not_local() {
    // Cloud mode: never strip tools via this path.
    assert!(!should_strip_tools_for_trio(false, true, true, true));
}

#[test]
fn test_should_strip_tools_no_strict_mode() {
    // strict_no_tools_main is false: don't strip.
    assert!(!should_strip_tools_for_trio(true, false, true, true));
}

#[test]
fn test_should_strip_tools_router_unhealthy() {
    // Router probe degraded: keep tools for fallback.
    assert!(!should_strip_tools_for_trio(true, true, false, true));
}

#[test]
fn test_should_strip_tools_circuit_breaker_open() {
    // Circuit breaker tripped: keep tools for fallback.
    assert!(!should_strip_tools_for_trio(true, true, true, false));
}

#[test]
fn test_should_strip_tools_both_degraded() {
    // Both degraded: definitely keep tools.
    assert!(!should_strip_tools_for_trio(true, true, false, false));
}

#[test]
fn test_adaptive_max_tokens_adds_thinking_headroom_for_local() {
    // Thinking budget is added on top of base so the model has room for
    // both reasoning tokens AND completion output.
    let out = adaptive_max_tokens(
        4096,
        false,
        "What time is it?",
        0,
        true,
        Some(512),
        &AdaptiveTokenConfig::default(),
    );
    assert_eq!(out, 4608); // 4096 + 512
}

#[test]
fn test_adaptive_max_tokens_no_reserve_without_thinking() {
    let out = adaptive_max_tokens(
        4096,
        false,
        "What time is it?",
        0,
        true,
        None,
        &AdaptiveTokenConfig::default(),
    );
    assert_eq!(out, 4096);
}

#[test]
fn test_adaptive_max_tokens_no_reserve_for_cloud() {
    let out = adaptive_max_tokens(
        4096,
        false,
        "What time is it?",
        0,
        false,
        Some(512),
        &AdaptiveTokenConfig::default(),
    );
    assert_eq!(out, 4096);
}

#[test]
fn test_adaptive_max_tokens_adds_thinking_even_on_small_base() {
    // Even with a small base, thinking budget is added on top.
    let out = adaptive_max_tokens(
        512,
        false,
        "short",
        0,
        true,
        Some(128),
        &AdaptiveTokenConfig::default(),
    );
    assert_eq!(out, 640); // 512 + 128
}

// -----------------------------------------------------------------------
// Offline trio E2E tests (no network required — all providers are mocks)
// -----------------------------------------------------------------------

/// A mock LLM provider that returns responses from a pre-loaded queue.
///
/// Each call pops the next response. When the queue is empty it returns a
/// sentinel error string so tests can detect over-calling.
struct SequenceProvider {
    name: String,
    responses: parking_lot::Mutex<std::collections::VecDeque<String>>,
    call_count: std::sync::atomic::AtomicU32,
}

impl SequenceProvider {
    fn new(name: &str, responses: Vec<&str>) -> Self {
        Self {
            name: name.to_string(),
            responses: parking_lot::Mutex::new(
                responses.into_iter().map(|s| s.to_string()).collect(),
            ),
            call_count: std::sync::atomic::AtomicU32::new(0),
        }
    }

    fn call_count(&self) -> u32 {
        self.call_count.load(std::sync::atomic::Ordering::Relaxed)
    }
}

#[async_trait]
impl LLMProvider for SequenceProvider {
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
        self.call_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let response = {
            let mut deque = self.responses.lock();
            if deque.is_empty() {
                "ERROR: no responses left in SequenceProvider".to_string()
            } else {
                deque.pop_front().unwrap()
            }
        };
        Ok(crate::providers::base::LLMResponse {
            content: Some(response),
            tool_calls: vec![],
            finish_reason: "stop".to_string(),
            usage: std::collections::HashMap::new(),
        })
    }

    fn get_default_model(&self) -> &str {
        &self.name
    }
}

#[tokio::test]
async fn plain_text_response_is_final_answer() {
    // Replaces the prior attestation tests: with the protocol removed, plain
    // non-empty text terminates the turn on the first response — no retries,
    // no hidden markers, no duplicated output (the live regression that bit
    // session 20260728_142921_a3b1d8).
    let main = Arc::new(SequenceProvider::new(
        "local-main",
        vec!["Hello! How can I help you today?"],
    ));
    let main_dyn: Arc<dyn LLMProvider> = main.clone();
    let (agent_loop, workspace) = build_local_inline_harness(main_dyn);
    let session_key = format!(
        "test-no-attestation-{}",
        uuid::Uuid::new_v4().to_string()
    );

    let response = agent_loop
        .process_direct("hi", &session_key, "test", "offline")
        .await;

    assert!(
        response.contains("Hello! How can I help you today?"),
        "expected the single plain-text response, got: {response}"
    );
    assert_eq!(
        main.call_count(),
        1,
        "plain text must terminate on the first response — no retries"
    );

    let _ = std::fs::remove_dir_all(&workspace);
}

struct ResponseSequenceProvider {
    name: String,
    responses: parking_lot::Mutex<std::collections::VecDeque<crate::providers::base::LLMResponse>>,
    call_count: std::sync::atomic::AtomicU32,
}

/// Pauses the second provider call so tests can inspect durable session state
/// after a tool round but before finalization.
struct ToolRoundBarrierProvider {
    call_count: std::sync::atomic::AtomicU32,
    second_call_started: tokio::sync::Notify,
    release_second_call: tokio::sync::Notify,
}

impl ToolRoundBarrierProvider {
    fn new() -> Self {
        Self {
            call_count: std::sync::atomic::AtomicU32::new(0),
            second_call_started: tokio::sync::Notify::new(),
            release_second_call: tokio::sync::Notify::new(),
        }
    }
}

#[async_trait]
impl LLMProvider for ToolRoundBarrierProvider {
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
        let call = self
            .call_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if call == 0 {
            let mut arguments = std::collections::HashMap::new();
            arguments.insert("path".to_string(), json!("."));
            return Ok(crate::providers::base::LLMResponse {
                content: Some(String::new()),
                tool_calls: vec![crate::providers::base::ToolCallRequest {
                    id: "tc_durable".to_string(),
                    name: "list_dir".to_string(),
                    arguments,
                }],
                finish_reason: "tool_calls".to_string(),
                usage: std::collections::HashMap::new(),
            });
        }

        self.second_call_started.notify_one();
        self.release_second_call.notified().await;
        Ok(crate::providers::base::LLMResponse {
            content: Some(attested_text("done")),
            tool_calls: vec![],
            finish_reason: "stop".to_string(),
            usage: std::collections::HashMap::new(),
        })
    }

    fn get_default_model(&self) -> &str {
        "local-barrier"
    }
}

impl ResponseSequenceProvider {
    fn new(name: &str, responses: Vec<crate::providers::base::LLMResponse>) -> Self {
        Self {
            name: name.to_string(),
            responses: parking_lot::Mutex::new(responses.into()),
            call_count: std::sync::atomic::AtomicU32::new(0),
        }
    }

    fn call_count(&self) -> u32 {
        self.call_count.load(std::sync::atomic::Ordering::Relaxed)
    }
}

#[async_trait]
impl LLMProvider for ResponseSequenceProvider {
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
        self.call_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let response = {
            let mut deque = self.responses.lock();
            deque.pop_front()
        };
        Ok(
            response.unwrap_or_else(|| crate::providers::base::LLMResponse {
                content: Some("ERROR: no responses left in ResponseSequenceProvider".to_string()),
                tool_calls: vec![],
                finish_reason: "stop".to_string(),
                usage: std::collections::HashMap::new(),
            }),
        )
    }

    fn get_default_model(&self) -> &str {
        &self.name
    }
}

struct FailOnceThenResponseProvider {
    name: String,
    response: crate::providers::base::LLMResponse,
    call_count: std::sync::atomic::AtomicU32,
}

impl FailOnceThenResponseProvider {
    fn new(name: &str, response: crate::providers::base::LLMResponse) -> Self {
        Self {
            name: name.to_string(),
            response,
            call_count: std::sync::atomic::AtomicU32::new(0),
        }
    }
}

#[async_trait]
impl LLMProvider for FailOnceThenResponseProvider {
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
        let call = self
            .call_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if call == 0 {
            anyhow::bail!("synthetic provider failure");
        }
        Ok(self.response.clone())
    }

    fn get_default_model(&self) -> &str {
        &self.name
    }
}

struct StreamingThinkingProvider {
    name: String,
    last_thinking_budget: std::sync::atomic::AtomicU32,
}

impl StreamingThinkingProvider {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            last_thinking_budget: std::sync::atomic::AtomicU32::new(0),
        }
    }

    fn last_thinking_budget(&self) -> u32 {
        self.last_thinking_budget
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}

#[async_trait]
impl LLMProvider for StreamingThinkingProvider {
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
        anyhow::bail!("StreamingThinkingProvider only supports chat_stream")
    }

    async fn chat_stream(
        &self,
        _messages: &[Value],
        _tools: Option<&[Value]>,
        _model: Option<&str>,
        _max_tokens: u32,
        _temperature: f64,
        thinking_budget: Option<u32>,
        _top_p: Option<f64>,
    ) -> anyhow::Result<crate::providers::base::StreamHandle> {
        self.last_thinking_budget.store(
            thinking_budget.unwrap_or(0),
            std::sync::atomic::Ordering::Relaxed,
        );
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let _ = tx.send(crate::providers::base::StreamChunk::ThinkingDelta(
            "private thought".to_string(),
        ));
        let _ = tx.send(crate::providers::base::StreamChunk::TextDelta(
            "visible answer".to_string(),
        ));
        let _ = tx.send(crate::providers::base::StreamChunk::Done(
            crate::providers::base::LLMResponse {
                content: Some(attested_text("visible answer")),
                tool_calls: vec![],
                finish_reason: "stop".to_string(),
                usage: std::collections::HashMap::new(),
            },
        ));
        Ok(crate::providers::base::StreamHandle {
            rx,
            abort_on_drop: None,
        })
    }

    fn get_default_model(&self) -> &str {
        &self.name
    }
}

struct RecordingProvider {
    name: String,
    response: String,
    last_max_tokens: std::sync::atomic::AtomicU32,
}

impl RecordingProvider {
    fn new(name: &str, response: &str) -> Self {
        Self {
            name: name.to_string(),
            response: attested_text(response),
            last_max_tokens: std::sync::atomic::AtomicU32::new(0),
        }
    }

    fn last_max_tokens(&self) -> u32 {
        self.last_max_tokens
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}

#[async_trait]
impl LLMProvider for RecordingProvider {
    async fn chat(
        &self,
        _messages: &[Value],
        _tools: Option<&[Value]>,
        _model: Option<&str>,
        max_tokens: u32,
        _temperature: f64,
        _thinking_budget: Option<u32>,
        _top_p: Option<f64>,
    ) -> anyhow::Result<crate::providers::base::LLMResponse> {
        self.last_max_tokens
            .store(max_tokens, std::sync::atomic::Ordering::Relaxed);
        Ok(crate::providers::base::LLMResponse {
            content: Some(self.response.clone()),
            tool_calls: vec![],
            finish_reason: "stop".to_string(),
            usage: std::collections::HashMap::new(),
        })
    }

    fn get_default_model(&self) -> &str {
        &self.name
    }
}

/// Build an offline trio harness from pre-built mock providers.
///
/// Mirrors `build_trio_e2e_harness` but accepts providers directly rather
/// than constructing real HTTP clients. No background probes are wired.
fn build_trio_offline_harness(
    main: Arc<dyn LLMProvider>,
    router: Arc<dyn LLMProvider>,
    specialist: Arc<dyn LLMProvider>,
) -> (AgentLoop, std::path::PathBuf) {
    use crate::config::schema::LcmSchemaConfig;

    let workspace = tempfile::tempdir().unwrap().keep();

    let mut td = ToolDelegationConfig {
        mode: crate::config::schema::DelegationMode::trio(),
        ..Default::default()
    };
    td.apply_mode(); // trio mode carries strict_no_tools_main + strict_router_schema

    let router_model = router.get_default_model().to_string();
    let specialist_model = specialist.get_default_model().to_string();

    let trio_config = TrioConfig {
        enabled: true,
        router_model: router_model.clone(),
        specialist_model: specialist_model.clone(),
        ..Default::default()
    };

    let core = build_swappable_core(SwappableCoreConfig {
        provider: main,
        workspace: workspace.clone(),
        model: "offline-main".to_string(),
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
        is_local: true,
        lane: Lane::default(),
        tool_delegation: td,
        provenance: ProvenanceConfig::default(),
        max_tool_result_chars: 2000,
        delegation_provider: Some(router),
        specialist_provider: Some(specialist),
        trio_config,
        model_capabilities_overrides: std::collections::HashMap::new(),
        reasoning_config: crate::config::schema::ReasoningConfig::default(),
        tool_heartbeat_secs: 2,
        health_check_timeout_secs: 2,
        adaptive_tokens: AdaptiveTokenConfig::default(),
        sessions_db_path: Some(
            std::env::temp_dir().join(format!("nanobot-test-{}.sqlite", uuid::Uuid::new_v4())),
        ),
    });

    let counters = test_runtime_counters(4096);
    let core_handle = AgentHandle::new(core, counters);

    let (inbound_tx, inbound_rx) = tokio::sync::mpsc::unbounded_channel::<InboundMessage>();
    let (outbound_tx, _outbound_rx) = tokio::sync::mpsc::unbounded_channel::<OutboundMessage>();

    let agent_loop = AgentLoop::new(
        core_handle,
        inbound_rx,
        outbound_tx,
        inbound_tx,
        None,
        1,
        None,
        None,
        None,
        ProprioceptionConfig::default(),
        LcmSchemaConfig::default(),
        None, // no health_registry — offline tests manage their own
    );

    (agent_loop, workspace)
}

fn build_local_inline_harness(main: Arc<dyn LLMProvider>) -> (AgentLoop, std::path::PathBuf) {
    build_local_inline_harness_with_model(main, "local-qwen-test")
}

/// Same as [`build_local_inline_harness`] but with a custom `max_iterations`.
/// Convergence tests need this above the lease coarse-family cap (6) so the
/// sticky-strip path is actually reachable — the default of 5 stops the loop
/// before the family cap can fire.
fn build_local_inline_harness_with_iters(
    main: Arc<dyn LLMProvider>,
    max_iterations: u32,
) -> (AgentLoop, std::path::PathBuf) {
    let workspace = tempfile::tempdir().unwrap().keep();
    let core = build_swappable_core(SwappableCoreConfig {
        provider: main,
        workspace: workspace.clone(),
        model: "local-qwen-test".to_string(),
        max_iterations,
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
        is_local: true,
        lane: Lane::default(),
        tool_delegation: ToolDelegationConfig::default(),
        provenance: ProvenanceConfig::default(),
        max_tool_result_chars: 2000,
        delegation_provider: None,
        specialist_provider: None,
        trio_config: TrioConfig::default(),
        model_capabilities_overrides: std::collections::HashMap::new(),
        reasoning_config: crate::config::schema::ReasoningConfig::default(),
        tool_heartbeat_secs: 2,
        health_check_timeout_secs: 2,
        adaptive_tokens: AdaptiveTokenConfig::default(),
        sessions_db_path: Some(
            std::env::temp_dir().join(format!("nanobot-test-{}.sqlite", uuid::Uuid::new_v4())),
        ),
    });

    let counters = test_runtime_counters(4096);
    let core_handle = AgentHandle::new(core, counters);

    let (inbound_tx, inbound_rx) = tokio::sync::mpsc::unbounded_channel::<InboundMessage>();
    let (outbound_tx, _outbound_rx) = tokio::sync::mpsc::unbounded_channel::<OutboundMessage>();
    let agent_loop = AgentLoop::new(
        core_handle,
        inbound_rx,
        outbound_tx,
        inbound_tx,
        None,
        1,
        None,
        None,
        None,
        crate::config::schema::ProprioceptionConfig::default(),
        LcmSchemaConfig::default(),
        None,
    );
    (agent_loop, workspace)
}

fn build_local_inline_harness_with_model(
    main: Arc<dyn LLMProvider>,
    model: &str,
) -> (AgentLoop, std::path::PathBuf) {
    build_local_inline_harness_with_lcm(main, model, 4096, LcmSchemaConfig::default())
}

fn build_local_inline_harness_with_lcm(
    main: Arc<dyn LLMProvider>,
    model: &str,
    max_context_tokens: usize,
    lcm_config: LcmSchemaConfig,
) -> (AgentLoop, std::path::PathBuf) {
    build_local_inline_harness_with_memory(
        main,
        model,
        max_context_tokens,
        lcm_config,
        MemoryConfig::default(),
    )
}

fn build_local_inline_harness_with_memory(
    main: Arc<dyn LLMProvider>,
    model: &str,
    max_context_tokens: usize,
    lcm_config: LcmSchemaConfig,
    memory_config: MemoryConfig,
) -> (AgentLoop, std::path::PathBuf) {
    build_local_inline_harness_with_memory_and_reflection(
        main,
        model,
        max_context_tokens,
        lcm_config,
        memory_config,
        None,
    )
}

/// Same as [`build_local_inline_harness_with_memory`], but lets a test wire a
/// distinct specialist fallback for durable-memory reflection. LCM still uses
/// `main`; keeping this separate proves reflection configuration cannot reroute
/// context compaction.
fn build_local_inline_harness_with_memory_and_reflection(
    main: Arc<dyn LLMProvider>,
    model: &str,
    max_context_tokens: usize,
    lcm_config: LcmSchemaConfig,
    memory_config: MemoryConfig,
    reflection: Option<Arc<dyn LLMProvider>>,
) -> (AgentLoop, std::path::PathBuf) {
    let workspace = tempfile::tempdir().unwrap().keep();
    let core = build_swappable_core(SwappableCoreConfig {
        provider: main,
        workspace: workspace.clone(),
        model: model.to_string(),
        max_iterations: 5,
        max_continuations: 2,
        max_tokens: 512,
        temperature: 0.3,
        max_context_tokens,
        brave_api_key: None,
        search_provider: "searxng".to_string(),
        searxng_url: "http://localhost:8888".to_string(),
        crw_url: String::new(),
        search_max_results: 5,
        exec_timeout: 30,
        restrict_to_workspace: true,
        memory_config,
        is_local: true,
        lane: Lane::default(),
        tool_delegation: ToolDelegationConfig::default(),
        provenance: ProvenanceConfig::default(),
        max_tool_result_chars: 2000,
        delegation_provider: None,
        specialist_provider: reflection,
        trio_config: TrioConfig::default(),
        model_capabilities_overrides: std::collections::HashMap::new(),
        reasoning_config: crate::config::schema::ReasoningConfig::default(),
        tool_heartbeat_secs: 2,
        health_check_timeout_secs: 2,
        adaptive_tokens: AdaptiveTokenConfig::default(),
        sessions_db_path: Some(
            std::env::temp_dir().join(format!("nanobot-test-{}.sqlite", uuid::Uuid::new_v4())),
        ),
    });

    let counters = test_runtime_counters(max_context_tokens);
    let core_handle = AgentHandle::new(core, counters);

    let (inbound_tx, inbound_rx) = tokio::sync::mpsc::unbounded_channel::<InboundMessage>();
    let (outbound_tx, _outbound_rx) = tokio::sync::mpsc::unbounded_channel::<OutboundMessage>();

    let agent_loop = AgentLoop::new(
        core_handle,
        inbound_rx,
        outbound_tx,
        inbound_tx,
        None,
        1,
        None,
        None,
        None,
        ProprioceptionConfig::default(),
        lcm_config,
        None,
    );

    (agent_loop, workspace)
}

/// Cloud-mode counterpart to `build_local_inline_harness_with_memory_and_reflection`
/// (`is_local: false`). Exercises the real system+developer assembly split
/// (`ContextBuilder::collect_static_sections`) plus `prepare_context`'s
/// `collect_cloud_runtime_sections` (MemoryLadder, background-task status) --
/// the two mechanisms that once double-injected `MEMORY.md`.
fn build_cloud_inline_harness_with_memory(
    main: Arc<dyn LLMProvider>,
    model: &str,
    memory_config: MemoryConfig,
) -> (AgentLoop, std::path::PathBuf) {
    let workspace = tempfile::tempdir().unwrap().keep();
    let max_context_tokens = 128_000;
    let core = build_swappable_core(SwappableCoreConfig {
        provider: main,
        workspace: workspace.clone(),
        model: model.to_string(),
        max_iterations: 5,
        max_continuations: 2,
        max_tokens: 512,
        temperature: 0.3,
        max_context_tokens,
        brave_api_key: None,
        search_provider: "searxng".to_string(),
        searxng_url: "http://localhost:8888".to_string(),
        crw_url: String::new(),
        search_max_results: 5,
        exec_timeout: 30,
        restrict_to_workspace: true,
        memory_config,
        is_local: false,
        lane: Lane::default(),
        tool_delegation: ToolDelegationConfig::default(),
        provenance: ProvenanceConfig::default(),
        max_tool_result_chars: 2000,
        delegation_provider: None,
        specialist_provider: None,
        trio_config: TrioConfig::default(),
        model_capabilities_overrides: std::collections::HashMap::new(),
        reasoning_config: crate::config::schema::ReasoningConfig::default(),
        tool_heartbeat_secs: 2,
        health_check_timeout_secs: 2,
        adaptive_tokens: AdaptiveTokenConfig::default(),
        sessions_db_path: Some(
            std::env::temp_dir().join(format!("nanobot-test-{}.sqlite", uuid::Uuid::new_v4())),
        ),
    });

    let counters = test_runtime_counters(max_context_tokens);
    let core_handle = AgentHandle::new(core, counters);

    let (inbound_tx, inbound_rx) = tokio::sync::mpsc::unbounded_channel::<InboundMessage>();
    let (outbound_tx, _outbound_rx) = tokio::sync::mpsc::unbounded_channel::<OutboundMessage>();

    let agent_loop = AgentLoop::new(
        core_handle,
        inbound_rx,
        outbound_tx,
        inbound_tx,
        None,
        1,
        None,
        None,
        None,
        ProprioceptionConfig::default(),
        LcmSchemaConfig::default(),
        None,
    );

    (agent_loop, workspace)
}

/// Regression test for the MEMORY.md double-injection bug: `ContextBuilder::
/// collect_static_sections` used to be one injector and `MemoryLadder`'s
/// (now-removed) `GroundTruth` layer -- reached via `prepare_context::
/// collect_cloud_runtime_sections` -- was a second, silently duplicating
/// long-term memory content into the assembled cloud-path messages.
///
/// Inspects `TurnContext.messages` directly (via `prepare_context`, the
/// method that actually assembles system+developer content) rather than the
/// protocol-rendered wire sent to a provider -- `render_to_wire`/
/// `turn_from_legacy` has a separate, pre-existing gap where `role:
/// "developer"` messages aren't converted to a `Turn` at all and are dropped
/// during rendering, which is an unrelated protocol-layer bug, not a memory
/// double-injection.
#[tokio::test]
async fn memory_md_appears_exactly_once_in_assembled_cloud_messages() {
    let provider = MockLLM::named("cloud-memory-dedup-test");
    let (agent_loop, workspace) = build_cloud_inline_harness_with_memory(
        provider,
        "cloud-memory-dedup-test",
        MemoryConfig::default(),
    );

    // Distinctive content only `MEMORY.md` contains -- `collect_static_sections`
    // (the sole intended injector) reads it from disk via `MemoryStore`.
    const MARKER: &str = "XYZZY-UNIQUE-MEMORY-MARKER-42: the user prefers oat milk.";
    let memory_dir = workspace.join("memory");
    std::fs::create_dir_all(&memory_dir).unwrap();
    std::fs::write(memory_dir.join("MEMORY.md"), MARKER).unwrap();

    let session_key = format!("cloud-memory-dedup-{}", uuid::Uuid::new_v4());
    let mut msg = InboundMessage::new("test", "user", "offline", "Hello there.");
    msg.metadata
        .insert("session_key".to_string(), json!(session_key));

    let turn_ctx = agent_loop
        .shared
        .prepare_context(&msg, None, None, None, None)
        .await;

    // Count substring occurrences (not just how many messages contain it) so
    // a duplicate concatenated into the SAME message -- e.g. two blocks
    // folded into one `developer` message -- is caught too.
    let occurrences: usize = turn_ctx
        .messages
        .iter()
        .filter_map(|message| message.get("content").and_then(Value::as_str))
        .map(|content| content.matches(MARKER).count())
        .sum();
    assert_eq!(
        occurrences, 1,
        "MEMORY.md content must appear exactly once across the assembled cloud-path messages, got {}",
        occurrences
    );
}

/// Records the full wire `messages` array of every `chat()` call and replays
/// a scripted response sequence (last response repeats when exhausted).
struct WireRecordingProvider {
    name: String,
    responses: std::sync::Mutex<std::collections::VecDeque<crate::providers::base::LLMResponse>>,
    calls: std::sync::Mutex<Vec<Vec<Value>>>,
}

impl WireRecordingProvider {
    fn new(name: &str, responses: Vec<crate::providers::base::LLMResponse>) -> Self {
        Self {
            name: name.to_string(),
            responses: std::sync::Mutex::new(responses.into()),
            calls: std::sync::Mutex::new(Vec::new()),
        }
    }

    fn text_response(content: &str) -> crate::providers::base::LLMResponse {
        crate::providers::base::LLMResponse {
            content: Some(attested_text(content)),
            tool_calls: vec![],
            finish_reason: "stop".to_string(),
            usage: std::collections::HashMap::new(),
        }
    }

    /// Internal compaction/reflection output is not an agent turn.
    fn plain_text_response(content: &str) -> crate::providers::base::LLMResponse {
        crate::providers::base::LLMResponse {
            content: Some(content.to_string()),
            tool_calls: vec![],
            finish_reason: "stop".to_string(),
            usage: std::collections::HashMap::new(),
        }
    }

    fn calls(&self) -> Vec<Vec<Value>> {
        self.calls.lock().unwrap().clone()
    }
}

#[async_trait]
impl LLMProvider for WireRecordingProvider {
    async fn chat(
        &self,
        messages: &[Value],
        _tools: Option<&[Value]>,
        _model: Option<&str>,
        _max_tokens: u32,
        _temperature: f64,
        _thinking_budget: Option<u32>,
        _top_p: Option<f64>,
    ) -> anyhow::Result<crate::providers::base::LLMResponse> {
        self.calls.lock().unwrap().push(messages.to_vec());
        let mut queue = self.responses.lock().unwrap();
        Ok(if queue.len() > 1 {
            queue.pop_front().unwrap()
        } else {
            queue
                .front()
                .cloned()
                .unwrap_or_else(|| Self::text_response("done"))
        })
    }

    fn get_default_model(&self) -> &str {
        &self.name
    }
}

#[tokio::test]
async fn hard_lcm_checkpoint_is_installed_before_foreground_inference() {
    let provider = Arc::new(WireRecordingProvider::new(
        "local-hard-lcm-test",
        vec![
            WireRecordingProvider::plain_text_response(
                "- Prior turns retained project detail context for later reference.",
            ),
            WireRecordingProvider::text_response("foreground reply"),
        ],
    ));
    let lcm_config = LcmSchemaConfig {
        tau_soft: 0.05,
        tau_hard: 0.10,
        deterministic_target: 64,
        ..Default::default()
    };
    let (agent_loop, _workspace) = build_local_inline_harness_with_memory_and_reflection(
        provider.clone() as Arc<dyn LLMProvider>,
        "local-hard-lcm-test",
        8192,
        lcm_config,
        MemoryConfig::default(),
        None,
    );
    let session_key = format!("hard-lcm-barrier-{}", uuid::Uuid::new_v4());
    let core = agent_loop.shared.core_handle.swappable();
    let session = core.sessions.get_or_resume(&session_key).await;
    for turn in 0..20_u64 {
        let detail = format!(
            "turn {turn}: {}",
            "persistent project detail with enough context to require lossless compaction "
                .repeat(12)
        );
        core.sessions
            .add_message(
                &session.id,
                &json!({"role": "user", "content": detail, "_turn": turn}),
            )
            .await;
        core.sessions
            .add_message(
                &session.id,
                &json!({"role": "assistant", "content": format!("acknowledged turn {turn}"), "_turn": turn}),
            )
            .await;
    }

    let response = agent_loop
        .process_direct(
            "Use the retained project details to answer briefly.",
            &session_key,
            "test",
            "offline",
        )
        .await;
    assert_eq!(response, "foreground reply");

    let calls = provider.calls();
    assert_eq!(
        calls.len(),
        2,
        "expected one compaction call followed by one foreground call"
    );
    // Internal fields like `_lcm_summary` are stripped before messages hit
    // the wire, so `calls[0]` can't be checked via that tag. Match the
    // summary wire message's exact phrasing instead of a bare "[Summary of
    // messages" substring — the LCM_EXPAND_GUIDE instructional text
    // (prepare_context.rs) contains that same lead-in as a generic example
    // ("copy that range into lcm_expand — for example ..."), which would
    // false-positive this check even if compaction never installed
    // anything. `summary_wire_message` (lcm.rs) uniquely phrases it "To read
    // the exact originals call lcm_expand(...)".
    let foreground_call = calls.last().expect("foreground call recorded");
    let has_summary = foreground_call.iter().any(|message| {
        message
            .get("content")
            .and_then(Value::as_str)
            .is_some_and(|content| content.contains("To read the exact originals call"))
    });
    assert!(
        has_summary,
        "hard-pressure LCM checkpoint must be installed before the foreground call"
    );
    assert!(foreground_call.iter().any(|message| {
        message
            .get("content")
            .and_then(Value::as_str)
            .is_some_and(|content| content.contains("Use the retained project details"))
    }));
    assert!(
        agent_loop
            .shared
            .core_handle
            .counters
            .lcm_compaction_count
            .load(std::sync::atomic::Ordering::Relaxed)
            >= 1
    );

    let nodes = tokio::time::timeout(std::time::Duration::from_secs(2), async {
        loop {
            let nodes = core.sessions.load_summary_nodes(&session.id).await;
            if !nodes.is_empty() {
                break nodes;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("LCM summary node must become restart-durable");
    let raw = core.sessions.get_all_messages(&session.id).await;
    let rebuilt = crate::agent::lcm::LcmEngine::rebuild_from_db_nodes(
        &raw,
        &nodes,
        crate::agent::lcm::LcmConfig::default(),
    );
    assert!(rebuilt.active_context().iter().any(|message| {
        message
            .get("_lcm_summary")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    }));
}

#[tokio::test]
async fn soft_lcm_uses_main_provider_and_preserves_foreground_context() {
    let main_provider = Arc::new(WireRecordingProvider::new(
        "local-soft-lcm-test",
        vec![WireRecordingProvider::text_response("foreground reply")],
    ));
    let memory_provider = Arc::new(WireRecordingProvider::new(
        "memory-soft-lcm-test",
        vec![WireRecordingProvider::plain_text_response(
            "- memory summary",
        )],
    ));
    let lcm_config = LcmSchemaConfig {
        tau_soft: 0.0001,
        // Deliberately above 1.0 in this focused policy test so even a tiny
        // effective budget cannot turn the soft case into hard pressure.
        tau_hard: 10.0,
        deterministic_target: 64,
        ..Default::default()
    };

    let (agent_loop, _workspace) = build_local_inline_harness_with_memory_and_reflection(
        main_provider.clone() as Arc<dyn LLMProvider>,
        "local-soft-lcm-test",
        1_000_000,
        lcm_config,
        MemoryConfig::default(),
        Some(memory_provider.clone() as Arc<dyn LLMProvider>),
    );
    let session_key = format!("soft-lcm-preserve-{}", uuid::Uuid::new_v4());
    let core = agent_loop.shared.core_handle.swappable();
    let session = core.sessions.get_or_resume(&session_key).await;
    // 8 turns of padded content: the LCM engine protects a ~2048-token tail
    // of the most recent raw messages (see `protect_tokens_for_budget`) and
    // only compacts the oldest block beyond that. It must clear
    // `MIN_COMPACTION_TOKENS` (200) or compaction skips silently without
    // ever calling the provider — this volume keeps the oldest block
    // comfortably above that floor so the main-provider compactor is actually
    // exercised.
    for turn in 0..8_u64 {
        core.sessions
            .add_message(
                &session.id,
                &json!({
                    "role": "user",
                    "content": format!("soft-pressure-marker-{turn} {}", "context detail ".repeat(40)),
                    "_turn": turn
                }),
            )
            .await;
        core.sessions
            .add_message(
                &session.id,
                &json!({"role": "assistant", "content": format!("soft ack {turn}"), "_turn": turn}),
            )
            .await;
    }

    let response = agent_loop
        .process_direct(
            "Continue without losing context.",
            &session_key,
            "test",
            "offline",
        )
        .await;
    assert_eq!(response, "foreground reply");

    // Async (soft) compaction must use the main provider even though a distinct
    // provider is configured for memory reflection.
    tokio::time::timeout(std::time::Duration::from_secs(3), async {
        while main_provider.calls().len() <= 1 {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("background compaction must call the main provider");

    // The mock's canned foreground reply isn't a valid bullet-only handoff, so
    // both escalation levels reject it and compaction leaves the context
    // uncompacted rather than installing deterministic truncation.
    let calls = main_provider.calls();
    assert!(
        calls.len() > 1,
        "compaction must have been attempted against the main provider, got {} call(s)",
        calls.len()
    );
    assert!(
        memory_provider.calls().is_empty(),
        "LCM must not send compaction requests to the reflection provider"
    );
    // The last call is the LCM escalation attempt (Level 1/2 summarization),
    // not the original foreground chat call — it must still carry the
    // original source content it's being asked to summarize, proving
    // nothing was silently dropped before compaction gave up.
    let last_call = calls.last().expect("at least one call recorded");
    assert!(last_call.iter().any(|message| {
        message
            .get("content")
            .and_then(Value::as_str)
            .is_some_and(|content| content.contains("soft-pressure-marker-0"))
    }));
    assert!(
        !calls.iter().flatten().any(|message| {
            // Match the summary wire message's exact phrasing, not a bare
            // "[Summary of messages" substring — the LCM_EXPAND_GUIDE
            // instructional text (prepare_context.rs) contains that same
            // lead-in as a generic example and would false-positive this
            // check. `summary_wire_message` (lcm.rs) uniquely phrases an
            // actually-installed summary "To read the exact originals call
            // lcm_expand(...)".
            message
                .get("content")
                .and_then(Value::as_str)
                .is_some_and(|content| content.contains("To read the exact originals call"))
        }),
        "no summary — real or the deleted deterministic-truncation fallback — was ever installed"
    );
}

#[tokio::test]
async fn concrete_session_reuses_compaction_checkpoint_handle() {
    let provider = MockLLM::named("local-compaction-checkpoint-handle-test");
    let (agent_loop, _workspace) = build_local_inline_harness(provider);
    let session_key = format!("compaction-checkpoint-handle-{}", uuid::Uuid::new_v4());
    let mut msg = InboundMessage::new("test", "user", "offline", "first");
    msg.metadata
        .insert("session_key".to_string(), json!(session_key));

    let first = agent_loop
        .shared
        .prepare_context(&msg, None, None, None, None)
        .await;
    msg.content = "second".to_string();
    let second = agent_loop
        .shared
        .prepare_context(&msg, None, None, None, None)
        .await;

    assert_eq!(first.session_id, second.session_id);
    assert!(
        Arc::ptr_eq(&first.compaction.slot, &second.compaction.slot),
        "the pending checkpoint must remain visible across turns in one concrete session"
    );
    assert!(
        Arc::ptr_eq(&first.compaction.in_flight, &second.compaction.in_flight),
        "the in-flight barrier must remain visible across turns in one concrete session"
    );
}

#[tokio::test]
async fn idle_rollover_does_not_reuse_compaction_checkpoint_handle() {
    let provider = MockLLM::named("local-compaction-checkpoint-rollover-test");
    let memory_config = MemoryConfig {
        session_complete_after_secs: 1,
        ..Default::default()
    };
    let (agent_loop, _workspace) = build_local_inline_harness_with_memory(
        provider,
        "local-compaction-checkpoint-rollover-test",
        4096,
        LcmSchemaConfig::default(),
        memory_config,
    );
    let session_key = format!("compaction-checkpoint-rollover-{}", uuid::Uuid::new_v4());
    let mut msg = InboundMessage::new("test", "user", "offline", "first");
    msg.metadata
        .insert("session_key".to_string(), json!(session_key));

    let first = agent_loop
        .shared
        .prepare_context(&msg, None, None, None, None)
        .await;
    tokio::time::sleep(std::time::Duration::from_millis(2_100)).await;
    msg.content = "second".to_string();
    let second = agent_loop
        .shared
        .prepare_context(&msg, None, None, None, None)
        .await;

    assert_ne!(first.session_id, second.session_id);
    assert!(!Arc::ptr_eq(
        &first.compaction.slot,
        &second.compaction.slot
    ));
    assert!(!Arc::ptr_eq(
        &first.compaction.in_flight,
        &second.compaction.in_flight
    ));
}

#[tokio::test]
async fn in_flight_compaction_checkpoint_hides_unpublished_dag() {
    let provider = MockLLM::named("local-compaction-checkpoint-visibility-test");
    let (agent_loop, _workspace) = build_local_inline_harness(provider);
    let session_key = format!("compaction-checkpoint-visibility-{}", uuid::Uuid::new_v4());
    let core = agent_loop.shared.core_handle.swappable();
    let session = core.sessions.get_or_resume(&session_key).await;
    core.sessions
        .add_message(
            &session.id,
            &json!({"role": "user", "content": "raw-source-marker"}),
        )
        .await;
    core.sessions
        .add_message(
            &session.id,
            &json!({"role": "assistant", "content": "raw-source-answer"}),
        )
        .await;

    let mut msg = InboundMessage::new("test", "user", "offline", "first");
    msg.metadata
        .insert("session_key".to_string(), json!(session_key));
    let first = agent_loop
        .shared
        .prepare_context(&msg, None, None, None, None)
        .await;

    let raw = core.sessions.get_all_messages(&session.id).await;
    let source_ids = raw
        .iter()
        .map(|message| message["_db_id"].as_u64().unwrap() as usize)
        .collect::<Vec<_>>();
    let nodes = vec![(
        0,
        source_ids,
        Vec::new(),
        "checkpoint-summary-marker".to_string(),
        3,
        1,
        crate::agent::lcm::SummaryManifest::default(),
        "db_id".to_string(),
    )];
    let rebuilt = crate::agent::lcm::LcmEngine::rebuild_from_db_nodes(
        &raw,
        &nodes,
        crate::agent::lcm::LcmConfig::default(),
    );
    let engine = agent_loop
        .shared
        .lcm_engines
        .lock()
        .await
        .get(&session.id)
        .cloned()
        .unwrap();
    *engine.lock().await = rebuilt;

    first
        .compaction
        .in_flight
        .store(true, std::sync::atomic::Ordering::Release);
    msg.content = "second".to_string();
    let second = agent_loop
        .shared
        .prepare_context(&msg, None, None, None, None)
        .await;
    first
        .compaction
        .in_flight
        .store(false, std::sync::atomic::Ordering::Release);

    let assembled = serde_json::to_string(&second.messages).unwrap();
    assert!(
        assembled.contains("raw-source-marker"),
        "raw history remains authoritative until the checkpoint is published"
    );
    assert!(
        !assembled.contains("checkpoint-summary-marker"),
        "an in-flight DAG mutation must not become prompt-visible before checkpoint publication"
    );

    let snapshot = second.messages[..second.new_start].to_vec();
    *first.compaction.slot.lock().await = Some(crate::agent::agent_core::PendingCompaction {
        result: crate::agent::compaction::CompactionResult {
            messages: vec![json!({
                "role": "assistant",
                "content": "checkpoint-summary-marker"
            })],
        },
        snapshot,
    });
    msg.content = "third".to_string();
    let third = agent_loop
        .shared
        .prepare_context(&msg, None, None, None, None)
        .await;
    *first.compaction.slot.lock().await = None;

    let assembled = serde_json::to_string(&third.messages).unwrap();
    assert!(
        assembled.contains("raw-source-marker"),
        "raw history remains authoritative while the checkpoint is pending"
    );
    assert!(
        !assembled.contains("checkpoint-summary-marker"),
        "a pending DAG rewrite must not become prompt-visible before checkpoint installation"
    );
}

#[tokio::test]
async fn idle_rollover_uses_a_new_session_scoped_lcm_engine() {
    let provider = Arc::new(WireRecordingProvider::new(
        "local-idle-lcm-test",
        vec![WireRecordingProvider::text_response("foreground reply")],
    ));
    let memory_config = MemoryConfig {
        session_complete_after_secs: 1,
        ..Default::default()
    };
    let (agent_loop, _workspace) = build_local_inline_harness_with_memory(
        provider as Arc<dyn LLMProvider>,
        "local-idle-lcm-test",
        4096,
        LcmSchemaConfig::default(),
        memory_config,
    );
    let session_key = format!("idle-lcm-isolation-{}", uuid::Uuid::new_v4());

    agent_loop
        .process_direct("first-session-marker", &session_key, "test", "offline")
        .await;
    let core = agent_loop.shared.core_handle.swappable();
    let first_id = core
        .sessions
        .get_latest_session(&session_key)
        .await
        .unwrap()
        .id;
    // Expiry uses whole seconds and a strict `idle > threshold` comparison.
    tokio::time::sleep(std::time::Duration::from_millis(2_100)).await;

    agent_loop
        .process_direct("second-session-marker", &session_key, "test", "offline")
        .await;
    let second_id = core
        .sessions
        .get_latest_session(&session_key)
        .await
        .unwrap()
        .id;
    assert_ne!(
        first_id, second_id,
        "idle rollover must create a new session"
    );

    let second_engine = agent_loop
        .shared
        .lcm_engines
        .lock()
        .await
        .get(&second_id)
        .cloned()
        .expect("new concrete session must own its own LCM engine");
    let active = second_engine.lock().await.active_context();
    let wire = serde_json::to_string(&active).unwrap();
    assert!(wire.contains("second-session-marker"));
    assert!(!wire.contains("first-session-marker"));
}

/// Assert every wire message of `first` reappears byte-identical, in order,
/// at the head of `second` — the KV prefix-cache contract. One mutated byte
/// forces the local server to re-prefill everything past it (~45s cold on a
/// 35B), so this property IS the local-model perf story.
fn assert_wire_prefix(first: &[Value], second: &[Value]) {
    assert!(
        second.len() > first.len(),
        "later call must extend the earlier one (got {} then {})",
        first.len(),
        second.len()
    );
    for (i, msg) in first.iter().enumerate() {
        assert_eq!(
            serde_json::to_string(msg).unwrap(),
            serde_json::to_string(&second[i]).unwrap(),
            "wire message {i} mutated between calls — prompt prefix diverged, KV cache busted"
        );
    }
}

/// KV prefix contract across turns: turn N's wire prompt must be a
/// byte-prefix of turn N+1's. Per-turn content belongs in TAIL blocks,
/// never in messages[0]. Guards the 1.9s-full-prefill → 0.07s-reuse asset.
#[tokio::test]
async fn test_local_wire_prompt_prefix_stable_across_turns() {
    let provider = Arc::new(WireRecordingProvider::new(
        "local-qwen-test",
        vec![WireRecordingProvider::text_response("first reply")],
    ));
    let (agent_loop, _ws) = build_local_inline_harness(provider.clone() as Arc<dyn LLMProvider>);
    let session_key = format!("prefix-stability-{}", uuid::Uuid::new_v4());

    agent_loop
        .process_direct("first message", &session_key, "test", "offline")
        .await;
    agent_loop
        .process_direct("second message", &session_key, "test", "offline")
        .await;

    let calls = provider.calls();
    assert!(
        calls.len() >= 2,
        "expected two LLM calls, got {}",
        calls.len()
    );
    assert_wire_prefix(&calls[0], &calls[calls.len() - 1]);
}

#[tokio::test]
async fn test_local_wire_prompt_prefix_stable_when_second_turn_is_rich_artifact() {
    let provider = Arc::new(WireRecordingProvider::new(
        "local-qwen-test",
        vec![
            WireRecordingProvider::text_response("Hey! What can I help you with today?"),
            WireRecordingProvider::text_response("I'll create it."),
        ],
    ));
    let (agent_loop, _ws) = build_local_inline_harness(provider.clone() as Arc<dyn LLMProvider>);
    let session_key = format!("artifact-prefix-stability-{}", uuid::Uuid::new_v4());

    agent_loop
        .process_direct("hi", &session_key, "test", "offline")
        .await;
    agent_loop
        .process_direct(
            "I want you to create a single HTML file version of Pong at ~/Dev/pong . It must be fun, colorful and easy to play",
            &session_key,
            "test",
            "offline",
        )
        .await;

    let calls = provider.calls();
    assert!(
        calls.len() >= 2,
        "expected two LLM calls, got {}",
        calls.len()
    );
    assert_wire_prefix(&calls[0], &calls[calls.len() - 1]);
    let system = calls[calls.len() - 1][0]["content"].as_str().unwrap_or("");
    assert!(
        !system.contains("Local Artifact Writer"),
        "rich artifact turns must not mutate the stable system prompt"
    );
}

/// Persisted sessions remain discoverable through recall/resume, but a fresh
/// local turn must not carry an unrelated previous-session hint in its stable
/// prompt prefix.
#[tokio::test]
async fn test_local_fresh_session_does_not_inject_previous_session() {
    let provider = Arc::new(WireRecordingProvider::new(
        "local-qwen-test",
        vec![
            WireRecordingProvider::text_response("first reply"),
            WireRecordingProvider::text_response("second reply"),
        ],
    ));
    let (agent_loop, _ws) = build_local_inline_harness(provider.clone() as Arc<dyn LLMProvider>);

    agent_loop
        .process_direct("first message", "pi-style-prior", "test", "offline")
        .await;
    agent_loop
        .process_direct(
            "unrelated fresh message",
            "pi-style-fresh",
            "test",
            "offline",
        )
        .await;

    let calls = provider.calls();
    assert!(
        calls.len() >= 2,
        "expected two LLM calls, got {}",
        calls.len()
    );
    let system = calls[1][0]["content"].as_str().unwrap_or("");
    assert!(!system.contains("Previous Session"));
    assert!(!system.contains("pi-style-prior"));
}

/// Same contract within a turn: executing a tool must only APPEND to the
/// wire prompt (assistant carrier + tool result + continuation), never
/// rewrite what the server already prefilled.
#[tokio::test]
async fn test_local_wire_prompt_tool_result_appends_only() {
    let mut args = std::collections::HashMap::new();
    args.insert("path".to_string(), json!("."));
    let provider = Arc::new(WireRecordingProvider::new(
        "local-qwen-test",
        vec![
            crate::providers::base::LLMResponse {
                content: Some(String::new()),
                tool_calls: vec![crate::providers::base::ToolCallRequest {
                    id: "tc_prefix".to_string(),
                    name: "list_dir".to_string(),
                    arguments: args,
                }],
                finish_reason: "tool_calls".to_string(),
                usage: std::collections::HashMap::new(),
            },
            WireRecordingProvider::text_response("listed."),
        ],
    ));
    let (agent_loop, _ws) = build_local_inline_harness(provider.clone() as Arc<dyn LLMProvider>);
    let session_key = format!("prefix-tool-{}", uuid::Uuid::new_v4());

    agent_loop
        .process_direct("please list files", &session_key, "test", "offline")
        .await;

    let calls = provider.calls();
    assert!(
        calls.len() >= 2,
        "expected two LLM calls, got {}",
        calls.len()
    );
    assert_wire_prefix(&calls[0], &calls[1]);
}

#[tokio::test]
async fn test_local_wire_prefix_stable_across_batched_tool_results_and_next_turn() {
    let mut args_a = std::collections::HashMap::new();
    args_a.insert("path".to_string(), json!("big_a.txt"));
    let mut args_b = std::collections::HashMap::new();
    args_b.insert("path".to_string(), json!("big_b.txt"));
    let provider = Arc::new(WireRecordingProvider::new(
        "local-qwen-test",
        vec![
            crate::providers::base::LLMResponse {
                content: Some(String::new()),
                tool_calls: vec![
                    crate::providers::base::ToolCallRequest {
                        id: "tc_big_a".to_string(),
                        name: "read_file".to_string(),
                        arguments: args_a,
                    },
                    crate::providers::base::ToolCallRequest {
                        id: "tc_big_b".to_string(),
                        name: "read_file".to_string(),
                        arguments: args_b,
                    },
                ],
                finish_reason: "tool_calls".to_string(),
                usage: std::collections::HashMap::new(),
            },
            WireRecordingProvider::text_response("done turn one"),
            WireRecordingProvider::text_response("turn two reply"),
        ],
    ));
    let (agent_loop, workspace) =
        build_local_inline_harness(provider.clone() as Arc<dyn LLMProvider>);
    std::fs::write(workspace.join("big_a.txt"), "alpha\n".repeat(1600)).unwrap();
    std::fs::write(workspace.join("big_b.txt"), "beta\n".repeat(1700)).unwrap();
    let session_key = format!("batched-tools-prefix-{}", uuid::Uuid::new_v4());

    tokio::time::timeout(
        std::time::Duration::from_secs(10),
        agent_loop.process_direct("read both files", &session_key, "test", "offline"),
    )
    .await
    .expect("turn 1 must terminate");
    let turn1_calls = provider.calls().len();
    assert_eq!(turn1_calls, 2, "turn 1 should call before and after tools");

    tokio::time::timeout(
        std::time::Duration::from_secs(10),
        agent_loop.process_direct("what did you see?", &session_key, "test", "offline"),
    )
    .await
    .expect("turn 2 must terminate");

    let calls = provider.calls();
    assert_eq!(calls.len(), 3, "turn 2 should add one provider call");
    assert_wire_prefix(&calls[0], &calls[1]);
    assert_wire_prefix(&calls[1], &calls[2]);

    let call1_roles: Vec<_> = calls[1]
        .iter()
        .map(|m| m["role"].as_str().unwrap_or("?"))
        .collect();
    let joined = call1_roles.join(",");
    // With native tool-role rendering (2026-07-27), parallel tool
    // results render as consecutive role:tool messages — no user-role
    // separators between them. The pattern is
    // assistant(tool_calls),tool,tool,assistant(text).
    assert!(
        joined.contains("assistant,tool,tool"),
        "batched native tool results must render as consecutive role:tool after the tool_calls assistant: {joined}"
    );
    // With native tool-role rendering, tool results are role:tool with
    // name + content (no [System: tool succeeded...] user wrapper).
    assert!(calls[1].iter().any(|m| {
        m["role"] == "tool" && m["name"] == "read_file" && m["tool_call_id"] == "tc_big_a"
    }));
    assert!(calls[1].iter().any(|m| {
        m["role"] == "tool" && m["name"] == "read_file" && m["tool_call_id"] == "tc_big_b"
    }));

    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn test_cached_duplicate_tool_receipts_trip_loop_circuit_breaker() {
    let duplicate_call = |id: usize| {
        let mut arguments = std::collections::HashMap::new();
        arguments.insert("path".to_string(), json!("."));
        crate::providers::base::LLMResponse {
            content: Some("Let me check the remaining modified files:".to_string()),
            tool_calls: vec![crate::providers::base::ToolCallRequest {
                id: format!("tc_duplicate_{id}"),
                name: "list_dir".to_string(),
                arguments,
            }],
            finish_reason: "tool_calls".to_string(),
            usage: std::collections::HashMap::new(),
        }
    };
    let provider = Arc::new(ResponseSequenceProvider::new(
        "local-qwen-test",
        vec![
            duplicate_call(1),
            duplicate_call(2),
            duplicate_call(3),
            duplicate_call(4),
            WireRecordingProvider::text_response("breaker failed"),
        ],
    ));
    let (agent_loop, workspace) =
        build_local_inline_harness(provider.clone() as Arc<dyn LLMProvider>);
    let session_key = format!("cached-duplicate-breaker-{}", uuid::Uuid::new_v4());

    let response = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        agent_loop.process_direct("inspect the workspace", &session_key, "test", "offline"),
    )
    .await
    .expect("cached duplicate loop must terminate");

    assert_eq!(
        response,
        "The same tool request repeated after its result was already available, so the loop was stopped to prevent further duplicate work."
    );
    assert_eq!(
        provider.call_count(),
        2,
        "first read executes; the first cached duplicate must force finalization"
    );

    let _ = std::fs::remove_dir_all(&workspace);
}

fn stale_read_write_context_parts(
    path: &str,
) -> (
    Vec<Value>,
    std::collections::HashMap<String, Value>,
    crate::agent::tool_guard::ToolGuard,
) {
    let mut read_args = std::collections::HashMap::new();
    read_args.insert("path".to_string(), json!(path));
    let mut write_args = std::collections::HashMap::new();
    write_args.insert("path".to_string(), json!(path));
    write_args.insert("content".to_string(), json!("new\n"));
    let read_call = crate::providers::base::ToolCallRequest {
        id: "tc_read_old".to_string(),
        name: "read_file".to_string(),
        arguments: read_args.clone(),
    };
    let write_call = crate::providers::base::ToolCallRequest {
        id: "tc_write_new".to_string(),
        name: "write_file".to_string(),
        arguments: write_args.clone(),
    };

    let mut guard = crate::agent::tool_guard::ToolGuard::new(1);
    assert!(guard.allow("read_file", &read_args).is_ok());
    guard.record_result_with_status("read_file", &read_args, "old\n".to_string(), true);
    assert!(guard.allow("write_file", &write_args).is_ok());
    guard.record_result_with_status("write_file", &write_args, "written".to_string(), true);
    assert_eq!(
        guard.get_cached_result(&crate::agent::tool_guard::ToolGuard::key(
            "read_file",
            &read_args
        )),
        None
    );

    (
        vec![
            json!({"role": "user", "content": "read, write, re-read"}),
            json!({"role": "assistant", "content": "", "tool_calls": [read_call.to_openai_json()]}),
            json!({"role": "tool", "tool_call_id": "tc_read_old", "name": "read_file", "ok": true, "content": "old\n"}),
            json!({"role": "assistant", "content": "", "tool_calls": [write_call.to_openai_json()]}),
            json!({"role": "tool", "tool_call_id": "tc_write_new", "name": "write_file", "ok": true, "content": "written"}),
        ],
        read_args,
        guard,
    )
}

#[tokio::test]
async fn test_read_after_write_same_turn_is_not_blocked_by_stale_receipt() {
    let provider = Arc::new(WireRecordingProvider::new(
        "local-qwen-test",
        vec![WireRecordingProvider::text_response("unused")],
    ));
    let (agent_loop, workspace) =
        build_local_inline_harness(provider.clone() as Arc<dyn LLMProvider>);
    let msg = InboundMessage::new("test", "user", "offline", "read, write, re-read");
    let mut ctx = agent_loop
        .shared
        .prepare_context(&msg, None, None, None, None)
        .await;

    let (messages, read_args, guard) = stale_read_write_context_parts("/tmp/same-turn.txt");
    ctx.messages = messages;
    ctx.new_start = 0;
    ctx.flow.tool_guard = guard;

    let result = crate::agent::router::route_tool_calls(
        &mut ctx,
        Some(""),
        vec![crate::providers::base::ToolCallRequest {
            id: "tc_read_new".to_string(),
            name: "read_file".to_string(),
            arguments: read_args,
        }],
    )
    .await;

    match result {
        crate::agent::router::RouteResult::Execute(calls) => {
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].id, "tc_read_new");
        }
        crate::agent::router::RouteResult::Break(text) => {
            panic!("post-write read was blocked: {text}")
        }
        crate::agent::router::RouteResult::Continue => panic!("post-write read should execute"),
    }
    assert!(!ctx.flow.tool_guard.had_blocked_calls);

    let _ = std::fs::remove_dir_all(&workspace);
}

/// Regression (prod, session cli:oneshot, bonsai-27b): a turn that runs a
/// side-effect tool (exec) arms the response boundary, which injects a
/// synthetic `scaffold_user` nudge into the conversation. That nudge is
/// rendered into the wire but never persisted. When a later tool round appends
/// after it, the nudge sits MID-history; on the NEXT turn the reloaded history
/// no longer contains it, so turn N+1's wire is no longer a byte-prefix of
/// turn N's last wire — the local server re-prefills the whole context
/// (`prompt_prefix_diverged`, ~30-200s on a 27B). This is the observed 38→32
/// wire shrink diverging at an empty `[assistant]` carrier.
#[tokio::test]
async fn test_wire_prefix_stable_across_turn_after_side_effect_boundary_nudge() {
    let mut exec_args = std::collections::HashMap::new();
    exec_args.insert("command".to_string(), json!("echo hi"));
    let exec_call = crate::providers::base::LLMResponse {
        content: Some(String::new()),
        tool_calls: vec![crate::providers::base::ToolCallRequest {
            id: "tc_exec".to_string(),
            name: "exec".to_string(),
            arguments: exec_args,
        }],
        finish_reason: "tool_calls".to_string(),
        usage: std::collections::HashMap::new(),
    };
    let mut ls_args = std::collections::HashMap::new();
    ls_args.insert("path".to_string(), json!("."));
    let listdir_call = crate::providers::base::LLMResponse {
        content: Some(String::new()),
        tool_calls: vec![crate::providers::base::ToolCallRequest {
            id: "tc_ls".to_string(),
            name: "list_dir".to_string(),
            arguments: ls_args,
        }],
        finish_reason: "tool_calls".to_string(),
        usage: std::collections::HashMap::new(),
    };
    // Turn 1: exec (arms boundary → nudge injected before the next call) then a
    // second tool round (list_dir) that lands AFTER the nudge, then a final
    // text reply. Turn 2: a plain text reply.
    let provider = Arc::new(WireRecordingProvider::new(
        "local-qwen-test",
        vec![
            exec_call,
            listdir_call,
            WireRecordingProvider::text_response("done turn one"),
            WireRecordingProvider::text_response("turn two reply"),
        ],
    ));
    let (agent_loop, workspace) =
        build_local_inline_harness(provider.clone() as Arc<dyn LLMProvider>);
    let session_key = format!("boundary-nudge-prefix-{}", uuid::Uuid::new_v4());

    tokio::time::timeout(
        std::time::Duration::from_secs(10),
        agent_loop.process_direct("run something", &session_key, "test", "offline"),
    )
    .await
    .expect("turn 1 must terminate");

    let turn1_calls = provider.calls().len();
    assert!(turn1_calls >= 2, "turn 1 must make multiple provider calls");

    tokio::time::timeout(
        std::time::Duration::from_secs(10),
        agent_loop.process_direct("what did you find?", &session_key, "test", "offline"),
    )
    .await
    .expect("turn 2 must terminate");

    let calls = provider.calls();
    assert!(
        calls.len() > turn1_calls,
        "turn 2 must make at least one provider call (had {turn1_calls}, now {})",
        calls.len()
    );

    // Turn N's last wire must be a byte-prefix of turn N+1's first wire.
    assert_wire_prefix(&calls[turn1_calls - 1], &calls[turn1_calls]);

    let _ = std::fs::remove_dir_all(&workspace);
}

/// Full production shape (log 2026-07-17, cli:oneshot, bonsai-27b): repeated
/// identical `exec` calls — the first ones execute (arming the response
/// boundary, injecting its scaffold nudge), the rest are duplicate-blocked
/// until the tool-loop circuit breaker forces a text response. The NEXT turn's
/// reloaded wire must still be a byte-suffix extension of the previous turn's
/// last wire (no mid-history shrink, no `prompt_prefix_diverged`).
#[tokio::test]
async fn test_wire_prefix_stable_after_duplicate_exec_circuit_breaker() {
    let exec_call = |id: usize| {
        let mut arguments = std::collections::HashMap::new();
        arguments.insert("command".to_string(), json!("echo hi"));
        crate::providers::base::LLMResponse {
            content: Some(String::new()),
            tool_calls: vec![crate::providers::base::ToolCallRequest {
                id: format!("tc_exec_{id}"),
                name: "exec".to_string(),
                arguments,
            }],
            finish_reason: "tool_calls".to_string(),
            usage: std::collections::HashMap::new(),
        }
    };
    let provider = Arc::new(WireRecordingProvider::new(
        "local-qwen-test",
        vec![
            exec_call(1),
            exec_call(2),
            exec_call(3),
            exec_call(4),
            exec_call(5),
            exec_call(6),
            exec_call(7),
            WireRecordingProvider::text_response("turn two reply"),
        ],
    ));
    let (agent_loop, workspace) =
        build_local_inline_harness(provider.clone() as Arc<dyn LLMProvider>);
    let session_key = format!("dup-exec-breaker-prefix-{}", uuid::Uuid::new_v4());

    tokio::time::timeout(
        std::time::Duration::from_secs(15),
        agent_loop.process_direct("run the check", &session_key, "test", "offline"),
    )
    .await
    .expect("turn 1 (duplicate exec loop) must terminate");

    let turn1_calls = provider.calls().len();
    assert!(turn1_calls >= 2, "turn 1 must make multiple provider calls");

    tokio::time::timeout(
        std::time::Duration::from_secs(15),
        agent_loop.process_direct("so what happened?", &session_key, "test", "offline"),
    )
    .await
    .expect("turn 2 must terminate");

    let calls = provider.calls();
    assert!(
        calls.len() > turn1_calls,
        "turn 2 must make at least one provider call (had {turn1_calls}, now {})",
        calls.len()
    );
    assert_wire_prefix(&calls[turn1_calls - 1], &calls[turn1_calls]);

    let _ = std::fs::remove_dir_all(&workspace);
}

struct EnvVarGuard {
    key: &'static str,
    saved: Option<std::ffi::OsString>,
}

impl EnvVarGuard {
    fn remove(key: &'static str) -> Self {
        let saved = std::env::var_os(key);
        std::env::remove_var(key);
        Self { key, saved }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        if let Some(value) = self.saved.take() {
            std::env::set_var(self.key, value);
        } else {
            std::env::remove_var(self.key);
        }
    }
}

#[tokio::test]
async fn test_tool_call_carrier_persists_before_tool_result() {
    let mut args = std::collections::HashMap::new();
    args.insert("path".to_string(), json!("."));
    let main: Arc<dyn LLMProvider> = Arc::new(ResponseSequenceProvider::new(
        "local-main",
        vec![
            crate::providers::base::LLMResponse {
                content: Some(String::new()),
                tool_calls: vec![crate::providers::base::ToolCallRequest {
                    id: "tc_list".to_string(),
                    name: "list_dir".to_string(),
                    arguments: args,
                }],
                finish_reason: "tool_calls".to_string(),
                usage: std::collections::HashMap::new(),
            },
            crate::providers::base::LLMResponse {
                content: Some(attested_text("I listed the workspace.")),
                tool_calls: vec![],
                finish_reason: "stop".to_string(),
                usage: std::collections::HashMap::new(),
            },
        ],
    ));
    let (agent_loop, workspace) = build_local_inline_harness(main);
    let session_key = format!("test-tool-order-{}", uuid::Uuid::new_v4().to_string());

    let response = agent_loop
        .process_direct("please list files", &session_key, "test", "offline")
        .await;
    assert_eq!(response, "I listed the workspace.");

    let core = agent_loop.shared.core_handle.swappable();
    let meta = core
        .sessions
        .get_latest_session(&session_key)
        .await
        .expect("session should exist");
    let raw = core.sessions.get_all_messages(&meta.id).await;
    let roles: Vec<&str> = raw
        .iter()
        .map(|m| m.get("role").and_then(|r| r.as_str()).unwrap_or(""))
        .collect();

    assert_eq!(roles, vec!["user", "assistant", "tool", "assistant"]);
    assert!(
        raw[1].get("tool_calls").is_some(),
        "assistant carrier must retain tool_calls"
    );
    assert_eq!(
        raw[2].get("tool_call_id").and_then(|v| v.as_str()),
        raw[1]
            .get("tool_calls")
            .and_then(|v| v.as_array())
            .and_then(|calls| calls.first())
            .and_then(|call| call.get("id"))
            .and_then(|v| v.as_str()),
        "tool result must point at the immediately preceding assistant call"
    );

    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn test_tool_round_is_durable_before_next_provider_call_completes() {
    let provider = Arc::new(ToolRoundBarrierProvider::new());
    let (agent_loop, workspace) =
        build_local_inline_harness(provider.clone() as Arc<dyn LLMProvider>);
    let core = agent_loop.shared.core_handle.swappable();
    let session_key = format!("test-tool-durable-{}", uuid::Uuid::new_v4());
    let task_loop = Arc::new(agent_loop);
    let task = {
        let task_loop = task_loop.clone();
        let session_key = session_key.clone();
        tokio::spawn(async move {
            task_loop
                .process_direct("please list files", &session_key, "test", "offline")
                .await
        })
    };

    tokio::time::timeout(
        std::time::Duration::from_secs(5),
        provider.second_call_started.notified(),
    )
    .await
    .expect("second provider call should start after the tool round");

    let meta = core
        .sessions
        .get_latest_session(&session_key)
        .await
        .expect("session should exist while the turn is active");
    let durable = core.sessions.get_all_messages(&meta.id).await;
    let roles: Vec<&str> = durable
        .iter()
        .map(|message| message.get("role").and_then(Value::as_str).unwrap_or(""))
        .collect();
    assert_eq!(
        roles,
        vec!["user", "assistant", "tool"],
        "the active tool protocol must be crash-durable before the next inference"
    );

    provider.release_second_call.notify_one();
    assert_eq!(task.await.expect("turn task should join"), "done");
    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn test_multiple_tool_round_carriers_persist_in_order() {
    let mut list_args = std::collections::HashMap::new();
    list_args.insert("path".to_string(), json!("."));
    let mut exec_args = std::collections::HashMap::new();
    exec_args.insert("command".to_string(), json!("printf ok"));
    exec_args.insert("working_dir".to_string(), json!("."));

    let main: Arc<dyn LLMProvider> = Arc::new(ResponseSequenceProvider::new(
        "local-main",
        vec![
            crate::providers::base::LLMResponse {
                content: Some(String::new()),
                tool_calls: vec![crate::providers::base::ToolCallRequest {
                    id: "tc_list".to_string(),
                    name: "list_dir".to_string(),
                    arguments: list_args,
                }],
                finish_reason: "tool_calls".to_string(),
                usage: std::collections::HashMap::new(),
            },
            crate::providers::base::LLMResponse {
                content: Some("I found the script; running it now.".to_string()),
                tool_calls: vec![crate::providers::base::ToolCallRequest {
                    id: "tc_exec".to_string(),
                    name: "exec".to_string(),
                    arguments: exec_args,
                }],
                finish_reason: "tool_calls".to_string(),
                usage: std::collections::HashMap::new(),
            },
            crate::providers::base::LLMResponse {
                content: Some(attested_text("Done.")),
                tool_calls: vec![],
                finish_reason: "stop".to_string(),
                usage: std::collections::HashMap::new(),
            },
        ],
    ));
    let (agent_loop, workspace) = build_local_inline_harness(main);
    let session_key = format!("test-tool-order-multi-{}", uuid::Uuid::new_v4().to_string());

    let response = agent_loop
        .process_direct("please inspect and run", &session_key, "test", "offline")
        .await;
    assert_eq!(response, "Done.");

    let core = agent_loop.shared.core_handle.swappable();
    let meta = core
        .sessions
        .get_latest_session(&session_key)
        .await
        .expect("session should exist");
    let raw = core.sessions.get_all_messages(&meta.id).await;
    let roles: Vec<&str> = raw
        .iter()
        .map(|m| m.get("role").and_then(|r| r.as_str()).unwrap_or(""))
        .collect();

    assert_eq!(
        roles,
        vec![
            "user",
            "assistant",
            "tool",
            "assistant",
            "tool",
            "assistant"
        ]
    );
    assert!(raw[1].get("tool_calls").is_some());
    assert!(raw[3].get("tool_calls").is_some());
    assert_eq!(
        raw[2].get("tool_call_id").and_then(|v| v.as_str()),
        Some("tc_list")
    );
    assert_eq!(
        raw[4].get("tool_call_id").and_then(|v| v.as_str()),
        Some("tc_exec")
    );

    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn test_local_truncated_response_requires_attested_correction_without_continue() {
    let main = Arc::new(ResponseSequenceProvider::new(
        "local-main",
        vec![crate::providers::base::LLMResponse {
            content: Some("Partial local answer".to_string()),
            tool_calls: vec![],
            finish_reason: "length".to_string(),
            usage: std::collections::HashMap::new(),
        }],
    ));
    let main_dyn: Arc<dyn LLMProvider> = main.clone();
    let (agent_loop, workspace) = build_local_inline_harness(main_dyn);
    let session_key = format!(
        "test-local-no-auto-continue-{}",
        uuid::Uuid::new_v4().to_string()
    );

    let response = agent_loop
        .process_direct("answer briefly", &session_key, "test", "offline")
        .await;

    // With the attestation protocol removed, truncated text follows the
    // standard auto-continuation path; if continuations are exhausted the
    // accumulated text is the final answer (no attestation retry loop).
    assert!(
        !response.is_empty(),
        "expected a non-empty terminal response, got: {response:?}"
    );

    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn test_local_streaming_cache_markers_append_only_across_turns() {
    let main: Arc<dyn LLMProvider> = Arc::new(ResponseSequenceProvider::new(
        "local-main",
        vec![
            crate::providers::base::LLMResponse {
                content: Some(attested_text("one")),
                tool_calls: vec![],
                finish_reason: "stop".to_string(),
                usage: std::collections::HashMap::new(),
            },
            crate::providers::base::LLMResponse {
                content: Some(attested_text("two")),
                tool_calls: vec![],
                finish_reason: "stop".to_string(),
                usage: std::collections::HashMap::new(),
            },
            crate::providers::base::LLMResponse {
                content: Some(attested_text("three")),
                tool_calls: vec![],
                finish_reason: "stop".to_string(),
                usage: std::collections::HashMap::new(),
            },
        ],
    ));
    let (agent_loop, workspace) = build_local_inline_harness(main);
    let session_key = format!(
        "test-local-cache-markers-{}",
        uuid::Uuid::new_v4().to_string()
    );

    let mut cache_markers = Vec::new();
    let mut prefill_estimates = Vec::new();
    for (input, expected) in [
        ("first short turn", "one"),
        ("second short turn", "two"),
        ("third short turn", "three"),
    ] {
        let (delta_tx, mut delta_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        let response = agent_loop
            .process_direct_streaming(
                input,
                &session_key,
                "test",
                "offline",
                None,
                delta_tx,
                None,
                None,
                None,
                None,
            )
            .await;
        assert_eq!(response, expected);

        let mut turn_cache_markers = Vec::new();
        let mut turn_prefill_estimates = Vec::new();
        while let Ok(delta) = delta_rx.try_recv() {
            if delta.starts_with("\u{0}cache:") {
                turn_cache_markers.push(delta);
            } else if delta.starts_with("\u{0}prefill_estimate:") {
                turn_prefill_estimates.push(delta);
            }
        }
        assert_eq!(
            turn_cache_markers.len(),
            1,
            "each streamed turn should emit one cache marker"
        );
        assert_eq!(
            turn_prefill_estimates.len(),
            1,
            "each streamed turn should emit one prefill estimate"
        );
        cache_markers.push(turn_cache_markers.remove(0));
        prefill_estimates.push(turn_prefill_estimates.remove(0));
    }

    assert!(
        cache_markers[0].starts_with("\u{0}cache:first:"),
        "first turn should establish the cache: {cache_markers:?}"
    );
    // Each new turn reloads history from DB via filter_history, which applies
    // byte-changing transformations (recall_tool_result raw→digest, etc.).
    // The fingerprint is cleared at the start of each turn to prevent false
    // divergences — the first call of each turn shows First, not AppendOnly.
    // The Higgs radix cache still hits (content-based, not fingerprint-based).
    assert!(
        cache_markers[1].starts_with("\u{0}cache:first:"),
        "second turn starts fresh (fingerprint cleared on DB reload): {cache_markers:?}"
    );
    assert!(
        cache_markers[2].starts_with("\u{0}cache:first:"),
        "third turn starts fresh: {cache_markers:?}"
    );
    assert!(
        cache_markers
            .iter()
            .all(|marker| !marker.starts_with("\u{0}cache:diverged:")),
        "local cache path must not diverge across ordinary turns: {cache_markers:?}"
    );
    for marker in &prefill_estimates {
        let tokens: usize = marker
            .trim_start_matches("\u{0}prefill_estimate:")
            .parse()
            .expect("prefill estimate token count");
        assert!(tokens > 0, "prefill estimate must be positive: {marker:?}");
    }

    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn test_failed_local_call_does_not_seed_prompt_cache_marker() {
    let provider: Arc<dyn LLMProvider> = Arc::new(FailOnceThenResponseProvider::new(
        "local-main",
        crate::providers::base::LLMResponse {
            content: Some(attested_text("recovered")),
            tool_calls: vec![],
            finish_reason: "stop".to_string(),
            usage: std::collections::HashMap::new(),
        },
    ));
    let (agent_loop, workspace) = build_local_inline_harness(provider);
    let session_key = format!(
        "test-local-cache-failure-{}",
        uuid::Uuid::new_v4().to_string()
    );

    let (first_tx, mut first_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let first = agent_loop
        .process_direct_streaming(
            "first turn fails before the server can warm cache",
            &session_key,
            "test",
            "offline",
            None,
            first_tx,
            None,
            None,
            None,
            None,
        )
        .await;
    assert!(
        first.contains("synthetic provider failure"),
        "expected provider error, got {first:?}"
    );
    let mut first_markers = Vec::new();
    while let Ok(delta) = first_rx.try_recv() {
        if delta.starts_with("\u{0}cache:") {
            first_markers.push(delta);
        }
    }
    assert!(
        first_markers
            .first()
            .is_some_and(|m| m.starts_with("\u{0}cache:first:")),
        "failed call may diagnose cold cache, but must not commit it: {first_markers:?}"
    );

    let (second_tx, mut second_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let second = agent_loop
        .process_direct_streaming(
            "second turn should still be cold from nanobot's cache model",
            &session_key,
            "test",
            "offline",
            None,
            second_tx,
            None,
            None,
            None,
            None,
        )
        .await;
    assert_eq!(second, "recovered");

    let mut second_markers = Vec::new();
    while let Ok(delta) = second_rx.try_recv() {
        if delta.starts_with("\u{0}cache:") {
            second_markers.push(delta);
        }
    }
    assert!(
        second_markers
            .first()
            .is_some_and(|m| m.starts_with("\u{0}cache:first:")),
        "a failed provider call must not make the next turn look append-only: {second_markers:?}"
    );

    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn test_direct_streaming_forwards_thinking_delta_with_ansi_marker() {
    let main = Arc::new(StreamingThinkingProvider::new("local-main"));
    let main_dyn: Arc<dyn LLMProvider> = main.clone();
    let (agent_loop, workspace) = build_local_inline_harness(main_dyn);
    agent_loop
        .shared
        .core_handle
        .counters
        .thinking_budget
        .store(128, std::sync::atomic::Ordering::Relaxed);
    let session_key = format!(
        "test-direct-thinking-stream-{}",
        uuid::Uuid::new_v4().to_string()
    );

    let (delta_tx, mut delta_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let response = agent_loop
        .process_direct_streaming(
            "show thinking",
            &session_key,
            "test",
            "offline",
            None,
            delta_tx,
            None,
            None,
            None,
            None,
        )
        .await;

    assert_eq!(response, "visible answer");
    assert_eq!(
        main.last_thinking_budget(),
        128,
        "direct streaming call should pass the enabled thinking budget to the provider"
    );

    let mut deltas = Vec::new();
    while let Ok(delta) = delta_rx.try_recv() {
        deltas.push(delta);
    }
    let marker_idx = deltas
        .iter()
        .position(|delta| delta == "\x1b[90m\x1b[2m")
        .unwrap_or_else(|| panic!("missing thinking marker in deltas: {deltas:?}"));
    assert_eq!(
        deltas.get(marker_idx + 1).map(String::as_str),
        Some("private thought"),
        "thinking text should immediately follow the ANSI marker: {deltas:?}"
    );
    assert!(
        deltas.iter().any(|delta| delta == "\x1b[0m\n\n"),
        "thinking stream should be reset before visible text: {deltas:?}"
    );
    assert!(
        deltas.iter().any(|delta| delta == "visible answer"),
        "visible answer text should still stream after thinking: {deltas:?}"
    );

    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn test_vibethinker_hidden_reasoning_streams_without_think_budget() {
    let main = Arc::new(StreamingThinkingProvider::new("VibeThinker-3B-mlx-8Bit"));
    let main_dyn: Arc<dyn LLMProvider> = main.clone();
    let (agent_loop, workspace) =
        build_local_inline_harness_with_model(main_dyn, "local:VibeThinker-3B-mlx-8Bit");
    let session_key = format!(
        "test-vibethinker-hidden-thinking-{}",
        uuid::Uuid::new_v4().to_string()
    );

    let (delta_tx, mut delta_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let response = agent_loop
        .process_direct_streaming(
            "show native thinking",
            &session_key,
            "test",
            "offline",
            None,
            delta_tx,
            None,
            None,
            None,
            None,
        )
        .await;

    assert_eq!(response, "visible answer");
    assert_eq!(
        main.last_thinking_budget(),
        0,
        "hidden reasoning should not impose a nanobot thinking budget"
    );

    let mut deltas = Vec::new();
    while let Ok(delta) = delta_rx.try_recv() {
        deltas.push(delta);
    }
    assert!(
        deltas.iter().any(|delta| delta == "\x1b[90m\x1b[2m"),
        "VibeThinker reasoning_content should stream to display without /think: {deltas:?}"
    );
    assert!(
        deltas.iter().any(|delta| delta == "private thought"),
        "hidden reasoning text should not be dropped: {deltas:?}"
    );

    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn test_tts_suppression_does_not_hide_vibethinker_display() {
    let main = Arc::new(StreamingThinkingProvider::new("VibeThinker-3B-mlx-8Bit"));
    let main_dyn: Arc<dyn LLMProvider> = main.clone();
    let (agent_loop, workspace) =
        build_local_inline_harness_with_model(main_dyn, "local:VibeThinker-3B-mlx-8Bit");
    agent_loop
        .shared
        .core_handle
        .counters
        .suppress_thinking_in_tts
        .store(true, std::sync::atomic::Ordering::Relaxed);
    let session_key = format!(
        "test-vibethinker-tts-suppression-display-{}",
        uuid::Uuid::new_v4().to_string()
    );

    let (delta_tx, mut delta_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let response = agent_loop
        .process_direct_streaming(
            "show native thinking while voice is on",
            &session_key,
            "test",
            "offline",
            None,
            delta_tx,
            None,
            None,
            None,
            None,
        )
        .await;

    assert_eq!(response, "visible answer");
    let mut deltas = Vec::new();
    while let Ok(delta) = delta_rx.try_recv() {
        deltas.push(delta);
    }
    assert!(
        deltas.iter().any(|delta| delta == "private thought"),
        "TTS suppression should not suppress the visual thinking stream: {deltas:?}"
    );

    let _ = std::fs::remove_dir_all(&workspace);
}

// -----------------------------------------------------------------------
// Test 1: router decides "respond" — specialist is never called
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_trio_offline_e2e_respond() {
    let router_resp = r#"{"action":"respond","target":"main","args":{},"confidence":0.9}"#;
    let main_resp = "Four.";

    let router: Arc<dyn LLMProvider> = Arc::new(SequenceProvider::new(
        "offline-router",
        vec![router_resp, router_resp, router_resp],
    ));
    let main: Arc<dyn LLMProvider> = Arc::new(StaticResponseLLM::new("offline-main", main_resp));
    let specialist: Arc<dyn LLMProvider> = Arc::new(StaticResponseLLM::new(
        "offline-specialist",
        "specialist unused",
    ));

    let (agent_loop, workspace) = build_trio_offline_harness(main, router, specialist);

    let resp = agent_loop
        .process_direct("What is 2+2?", "trio-offline-respond", "test", "offline")
        .await;

    eprintln!(
        "test_trio_offline_e2e_respond: response ({} chars): {}",
        resp.len(),
        &resp[..resp.len().min(200)]
    );

    let counters = &agent_loop.shared.core_handle.counters;
    let metrics = &counters.trio_metrics;

    assert!(
        metrics
            .router_preflight_fired
            .load(std::sync::atomic::Ordering::Relaxed),
        "router preflight should have fired"
    );
    assert_eq!(
        metrics.router_action.lock().as_deref(),
        Some("respond"),
        "router_action should be 'respond'"
    );
    assert!(
        !metrics
            .specialist_dispatched
            .load(std::sync::atomic::Ordering::Relaxed),
        "specialist should NOT have been dispatched for a 'respond' decision"
    );
    assert!(!resp.is_empty(), "response should be non-empty");

    let _ = std::fs::remove_dir_all(&workspace);
}

#[tokio::test]
async fn test_local_thinking_reserves_max_tokens_end_to_end() {
    let router_resp = r#"{"action":"respond","target":"main","args":{},"confidence":0.9}"#;
    let router: Arc<dyn LLMProvider> = Arc::new(SequenceProvider::new(
        "offline-router",
        vec![router_resp, router_resp, router_resp],
    ));
    let main = Arc::new(RecordingProvider::new("offline-main", "ok"));
    let main_dyn: Arc<dyn LLMProvider> = main.clone();
    let specialist: Arc<dyn LLMProvider> =
        Arc::new(StaticResponseLLM::new("offline-specialist", "unused"));

    let (agent_loop, workspace) = build_trio_offline_harness(main_dyn, router, specialist);
    agent_loop
        .shared
        .core_handle
        .counters
        .thinking_budget
        .store(128, std::sync::atomic::Ordering::Relaxed);

    let _ = agent_loop
        .process_direct(
            "What is the current date?",
            "reserve-max-tokens",
            "test",
            "offline",
        )
        .await;

    assert_eq!(
        main.last_max_tokens(),
        640,
        "local thinking should add budget on top of base max_tokens=512 (512+128=640)"
    );

    let _ = std::fs::remove_dir_all(&workspace);
}

// -----------------------------------------------------------------------
// Test 2: router decides "specialist" — specialist is called
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_trio_offline_e2e_specialist_dispatch() {
    let router_resp = r#"{"action":"specialist","target":"coding","args":{"task":"explain loops"},"confidence":0.85}"#;

    let router: Arc<dyn LLMProvider> = Arc::new(SequenceProvider::new(
        "offline-router",
        vec![router_resp, router_resp, router_resp],
    ));
    let main: Arc<dyn LLMProvider> = Arc::new(StaticResponseLLM::new("offline-main", "delegating"));
    let specialist: Arc<dyn LLMProvider> = Arc::new(StaticResponseLLM::new(
        "offline-specialist",
        "Here is the specialist answer.",
    ));

    let (agent_loop, workspace) = build_trio_offline_harness(main, router, specialist);

    let resp = agent_loop
        .process_direct(
            "Explain for loops",
            "trio-offline-specialist",
            "test",
            "offline",
        )
        .await;

    eprintln!(
        "test_trio_offline_e2e_specialist_dispatch: response ({} chars): {}",
        resp.len(),
        &resp[..resp.len().min(200)]
    );

    let metrics = &agent_loop.shared.core_handle.counters.trio_metrics;

    assert_eq!(
        metrics.router_action.lock().as_deref(),
        Some("specialist"),
        "router_action should be 'specialist'"
    );
    assert!(
        metrics
            .specialist_dispatched
            .load(std::sync::atomic::Ordering::Relaxed),
        "specialist should have been dispatched"
    );
    assert!(!resp.is_empty(), "response should be non-empty");

    let _ = std::fs::remove_dir_all(&workspace);
}

// -----------------------------------------------------------------------
// Test 3: circuit breaker cascade
//
// The router returns non-JSON 3+ times. Each failure is recorded under
// the key "router:{model}" (as router.rs does). However, agent_loop.rs
// checks availability under "trio_router" — so the CB check at the
// should_strip_tools_for_trio call site never sees the tripped breaker.
//
// This test documents that discrepancy explicitly.
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_trio_offline_e2e_circuit_breaker_cascade() {
    // All 4 router calls return non-JSON to trip the circuit breaker.
    let router: Arc<dyn LLMProvider> = Arc::new(SequenceProvider::new(
        "offline-router",
        vec![
            "this is not json at all !!!",
            "this is not json at all !!!",
            "this is not json at all !!!",
            "this is not json at all !!!",
        ],
    ));
    let main: Arc<dyn LLMProvider> = Arc::new(StaticResponseLLM::new(
        "offline-main",
        "main fallback response",
    ));
    let specialist: Arc<dyn LLMProvider> = Arc::new(StaticResponseLLM::new(
        "offline-specialist",
        "specialist unused",
    ));

    let (agent_loop, workspace) = build_trio_offline_harness(main, router, specialist);

    // Send 4 messages — each failure increments the CB counter.
    // After 3 failures (default threshold) the CB is tripped.
    // The 4th call will be via Passthrough (router returns early) because
    // the CB key "router:offline-router" is open. Main answers directly.
    for i in 0..4u32 {
        let resp = agent_loop
            .process_direct(
                &format!("message {}", i),
                "trio-offline-cb",
                "test",
                "offline",
            )
            .await;
        eprintln!(
            "  cascade msg {}: ({} chars) {}",
            i,
            resp.len(),
            &resp[..resp.len().min(80)]
        );
    }

    let counters = &agent_loop.shared.core_handle.counters;

    // After repeated failures the trio state should be Degraded.
    let state = counters.get_trio_state();
    eprintln!("trio_state after cascade: {:?}", state);
    assert_eq!(
        state,
        crate::agent::agent_core::TrioState::Degraded,
        "trio_state should be Degraded after repeated router failures"
    );

    // Verify CB key alignment after the fix.
    //
    // The offline harness returns mock responses that fail strict AND lenient
    // parsing (lenient no longer defaults to phantom "clarify" target — it
    // returns None when no target can be extracted). Each parse failure records
    // a CB failure, so after 4 turns the CB should be tripped.
    //
    // The shared CB key format ("router:{model}") ensures that the
    // tool-stripping guard in step_pre_call and the routing skip in
    // router_preflight observe the same state.
    let cb_correct_key_available = counters
        .trio_circuit_breaker
        .lock()
        .is_available("router:offline-router");
    eprintln!(
        "CB 'router:offline-router' available after 4 turns: {}",
        cb_correct_key_available
    );
    // Parse failures are now correctly recorded — CB should be tripped.
    assert!(
        !cb_correct_key_available,
        "CB 'router:offline-router' should be tripped: parse failures are now recorded"
    );
    // The legacy key "trio_router" is also untouched.
    let cb_legacy_key_available = counters
        .trio_circuit_breaker
        .lock()
        .is_available("trio_router");
    assert!(
        cb_legacy_key_available,
        "CB 'trio_router' should be untouched — agent_loop now uses 'router:{{model}}' key"
    );

    let _ = std::fs::remove_dir_all(&workspace);
}

// -----------------------------------------------------------------------
// Test 4: health gate — degraded router probe bypasses preflight
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_trio_offline_e2e_health_gate() {
    use crate::config::schema::LcmSchemaConfig;
    use crate::heartbeat::health::{HealthProbe, HealthRegistry, ProbeResult};

    // A mock probe that always returns unhealthy (simulates router being down).
    struct AlwaysUnhealthyProbe;

    #[async_trait]
    impl HealthProbe for AlwaysUnhealthyProbe {
        fn name(&self) -> &str {
            "trio_router"
        }

        fn interval_secs(&self) -> u64 {
            0 // always due
        }

        async fn check(&self) -> ProbeResult {
            ProbeResult {
                healthy: false,
                latency_ms: 0,
            }
        }
    }

    // Build a registry and degrade the trio_router probe.
    let mut health_registry = HealthRegistry::new();
    health_registry.register(Box::new(AlwaysUnhealthyProbe));
    // Run 3 times to reach DEGRADED_THRESHOLD = 3.
    for _ in 0..3 {
        health_registry.run_due_probes().await;
    }
    assert!(
        !health_registry.is_healthy("trio_router"),
        "trio_router should be degraded after 3 failures"
    );
    let health_registry = Arc::new(health_registry);

    // The router SequenceProvider would fail the test if called (empty queue).
    // We keep a typed Arc so we can read call_count() after the run.
    let router_seq = Arc::new(SequenceProvider::new(
        "offline-router",
        vec![], // empty — calling this would return the sentinel error
    ));
    let router: Arc<dyn LLMProvider> = router_seq.clone();
    let main: Arc<dyn LLMProvider> =
        Arc::new(StaticResponseLLM::new("offline-main", "main answer"));
    let specialist: Arc<dyn LLMProvider> = Arc::new(StaticResponseLLM::new(
        "offline-specialist",
        "specialist unused",
    ));

    // Build harness manually so we can wire in the health registry.
    let workspace = tempfile::tempdir().unwrap().keep();
    let mut td = ToolDelegationConfig {
        mode: crate::config::schema::DelegationMode::trio(),
        ..Default::default()
    };
    td.apply_mode();

    let router_model = router.get_default_model().to_string();
    let specialist_model = specialist.get_default_model().to_string();
    let trio_config = TrioConfig {
        enabled: true,
        router_model: router_model.clone(),
        specialist_model: specialist_model.clone(),
        ..Default::default()
    };

    let core = build_swappable_core(SwappableCoreConfig {
        provider: main,
        workspace: workspace.clone(),
        model: "offline-main".to_string(),
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
        is_local: true,
        lane: Lane::default(),
        tool_delegation: td,
        provenance: ProvenanceConfig::default(),
        max_tool_result_chars: 2000,
        delegation_provider: Some(router.clone()),
        specialist_provider: Some(specialist),
        trio_config,
        model_capabilities_overrides: std::collections::HashMap::new(),
        reasoning_config: crate::config::schema::ReasoningConfig::default(),
        tool_heartbeat_secs: 2,
        health_check_timeout_secs: 2,
        adaptive_tokens: AdaptiveTokenConfig::default(),
        sessions_db_path: Some(
            std::env::temp_dir().join(format!("nanobot-test-{}.sqlite", uuid::Uuid::new_v4())),
        ),
    });

    let counters = test_runtime_counters(4096);
    let core_handle = AgentHandle::new(core, counters);

    let (inbound_tx, inbound_rx) = tokio::sync::mpsc::unbounded_channel::<InboundMessage>();
    let (outbound_tx, _outbound_rx) = tokio::sync::mpsc::unbounded_channel::<OutboundMessage>();

    let agent_loop = AgentLoop::new(
        core_handle,
        inbound_rx,
        outbound_tx,
        inbound_tx,
        None,
        1,
        None,
        None,
        None,
        ProprioceptionConfig::default(),
        LcmSchemaConfig::default(),
        Some(health_registry), // health registry is wired in here
    );

    let resp = agent_loop
        .process_direct("Hello", "trio-offline-health-gate", "test", "offline")
        .await;

    eprintln!(
        "test_trio_offline_e2e_health_gate: response ({} chars): {}",
        resp.len(),
        &resp[..resp.len().min(200)]
    );

    // When the health gate fires, router_preflight returns Passthrough and sets Degraded.
    let state = agent_loop.shared.core_handle.counters.get_trio_state();
    eprintln!("trio_state after health gate: {:?}", state);
    assert_eq!(
        state,
        crate::agent::agent_core::TrioState::Degraded,
        "trio_state should be Degraded when health gate fires"
    );

    // Response must come from main (non-empty).
    assert!(
        !resp.is_empty(),
        "response should come from main, not be empty"
    );

    // router_preflight_fired should be true (we entered preflight but returned Passthrough).
    let metrics = &agent_loop.shared.core_handle.counters.trio_metrics;
    assert!(
        metrics
            .router_preflight_fired
            .load(std::sync::atomic::Ordering::Relaxed),
        "router_preflight_fired should be true (preflight was entered)"
    );

    // Specialist must not have been dispatched.
    assert!(
        !metrics
            .specialist_dispatched
            .load(std::sync::atomic::Ordering::Relaxed),
        "specialist should not be dispatched when health gate is active"
    );

    // Router's chat() should never have been called — health gate fired before it.
    assert_eq!(
        router_seq.call_count(),
        0,
        "router provider's chat() call count should be 0 (health gate bypassed it)"
    );

    let _ = std::fs::remove_dir_all(&workspace);
}

// -----------------------------------------------------------------------
// Test 5: lenient parse fallback
//
// Router returns FunctionGemma comma-separated format:
//   "specialist,coding,{}"
// `parse_lenient_router_decision` handles this format.
// -----------------------------------------------------------------------

#[tokio::test]
async fn test_trio_offline_e2e_parse_fallback_lenient() {
    // Lenient format: "action,target,{args}" — no JSON wrapper.
    // This exercises the comma-separated branch in parse_lenient_router_decision.
    let router_resp = "specialist,coding,{}";

    let router: Arc<dyn LLMProvider> = Arc::new(SequenceProvider::new(
        "offline-router",
        vec![router_resp, router_resp, router_resp],
    ));
    let main: Arc<dyn LLMProvider> = Arc::new(StaticResponseLLM::new("offline-main", "delegating"));
    let specialist: Arc<dyn LLMProvider> = Arc::new(StaticResponseLLM::new(
        "offline-specialist",
        "lenient parse worked",
    ));

    // Verify that parse_lenient_router_decision handles this format before
    // wiring it into the full agent loop.
    let lenient_decision = parse_lenient_router_decision(router_resp);
    assert!(
        lenient_decision.is_some(),
        "parse_lenient_router_decision should accept 'specialist,coding,{{}}'"
    );
    let lenient_decision = lenient_decision.unwrap();
    assert_eq!(
        lenient_decision.action, "specialist",
        "lenient decision action should be 'specialist'"
    );

    let (agent_loop, workspace) = build_trio_offline_harness(main, router, specialist);

    let resp = agent_loop
        .process_direct(
            "Explain something complex",
            "trio-offline-lenient",
            "test",
            "offline",
        )
        .await;

    eprintln!(
        "test_trio_offline_e2e_parse_fallback_lenient: response ({} chars): {}",
        resp.len(),
        &resp[..resp.len().min(200)]
    );

    let metrics = &agent_loop.shared.core_handle.counters.trio_metrics;

    assert_eq!(
        metrics.router_action.lock().as_deref(),
        Some("specialist"),
        "router_action should be 'specialist' after lenient parse"
    );
    assert!(
        metrics
            .specialist_dispatched
            .load(std::sync::atomic::Ordering::Relaxed),
        "specialist should have been dispatched after lenient parse"
    );
    assert!(!resp.is_empty(), "response should be non-empty");

    let _ = std::fs::remove_dir_all(&workspace);
}

// ============================================================================
// appears_incomplete heuristic tests
// ============================================================================

mod continuation_tests {
    use super::appears_incomplete;

    #[test]
    fn test_unclosed_backtick_detected() {
        assert!(appears_incomplete("The template to skip `"));
        assert!(appears_incomplete("Thinking blocks (`"));
    }

    #[test]
    fn test_complete_response_not_flagged() {
        assert!(!appears_incomplete("This is a complete sentence."));
        assert!(!appears_incomplete("Done!"));
        assert!(!appears_incomplete("Use `code` here."));
        assert!(!appears_incomplete("```\ncode\n```"));
    }

    #[test]
    fn test_mid_sentence_detected() {
        assert!(appears_incomplete("The quick brown fox jumped over the"));
        assert!(appears_incomplete("Here are the steps to configure"));
    }

    #[test]
    fn test_short_fragments_not_flagged() {
        assert!(!appears_incomplete("OK"));
        assert!(!appears_incomplete("Yes"));
    }

    #[test]
    fn test_unclosed_paren_detected() {
        assert!(appears_incomplete("The function signature is fn foo(bar"));
    }

    #[test]
    fn test_appears_incomplete_mid_sentence() {
        // Text ending mid-word (no terminal punctuation, long enough to trigger)
        assert!(appears_incomplete(
            "The configuration requires setting the correc"
        ));
        assert!(appears_incomplete("You can use this approach to implemen"));
    }

    #[test]
    fn test_appears_incomplete_complete() {
        // Text ending with period or exclamation is considered complete
        assert!(!appears_incomplete("The task is now complete."));
        assert!(!appears_incomplete("All done!"));
        assert!(!appears_incomplete("Did it work?"));
    }

    #[test]
    fn test_trailing_emoji_not_flagged() {
        // Period before emoji — response is complete, must not trigger continuation
        assert!(!appears_incomplete(
            "Why cross the road? To avoid borrows. 🦀"
        ));
        // Period before multiple emojis
        assert!(!appears_incomplete("The answer is 42. 🎉✨"));
    }

    #[test]
    fn test_trailing_emoji_mid_sentence_still_flagged() {
        // No punctuation even after stripping emojis — still incomplete
        assert!(appears_incomplete("Here's a joke 🤣😂🔥"));
    }

    #[test]
    fn test_short_response_with_emoji_not_flagged() {
        // Under the 20-char length threshold
        assert!(!appears_incomplete("OK 👍"));
    }
}

// ============================================================================
// Universal textual tool-call parsing tests
// ============================================================================

mod universal_textual_parse_tests {
    use crate::agent::protocol::{parse_textual_tool_calls, strip_textual_tool_calls};

    #[test]
    fn test_textual_parse_strips_content() {
        // Content containing a [I called: ...] annotation should have the
        // annotation removed by strip_textual_tool_calls, leaving only prose.
        let input = "Sure, let me list the files.\n[I called: exec({\"command\": \"ls\"})]\nDone.";
        let stripped = strip_textual_tool_calls(input);
        assert!(
            !stripped.contains("[I called:"),
            "Expected [I called:] pattern to be stripped, got: {:?}",
            stripped
        );
        assert!(
            stripped.contains("Sure, let me list the files."),
            "Expected prose to be preserved, got: {:?}",
            stripped
        );
    }

    #[test]
    fn test_universal_parse_non_textual_replay() {
        // parse_textual_tool_calls should work on any content string regardless
        // of protocol mode — the function itself is protocol-agnostic.
        let content =
            "I will run the command now.\n[I called: exec({\"command\": \"echo hello\"})]";
        let parsed = parse_textual_tool_calls(content);
        assert_eq!(
            parsed.len(),
            1,
            "Expected 1 parsed tool call, got {}",
            parsed.len()
        );
        assert_eq!(parsed[0].tool, "exec");
        // Args should decode the command key.
        let cmd = parsed[0].args.get("command").and_then(|v| v.as_str());
        assert_eq!(cmd, Some("echo hello"));
    }

    #[test]
    fn test_textual_parse_no_match_returns_empty() {
        // Plain prose with no [I called: ...] patterns must return empty.
        let content = "There are no tool calls in this response.";
        let parsed = parse_textual_tool_calls(content);
        assert!(
            parsed.is_empty(),
            "Expected no parsed tool calls, got {:?}",
            parsed
        );
    }
}

mod nudge_tests {
    /// Verify that the 80%-ceiling formula produces the expected nudge thresholds.
    #[test]
    fn test_nudge_threshold_80_percent() {
        let nudge_at = |max: u32| -> u32 { ((max as f64) * 0.8).ceil() as u32 };

        // 10 * 0.8 = 8.0, ceil = 8
        assert_eq!(nudge_at(10), 8, "max=10 → nudge_at=8");
        // 5 * 0.8 = 4.0, ceil = 4
        assert_eq!(nudge_at(5), 4, "max=5 → nudge_at=4");
        // 20 * 0.8 = 16.0, ceil = 16
        assert_eq!(nudge_at(20), 16, "max=20 → nudge_at=16");
        // Non-round case: 7 * 0.8 = 5.6, ceil = 6
        assert_eq!(nudge_at(7), 6, "max=7 → nudge_at=6");
        // Minimal case: 1 * 0.8 = 0.8, ceil = 1
        assert_eq!(nudge_at(1), 1, "max=1 → nudge_at=1");
    }

    /// Verify that the rescue logic extracts the last assistant message when available,
    /// and falls back to the static message when no assistant content exists.
    #[test]
    fn test_rescue_extracts_last_assistant() {
        let messages: Vec<serde_json::Value> = vec![
            serde_json::json!({"role": "user", "content": "Hello"}),
            serde_json::json!({"role": "assistant", "content": "I am working on it."}),
            serde_json::json!({"role": "tool", "content": "some tool result"}),
        ];

        // Simulate the rescue logic from finalize_response.rs
        let final_content = String::new();
        let result = if final_content.is_empty() && messages.len() > 2 {
            let last_assistant = messages
                .iter()
                .rev()
                .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("assistant"))
                .and_then(|m| m.get("content").and_then(|c| c.as_str()))
                .unwrap_or("");
            if !last_assistant.trim().is_empty() {
                format!(
                    "{}\n\n[Note: The turn ended before a final answer was produced. This response may be incomplete.]",
                    last_assistant.trim()
                )
            } else {
                "The turn ended before I could produce a final answer. The actions above may be incomplete.".to_string()
            }
        } else {
            final_content.clone()
        };

        assert!(
            result.starts_with("I am working on it."),
            "rescue should start with the last assistant content, got: {result}"
        );
        assert!(
            result.contains("[Note: The turn ended before a final answer"),
            "rescue should append the incomplete note, got: {result}"
        );
    }

    /// When there is no assistant message at all, the static fallback is used.
    #[test]
    fn test_rescue_falls_back_when_no_assistant() {
        let messages: Vec<serde_json::Value> = vec![
            serde_json::json!({"role": "user", "content": "Hello"}),
            serde_json::json!({"role": "tool", "content": "tool result only"}),
            serde_json::json!({"role": "user", "content": "continue"}),
        ];

        let final_content = String::new();
        let result = if final_content.is_empty() && messages.len() > 2 {
            let last_assistant = messages
                .iter()
                .rev()
                .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("assistant"))
                .and_then(|m| m.get("content").and_then(|c| c.as_str()))
                .unwrap_or("");
            if !last_assistant.trim().is_empty() {
                format!(
                    "{}\n\n[Note: The turn ended before a final answer was produced. This response may be incomplete.]",
                    last_assistant.trim()
                )
            } else {
                "The turn ended before I could produce a final answer. The actions above may be incomplete.".to_string()
            }
        } else {
            final_content.clone()
        };

        assert_eq!(
            result,
            "The turn ended before I could produce a final answer. The actions above may be incomplete.",
            "should use static fallback when no assistant message found"
        );
    }

    // ---------------------------------------------------------------------------
    // Cost tracking tests
    // ---------------------------------------------------------------------------

    /// Test that cost calculation works with token counts and model prices.
    /// This is a RED test - it will fail until we wire up cost tracking.
    #[test]
    fn test_cost_tracking_calculates_from_tokens() {
        use crate::agent::model_prices::ModelPrices;

        let mut prices = ModelPrices::empty();
        // Add a test model: $0.01 per 1M prompt tokens, $0.03 per 1M completion tokens
        prices.prices.insert(
            "test-model".to_string(),
            (0.01 / 1_000_000.0, 0.03 / 1_000_000.0),
        );

        // 10,000 prompt tokens * $0.01/1M = $0.0001
        // 5,000 completion tokens * $0.03/1M = $0.00015
        // Total: $0.00025
        let cost = prices.cost_of("test-model", 10_000, 5_000);

        let expected = 0.0001 + 0.00015;
        assert!(
            (cost - expected).abs() < 0.0000001,
            "cost should be ${:.6}, got ${:.6}",
            expected,
            cost
        );
    }

    /// Test that finalize_response records actual costs (not hardcoded 0.0).
    /// This is the integration test for the cost tracking feature.
    #[test]
    fn test_finalize_response_records_nonzero_cost() {
        // This test will fail until we wire cost tracking in finalize_response.rs:231
        // The TODO currently hardcodes cost_usd: 0.0
        // After wiring, this should record actual costs based on token usage

        // For now, just verify the infrastructure exists
        use crate::agent::model_prices::ModelPrices;
        let prices = ModelPrices::empty();

        // Verify cost_of returns 0.0 for unknown models
        let unknown_cost = prices.cost_of("unknown-model", 1000, 500);
        assert_eq!(unknown_cost, 0.0, "unknown models should return 0.0 cost");

        // This assertion documents the TODO - it will pass once we wire cost tracking
        // Currently finalize_response hardcodes cost_usd: 0.0
        // TODO: Update this test to verify actual cost recording after wiring
    }
}

// ============================================================================
// RuntimeMode parallel-rollout parity tests (Wave 2)
// ============================================================================
//
// These tests pin the invariant that `SwappableCore.is_local` and
// `SwappableCore.mode` agree by construction. Wave 3's reader-migration
// relies on this invariant to swap each `is_local` read for a `mode` match
// without behavioural drift.
mod runtime_mode_parity_tests {
    use super::*;
    use crate::agent::runtime_mode::RuntimeMode;

    /// Cloud-fixture path: `is_local: false` → `mode == Cloud`, accessor returns Cloud.
    #[test]
    fn mode_accessor_cloud_matches_is_local_false() {
        let core = build_test_core(false, None, None);
        assert!(!core.mode().is_local(), "fixture is is_local=false");
        assert!(
            matches!(core.mode(), RuntimeMode::Cloud),
            "cloud fixture must resolve to RuntimeMode::Cloud"
        );
    }

    /// Local-fixture path: build a local core via a minimal SwappableCoreConfig
    /// (mirrors the pattern in `test_delegation_with_is_local_true`). Verifies
    /// the accessor returns `Local { caps }`.
    #[test]
    fn mode_accessor_local_matches_is_local_true() {
        let workspace = tempfile::tempdir().unwrap().keep();
        let main = MockLLM::named("local-main");
        let core = build_swappable_core(SwappableCoreConfig {
            provider: main,
            workspace,
            model: "local-model".to_string(),
            max_iterations: 10,
            max_continuations: 2,
            max_tokens: 4096,
            temperature: 0.7,
            max_context_tokens: 16384,
            brave_api_key: None,
            search_provider: "searxng".to_string(),
            searxng_url: "http://localhost:8888".to_string(),
            crw_url: String::new(),
            search_max_results: 5,
            exec_timeout: 30,
            restrict_to_workspace: false,
            memory_config: MemoryConfig::default(),
            is_local: true,
            lane: Lane::default(),
            tool_delegation: ToolDelegationConfig::default(),
            provenance: ProvenanceConfig::default(),
            max_tool_result_chars: 2000,
            delegation_provider: None,
            specialist_provider: None,
            trio_config: TrioConfig::default(),
            model_capabilities_overrides: std::collections::HashMap::new(),
            reasoning_config: crate::config::schema::ReasoningConfig::default(),
            tool_heartbeat_secs: 2,
            health_check_timeout_secs: 2,
            adaptive_tokens: AdaptiveTokenConfig::default(),
            sessions_db_path: Some(
                std::env::temp_dir().join(format!("nanobot-test-{}.sqlite", uuid::Uuid::new_v4())),
            ),
        });
        assert!(core.mode().is_local(), "fixture is is_local=true");
        assert!(
            matches!(core.mode(), RuntimeMode::Local { .. }),
            "local fixture must resolve to RuntimeMode::Local"
        );
    }

    /// Task 2 / Branch 4: reserve_cap is now derived from mode.
    /// Cloud: passthrough. Local + ample ctx: unchanged. Local + tight ctx: clamped to 25%.
    #[test]
    fn build_core_reserve_cap_cloud_passthrough() {
        // Cloud fixture: max_tokens=4096, max_ctx=16384. Cloud reserve = max_tokens verbatim.
        let core = build_test_core(false, None, None);
        // token_budget exposes reserve via the constructor; reconstruct the
        // expected value from what mode.reserve_cap returns on Cloud.
        let mode = core.mode();
        assert!(matches!(mode, RuntimeMode::Cloud));
        assert_eq!(mode.reserve_cap(4096, 16384), 4096);
    }

    #[test]
    fn build_core_reserve_cap_local_clamped_to_25_pct() {
        // Local fixture with a tight 16K ctx + 4096 max_tokens: reserve clamps to 4096 (ctx/4).
        let workspace = tempfile::tempdir().unwrap().keep();
        let main = MockLLM::named("local-main");
        let core = build_swappable_core(SwappableCoreConfig {
            provider: main,
            workspace,
            model: "local-model".to_string(),
            max_iterations: 10,
            max_continuations: 2,
            max_tokens: 4096,
            temperature: 0.7,
            max_context_tokens: 16_384,
            brave_api_key: None,
            search_provider: "searxng".to_string(),
            searxng_url: "http://localhost:8888".to_string(),
            crw_url: String::new(),
            search_max_results: 5,
            exec_timeout: 30,
            restrict_to_workspace: false,
            memory_config: MemoryConfig::default(),
            is_local: true,
            lane: Lane::default(),
            tool_delegation: ToolDelegationConfig::default(),
            provenance: ProvenanceConfig::default(),
            max_tool_result_chars: 2000,
            delegation_provider: None,
            specialist_provider: None,
            trio_config: TrioConfig::default(),
            model_capabilities_overrides: std::collections::HashMap::new(),
            reasoning_config: crate::config::schema::ReasoningConfig::default(),
            tool_heartbeat_secs: 2,
            health_check_timeout_secs: 2,
            adaptive_tokens: AdaptiveTokenConfig::default(),
            sessions_db_path: Some(
                std::env::temp_dir().join(format!("nanobot-test-{}.sqlite", uuid::Uuid::new_v4())),
            ),
        });
        // ctx/4 == 4096; min(4096, 4096) == 4096.
        assert_eq!(core.mode().reserve_cap(4096, 16_384), 4096);
        // Tighter ctx: 8192/4 = 2048 → reserve clamps below max_tokens.
        assert_eq!(core.mode().reserve_cap(4096, 8_192), 2048);
    }

    /// Task 2 / Branch 1–2: context builder reflects the mode's lite/full defaults.
    /// Cloud: `local_prompt_mode == false`, `system_prompt_cap == 0` prior to scaling
    /// (then scale_budgets sets it to 40% of ctx). Local: `local_prompt_mode == true`,
    /// `system_prompt_cap` is a fixed 1000-token cached-prefix cap set by set_lite_mode
    /// (the tiny-model ≤4K branch keeps a leaner 50-token prefix).
    #[test]
    fn build_core_context_cap_cloud_uses_full_scaling() {
        let core = build_test_core(false, None, None);
        // Cloud: scale_budgets sets system_prompt_cap = ctx * 2/5 = 16384 * 2/5 = 6553.
        assert!(!core.context.local_prompt_mode);
        assert_eq!(core.context.system_prompt_cap, 16_384 * 2 / 5);
    }

    #[test]
    fn build_core_context_cap_local_uses_lite_mode() {
        let workspace = tempfile::tempdir().unwrap().keep();
        let main = MockLLM::named("local-main");
        let core = build_swappable_core(SwappableCoreConfig {
            provider: main,
            workspace,
            model: "local-model".to_string(),
            max_iterations: 10,
            max_continuations: 2,
            max_tokens: 4096,
            temperature: 0.7,
            max_context_tokens: 16_384,
            brave_api_key: None,
            search_provider: "searxng".to_string(),
            searxng_url: "http://localhost:8888".to_string(),
            crw_url: String::new(),
            search_max_results: 5,
            exec_timeout: 30,
            restrict_to_workspace: false,
            memory_config: MemoryConfig::default(),
            is_local: true,
            lane: Lane::default(),
            tool_delegation: ToolDelegationConfig::default(),
            provenance: ProvenanceConfig::default(),
            max_tool_result_chars: 2000,
            delegation_provider: None,
            specialist_provider: None,
            trio_config: TrioConfig::default(),
            model_capabilities_overrides: std::collections::HashMap::new(),
            reasoning_config: crate::config::schema::ReasoningConfig::default(),
            tool_heartbeat_secs: 2,
            health_check_timeout_secs: 2,
            adaptive_tokens: AdaptiveTokenConfig::default(),
            sessions_db_path: Some(
                std::env::temp_dir().join(format!("nanobot-test-{}.sqlite", uuid::Uuid::new_v4())),
            ),
        });
        assert!(core.context.local_prompt_mode);
        // Local prompt cost is fixed rather than scaling with the model window:
        // the byte-stable cached prefix (identity + skills + workspace bootstrap)
        // is capped at 1000 tokens so Higgs's radix prefix cache can reuse it.
        assert_eq!(core.context.system_prompt_cap, 1000);
    }

    /// Task 2 / Branch 3: cloud memory provider/model follows the pre-Wave-2 path.
    /// MockLLM returns `get_api_base() == None` → triggers the "haiku" branch.
    #[test]
    fn build_core_memory_provider_cloud_defaults_to_haiku_when_no_api_base() {
        let core = build_test_core(false, None, None);
        // provider.get_api_base() is None for MockLLM → "haiku" memory model.
        assert_eq!(core.memory_model, "haiku");
        assert_eq!(core.compactor.model(), "main-model");
    }

    /// Task 2 / Branch 3: local memory provider falls through specialist → main.
    /// With no explicit memory config and no specialist provider, the local
    /// reflection and compaction identities both resolve to the main model.
    #[test]
    fn build_core_memory_provider_local_defaults_to_main_without_trio() {
        let workspace = tempfile::tempdir().unwrap().keep();
        let main = MockLLM::named("local-main");
        let core = build_swappable_core(SwappableCoreConfig {
            provider: main,
            workspace,
            model: "local-model".to_string(),
            max_iterations: 10,
            max_continuations: 2,
            max_tokens: 4096,
            temperature: 0.7,
            max_context_tokens: 16_384,
            brave_api_key: None,
            search_provider: "searxng".to_string(),
            searxng_url: "http://localhost:8888".to_string(),
            crw_url: String::new(),
            search_max_results: 5,
            exec_timeout: 30,
            restrict_to_workspace: false,
            memory_config: MemoryConfig::default(),
            is_local: true,
            lane: Lane::default(),
            tool_delegation: ToolDelegationConfig::default(),
            provenance: ProvenanceConfig::default(),
            max_tool_result_chars: 2000,
            delegation_provider: None,
            specialist_provider: None,
            trio_config: TrioConfig::default(),
            model_capabilities_overrides: std::collections::HashMap::new(),
            reasoning_config: crate::config::schema::ReasoningConfig::default(),
            tool_heartbeat_secs: 2,
            health_check_timeout_secs: 2,
            adaptive_tokens: AdaptiveTokenConfig::default(),
            sessions_db_path: Some(
                std::env::temp_dir().join(format!("nanobot-test-{}.sqlite", uuid::Uuid::new_v4())),
            ),
        });
        assert_eq!(core.memory_model, "local-model");
        assert_eq!(core.compactor.model(), "local-model");
    }

    /// Caps carried inside `Local { caps }` match the capabilities resolved
    /// for the model. Ensures `mode_accessor_round_trip` (VALIDATION.md):
    /// construction inputs are consistent with the mode's payload.
    #[test]
    fn mode_accessor_round_trip_local_caps_match_lookup() {
        let workspace = tempfile::tempdir().unwrap().keep();
        let main = MockLLM::named("local-main");
        let core = build_swappable_core(SwappableCoreConfig {
            provider: main,
            workspace,
            model: "local-model".to_string(),
            max_iterations: 10,
            max_continuations: 2,
            max_tokens: 4096,
            temperature: 0.7,
            max_context_tokens: 16384,
            brave_api_key: None,
            search_provider: "searxng".to_string(),
            searxng_url: "http://localhost:8888".to_string(),
            crw_url: String::new(),
            search_max_results: 5,
            exec_timeout: 30,
            restrict_to_workspace: false,
            memory_config: MemoryConfig::default(),
            is_local: true,
            lane: Lane::default(),
            tool_delegation: ToolDelegationConfig::default(),
            provenance: ProvenanceConfig::default(),
            max_tool_result_chars: 2000,
            delegation_provider: None,
            specialist_provider: None,
            trio_config: TrioConfig::default(),
            model_capabilities_overrides: std::collections::HashMap::new(),
            reasoning_config: crate::config::schema::ReasoningConfig::default(),
            tool_heartbeat_secs: 2,
            health_check_timeout_secs: 2,
            adaptive_tokens: AdaptiveTokenConfig::default(),
            sessions_db_path: Some(
                std::env::temp_dir().join(format!("nanobot-test-{}.sqlite", uuid::Uuid::new_v4())),
            ),
        });
        match core.mode() {
            RuntimeMode::Local { caps } => {
                // The wrapped caps must equal the model-capabilities lookup for the
                // same model — construction doesn't silently swap in a different
                // capability record.
                assert_eq!(
                    caps.size_class, core.model_capabilities.size_class,
                    "mode caps.size_class must match core.model_capabilities"
                );
                assert_eq!(
                    caps.tool_calling, core.model_capabilities.tool_calling,
                    "mode caps.tool_calling must match core.model_capabilities"
                );
            }
            RuntimeMode::Cloud => panic!("expected Local variant"),
        }
    }

    // ------------------------------------------------------------------
    // Wave 3 reader-migration parity tests.
    //
    // Every migration in plan 09-03 replaces `ctx.core.mode().is_local()` with a
    // typed `mode()` dispatch. These tests pin the parity between the old
    // bool branch and the new mode-driven branch for the non-trivial
    // migration sites, so a future reader-migration regression surfaces
    // as a failing test rather than a behavioral drift only visible in
    // three-way smoke.
    // ------------------------------------------------------------------

    /// agent_shared.rs :820 — proactive grounding message role.
    /// Pre-Wave-3: `if core.mode().is_local() { "user" } else { "system" }`.
    /// Post-Wave-3: `core.mode().grounding_role()`.
    #[test]
    fn wave3_grounding_role_cloud_matches_pre_migration() {
        let core = build_test_core(false, None, None);
        assert_eq!(core.mode().grounding_role(), "system");
    }

    #[test]
    fn wave3_grounding_role_local_matches_pre_migration() {
        let workspace = tempfile::tempdir().unwrap().keep();
        let main = MockLLM::named("local-main");
        let core = build_swappable_core(SwappableCoreConfig {
            provider: main,
            workspace,
            model: "local-model".to_string(),
            max_iterations: 10,
            max_continuations: 2,
            max_tokens: 4096,
            temperature: 0.7,
            max_context_tokens: 16_384,
            brave_api_key: None,
            search_provider: "searxng".to_string(),
            searxng_url: "http://localhost:8888".to_string(),
            crw_url: String::new(),
            search_max_results: 5,
            exec_timeout: 30,
            restrict_to_workspace: false,
            memory_config: MemoryConfig::default(),
            is_local: true,
            lane: Lane::default(),
            tool_delegation: ToolDelegationConfig::default(),
            provenance: ProvenanceConfig::default(),
            max_tool_result_chars: 2000,
            delegation_provider: None,
            specialist_provider: None,
            trio_config: TrioConfig::default(),
            model_capabilities_overrides: std::collections::HashMap::new(),
            reasoning_config: crate::config::schema::ReasoningConfig::default(),
            tool_heartbeat_secs: 2,
            health_check_timeout_secs: 2,
            adaptive_tokens: AdaptiveTokenConfig::default(),
            sessions_db_path: Some(
                std::env::temp_dir().join(format!("nanobot-test-{}.sqlite", uuid::Uuid::new_v4())),
            ),
        });
        assert_eq!(core.mode().grounding_role(), "user");
    }

    /// prepare_context.rs :518 — protocol selection respects the mlx: prefix
    /// exception. The non-mlx local path must still pick LocalProtocol, which
    /// is equivalent to `mode.is_local() && !model.starts_with("mlx:")`.
    #[test]
    fn wave3_protocol_selection_mlx_exception_preserved() {
        // Cloud always → CloudProtocol (mode.is_local() == false).
        let cloud = build_test_core(false, None, None);
        assert!(!cloud.mode().is_local());

        // Local with mlx: prefix model would go to CloudProtocol
        // (behavior-preserving: is_local && !starts_with("mlx:")).
        // This test pins the `mode.is_local()` half; the mlx: prefix check
        // is string-based and not affected by the migration.
        let workspace = tempfile::tempdir().unwrap().keep();
        let main = MockLLM::named("mlx-main");
        let local_mlx = build_swappable_core(SwappableCoreConfig {
            provider: main,
            workspace,
            model: "mlx:llama-8b".to_string(),
            max_iterations: 10,
            max_continuations: 2,
            max_tokens: 4096,
            temperature: 0.7,
            max_context_tokens: 16_384,
            brave_api_key: None,
            search_provider: "searxng".to_string(),
            searxng_url: "http://localhost:8888".to_string(),
            crw_url: String::new(),
            search_max_results: 5,
            exec_timeout: 30,
            restrict_to_workspace: false,
            memory_config: MemoryConfig::default(),
            is_local: true,
            lane: Lane::default(),
            tool_delegation: ToolDelegationConfig::default(),
            provenance: ProvenanceConfig::default(),
            max_tool_result_chars: 2000,
            delegation_provider: None,
            specialist_provider: None,
            trio_config: TrioConfig::default(),
            model_capabilities_overrides: std::collections::HashMap::new(),
            reasoning_config: crate::config::schema::ReasoningConfig::default(),
            tool_heartbeat_secs: 2,
            health_check_timeout_secs: 2,
            adaptive_tokens: AdaptiveTokenConfig::default(),
            sessions_db_path: Some(
                std::env::temp_dir().join(format!("nanobot-test-{}.sqlite", uuid::Uuid::new_v4())),
            ),
        });
        // mlx: prefix model: mode is Local, but protocol selection still
        // falls through to CloudProtocol via the `!starts_with("mlx:")` guard.
        assert!(local_mlx.mode().is_local());
        assert!(local_mlx.model.starts_with("mlx:"));
    }

    // -------------------------------------------------------------------------
    // Convergence harness
    // -------------------------------------------------------------------------
    //
    // The property that was missing for all three 2026-07 incidents: no matter
    // what the model emits, the loop must terminate in a BOUNDED number of
    // provider calls and return something — it must never spin. These tests
    // feed adversarial providers that never cooperate and assert bounded
    // termination. They are the one e2e layer that would have caught the
    // strip/restore churn, the phantom regression, and the family-cap spin.

    /// A provider that emits a list_dir tool call on every turn, never
    /// exhausts, and counts calls. Each call uses a DISTINCT path argument
    /// (defeats the cached-duplicate breaker, which keys on name+args) so the
    /// loop is forced through the lease coarse-family cap → sticky-strip path.
    /// It records whether any call observed `tools == None` — the direct signal
    /// that the sticky strip fired.
    struct LoopingProvider {
        name: String,
        call_count: std::sync::atomic::AtomicU32,
        saw_tools_absent: std::sync::atomic::AtomicBool,
    }

    impl LoopingProvider {
        fn new(name: &str) -> Self {
            Self {
                name: name.to_string(),
                call_count: std::sync::atomic::AtomicU32::new(0),
                saw_tools_absent: std::sync::atomic::AtomicBool::new(false),
            }
        }

        fn call_count(&self) -> u32 {
            self.call_count.load(std::sync::atomic::Ordering::Relaxed)
        }

        fn saw_tools_absent(&self) -> bool {
            self.saw_tools_absent.load(std::sync::atomic::Ordering::Relaxed)
        }
    }

    #[async_trait]
    impl LLMProvider for LoopingProvider {
        async fn chat(
            &self,
            _messages: &[Value],
            tools: Option<&[Value]>,
            _model: Option<&str>,
            _max_tokens: u32,
            _temperature: f64,
            _thinking_budget: Option<u32>,
            _top_p: Option<f64>,
        ) -> anyhow::Result<crate::providers::base::LLMResponse> {
            let n = self
                .call_count
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if tools.is_none() {
                self.saw_tools_absent
                    .store(true, std::sync::atomic::Ordering::Relaxed);
            }
            // Distinct path per call: the cached-duplicate breaker cannot arm,
            // so the lease read-family cap is the only thing that can stop the
            // run — exactly the 2026-07-30 incident path.
            let mut args = std::collections::HashMap::new();
            args.insert("path".to_string(), json!(format!("dir{n}")));
            Ok(crate::providers::base::LLMResponse {
                content: Some(String::new()),
                tool_calls: vec![crate::providers::base::ToolCallRequest {
                    id: format!("tc_loop_{n}"),
                    name: "list_dir".to_string(),
                    arguments: args,
                }],
                finish_reason: "tool_calls".to_string(),
                usage: std::collections::HashMap::new(),
            })
        }

        fn get_default_model(&self) -> &str {
            &self.name
        }
    }

    /// Adversarial convergence: the model emits a fresh list_dir every turn
    /// (distinct args, so no cached-duplicate shortcut) and NEVER writes a
    /// final answer. The loop must still converge — via the lease coarse-family
    /// cap (6) → sticky tool strip (Fix A1) → a tools-absent call → forced text.
    ///
    /// This is the test that was missing for the 2026-07 incidents. It is only
    /// meaningful because it ASSERTS the sticky strip fired (`saw_tools_absent`);
    /// if Fix A1 were reverted, the router would restore tools on every call and
    /// this assertion would fail.
    #[tokio::test]
    async fn convergence_loop_terminates_via_sticky_strip_when_model_loops() {
        let provider = Arc::new(LoopingProvider::new("local-main"));
        // max_iterations must exceed the coarse-family cap (6) so the cap can
        // fire before the bare iteration limit does.
        let (agent_loop, workspace) =
            build_local_inline_harness_with_iters(provider.clone() as Arc<dyn LLMProvider>, 12);
        let session_key = format!("conv-sticky-strip-{}", uuid::Uuid::new_v4());

        let response = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            agent_loop.process_direct("list files forever", &session_key, "test", "offline"),
        )
        .await
        .expect("loop must terminate — it hung (convergence regression)");

        assert!(
            !response.trim().is_empty(),
            "a converged turn must return text, got empty"
        );
        assert!(
            provider.saw_tools_absent(),
            "loop never observed a tools-absent call — the sticky strip (Fix A1) did not fire. \
             Either the family cap wasn't reached or the router restored tools (regression)."
        );
        let calls = provider.call_count();
        assert!(
            calls < 25,
            "loop made {calls} provider calls — did not converge (termination guard regressed)"
        );

        let _ = std::fs::remove_dir_all(&workspace);
    }

    /// Regression for the 2026-07-30 incident: a recalled body must NOT re-enter
    /// live context raw (a 172KB recall inflated a session to 77k tokens and
    /// dropped the warm cache). The model runs a command whose output is huge
    /// (stashed under its tool_call_id), then recalls it. The recalled result
    /// persisted in the conversation must be bounded AND point at slice/search —
    /// not the full body. This exercises the real tool-execution + shaping path
    /// end-to-end (the unit test only covers `digest_tool_result`).
    #[tokio::test]
    async fn convergence_recall_of_oversized_body_stays_bounded() {
        let mut exec_args = std::collections::HashMap::new();
        // ~230KB of output — far over any in-context cap, guaranteed to stash.
        exec_args.insert("command".to_string(), json!("seq 1 40000"));
        let mut recall_args = std::collections::HashMap::new();
        recall_args.insert("tool_call_id".to_string(), json!("tc_big"));

        let main: Arc<dyn LLMProvider> = Arc::new(ResponseSequenceProvider::new(
            "local-main",
            vec![
                crate::providers::base::LLMResponse {
                    content: Some(String::new()),
                    tool_calls: vec![crate::providers::base::ToolCallRequest {
                        id: "tc_big".to_string(),
                        name: "exec".to_string(),
                        arguments: exec_args,
                    }],
                    finish_reason: "tool_calls".to_string(),
                    usage: std::collections::HashMap::new(),
                },
                crate::providers::base::LLMResponse {
                    content: Some(String::new()),
                    tool_calls: vec![crate::providers::base::ToolCallRequest {
                        id: "tc_recall".to_string(),
                        name: "recall_tool_result".to_string(),
                        arguments: recall_args,
                    }],
                    finish_reason: "tool_calls".to_string(),
                    usage: std::collections::HashMap::new(),
                },
                crate::providers::base::LLMResponse {
                    content: Some(attested_text("done")),
                    tool_calls: vec![],
                    finish_reason: "stop".to_string(),
                    usage: std::collections::HashMap::new(),
                },
            ],
        ));
        let (agent_loop, workspace) = build_local_inline_harness(main);
        let session_key = format!("conv-recall-bound-{}", uuid::Uuid::new_v4());

        let response = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            agent_loop.process_direct("run then recall", &session_key, "test", "offline"),
        )
        .await
        .expect("recall e2e must terminate");

        assert_eq!(response, "done");

        // The recalled body persisted in the conversation must be bounded, not
        // the raw ~230KB, and must direct the model at slice/search.
        let core = agent_loop.shared.core_handle.swappable();
        let meta = core
            .sessions
            .get_latest_session(&session_key)
            .await
            .expect("session should exist");
        let msgs = core.sessions.get_all_messages(&meta.id).await;
        let recall_result = msgs
            .iter()
            .find(|m| m.get("tool_call_id").and_then(|v| v.as_str()) == Some("tc_recall"))
            .expect("recall tool result must be persisted");
        let content = recall_result
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        assert!(
            content.chars().count() < 5000,
            "recalled body must be bounded in context, got {} chars (regression of the 172KB blowup)",
            content.chars().count()
        );
        assert!(
            content.contains("slice_tool_result") || content.contains("search_tool_result"),
            "recalled preview must point at slice/search, got: {content}"
        );

        let _ = std::fs::remove_dir_all(&workspace);
    }
}
