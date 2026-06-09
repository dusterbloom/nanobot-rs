/// Returns true if a message is a synthetic router/specialist injection that
/// must not be merged with adjacent messages. Synthetic messages carry a
/// `_synthetic: true` field.
pub(crate) fn is_synthetic_injection(message: &serde_json::Value) -> bool {
    message
        .get("_synthetic")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

pub mod agent_core;
pub mod agent_loop;
pub mod agent_profiles;
pub mod anti_drift;
pub mod audit;
pub mod budget_calibrator;
pub mod bulletin;
pub mod capabilities;
pub mod circuit_breaker;
pub mod compaction;
pub mod context;
pub mod context_gate;
pub mod context_hygiene;
pub mod context_store;
pub mod embedder;
pub mod finalize_response;
pub(crate) mod gateway_commands;
pub mod hooks;
pub mod instructions;
pub mod knowledge_graph;
pub mod knowledge_store;
pub mod lane;
pub mod lcm;
pub mod learning;
pub mod markers;
pub mod memory;
pub mod memory_ladder;
pub mod metrics;
pub mod model_capabilities;
pub mod model_feature_cache;
pub mod model_prices;
pub mod observer;
pub(crate) mod pid_file;
pub mod pipeline;
pub mod policy;
pub mod prepare_context;
pub mod proactive;
pub mod prompt_contract;
pub mod prefix_guard;
pub mod prompt_fingerprint;
pub mod protocol;
pub mod provenance;
pub mod reasoning;
pub mod reflector;
pub mod role_policy;
pub mod router;
pub mod router_fallback;
pub mod runtime_mode;
pub mod sanitize;
pub mod session_indexer;
pub mod skills;
pub mod subagent;
pub mod system_state;
pub mod taint;
pub mod token_budget;
pub mod tool_engine;
pub mod tool_gate;
pub mod tool_guard;
pub mod tool_runner;
pub mod tool_wiring;
pub mod toolplan;
pub mod tools;
pub(crate) mod trace_store;
pub mod tuning;
pub mod turn;
pub mod validation;
pub mod worker_tools;
pub mod working_memory;
