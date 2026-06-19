/// Returns true if a message is a synthetic router/specialist injection that
/// must not be merged with adjacent messages. Synthetic messages carry a
/// `_synthetic: true` field.
pub(crate) fn is_synthetic_injection(message: &serde_json::Value) -> bool {
    message
        .get("_synthetic")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

pub(crate) mod agent_core;
pub(crate) mod agent_loop;
pub(crate) mod agent_profiles;
pub(crate) mod anti_drift;
pub(crate) mod audit;
pub(crate) mod bulletin;
pub(crate) mod capabilities;
pub(crate) mod circuit_breaker;
pub mod compaction;
pub(crate) mod context;
pub mod context_gate;
pub(crate) mod context_hygiene;
pub(crate) mod context_store;
pub(crate) mod embedder;
pub(crate) mod finalize_response;
pub(crate) mod gateway_commands;
pub(crate) mod hooks;
pub(crate) mod instructions;
pub mod knowledge_graph;
pub mod knowledge_store;
pub(crate) mod lane;
pub mod lcm;
pub(crate) mod learning;
pub(crate) mod markers;
pub(crate) mod memory;
pub(crate) mod memory_ladder;
pub(crate) mod metrics;
pub(crate) mod model_capabilities;
pub(crate) mod model_prices;
pub(crate) mod observer;
pub(crate) mod pid_file;
pub(crate) mod pipeline;
pub(crate) mod policy;
pub(crate) mod prefix_guard;
pub(crate) mod prepare_context;
pub(crate) mod proactive;
pub(crate) mod prompt_contract;
pub(crate) mod prompt_fingerprint;
pub mod protocol;
pub(crate) mod provenance;
pub(crate) mod reasoning;
pub mod reflector;
pub(crate) mod role_policy;
pub mod router;
pub(crate) mod router_fallback;
pub(crate) mod runtime_mode;
pub(crate) mod sanitize;
pub mod session_indexer;
pub(crate) mod skills;
pub(crate) mod subagent;
pub(crate) mod system_state;
pub(crate) mod taint;
pub mod token_budget;
pub(crate) mod tool_engine;
pub(crate) mod tool_gate;
pub(crate) mod tool_guard;
pub(crate) mod tool_runner;
pub(crate) mod tool_wiring;
pub(crate) mod toolplan;
pub(crate) mod tools;
pub(crate) mod trace_store;
pub(crate) mod tuning;
pub mod turn;
pub(crate) mod validation;
pub(crate) mod worker_tools;
pub mod working_memory;
