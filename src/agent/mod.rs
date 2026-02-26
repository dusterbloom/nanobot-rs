/// Returns true if a message is a synthetic router/specialist injection that
/// must not be merged with adjacent messages. Synthetic messages carry a
/// `_synthetic: true` field.
pub(crate) fn is_synthetic_injection(message: &serde_json::Value) -> bool {
    message.get("_synthetic").and_then(|v| v.as_bool()).unwrap_or(false)
}

pub mod agent_core;
pub mod agent_loop;
pub mod agent_profiles;
pub mod instructions;
pub mod anti_drift;
pub mod audit;
pub mod budget_calibrator;
pub mod bulletin;
pub mod circuit_breaker;
pub mod compaction;
pub mod confidence_gate;
pub mod context;
pub mod context_gate;
pub mod context_hygiene;
pub mod context_store;
pub mod eval;
pub mod knowledge_store;
pub mod lcm;
pub mod learning;
pub mod lora_bridge;
pub mod memory;
pub mod markers;
pub mod metrics;
pub mod model_capabilities;
pub mod model_feature_cache;
pub mod model_prices;
pub mod observer;
pub mod pipeline;
pub mod protocol;
pub mod policy;
pub mod process_tree;
pub mod provenance;
pub mod reflector;
pub mod role_policy;
pub(crate) mod trace_store;
pub mod router;
pub mod sanitize;
pub mod router_fallback;
pub mod session_indexer;
pub mod skills;
pub mod step_voter;
pub mod taint;
pub mod subagent;
pub mod system_state;
pub mod thread_repair;
pub mod tool_engine;
pub mod tool_guard;
pub mod tool_wiring;
pub mod toolplan;
pub mod token_budget;
pub mod tool_runner;
pub mod tools;
pub mod turn;
pub mod tuning;
pub mod validation;
pub mod worker_tools;
pub mod working_memory;
pub(crate) mod gateway_commands;
