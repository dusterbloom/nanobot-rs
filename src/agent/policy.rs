//! Session-level policy enforcement for reliable local tool execution.

use std::collections::HashMap;

use serde_json::Value;

/// Sticky per-session policy flags.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SessionPolicy {
    /// When true, all spawned subagents must run with a local model.
    pub local_only: bool,
}

/// Update session policy from user text. Returns true if policy changed.
pub fn update_from_user_text(policy: &mut SessionPolicy, text: &str) -> bool {
    let lower = text.to_lowercase();
    let mut changed = false;
    if lower.contains("always local")
        || lower.contains("strictly local")
        || lower.contains("only local")
        || lower.contains("local only")
    {
        if !policy.local_only {
            policy.local_only = true;
            changed = true;
        }
    }
    if lower.contains("cloud allowed")
        || lower.contains("allow cloud")
        || lower.contains("not local only")
    {
        if policy.local_only {
            policy.local_only = false;
            changed = true;
        }
    }
    changed
}

/// Whether the model string denotes a local model.
pub fn is_local_model(model: &str) -> bool {
    let m = model.trim().to_lowercase();
    m == "local" || m.starts_with("local:")
}

/// Enforce local-only policy for spawned subagent model overrides.
///
/// Returns the effective model value to pass to spawn.
pub fn enforce_subagent_model(policy: &SessionPolicy, requested: Option<String>) -> Option<String> {
    if !policy.local_only {
        return requested;
    }
    match requested {
        Some(m) if is_local_model(&m) => Some(m),
        _ => Some("local".to_string()),
    }
}

/// Validate that a spawn tool request has a non-empty task.
pub fn validate_spawn_args(params: &HashMap<String, Value>) -> Result<(), String> {
    let action = params
        .get("action")
        .and_then(|v| v.as_str())
        .unwrap_or("spawn");
    if action != "spawn" {
        return Ok(());
    }
    let task = params
        .get("task")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim();
    if task.is_empty() {
        return Err("Tool 'spawn' requires non-empty 'task' parameter".to_string());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_update_from_user_text_local_only() {
        let mut p = SessionPolicy::default();
        assert!(update_from_user_text(&mut p, "Always local"));
        assert!(p.local_only);
        assert!(!update_from_user_text(&mut p, "Always local"));
    }

    #[test]
    fn test_enforce_subagent_model() {
        let p = SessionPolicy { local_only: true };
        assert_eq!(
            enforce_subagent_model(&p, Some("claude-opus-4-6".to_string())),
            Some("local".to_string())
        );
        assert_eq!(
            enforce_subagent_model(&p, Some("local:nano".to_string())),
            Some("local:nano".to_string())
        );
    }
}

