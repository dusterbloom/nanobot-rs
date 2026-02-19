//! Shared proprioception: SystemState, TaskPhase, and ensemble coordination types.
//!
//! Every model in the ensemble can sense the system's current state through
//! `SystemState`. This enables phase-aware tool scoping, heartbeat grounding,
//! and priority interrupt signaling.

use std::time::Instant;

// ---------------------------------------------------------------------------
// Task Phase
// ---------------------------------------------------------------------------

/// The current phase of the agent's task, inferred from recent tool usage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskPhase {
    Idle,
    Understanding,
    Planning,
    FileEditing,
    CodeExecution,
    WebResearch,
    Communication,
    Reflection,
}

impl std::fmt::Display for TaskPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TaskPhase::Idle => write!(f, "idle"),
            TaskPhase::Understanding => write!(f, "understanding"),
            TaskPhase::Planning => write!(f, "planning"),
            TaskPhase::FileEditing => write!(f, "file editing"),
            TaskPhase::CodeExecution => write!(f, "code execution"),
            TaskPhase::WebResearch => write!(f, "web research"),
            TaskPhase::Communication => write!(f, "communication"),
            TaskPhase::Reflection => write!(f, "reflection"),
        }
    }
}

impl TaskPhase {
    /// Parse a phase name from a string (for `set_phase` micro-tool).
    pub fn from_str_loose(s: &str) -> Option<TaskPhase> {
        match s.to_lowercase().trim() {
            "idle" => Some(TaskPhase::Idle),
            "understanding" => Some(TaskPhase::Understanding),
            "planning" => Some(TaskPhase::Planning),
            "file_editing" | "fileediting" | "file editing" => Some(TaskPhase::FileEditing),
            "code_execution" | "codeexecution" | "code execution" => Some(TaskPhase::CodeExecution),
            "web_research" | "webresearch" | "web research" => Some(TaskPhase::WebResearch),
            "communication" => Some(TaskPhase::Communication),
            "reflection" => Some(TaskPhase::Reflection),
            _ => None,
        }
    }
}

/// Infer the current task phase from the last N tool calls.
///
/// Most-specific match wins. Examines the last 3 tools to determine phase.
pub fn infer_phase(recent_tools: &[&str]) -> TaskPhase {
    if recent_tools.is_empty() {
        return TaskPhase::Idle;
    }

    let last3: Vec<&str> = recent_tools.iter().rev().take(3).copied().collect();

    // Most-specific first: check for file editing (read + write/edit combo).
    let has_file_write = last3
        .iter()
        .any(|t| *t == "write_file" || *t == "edit_file");
    let has_file_read = last3.iter().any(|t| *t == "read_file" || *t == "list_dir");
    if has_file_write {
        return TaskPhase::FileEditing;
    }

    // Code execution.
    if last3.iter().any(|t| *t == "exec") {
        return TaskPhase::CodeExecution;
    }

    // Web research.
    if last3
        .iter()
        .any(|t| *t == "web_search" || *t == "web_fetch")
    {
        return TaskPhase::WebResearch;
    }

    // Communication.
    if last3.iter().any(|t| *t == "message" || *t == "send_email") {
        return TaskPhase::Communication;
    }

    // Planning (spawn = delegating work).
    if last3.iter().any(|t| *t == "spawn") {
        return TaskPhase::Planning;
    }

    // Read-only = understanding.
    if has_file_read {
        return TaskPhase::Understanding;
    }

    // Recall/skill = reflection.
    if last3.iter().any(|t| *t == "recall" || *t == "read_skill") {
        return TaskPhase::Reflection;
    }

    TaskPhase::Idle
}

// ---------------------------------------------------------------------------
// System State
// ---------------------------------------------------------------------------

/// Shared system state that every model can sense (proprioception).
#[derive(Debug, Clone)]
pub struct SystemState {
    pub task_phase: TaskPhase,
    /// Context pressure: 0.0 = empty, 1.0 = full.
    pub context_pressure: f32,
    pub turn_number: u64,
    pub message_count: u64,
    pub turns_since_compaction: u32,
    pub delegation_healthy: bool,
    pub recent_tool_failures: u32,
    pub last_tool_ok: bool,
    pub active_subagents: u8,
    pub pending_aha_signals: u8,
    pub updated_at: Instant,
}

impl SystemState {
    /// Create a new state snapshot from runtime data.
    pub fn snapshot(
        task_phase: TaskPhase,
        context_used: u64,
        context_max: u64,
        turn_number: u64,
        message_count: u64,
        turns_since_compaction: u32,
        delegation_healthy: bool,
        recent_tool_failures: u32,
        last_tool_ok: bool,
        active_subagents: u8,
        pending_aha_signals: u8,
    ) -> Self {
        let pressure = if context_max > 0 {
            (context_used as f32 / context_max as f32).clamp(0.0, 1.0)
        } else {
            0.0
        };
        Self {
            task_phase,
            context_pressure: pressure,
            turn_number,
            message_count,
            turns_since_compaction,
            delegation_healthy,
            recent_tool_failures,
            last_tool_ok,
            active_subagents,
            pending_aha_signals,
            updated_at: Instant::now(),
        }
    }
}

impl Default for SystemState {
    fn default() -> Self {
        Self {
            task_phase: TaskPhase::Idle,
            context_pressure: 0.0,
            turn_number: 0,
            message_count: 0,
            turns_since_compaction: 0,
            delegation_healthy: true,
            recent_tool_failures: 0,
            last_tool_ok: true,
            active_subagents: 0,
            pending_aha_signals: 0,
            updated_at: Instant::now(),
        }
    }
}

// ---------------------------------------------------------------------------
// Heartbeat / Grounding (Phase 4)
// ---------------------------------------------------------------------------

/// Format a grounding message from the current system state.
pub fn format_grounding(state: &SystemState) -> String {
    format!(
        "[grounding] Turn {}. Context: {:.0}% used. Phase: {}. \
         Delegation: {}. Subagents: {}.{}",
        state.turn_number,
        state.context_pressure * 100.0,
        state.task_phase,
        if state.delegation_healthy {
            "ok"
        } else {
            "down"
        },
        state.active_subagents,
        if state.pending_aha_signals > 0 {
            format!(" {} pending signals.", state.pending_aha_signals)
        } else {
            String::new()
        }
    )
}

/// Determine if a grounding message should be injected this iteration.
///
/// Higher context pressure = less frequent grounding (save tokens).
/// Returns false if pressure > 0.85 (too tight, skip entirely).
pub fn should_ground(iteration: u32, base_interval: u32, pressure: f32) -> bool {
    if base_interval == 0 {
        return false;
    }
    if pressure > 0.85 {
        return false; // too tight, skip
    }
    let interval = if pressure > 0.7 {
        base_interval * 2
    } else {
        base_interval
    };
    iteration > 0 && iteration % interval == 0
}

// ---------------------------------------------------------------------------
// Aha Channel (Phase 6)
// ---------------------------------------------------------------------------

/// Priority level for subagent signals.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AhaPriority {
    Critical,
    High,
    Normal,
}

/// A priority signal from a subagent.
#[derive(Debug, Clone)]
pub struct AhaSignal {
    pub priority: AhaPriority,
    pub agent_id: String,
    pub category: String,
    pub message: String,
}

/// Classify signal priority from subagent output content.
///
/// Returns `None` if no signal-worthy content is detected.
pub fn classify_signal(content: &str) -> Option<AhaPriority> {
    let lower = content.to_lowercase();

    // Critical: errors, failures, security issues.
    if lower.contains("error")
        || lower.contains("failed")
        || lower.contains("panic")
        || lower.contains("security")
        || lower.contains("vulnerability")
        || lower.contains("unsafe")
    {
        return Some(AhaPriority::Critical);
    }

    // High: discoveries, insights.
    if lower.contains("found") || lower.contains("discovered") || lower.contains("insight") {
        return Some(AhaPriority::High);
    }

    // Normal: completion signals.
    if lower.contains("complete") || lower.contains("done") || lower.contains("finished") {
        return Some(AhaPriority::Normal);
    }

    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- infer_phase tests --

    #[test]
    fn test_infer_phase_file_editing() {
        assert_eq!(
            infer_phase(&["read_file", "edit_file", "write_file"]),
            TaskPhase::FileEditing
        );
    }

    #[test]
    fn test_infer_phase_code_execution() {
        assert_eq!(infer_phase(&["exec"]), TaskPhase::CodeExecution);
    }

    #[test]
    fn test_infer_phase_web_research() {
        assert_eq!(infer_phase(&["web_search"]), TaskPhase::WebResearch);
    }

    #[test]
    fn test_infer_phase_empty() {
        assert_eq!(infer_phase(&[]), TaskPhase::Idle);
    }

    #[test]
    fn test_infer_phase_communication() {
        assert_eq!(infer_phase(&["message"]), TaskPhase::Communication);
    }

    #[test]
    fn test_infer_phase_planning() {
        assert_eq!(infer_phase(&["spawn"]), TaskPhase::Planning);
    }

    #[test]
    fn test_infer_phase_understanding() {
        assert_eq!(infer_phase(&["read_file"]), TaskPhase::Understanding);
    }

    #[test]
    fn test_infer_phase_reflection() {
        assert_eq!(infer_phase(&["recall"]), TaskPhase::Reflection);
    }

    #[test]
    fn test_infer_phase_most_recent_wins() {
        // exec after read_file should be CodeExecution, not Understanding
        assert_eq!(
            infer_phase(&["read_file", "read_file", "exec"]),
            TaskPhase::CodeExecution
        );
    }

    // -- SystemState::snapshot tests --

    #[test]
    fn test_snapshot_pressure_computation() {
        let state = SystemState::snapshot(
            TaskPhase::Idle,
            50_000,
            100_000,
            5,
            10,
            3,
            true,
            0,
            true,
            0,
            0,
        );
        assert!((state.context_pressure - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_snapshot_zero_max_context() {
        let state = SystemState::snapshot(TaskPhase::Idle, 50_000, 0, 1, 2, 0, true, 0, true, 0, 0);
        assert_eq!(state.context_pressure, 0.0);
    }

    #[test]
    fn test_snapshot_clamped_pressure() {
        let state = SystemState::snapshot(
            TaskPhase::Idle,
            200_000,
            100_000,
            1,
            2,
            0,
            true,
            0,
            true,
            0,
            0,
        );
        assert_eq!(state.context_pressure, 1.0);
    }

    // -- should_ground tests --

    #[test]
    fn test_should_ground_at_interval() {
        assert!(should_ground(8, 8, 0.5));
    }

    #[test]
    fn test_should_ground_too_tight() {
        assert!(!should_ground(8, 8, 0.9));
    }

    #[test]
    fn test_should_ground_not_time_yet() {
        assert!(!should_ground(4, 8, 0.5));
    }

    #[test]
    fn test_should_ground_disabled() {
        assert!(!should_ground(8, 0, 0.5));
    }

    #[test]
    fn test_should_ground_high_pressure_doubles_interval() {
        // At pressure 0.75, interval doubles from 8 to 16.
        assert!(!should_ground(8, 8, 0.75));
        assert!(should_ground(16, 8, 0.75));
    }

    #[test]
    fn test_should_ground_iteration_zero() {
        assert!(!should_ground(0, 8, 0.5));
    }

    // -- format_grounding tests --

    #[test]
    fn test_format_grounding_contains_key_info() {
        let state = SystemState::snapshot(
            TaskPhase::FileEditing,
            67_000,
            100_000,
            12,
            25,
            5,
            true,
            0,
            true,
            1,
            0,
        );
        let text = format_grounding(&state);
        assert!(text.contains("Turn 12"));
        assert!(text.contains("67%"));
        assert!(text.contains("file editing"));
        assert!(text.contains("Subagents: 1"));
    }

    #[test]
    fn test_format_grounding_with_signals() {
        let state =
            SystemState::snapshot(TaskPhase::Idle, 0, 100_000, 1, 1, 0, true, 0, true, 0, 3);
        let text = format_grounding(&state);
        assert!(text.contains("3 pending signals"));
    }

    // -- classify_signal tests --

    #[test]
    fn test_classify_signal_critical_error() {
        assert_eq!(
            classify_signal("error: compilation failed"),
            Some(AhaPriority::Critical)
        );
    }

    #[test]
    fn test_classify_signal_high_discovery() {
        assert_eq!(
            classify_signal("found the root cause"),
            Some(AhaPriority::High)
        );
    }

    #[test]
    fn test_classify_signal_normal_completion() {
        assert_eq!(classify_signal("task complete"), Some(AhaPriority::Normal));
    }

    #[test]
    fn test_classify_signal_none() {
        assert_eq!(classify_signal("reading file"), None);
    }

    #[test]
    fn test_classify_signal_security() {
        assert_eq!(
            classify_signal("potential security vulnerability detected"),
            Some(AhaPriority::Critical)
        );
    }

    // -- TaskPhase display + parse --

    #[test]
    fn test_task_phase_display() {
        assert_eq!(format!("{}", TaskPhase::FileEditing), "file editing");
        assert_eq!(format!("{}", TaskPhase::Idle), "idle");
    }

    #[test]
    fn test_task_phase_from_str_loose() {
        assert_eq!(
            TaskPhase::from_str_loose("file_editing"),
            Some(TaskPhase::FileEditing)
        );
        assert_eq!(
            TaskPhase::from_str_loose("FileEditing"),
            Some(TaskPhase::FileEditing)
        );
        assert_eq!(TaskPhase::from_str_loose("idle"), Some(TaskPhase::Idle));
        assert_eq!(TaskPhase::from_str_loose("nonsense"), None);
    }
}
