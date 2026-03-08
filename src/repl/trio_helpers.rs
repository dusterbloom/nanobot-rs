//! Pure trio config helpers (testable without ReplContext).

/// Enable trio mode on a Config. Returns `true` if router/specialist are missing (needs warning).
pub(crate) fn trio_enable(cfg: &mut crate::config::schema::Config) -> bool {
    use crate::config::schema::DelegationMode;

    let needs_warning = cfg.trio.router_model.is_empty() || cfg.trio.specialist_model.is_empty();
    cfg.trio.enabled = true;
    cfg.tool_delegation.mode = DelegationMode::Trio;
    cfg.tool_delegation.apply_mode();
    needs_warning
}

/// Auto-detect router and specialist from available LM Studio models.
///
/// Scans model names for known patterns. Only picks models that aren't
/// the main model. Returns (router, specialist) — either may be None.
pub(crate) fn pick_trio_models(
    available: &[String],
    main_model: &str,
) -> (Option<String>, Option<String>) {
    let main_lower = main_model.to_lowercase();

    // Fuzzy "is this the main model?" — matches the same way is_model_available does:
    // exact, or either side is a substring of the other (handles org prefixes,
    // resolved-vs-config name mismatches, etc.)
    let is_main = |candidate_lower: &str| -> bool {
        if main_lower.is_empty() {
            return false;
        }
        candidate_lower == main_lower
            || candidate_lower.contains(&main_lower)
            || main_lower.contains(candidate_lower)
    };

    // Router detection (first match wins)
    let router_patterns: &[&str] = &["orchestrator", "router"];
    let router = router_patterns
        .iter()
        .find_map(|pat| {
            available.iter().find(|m| {
                let low = m.to_lowercase();
                low.contains(pat) && !is_main(&low)
            })
        })
        .cloned();

    // Specialist detection (first match wins, excludes main + router)
    let router_lower = router.as_ref().map(|r| r.to_lowercase());
    let specialist_patterns: &[&str] =
        &["function-calling", "functiongemma", "instruct", "ministral"];
    let specialist = specialist_patterns
        .iter()
        .find_map(|pat| {
            available.iter().find(|m| {
                let low = m.to_lowercase();
                low.contains(pat) && !is_main(&low) && router_lower.as_deref() != Some(low.as_str())
            })
        })
        .cloned();

    (router, specialist)
}

/// Whether trio mode should be auto-activated at REPL startup.
///
/// Returns `true` when all conditions hold:
/// - Running in local mode (`is_local`)
/// - Both router and specialist are configured (non-empty model name OR explicit endpoint)
/// - Currently in the default `DelegationMode::Delegated` state (i.e. the user has not
///   explicitly opted in to `Trio` or opted out via `Inline`)
pub(crate) fn should_auto_activate_trio(
    is_local: bool,
    router_model: &str,
    specialist_model: &str,
    has_router_endpoint: bool,
    has_specialist_endpoint: bool,
    current_mode: &crate::config::schema::DelegationMode,
) -> bool {
    use crate::config::schema::DelegationMode;

    let has_router = !router_model.is_empty() || has_router_endpoint;
    let has_specialist = !specialist_model.is_empty() || has_specialist_endpoint;
    // Only auto-activate from the default Delegated mode.
    // If the user explicitly set Inline or Trio, respect their choice.
    is_local && has_router && has_specialist && *current_mode == DelegationMode::Delegated
}

/// Disable trio mode on a Config, switching to inline (single model).
pub(crate) fn trio_disable(cfg: &mut crate::config::schema::Config) {
    use crate::config::schema::DelegationMode;

    cfg.trio.enabled = false;
    cfg.tool_delegation.mode = DelegationMode::Inline;
    cfg.tool_delegation.apply_mode();
}

/// Set a trio role model (router or specialist) on a Config.
pub(crate) fn set_trio_role_model_pure(
    cfg: &mut crate::config::schema::Config,
    role: &str,
    model: &str,
) {
    match role {
        "router" => cfg.trio.router_model = model.to_string(),
        "specialist" => cfg.trio.specialist_model = model.to_string(),
        _ => {}
    }
}

/// Copy trio-related fields from `live` config to `disk` config for persistence.
/// Does NOT touch non-trio fields (model, api keys, etc).
pub(crate) fn persist_trio_fields(
    live: &crate::config::schema::Config,
    disk: &mut crate::config::schema::Config,
) {
    disk.trio = live.trio.clone();
    disk.tool_delegation.mode = live.tool_delegation.mode;
    disk.tool_delegation.strict_no_tools_main = live.tool_delegation.strict_no_tools_main;
    disk.tool_delegation.strict_router_schema = live.tool_delegation.strict_router_schema;
    disk.tool_delegation.role_scoped_context_packs = live.tool_delegation.role_scoped_context_packs;
    disk.agents.defaults.local_max_context_tokens = live.agents.defaults.local_max_context_tokens;
}

// ============================================================================
// VRAM Budget Display (pure, testable)
// ============================================================================

/// Format a VRAM budget result for display. Pure function.
pub(crate) fn format_vram_budget(result: &crate::server::VramBudgetResult) -> String {
    use std::fmt::Write;
    let mut out = String::new();

    let cap_gb = result.effective_limit_bytes as f64 / 1e9;
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "  \x1b[1mVRAM BUDGET\x1b[0m  (limit: {:.1} GB)",
        cap_gb
    );
    let _ = writeln!(out, "  {}", "\u{2500}".repeat(52));

    for b in &result.breakdown {
        let role_upper = b.role.to_uppercase();
        let weights_gb = b.weights_bytes as f64 / 1e9;
        let kv_gb = b.kv_cache_bytes as f64 / 1e9;
        let total_gb = weights_gb + kv_gb + b.overhead_bytes as f64 / 1e9;
        let ctx_k = b.context_tokens / 1024;
        let _ = writeln!(
            out,
            "  {:<11} {:<20} {:.1} GB + {:.1} GB KV ({}K ctx) = {:.1} GB",
            role_upper, b.name, weights_gb, kv_gb, ctx_k, total_gb,
        );
    }

    let overhead_count = result.breakdown.len();
    let overhead_per = result.breakdown.first().map_or(0, |b| b.overhead_bytes);
    let overhead_total = overhead_count as f64 * overhead_per as f64 / 1e9;
    let _ = writeln!(
        out,
        "  {:<11} {} x {:.0} MB                                     = {:.1} GB",
        "OVERHEAD",
        overhead_count,
        overhead_per as f64 / 1e6,
        overhead_total,
    );

    let _ = writeln!(out, "  {}", "\u{2500}".repeat(52));
    let total_gb = result.total_vram_bytes as f64 / 1e9;
    let status = if result.fits {
        format!("\x1b[32mOK\x1b[0m")
    } else {
        format!("\x1b[31mOVER\x1b[0m")
    };
    let _ = writeln!(
        out,
        "  {:<11} {:.1} GB / {:.1} GB  {}",
        "TOTAL", total_gb, cap_gb, status,
    );
    let _ = writeln!(out);

    out
}

// ============================================================================
// Trio Parameter Helpers (pure, testable)
// ============================================================================

/// Apply a trio parameter change to a Config. Returns Ok(description) or Err(message).
pub(crate) fn apply_trio_param(
    config: &mut crate::config::schema::Config,
    role: &str,
    param: &str,
    value: &str,
) -> Result<String, String> {
    match (role, param) {
        ("router", "temperature") => {
            let v: f64 = value
                .parse()
                .map_err(|_| format!("Invalid number: {}", value))?;
            config.trio.router_temperature = v.clamp(0.0, 2.0);
            Ok(format!(
                "router temperature = {:.1}",
                config.trio.router_temperature
            ))
        }
        ("specialist", "temperature") => {
            let v: f64 = value
                .parse()
                .map_err(|_| format!("Invalid number: {}", value))?;
            config.trio.specialist_temperature = v.clamp(0.0, 2.0);
            Ok(format!(
                "specialist temperature = {:.1}",
                config.trio.specialist_temperature
            ))
        }
        ("router", "ctx") => {
            let ctx = super::super::parse_ctx_arg(value)
                .map_err(|e| e.to_string())?
                .ok_or_else(|| "Missing context size".to_string())?;
            config.trio.router_ctx_tokens = ctx;
            Ok(format!("router ctx = {}K", ctx / 1024))
        }
        ("specialist", "ctx") => {
            let ctx = super::super::parse_ctx_arg(value)
                .map_err(|e| e.to_string())?
                .ok_or_else(|| "Missing context size".to_string())?;
            config.trio.specialist_ctx_tokens = ctx;
            Ok(format!("specialist ctx = {}K", ctx / 1024))
        }
        ("router", "no_think") => {
            config.trio.router_no_think = !config.trio.router_no_think;
            Ok(format!("router no_think = {}", config.trio.router_no_think))
        }
        ("main", "no_think") => {
            config.trio.main_no_think = !config.trio.main_no_think;
            Ok(format!("main no_think = {}", config.trio.main_no_think))
        }
        ("trio", "vram_cap") => {
            let gb: f64 = value
                .parse()
                .map_err(|_| format!("Invalid number: {}", value))?;
            if gb < 1.0 || gb > 256.0 {
                return Err("VRAM cap must be between 1 and 256 GB".to_string());
            }
            config.trio.vram_cap_gb = gb;
            Ok(format!("vram cap = {:.1} GB", gb))
        }
        _ => Err(format!("Unknown parameter: {}.{}", role, param)),
    }
}
