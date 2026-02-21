# B12: Configuration Debt Elimination — Design Proposal

**Status:** Approved
**Author:** peppi
**Date:** 2026-02-21

## Problem

Audit found **60+ hardcoded magic values** across `src/`. Adding a config field touches 5 files (schema.rs field+default, Default impl, SwappableCoreConfig, build_swappable_core(), SwappableCore). Developers hardcode instead.

## Recommendation: Approach E (Hybrid A+D)

**Module-local config structs** (A) for numeric knobs + **Model capability registry** (D) for model name sniffing. Together covers ~95% of the 60+ values.

### Key Insight

Most hardcoded values are **static module-internal** — they never change at runtime. They don't need SwappableCore threading. Modules read their config section once at construction, store it locally. This drops the "5 files per knob" problem to **2 files** (define in schema.rs + use in module).

### Three Value Categories

| Category | Count | Solution | Threading |
|----------|-------|----------|-----------|
| Static module-internal | 50+ | Approach A: module-local config structs | Module reads at construction, no SwappableCore |
| Dynamic runtime | ~20 | Already in SwappableCore | No change needed |
| Model-dependent | 7+ sites | Approach D: ModelCapabilities registry | 1 field on SwappableCore, rebuilt on model swap |

## Part 1: ModelCapabilities Registry

New file: `src/agent/model_capabilities.rs` (~200 lines)

Replaces 7 model-name-sniffing sites across 5 files:
- `agent_core.rs` — `is_small_local_model()` → `caps.size_class == Small`
- `compaction.rs` — `ReaderProfile::from_model()` → `caps.reader_tier`
- `tool_runner.rs` — `scratch_pad_round_budget()` → `caps.scratch_pad_rounds`
- `openai_compat.rs` — `needs_native_lms_api()` → `caps.needs_native_lms_api`
- `thread_repair.rs` — strict alternation → `caps.strict_alternation`

```rust
pub struct ModelCapabilities {
    pub size_class: ModelSizeClass,        // Small/Medium/Large
    pub tool_calling: bool,                // default true
    pub thinking: bool,                    // supports <think> blocks
    pub needs_native_lms_api: bool,        // needs LMS /no_think endpoint
    pub strict_alternation: bool,          // requires user/assistant alternation
    pub max_reliable_output: usize,        // tokens before degradation
    pub scratch_pad_rounds: usize,         // tool runner analysis budget
    pub reader_tier: ReaderTier,           // Minimal/Standard/Advanced
}
```

Built-in pattern table:

| Pattern | size | thinking | strict_alt | max_output | rounds | reader |
|---------|------|----------|------------|------------|--------|--------|
| nanbeige | Small | no | yes | 512 | 3 | Minimal |
| ministral-3 | Small | no | yes | 1024 | 4 | Minimal |
| qwen3-1.7b | Small | yes | yes | 1024 | 4 | Minimal |
| nemotron/orchestrator | Medium | yes | no | 4096 | 10 | Standard |
| claude/gpt-4/gemini | Large | yes | no | 16384 | 10 | Advanced |
| (unknown default) | Medium | no | no | 4096 | 10 | Standard |

Config override in `config.json`:
```json
{ "modelCapabilities": { "my-custom-3b": { "sizeClass": "small", "maxReliableOutput": 512 } } }
```

## Part 2: Module-Local Config Structs

8 new small structs in `schema.rs`, nested under existing parent configs:

### SubagentTuning → nested under ToolDelegationConfig
- `max_iterations: 15`, `max_spawn_depth: 3`, `local_fallback_context: 8192`
- `local_min_context: 2048`, `local_max_response_tokens: 1024`, `local_min_response_tokens: 256`

### CircuitBreakerConfig → nested under TrioConfig
- `threshold: 3`, `cooldown_secs: 300`

### CompactionTuning → nested under MemoryConfig
- `max_merge_rounds: 6`

### SessionTuning → nested under MemoryConfig
- `rotation_size_bytes: 1_000_000`, `rotation_carry_messages: 10`

### ContextHygieneConfig → nested under MemoryConfig
- `keep_last_messages: 20`

### HeartbeatConfig → new top-level section
- `interval_secs: 300`, `degraded_threshold: 3`, `compaction_timeout_secs: 30`

### PipelineTuning → nested under ToolDelegationConfig
- `step_max_iterations: 5`, `max_tool_result_chars: 30_000`

### Audit extension → extend existing ProvenanceConfig
- `audit_max_result_size: 8192`

### NOT configuring (intentional)
**Timeouts at I/O call sites** — tightly coupled to I/O context. Exception: LMS load timeout (120s) → `AgentDefaults.lms_load_timeout_secs`.

## Implementation Phases

### Phase 1: ModelCapabilities (highest impact)
Create registry, add to SwappableCore, replace 5 sniffing sites. Tests per pattern.

### Phase 2: SubagentTuning + CircuitBreakerConfig
Add structs, replace constants, delete old consts.

### Phase 3: Session + Compaction + Hygiene
Add structs under MemoryConfig, replace constants.

### Phase 4: Health + Audit + Pipeline
HeartbeatConfig top-level, extend ProvenanceConfig, add PipelineTuning.

## Verification

Per phase:
1. `cargo test` — all green
2. Default assertion tests for every migrated value
3. Grep confirms old constants deleted
4. Existing config.json works unchanged (all `#[serde(default)]`)

Also read the file BACKLOG.md. Find the B12 entry and update it to add a line referencing this proposal: `Design: [docs/plans/b12-config-debt-elimination.md](docs/plans/b12-config-debt-elimination.md)`
