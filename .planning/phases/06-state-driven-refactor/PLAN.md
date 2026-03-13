# Phase 06: State-Driven Refactor

**Goal:** Replace if/else cascades, duplicated fallback chains, and boolean-flag state tracking with proper enums, strategy traits, and event-driven dispatch across the nanobot codebase.

**Scope:** ~35 refactoring targets across 12 files, organized into 4 execution waves by blast radius.

**Baseline (2026-03-13):** `quality-sentinel.sh --all` → **1,440 warnings** (G1: 133, G3: 820, G4: 445, G5: 42). Target: reduce by 50%+ across all gates.

**Prerequisite:** v0.1.1 contracts (PromptContract, ToolGate, MemoryLadder, LearnLoop, ParserCanon, LaneSplit) — all shipped.

**Aligns with:** PROJECT.md active requirements MODE-01, MODE-02, PARSE-02, PARSE-03.

---

## Systemic Anti-Patterns Identified

### 1. Boolean Blindness (`is_local` cascade)
A single `bool` propagates through agent_core.rs, context.rs, cmd_read.rs, agent_shared.rs — controlling 20+ independent decisions. Each location re-tests the same boolean.

### 2. Priority Fallback Chains
"Try A, else try B, else try C, else default" appears in: memory provider (agent_core), delegation provider (agent_core), model resolution (subagent), TTS engine (voice_pipeline ×3), model dir (cmd_read ×2).

### 3. Implicit State Machines
Boolean flags in loops where enum state machines would be clearer: `in_thinking_block`, `in_code_block`, `async_compaction_pending`, `dirty`+`accumulated`, `nudge_sent`+`consecutive_empty`.

---

## Wave 1 — RuntimeMode Spine

**Impact:** Kills ~20 if/else branches across 4 files. Addresses MODE-01/MODE-02 from PROJECT.md.

### 1.1 Define `RuntimeMode` enum + `ModeStrategy` trait
- **File:** `src/agent/mod.rs` or new `src/agent/runtime_mode.rs`
- **What:** `enum RuntimeMode { Local(LocalConfig), Cloud(CloudConfig) }` with trait:
  ```rust
  trait ModeStrategy {
      fn memory_provider(&self) -> Arc<dyn LLMProvider>;
      fn token_reserve(&self, max: u32) -> usize;
      fn delegation_provider(&self) -> Option<Arc<dyn LLMProvider>>;
      fn delegation_model(&self) -> Option<String>;
      fn context_window_default(&self) -> usize;
      fn system_prompt_cap_pct(&self) -> f64;
  }
  ```

### 1.2 Refactor `agent_core.rs:441-556` — Provider fallback chains
- **Current:** 16+ branches across memory provider (441-491) and delegation provider (527-556)
- **Target:** Each `ModeStrategy` impl resolves providers once at construction
- **Kills:** `is_local` parameter from `build_core()`, 5 separate if/else chains

### 1.3 Refactor `context.rs:909-979` — Message assembly strategy
- **Current:** 2 completely different 260-line code paths behind `if is_local`
- **Target:** `trait MessageAssemblyStrategy` with `LocalAssembly` and `CloudAssembly` impls
- **Kills:** The largest if/else in the codebase (144 combined branches in context.rs)

### 1.4 Refactor `context.rs:506+564` — Context window calculation
- **Current:** Duplicated with different defaults (16K vs 128K) and different cap percentages (0.3 vs 0.4)
- **Target:** `ModeStrategy::context_window_default()` + `system_prompt_cap_pct()`
- **Kills:** 2 duplicated if/else + magic constants

### 1.5 Refactor `cmd_read.rs:11-40` — Mode detection
- **Current:** if/else chain computing mode_label and lane_label from booleans
- **Target:** Derive from `RuntimeMode` enum
- **Kills:** 6 branches

### UAT
- [ ] `cargo test` passes with no regressions
- [ ] `RuntimeMode` resolves correctly for: cloud, local (LM Studio), local (MLX), local (oMLX)
- [ ] No remaining `is_local: bool` parameter in agent_core.rs build functions
- [ ] Context assembly produces identical output for both modes

---

## Wave 2 — Deduplicate Fallback Chains

**Impact:** Kills copy-paste bugs and makes fallback logic testable.

### 2.1 Extract `resolve_model_dir()` helper
- **File:** `src/repl/cmd_read.rs:938-950` and `1083-1097`
- **Current:** Two identical cfg-gated blocks for model dir resolution (copy-paste)
- **Target:** Single `fn resolve_model_dir(mlx_handle: Option<&MlxHandle>) -> Result<PathBuf>`

### 2.2 Consolidate TTS engine selection
- **File:** `src/voice_pipeline.rs:926-946`, `1164-1181`, `1408-1448`
- **Current:** Language→engine routing duplicated 3 times
- **Target:** Single `fn select_tts_engine(&self, lang: &str) -> Result<(Arc<Mutex<dyn TtsEngine>>, String)>`
  - Returns (engine, voice_id) pair
  - Encapsulates English-first vs multilingual-first fallback

### 2.3 Share escalation logic in LCM
- **File:** `src/agent/lcm.rs:925-965`
- **Current:** Level 1 and Level 2 escalation are identical 3-branch if/else
- **Target:** `enum EscalationResult { Accepted(String), Refusal, SizeFailure }` + shared `try_escalate()` fn

### 2.4 Merge head/tail truncation
- **File:** `src/agent/context.rs:1500-1557`
- **Current:** Two functions sharing 80% of code (guards + budget loop)
- **Target:** `enum TruncationStrategy { Head, Tail }` parameterizing single `truncate_to_budget()`

### 2.5 Extract `SkillDisclosureMode` enum
- **File:** `src/agent/context.rs:326-335` and `842-850`
- **Current:** String-based mode normalization duplicated in 2 places
- **Target:** `enum SkillDisclosureMode { Eager, Xml, Compact }` with `FromStr` impl, validated at init

### UAT
- [ ] `cargo test` passes
- [ ] No duplicated blocks remain in cmd_read.rs, voice_pipeline.rs, lcm.rs, context.rs
- [ ] Each deduplication target has a single source of truth

---

## Wave 3 — Bool Flags to State Enums

**Impact:** Makes implicit state machines explicit and testable.

### 3.1 `ThinkingState` enum
- **File:** `src/voice_pipeline.rs:384-410`
- **Current:** `in_thinking_block: bool` controlling 3×2 nested if/else in strip_thinking_from_buffer()
- **Target:**
  ```rust
  enum ThinkingState { Outside, Inside }
  // match state { Outside => look for <thinking>, Inside => look for </thinking> }
  ```

### 3.2 `ExtractionState` enum
- **File:** `src/voice_pipeline.rs:483-527`
- **Current:** `in_code_block: bool` controlling sentence extraction loop
- **Target:**
  ```rust
  enum ExtractionState { InText, InCodeBlock }
  ```

### 3.3 `DelegationHealthState` enum
- **File:** `src/agent/tool_engine.rs:176-216`
- **Current:** `AtomicBool` + `AtomicU64` counter managing 4 health states
- **Target:**
  ```rust
  enum DelegationHealthState { Healthy, Degraded { retries_since_probe: u32 }, Dead }
  ```

### 3.4 `CompactionState` enum (absorb `async_compaction_pending` flag)
- **File:** `src/agent/lcm.rs:198, 430-436`
- **Current:** Bool flag `async_compaction_pending` + if/else in `check_compaction_needed()`
- **Target:**
  ```rust
  enum CompactionState { Idle, AsyncPending, Compacting }
  ```

### 3.5 `ModelResolution` enum for subagent fallback
- **File:** `src/agent/subagent.rs:364-381`
- **Current:** 4-way nested if/else with warnings as side effects
- **Target:**
  ```rust
  enum ModelResolution { Override(String), Profile(String), Default(String), MainFallback(String) }
  fn resolve_model(...) -> ModelResolution { ... }
  ```

### UAT
- [ ] `cargo test` passes
- [ ] No `bool` flags remain for: thinking_block, code_block, compaction_pending, delegation_alive
- [ ] Each new enum has at least one unit test for state transitions

---

## Wave 4 — Event-Driven Dispatch

**Impact:** Makes the main loop and engine init event-driven and testable.

### 4.1 `AgentEvent` classifier enum
- **File:** `src/agent/agent_loop.rs:476-750`
- **Current:** 6-way linear if/else testing message type
- **Target:**
  ```rust
  enum AgentEvent {
      System(OutboundMessage),
      Command { name: String, args: String },
      UserMessage(InboundMessage),
      CoalescedBatch(Vec<InboundMessage>),
      Timeout,
      Shutdown,
  }
  fn classify(msg: InboundMessage) -> AgentEvent { ... }
  // Main loop: match classify(msg) { ... }
  ```

### 4.2 `LcmEngineLoadState` enum
- **File:** `src/agent/agent_shared.rs:885-1109`
- **Current:** Complex nested if/else for DB→legacy→fresh engine init
- **Target:**
  ```rust
  enum LcmEngineLoadState { FromDb(Vec<SummaryNode>), FromLegacy(Vec<Turn>), Fresh }
  fn determine_load_state(...) -> LcmEngineLoadState { ... }
  ```

### 4.3 `ToolSourceMode` enum
- **File:** `src/agent/agent_shared.rs:773-872`
- **Current:** 3-way if/else-if/else for tool definition source + nested 4-way trio stripping
- **Target:**
  ```rust
  enum ToolSourceMode { Local, Scoped(TaskPhase), Relevant }
  fn select_tool_source_mode(is_local: bool, proprioception: &Config) -> ToolSourceMode { ... }
  ```

### 4.4 `ServerHealthState` enum
- **File:** `src/server.rs:847-893`
- **Current:** Nested if/else with side effects (alerts, restarts)
- **Target:**
  ```rust
  enum ServerHealthState { Starting, Healthy, Degraded { failures: u32 }, Failed }
  impl ServerHealthState { fn transition(&mut self, healthy: bool) -> Option<HealthAction> { ... } }
  ```

### 4.5 Extract `/local` command into steps
- **File:** `src/repl/cmd_lifecycle.rs:1091-1253`
- **Current:** 10+ nested decision points in single function
- **Target:** Extract `enter_local_mode()`, `exit_local_mode()`, `setup_inference_engine()`, `load_trio_models_if_enabled()`

### 4.6 Per-source `/model` handlers
- **File:** `src/repl/cmd_lifecycle.rs:424-620`
- **Current:** 5 match arms with 60-line MLX arm
- **Target:** Extract handler methods: `handle_lms_source()`, `handle_mlx_source()`, `handle_remote_source()`, etc.

### UAT
- [ ] `cargo test` passes
- [ ] `classify()` function has unit tests for all AgentEvent variants
- [ ] LCM engine loads correctly from DB, legacy, and fresh states
- [ ] `/local` and `/model` commands work identically to before refactor

---

## Deferred (Tier 3 — Quality Improvements)

These are lower-impact and can be done opportunistically:

| Target | File | What |
|--------|------|------|
| MIME lookup table | context.rs:1630-1644 | Replace 6-branch if/else with HashMap |
| Email field resolver | cli/mod.rs:1063-1129 | Generic `resolve_field()` for 4×3 branches |
| Provider/channel status | cli/mod.rs:1209-1389 | Loop over data struct instead of 8 ternaries |
| TTS init fallback | cli/mod.rs:843-876 | Chain-of-responsibility |
| Context tier lookup | server.rs:270-280 | Static sorted lookup table |
| `/think` parsing | cmd_mutation.rs:11-67 | `ThinkMode` enum + FromStr |
| `/sessions` parsing | cmd_mutation.rs:162-214 | `SessionsCmd` enum |
| `/replay` mode | cmd_mutation.rs:216-324 | `ReplayMode` enum + extract display fns |
| Proactive grounding | agent_shared.rs:1181-1215 | Extract `should_inject_grounding()` guard fn |
| Media pipeline | context.rs:1459-1493 | Functional filter_map chain |
| Bootstrap loading | context.rs:1370-1438 | `FileInclusionState` enum (state machine) |

---

## Verification Strategy

Each wave:
1. `cargo test` — full suite, no regressions
2. `cargo build --release` — clean compile
3. `cargo build --features ane` — ANE feature gate preserved
4. Manual smoke: `nanobot agent -m "hello"` in both local and cloud mode
5. Diff review: each refactored function produces identical output to original

## Risk

- **Wave 1** has the widest blast radius (touches 4 files, changes function signatures). Build incrementally: define enum/trait first, then migrate one call site at a time.
- **Waves 2-3** are low-risk (local to single files, no API changes).
- **Wave 4** is medium-risk (touches main loop). Use feature flag or cfg test guard if needed.

## Estimated Complexity

| Wave | Files | Targets | Risk |
|------|-------|---------|------|
| 1 — RuntimeMode | 4 | 5 | High (signature changes) |
| 2 — Dedup | 4 | 5 | Low (local extraction) |
| 3 — State enums | 5 | 5 | Low (local extraction) |
| 4 — Event dispatch | 5 | 6 | Medium (main loop) |
| Deferred | 6 | 11 | Low |
