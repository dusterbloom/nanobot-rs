# Nanobot Backlog

> Single source of truth for all actionable work.
> ROADMAP.md = vision. This file = what to do next.

---

## Ways of Working

> The codebase has coherence debt. Features are designed but not wired end-to-end.
> Trio mode exists in docs and config flags but never completed a real session.
> Compaction exists but crashes when it matters most. Context budgets exist but
> aren't enforced per role. The pattern: design → partial implement → patch holes → next feature.
>
> **New rule: every blocking item must have a verification step.**
> "Done" means: ran a real session with local models, checked metrics, checked logs,
> confirmed the feature actually fired. Not "code compiles and tests pass."
>
> When working with Claude Code on these items:
> 1. **One item at a time.** Don't let it "also fix" adjacent things.
> 2. **Start with a failing test or reproduction.** Show the broken state first.
> 3. **Verify with `metrics.jsonl` + `nanobot.log`** after every change.
> 4. **Read the existing code before writing.** Half these bugs are "feature exists but isn't called."
> 5. **B3-B10 are green. I7/LCM E2E verified. L1 concept router 80% accurate (2x orchestrator).** Blockers remaining: B5 (experiments), **B11 (heartbeat health layer)**, **B12 (config debt — proposal needed)**. Next priority: **B11 → B12 proposal → I9 (tiered routing) → I8 (SearXNG)**.

---

## Phase 0: Foundation (current)

### 🔴 Blocking — do first

- [ ] **B11: Heartbeat as foundational liveness service** ⚡ — **Priority.** The current heartbeat is a glorified cron — shell commands + optional HEARTBEAT.md check. It needs to become the central liveness and health service that all modes (text/voice), channels (CLI/Telegram/WhatsApp), and configurations (local trio/cloud) depend on.
  **Current state (verified 2026-02-21):** Runs one hardcoded command (`qmd update -c sessions`), LLM callback always `None` (never wired), `#![allow(dead_code)]` on the module. Zero awareness of providers, endpoints, channels, or self-health.
  **Health probes needed:**
  | Probe | What | Frequency | Action on failure |
  |-------|------|-----------|-------------------|
  | Provider health | Ping each configured provider (`/v1/models` or equivalent) | Every tick | Mark provider unavailable, trigger fallback chain |
  | LCM compactor | Ping `compactionEndpoint.url` + verify model loaded | Every tick | Set `lcm.available = false`, skip compaction gracefully, warn user |
  | Trio models | Verify router/specialist models loaded in LM Studio | Every tick (local only) | Degrade to inline mode, log which role is missing |
  | Search backend | Ping SearXNG / check Brave API key validity | Every 5 ticks | Disable `web_search` tool, surface in `/status` |
  | Channel liveness | Telegram/WhatsApp/Email connection state | Per-channel interval | Reconnect with backoff, surface in `/status` |
  | Self-health | Session size, memory pressure, disk space | Every tick | Trigger compaction, warn user, auto-rotate session |
  **Critical gap — LCM compaction has no pre-flight health check:** When qwen3-0.6b is unreachable, Level 1+2 silently fail → Level 3 deterministic fallback fires (works, but lossy). No warning, no status surfaced. If the LLM **hangs** (not down, just slow), the compaction spawn blocks indefinitely and `in_flight` never resets — **blocking ALL future compaction for the session.**
  **Design principles:**
  1. **Channel-agnostic** — same health loop whether CLI, Telegram, or voice mode
  2. **Mode-agnostic** — works for local trio, cloud, and hybrid configurations
  3. **Graceful degradation** — failures disable features, never crash. Compactor down → LCM pauses. Router gone → deterministic routing. Provider down → queue + retry.
  4. **Observable** — `/status` REPL command shows all probe states. Status injectable into context (connects to N6).
  5. **No LLM required** — health probes are HTTP pings and process checks, not agent tasks. Existing Layer 2 (HEARTBEAT.md → agent) stays as optional add-on.
  **Implementation sketch:**
  - `HealthRegistry` — register probes at startup based on config (if `lcm.enabled`, register LCM probe; if trio, register trio probes; etc.)
  - Each probe: `name`, `check() → Result<(), String>`, `interval`, `on_failure` callback
  - `HeartbeatService` runs the registry on each tick, stores probe states in `SystemState`
  - `/status` command reads probe states
  - Provider/router/compaction code checks probe state before attempting calls
  - Timeout guard on compaction spawn (default 30s) — kill task, reset `in_flight`, log error
  **Compounds with:** N6 (status injection), I8 (SearXNG health), I4 (multi-provider fallback), I7 (LCM compactor availability), I9 (tiered routing needs health-aware escalation).
  _Ref: `src/heartbeat/service.rs`, `src/agent/agent_loop.rs` (compaction spawn), `src/agent/system_state.rs`_

- [ ] **B12: Configuration debt — eliminate hardcoded magic values** ⚡ — **Requires design proposal before implementation.** Audit (2026-02-21) found **40+ hardcoded values** across `src/` that should be configurable: timeouts (8), token limits (12), model name sniffing (5 files), subagent constraints (5), circuit breaker params (2), session rotation (2), URLs (6+), maintenance commands (1). These accumulate because adding a `const` in a `.rs` file is faster than threading a value through config schema → serde → builder → runtime.
  **The problem is structural, not disciplinary.** Adding a config field today requires: 1) Add field + default fn to `schema.rs`. 2) Update `Default` impl. 3) Thread through `build_swappable_core()` or `AgentLoop::new()`. 4) Access via `ctx.core.xxx` or `self.xxx` at use site. 5) Write a serde test. That's 5 files touched for one knob — so devs take shortcuts and hardcode instead.
  **Audit — worst offenders:**
  | Category | Count | Where | Examples |
  |----------|-------|-------|---------|
  | Timeouts | 8 | lms.rs, tool_engine.rs, agent_loop.rs | LMS load 120s, delegation 30s, heartbeat 300s |
  | Token limits | 12 | subagent.rs, compaction.rs, pipeline.rs, context.rs | Subagent ctx 8192, summary 1024, pipeline 30K |
  | Model name sniffing | 5 files | tool_runner.rs, compaction.rs, agent_core.rs, thread_repair.rs | `nanbeige`, `ministral-3`, `qwen3-1.7b` string matches for capability detection |
  | Subagent constraints | 5 | subagent.rs | MAX_ITERATIONS=15, MAX_DEPTH=3, ctx 8192/2048, response 1024/256 |
  | Circuit breaker | 2 | circuit_breaker.rs | threshold=3, cooldown=300s |
  | Session mgmt | 2 | session/manager.rs | Rotation at 1MB, carry 10 messages |
  | Context bootstrap | 8 | context.rs | AGENTS.md 1500 tokens, memory 400, skills 1000, profiles 500 |
  **Requires a proposal that satisfies all of:**
  1. **KISS for users** — `config.json` stays simple. Normal users see ~20 top-level knobs (model, workspace, channels). Power users can override anything. Nobody needs to know about `subagent.local_fallback_context_tokens`.
  2. **DRY for devs** — Adding a new configurable value should touch ≤2 files (define + use). No manual threading through 5 layers. No forgetting to update `Default` impls.
  3. **SOLID** — Config values belong to the module that uses them, not a god struct. `subagent.rs` owns its defaults, `compaction.rs` owns its thresholds. The config system discovers them, not the other way around.
  4. **Zero regression risk** — Defaults must be identical to current hardcoded values. Existing `config.json` files must work without changes. New fields use `#[serde(default)]`.
  5. **Model capabilities, not model names** — Replace string sniffing (`if model.contains("nanbeige")`) with capability flags: `supports_tool_calling: bool`, `supports_thinking: bool`, `max_reliable_output: usize`. Populated from model metadata or config override.
  **Candidate approaches (evaluate before implementing):**
  - **A: Layered config with module-local defaults** — Each module declares its own `ModuleConfig` struct with `#[serde(default)]` fields. Root `Config` nests them. Modules access their own config directly. Pro: DRY, SOLID. Con: config.json grows nested sections.
  - **B: Feature flags + profiles** — Named profiles (`potato`, `sweet`, `power`, `beast`) that set all knobs at once. `config.json` says `profile: "sweet"`, everything derives. Individual overrides still possible. Pro: KISS for users. Con: profile matrix is hard to maintain.
  - **C: Convention-driven defaults with config overlay** — Keep `const` values in modules as defaults. Add a flat `overrides: {}` map in config that any module can query: `config.get_or("subagent.max_iterations", 15)`. Pro: zero structural change, 1-line to add a knob. Con: stringly-typed, no serde validation.
  - **D: Model capability registry** — Separate concern: replace 5-file model name sniffing with `ModelCapabilities` struct looked up by model ID. Config can override: `"modelCapabilities": {"nanbeige": {"toolCalling": true, "maxReliableOutput": 512}}`. Pro: solves the worst offender cleanly. Con: only addresses model sniffing, not timeouts/thresholds.
  - **E: Hybrid A+D** — Module-local configs (approach A) for numeric knobs + capability registry (approach D) for model sniffing. Two changes that together cover ~90% of the debt.
  **Do NOT implement until a proposal is reviewed.** The wrong abstraction here is worse than the current hardcoded values.
  **Compounds with:** B11 (heartbeat needs config-driven probes), I9 (tiered routing needs configurable thresholds), N1 (hardware auto-detection feeds profile selection).
  _Ref: Audit data in this backlog entry. Files: `src/agent/subagent.rs`, `src/agent/compaction.rs`, `src/agent/tool_runner.rs`, `src/agent/agent_core.rs`, `src/agent/thread_repair.rs`, `src/server.rs`, `src/heartbeat/service.rs`, `src/config/schema.rs`_

- [x] **B3: Update default local trio** — ✅ Trio configured: Main `gemma-3n-e4b-it` (`server.rs:18`), Router `nvidia_orchestrator-8b`, Specialist `ministral-3-8b-instruct-2512` (both in `config.json` trio section + B10 auto-detect as fallback). `TrioConfig::default()` has empty strings but runtime always populated via explicit config or auto-detect.
- [x] **B3.1: Smarter deterministic fallback** — ✅ `router_fallback.rs` now has 9 deterministic patterns + default ask_user (was 2). Includes: research+URL→spawn researcher, plain URL/HN→web_fetch, latest news→spawn, read/show+path→read_file, write/create+path→write_file, edit/modify+path→edit_file, list/ls→list_dir, run/execute/cargo→exec, search→web_search. All patterns guarded by `has_tool()` for graceful fallthrough. 19 tests. _Ref: `src/agent/router_fallback.rs`_
- [x] **B4: Multi-model config schema** — ✅ Obsolete as scoped. `TrioConfig` already provides per-role model/port/ctx_tokens/temperature/endpoint for router and specialist. LM Studio JIT-loads models on demand — no need for nanobot to spawn separate llama-server instances. The `local.rlm` slot became the specialist role; `local.memory` uses `memory.model` config.
- [ ] **B5: RLM model evaluation** — Systematic experiments to find best RLM model per VRAM tier. Critical for "3 impossible things". See experiment plan below. _Routing benchmarks started in `experiments/lcm-routing/` (orchestrator_bench.py, test_bench.py)._
- [x] **B8: Trio mode activation & role-scoped context** ⚡ — ✅ All 5 steps complete. Metrics + circuit breaker (commit `0f80ad9`). Auto-activation + auto-detect as B10 (commit `3774742`). E2E verified: local session → `delegation_mode=Trio` in log → Main emits natural language → Router preflight intercepts → Specialist executes tool. _Ref: `src/agent/router.rs`_

### 🟡 Important — do soon

- [ ] **I0: Trio pipeline actions** — Router can only emit ONE action per turn. Multi-step tasks (research + synthesize) fail because the router picks one tool and stops. Need pipeline-as-first-class router output + shared scratchpad between trio roles. **Superseded by I9** for the routing layer — I0 remains relevant for the execution/scratchpad side. _Ref: `thoughts/shared/plans/2026-02-20-trio-pipeline-architecture.md`_
- [ ] **I9: Tiered routing with orchestrator escalation** ⚡ — **Priority.** The L1 concept router (all-MiniLM-L6-v2, centroid classification) proved 80% accurate at 5ms/0 VRAM vs orchestrator-8b's 43% at 637ms/6GB. But rigid template matching (L2) caps at ~7 predefined multi-step patterns. **Real workflows need 10-100+ steps with dynamic re-planning, conditional branching, and error recovery — templates can't express this.**
  **Architecture: Three-tier routing with orchestrator escalation:**
  | Tier | Engine | Latency | When | Traffic % |
  |------|--------|---------|------|-----------|
  | T1: Concept Router | Embedding centroid (CPU) | ~5ms | Unambiguous single-action queries | ~70% |
  | T2: Template Expander | Embedding → template match | ~10ms | Known multi-step patterns (L2 templates) | ~15% |
  | T3: Orchestrator | Reasoning LLM (nemotron-orchestrator-8b or nanbeige) | ~600ms | Complex/novel/failing workflows, low-confidence T1 | ~15% |
  **Escalation triggers (T1→T3):** (a) Cosine similarity margin <0.4 (low confidence). (b) No template match for detected multi-step intent. (c) Step failure mid-execution (re-plan). (d) User query references prior context (pragmatic ambiguity).
  **T3 orchestrator responsibilities:** Dynamic step decomposition (not limited to templates). Mid-workflow re-planning when steps fail or return unexpected results. Conditional branching (if build fails → fix errors → retry). State tracking across 10-100+ steps via scratchpad. Budget/token cost monitoring.
  **Key insight from L1 experiments:** The 6 concept router failures were all pragmatic (hedging, vagueness, context-dependent) — exactly the cases where an LLM reasoning model adds value. The concept router handles the ~70% easy cases at zero cost; the orchestrator handles the ~15% hard cases where reasoning matters. This is cheaper than running the orchestrator on 100% of traffic.
  **Implementation path:** 1) Wire concept router into `router_preflight()` as fast path. 2) Add confidence threshold — below 0.4, escalate to LLM orchestrator. 3) Add step executor that consumes `Vec<RouterDecision>` from either T2 templates or T3 orchestrator. 4) Add shared scratchpad for multi-step state. 5) Add re-planning hook: when a step fails, send context + failure to T3 for new plan.
  **Compounds with:** I0 (pipeline execution), L1/L2 experiments (`experiments/lcm-routing/`), B5 (model evaluation).
  _Ref: `experiments/lcm-routing/results/L1_analysis.md`, `experiments/lcm-routing/multi_step_templates.json`_
- [ ] **I1: Local role/protocol crashes** — Fix `system` role crash, alternation crash, orphan tool messages. Thread repair pipeline exists but needs hardening. _Ref: `docs/plans/local-trio-strategy-2026-02-18.md`, `docs/plans/local-model-reliability-tdd.md`_
- [x] **I2: Non-blocking compaction** — ✅ Absorbed into I7 (matryoshka compaction). Per-cluster parallel summarization replaces the three-tier approach.
- [ ] **I3: Context Gate** — Replace dumb char-limit truncation with `ContentGate`: pass raw / structural briefing / drill-down. Zero agent-facing API changes. _Ref: `docs/plans/context-gate.md`, `docs/plans/context-protocol.md`_
- [ ] **I4: Multi-provider refactor** — Break up `SwappableCore` god struct, extensible provider registry, fallback chains. _Ref: `docs/plans/multi-provider-refactor.md`, `docs/plans/nanobot_architecture_review.md`_
- [ ] **I5: Dynamic model router** — Prompt complexity scoring (light/standard/heavy), auto-downgrade simple messages to cheaper models. _Ref: `docs/plans/dynamic-model-router.md`_
- [x] **I6: Context Hygiene Hooks** — ✅ Implemented as `anti_drift.rs` (851 lines, 25 tests). PreCompletion: pollution scoring, turn eviction, repetitive-attempt collapse, format anchor re-injection. PostCompletion: thinking tag stripping, babble collapse. _Ref: commit `56dedce`, `src/agent/anti_drift.rs`_
- [ ] **I8: SearXNG search backend** — Replace Brave Search (API key required, rate-limited) with SearXNG (free, local, unlimited). 3 touchpoints: 1) `schema.rs`: add `provider: String` ("brave"|"searxng", default "searxng") + `searxng_url: String` (default "http://localhost:8888") to `WebSearchConfig`. 2) `web.rs`: add SearXNG path in `execute()` — `GET {url}/search?q={query}&format=json`, parse `results[].title/url/content`. No API key needed. 3) `registry.rs`+`tool_wiring.rs`: extend `ToolConfig` to carry `search_provider`+`searxng_url`. Fallback: if SearXNG unreachable and Brave key set, use Brave. If neither, helpful error: "Run `docker run -d -p 8888:8080 searxng/searxng` or set a Brave API key." Onboard integration: `cmd_onboard()` prints Docker one-liner. Optional `nanobot onboard --search` auto-pulls+starts+configures SearXNG. _SearXNG container tested working 2026-02-20: `docker run -d -p 8888:8080 --name searxng searxng/searxng` + enable JSON format in settings.yml._
- [x] **I7: Lossless Context Management (LCM)** _(supersedes matryoshka design)_ — ✅ DAG-based lossless compaction per Ehrlich & Blackman (2026). Immutable store (session JSONL) + Summary DAG with pointers to originals + active context assembly. **Implemented:** `src/agent/lcm.rs` (~1100 lines, 17 tests): `SummaryDag`, `LcmEngine` (ingest/compact/expand), three-level escalation (preserve_details → bullet_points → deterministic truncate), dual-threshold control loop (τ_soft 50% / τ_hard 85%). `LcmSchemaConfig` in config schema. Wired into `agent_loop.rs`. `lcm_expand` tool registered when LCM enabled.
  **E2E verified (2026-02-21):** Real E2E test against nemotron-nano-12b on LM Studio: 12 messages through `process_direct` → compaction triggered at τ_soft → Level 2 summary created → DAG node with lossless source IDs → `expand()` retrieves originals. 6 invariants checked (store lossless, active shrinks, DAG populated, source IDs resolve, Summary entries present, expand works). Benchmark across 4 models: qwen3-0.6b best compressor (83.2% compression, 3.4s), nemotron-nano-12b fastest (81.4%, 2.8s). Bigger models (gemma-3n-e4b 54.6%, qwen3-1.7b 72.8%) produce more verbose summaries — worse for compaction.
  **Remaining:** Performance profiling under sustained load. Verify `lcm_expand` actually invoked by LLM during conversation. Persist DAG across session rotations.
  **Compounds with:** I6 (anti-drift cleans within summaries). B9 (pre-flight truncation as safety net). I3 (ContentGate decides raw vs summary).
  _Ref: `src/agent/lcm.rs`, `src/config/schema.rs:1219`_
- [ ] **I10: `/clear` and `/new` REPL commands** — Manual context reset for local models with small context windows (4K-8K). Essential for trio mode where the Main model accumulates full conversation history while Router and Specialist are already stateless (ephemeral per-turn message arrays).
  **`/clear` — reset working context, keep session:**
  1. If LCM enabled: compact all `ctx.messages` into a single summary node. Model starts "fresh" but can `lcm_expand` to retrieve originals.
  2. If LCM disabled: truncate messages, carry forward last 2 as context seed.
  3. Reset working memory section of CONTEXT.md.
  4. Emit `--- context cleared (N messages compacted) ---` in REPL.
  5. Session JSONL continues — no data lost from the audit trail.
  **`/new [name]` — fresh session entirely:**
  1. Start a new session file (new JSONL, clean slate).
  2. Optional `name` parameter, otherwise auto-generate.
  3. Existing session stays on disk, accessible via `/sessions`.
  **Scope:** Only Main model context needs reset — Router and Specialist are already ephemeral (verified in `router.rs`: `tool_messages`/`router_messages` built fresh each turn, `specialist_messages` built fresh each dispatch). No per-role action needed.
  **Compounds with:** I7/LCM (compaction on clear), B11 (health status after clear), N6 (status injection reset).
  _Ref: `src/agent/router.rs` (trio context isolation), `src/agent/agent_loop.rs` (LCM wiring), `src/session/manager.rs` (session lifecycle)_

### 🟢 Nice to have — Phase 0

- [ ] **N1: Auto hardware detection** — Detect VRAM/RAM/CPU, auto-assign tier (Potato/Sweet/Power/Beast), select quant level. `nanobot doctor` command.
- [ ] **N2: `nanobot setup`** — Interactive first-run: detect hardware, download models, generate optimal config.
- [ ] **N3: Streaming rewrite** — Incremental markdown renderer, line-by-line syntax highlighting, no full-response rerender. _Ref: `docs/plans/streaming-rewrite.md`_
- [ ] **N4: Full-duplex REPL** — ESC+ESC instant cancel, backtick injection prompt, priority message channel. _Ref: `docs/plans/full-duplex-repl.md`_
- [ ] **N5: Thinking toggle** — `/think` command + Ctrl+T toggle for extended thinking mode. _Ref: `docs/plans/thinking-toggle.md`_
- [ ] **N6: Status injection** — Auto-inject background worker status into context each turn. _(Spacebot idea)_
- [ ] **N7: Message coalescing** — Batch rapid messages in channels into single LLM turn. _(Spacebot idea)_
- [ ] **N8: Narration stress test** — Validate narration compliance across local models. _Ref: `docs/plans/narration-stress-test.md`_

---

## Phase 1: Million-Token Context (next)

- [ ] **P1.1: File-backed volumes** — `MappedVolume` struct with mmap + line-offset index
- [ ] **P1.2: Chunk index** — 4K-char chunks, simhash signatures, `ctx_search`
- [ ] **P1.3: Semantic index** — Optional e5-small embeddings, vector similarity
- [ ] **P1.4: Proof** — Needle-in-haystack at 1M tokens, 95%+ recall, <60s

---

## Phase 2: Million-Step Processes (later)

- [ ] **P2.0: Calibration run** — Measure per-step `p` on 1K-10K steps using winning RLM model from E3
- [ ] **P2.1: MAKER voting** — `first_to_ahead_by_k`, red-flagging, output token cap
- [ ] **P2.2: MAD decomposition** — Atomic step definitions per domain
- [ ] **P2.3: Process tree** — Persistent execution tree, checkpoint/resume
- [ ] **P2.4: RLM completion** — `ctx_summarize`, recursive depth, smart short-circuit. _Ref: `docs/plans/rlm-completion-proposal.md`, `docs/plans/adaptive_rlm_design.md`_
- [ ] **P2.5: Swarm architecture** — Workers spawn Workers, budget propagation. _Ref: `docs/plans/swarm-architecture.md`_
- [ ] **P2.6: Event log pipeline** — Append-only JSONL, pipeline runner. _Ref: `docs/plans/event-log-pipeline.md`_
- [ ] **P2.7: Proof** — Towers of Hanoi 20 disks, 1M+ steps, zero errors, local only

---

## Phase 3: Self-Evolving Agent (future)

- [ ] **P3.1: Trace logger** — Structured JSONL per process
- [ ] **P3.2: Skill crystallization** — Auto-create skills from repeated successes
- [ ] **P3.3: Budget calibration** — Per-task-type stats in SQLite
- [ ] **P3.4: LoRA distillation** — Export traces → Zero pipeline → hot-swap LoRA

---

## Experiment Plan: Local Trio Evaluation

> Reduce assumptions one at a time. No coding until we know what works.

### New Default Trio (RTX 3090, 24GB VRAM)

| Role | Model | Size | Why |
|------|-------|------|-----|
| **Main** | gemma-3n-e4b-it | ~4B effective | Fast, good chat, small footprint |
| **Orchestrator** | nvidia_orchestrator-8b | 8B | 10/10 routing accuracy (proven in experiments/) |
| **Specialist** | ministral-3-8b-instruct-2512 | 8B | Strong tool-calling, instruction following |

### What We Know
- Nemotron Orchestrator: 10/10 routing (vs NanBeige 6/10). Purpose-built. **Proven.**
- NanBeige 3B: Good with `<think>\n</think>\n\n` prefill, but weak as router.
- Main + Orchestrator work well together in practice.
- Sequential self-routing would add latency vs parallel separation. Keep roles split.
- **Router single-action bottleneck**: `request_strict_router_decision()` returns ONE `RouterDecision`. Multi-step tasks (fetch 2 URLs + synthesize) cannot be expressed. The router picks one tool and the pipeline stalls. **See I9 for solution.**
- ~~**Deterministic fallback too narrow**~~: **Fixed** in B3.1. `router_fallback.rs` now has 9 patterns (research+URL, plain URL, HN, latest news, read, write, edit, list, search, exec) + default ask_user. All guarded by `has_tool()`.
- **L1 Concept router validated (2026-02-21):** all-MiniLM-L6-v2 centroid classification: 24/30 (80%) vs orchestrator-8b 13/30 (43%). 5ms vs 637ms. 0 VRAM vs 6GB. 100% on non-ambiguous queries. 5/5 multilingual. Failures are all pragmatic/vague — exactly the cases where LLM reasoning adds value. Data: `experiments/lcm-routing/`.
- **L2 Multi-step templates built:** 7 templates (research_and_summarize, read_and_analyze, fetch_and_compare, search_and_update, check_and_report, plan_and_implement, verify_and_fix). Max 4 steps. **Limitation: rigid patterns can't scale to 10-100+ step workflows or handle failures/branching.** Orchestrator model needed for dynamic planning. See I9.
- **Specialist has no tools**: `dispatch_specialist()` sends a single-shot chat — no tool access. Can synthesize given context but cannot fetch/execute.
- **Trio never tested end-to-end**: As of 2026-02-19 handoff, the full trio flow (Main→Router→Specialist) has never completed a real task through LM Studio.
- **2026-02-21 diagnostic: Trio mode didn't activate.** NanBeige ran as Inline main with full tool schemas. 21 metrics entries show `tool_calls_requested: 1, tool_calls_executed: 0` — model generated tool calls (proving it had tool schemas) that were blocked as duplicates. Compaction crashed twice (`n_keep 12620 >= n_ctx 8192`). **Fixed:** B8 (metrics + circuit breaker) and B9 (tool guard replay + compaction overflow) shipped. Death spiral no longer occurs. Remaining: wire trio activation so NanBeige runs in Trio mode, not Inline.
- **System prompt is ~15-20K tokens** even before conversation starts. Opus first call: `prompt_tokens: 21705`. A 3B model with 8K context has zero room. Even with 32K context, 15K of prompt leaves only 17K for conversation — and most of that prompt is AGENTS.md/SOUL.md/TOOLS.md that small models can't follow anyway.
- ~~**Metrics broken for local models**~~ — **Fixed** in B8 (commit `0f80ad9`). Token counts now captured from llama.cpp `usage` field.

### Experiments Needed (one assumption at a time)

#### E1: Role Evaluation Matrix
Test each candidate model in each role independently.

| Model | As Main | As Orchestrator | As Specialist | As RLM |
|-------|---------|-----------------|---------------|--------|
| gemma-3n-e4b-it | ? | ? | ? | ? |
| nvidia_orchestrator-8b | ✅ 10/10 routing | ✅ proven | ? | ? |
| ministral-3-8b-instruct-2512 | ? | ? | ? | ? |
| nanbeige4.1-3b | ? | 6/10 | ? | ? |

Test bench per role:
- **Main**: 10 conversation tasks (chat quality, coherence, narration compliance)
- **Orchestrator**: 10 routing cases (existing test suite from experiments/)
- **Specialist**: 10 tool-calling tasks (file ops, exec, multi-step)
- **RLM**: 5 delegation loops (multi-step file edit, research, build cycle)

#### E2: VRAM Profile Testing
Critical for "3 impossible things" — must work across hardware tiers.

| Tier | VRAM | Trio Budget | Candidate Combos |
|------|------|-------------|------------------|
| Potato | 4-6 GB | ~4B total | 1 model does all? |
| Sweet | 8-12 GB | ~12B total | 2 small models |
| Power | 16-24 GB | ~24B total | Full trio (current target) |
| Beast | 48+ GB | Unlimited | Bigger specialists |

#### E3: RLM Model Shootout
The key unknown. Test candidates on delegation loop benchmarks:
- Multi-step file edit (read → plan → edit → verify)
- Web research synthesis (search → fetch → summarize)
- Build cycle (edit → compile → fix errors → retry)

Metrics: completion rate, token cost, latency, error recovery.

#### E4: Integration Test
Once E1-E3 identify winners, run full nanobot session with the new trio.
Compare against current setup on real tasks.

### Experiment Order
1. **E1** first — know what each model can do in each role
2. **E3** next — find the RLM (biggest unknown)
3. **E2** then — scale findings across VRAM tiers
4. **E4** last — validate the winning combo end-to-end

---

## Spacebot Ideas (parking lot)

Captured from [spacebot](https://github.com/spacedriveapp/spacebot). Ideas only, no code.

| Idea | Status | Mapped to |
|------|--------|-----------|
| Non-blocking compaction | ✅ Absorbed into I7 (matryoshka) | Phase 0 |
| Status injection | Backlog N6 | Phase 0 |
| Message coalescing | Backlog N7 | Phase 0 |
| Branch concept (context-fork) | Not started | Phase 2 (related to swarm) |
| Prompt complexity routing | Backlog I5 | Phase 0 |
| Memory bulletin (Cortex) | Not started | Phase 3 (related to memory) |

---

## Done ✅

- ~~I7: Lossless Context Management (LCM)~~ — DAG-based lossless compaction. `LcmEngine` with three-level escalation (LLM preserve_details → bullet_points → deterministic truncate). Dual-threshold control loop (τ_soft/τ_hard). `lcm_expand` tool for lossless retrieval. E2E verified against 4 local models: qwen3-0.6b best compressor (83.2%, 3.4s), nemotron-nano-12b fastest (81.4%, 2.8s). 17 tests (4 mock E2E + 1 real E2E + 1 benchmark + 4 config + 9 unit). 1407 total green. (2026-02-21, `src/agent/lcm.rs`, commits `0697bd4`, `9893d91`, `bde583f`)
- ~~B3: Update default local trio~~ — Trio configured: Main `gemma-3n-e4b-it`, Router `nvidia_orchestrator-8b`, Specialist `ministral-3-8b-instruct-2512`. Explicit config + B10 auto-detect. (2026-02-21)
- ~~B4: Multi-model config schema~~ — Closed as obsolete. TrioConfig provides per-role model/port/endpoint. LM Studio JIT-loads models; no separate server spawning needed. (2026-02-21)
- ~~B8: Trio mode activation & role-scoped context~~ — All 5 steps complete. Metrics + circuit breaker (commit `0f80ad9`). Auto-activation + auto-detect as B10 (commit `3774742`). E2E verified: delegation_mode=Trio in log, Main emits natural language, Router preflight intercepts, Specialist executes tool. (2026-02-21, `src/agent/router.rs`)
- ~~Session indexer + REPL /sessions command~~ — Bridge between raw JSONL sessions (230 files, 116MB) and searchable SESSION_*.md memory files. `session_indexer.rs`: pure `extract_session_content()` + `index_sessions()` orchestrator (extracts user+assistant messages, skips tool results, caps at 50 messages, truncates to 500 chars each). REPL: `/sessions` command with list/export/purge/archive/index subcommands (`/ss` alias). CLI: `nanobot sessions index`. Fixed `process::exit(1)` in `sessions_cmd.rs` for REPL safety. Updated `recall` tool description. E2E verified: 149 sessions indexed (6→155 SESSION_*.md), idempotent re-run, grep finds content. 17 new tests, 1395 total green. (2026-02-21, `src/agent/session_indexer.rs`)
- ~~B10: Auto-detect trio models from LM Studio~~ — `pick_trio_models()` scans available LMS models at startup for "orchestrator"/"router" (router) and "function-calling"/"instruct"/"ministral" (specialist) patterns. Only fills empty config slots — explicit config always wins. Fuzzy main-model exclusion handles org prefixes and unresolved GGUF hints. Wired into REPL startup before auto-activation. 13 tests including e2e flow and real LMS model list. (2026-02-21, commit `3774742`)
- ~~B9: Compaction safety guard + tool guard death spiral~~ — Tool guard replays cached results instead of injecting error messages small models can't parse. Compaction respects summarizer model's actual context window via `compaction_model_context_size` config + pre-flight truncation (0.7 safety margin). Circuit breaker threshold 3→2. E2E verified against NanBeige on LM Studio. (2026-02-21, commit `0f7f365`)
- ~~B8: Metrics accuracy + tool loop circuit breaker~~ — Fixed local model metrics capture (`prompt_tokens`, `completion_tokens`, `elapsed_ms`). Added circuit breaker for consecutive all-blocked tool call rounds. (2026-02-21, commit `0f80ad9`)
- ~~B7: Provider retry with `backon`~~ — Replaced 3 hand-rolled retry loops with `backon` crate. Shared `is_retryable_provider_error()` predicate. Added retry to streaming path. (2026-02-21, commit `640bdc9`)
- ~~B6: SLM provider observability~~ — 8 silent failure paths now logged. `#[instrument]` spans on `chat()`/`chat_stream()`. Promoted `llm_call_failed` to `warn!`. (2026-02-21, commit `0b6bc5f`)
- ~~Fix: Audit log hash chain race condition~~ — `record()` had a TOCTOU bug: seq allocation + prev_hash read were not serialized under the file lock. Two concurrent executors (tool_runner + inline) both read seq 940 and wrote seq 941 with the same prev_hash, forking the chain at entry 942. Fix: acquire file lock first, re-read authoritative seq + prev_hash from file under lock, then compute hash and write. 12/12 audit tests pass. (2026-02-21, commit `835cf6d`, `src/agent/audit.rs`)
- ~~B1: 132 compiler warnings~~ → 0 warnings (2026-02-20)
- ~~B2: 2 test failures~~ → 1395 pass, 0 fail (2026-02-21)
- ~~Fix: Subprocess stdin steal~~ — `.stdin(Stdio::null())` on all 4 spawn sites in shell.rs + worker_tools.rs (2026-02-20)
- ~~Fix: Esc-mashing freezes REPL~~ — drain_stdin() after cancel (2026-02-20, commit 57ec883)
- ~~Fix stale comment in `ensure_compaction_model`~~ (2026-02-17)
- ~~Raise tool result truncation threshold~~ (2026-02-17)
- ~~Document multi-session CONTEXT.md race~~ (2026-02-17)
- ~~Input box disappears during streaming~~ (2026-02-17)
- ~~Agent interruption too slow~~ (2026-02-17)
- ~~Subagent improvements (wait, output files, budget, compaction)~~ (2026-02-18)
- ~~Tool runner infinite loop fix~~ (2026-02-18)
