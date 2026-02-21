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
> 5. **B6-B10 are green.** Next priority: B8 step 5 (end-to-end trio verification), then B3 (TrioConfig defaults), then B4 (multi-model config).

---

## Phase 0: Foundation (current)

### 🔴 Blocking — do first

- [ ] **B3: Update default local trio** — New defaults: 1) Main: `gemma-3n-e4b-it`, 2) Orchestrator: `nvidia_orchestrator-8b`, 3) Specialist: `ministral-3-8b-instruct-2512`. ~~Update `DEFAULT_LOCAL_MODEL`~~ ✅ (already `gemma-3n-e4b-it-Q4_K_M.gguf` in `server.rs:18`). Remaining: update `agents.json` defaults, config schema defaults, and `TrioConfig` default model names. _Ref: experiments/tool-calling/_
- [x] **B3.1: Smarter deterministic fallback** — ✅ `router_fallback.rs` now has 9 deterministic patterns + default ask_user (was 2). Includes: research+URL→spawn researcher, plain URL/HN→web_fetch, latest news→spawn, read/show+path→read_file, write/create+path→write_file, edit/modify+path→edit_file, list/ls→list_dir, run/execute/cargo→exec, search→web_search. All patterns guarded by `has_tool()` for graceful fallthrough. 19 tests. _Ref: `src/agent/router_fallback.rs`_
- [ ] **B4: Multi-model config schema** — Add `local.main`, `local.rlm`, `local.memory` to config. Each slot: `{ model, path, gpu, context_size, temperature }`. Server manager spawns up to 3 llama-server instances. _Ref: `docs/plans/local-model-matrix.md`_
- [ ] **B5: RLM model evaluation** — Systematic experiments to find best RLM model per VRAM tier. Critical for "3 impossible things". See experiment plan below. _Routing benchmarks started in `experiments/lcm-routing/` (orchestrator_bench.py, test_bench.py)._
- [ ] **B8: Trio mode activation & role-scoped context** ⚡ — Metrics accuracy and tool loop circuit breaker shipped (commit `0f80ad9`). Auto-activation wiring + auto-detect shipped as B10 (commit `3774742`). Remaining: end-to-end verification. Steps:
  1. ~~**Trace config loading**~~: ✅ Done in B10. `delegation_config_at_core_build` log at startup, `trio_auto_activated` when trio fires.
  2. ~~**Verify `router_preflight()` fires**~~: ✅ `info!("router_preflight_firing")` exists in `router.rs:474` (uncommitted). Also logs `router_preflight_skipped` with reason at `router.rs:468`.
  3. ~~**Verify role-scoped context packs differ by role**~~: ✅ `role_scoped_context_packs` field exists on `ToolDelegationConfig` and is used in `router.rs:368,499,661` and `tool_engine.rs:120` to build per-role context.
  4. ~~**Slim Main's system prompt for local**~~: ✅ `ContextBuilder::new_lite()` (`context.rs:113-128`) and `set_lite_mode()` (`context.rs:131-139`) implemented. Lite mode: bootstrap 2%, memory 1%, skills 2%, profiles 1%, cap 30% of context.
  5. **Verification**: Start local session → `delegation_mode=Trio` in log → Main emits natural language (not tool call) → Router preflight intercepts → Specialist executes tool. **This is the remaining work.**
  _Ref: 2026-02-21 diagnostic session, `src/agent/router.rs`_

### 🟡 Important — do soon

- [ ] **I0: Trio pipeline actions** — Router can only emit ONE action per turn. Multi-step tasks (research + synthesize) fail because the router picks one tool and stops. Need pipeline-as-first-class router output + shared scratchpad between trio roles. _Ref: `thoughts/shared/plans/2026-02-20-trio-pipeline-architecture.md`_
- [ ] **I1: Local role/protocol crashes** — Fix `system` role crash, alternation crash, orphan tool messages. Thread repair pipeline exists but needs hardening. _Ref: `docs/plans/local-trio-strategy-2026-02-18.md`, `docs/plans/local-model-reliability-tdd.md`_
- [x] **I2: Non-blocking compaction** — ✅ Absorbed into I7 (matryoshka compaction). Per-cluster parallel summarization replaces the three-tier approach.
- [ ] **I3: Context Gate** — Replace dumb char-limit truncation with `ContentGate`: pass raw / structural briefing / drill-down. Zero agent-facing API changes. _Ref: `docs/plans/context-gate.md`, `docs/plans/context-protocol.md`_
- [ ] **I4: Multi-provider refactor** — Break up `SwappableCore` god struct, extensible provider registry, fallback chains. _Ref: `docs/plans/multi-provider-refactor.md`, `docs/plans/nanobot_architecture_review.md`_
- [ ] **I5: Dynamic model router** — Prompt complexity scoring (light/standard/heavy), auto-downgrade simple messages to cheaper models. _Ref: `docs/plans/dynamic-model-router.md`_
- [x] **I6: Context Hygiene Hooks** — ✅ Implemented as `anti_drift.rs` (851 lines, 25 tests). PreCompletion: pollution scoring, turn eviction, repetitive-attempt collapse, format anchor re-injection. PostCompletion: thinking tag stripping, babble collapse. _Ref: commit `56dedce`, `src/agent/anti_drift.rs`_
- [ ] **I8: SearXNG search backend** — Replace Brave Search (API key required, rate-limited) with SearXNG (free, local, unlimited). 3 touchpoints: 1) `schema.rs`: add `provider: String` ("brave"|"searxng", default "searxng") + `searxng_url: String` (default "http://localhost:8888") to `WebSearchConfig`. 2) `web.rs`: add SearXNG path in `execute()` — `GET {url}/search?q={query}&format=json`, parse `results[].title/url/content`. No API key needed. 3) `registry.rs`+`tool_wiring.rs`: extend `ToolConfig` to carry `search_provider`+`searxng_url`. Fallback: if SearXNG unreachable and Brave key set, use Brave. If neither, helpful error: "Run `docker run -d -p 8888:8080 searxng/searxng` or set a Brave API key." Onboard integration: `cmd_onboard()` prints Docker one-liner. Optional `nanobot onboard --search` auto-pulls+starts+configures SearXNG. _SearXNG container tested working 2026-02-20: `docker run -d -p 8888:8080 --name searxng searxng/searxng` + enable JSON format in settings.yml._
- [ ] **I7: Matryoshka Compaction** _(absorbs I2)_ — Recursive, parallel context compaction that works with any model size. Replaces monolithic "summarize everything" with nested layers, like matryoshka dolls. **Architecture:**
  ```
  Layer 0: Raw conversation (32K+ tokens)
           ↓ kodama cluster into semantic blocks (TF-IDF, hand-rolled, ~40 lines)
  Layer 1: N topic clusters (~2-3K each)
           ↓ summarize each in parallel via tokio::spawn (even 2K model works)
  Layer 2: N summaries (~200-400 tokens each)
           ↓ if still too big, summarize the summaries
  Layer 3: 1 meta-summary (~500 tokens)
  ```
  **Key properties:** Each step fits any model (never exceeds `model_ctx * 0.7`). All cluster summaries run as concurrent tasks. Last 2-3 turns stay verbatim. Incremental — only active cluster grows, dormant clusters stay compressed. Infinite conversation regardless of model context size.
  **Role-scoped assembly:** Main (3B, 8K) gets meta-summary + last 2 turns + active cluster verbatim. Router (8B) gets task-relevant cluster summary + current intent. Specialist (8B) gets tool-relevant cluster + tool schemas. Each role sees exactly the context it needs.
  **Deps:** `kodama = "0.3"` (MIT/Unlicense, zero transitive deps). TF-IDF vectors hand-rolled (~40 lines). Brute-force cosine similarity over clusters (sub-ms at <1000 clusters, no vector DB needed). New module `src/agent/semantic_context.rs`, ~400 lines.
  **Compounds with:** I6 (hooks clean within a cluster, fragmentation cleans across clusters). B9 (pre-flight truncation becomes the per-chunk safety net). I3 (ContentGate decides raw vs summary per cluster).
  **Agno L3 parallel:** This is nanobot's equivalent of Agno's hybrid-search memory layer — surface relevant context, compress the rest — but designed for 4-8K local models instead of 128K cloud models.
  _Ref: 2026-02-21 matryoshka design session, `src/agent/compaction.rs`_

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
- **Router single-action bottleneck**: `request_strict_router_decision()` returns ONE `RouterDecision`. Multi-step tasks (fetch 2 URLs + synthesize) cannot be expressed. The router picks one tool and the pipeline stalls.
- ~~**Deterministic fallback too narrow**~~: **Fixed** in B3.1. `router_fallback.rs` now has 9 patterns (research+URL, plain URL, HN, latest news, read, write, edit, list, search, exec) + default ask_user. All guarded by `has_tool()`.
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
