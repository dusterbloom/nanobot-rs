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
> 5. **No new features until B8 and B9 are green.** Everything else is noise if trio mode doesn't activate.

---

## Phase 0: Foundation (current)

### 🔴 Blocking — do first

- [ ] **B6: SLM Observability** — Flying blind with trio. 8 silent failure paths identified (2026-02-20 audit). Zero new deps — uses existing `tracing` crate. Changes:
  1. `warn!` on malformed tool-call JSON (openai_compat.rs:1028, 1154, 1300) — currently silently wrapped in `{"raw":…}`
  2. Log HTTP error response body (openai_compat.rs:642, anthropic.rs:442) — currently body discarded, only status code logged
  3. `warn!` on SSE stream ending without `[DONE]` (openai_compat.rs:1270) — SLM crashes mid-response invisible
  4. `warn!` on empty LLM response fallback (agent_loop.rs:1214) — silent hardcoded string injection
  5. ~~`warn!` on Anthropic SSE parse errors (anthropic.rs:700)~~ — Done: upgraded to `warn!` in both anthropic.rs and openai_compat.rs
  6. Add `#[instrument]` spans to `chat()`/`chat_stream()` in both providers
  7. Add `.with_span_events(FmtSpan::CLOSE)` to JSON subscriber in main.rs — free latency tracking
  8. Remove dead `error_detail()` code (base.rs:47-62) — `finish_reason == "error"` never fires
  9. Fix duplicate `llm_stream_started` event in anthropic.rs:610-624
  10. Promote `llm_call_failed` from `info!` to `warn!` (openai_compat.rs:696, anthropic.rs:485)
  _Ref: session audit 2026-02-20, logs at ~/.nanobot/nanobot.log_
- [ ] **B7: Provider retry with `backon`** — Three hand-rolled retry loops (JIT gate, specialist warm-up, audit persistence) with no jitter. `ProviderError::RateLimited { retry_after_ms }` created but never read. `chat_stream` has zero retry. Add `backon = "1.6"` (MIT, near-zero deps, used by `uv`). Shared `is_retryable_provider_error()` predicate. Replace all 3 loops + add retry to streaming path. _Ref: session audit 2026-02-20_
- [ ] **B3: Update default local trio** — New defaults: 1) Main: `gemma-3n-e4b-it`, 2) Orchestrator: `nvidia_orchestrator-8b`, 3) Specialist: `ministral-3-8b-instruct-2512`. Update `agents.json`, `DEFAULT_LOCAL_MODEL`, config schema defaults, and `TrioConfig`. _Ref: experiments/tool-calling/_
- [ ] **B3.1: Smarter deterministic fallback** — `router_fallback.rs` only has 2 patterns (URL→web_fetch, "latest news"→spawn). Fails on "research report + URLs" which should spawn a researcher. Add pattern: research/report/summarize + URLs → `Subagent(researcher)`. _Ref: `src/agent/router_fallback.rs`_
- [ ] **B4: Multi-model config schema** — Add `local.main`, `local.rlm`, `local.memory` to config. Each slot: `{ model, path, gpu, context_size, temperature }`. Server manager spawns up to 3 llama-server instances. _Ref: `docs/plans/local-model-matrix.md`_
- [ ] **B5: RLM model evaluation** — Systematic experiments to find best RLM model per VRAM tier. Critical for "3 impossible things". See experiment plan below.
- [ ] **B8: Trio mode activation & role-scoped context** ⚡ — **Root cause of 2026-02-20 local session failure.** NanBeige ran in Inline mode (full tool schemas in context) instead of Trio mode. Metrics showed `tool_calls_requested: 1, tool_calls_executed: 0` for 21 consecutive calls — Main was generating tool calls that got blocked as duplicates, proving `strict_no_tools_main` was NOT active. The existing architecture (`DelegationMode::Trio`, `strict_no_tools_main`, `strict_router_schema`, `role_scoped_context_packs`, `router_preflight()`) is **designed but not wired for local sessions**. Steps:
  1. **Trace config loading**: Follow the path from LM Studio detection → `DelegationMode` selection → `apply_mode()`. Find where Trio mode should activate and doesn't. Add `info!` log: `"delegation_mode={:?}"` at startup.
  2. **Verify `router_preflight()` fires**: Add `info!` when preflight runs AND when it's skipped (with reason). Currently silent on skip.
  3. **Verify role-scoped context packs differ by role**: When `role_scoped_context_packs = true`, Main must NOT receive tool schemas. Check `build_context()` or equivalent — does it actually branch on role?
  4. **Slim Main's system prompt for local**: Main (3B) needs ~500 tokens of identity + conversation rules + task state. NOT the full AGENTS.md + SOUL.md + TOOLS.md (~15-20K tokens). Add a `build_local_main_prompt()` that strips everything except personality core and working memory.
  5. **Verification**: Start local session → check `nanobot.log` for `delegation_mode=Trio` → send a message requiring a tool → confirm Main emits natural language intent (not a tool call) → confirm Router preflight intercepts → confirm Specialist executes tool → check `metrics.jsonl` for correct role labels.
  _Ref: 2026-02-21 diagnostic session, `docs/local-ai-swarm-architecture.md`, `src/agent/router.rs`_
- [ ] **B9: Compaction safety guard** — Compaction using ministral-3-8b crashes with `n_keep (12620) >= n_ctx (8192)` when context exceeds specialist's window. This creates a **death spiral**: failed tools → context grows → compaction fails → context grows more → model performance degrades. Two crashes observed at 22:49:46 and 22:53:56 in the same session. Steps:
  1. **Pre-flight check**: Before sending compaction prompt, compare `estimated_tokens` against `model_ctx * 0.85`. If over, skip compaction gracefully (log `warn!`, don't error).
  2. **Emergency truncation**: If compaction is skipped AND context > 90% of Main's window, drop oldest non-pinned turns (keep system prompt + last 3 turns + pinned context).
  3. **Circuit breaker on duplicate tool calls**: After 2 consecutive all-blocked turns (not 7-8 as observed), force-stop the tool loop and emit the last text response. The current `max_same_tool_call_per_turn: 1` blocks individual calls but doesn't stop the retry loop.
  4. **Better duplicate feedback**: Instead of just "blocked", inject: `"Tool already called with these args. Result was: [previous result]. Use it or try different arguments."` — give the model the cached result instead of an error.
  5. **Verification**: Start local session → trigger a tool call → observe at most 2 retries in `nanobot.log` → confirm compaction either succeeds or gracefully skips → confirm no `n_keep >= n_ctx` errors.
  _Ref: 2026-02-21 diagnostic session, `nanobot.log.2026-02-20` lines at 22:49:46 and 22:53:56_

### 🟡 Important — do soon

- [ ] **I0: Trio pipeline actions** — Router can only emit ONE action per turn. Multi-step tasks (research + synthesize) fail because the router picks one tool and stops. Need pipeline-as-first-class router output + shared scratchpad between trio roles. _Ref: `thoughts/shared/plans/2026-02-20-trio-pipeline-architecture.md`_
- [ ] **I1: Local role/protocol crashes** — Fix `system` role crash, alternation crash, orphan tool messages. Thread repair pipeline exists but needs hardening. _Ref: `docs/plans/local-trio-strategy-2026-02-18.md`, `docs/plans/local-model-reliability-tdd.md`_
- [ ] **I2: Non-blocking compaction** — Spawn compaction as background task via `tokio::spawn`, swap result when done. Three tiers: background (80%), aggressive (85%), emergency truncation (95%). _(Spacebot idea)_
- [ ] **I3: Context Gate** — Replace dumb char-limit truncation with `ContentGate`: pass raw / structural briefing / drill-down. Zero agent-facing API changes. _Ref: `docs/plans/context-gate.md`, `docs/plans/context-protocol.md`_
- [ ] **I4: Multi-provider refactor** — Break up `SwappableCore` god struct, extensible provider registry, fallback chains. _Ref: `docs/plans/multi-provider-refactor.md`, `docs/plans/nanobot_architecture_review.md`_
- [ ] **I5: Dynamic model router** — Prompt complexity scoring (light/standard/heavy), auto-downgrade simple messages to cheaper models. _Ref: `docs/plans/dynamic-model-router.md`_
- [x] **I6: Context Hygiene Hooks** — ✅ Implemented as `anti_drift.rs` (851 lines, 25 tests). PreCompletion: pollution scoring, turn eviction, repetitive-attempt collapse, format anchor re-injection. PostCompletion: thinking tag stripping, babble collapse. _Ref: commit `56dedce`, `src/agent/anti_drift.rs`_
- [ ] **I8: SearXNG search backend** — Replace Brave Search (API key required, rate-limited) with SearXNG (free, local, unlimited). 3 touchpoints: 1) `schema.rs`: add `provider: String` ("brave"|"searxng", default "searxng") + `searxng_url: String` (default "http://localhost:8888") to `WebSearchConfig`. 2) `web.rs`: add SearXNG path in `execute()` — `GET {url}/search?q={query}&format=json`, parse `results[].title/url/content`. No API key needed. 3) `registry.rs`+`tool_wiring.rs`: extend `ToolConfig` to carry `search_provider`+`searxng_url`. Fallback: if SearXNG unreachable and Brave key set, use Brave. If neither, helpful error: "Run `docker run -d -p 8888:8080 searxng/searxng` or set a Brave API key." Onboard integration: `cmd_onboard()` prints Docker one-liner. Optional `nanobot onboard --search` auto-pulls+starts+configures SearXNG. _SearXNG container tested working 2026-02-20: `docker run -d -p 8888:8080 --name searxng searxng/searxng` + enable JSON format in settings.yml._
- [ ] **I7: Semantic Fragmentation** — Topic-aware context packing using TF-IDF clustering. Each message gets a TF-IDF vector (hand-rolled, ~40 lines — no LGPL deps). Pairwise cosine similarity → `kodama` (MIT, zero deps, fastcluster-speed) agglomerative clustering → topic clusters. On each LLM call: score clusters against current query, budget-pack active cluster verbatim + inactive clusters as 1-line summaries + last 2 turns always verbatim. Replaces all-or-nothing compaction with surgical per-cluster compaction. New module `src/agent/semantic_context.rs`, ~300 lines. **Deps**: `kodama = "0.3"` (MIT/Unlicense, zero transitive deps). **Why**: Nemotron 8B router gets focused context where routing is obvious; Ministral 8B specialist gets exactly the context needed for tool execution — not everything that ever happened. Compounds with I6 (hooks clean within a cluster, fragmentation cleans across clusters).

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
- **Deterministic fallback too narrow**: `router_fallback.rs` has 2 patterns only — URLs→web_fetch (first URL), "latest news"→spawn. Everything else→ask_user. Misses research/report patterns.
- **Specialist has no tools**: `dispatch_specialist()` sends a single-shot chat — no tool access. Can synthesize given context but cannot fetch/execute.
- **Trio never tested end-to-end**: As of 2026-02-19 handoff, the full trio flow (Main→Router→Specialist) has never completed a real task through LM Studio.
- **2026-02-21 diagnostic: Trio mode didn't activate.** NanBeige ran as Inline main with full tool schemas. 21 metrics entries show `tool_calls_requested: 1, tool_calls_executed: 0` — model generated tool calls (proving it had tool schemas) that were blocked as duplicates. Compaction crashed twice (`n_keep 12620 >= n_ctx 8192`). Death spiral: blocked tools → context grows → compaction fails → context grows more. Session lasted 8 minutes before manual switch to Opus. See B8 and B9.
- **System prompt is ~15-20K tokens** even before conversation starts. Opus first call: `prompt_tokens: 21705`. A 3B model with 8K context has zero room. Even with 32K context, 15K of prompt leaves only 17K for conversation — and most of that prompt is AGENTS.md/SOUL.md/TOOLS.md that small models can't follow anyway.
- **Metrics broken for local models**: All NanBeige calls show `prompt_tokens: 0, completion_tokens: 0, elapsed_ms: 0`. Token counts from llama.cpp `usage` field aren't being captured. Can't diagnose context issues without this data.

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
| Non-blocking compaction | Backlog I2 | Phase 0 |
| Status injection | Backlog N6 | Phase 0 |
| Message coalescing | Backlog N7 | Phase 0 |
| Branch concept (context-fork) | Not started | Phase 2 (related to swarm) |
| Prompt complexity routing | Backlog I5 | Phase 0 |
| Memory bulletin (Cortex) | Not started | Phase 3 (related to memory) |

---

## Done ✅

- ~~B1: 132 compiler warnings~~ → 0 warnings (2026-02-20)
- ~~B2: 2 test failures~~ → 1248 pass, 0 fail (2026-02-20)
- ~~Fix: Subprocess stdin steal~~ — `.stdin(Stdio::null())` on all 4 spawn sites in shell.rs + worker_tools.rs (2026-02-20)
- ~~Fix: Esc-mashing freezes REPL~~ — drain_stdin() after cancel (2026-02-20, commit 57ec883)
- ~~Fix stale comment in `ensure_compaction_model`~~ (2026-02-17)
- ~~Raise tool result truncation threshold~~ (2026-02-17)
- ~~Document multi-session CONTEXT.md race~~ (2026-02-17)
- ~~Input box disappears during streaming~~ (2026-02-17)
- ~~Agent interruption too slow~~ (2026-02-17)
- ~~Subagent improvements (wait, output files, budget, compaction)~~ (2026-02-18)
- ~~Tool runner infinite loop fix~~ (2026-02-18)
