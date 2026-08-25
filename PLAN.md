# Current Plan — v0.5 "Persistent Local Agent"

Sequencing decision (Q1=b): agency first, error-protocol codemod later.

- [x] E1 Idle agency (`a51b792`): designated-session idle turns, warm-only +
      hour cap, quiet-by-default, idle write-allowlist, doubling backoff.
      E2E through real run() loop; config `idle.*`, default off.
- [x] E2 Memory + skills (`3fff1b1`): create_skill tool (typed funnel);
      dream cron kind `--dream` + default nightly job when `dream.enabled`;
      MEMORY.md append+dedup; DREAM_PROPOSALS.md. CLI path smoke-tested.
- [x] E3 Fork-test-merge: `evolution` skill installed in workspace
      (`nanobot skills list/validate` OK). No new permissions needed
      (git/cargo already allowed); merge stays a human act via git;
      `/evolution` REPL command skipped — git is the UI for git things.
- [x] P2 Error protocol COMPLETE (batches a9f664d..0889ac9): 40/40 impls
      define execute_typed; zero String execute overrides in production
      tools; funnel only lives in the trait default. Tests unchanged except
      the three ctx-path suites (write/web/shell call execute_typed
      directly; remember's empty-add assertion moved to the MissingArg
      contract).
- [ ] P3 flip (one mechanical change, do with fresh context):
      1. base.rs: rename trait method execute_typed -> `execute(&self,
         params, ctx) -> ToolResult` (REQUIRED, no default); delete the
         String execute default, execute_with_context,
         execute_with_result, execute_with_result_and_context,
         funnel_legacy, and require_str! (rg users first).
      2. All ~40 impls + their test call sites: rename execute_typed ->
         execute (sd). Tests asserting on strings use per-file
         render(ToolResult) -> String helpers (shell.rs has the pattern);
         plain `tool.execute(p)` sites become render(tool.execute(p, &ctx))
         or unwrap().text.
      3. Registry execute_inner: call tool.execute(params, &ctx) directly,
         wrap ToolExecutionResult::from; delete the catch_unwind shape only
         if trivially adaptable (keep catch_unwind — panics still map to
         failure).
      4. errors.rs: move classify_tool_error + from_legacy + ToolErrorKind
         ROUND-TRIP... NO: keep ToolErrorKind + legacy_kind_from_tool_error
         (typed->kind, feeds ToolExecutionResult.error_kind + registry
         worked-example arm at registry.rs:751). Move ONLY classify_tool_error
         + from_legacy to host_bridge as private fns (serde wire round-trip
         at host_bridge.rs:260,394 + into_model_text).
      5. Update AGENTS.md ("tools return String" line) +
         docs/error-protocol-backlog.md status.
      6. Full gate: build + test + clippy (no new warnings) + higgs E2E
         smoke (one exec + one recall turn).
      Landmine: trait defaults currently form a 3-cycle (execute -> typed ->
      with_context -> execute) if an impl overrides NOTHING — the flip
      deletes the cycle by construction.
- [ ] E4 Anthropic demotion (−~1.6k): delete anthropic.rs + factory arm +
      schema; openai-compat + higgs remain. Revert hatch = the commit.
- [ ] E5 Cluster idle compute (~150-200 LOC): per-idle-turn router consult,
      mark_unhealthy on error, cluster.idleModels allowlist (Q6=a),
      delete dead Shared.cluster_router store. Behind cluster feature.

Deferred: surface diet (channels/voice/REPL), god-fn splits, single-stream
rearchitecture.

Known ceiling (ponytail: deliberate): idle-turn exec is unrestricted (the
E1 allowlist gates file tools only); damage bounded by lease (12 tools) and
the observation contract. Upgrade path: exec allowlist in idle turns if a
real incident demands it.
