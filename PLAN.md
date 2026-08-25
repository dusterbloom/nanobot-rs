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
- [x] P3 flip COMPLETE (`cc58b02`): `Tool::execute(params, ctx) -> ToolResult`
      is the one required method; String ladder + funnel_legacy + require_str!
      deleted (require_str! lives private in the worker lane); from_legacy is
      private in host_bridge; classify_tool_error + ToolErrorKind remain for
      the registry audit shape. Gate: 2804/0, 0 warnings, higgs E2E
      (exec + remember) green.
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
