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
- [ ] P2/P3 Error protocol: migrate 18 tool files to execute_typed, flip
      the trait, delete the legacy ladder + from_legacy/classify_tool_error
      (research: docs/research/2026-08-06-error-conventions-and-host-bridge.md §2.6).
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
