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
- [~] P2/P3 Error protocol (batches `a9f664d`, `6d06d21`; 18/40 impls typed):
      Migrated: system_info, tool_status, file_preview, todo, check_inbox,
      send_email, get_skills, cron, inspect_tool_result, apply_patch,
      create_skill, checkpoint, backtrack, plan, python_kernel, execute_code.
      Remaining P2: browser, cua, filesystem/mod (7 tools), filesystem/write,
      web_search + web_fetch (ctx overrides), shell (ctx), and merging the 4
      partial tools (message, spawn, remember, recall — logic into their
      existing execute_typed, delete String bodies).
      Then P3 flip: trait `execute(params, ctx) -> ToolResult` becomes the one
      method (delete String execute default, execute_with_context,
      execute_with_result(_and_context), funnel_legacy, require_str!);
      registry execute_inner calls tool.execute directly; classify_tool_error
      + ToolError::from_legacy move private into host_bridge (its serde wire
      round-trip needs them; errors.rs goes clean); tests switch
      `tool.execute(p)` -> bridge fn or typed asserts; update AGENTS.md
      "tools return String" line + docs/error-protocol-backlog.md.
      MIGRATION RULES (proven byte-stable, follow exactly):
      1. Param/validation sites -> InvalidArgs{message: <legacy text minus
         "Error: " prefix>} (keeps "is required" worked-example appends).
      2. Not-found -> NotFound, policy denials -> PermissionDenied, everything
         else -> Execution — ALL carry the exact stripped legacy message.
      3. Timeout/Network/RateLimited ONLY if the legacy string is already the
         canonical render ("Command timed out after Ns"); else Execution.
      4. No-"Error:"-prefix strings (e.g. "Error reading X: ...") were
         SUCCESS-channel — keep Ok(...) and mark the quirk.
      5. Inner closures/threads returning flat strings: one tool-local
         strip_prefix("Error:") boundary split (python_kernel pattern).
      6. Tests calling tool.execute(p) keep passing via the trait default
         bridge; no test churn until the P3 flip.
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
