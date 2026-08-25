# Current Plan — v0.5 "Persistent Local Agent"

**SHIPPED** (all phases landed on main):

- [x] E1 Idle agency (`a51b792`): designated-session idle turns, warm-only +
      hour cap, quiet-by-default, idle write-allowlist, doubling backoff.
- [x] E2 Memory + skills (`3fff1b1`): create_skill tool; dream cron
      (`--dream`, nightly job when dream.enabled); MEMORY.md append+dedup;
      DREAM_PROPOSALS.md.
- [x] E3 Fork-test-merge: `evolution` skill in workspace. Merge stays human.
- [x] P2 Error protocol (`a9f664d`..`0889ac9`): 40/40 tool impls typed-native.
- [x] P3 Trait flip (`cc58b02`): `Tool::execute(params, ctx) -> ToolResult`
      is the one required method; String ladder + funnel + require_str!
      deleted; from_legacy private in host_bridge.
- [x] E4 Anthropic demotion (`5e9eb7a`): native client + OAuth lane deleted
      (−~1.6k); anthropic keys/models route via openai-compat endpoint.
      Revert hatch = the commit.
- [x] E5 Cluster idle primitives (`617f51b`): cluster.idleModels allowlist,
      route_idle, mark_unhealthy; dead Shared.cluster_router deleted. The
      per-idle-turn provider consult is deferred until a real LAN peer makes
      it testable (needs a per-turn provider override mechanism).

Deferred backlog (unchanged): god-fn/file splits, surface diet, single-stream
rearchitecture, idle-turn exec allowlist (ceiling documented in E1).

Branch state: commits unpushed through `617f51b`; push when ready.
