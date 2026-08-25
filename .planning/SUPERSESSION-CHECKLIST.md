# Supersession Concern-Checklist — main d7a801c..c6103a2 vs branch HEAD

2026-08-25. Pre-squash evidence for wholesale branch-win on all conflicted files.
Main's delegated-tool theme (7 commits) is superseded by design (branch deleted
`recall_tool_result` deliberately); excluded here.

| sha | Concern fixed on main | Verdict | HEAD evidence |
|---|---|---|---|
| adc44c9 | Batch crossing lease boundary must not partially consume allowance or orphan tool results | COVERED | Reject consumes nothing (lease.rs:156-162); carrier journaled before execution (tool_engine.rs:557-575); per-tool_call_id rejection receipts (shared.rs:4188-4214); pairing invariant tested (tests.rs:9653-9672). Partial admission is deliberate — every blocked member gets a "not executed" receipt. |
| 7bc1d19 | Strip/restore churn + tool-schema instability at exhaustion | COVERED | Strip machinery deleted; tool_defs never mutated at execution time (shared.rs:4119-4126); receipt-ignoring model bounded by NO_PROGRESS_HARD_STOP=4 (shared.rs:1495-1500); catalog byte-stability asserted (tests.rs:9631-9637). |
| 2b72e3b | Local renewal loops past 12 tools | MOOT | Renewals re-enabled by design with split budgets (lease.rs:219-237, commit e455324); turns hard-bounded by max_iterations=15 (schema.rs:1408, shared.rs:1361). Policy difference, no unbounded path. |
| a05fc81 | (i) session locks shared across entrypoints (ii) atomic protocol-group persist | MOOT / GAP | (i) MOOT: run() and process_direct* never coexist per process (cron/executor.rs:14-19; cli/mod.rs:863). (ii) GAP: per-message add_message loop breaks mid-group (shared.rs:805-818); add_messages swallows failures (db.rs:2410-2412); carrier+receipts are two persists (tool_engine.rs:574, shared.rs:4213). |
| 3cbc59d | /clear racing in-flight turn → resurrected history | GAP | Dispatch on main loop outside session lock (mod.rs:351-365) vs spawned turns holding it (mod.rs:392); cmd_clear wipes (gateway_commands.rs:102-116); later persists re-add cleared rows. No branch test races them. |
| 3fb926d | Permit starvation / head-of-line blocking | GAP | Permit acquired in run() loop head before spawn (mod.rs:367-374); default 4 permits (schema.rs:383) — exhaustion stalls all sessions + dispatch. Main moved acquire inside task after lock. |
| b9d0055 | Coalescing side-path bypasses unified gateway | GAP | Different-session message during coalescing spawns parallel path (mod.rs:291-318) skipping is_system + /-command interception — a /clear from another session reaches the LLM as text. |
| 8462105 | Tool-topology change must rotate Higgs session | COVERED | Stronger: frozen per-session catalogs (agent_core.rs:528-536, shared.rs:2198-2226); every reservation validates frozen tool hash + generation, drops stale (agent_core.rs:816-846, 891-916). |
| 1204aa0 | Retained replay append-only | COVERED | Byte-identical handle passthrough + deterministic cap (filters.rs:181-215); fingerprint survives reload (prepare_context.rs:369-390); sanctioned trims clear it (shared.rs:1903). Missing main's rotate-on-divergence, but divergence is WARN-diagnosed (shared.rs:3237-3337) — design difference, not a hole. |
| 1957f5d | (i) preview cap vs long provider call_id (ii) stash raw before shaping | GAP (i) / COVERED (ii) | (i) tool_engine.rs:391-392 `return header;` unbounded — the exact line main fixed to .take(cap); content gate doesn't re-bound (context_gate.rs:91-104). (ii) store_then_render_tool_result + abort_turn_on_stash_failure (tool_engine.rs:866-886, :355). |
| 4ac502c | formatting | MOOT | rustfmt over delegated-tool files. |

## Verdict: wholesale branch-win NOT safe as-is — 5 ports required pre-squash

1. Move `/`-command dispatch inside the spawned task under the session lock (3cbc59d).
2. Acquire the permit inside the task after lock acquisition (3fb926d).
3. Replace coalescing side-path spawn with pending_msg push-back (b9d0055).
   (1-3 = one coherent ~80-line port of main's final mod.rs gateway section.)
4. Add `add_messages_checked`, use in `persist_pending_protocol_messages` (a05fc81).
5. `header.chars().take(cap).collect()` (1957f5d).
