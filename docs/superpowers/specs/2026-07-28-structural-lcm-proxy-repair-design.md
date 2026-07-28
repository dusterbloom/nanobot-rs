# Structural LCM and Proxy Repair Design

Date: 2026-07-28

## Problem

One failed session exposed three independent violations of the runtime's intended
single-path design:

1. Proxy calls are decoded twice. Execution accepts both the current
   `tool_name` / `tool_args` shape and the legacy `name` / `args` shape, while
   routing only accepts the legacy shape. Current calls therefore execute but
   lose their semantic tool identity before deduplication, caching, guarding,
   leasing, and persistence.
2. LCM pressure is partly derived from a hard-coded 24,576-token Higgs retained
   session cap instead of the active model's context budget. This can force
   blocking compaction while the model still has ample semantic context.
3. LCM and memory reflection are coupled to an on-demand Higgs sidecar. Every
   last lease stops an owned process, so repeated compactions repeatedly load a
   model. A sidecar crash also inserts a long acquisition timeout before the
   already-available main model is used.

These are structural mismatches, not tuning problems. The repair removes each
duplicate authority.

## Invariants

### Proxy calls have one decoder

`ToolRegistry` owns one canonical proxy-call resolver. Given the outer tool name
and arguments, it returns the semantic tool name and semantic arguments for both
the current and legacy wire shapes.

Both execution and routing use that resolver. A call that executes as
`web_fetch` must therefore also be guarded, cached, leased, and persisted as
`web_fetch`. The legacy keys remain accepted at the wire boundary only; no
downstream component knows about the compatibility shape.

Malformed proxy envelopes remain ordinary `tool` calls so execution returns the
existing validation error instead of routing inventing semantics from incomplete
input.

### LCM pressure has one context authority

LCM soft and hard thresholds are calculated only from the active
`TokenBudget::available_budget(tool_definition_tokens)` and the configured
`tau_soft` / `tau_hard` fractions.

The Higgs retained-session cache is a transport optimization. Its capacity,
admission fraction, environment variables, and prompt-token calibration do not
participate in semantic context pressure. If Higgs evicts or compresses retained
KV, the next request may cold-prefill but the persisted message history remains
correct.

The hard-pressure decision remains:

```text
estimated active tokens >= available model context * tau_hard
```

The soft-pressure decision uses the same denominator and `tau_soft`. No local
backend constant or server-specific fallback can trigger LCM.

### LCM always uses the active main model

Every `ContextCompactor` used for LCM is constructed from the active core's main
provider, main model, and main context ceiling. Core rebuilds replace those
three values atomically, so `/local` and `/model` changes cannot leave compaction
pointing at an old endpoint or model.

There is no compaction sidecar acquisition, health timeout, fallback branch,
lease, or shutdown path. The former fallback becomes the only path.

Memory reflection remains a separate concern. It continues to use the resolved
memory provider/model (explicit memory configuration, specialist, or main
fallback), but it invokes that provider directly and never acquires a compaction
lease. This preserves intentional cheap/specialist reflection without allowing
it to select the LCM model.

## Runtime Shape

The active core carries:

- the main provider/model and its `TokenBudget`;
- one main-model `ContextCompactor`;
- the separately resolved memory provider/model used by reflection.

It does not carry a `CompactionSidecarManager` or optional compaction provider.
Local-provider construction resolves only main, delegation, and specialist
roles. LCM execution clones the core compactor and retains the existing
cancellation-safe mutation and SQLite checkpoint transaction.

Background, cron, exit, and `/learn` reflection construct `Reflector` directly
from the core memory provider/model. Their availability depends on the memory
configuration and provider result, not on an unrelated sidecar lease.

## Configuration Migration

Remove `lcm.compactionModelDir`, `lcm.compactionPort`, and the legacy
`agents.defaults.higgsCompaction*` fields from the typed schema. Serde's existing
unknown-field behavior makes old JSON configurations load without activating a
hidden alternate path. Serializing the configuration drops those obsolete keys.

Remove the compaction-sidecar manager, lease, registry, retry, and tests from
`higgs.rs`. Higgs's main-server lifecycle and retained-session transport remain
unchanged.

## Failure Semantics

- A malformed proxy envelope fails through the existing tool validation path.
- A main-model compaction provider failure follows the existing LCM failure mode:
  the tentative DAG mutation rolls back unless the SQLite checkpoint commits.
- Soft compaction failure leaves the prior active window usable.
- Hard compaction failure cannot silently switch models or truncate solely
  because a sidecar is unavailable.
- Reflection failures are logged/reported by their existing caller and do not
  affect LCM state.

## Verification

Add regression coverage that proves:

1. Current and legacy proxy envelopes canonicalize identically for routing and
   execution.
2. Repeated current-shape `web_fetch` calls reach web-specific cache/guard
   behavior rather than generic `tool` behavior.
3. LCM soft/hard decisions depend on the active model budget and are unchanged
   by local retained-session settings.
4. A local core's LCM compactor uses the main provider, model, and context size.
5. Reflection works without a compaction manager.
6. Obsolete sidecar configuration is ignored and not serialized.

Run focused tests first, then `cargo test`, `cargo build`, and
`scripts/turn_bench.sh` because the router, context-budget hot path, and agent
loop are changing. Before any commit, run GitNexus change detection against
`main`.
