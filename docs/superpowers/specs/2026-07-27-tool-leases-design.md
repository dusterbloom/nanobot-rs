# Tool Leases — Structural Loop Prevention

Date: 2026-07-27
Owner: nanobot-rs (agent loop)
Review: codex v0.145.0 (gpt-5) — recommended renewable leases over both
pure reactive (Design A) and pure fixed-budget prevention (Design B).

## Problem (source-grounded)

Live session `20260727_161730_6e61a0` on 2026-07-27 16:19-16:21: model
made **13 consecutive `exec` tool calls**, all variations of
`grep "compaction|tau_soft|context_size" <path>`. Existing defenses:

| Defense | Why it didn't trip |
|---|---|
| `ToolGuard::allow()` | Exact `(name, args)` dedup; args varied |
| `collapse_repetitive_attempts` | Fingerprint `name(args)|50chars`; args varied |
| `inject_format_anchor` | Only fires when recent turns had NO tool calls |
| `advance_response_boundary` | Only fires after side-effect tools (writes); loop was read-only |

Model self-corrected after 13 calls ("My bad — I was stuck in a loop").
No structural defense broke the loop.

## Principle

> "We do not resort to patches, we speak the truth and innovate whenever
> possible." Reactive injections (`[system] …`, `[format-anchor] …`) are
> patches. The lease design **removes the failure class** structurally.

## Design — Renewable Tool Leases

### Core mechanic

Each user turn starts with a **tool lease** of `TOOLS_PER_LEASE = 5` tool
iterations. After `TOOLS_PER_LEASE` iterations in one turn:

1. **Tool definitions are removed from the request.** The model sees no
   tool schema → cannot emit tool calls.
2. The model must do one of:
   - **Produce a final text answer** (turn ends normally), OR
   - **Emit a renewal request**: a structured checkpoint describing
     findings so far, remaining question, and next bounded actions.

3. If the model emits a valid renewal request, the lease is renewed for
   another `TOOLS_PER_LEASE` iterations, tools come back, and the loop
   continues. If the renewal request fails validation (no findings, no
   next-actions, or repeats prior work), the model gets one more
   text-only turn, then the turn is force-ended.

### Progress signal in tool results (B3)

Every tool result includes a prefix:
`[Tool call N of M this lease — L leases remaining this turn]`

For example: `[Tool call 3 of 5 this lease — 2 leases remaining]`

The model has a continuous view of remaining budget and self-regulates
instead of being interrupted. `L` (remaining leases) starts at a
configurable cap `MAX_LEASES_PER_TURN = 3` (so up to 15 tool calls per
turn if every lease is renewed with valid checkpoints).

### Coarse-family sub-cap

Within a lease, **at most `CONSECUTIVE_COARSE_FAMILY_CAP = 3`** tool
calls from the same coarse family in a row. Coarse family:
- `exec:grep`, `exec:rg`, `exec:find`, `exec:ls`, `exec:cat` (read-only
  exec, classified by first word of command)
- `read_file`, `list_dir`, `find_files`, `search_files`, `recall`,
  `read_skill` (read tools)
- `web_search`, `web_fetch` (web tools)
- Each write tool is its own family

Cap hit → tool stripped for one iteration, replaced with a receipt
explaining the cap. Next iteration can use a different family. This
makes the specific live failure (`13 × exec:grep`) structurally
impossible: cap fires at 3.

### Lease renewal validation

A renewal request is an assistant message that contains all three:
- **Findings**: a non-empty list of what the model learned in the
  exhausted lease (`findings:`, `learned:` or `discovered:`)
- **Remaining question**: what's still unknown (`next:` or `still need:`)
- **Next bounded actions**: at least one specific tool call the model
  plans to make (`will:` or `plan:`)

Validation is **regex-based and lenient** (small models produce messy
text). It must be deterministic and visible: when a renewal is rejected,
the rejection message states exactly which of the three is missing.

### What this replaces

| Old mechanism | Status under leases |
|---|---|
| `[system] Report what the previous tool results showed` | Removed — leases force this naturally |
| `consecutive_tool_iterations` counter (would-be Fix 2) | Not needed |
| `advance_response_boundary` for read-only loops | Removed for this case (kept for write tools) |
| `inject_format_anchor` for tool-only drift | Removed for this case (kept for genuine XML-imitation drift) |

### What stays

| Mechanism | Why kept |
|---|---|
| `ToolGuard` exact-arg dedup | Still useful — saves real work within a lease |
| `collapse_repetitive_attempts` | Defense in depth during compaction (not primary) |
| `max_tool_iterations=60` cap | Outer bound; leases live well below it |
| `inject_format_anchor` for text-only drift | Different failure mode |

### Configuration

Three new fields on `AgentDefaults` config:
- `tool_lease_size: u32` (default 5)
- `max_leases_per_turn: u32` (default 3)
- `coarse_family_cap: u32` (default 3)

Zero config = current loop failure structurally impossible.

## TDD test plan

Tests live alongside existing tests in:
- `src/agent/tool_guard.rs` — coarse family classifier
- `src/agent/agent_loop/shared.rs` tests — lease lifecycle
- New `src/agent/lease.rs` — lease state machine, renewal validator

### RED tests (write first, watch fail)

1. `coarse_family_classifies_exec_grep_as_readonly_search`
   `exec("grep pattern path")` → `coarse_family = "readonly_search"`.
2. `coarse_family_distinguishes_readonly_exec_from_write`
   `exec("rm file")` → `coarse_family = "exec:write"`. `exec("grep x")` →
   `coarse_family = "readonly_search"`.
3. `lease_allows_n_tool_calls_then_strips_tools`
   With `lease_size=3`, three tool iterations are allowed; on the 4th
   LLM call the tool_defs slice is empty.
4. `lease_renewal_restores_tools_when_checkpoint_is_valid`
   After lease exhaustion, assistant message `"findings: X. next: Y. will:
   grep Z"` is accepted, tools return for the next iteration.
5. `lease_renewal_rejected_when_findings_missing`
   `"next: Y. will: grep Z"` (no findings) → renewal rejected, tools stay
   stripped, turn force-ends after one more text-only iteration.
6. `coarse_family_cap_blocks_fourth_consecutive_grep`
   Within a lease, three `exec:grep` calls are allowed; the 4th is
   blocked with a receipt. A different family (`exec:find`) is still
   allowed.
7. `tool_result_includes_lease_progress_signal`
   Tool results contain `[Tool call N of M this lease — L leases remaining]`.

### GREEN sequence

1. Coarse family classifier (Fix-equivalent: a stripped-down version of
   A3's read-only exec detector).
2. Lease state machine (new module `lease.rs`).
3. Lease lifecycle hooks in `agent_loop/shared.rs`.
4. Renewal validator (regex-based).
5. Tool-result prefix injection.
6. Coarse-family cap enforcement.
7. Config schema additions.

## Out of scope (deliberately)

- `{"name":"tool"}` hallucination — separate protocol bug
- Path/arg canonicalization for ToolGuard keys
- Cross-reordering dedup
- A3 full readonly-exec result caching (may add later as optimization)

## Risk mitigation (per codex)

> "Keep renewal rules narrow, deterministic, and visible to the model."

Implementation rules:
- Renewal validator is regex-only, no fuzzy matching
- Rejection messages name the missing field explicitly
- Every lease transition is logged at INFO level
- Config-exposed thresholds (no magic numbers in code)
- Unit tests cover both small-model messy text and well-formed text
