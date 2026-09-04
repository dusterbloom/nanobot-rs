# Certain Detail PR Fixes Design

## Goal

Integrate only confirmed, behavior-changing fixes from Detail PRs #7-#33 onto
`feature/adaptive-capacity-nanobot`, without importing cosmetic changes,
advisory-only behavior, inert configuration, or unnecessary patch bulk.

## Release Constraints

- Escha must continue to use the native trellis kernel on all 40 expert layers;
  no affine fallback may be introduced.
- Preserve the adaptive-capacity implementation and its live endpoint contract.
- Keep the production path singular: channel -> agent loop -> provider -> tools
  -> reply.
- Reimplement each fix minimally on the integration branch; do not cherry-pick
  a Detail PR wholesale when its patch contains avoidable scaffolding.
- Every behavior change requires a focused regression test that is observed
  failing before the production change and passing afterward.
- Build and test only in release mode.
- Do not stage or modify the existing worktree changes to `AGENTS.md` or
  `CLAUDE.md`.

## Included Fixes

| PR | Behavior to preserve | Minimal integration shape |
|---|---|---|
| #9 | Bound model-controlled pipeline voting and refinement cost | Small schema/parser caps plus overflow-safe arithmetic; use a genuinely small voting cap |
| #10 | Do not restart a Higgs server after it has recovered | Health check immediately before automatic restart |
| #12 | Use local wire protocol for bare model IDs served by local/LAN endpoints | Select protocol from resolved provider endpoint while retaining the MLX exception |
| #13 | Overflow fallback must reserve the response budget actually sent | Derive fallback target from the effective per-call `max_tokens` within adaptive capacity |
| #14 | FTS tokenizer migration must not strand an empty index | Transactional migration with a focused empty-index recovery check |
| #16 | Lease-renewal checkpoints must reach the next model call | Append the checkpoint assistant message before the renewal scaffold |
| #17 | Overlapping unverified spans must not leak text | Merge overlapping/adjacent spans before reverse-order redaction |
| #18 | Anti-drift collapse must preserve assistant/tool pairing | Retain `tool_calls` when replacing repetitive assistant preambles |
| #20 | Explicit `finish_reason=stop` must survive a missing `[DONE]` | Track whether a finish reason was observed separately from its value |
| #22 | `lcm_expand` must understand spaced ID ranges | Extend the existing ID parser with the smallest safe spaced-dash handling |
| #24 | Proxy-catalog exclusions must match complete tool names | Filter on `Tool::name()` before rendering hints |
| #27 | Repeatedly blocked tools must receive the intended final-answer scaffold | Push the already-constructed scaffold into the message log |
| #28 | System announcements must never coalesce with user messages | Exclude system, idle, and command messages at both coalescing boundaries |
| #29 | UTF-8 voice previews must not panic | Reuse `floor_char_boundary` at each affected log slice |
| #30 | Local subagent aliases must resolve to the active served model | Apply the existing environment-aware resolver to every precedence tier |
| #31 | Failed plan steps must terminate instead of stalling | Make failure terminal and stop the turn when no checkpoint exists |
| #32 | Idle-turn writes must reject `..` traversal | Normalize the checked and executed path identically; retain a documented symlink limitation for separate hardening |
| #33 | Unified-diff body lines beginning `---`/`+++` must not be dropped | Treat headers specially only outside an active hunk |

## Explicit Exclusions

- #7: documentation-only.
- #8: terminal link-rendering polish.
- #11: advisory warning only; does not enforce the taint boundary.
- #15: optional malformed skill-path presentation with no established runtime
  failure in the current workload.
- #19: activates an otherwise unused configuration surface.
- #21: warns without correcting the invalid memory-provider configuration.
- #23: optional knowledge-graph path and a process-global `HOME` test race.
- #25: removes a false safety claim but leaves unrestricted Python execution.
- #26: broadens generic context extraction beyond the proven nested-error case.

## Integration Structure

Changes land as independent commits ordered by risk and dependency:

1. Security and destructive-tool correctness: #32, #33, #9.
2. Protocol and message invariants: #18, #12, #20, #27, #28.
3. Adaptive-capacity and control-flow correctness: #13, #16, #31.
4. Durable memory and evidence recovery: #14, #22, #17.
5. Operational reliability: #10, #24, #29, #30.

Where a Detail branch conflicts with adaptive-capacity code, reproduce the
behavior from a failing test and implement it against the adaptive interfaces;
do not resolve by accepting either side wholesale.

## Verification

For every fix:

1. Run GitNexus impact analysis for each production symbol before editing.
2. Add the smallest regression test and run it in release mode to demonstrate
   the expected failure.
3. Apply the minimal production change.
4. Run the focused test in release mode to demonstrate success.
5. Run the nearest affected release-mode test suite and commit the isolated
   change.

After the stack:

- Run `cargo fmt --all -- --check`.
- Run `cargo build --release`.
- Run `cargo test --release`.
- Run `scripts/turn_bench.sh` because provider, agent-loop, and context behavior
  changes are included.
- Run GitNexus change detection and confirm only the intended symbols and flows
  changed.
- Reconfirm the Higgs/Escha native-trellis routing and absence of affine expert
  fallback.

## Completion Condition

The integration is complete only when all 18 behaviors have isolated regression
evidence, the full release suite passes, the speed track shows no unexplained
regression, adaptive-capacity behavior remains intact, and Escha still executes
all expert layers through the native trellis kernel.
