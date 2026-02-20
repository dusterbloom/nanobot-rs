---
name: slm-agentic-reliability
description: Design, tune, and harden local small language model (SLM) agentic systems on constrained hardware, especially NVIDIA RTX 3090 (24 GB VRAM) and Apple Silicon with 32 GB unified memory. Use when choosing model and quantization, setting context and concurrency limits, building or debugging multi-step agent loops, adding reliability controls (timeouts, retries, checkpoints), or triaging failures such as OOM, loop runaway, tool-call hallucinations, and context overflow.
---

# SLM Agentic Reliability

## Overview

Start from observed failures in today's local-model sessions, then harden the system with constrained-hardware guardrails. Keep changes small, measurable, and reversible.

## Start With Failure Audit

Run:

```bash
python3 scripts/audit_local_failures.py --date "$(date +%F)"
```

Use this output as the mandatory starting point:
- category counts
- session hotspots
- representative failure samples
- first-action hint per category

## Failure-First Workflow

1. Audit today's failures with `scripts/audit_local_failures.py`.
2. Rank categories by `frequency x user impact`.
3. Map each category to fixes in `references/local-failure-taxonomy.md`.
4. Apply one fix per iteration.
5. Re-run the same-day audit to confirm reduction.
6. Promote only fixes that pass gates in `references/reliability-gates.md`.

## Set Constraints

Collect and keep explicit:
- target profile (`rtx3090` or `apple-m4-32g`)
- `p95_ms` target
- minimum success rate
- max context length
- concurrent sessions
- tool reliability requirements (strict JSON, deterministic schema)

If user targets are missing, propose:
- `success_rate >= 0.90`
- `p95_ms <= 4000` for short requests
- `oom_rate = 0` in 200 runs
- `loop_abort_rate <= 0.02`

## Apply Core Guardrails

Enforce in every agentic loop:
- hard budgets (`max_turns`, `max_tool_calls`, `max_runtime_ms`)
- tool allowlist plus strict argument validation
- checkpointed state after side effects
- bounded retries with retryable-error classes only
- watchdog timeout and cancellation path
- output schema validation before commit
- fallback model route under memory or latency pressure

Use patterns in `references/loop-reliability-patterns.md`.

## Adapt By Hardware

Apply profile guidance from `references/hardware-profiles.md`:
- RTX 3090: preserve VRAM headroom for KV cache growth and tool overhead.
- Apple M4 32GB: preserve unified-memory headroom for OS plus runtime.

## Output Contract

When asked to troubleshoot reliability, produce:
1. Failure summary from today's sessions (counts and top categories).
2. Root-cause hypothesis for each top category.
3. Minimal fix plan (one change at a time).
4. Verification plan with explicit pass/fail thresholds.
5. Next two iterations if first fix does not pass gates.

## References

- `references/failure-first-workflow.md`: command sequence for same-day failure triage.
- `references/local-failure-taxonomy.md`: failure classes, symptoms, and fixes.
- `references/hardware-profiles.md`: memory and throughput planning by profile.
- `references/loop-reliability-patterns.md`: bounded-loop patterns and guardrails.
- `references/reliability-gates.md`: release gates and regression checks.
