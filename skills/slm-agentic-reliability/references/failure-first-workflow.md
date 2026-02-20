# Failure-First Workflow

Run this sequence for every reliability task.

## 1) Audit Today's Failures

```bash
python3 scripts/audit_local_failures.py --date "$(date +%F)"
```

If needed, include all scopes:

```bash
python3 scripts/audit_local_failures.py --date "$(date +%F)" --scope all
```

## 2) Prioritize

Rank by:
- `frequency`
- `user impact` (hard failure > degraded quality)
- `blast radius` (single route vs all routes)

Focus on the top one or two categories first.

## 3) Diagnose

For each top category:
- inspect samples
- verify where the failure enters the loop
- confirm whether failure is model, runtime, tool, or prompt contract

Use `local-failure-taxonomy.md` to map category to likely cause.

## 4) Patch Minimally

Change exactly one variable per iteration:
- context size
- model/quant
- concurrency
- tool budget
- timeout/retry
- output schema validator

Avoid bundled changes that hide causality.

## 5) Re-Audit and Gate

Re-run:

```bash
python3 scripts/audit_local_failures.py --date "$(date +%F)"
```

Then apply `reliability-gates.md` pass/fail criteria.

## 6) Promote or Roll Back

- promote only if gates pass
- roll back if regression appears in success rate, p95 latency, or failure rate
