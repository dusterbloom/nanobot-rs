# Reliability Gates

Use explicit gates before promoting configuration changes.

## Minimum Acceptance Gates

- `success_rate >= 0.90` on the golden task set
- `p95_latency_ms` meets agreed target
- `oom_rate = 0` over 200 consecutive runs
- `loop_abort_rate <= 0.02`
- no increase in `tool_call_protocol_mismatch` rate

## Regression Guard

Require all of the following:
- no category worsens by more than 10% relative to baseline
- no new high-severity failure category appears
- fallback route remains functional under load

## Validation Procedure

1. Capture baseline metrics.
2. Apply one change.
3. Re-run same benchmark and same-day failure audit.
4. Compare against baseline and gates.
5. Promote only on pass; otherwise roll back.

## Incident Exit Criteria

Consider incident mitigated only when:
- top failure category rate drops to acceptable threshold
- no critical hard failures observed in verification window
- documented root cause and mitigation are recorded
