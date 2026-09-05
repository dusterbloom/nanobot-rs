# Context decision policy discriminator — 2026-09-05

Clearer instructions changed Escha's tool use. Optional context management produced
no inspections; requiring a post-submission check produced five checks and four
explicit continue decisions. Neither run reset, and both stopped on Higgs capacity
unavailability. The fifth decision-policy check completed, but its following model
request failed before the model could choose an action. This is not evidence that
Escha would ignore low headroom after seeing it.

| Outcome | Optional policy | Explicit decision policy |
|---|---:|---:|
| Requested updates | 20 | 20 |
| Exact snapshots submitted | 5 | 5 |
| Wrong snapshots | 0 | 0 |
| Normally finished user turns | 5 | 4 |
| Post-submission headroom checks | 0 | 5 |
| Explicit continue decisions | 0 | 4 |
| Notes / reset / retrieval calls | 0 / 0 / 0 | 0 / 0 / 0 |
| Automatic compactions | 0 | 0 |
| Model requests | 12 | 15 |
| Observed time to interruption | 254.4 s | 341.6 s |
| Tool errors / duplicate submissions / forbidden actions | 0 / 0 / 0 | 0 / 0 / 0 |
| Pending turns preserved | 1 | 1 |
| In-arm server restarts | 0 | 0 |
| Minimum server prompt capacity | 0 | 0 |
| Repeated-boundary endurance coverage | No | No |

The extra time is an observed cost, not a controlled latency estimate: independent
boots experienced different memory-pressure histories, and the runs stopped at
different points in completing the fifth/sixth user turn. The explicit policy
added model/tool round trips; once retained-cache reuse stopped, these incurred
full prefills. Do not extrapolate one pair to a population effect.

The fourth continue decision cited approximately 3800 estimated tokens remaining;
the advertised next update was approximately 2400 estimated tokens. The fifth
headroom result reported estimated_tokens=11951, remaining_estimate=1802,
prompt_budget_after_output_and_tool_reserve=13753, last_actual_prompt_tokens=13041.
It was durably recorded. The subsequent model request received a capacity failure;
there is no fifth continue/reset decision to score. Both runs entered the
capacity-suspension path with a pending row in SQLite. Eventual gateway-driven
resumption is not exercised by this harness.

## Controls and evidence

Both runs use the B interface: existing LCM/recall/lcm_expand plus
notes/history/context_status/new_context. Only the initial system policy changed.
The explicit variant requires checking headroom after submission, then deciding
continue/retrieve/checkpoint. No action or reset threshold is prescribed. No
per-update reminder, forced reset, or oracle correction is sent.

Binary and harness-source SHA-256 hashes, Higgs configuration, complete fixture,
native tool schemas, and full proxy catalog are identical across the two runs.
Wire audits confirm each selected policy, the common guide, exact submission
schema and fallback retrieval tools. Raw scoring is reconciled independently;
failed requests, action attempts, capacity metrics and server boot IDs are retained.
All twenty updates were generated; only five snapshots were submitted. The source
window-equivalent metric describes the planned stream, not achieved endurance.

- [Machine-readable results and parity checks](policy-summary.json)
- [Approved protocol](POLICY-DISCRIMINATOR.md)
- Raw optional evidence: endurance-policy-optional/B (gitignored)
- Raw decision evidence: endurance-policy-decision/B (gitignored)
- Each parent directory contains provenance.json and sampled telemetry.

Reproduce from the repository root in tmux, using a new output directory per run:

```sh
python3 -u experiments/context-recovery/endurance.py \
  experiments/context-recovery/endurance-policy-next-optional \
  --updates 20 --arms B --policy optional --ceiling 16384 --minutes 45
python3 -u experiments/context-recovery/endurance.py \
  experiments/context-recovery/endurance-policy-next-decision \
  --updates 20 --arms B --policy decision --ceiling 16384 --minutes 45
```

## Consequence

The user's instruction hypothesis has support: an explicit obligation causes
correct use of the inspection interface. Autonomous reset timing and long-session
endurance remain unestablished. The immediate next obstacle is that an informed
choice still needs a successful model request after obtaining headroom. Token
headroom also does not predict all memory/admission changes during a full prefill.
A follow-up should expose runtime capacity/cache state and test recovery at the
actual resume boundary; it must preserve the existing hardening and distinguish
model choice from a runtime-enforced reset policy.

No production implementation was changed. Final validation: release build passed;
268 agent-loop tests passed, 0 failed, 12 opt-in ignored; Python syntax and final
wire/fixture/configuration parity audits passed. This is an exploratory pair,
not proof of equivalence or a successful long-endurance run.
