# Context-recovery validation: response budget and handoff

This comparison uses frozen project deltas and independent Higgs profiles per arm. A exposes LCM recovery; B additionally exposes voluntary notes/reset, while retaining the same automatic LCM fallback. Thus B is a hybrid, not a compaction-disabled mode. Every run plans20 updates.

| Response-reservation condition | Arm | Correct / submitted | Voluntary resets | Completed LCM boundaries | Seconds | Outcome |
|---|---|---:|---:|---:|---:|---|
| Default long-form6144 | A | 4 / 4 | 0 | 1 | 167.2 | capacity suspension |
| Default long-form6144 | B | 5 / 5 | 0 | 1 | 236.7 | capacity suspension |
| Long-form2048 | A | 4 / 4 | 0 | 1 | 243.6 | capacity suspension |
| Long-form2048 | B | 3 / 20 | 5 | 3 | 1150.2 | incorrect snapshots |

Default-condition wire reservations included6144 and2048. The new condition independently verifies that **all main-request reservations were2048** in both arms. It does not change production defaults, the task input, reset-choice policy or model precision. The2,048 setting is exposed only in the ignored test harness via `ENDURANCE_LONG_FORM_MIN_TOKENS`.

The2048 A arm remained correct through four snapshots, then suspended with one durable pending input. Higgs downshifted six times, reaching safe total6144 / published prompt2048. Its result does not establish that smaller response reservation alone prevents capacity suspension.

The2048 B arm completed the planned stream and crossed repeated boundaries, but only its first three snapshots were correct. At revision2 it wrote a correct checkpoint and voluntarily reset. The fresh revision3 request contained the intact current delta, but no earlier snapshot or checkpoint pointer. The model did not read notes/history before submitting; it filled inherited fields from stale distractors. Subsequent checkpoints propagated the wrong state. Full evidence and plausible interpretations: [LCM-NOTES-CHOICE-AUDIT.md](LCM-NOTES-CHOICE-AUDIT.md).

Completed B also had one duplicate submission attempt, one tool error, one recall call, five voluntary resets and three later LCM boundaries. The first failure at revision3 preceded those later LCM boundaries. No forbidden external action, wire-schema error, telemetry error, watchdog termination or unexpected server restart occurred. Test-process exit0 means artifacts were recorded, not workload correctness.

LCM terminology: the earlier verified LCM checkpoint was level3 deterministic truncation because a model summarization request would not fit. Historical `compaction_attempts` counts only journaled LLM summarizer requests; it is renamed to `llm_compaction_requests` in future summaries. Durable completed LCM counts remain separate. Do not infer “no compaction” from a zero model-request count.

The next test adds a one-time, pointer-only recovery instruction after a completed-turn reset. It injects no saved facts and does not prescribe when to reset. This isolates a missing handoff from checkpoint storage and autonomous reset timing. Results will be added after the run.

Host pressure and capacity histories differed between arms and cohorts. These runs establish observed behavior and failure chains, not a clean speed ranking or a causal estimate of the response reservation alone. Raw2048 files are in `endurance-budget2048-isolated/`; compact summaries/provenance are committed in `budget2048-evidence.json`.

## Autonomous run with handoff enabled: not exercised

The handoff-enabled B run submitted5/5 correct snapshots, then suspended on capacity. It executed zero voluntary resets and delivered zero handoff markers. At revisions2 and4 it announced checkpoint/reset in final text without calling notes/new_context; revision3 then claimed a reset had already happened. Actual receipts show one session and two LCM boundaries. No conclusion about handoff effectiveness follows from these correct snapshots.

Raw: `endurance-reset-handoff/`; compact: `reset-handoff-evidence.json`. All wire output reservations remained2048. There were no duplicate submissions, forbidden actions, tool errors, wire errors or unexpected restarts; one pending capacity turn remained.

Next bounded discriminator replays the recorded first-reset state through the same agent loop: fresh session, correct model-written revision2 checkpoint on disk, exact frozen revision3 delta, with/without the pointer. This deliberately controls the recovery boundary and therefore tests recovery ability, not whether the model chooses to reset.

## Grammar recovery and frozen handoff discriminator

The required-tool field was previously ignored by Higgs. The patched decoder now
enforces a valid tool envelope; an adversarial prompt differentiates auto text
from required calls in both response modes. Nanobot's recorded reset announcement
now produces a durable checkpoint/reset boundary and an exact recovery in a distinct
fresh session. See `ANNOUNCEMENT-RESULTS.md` and its raw evidence links.

Frozen revision-3 replay, OFF/ON/OFF/ON: no handoff 0/2 correct (185.3s,139.9s),
pointer-only handoff 2/2 correct (43.1s,44.0s). No facts were added by the pointer.
This tests recovery after reset, not autonomous timing or superiority over LCM.
