# OpenCode review record — 2026-09-06

Requested and verified provider/model: `zai-coding-plan/glm-5.3-flash`.
OpenCode session: `ses_f8c0ac0beffeagHmsxev78FrD6`; tmux: `opencode-endurance-review`; exit: 0.

## Reconciliation by primary agent

- Agree on 13/20 for the original full scheduled run, no superseding full run, and the targeted probe counts. Independently rebuilt expected state from input deltas and diagnostic lines and matched action receipts: `score-review-independent.json`.
- Correct the review's failure split: revision 10 also has an empty checksum inside the extra wrapper. All seven failures involve checksum values; one additionally has malformed nesting.
- Agree that the original prompt already required preserving unchanged fields. The new explicit opaque-identifier sentence strengthens instruction robustness; it does not introduce permission to change prior values or new task facts.
- Verified and clarified run-time HEAD versus source committed later as fdb4ebf; source hashes match. Annotated delivery counters that miss direct updates. Fixed tracker table formatting.
- Keep exact full-snapshot correctness as the primary endpoint; checksum and shape are secondary diagnostics. A tiny baseline/treatment comparison is evidence for this fixture, not proof of general reliability or superiority.
- Reviewer inspected local benchmark prose, not the external benchmark implementations; compatibility remains unvalidated. The review's dd6730133 identifier is the Higgs server build, not a model-artifact hash.

## Original reviewer output (preserved verbatim)

# Independent review: scheduled endurance 13/20 claim

## Verdict up front

**The 13/20 statement is factually correct and adequately qualified.** I recomputed the score from raw `session_replay_artifacts` in `endurance-feasibility-scheduled/B/sessions.db` without trusting `snapshots.json`, the terminal artifact, or any prose: 20 submissions, **13 exact matches, failures at revisions 8, 9, 10, 14, 16, 17, 18** — byte-identical to `snapshots.json`, `summary.json` (`correct_updates: 13`, `first_wrong_revision: 8`), and FEASIBILITY-RESULTS.md. No newer evidence supersedes it: the only later work (checksum probes, 00:53–01:08) explains the failures; no later 20-update run exists. It remains the only completed-run score under this policy.

## Independently recomputed counts

| Claim | Source | Recomputed | Match |
|---|---|---|---|
| 13/20 exact, wrong at 8,9,10,14,16,17,18 | FEASIBILITY-RESULTS.md:3,31 | raw DB replay: 13/20, same revisions | ✅ |
| snapshots↔fixture reconciliation | endurance.py:102-106 | `pass == (actual == expected[i])` holds for all 20 | ✅ |
| 20 submitted / 20 resets / 19 handoffs | FEASIBILITY-RESULTS.md:9 | 20 ok `submit_result` executes, 20 `reset_events` (rev 1–20), `reset_handoff_count: 19` | ✅ |
| Notes writes/reads 20/19 | FEASIBILITY-RESULTS.md:10 | `control_tool_calls.notes = 39` | ✅ |
| 79 requests / 10,809 output tokens / 985.3 s / peak prompt 5445 | FEASIBILITY-RESULTS.md:11,15-16 | summary.json identical | ✅ |
| Footprints 17.33 / 13.69 / 2.25 GiB, 197 samples, 0 errors | FEASIBILITY-RESULTS.md:19-22 | raw bytes /2³⁰: 17.33, 13.69, 2.253 | ✅ |
| Cold probes 2/4, 3/4, 4/4, 4/4; retained 0/2 vs 2/2 | CHECKSUM-RESULTS.md:9-14,24-25 | re-tallied all 16 cold + 4 retained `exact` flags | ✅ |
| Correct checksum visible before rev-8 submission | FEASIBILITY-RESULTS.md:29 | `feasibility-first-error.json`: request event 178 contains `7aF07-bC9x-00Q`, not the fabricated value; receipt shows it | ✅ |
| Release suite 3004/0/31 | FEASIBILITY-RESULTS.md:43 | summed all `test result` lines in feasibility-test.log: 3004/0/31 | ✅ |

## Failure classification (per-revision diffs, actual vs expected)

- **Semantic — fabricated/recomputed opaque checksum (6):** rev 8 `8bR29-mD7y-01P`, rev 14 `9mK22-dE8y-11R`, rev 16 `7aF16-bC9x-01Q` (near-miss recompute vs expected `7aF15-bC9x-00Q`), rev 17/18 `8bG17/8bG18-dD0y-02R`, rev 9 **empty string**. Inventions correlate with revision number — consistent with CHECKSUM-RESULTS.md:3's "treats checksum as something to recompute."
- **Malformed tool output (1):** rev 10 — entire snapshot nested under an extra `result` key; `submit` (endurance_eval.rs:83-93) records arbitrary JSON and gives no repair feedback. Correctly separated from the checksum errors.
- **Execution feasibility:** PASS, cleanly separated — 0 capacity rejections, 0 watchdog/restarts, `valid_measurement: true`, grader returns only `{"recorded":true}` (endurance_eval.rs:91-92), no oracle leakage.
- **Instruction ambiguity — partially mislabeled:** the scheduled prompt already says "preserve unchanged fields" and "Never invent missing fields" (wire-8.json system prompt). What was missing is emphasis/opaque-identifier wording, and the model-authored checkpoint placed the checksum outside its explicit `Snapshot submitted:` field list (feasibility-first-error.json receipt). The probes show stronger *wording* flips outcomes (4/4, 2/2) — that's instruction **sufficiency/robustness**, not ambiguity of an unstated rule. Minor but worth precise language.

## Findings (severity-ordered)

1. **LOW — provenance vs claimed commit:** `provenance.json` records `nanobot_head = 0be0e65` but FEASIBILITY-RESULTS.md:43 claims harness `fdb4ebf`; the commit landed 8 min *after* run start (00:26 vs 00:34). Mitigated: recorded SHA-256s of `endurance_eval.rs` and `endurance.py` match the fdb4ebf-committed files exactly, so content identity holds. Docs should cite the head at run time or say "uncommitted tree later committed as fdb4ebf."
2. **LOW — dead metrics in scheduled summary:** `delivered_update_bytes: 0` / `..._window_equivalents: 0.0` (endurance.py:84-86,126-128 count `next_task` tool deliveries; the scheduled driver delivers updates directly). Not quoted in docs; note it before anyone reads those fields as meaningful.
3. **INFO — VALIDATION-TRACKER.md:20** places the scheduled-control row after the table instead of inside it. Cosmetic.
4. **INFO — external benchmark proposals:** Harbor appears only in SCORE-REVIEW-PLAN.md:14, correctly gated ("verify adapter and ARM64 environments with an oracle smoke check before freezing the task set") and claims nothing. Terminal-Bench and EvalScope appear **nowhere in the repo** — they exist only in the reviewer's chat and cannot be assessed against evidence here. BFCL numbers in `docs/plans/local-model-matrix.md:28,53` and `three-impossible-things.md:269` are uncited vendor-claimed model scores used for model *selection* — acceptable as plans, but they must never be quoted as nanobot evidence. The refreshed README (f098fd2) is accurate and properly hedged: "Superiority is a hypothesis we are testing, not a benchmark result" (README.md:7), eval tools flagged non-production (README.md:27). No repo doc makes a blanket performance claim — the tracker explicitly withholds the autonomous rerun (VALIDATION-TRACKER.md:71) and refuses cross-policy comparison (FEASIBILITY-RESULTS.md:41).

## Bounded next experiment

SCORE-REVIEW-PLAN.md:10-12's design is sound; sharpen it minimally:

- **Arm T:** scheduled policy + only the byte-for-byte opaque-identifier sentence; **Arm B′:** exact baseline rerun. Alternate order, 2 runs total minimum (4 if both directions must be shown).
- Primary endpoint: checksum-field correctness across 20; secondary: shape validity (catches a rev-10 recurrence) — exactly as proposed.
- Justification for the baseline repeat: rev-14's original request **passed** cold but failed retained, and neutral-penalty flipped it — temperature-0 + repeat-penalty 1.1 still yields decision instability, so one treatment run cannot attribute a 20/20 to the instruction.
- Ceiling: retained probes went 2/2 with the instruction on the two worst fixtures; if treatment also scores 20/20 while baseline repeats ≤13/20, instruction sufficiency is established for this fixture; generalization still requires the held-out Harbor track.

No blanket claims: everything above is one run on one machine, one model build (`dd6730133`), one fixture, n=1 per condition except where probes say otherwise.
