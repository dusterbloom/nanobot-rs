# Explicit context-decision recovery

Status: release checks, live grammar enforcement and fresh-session recovery passed.
Matched autonomous arms completed measurement; neither completed the endurance target.

The recorded model output began `Context decision: checkpoint/reset.` but had no
notes or new_context calls. Nanobot's old recovery detector accepted this as final
prose. More fundamentally, Higgs's request type did not contain `tool_choice`, so
the client's existing `required` field was silently ignored.

The change reuses the existing recovery and tool-execution paths:

1. Match the explicit decision at the start of the current assistant response.
   Ordinary prose, quoted examples, continue decisions, and responses containing
   real calls are excluded.
2. Persist the announcement and a runtime execution instruction. Restrict the
   retry catalog to notes(write, content) and new_context, and validate returned
   calls against that restriction before execution.
3. Higgs's existing FSM masks tokens to a valid tagged JSON tool call. Required
   and named choices disable thinking; default auto keeps its existing path.
4. Constrained retries remove retained-cache controls from their request copy.
   Durable session state is preserved. Failed retries retract any streamed
   announcement and deliver an explicit failure response.
5. The experiment's reset tool requires a current-turn checkpoint write, file
   fsync, atomic rename and directory fsync. Failed writes revoke reset
   eligibility; reset requests remain idempotent and wait for successful batch
   receipts.

The notes/reset tools are still experiment tools. This change does not establish
that notes/reset is a better long-session strategy or that Escha chooses safe
reset timing. The model still authors checkpoint contents; grammar does not prove
those contents correct. A finite output limit can still truncate generation.

## Evidence

| Check | Result |
|---|---|
| Nanobot release suite | 3,002 passed across all targets, 31 ignored; release build passed |
| Higgs release suites | 1,528 passed, 35 ignored; release build passed |
| Live grammar, normal and streaming | Both returned the exact permitted notes call |
| Adversarial prompt: asks for plain text | Auto returned the sentinel; required returned the permitted call in both modes |
| Recorded announcement obligation | Actual notes write then new_context; both announcement and runtime instruction persisted |
| Fresh session after that boundary | Read unchanged notes, submitted exact snapshot once, no external action |
| Required-cache marker on recovery input | Removed from the recorded forced request; fresh recovery succeeded |
| Independent review | Two integration defects found, fixed and rechecked; no remaining concrete blocker |

The full suite initially exposed two old retained-route assertions. They expected
required retries to keep their retained-cache route, incompatible with actual
constrained decoding. Only those request expectations changed; original fallback
and stream-retraction assertions still pass.

## Frozen handoff comparison

One fixed revision-3 task, with the same correct checkpoint on disk, repeated twice
per condition. Fresh server/profile each time; order OFF/ON/OFF/ON. Original update,
prompt suffix and expected snapshot were identical. The handoff contains no facts.

| Handoff | Exact answers | Seconds | Mean seconds |
|---|---:|---|---:|
| Absent | 0/2 | 185.3, 139.9 | 162.6 |
| Read-notes instruction | 2/2 | 43.1, 44.0 | 43.5 |

Both OFF runs submitted the same stale inherited fields and did not read notes.
Both ON runs read notes and submitted the exact snapshot. This supports a concrete
instruction/interface failure in this fixture; it is not a broad model-quality
benchmark or a measured speed comparison against LCM.

Evidence: `announcement-evidence.json`, `grammar-evidence.json`,
`handoff-replay-evidence.json`, `harness-validation/`, and `fix-impact/`.
Raw SQLite receipts are preserved in the corresponding `endurance-*` directories.

Higgs local fix: `dd6730133` (local nightly includes it). Tested binary SHA-256
`fd9747c68f4f85770eb47d30cf7c077b9a014e559a71dc03e2b46380dcfa86fc`.
Nanobot release binary SHA-256
`f557bebfe98891c759d6e758114907efa7e24f41841371b425268aec62c62f0e`.

## Ordinary CLI speed check

Matched three-turn sessions used the old/new nanobot binaries against the same
patched Higgs build, with a fresh server/profile for each. All final replies and
prompt/completion token counts matched. Warm mean elapsed time was 1236.5ms before
and 1228.5ms after (-0.65%); cold calls were 9881ms/9965ms. This small sample shows
no meaningful ordinary-turn regression, not a proven speed improvement.

## Reproduction

Run long-lived commands in tmux, with Higgs serving the Escha fixture config.
Use fresh output paths; the tools refuse to overwrite results.

```sh
HIGGS_EVAL_URL=http://127.0.0.1:9000/v1 RECOVERY_ANNOUNCEMENT_OUT=/private/tmp/new-announcement-case target/release/deps/nanobot-67848bc0f9b02956 context_reset_announcement_recovery_live --ignored --nocapture --test-threads=1
python3 experiments/context-recovery/grammar-smoke.py /private/tmp/new-grammar-case
python3 experiments/context-recovery/replay.py /private/tmp/new-handoff-replay
```

## Matched autonomous endurance

Same frozen 20-update fixture, 12288 ceiling, actual 2048 output reservation,
decision policy and current binaries; independent fresh server/profile per arm.

| Measure | LCM A | Notes/reset B (LCM fallback retained) |
|---|---:|---:|
| Correct / submitted / planned | 6 / 6 / 20 | 6 / 6 / 20 |
| Elapsed seconds | 193.6 | 295.8 |
| Autonomous resets / applied handoffs | 0 / 0 | 1 / 1 |
| Completed LCM boundaries | 1 | 1 |
| Rejected duplicate attempts | 0 | 1 |
| Forced announcement recovery requests | 0 | 0 |
| Pending capacity turns | 1 | 1 |
| Model requests | 20 | 30 |
| Logical input / output tokens | 106713 / 1268 | 178683 / 2490 |
| Endurance pass | No | No |

Both measurements were valid, without watchdog termination or server restart.
LCM was faster in this matched run. Notes/reset recovered correctly after its
voluntary reset, but the earlier frozen handoff speed advantage does not establish
an endurance advantage. Neither exercised the announcement detector autonomously;
that behavior is established by the separate live announcement test.

B's duplicate was the revision-3 payload during turn tag 4, after successful
submission and the required context-status inspection. The original update and
success receipt remained visible; no compaction occurred in that turn. Revision 5
was also recorded successfully before the final turn suspended; revision 6 was
never delivered. Thus a pending turn is not necessarily an unrecorded update.

B's final failures were a rendered prompt of 10300 exceeding a requested 10240
limit, followed by two admission rejections after safe total capacity fell from
18432 to 8192. Published cause was memory pressure. Raw OS pressure stayed normal,
swap delta was zero, and compressor activity increased. There was no allocator
error, crash or restart. Peak sampled MLX active allocation was 14,834,099,796 bytes.
Process footprint, system free RAM and compressor resident bytes were not captured:
these results do not establish physical RAM exhaustion.

Evidence: `autonomous-announcement-evidence.json`, `lcm-announcement-evidence.json`;
raw replay/telemetry in `endurance-announcement-autonomous/` and
`endurance-announcement-lcm/`. The remaining discriminator is longer completed
coverage under measured capacity, plus completion handling after status inspection.

## Overflow retry defect discovered by the audit

The B replay exposed a separate deterministic defect: after the 10300 > 10240
rejection, emergency trimming reversed the protected current-turn suffix. The
original request ended user → notes call → receipt → submit call → success receipt;
the retry contained that suffix in reverse order. Thus the success bytes survived
but their causal order did not. This did not cause the earlier revision-3 duplicate.

`keep_recent_within_budget` seeded a backward accumulator with a forward-ordered
protected suffix, then reversed the entire accumulator. The correction seeds it in
reverse order so the final reversal restores chronology. A regression exercises
the public trimming path with older history, an oversized message and two current
call/receipt pairs. Pre-fix endurance scores above remain pre-fix evidence.

Overflow-order verification: the new regression failed on the old code with the
exact reversed sequence, then passed with the correction. Full nanobot release
suite: 3003 passed, 0 failed, 31 ignored; release build passed. Independent review
confirmed the reverse-walk correction. Graph analysis was complete and LOW risk.

Correction installed as nanobot `7b8b24a`; final binary/runtime hashes are in
`BINARY-PROVENANCE-AFTER.json`. The matched ordinary CLI check returned identical
replies/token counts, warm means 1221ms before and 1220ms after. One initial
after-build model load was rejected by critical-pressure policy; a single fresh
retry passed without relaxing policy. No endurance rerun on this final correction
is claimed. Installed Higgs defaults were restored and READY smoke passed.
