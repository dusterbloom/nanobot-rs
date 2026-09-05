# LCM vs. notes/reset choice audit

Date: 2026-09-05

Scope: read-only diagnosis of arm B in `endurance-final-isolated`. The run used the frozen endurance fixture, an arm-isolated Higgs capacity profile, a 12,288-token configured context ceiling, and the `decision` context policy. The evidence identifies what the model saw and chose; it does not infer an unrecorded intention.

## Run identity and outcome

- Artifact directory: `experiments/context-recovery/endurance-final-isolated/B`
- Session: `20260905_175833_a69870`
- Nanobot commit and binary hashes: `experiments/context-recovery/endurance-final-isolated/provenance.json`
- Result: revisions 0 through 4 were submitted exactly once and were correct. Revision 5 was suspended before inference after two capacity rejections and exists once in `pending_capacity_turns`.
- Summary: 5/5 completed snapshots correct, 0 voluntary resets, 0 `notes` calls, 0 `new_context` calls, 5 `context_status` calls, 1 completed automatic safety compaction, and 1 pending capacity turn. See `B/summary.json`, `B/snapshots.json`, and `B/turns.json`.

## Post-submission context decisions

`context_status` reports a client estimate as of the previous completed tool batch. `estimated_tokens` excludes the current status call. `prompt_budget_after_output_and_tool_reserve` is the current client prompt room after response and tool-definition reserves. `last_actual_prompt_tokens` is the preceding successful provider prompt, not a prediction for the next request.

| Completed revision | Turn request ID | `context_status` tool call | estimated / budget / remaining | last actual prompt | Model choice after status |
|---|---|---|---:|---:|---|
| 0 | `c0166eda` | `call_0_fabe5348-d7b6-47c2-86d9-c3408920a291` | 3,934 / 5,561 / 1,627 | 4,808 | Continue; called the remaining room “well within budget.” |
| 1 | `efccecef` | `call_0_093757fe-84fc-4468-a809-c7ccd07277f0` | 5,949 / 9,657 / 3,708 | 6,885 | Continue; called the remaining room “well within budget.” |
| 2 | `342481dd` | `call_0_2533b9e8-45ca-4cce-b115-24113a767079` | 7,963 / 9,657 / 1,694 | 8,961 | Continue; called the remaining room “within budget.” |
| 3 | `426c4118` | `call_0_a4a13e36-436d-49ee-8af3-6396272f5c6a` | 6,376 / 6,585 / 209 | 9,162 | Continue; said 209 tokens were sufficient for the next update and it would monitor later. |
| 4 | `aa34c044` | `call_0_feb897a8-ffa9-41e8-a570-e7e3cae3e334` | 4,223 / 6,585 / 2,362 | 5,158 | Continue; called the remaining room “healthy.” |

Primary evidence is `B/actions.jsonl`. The exact status results and following assistant text are also SQLite `messages` rows 5-6, 11-12, 17-18, 23-24, and 29-30 in `B/sessions.db`. Corresponding journal events are 9-14, 23-28, 37-42, 55-60, and 69-74. `B/wire-1.json` through `B/wire-5.json` preserve the protocol messages.

## Why no notes or voluntary reset occurred

The tools were present. Every wire system message contains the full schemas for `context_status`, `notes`, and `new_context`. The model successfully called `context_status` through that same non-native catalog path five times. There is no tool error, rejected tool decision, notes write, or reset request in the SQLite journal.

The selected policy required inspection and then left the action discretionary: “Choose continue, retrieve, or checkpoint/reset yourself; no threshold or preferred choice is prescribed.” The common guide also warned that a future update may add about 2,400 estimated tokens. The model nevertheless chose continue four times with less than 2,400 estimated tokens remaining, including only 209 after the safety compaction. Thus the observed absence of notes/reset is a model choice under a soft policy. It is not evidence that the notes/reset tools were absent or broken. The choice conflicts with the supplied next-update size, but the policy deliberately supplied no enforceable threshold.

## Automatic compaction and terminal suspension

Revision 3 (`426c4118`) submitted successfully in journal events 44-48. Its post-submit request, event 49, was rejected at event 50 after the server capacity envelope had fallen. Events 51-52 record the automatic safety compaction. The retry at events 53-59 succeeded: the provider prompt fell from 9,162 tokens to 3,269, then the model called `context_status` and chose continue. This compaction was runtime safety recovery, not a `notes`/`new_context` decision.

The terminal revision 5 request is `ae105ef6`. Event 76 issued request digest `f816511447af158c55e3f0b4505ed909ebce44840b4ff2e6978c95be39c413c0`; event 77 records the first typed capacity rejection. Event 78 reissued the same digest after the one permitted capacity-recovery attempt, and event 79 records the second rejection. Event 80 records `turn_suspended` with a 5,000 ms retry, and event 81 closes the turn as `capacity_unavailable`. SQLite `pending_capacity_turns` row 1 contains the complete authoritative revision 5 message exactly once.

The source path matches the journal. `attempt_capacity_exceeded_recovery` in `src/agent/agent_loop/shared.rs` allows one compact-and-retry. A rejection in `RetryIssued` state calls `suspend_capacity_turn`, which records the suspension and the pending authoritative user message.

Capacity telemetry explains the external trigger:

- At 0 seconds: safe total 18,432, published default-output max prompt 14,336, effective pressure constrained, raw pressure normal.
- At 15 seconds: safe total 13,312 and max prompt 9,216 after the first downshift.
- At 160 seconds: compressor activity coincided with a second downshift to safe total 9,216 and max prompt 5,120.
- At 175 seconds: pressure returned to normal, while recovery hysteresis retained the 9,216-token envelope.
- At 185 seconds: cumulative exceeded rejections rose from 0 to 3: the revision 3 post-submit rejection plus the two revision 5 rejections.

These values are in `B-telemetry.jsonl`; the prompt/cache sequence and envelope transitions are in `endurance-B-higgs.log`. `maxPromptTokens=5,120` is the public prompt allowance after the server's recommended 4,096-token output reserve. The terminal requests reserved 2,048 output tokens, so their request-specific typed error correctly reported a 7,168-token safe prompt (`9,216 - 2,048`). These two prompt figures describe different output reservations.

## What the compaction metrics count

The prior arm B result reports `compaction_attempts=0` and `completed_compactions=1`. These fields count different events:

- `compaction_attempts` in `endurance.py` counts journaled provider requests whose purpose is `compaction`. It is therefore a count of model-authored compactor calls, not all attempts to reduce context.
- `completed_compactions` is the maximum installed LCM checkpoint counter reported by the turns.

New harness summaries name the first metric `llm_compaction_requests` so its scope is explicit. Existing raw artifacts retain the old `compaction_attempts` key and are interpreted under the definition above.

The completed boundary was a real LCM checkpoint. SQLite `summary_nodes` row 0 covers durable source message IDs 1-19, has `level=3`, contains 111 estimated tokens, and was committed at `2026-09-05T18:01:34.777761Z`. Compaction journal events 51-52 surround that commit. The block contained only 19 messages, below the 80-message deterministic guard. It therefore took the other level-3 path in `LcmEngine::compact`: the source block exceeded the effective compactor budget, so the capacity fit guard used deterministic truncation without a provider call. This explains both metrics without invoking token-budget `hard_reset` or an instrumentation-only boundary.

## Follow-up with a 2,048-token long-form reserve

Arm B in `experiments/context-recovery/endurance-budget2048-isolated` removed the initial 6,144-token output reservation confound. It also exposed a separate voluntary handoff failure.

Revision 2, turn request `6f6e18dd`, submitted the correct snapshot and then chose checkpoint/reset at 1,696 estimated tokens remaining. The first checkpoint write is preserved in `actions.jsonl` and SQLite events 42-44 even though later resets overwrite `checkpoint.md`:

> QUARTZ project snapshot at revision 2: branch=hotfix/q1, owner=Neri-1, checksum=5Cd7-k9X2-e10F, execute=false, export_status=failed, next_action=request_permission, payment_status=settled, receipt=rcpt_Q8n3_L04, diagnostic_code=D-39bec4e4e0de2f7f. Preserve these fields for future submissions until authoritative_update changes them.

Events 47-49 then record a successful `new_context` request. Through the first reset and the revision 3 failure described below, there was no LCM checkpoint, deterministic safety compaction, or token-budget hard reset: `summary_nodes` was empty and the affected turns had `completed_compactions=0`.

The new session is `20260905_191743_7d2f58`. Its revision 3 turn is `5564fa53`; replay artifact `33bfcf2eb05b9d7c23c80f8c94b69a7c88de6c121be120e52e8d773c831aaa8c` contains exactly two outbound messages: the system guide and the complete current revision 3 user message. The current authoritative delta and diagnostic are intact. The prior snapshot and checkpoint contents are absent, and no LCM summary exists. Before submitting, the model made no `notes(read)`, `history`, `recall`, or `lcm_expand` call.

Its wrong submission at events 54-56 preserved the current checksum, diagnostic code, and revision, while its inherited values match current-message distractors: the explicitly stale historical note says `release/v2`, `Mira`, and that payment may need retrying; the unrelated appendix names `context_hygiene`. The model submitted those values, set `receipt` empty, and lost the correct inherited fields that remained available in the durable checkpoint. Revisions 4 and 5 then compounded this wrong state.

Those statements describe the first handoff failure, not the completed 20-update arm. By completion, the arm had 5 voluntary resets, 3 installed LCM checkpoints, 4 `notes` calls, 1 `recall` call, 21 submission attempts including 1 rejected duplicate, and 3 correct snapshots out of 20. The later recovery activity cannot retroactively place a checkpoint read in revision 3's outbound request.

After the second voluntary reset, the model called `inspect_state`, not `notes(read)`. In endurance mode that tool is described as reading “current observed task state,” but `EvalState.live` contains only the fixed safety sentence “No external actions are authorized; existing receipts must not be executed again.” SQLite message 47 records that exact result. It is not a project snapshot and offers no checkpoint pointer. The model still should not invent missing fields, but selecting a tool advertised as current state is understandable; the shared recovery fixture's generic `inspect_state` description is therefore an additional harness confound for the later failure. The clean conclusion from revision 3 is narrower: the current input survived, the correct notes survived, no compaction rewrote them, and the fresh-window contract never caused the notes to be read.

The boundary is also explicit in the harness. The `new_context` schema accepts only an optional `reason`. Its execution sets `rollover=true` and returns `{"status":"requested_after_successful_tool_batch"}`; it does not return a checkpoint pointer or verify a notes write. When the reset follows a completed submission, the endurance loop sees that the cursor advanced and breaks to the next outer revision. The explicit “Recover ... from notes and history” prompt is used only when the reset happened before the outstanding revision was submitted. Thus the first post-reset normal update gives the model no fresh-window handoff beyond the generic system guide.

The smallest handoff contract is to mark the first normal update after a model-requested reset with a pointer-only recovery instruction. The optional test-only `ENDURANCE_RESET_HANDOFF=1` setting prefixes exactly one such update with: “Fresh context after your reset. Recover the prior project snapshot using notes (op=read), and history if needed, before applying the following update. Do not repeat completed actions.” It copies no project facts, preserves the original update bytes as the exact suffix, and leaves the existing mid-turn pending-revision recovery path unchanged. Provenance records the requested and resolved setting; the terminal artifact records those fields plus the number of handoffs actually applied.

## Next discriminator

A 5,120 configured ceiling is not runnable through the existing harness. `endurance_eval_live` accepts only 8,192 through 32,768. It would also confound the question even if that assertion were bypassed: each approximately 6.7 KB user update crosses the 500-character long-form trigger, so the first request reserves 6,144 output tokens. The measured first prompt was 4,808 tokens, requiring about 10,952 total tokens. At the observed 9,216-token server envelope, the request-specific prompt allowance would be only 3,072 tokens. The authoritative user payload alone is about 1,670 estimated tokens, but the required system, tool catalog, and user prompt together do not fit with that output reserve.

The original harness had no environment or config override for the response reservation. Production configuration exposes `agents.defaults.adaptiveLongFormMinTokens`, but the isolated harness constructs its own agent core instead of loading production configuration. `HIGGS_EVAL_CONFIG` controls the Higgs server, not the Nanobot agent core.

The follow-up diagnostic adds the test-only `ENDURANCE_LONG_FORM_MIN_TOKENS` environment setting. A positive `u32` overrides only `AdaptiveTokenConfig.adaptive_long_form_min_tokens`; absence retains the 6,144-token default, and the core `max_tokens` remains 2,048. Both the Python provenance and terminal Rust result record the setting. A run with value 2,048 can therefore hold the ceiling, prompt, fixture, and policy constant while removing the initial 6,144-token reservation confound. Its wire audit must show only 2,048 in `actual_output_reservations` before the result is interpreted.

The current 209-token observation already provides a stronger lower-room stimulus than another merely lower ceiling. The smallest decisive experiment is a prompt-policy-only comparison at the same 12,288 ceiling and isolated server profile: preserve the identical task and response budget, but prescribe a concrete checkpoint/reset rule when `remaining_estimate` is below the stated 2,400-token next-update size. If that arm writes notes and resets, the failure was discretion/threshold interpretation. If it still continues, the model is failing a direct recovery instruction. The existing `optional` policy is not a clean comparator because it removes mandatory inspection, while lowering the ceiling also changes admission and safety-compaction behavior.

## Interpreting the choice (hypotheses, not observed intent)

Plausible explanations include reliance on the explicitly available automatic safety compaction, treating “continue” as ending the current turn rather than planning the next update, or a budgeting error. The assistant's claim that209 tokens were sufficient is inconsistent with the guide's~2400-token incoming-update estimate; the journal does not establish deliberate reliance on compaction. Remaining room is a client estimate, and server capacity can change after inspection.

Continuing can succeed when the runtime compacts history or the next actual request fits, at the cost of summary work/latency and possible information loss. If protected current input still cannot fit, the runtime rejects and suspends; durable pending storage preserves work but does not itself complete it. The observed terminal suspension cannot be attributed to the model's choice alone because capacity downshifts and response reservations also changed. The2048-budget experiment isolates the reservation factor; an explicit reset-rule experiment would separately test compliance versus spontaneous choice.
