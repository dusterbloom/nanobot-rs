# Escha context recovery experiment

Goal: compare current nanobot LCM recovery with explicit checkpoint + fresh context,
using the real nanobot loop and hardened Higgs branch 327e5021e, under tmux.

## Design and scope
- Five synthetic, objectively scored tasks: changed requirement, failed action,
  completed action, buried exact identifier, and superseded project state.
- Common seeded SQLite transcript per pair, including real repository code as
  distractor material. No personal session database is read or written.
- A: invoke production LCM compaction/publication, restart loop, resume using
  normal LCM/recall retrieval. Require a durable summary checkpoint.
- B: same original transcript; Escha writes notes and requests new_context.
  Harness applies fresh-session boundary after completed successful tool batch;
  restart loop, recover using notes and paged history. This is a test-only
  approximation of the proposed interface, not a shipping rollover feature.
- C: two unexpected-reset cases without checkpoint notes.
- Identical model, temperature, prompt ceiling, output/iteration budget and task
  interfaces across conditions; recovery interfaces are the treatment.
- Exact JSON artifact scoring, tool-action log, replay availability, model usage,
  preparation + recovery wall time. Report checkpoint costs separately and total.
- Forced boundaries test recovery, not autonomous timing or long-session quality.
- No production behavior changes, external actions, cloud calls, commits or pushes.

## Execution
- [x] Inspect running service, local/fork revisions, API and test integration.
- [x] Attempt GitNexus impact: UNKNOWN, stale index; confirm CLI callers by source.
  Only a cfg(test) module declaration will touch existing code.
- [x] Build hardened Higgs 327e5021e and nanobot fa53da4 in tmux; verify model identity.
- [x] Add test-only harness and score/reader negative checks.
- [x] Run five paired trials, plus two unexpected resets if B is viable.
- [x] Verify raw logs, durable checkpoints, results and replay evidence.
- [x] Write results and limitations; preserve logs and keep Higgs inspectable in tmux.

## Pilot invalidation and corrected run
The initial trials under `/private/tmp/recovery-eval-results` are invalid for
model attribution: local lazy persona loading omitted GUIDE, the replacement
registry contradicted production tool advertisements, reset happened after the
whole turn, and a preparation result could satisfy recovery's raw scorer.
Preserve those logs as harness-debugging evidence only.

Corrected harness explicitly installs GUIDE plus full available schemas and a
proxy-call example before first send. A test-only loop hook ends a requested
reset after every tool call has successful durable execution/postprocessing.
Negative checks reject incomplete/failed batches. Preparation artifacts are
quarantined; passing requires zero preparation submissions, exactly one fresh
submission, exact JSON, no prohibited actions, and Finished outcome.

The LCM arm still exercises actual restart behavior: restored summary nodes in
`LcmEngine::rebuild` currently set `created_at_turn: 0`, disabling the freshness
cooldown. Do not silently repair this while measuring the existing baseline.
Synthetic seed turns now determine the compaction turn instead of hardcoded 1.
The experiment remains a bounded recovery probe, not evidence about autonomous
long-session timing or a production implementation of posthorse.

- [x] Corrected harness compiles in release; score/page and failed/incomplete
  reset-batch checks pass. Existing compaction regressions: 52 pass, 1 ignored.
- [x] First corrected changed-requirement B trial passes exact recovery scoring;
  all nine wire requests contain guide, schema enum and proxy example. No
  malformed tool calls; old window stops at reset batch. Preparation 48.8s,
  recovery 32.9s. Timing cannot be compared causally with invalid pilots because
  server cache settings also changed.
- [x] Preserve raw corrected artifacts in `experiments/context-recovery/results/`
  (gitignored), not solely temporary directories. Driver resumes completed trials
  and audits wire requests and reset ordering after every new trial.

## Additional validity corrections
- `failed_action.operation` was underspecified: the scorer demanded EX-7041,
  while the instruction allowed a reasonable reading as an action name. Make
  the ID requirement explicit before scoring it. Preserve the old interrupted
  run under `interrupted/`; it is not a model capability failure.
- The continuous-server series showed capacity declining from 182272 total
  tokens to 3072 across intermittent compressor-pressure episodes; nanobot
  suspended with CapacityUnavailable and saved pending work. At inspection,
  Higgs reported normal pressure, zero active reservations, zero current swap/
  compression deltas, and zero prompt headroom. The next B trial could not start.
  This is separate endurance evidence, not grounds to change recovery scores.
- Remaining trials use a fresh same-config Higgs boot with >=16384 advertised
  prompt tokens before each trial. Record per-trial boot logs and envelopes;
  exclude server startup from task recovery timing. The first completed pair
  remains valid behavioral evidence but timings across the suite are exploratory.
- Existing model paths, temperature, native execution, and actual production
  compaction/replay implementation are unchanged.

## Scoring interpretation
Retain predeclared exact-JSON scores without rewriting failures into passes.
However, the fixture never requires literal status codes: e.g. `Failed (ERROR
permission denied)` truthfully reports failure. Report a separate manual factual
assessment across ALL artifacts, distinguishing semantic recovery from unspecified
status formatting. Do not call a free-form status spelling a lost-context error.
Future automated regression fixtures should declare status enums before inference.

Final verification: release build passed; 267 agent-loop regressions passed, 11 ignored. All 12 live trials separately ran and passed wire/reset/scoring audits. No production behavior changed; no commits or pushes.
