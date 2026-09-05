# Autonomous choice and bounded endurance — 2026-09-05

Both valid arms produced five exact snapshots, then stopped because Higgs capacity
became unavailable. Both saved pending work. Notes/reset was available but Escha
never chose context inspection, notes, history or reset. This run does not show
that autonomous resets are faster or that either approach sustains long sessions.

| Outcome | A: existing LCM + retrieval | B: same + notes/reset |
|---|---:|---:|
| Requested updates | 20 | 20 |
| Exact snapshots submitted | 5/20 | 5/20 |
| Wrong submitted snapshots | 0 | 0 |
| Normally finished user turns | 5 | 4 |
| Time to interruption | 296.6 s | 266.2 s |
| Voluntary resets | unavailable | 0 |
| Context inspection / notes calls | 0 / unavailable | 0 / 0 |
| Completed automatic compactions | 1 | 0 |
| Model requests | 11 | 10 |
| Tool errors / duplicate submissions / forbidden actions | 0 / 0 / 0 | 0 / 0 / 0 |
| Peak actual prompt | 13,047 tokens | 13,007 tokens |
| Minimum server prompt capacity | 0 | 0 |
| Pending capacity turns in SQLite | 1 | 1 |
| In-arm server restarts | 0 | 0 |
| Three-boundary endurance coverage | No | No |

B submitted its fifth correct snapshot before the continuation was interrupted;
that user turn did not finish normally. A suspended on the next update. The
shorter B duration is therefore not a speed win. The arms also encountered
different system memory-pressure histories. One pair cannot establish a reliable
latency or accuracy ranking.

The three-update plumbing canary passed 3/3 exact snapshots, with zero tool
errors and no reset. This is interface/plumbing evidence, not endurance evidence.

## What changed in the test

Updates are actual successive user turns through the existing agent loop, sharing
one core and session until an agent-selected reset. Every update changes some
fields while retaining original exact identifiers and action receipts. Each has
a fresh required diagnostic code buried in a large appendix. Every submitted
snapshot is scored exactly, independently reconciled against the fixture, without
revealing oracle feedback to the model. Duplicate and prohibited tool attempts
are counted even when the runtime blocks them.

The initial system instructions explain the schemas and tools and explicitly
leave checkpoint/reset timing to the agent. There are no per-turn reset reminders
or forced checkpoint turns. Both arms retain recall/lcm_expand and the ordinary
production compaction/capacity paths. B adds notes, paged history and reset. Context
telemetry identifies its previous-batch observation and accounts for adaptive
output reservation. Recorded request audits check the instructions and retrieval
catalog. Tool results survive in SQLite across reset windows.

Each arm uses a fresh hardened Higgs boot, followed by continuous service for the
whole arm; there are no hidden in-arm reboots. Metrics sample capacity, memory,
cache and boot identity. The attempt stops on its first runtime interruption and
checks durable pending state. It does not run the gateway's automatic resume
scheduler, so preservation is verified but eventual autonomous resumption is not.

## Invalid attempts and configuration findings

Preserved raw artifacts are excluded from the capability table:

- Initial telemetry/schema lock reentrancy, missing tool-result reader, and
  single-turn polling interacting with duplicate-call replay were harness bugs.
  The corrected test uses immutable tool metadata and real user turns.
- An artificial 8K window left only 2K prompt space after adaptive output reserved
  6144 tokens; the first 4032-token prompt was rejected. Scored runs use 16K.
- The original isolated server output ceiling of 4096 rejected adaptive 6144-token
  requests. Both scored arms use higgs-endurance.toml with ceiling 8192. This
  changes only the isolated test configuration; production adaptive budgeting
  remains enabled.
- One boot rejected model allocation under critical system memory pressure.
- The first main B attempt lacked lcm_expand despite fallback summaries naming it.
  Its post-compaction mistakes are confounded and must not be attributed to model
  capability. It was stopped, the tool restored, the wire audit strengthened,
  and B rerun. The final B above is that corrected attempt.

## Interpretation and next discriminator

The earlier forced-recovery results still support notes/reset interface capability
and an exploratory latency advantage. This autonomous run answers a different
question: merely exposing the controls did not cause Escha to use them before the
observed capacity interruption. It cannot distinguish a weak planning policy
from insufficiently salient instructions or a capacity loss that arrives too soon.

The next controlled comparison should cross two instruction policies with the
same interfaces: discretionary use versus an explicit obligation to review
headroom and choose continue/checkpoint/retrieve after each completed update.
Leave the threshold and action to the model. Compare chosen actions, recovery
correctness and cost; do not call a prescribed reset threshold autonomous choice.
A stable capacity envelope and at least three observed boundaries are required
before claiming long-session endurance. Separately test capacity recovery with
the real resume scheduler and duplicate-action checks.

## Reproduce and evidence

From /Users/peppi/Dev/nanobot-rs, run inside tmux:

```sh
python3 -u experiments/context-recovery/endurance.py \
  experiments/context-recovery/endurance-next \
  --updates 20 --arms A,B --ceiling 16384 --minutes 45
```

The current release test binary and hardened Higgs worktree must exist. The driver
uses higgs-endurance.toml by default, refuses artifact overwrite, and saves binary
hashes/configuration. The hardened Higgs tmux session is recovery-higgs.

- [Machine-readable final comparison](endurance-summary.json)
- [Protocol and corrections](ENDURANCE.md)
- A raw evidence: endurance-main/A (gitignored)
- Corrected B raw evidence: endurance-corrected/B (gitignored)
- Corrected B provenance: endurance-corrected/provenance.json (gitignored)
- Previous forced-recovery comparison: [RESULTS.md](RESULTS.md)

Nanobot production base: fa53da492f100a7388571913f425c7419e1db1c3.
Higgs hardening base: 327e5021ef957a7d6968f6780a7644f00b410a9e,
feature/adaptive-capacity-higgs. Escha model: Qwen3.6-35B-A3B-Escha-W2.
The test-only harness changed between the A and corrected B runs; A's retrieval
catalog was already intact. Treat these as exploratory attempts, not locked trials.

Validation after final Rust changes: cargo test --release agent::agent_loop:
268 passed, 0 failed, 12 opt-in ignored; cargo build --release passed. Python
syntax checks and final SQLite/request audits passed. No production feature,
commit or deployment was made.
