# Prior hardening: location and baseline

The comparison must preserve this work, not substitute a toy LCM implementation.

## Nanobot (merged into main at fa53da4)

- 2df8777: publish compaction checkpoints across turns.
- 4f431df: reliable first-attempt compaction; stop auto-expansion loops.
- 349689c: rotate prefix cache on divergence; enforce LCM/window ordering.
- 605c7ef: sent/draft MessageLog prevents mid-turn committed-history rewrites.
- becf08e, ff164b9, 7dd9e42, 8f9fe9f, 8c4d2ad, 819bd45:
  truthful tool results, replay lifecycle closure, original compacted/ranged
  evidence, atomic rejected rows and rejection convergence.
- eb3dce2, f748ed6: live Higgs capacity and compaction before unsafe prefill.
- 6676a16: one safe capacity retry without rerunning the outer turn/tools.
- 9c3975f: durable capacity-interrupted turns.
- b686dbf: normalize Higgs wire contracts.
- 3cf5f7d: reserve effective overflow budget.
- 76bac72: retain lease-renewal checkpoint.

## Higgs (NOT merged into the active nightly checkout)

Branch: feature/adaptive-capacity-higgs
Head: 327e5021ef957a7d6968f6780a7644f00b410a9e

- 5508a78a7, 723a92af7: versioned capacity wire contract.
- ddc927885, c2870629d: engine capacity memory facts.
- d442cf362 and follow-up fixes: adaptive byte controller.
- c0a1f927d and follow-up fixes: macOS pressure observations.
- 3fefd3c21 and follow-up fixes: model lifecycle accounting.
- 08c898784 and follow-up fixes: bounded model loading.
- b936f67d7, cb688412f: request admission and review fixes.
- a1556f2e2: cancel inference before releasing capacity.
- fdcfb257f: adaptive diagnostics.
- 327e5021e: preserve native Escha headroom and optional-cache budget.

The committed document
`docs/superpowers/plans/2026-09-04-adaptive-capacity-task8-continuation.md`
on this Higgs branch records completed release, hardware, and synchronized
Nanobot conformance gates. Historical /private/tmp/nac and
/private/tmp/higgs-adaptive-capacity worktrees/logs no longer exist here.
Recorded checkmarks are prior evidence, not fresh validation in this experiment.

The initially running ~/.local/bin/higgs and local nightly checkout 5c7605f91
were NOT an acceptable baseline for the synchronized adaptive-capacity work.
