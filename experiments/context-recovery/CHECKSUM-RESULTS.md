# Checksum investigation — 2026-09-06

The first failure is a semantic reinterpretation: Escha treats the checksum as something to recompute when project state changes. The actual recorded response says it needs to compute a checksum, calls the previous value a checksum format, and proposes generating a new value. Correct notes bytes were present before that decision.

The checkpoint's checksum was in an introductory sentence but omitted from its explicit `Snapshot submitted:` field list. Two isolated interventions address the ambiguity: move the existing checksum into that list, or explicitly require opaque identifiers to be copied byte-for-byte unless changed by `authoritative_update`. Neither intervention supplies new task facts.

## Cold decision probes

| Single change | Exact snapshots |
|---|---:|
| Original request/settings | 2/4 |
| Neutral repetition penalty (1.0 instead of 1.1) | 3/4 |
| Existing checksum moved into snapshot field list | 4/4 |
| Explicit opaque-identifier copy instruction | 4/4 |

These are four trials across three recorded decision fixtures: revision 8 twice, revisions 14 and 16 once each. At revision 8, original fails twice and every intervention passes twice. At revision 14, neutralizing the penalty introduces a failure while the original cold request passes. Therefore a blanket penalty removal is not supported. No broad reliability percentage or endurance pass follows from this small sample.

The stored provider request was translated to local HTTP using the provider's actual body construction: no client temperature override, thinking disabled, repetition alias 1.1, normalized object schemas and max-prompt control. The server fixture defaults to temperature 0.0. Cold probes omit retained-session controls deliberately. Exact probe requests/responses are saved. Later original cold decisions pass despite their original retained-run failures, so cold replay does not reproduce all numerical/execution conditions.

## Retained continuation confirmation

Reconstructed the real two-request sequence on revisions 8 and 14: request notes,
then supply the recorded notes receipt with the actual returned tool-call id and
continue the same Higgs session. Original instructions: **0/2 exact**. Explicit
copy instruction: **2/2 exact**. All four decisions report cached prompt tokens
(4557–4639). No tools were executed; checkpoint evidence was injected verbatim.
These are reconstructed fresh sessions, not a byte-identical restoration of the
original long-run cache. They confirm the instruction result on retained execution.

## Sampling and configuration findings

Nanobot's served model id is `escha-35b-a3b`, which enters the generic local repetition-control branch and emits `repeat_penalty: 1.1`. Higgs accepts this alias. Its merge function takes the maximum of the alias and canonical `repetition_penalty`: adding canonical 1.0 beside alias 1.1 cannot disable the effective penalty. A nearby nanobot comment saying Higgs ignores the alias is stale relative to the installed Higgs implementation.

Higgs applies repetition penalties to generated-token history, changing logits before sampling. This can alter the model's explanatory text and subsequent decision even at temperature zero; it is not simply penalizing every token already present in the input notes. The ablations show sensitivity, not that this penalty alone explains every checksum error.

## Separate output-shape weakness

Revision 10's result was nested inside an extra `result` object. The fixture's `EnduranceStream::submit` records arbitrary JSON, advances the cursor and acknowledges recording; it deliberately supplies no correctness feedback. Exact scoring catches the malformed shape, but the tool does not reject it for repair. This is distinct from checksum semantics. Strict schema validation can reject malformed shape without revealing expected task values.

## Recommended correction and limits

The supported first correction is the explicit field contract: identifiers including checksum are opaque observations, preserved verbatim unless an authoritative delta changes them. Keep every required state field in the checkpoint's explicit snapshot structure. Validate submitted shape separately. Preserve the penalty setting until broader behavior testing supports a change.

These probes inspect model decisions; returned tools are never executed. They do not establish a new 20-update endurance score or validate checkpoint persistence again. Production binaries and defaults were not modified by this investigation.

Evidence: `checksum-probe-evidence.json`, `checksum-diagnosis-trace.json`; raw directories `endurance-checksum-probe/`, `endurance-checksum-rev14/`, `endurance-checksum-rev16/`, and `endurance-checksum-retained/`. `checksum-probe.py` and `checksum-retained.py` reproduce the diagnostic procedures against the installed server. Revision-8's original probe-source snapshot is preserved with its raw artifacts.
