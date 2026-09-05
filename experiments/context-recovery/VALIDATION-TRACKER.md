# Escha precision and context-recovery validation

Updated: 2026-09-05. Keep precision, reset timing and recovery correctness separate.

## Evidence so far

| Track | Result | Interpretation |
|---|---|---|
| FP16 attention, 4K/8K/16K | Same-precision digests repeat, cross-precision digests differ; variable KV halves, process peak only 0.728% lower | Long-context exact parity failed; semantic impact unmeasured; candidate uninstalled |
| LCM/default reserve | A 4/4 correct, B 5/5; both suspended, B no resets | Capacity-limited; no strategy winner |
| LCM/2048 reserve | A 4/4 correct, then suspended | Actual wire reservations 2048 |
| Notes/reset/2048, no handoff | 20 submitted, 3 correct, 5 resets, 3 later LCM boundaries | Correct checkpoint was not read after first reset; bad state propagated |
| Notes/reset/2048, handoff enabled | 5/5 correct, then suspended; zero actual resets/handoffs | Model announced reset without executing it; handoff not exercised |
| New grammar wire test | Required calls passed blocking/SSE, including conflicting plain-text prompt; auto returned requested plain text | Actual decoder enforcement verified |
| New announcement end-to-end test | Durable notes → reset boundary → distinct session → notes read → exact snapshot, no repeated action | Execution and recovery work with adequate headroom |
| Frozen revision-3 OFF/ON/OFF/ON | OFF 0/2 correct, 185.3s/139.9s; ON 2/2 correct, 43.1s/44.0s | Pointer-only instruction fixes this repeated recovery fixture; not an LCM speed comparison |
| New autonomous notes/reset arm | 6/6 correct of 20 planned; 1 reset/handoff, 1 LCM boundary, 1 rejected duplicate, then capacity suspension; no forced retry | Correct recovery after autonomous reset; endurance goal not completed |
| Matched LCM arm on new binaries | 6/6 correct of 20, 193.6s, 1 LCM boundary, zero duplicates, capacity suspension | Faster than B (295.8s) here; neither passed endurance |

## Runtime restored

Both matched arms are terminal. Installed Higgs is serving in `recovery-higgs:0.0`,
PID 69407, boot `dd9cdc15-7655-4b4b-9b48-e753664d4c04`; installed executable and Metal
mappings/hashes verified. Default auto resolves to throughput, native FP32 cache
40960 bytes/token, scratch_matmul prefill, chunk 1024. Exact READY smoke passed.
Full runtime evidence is in `BINARY-PROVENANCE-AFTER.json`.

## Implemented and verified

- Nanobot `721bc3b`: explicit announcement detection, persisted execution instruction, restricted/validated recovery calls, stateless forced requests, truthful streamed failure replacement, durable fixture checkpoint sequencing.
- Higgs `dd6730133`: required/named tool-choice grammar using the existing FSM/parser; thinking disabled for forced calls. Local `nightly` includes the commit.
- Installed Higgs SHA-256: `fd9747c68f4f85770eb47d30cf7c077b9a014e559a71dc03e2b46380dcfa86fc`.
- Installed nanobot `7b8b24a`, SHA-256: `e26c4082bff1dea1540a593cc176e4fd092b3f12e7fb9504a1b9717fc42755a0`.
- Release suites after overflow fix: nanobot 3003 passed/31 ignored across targets; Higgs 1528 passed/35 ignored. Both release builds passed.
- Matched CLI sample: identical final replies/token counts; warm mean 1236.5ms before, 1228.5ms after. Too small to claim a speed improvement; no meaningful regression observed.
- Full graph analysis and independent review completed. Main retry/chat routes have CRITICAL graph impact; two review findings were fixed and rechecked. User instruction files restored byte-identically after indexing.

## Remaining for this run

- [x] Record autonomous outcome, actual forced calls/resets/handoffs and capacity coverage.
- [x] Restore service using installed Higgs with default kernel/precision settings; verify live PID, binary/Metal mappings, hashes and response.
- [x] Prepare final evidence and comparison update for the evidence commit.

## Interpretation and reports

B retains automatic LCM fallback; this is not compaction-disabled versus compaction-enabled.
The old `compaction_attempts` field counted only LLM summarizer requests. Current
reports use `llm_compaction_requests` and count completed LCM boundaries separately.
The earlier boundary could publish deterministic truncation when summarization did
not fit; that was real LCM compaction, not a hard reset.

`ANNOUNCEMENT-RESULTS.md`, `announcement-evidence.json`, `grammar-evidence.json`,
`handoff-replay-evidence.json`, and `announcement-speed-evidence.json` cover the new
work. `CONTEXT-VALIDATION-RESULTS.md`, `LCM-NOTES-CHOICE-AUDIT.md`,
`budget2048-evidence.json`, and `reset-handoff-evidence.json` preserve earlier controls.
Precision results are in `fp16-probe/LONG-CONTEXT.md`; diverse semantic tasks and
HTTP retained-cache validation remain separate precision gates.

Public Higgs follow-up push remains blocked by automatic approval review pending
explicit public-destination authorization. Local binaries/branches include the work.

## Endurance audit follow-up

The pre-fix B overflow retry reversed the current user/call/receipt suffix.
Commit `7b8b24a` fixes the backward accumulator; public-path regression RED→GREEN,
3003 release tests passed and corrected executable installed. Earlier endurance
results are preserved and have not been rerun on this correction.
