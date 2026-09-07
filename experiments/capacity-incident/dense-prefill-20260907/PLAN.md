# Dense attention prefill investigation

- [x] Trace pinned MLX FP32 D256 fallback and score scratch.
- [x] Build isolated fused-kernel variants; compare numerics and timing; reject slower candidates.
- [x] Test attention-only query blocks without shrinking the model prefill chunk.
- [x] Add bounded-scratch, offset/tail, noncausal and broadcast-mask regressions; observe RED then GREEN.
- [x] Validate release build, canonical paths and base-M4 selection.
- [x] Compare 32K and matched AC 45K serving; retrieve facts and reuse cached context.
- [x] Validate final automatic selector at 45K with no experimental override.
- [x] Independent review and pre-commit graph check.
- [x] Commit only source and model documentation to nightly: d77352ef8.
- [x] Verify installed binary/library hashes and live answer.

Outcome: reduced peak process memory with essentially flat 45K latency. No ANE or custom fused kernel shipped. Detailed evidence and limitations in STATUS.md.
