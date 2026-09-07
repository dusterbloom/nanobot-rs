# Fused D256 discriminator
User approved the preceding design with keep going. Isolated native benchmark, FP32 Q16/KV2/D256; no production edits.
- [x] Impact lookup external symbols UNKNOWN; source traced dispatcher -> kernel, benchmark -> SDPA. Only copies change.
- [x] Build original16x8, separate-unpadded16x16, safe-register16x16 in one executable; distinct pipeline cache identities.
- [x] Gate all arms against fallback at offset/tail and 8K/32K/45K, normal and amplified Q/K distributions.
- [x] Warm arms; run six balanced interleaved brackets against explicit-mask query128, normal pressure/no new swapouts/stable power.
- [x] Analyze paired ratios and dispersion; promote only repeatable winner to serving validation, otherwise retain installed d77352ef8.
- [x] Restore and verify live server; preserve source, logs and outcome in workspace.

Explicit-mask pass completed with all numerical gates passing; no candidate win. Native-causal diagnostic added with same-process controls to distinguish mask cost. 32K complete: no candidate beats query128 in any of6rounds. 45K underway. Initial benchmark build scalar-array compile error fixed; initial server restart rejected transient critical pressure, manually retried successfully. Final restore receives20s settling.

COMPLETE: both mask modes, six rounds, no fused win. Final installed server restored and smoke checked. No serving integration or production commit warranted.
