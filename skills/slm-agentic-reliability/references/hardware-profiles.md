# Hardware Profiles

Use these profiles as conservative starting points. Validate with real workload traces.

## Profile: `rtx3090` (24 GB VRAM)

Guidelines:
- keep VRAM headroom (target ~15-20%) for KV cache growth and tool overhead
- start with concurrency `1`; increase only after stable runs
- prefer 4-bit or 5-bit quant for larger contexts

Tuning order:
1. set context limit
2. set model/quant
3. set output token cap
4. set concurrency

If instability appears:
- cut context first
- then lower concurrency
- then move to smaller model

## Profile: `apple-m4-32g` (32 GB unified memory)

Guidelines:
- reserve memory for OS and background tasks before model budgeting
- keep concurrency `1` by default
- prefer Metal-optimized runtimes and conservative context

Tuning order:
1. set context limit
2. set model/quant
3. set output token cap
4. set concurrency only after long-run stability

If instability appears:
- reduce context and output tokens
- lower model size or quant
- disable heavy parallel tool calls

## Cross-Profile Rules

- do not run at maximum memory utilization
- enforce explicit token caps per turn
- keep fallback model available for overload conditions
- benchmark latency and failure rate together, not separately
