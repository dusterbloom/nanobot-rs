# Loop Reliability Patterns

Apply these patterns to every agentic loop.

## Bounded Loop Contract

Maintain state fields:
- `phase`
- `turn_index`
- `tool_budget_remaining`
- `attempt`
- `deadline_ms`

Enforce hard limits:
- `max_turns`
- `max_tool_calls`
- `max_runtime_ms`

Abort cleanly when any budget hits zero.

## Tool Safety Contract

- allow only whitelisted tools
- validate args against schema before execution
- reject unknown args
- normalize tool output size before re-injecting into prompt

## Retry Strategy

- retry only retryable classes (`timeout`, transient connection failures)
- cap retry attempts
- apply backoff and jitter
- reset transient loop state between attempts

Do not retry schema violations or deterministic logic errors without a code/config change.

## Output Validation

Before final answer:
- validate required structure (for example strict JSON)
- run lightweight repair pass if validation fails
- emit explicit failure code if repair fails

## Fallback Routing

Route to fallback model when:
- memory pressure exceeds threshold
- latency exceeds target budget repeatedly
- primary model triggers repeated protocol errors

Keep fallback behavior deterministic and logged.
