# Local Failure Taxonomy

## Context Overflow

Signals:
- `exceed_context_size_error`
- `request (...) exceeds the available context size`

Likely causes:
- prompt growth without truncation/summarization
- excessive tool output in context
- context target larger than runtime `n_ctx`

First fixes:
- truncate/summarize history before model call
- cap tool output inclusion
- lower max input tokens or increase runtime context

## Tool Call Protocol Mismatch

Signals:
- `tool_call_id ... not found in tool_calls`

Likely causes:
- provider-specific tool format mismatch
- stale tool-call state across retries
- malformed assistant/tool message ordering

First fixes:
- enforce strict tool-call pairing per provider contract
- clear transient tool state on retry
- validate message sequence before send

## OOM or Memory Pressure

Signals:
- `out of memory`
- CUDA/Metal memory allocation failures
- abrupt local server termination under load

Likely causes:
- oversized model/context for hardware
- no memory headroom for KV cache growth
- excessive concurrency

First fixes:
- reduce context length and/or batch
- switch to smaller quant/model
- lower concurrency to 1 until stable

## Timeout

Signals:
- `timed out`
- `deadline exceeded`

Likely causes:
- model too large for latency target
- slow tool path blocks loop
- missing watchdog and retries

First fixes:
- lower output token cap
- split long tasks into smaller turns
- add watchdog timeout and bounded retries

## Connection Failure

Signals:
- `connection refused`
- `failed to connect`
- health check failure

Likely causes:
- local server not running
- wrong host/port
- server crash loop

First fixes:
- verify process and bind address
- add startup health checks
- add auto-restart with backoff

## Invalid Request

Signals:
- `HTTP 400 Bad Request`
- `invalid_request_error`

Likely causes:
- malformed payload
- unsupported options for provider
- schema drift in tool parameters

First fixes:
- diff payload against known-good request
- remove unsupported params
- enforce schema validation pre-send
