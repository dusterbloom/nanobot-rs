# Shader profiling result

Used a cost-effective fast_scan sub-agent for read-only feasibility/source validation; main captured local GPU workloads. No GLM API call, production edit or installed binary change.

Measured Metal pipeline resource reports:

| Variant | Static threadgroup bytes | Max threads | SIMD width |
|---|---:|---:|---:|
| Original16x8 causal | 28,928 | 64 | 32 |
| Register-Q16x16 explicit | 20,480 | 64 | 32 |
| Register-Q16x16 causal | 20,480 | 64 | 32 |

The register-Q explicit/causal performance split is not explained by these identical static limits. Reported max threads is not occupancy; no physical register/spill or hardware-bank-conflict measurement was obtained.

Device counter enumeration via public Metal API returned only timestamp/GPUTimestamp. No occupancy, bandwidth or stall counter set was exposed. The sub-agent found no supported `.gputrace` CLI counter exporter; Xcode Metal Debugger is needed for deeper shader profiling/replay. Xcode UI automation failed/timed out here, so the actual hardware limiter remains unresolved.

Four warmed32K captures were attempted. The original two causal captures were EMPTY and must not be used. Root cause: pinned MLX eval is synchronous but GPU finalize commits then pre-creates the next command buffer before start_capture. The causal dispatch fit in that pre-capture buffer. Explicit/query128 workloads created additional buffers inside capture and contained work. Adding synchronize() before start_capture and after eval before stop_capture fixed the causal captures; both now have command/resource records (capture4268bytes, deviceId1) and ~296MB bundles. This capture boundary issue does not invalidate previous eval-synchronized timing experiments. Source corroborated by sub-agent in mlx/backend/metal/eval.cpp finalize/synchronize.

Valid frame artifacts remain in /private/tmp/higgs-fused-profile:
- query128.gputrace (~555MB)
- reg-explicit.gputrace (~296MB)
- original-causal-sync.gputrace (~296MB)
- reg-causal-sync.gputrace (~296MB)

Old original-causal.gputrace and reg-causal.gputrace (40KB, 8-byte capture stream) are invalid diagnostic artifacts, retained solely to document the capture bug. Kernel metallibs did not embed full debug sources; matching shader source remains under /private/tmp/higgs-fused-v2 and durable fused-v2 source artifacts.

Installed Higgs was restored after captures; supervisor confirmed available=true, normal pressure and maxPromptTokens47104. Source/build/resource/counter logs accompany this report. No speedup or hardware-limiter claim follows from static resource reports. Further kernel changes should wait for actual shader profiler evidence.
