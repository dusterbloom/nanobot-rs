# Sanitized GPU interval analysis

Source: `gpu-intervals.xml`, exported from the GPU interval table of the
10-second Metal System Trace. This summary contains no environment variables,
request bodies, or trace table-of-contents dump.

## Method

- Parsed all 4,943 rows and resolved XML `id`/`ref` indirection globally.
- Resolution check: 41,329 unique IDs, 81,154 references, zero unresolved
  prior references, and zero duplicate IDs.
- Every exported row has GPU state `Active`.
- Treated each row as the half-open interval `[start, start + duration)` in
  nanoseconds. Utilization is the union of intervals, so nested encoder rows
  and overlapping Compute/Vertex/Fragment channels are never double-counted.
- Process figures use the union of intervals tagged with each resolved process.
  They are non-additive because processes can overlap on different GPU channels.

## Timeline utilization

The trace metadata reports a requested time limit of 10 seconds and an actual
recording duration of 11.546139 seconds. GPU rows begin at 0.620978583 seconds
and end at 11.545072500 seconds.

| Measure | Time | Fraction |
| --- | ---: | ---: |
| Any GPU interval active, full recorded run | 9.608811295 s | 83.221% |
| No GPU interval active, full recorded run | 1.937327705 s | 16.779% |
| Any GPU interval active, first-to-last GPU interval envelope | 9.608811295 s | 87.960% |
| Idle inside first-to-last envelope | 1.315282622 s | 12.040% |
| Initial recording time before first GPU interval | 0.620978583 s | 5.378% of run |
| Trailing recording time after last GPU interval | 0.001066500 s | 0.009% of run |

The union contains 1,605 active segments and 1,604 internal gaps. Internal gap
median is 0.000750 ms, p90 is 2.393541 ms, p95 is 5.920958 ms, p99 is
12.320000 ms, and the maximum is 24.462541 ms. The 94 gaps at least 5 ms long
account for 0.908674664 s, or 69.1% of internal idle time. The 36 gaps at least
10 ms long account for 0.485625372 s.

For reference only, clipping timestamps to the literal `[0, 10 s]` interval
gives 8.213540460 s active (82.135%). This is not the primary denominator
because the exported recording continued through 11.546 seconds and contains
GPU work after 10 seconds.

## Process attribution

| Resolved process | Rows | Union-active time | % of recorded run | % of all GPU-active time |
| --- | ---: | ---: | ---: | ---: |
| `higgs` (PID 75074) | 1,778 | 9.477316295 s | 82.082% | 98.632% |
| Google Chrome Helper (PID 1871) | 339 | 0.688804455 s | 5.966% | 7.168% |
| WindowServer (PID 445) | 2,538 | 0.591719055 s | 5.125% | 6.158% |
| Terminal (PID 707) | 282 | 0.248151535 s | 2.149% | 2.583% |
| Unattributed | 6 | 0.042439208 s | 0.368% | 0.442% |

The percentages in the final column overlap and must not be summed. Sweeping
the union of each process's intervals gives:

- `higgs` alone: 8.081822872 s (84.108% of all GPU-active time).
- `higgs` overlapping one or more other processes: 1.395493423 s (14.523%).
- GPU activity with no `higgs` interval: 0.131495000 s (1.368%).

All 1,778 `higgs` rows are on the Compute channel. Across all processes, the
Compute-channel union is 9.520429423 s (99.080% of GPU-active time), Fragment
is 1.122429051 s (11.681%), and Vertex is 0.395159192 s (4.112%). Channel
fractions overlap and are non-additive.

## Supported conclusions

- The GPU had at least one active interval for 83.2% of the complete exported
  recording, or 88.0% between the first and last GPU activity.
- `higgs` accounts for intervals covering 98.6% of union GPU-active time; only
  1.37% of active time occurs without a `higgs` interval.
- Most internal idle time is concentrated in millisecond-scale gaps rather
  than the many sub-microsecond boundaries between adjacent intervals.
- Other desktop processes overlap `higgs` activity, but their interval union
  is small compared with `higgs` and cannot be interpreted as an additive GPU
  utilization share.

## What this trace cannot prove

- An `Active` interval means submitted GPU work was executing; it does not mean
  shader cores, SIMD lanes, memory bandwidth, or the whole GPU were saturated.
- This table has no FLOP, byte-traffic, bandwidth, cache, occupancy, stall,
  power, or thermal counters. It cannot establish a roofline point or classify
  the workload as compute-bound versus memory-bound.
- Overlapping process intervals do not apportion physical GPU resources, so
  the trace cannot quantify how much WindowServer, Chrome, or Terminal slowed
  `higgs`.
- The requested 10-second limit produced an 11.546-second exported run. Without
  an explicit application signpost, the exact workload-only window cannot be
  separated from launch and teardown overhead.
- The interval table does not provide per-kernel arithmetic intensity or
  enough labeled encoder detail to identify which model operation caused an
  idle gap.
