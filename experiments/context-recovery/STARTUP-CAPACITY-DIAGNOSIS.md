# Startup capacity diagnosis

The 9,216-token startup envelope was not caused by a smaller backend memory
authority. The current boot and the earlier endurance boots all logged the same
Metal authority, 26,800,603,136 bytes, with zero MLX-active bytes before model
load. The current idle allocator measurement is 12,377,847,584 active bytes.

## Shared learned profile

`higgs serve -c experiments/context-recovery/higgs-endurance.toml` derives its
learned-profile directory from the config file's parent:

```
experiments/context-recovery/capacity/
```

Consequently, every previous A and B arm using that config shared
`capacity/bc0bcc7556ca726e0011bb9eb0d5c880f47bf472a3f98c29e93d521d6e0812dc.json`.
At the diagnosed boot the profile key exactly matched the hardware, OS, backend
authority, Higgs build, model fingerprint, execution mode, and cache settings.
Its startup headroom also exactly matched 26,800,603,136 bytes, so Higgs restored
these entries:

| Prompt band | Retained high-water bytes | Cold high-water bytes | Cold qualified |
| --- | ---: | ---: | --- |
| 8,192 | 406,945,792 | 0 | false |
| 16,384 | 499,220,480 | 0 | false |

This is retained-session evidence. No qualified cold replacement constrained
this startup.

## Exact 9,216-token ledger

The pressure observer starts before the 17-second model load. A nominally Normal
OS event with a positive compression delta is treated as effective Constrained
pressure. A model registered in that state uses a 30% protected reserve:

```
26,800,603,136 - 30% = 18,760,422,196 usable bytes
```

The fixed ledger at the 8,192 prompt band was:

| Term | Bytes |
| --- | ---: |
| Loaded MLX model | 12,377,847,584 |
| Fixed live session | 278,880,256 |
| Decode workspace | 268,435,456 |
| Retained and prefix cache ceilings | 939,524,096 |
| Static prefill transient (`artifact_bytes / 3`) | 4,098,984,160 |
| Persisted retained high water | 406,945,792 |
| **Fixed subtotal** | **18,370,617,344** |

At 40,960 bytes per token, 9,216 total tokens add 377,487,360 bytes. The
resulting 18,748,104,704-byte ledger fits with 12,317,492 bytes remaining.
Adding the next 1,024-token alignment step requires another 41,943,040 bytes
and does not fit. Therefore the published result is exactly 9,216 total and
5,120 prompt tokens.

Without the persisted 406,945,792-byte retained term, the same Constrained
ledger yields exactly 18,432 total and 14,336 prompt tokens. Under Normal
pressure and without learned evidence, the same inputs yield 83,968 total
tokens. Thus the current 9,216 result requires both facts: registration during
the compression-derived Constrained state and the compatible shared profile.

The current diagnostics later show raw and effective Normal, zero raw
non-Normal event epochs, and zero model downshifts. Those values are consistent
with this sequence. Compression-derived pressure does not increment the raw
pressure-event epoch, and the model was born Constrained rather than downshifted
after registration. The first clean cadence restored Normal pressure, but the
controller's recovery rule retains `min(previous, recomputed)`. The log records
that recovery as `from_tokens=9216 to_tokens=9216` with cause `memory pressure`.

The earlier reported 24,576-token value has no preserved capacity or telemetry
artifact in this experiment tree or the temporary Higgs logs. It cannot be
reconstructed as a static result from the current authority, costs, and profile,
so it must not be cited as a verified comparable startup envelope without its
original profile and pressure snapshot.

## Experiment isolation

There is no separate capacity-profile environment variable. The supported
isolation boundary is the config path: Higgs stores profiles in a `capacity/`
directory beside the selected config. The endurance driver now copies the same
config bytes into `server-A/` and `server-B/`, selects that copy before starting
each server, and records both per-arm paths in provenance. This preserves
profile persistence within an arm while preventing evidence from crossing arms.

## Profile-key label

The diagnosed pre-correction key said `kvRepresentation: "fp16"` because it described the configured KV
mode. Native Escha execution actually promotes K/V storage and is charged at
40,960 bytes per token. This label is semantically stale, but it did not cause
this reuse: `executionMode`, the resolved runtime settings fingerprint, and the
Higgs executable build identity also participate in exact profile compatibility.
It remains a naming and future key-collision risk if runtime KV behavior changes
without changing any of those identity fields.

Local nightly `78da18f13` corrects the native Escha key label to `fp32`, matching actual baseline storage. A release regression verifies that metadata. The isolated FP16 experiments do not change the installed baseline or this diagnosis.
