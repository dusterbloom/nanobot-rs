#!/usr/bin/env python3
"""Reconcile recorded Escha retention bytes with its actual model geometry."""
import json
from pathlib import Path
import sys

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name('endurance-policy-decision') / 'B-telemetry.jsonl'
samples = [json.loads(line) for line in path.read_text().splitlines()]
# Pinned Escha config: 30 GDN layers, 32 value heads, 128x128 FP32 state;
# 3 conv history rows, 2*16*128 key channels + 32*128 value channels.
fixed = 30 * (32 * 128 * 128 * 4 + 3 * (2 * 16 * 128 + 32 * 128) * 4)
# 10 full-attention layers, keys+values, 2 KV heads, 256 head dimension.
geometry = 10 * 2 * 2 * 256
rows = {}
for sample in samples:
    cache = sample['cache']
    actual = cache['retained_bytes']
    tokens = cache['session_last_retained_tokens']
    if not actual:
        continue
    allocated_tokens = ((tokens + 255) // 256) * 256
    row = {'retained_tokens': tokens, 'allocated_tokens': allocated_tokens,
           'observed_bytes': actual, 'fp32_prediction': fixed + allocated_tokens * geometry * 4,
           'fp16_prediction_same_fixed': fixed + allocated_tokens * geometry * 2}
    assert actual == row['fp32_prediction'], row
    assert actual != row['fp16_prediction_same_fixed'], row
    rows[tokens, actual] = row
assert len(rows) >= 3, 'Insufficient independent captured retention points'
cap = 512 * 1024 * 1024
first_oversized = next(n for n in range(256, 16385, 256) if fixed + n * geometry * 4 > cap)
print(json.dumps({'source': str(path), 'fixed_gdn_bytes': fixed,
                  'observed_bytes_per_token': geometry * 4, 'estimator_bytes_per_token': geometry * 2,
                  'retained_byte_ceiling': cap, 'last_fitting_allocated_tokens': first_oversized - 256,
                  'first_oversized_allocated_tokens': first_oversized, 'matched_points': list(rows.values())}, indent=2))
