#!/usr/bin/env python3
"""Compare two normalized benchmark CSVs and print a delta table.

Usage:
    python3 scripts/bench_diff.py benches/baseline.csv /tmp/nanobot-speed.csv

Per-task deltas are reported for the metrics that matter to a user:
context_ms, ttfb_ms, total_ms. The headline number is the sum of total_ms
across all tasks present in BOTH files — a PR that regresses that number
by >5% needs explanation in the PR description.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict


METRICS = ("cold_start_ms", "context_ms", "ttfb_ms", "total_ms")


def load(path: str) -> dict[str, dict[str, float]]:
    """Map task_id -> {metric: value}. If multiple rows per task (different
    runs of the same task), the most recent (last) wins."""
    by_task: dict[str, dict[str, float]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            tid = row["task_id"]
            by_task[tid] = {m: float(row[m]) for m in METRICS}
    return by_task


def fmt_pct(old: float, new: float) -> str:
    if old == 0:
        return "  +∞%" if new > 0 else "    0%"
    pct = (new - old) / old * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:6.1f}%"


def main(base_path: str, new_path: str) -> int:
    base = load(base_path)
    new = load(new_path)

    common = sorted(set(base) & set(new))
    only_new = sorted(set(new) - set(base))
    missing = sorted(set(base) - set(new))

    if not common and not only_new:
        print("no rows to compare; check that the new CSV has rows beyond the header")
        return 1

    header = f"{'task':<14}" + "".join(f"{m:>14}{'Δ':>9}" for m in METRICS)
    print(header)
    print("-" * len(header))

    base_total = new_total = 0.0
    for tid in common:
        line = f"{tid:<14}"
        for m in METRICS:
            old_v = base[tid][m]
            new_v = new[tid][m]
            line += f"{new_v:>14.0f}{fmt_pct(old_v, new_v):>9}"
            if m == "total_ms":
                base_total += old_v
                new_total += new_v
        print(line)

    for tid in only_new:
        line = f"{tid:<14}"
        for m in METRICS:
            new_v = new[tid][m]
            line += f"{new_v:>14.0f}{'  (new)':>9}"
            if m == "total_ms":
                new_total += new_v
        print(line)

    if missing:
        print(f"\nmissing from new run: {', '.join(missing)}")

    print()
    if base_total > 0:
        delta = (new_total - base_total) / base_total * 100
        sign = "+" if delta >= 0 else ""
        print(f"headline total_ms: base={base_total:.0f}  new={new_total:.0f}  Δ={sign}{delta:.1f}%")
        if delta > 5.0:
            print("REGRESSION >5%: requires explanation in the PR description.")
            return 1
    else:
        print(f"headline total_ms: base=0  new={new_total:.0f}  (no baseline to compare)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
