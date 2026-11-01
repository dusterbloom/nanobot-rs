#!/usr/bin/env python3
"""Read one arm's isolated ~/.nanobot (metrics.jsonl + sessions.db + logs) and
print a one-line JSON summary. Used by run.sh; not meant to be clever.

Metrics source priority, per docs/superpowers/plans/2026-11-01-model-managed-context.md T9:
  1. RequestMetrics rows in metrics.jsonl (prompt_tokens, cache_read_tokens).
  2. If cache_read_tokens is always None (local server reports no cache usage),
     fall back to the PromptDelta proxy logged by
     src/agent/agent_loop/shared.rs at target nanobot::agent::agent_loop::shared
     (prompt_cache_cold_start / prompt_append_only / prompt_prefix_diverged /
     prompt_cache_sanctioned_reset), captured in stderr.log when the arm was
     run with RUST_LOG="warn,nanobot::agent::agent_loop::shared=debug".
"""
import json
import re
import sqlite3
import sys
from pathlib import Path


def load_metrics(home: Path):
    path = home / ".nanobot" / "metrics.jsonl"
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def count_context_tool_calls(home: Path) -> int:
    db_path = home / ".nanobot" / "sessions.db"
    if not db_path.exists():
        return 0
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = db.execute(
            "select payload_json from session_events where event_kind = 'tool_pre_execute'"
        ).fetchall()
    except sqlite3.OperationalError:
        return 0
    n = 0
    for (raw,) in rows:
        try:
            e = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if e.get("tool_name") == "context":
            n += 1
    return n


def prompt_delta_proxy(home: Path) -> dict:
    # nanobot's non-interactive tracing writer defaults to stdout, so the
    # debug "prompt_*" lines and the rendered transcript both land in run.log.
    path = home / "run.log"
    counts = {"cold_start": 0, "append_only": 0, "diverged": 0, "sanctioned_reset": 0}
    if not path.exists():
        return counts
    text = path.read_text(errors="replace")
    counts["cold_start"] = len(re.findall(r"prompt_cache_cold_start", text))
    counts["append_only"] = len(re.findall(r"prompt_append_only", text))
    counts["diverged"] = len(re.findall(r"prompt_prefix_diverged", text))
    counts["sanctioned_reset"] = len(re.findall(r"prompt_cache_sanctioned_reset", text))
    return counts


def summarize(home: Path, label: str, expect_substring: str) -> dict:
    metrics = load_metrics(home)
    main_rows = [m for m in metrics if m.get("role") == "main"] or metrics

    prompt_tokens = [m.get("prompt_tokens", 0) for m in main_rows]
    cache_read = [m.get("cache_read_tokens") for m in main_rows]
    cache_read_known = [c for c in cache_read if c is not None]

    requests = len(main_rows)
    total_prompt = sum(prompt_tokens)
    mean_prompt = (total_prompt / requests) if requests else 0.0

    cache_hit_pct = None
    if cache_read_known and total_prompt > 0:
        cache_hit_pct = 100.0 * sum(cache_read_known) / total_prompt

    proxy = prompt_delta_proxy(home)

    run_log_path = home / "run.log"
    run_log_text = run_log_path.read_text(errors="replace") if run_log_path.exists() else ""
    task_success = expect_substring in run_log_text

    return {
        "label": label,
        "requests": requests,
        "mean_prompt_tokens": round(mean_prompt, 1),
        "total_prompt_tokens": total_prompt,
        "cache_hit_pct": None if cache_hit_pct is None else round(cache_hit_pct, 1),
        "prompt_delta_proxy": proxy,
        "context_tool_calls": count_context_tool_calls(home),
        "task_success": task_success,
    }


def fmt_cache(s: dict) -> str:
    if s["cache_hit_pct"] is not None:
        return f"{s['cache_hit_pct']}%"
    p = s["prompt_delta_proxy"]
    return f"proxy: cold={p['cold_start']} append={p['append_only']} diverged={p['diverged']} reset={p['sanctioned_reset']}"


def main():
    if len(sys.argv) != 6:
        print(
            "usage: summarize.py <home_A> <label_A> <home_B> <label_B> <expect_substring>",
            file=sys.stderr,
        )
        sys.exit(1)
    home_a, label_a, home_b, label_b, expect = sys.argv[1:]
    sa = summarize(Path(home_a), label_a, expect)
    sb = summarize(Path(home_b), label_b, expect)

    print("| arm | requests | mean prompt tok/request | prefill cache hit % | total prompt tok | task success |")
    print("|-----|----------|--------------------------|----------------------|-------------------|--------------|")
    for s in (sa, sb):
        print(
            f"| {s['label']} | {s['requests']} | {s['mean_prompt_tokens']} | "
            f"{fmt_cache(s)} | {s['total_prompt_tokens']} | {s['task_success']} |"
        )
    print()
    print(f"context tool calls — {label_a}: {sa['context_tool_calls']}  |  {label_b}: {sb['context_tool_calls']}")
    print()
    print("raw:", json.dumps({"a": sa, "b": sb}))


if __name__ == "__main__":
    main()
