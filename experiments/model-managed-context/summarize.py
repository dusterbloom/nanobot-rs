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

Also reports (added for the "pressure" task, T9 follow-up): peak prompt
tokens/request, how many `[System notice] Context ~` size notices fired, and
each `context` tool call's drop spec + whether a note was present. The size
notice and the context calls' arguments are read from the isolated
sessions.db's `messages` table (content / tool_calls columns) — not run.log,
see size_notice_count()'s docstring for why.

task_success grading also reads sessions.db, not run.log: see
last_assistant_message()'s docstring for why (run.log is every streamed
assistant message, not just the final answer, and can contain ANSI codes).
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
        if e.get("tool_name") == "context_edit":
            n += 1
    return n


def context_call_details(home: Path) -> list:
    # messages.tool_calls holds the OpenAI-shape tool_calls array (`function.
    # arguments` is itself a JSON-encoded string) -- src/agent/tools/
    # context_edit.rs's ContextTool params are {"drop": ..., "note": ...}.
    db_path = home / ".nanobot" / "sessions.db"
    if not db_path.exists():
        return []
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = db.execute(
            "select tool_calls from messages where tool_calls is not null"
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    calls = []
    for (raw,) in rows:
        try:
            tool_calls = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        for tc in tool_calls:
            fn = tc.get("function", {})
            if fn.get("name") != "context_edit":
                continue
            try:
                args = json.loads(fn.get("arguments", "{}"))
            except json.JSONDecodeError:
                args = {}
            calls.append(
                {
                    "drop": args.get("drop"),
                    "note_present": bool(args.get("note")),
                }
            )
    return calls


def last_assistant_message(home: Path) -> str:
    # Grade against the model's actual final answer, not run.log: run.log is
    # the streamed transcript of EVERY assistant message (ANSI color codes
    # inside syntax-highlighted quotes, which can split a fact across escape
    # sequences) -- a fact quoted mid-run right after its read would pass
    # even if the final answer lost it, defeating the memory test. The last
    # non-empty assistant message's `content` in the isolated sessions.db
    # (same DB used for size_notice_count above) is the model's actual final
    # answer, stored pre-render with no ANSI.
    db_path = home / ".nanobot" / "sessions.db"
    if not db_path.exists():
        return ""
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        row = db.execute(
            "select content from messages where role = 'assistant' "
            "and content is not null and content != '' "
            "order by id desc limit 1"
        ).fetchone()
    except sqlite3.OperationalError:
        return ""
    return row[0] if row else ""


def size_notice_count(home: Path) -> int:
    # The `[System notice] Context ~...` scaffold is injected server-side
    # (src/agent/agent_loop/shared.rs's inject_context_size_notice) straight
    # into ctx.messages -- it is never printed to stdout (the CLI only echoes
    # the -m prompt when stdout is a tty, which it isn't under run.sh's
    # redirect), so run.log can't be used here. It IS persisted verbatim as a
    # synthetic user message, so sessions.db's messages table is the reliable
    # source.
    db_path = home / ".nanobot" / "sessions.db"
    if not db_path.exists():
        return 0
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        (n,) = db.execute(
            "select count(*) from messages where content like '[System notice] Context ~%'"
        ).fetchone()
    except sqlite3.OperationalError:
        return 0
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


def summarize(home: Path, label: str, expect_substrings: list) -> dict:
    metrics = load_metrics(home)
    main_rows = [m for m in metrics if m.get("role") == "main"] or metrics

    prompt_tokens = [m.get("prompt_tokens", 0) for m in main_rows]
    cache_read = [m.get("cache_read_tokens") for m in main_rows]
    cache_read_known = [c for c in cache_read if c is not None]

    requests = len(main_rows)
    total_prompt = sum(prompt_tokens)
    mean_prompt = (total_prompt / requests) if requests else 0.0
    peak_prompt = max(prompt_tokens) if prompt_tokens else 0

    cache_hit_pct = None
    if cache_read_known and total_prompt > 0:
        cache_hit_pct = 100.0 * sum(cache_read_known) / total_prompt

    proxy = prompt_delta_proxy(home)

    final_text = last_assistant_message(home)
    fact_hits = [s in final_text for s in expect_substrings]
    task_success = all(fact_hits)

    return {
        "label": label,
        "requests": requests,
        "mean_prompt_tokens": round(mean_prompt, 1),
        "peak_prompt_tokens": peak_prompt,
        "total_prompt_tokens": total_prompt,
        "cache_hit_pct": None if cache_hit_pct is None else round(cache_hit_pct, 1),
        "prompt_delta_proxy": proxy,
        "context_tool_calls": count_context_tool_calls(home),
        "context_call_details": context_call_details(home),
        "size_notices": size_notice_count(home),
        "fact_hits": fact_hits,
        "task_success": task_success,
    }


def fmt_cache(s: dict) -> str:
    if s["cache_hit_pct"] is not None:
        return f"{s['cache_hit_pct']}%"
    p = s["prompt_delta_proxy"]
    return f"proxy: cold={p['cold_start']} append={p['append_only']} diverged={p['diverged']} reset={p['sanctioned_reset']}"


def fmt_fact_hits(fact_hits: list) -> str:
    # Letters label expect_substrings by position (A = 1st substring, etc.)
    # so partial loss (e.g. one fact dropped from memory) is visible even
    # when the overall task_success is already False.
    labels = [chr(65 + i) for i in range(len(fact_hits))]
    return " ".join(f"{label}{'✓' if hit else '✗'}" for label, hit in zip(labels, fact_hits))


def fmt_context_calls(s: dict) -> str:
    if not s["context_call_details"]:
        return "none"
    parts = []
    for c in s["context_call_details"]:
        drop = c["drop"] if c["drop"] is not None else "(none)"
        note = "note" if c["note_present"] else "no note"
        parts.append(f"drop={drop} {note}")
    return "; ".join(parts)


def main():
    if len(sys.argv) != 6:
        print(
            "usage: summarize.py <home_A> <label_A> <home_B> <label_B> <expect_substrings>",
            file=sys.stderr,
        )
        print(
            "  <expect_substrings>: one or more substrings, newline-separated in a single "
            "arg; success requires ALL to be present",
            file=sys.stderr,
        )
        sys.exit(1)
    home_a, label_a, home_b, label_b, expect = sys.argv[1:]
    expect_substrings = [s for s in expect.split("\n") if s.strip()]
    sa = summarize(Path(home_a), label_a, expect_substrings)
    sb = summarize(Path(home_b), label_b, expect_substrings)

    print(
        "| arm | requests | mean prompt tok/request | peak prompt tok/request | "
        "prefill cache hit % | total prompt tok | task success |"
    )
    print(
        "|-----|----------|--------------------------|--------------------------|"
        "----------------------|-------------------|--------------|"
    )
    for s in (sa, sb):
        print(
            f"| {s['label']} | {s['requests']} | {s['mean_prompt_tokens']} | "
            f"{s['peak_prompt_tokens']} | {fmt_cache(s)} | {s['total_prompt_tokens']} | "
            f"{s['task_success']} |"
        )
    print()
    print(f"fact hits — {label_a}: {fmt_fact_hits(sa['fact_hits'])}  |  {label_b}: {fmt_fact_hits(sb['fact_hits'])}")
    print()
    print(f"context tool calls — {label_a}: {sa['context_tool_calls']}  |  {label_b}: {sb['context_tool_calls']}")
    print(f"  {label_a}: {fmt_context_calls(sa)}")
    print(f"  {label_b}: {fmt_context_calls(sb)}")
    print()
    print(f"size notices fired — {label_a}: {sa['size_notices']}  |  {label_b}: {sb['size_notices']}")
    print()
    print("raw:", json.dumps({"a": sa, "b": sb}))


if __name__ == "__main__":
    main()
