#!/usr/bin/env python3
"""Memory probe: can bonsai-27b answer questions about the user's past
using ONLY agentic search over raw sessions.db (FTS5)?

Two tools, no curation, no injection. This is deterministic eval #1 for
the proposed memory architecture. stdlib only.
"""
import json
import re
import sqlite3
import sys
import time
import urllib.request

DB = "/Users/peppi/.nanobot/sessions.db"
API = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "bonsai-27b"
MAX_HOPS = 6
OUT = "/tmp/memory_probe_results_v2.jsonl"

# (question, match_mode, expected_substrings)
PROBES = [
    ("What timezone does the user live in?", "any", ["rome", "cet", "italy"]),
    ("Which languages does the user speak besides English?", "all", ["italian", "german", "spanish"]),
    ("The user works on two indie inference engines. What are their names?", "all", ["higgs", "lucebox"]),
    ("Which GPU/machine does the engine 'lucebox' run on?", "any", ["3090"]),
    ("What name was the agent called in early March, before it was renamed?", "any", ["ava"]),
    ("Which Node.js version command does the user always want used?", "any", ["lts"]),
    ("Which voice should the `tss` command always use?", "any", ["siri"]),
    ("During the project naming discussion in mid-March, which name did the user say 'clicks somehow'?", "any", ["qlick"]),
    ("In which directory does the user's inference-optimization work live?", "any", ["dev/higgs"]),
    ("In April the user asked to scan the hard drive for files related to what service?", "any", ["vercel"]),
    ("Which latency metric did the user want to minimize in early June?", "any", ["ttft", "time to first token"]),
    ("Which Italian news outlet's RSS feed did the user switch the news skill to in mid-March?", "any", ["ansa"]),
    ("Which town does the user ask weather forecasts for?", "any", ["serramanna"]),
    ("Which music genre does the user ask the webradio skill to play?", "any", ["jazz"]),
    ("What is the user's name?", "any", ["peppi"]),
    ("Which fine-tuning technique did the user implement in March to give the agent parametric learning?", "any", ["lora"]),
    ("Which three news outlets did the user ask for a combined global news summary from in March?", "all", ["bbc", "cnn", "jazeera"]),
    ("Which skill does the agent use to fetch news summaries?", "any", ["newsreader"]),
    ("Which tool was supposed to transcribe the user's speech but kept failing in March?", "any", ["listen", "whisper"]),
    ("What is the name of the user's personal AI assistant project written in Rust?", "any", ["nanobot"]),
]

TOOLS = [
    {"type": "function", "function": {
        "name": "memory_search",
        "description": "Full-text search over all past conversations (March-July 2026). Words are ANDed and stemmed. Returns dated snippets with session ids.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "search words, e.g. 'timezone rome'"},
            "after": {"type": "string", "description": "optional ISO date lower bound, e.g. 2026-03-01"},
            "before": {"type": "string", "description": "optional ISO date upper bound"},
            "role": {"type": "string", "enum": ["user", "assistant"],
                     "description": "optional: restrict to one speaker. Use 'user' for facts about the user — their own words are ground truth."}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "read_session",
        "description": "Read messages of one session, optionally centered on a message id from memory_search results.",
        "parameters": {"type": "object", "properties": {
            "session_id": {"type": "string"},
            "around_id": {"type": "integer"}},
            "required": ["session_id"]}}},
]

SYSTEM = (
    "You answer questions about the user's past conversations with their assistant. "
    "You have tools that search the raw conversation history. ALWAYS search before answering; "
    "try different keywords if a search returns nothing. Rules: "
    "(1) The user's own messages are ground truth. Assistant messages may contain errors, "
    "failed attempts, or hallucinations — verify assistant claims against user turns "
    "(use role='user'). "
    "(2) If a snippet references something indirectly ('the three sources', 'that file'), "
    "call read_session around that message to resolve it before answering. "
    "(3) Search with words the user would have actually typed, not abstract descriptors. "
    "(4) Cross-check with a second search before concluding. "
    "Answer concisely with the specific fact."
)


def strip_think(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    return text.split("</think>")[-1].strip()


def db():
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def memory_search(query, after=None, before=None, role=None):
    words = re.findall(r"[a-zA-Z0-9_.\-/]+", query)
    if not words:
        return "error: empty query"
    conn = db()

    def run(match):
        sql = ("SELECT m.id, m.session_id, substr(m.timestamp,1,10) AS d, m.role, "
               "snippet(messages_fts, 0, '>>', '<<', '…', 40) AS snip "
               "FROM messages_fts f JOIN messages m ON m.id = f.rowid "
               "WHERE messages_fts MATCH ? AND m.role IN ('user','assistant') "
               "AND m.synthetic = 0 AND m.content NOT LIKE '[VERBATIM%'")
        args = [match]
        if role in ("user", "assistant"):
            sql += " AND m.role = ?"
            args.append(role)
        if after:
            sql += " AND m.timestamp >= ?"
            args.append(after)
        if before:
            sql += " AND m.timestamp <= ?"
            args.append(before)
        sql += " ORDER BY rank LIMIT 10"
        try:
            return conn.execute(sql, args).fetchall()
        except sqlite3.OperationalError as e:
            return f"error: {e}"

    quoted = " ".join(f'"{w}"' for w in words)
    rows = run(quoted)
    if isinstance(rows, str):
        return rows
    if not rows and len(words) > 1:  # fallback: OR the words for recall
        rows = run(" OR ".join(f'"{w}"' for w in words))
        if isinstance(rows, str):
            return rows
    if not rows:
        return "no results — try different keywords"
    return "\n".join(f"[msg {r['id']} | {r['d']} | {r['role']} | session {r['session_id']}] {r['snip']}"
                     for r in rows)


def read_session(session_id, around_id=None):
    conn = db()
    rows = conn.execute(
        "SELECT id, role, substr(COALESCE(content,''),1,300) AS c FROM messages "
        "WHERE session_id = ? AND role IN ('user','assistant') AND synthetic = 0 "
        "ORDER BY id", (session_id,)).fetchall()
    if not rows:
        return "no such session"
    if around_id:
        idx = next((i for i, r in enumerate(rows) if r["id"] == around_id), len(rows) // 2)
        rows = rows[max(0, idx - 8):idx + 8]
    else:
        rows = rows[:24]
    return "\n".join(f"[{r['id']} {r['role']}] {r['c']}" for r in rows)


def chat(messages, with_tools=True):
    body = {"model": MODEL, "temperature": 0, "max_tokens": 400, "messages": messages}
    if with_tools:
        body["tools"] = TOOLS
    req = urllib.request.Request(API, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    last = None
    for attempt in range(4):  # survive server restarts mid-run
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read())["choices"][0]["message"]
        except Exception as e:
            last = e
            time.sleep(15 * (attempt + 1))
    raise last


def run_probe(q):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}]
    trail = []
    for hop in range(MAX_HOPS):
        msg = chat(messages)
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        if not calls:
            answer = strip_think(msg.get("content") or "")
            if not answer:  # degenerate empty/think-only output: one retry
                messages.append({"role": "user",
                                 "content": "Answer now with your best conclusion."})
                msg = chat(messages, with_tools=False)
                answer = strip_think(msg.get("content") or "")
            return answer, trail
        for c in calls:
            fn = c["function"]["name"]
            try:
                args = json.loads(c["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            if fn == "memory_search":
                out = memory_search(args.get("query", ""), args.get("after"),
                                    args.get("before"), args.get("role"))
            elif fn == "read_session":
                out = read_session(args.get("session_id", ""), args.get("around_id"))
            else:
                out = "unknown tool"
            trail.append({"tool": fn, "args": args})
            messages.append({"role": "tool", "tool_call_id": c["id"], "name": fn,
                             "content": str(out)[:4000]})
    messages.append({"role": "user", "content": "No more searches. Answer now with your best conclusion."})
    msg = chat(messages, with_tools=False)
    return strip_think(msg.get("content") or ""), trail


def score(answer, mode, expected):
    a = answer.lower()
    hits = [e for e in expected if e in a]
    return len(hits) == len(expected) if mode == "all" else len(hits) > 0


def main():
    results = []
    with open(OUT, "w") as f:
        for i, (q, mode, exp) in enumerate(PROBES, 1):
            t0 = time.time()
            try:
                answer, trail = run_probe(q)
            except Exception as e:  # server hiccup: record, keep going
                answer, trail = f"ERROR: {e}", []
            ok = score(answer, mode, exp)
            dt = time.time() - t0
            rec = {"n": i, "q": q, "answer": answer.strip(), "expect": exp,
                   "pass": ok, "hops": len(trail), "trail": trail, "secs": round(dt, 1)}
            results.append(rec)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"[{i:2}/20] {'PASS' if ok else 'FAIL'} {dt:5.1f}s hops={len(trail)} {q[:60]}",
                  flush=True)
    n = sum(r["pass"] for r in results)
    print(f"\nSCORE: {n}/20  (avg {sum(r['secs'] for r in results)/20:.1f}s, "
          f"avg hops {sum(r['hops'] for r in results)/20:.1f})")


if __name__ == "__main__":
    sys.exit(main())
