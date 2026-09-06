"""Sequential installed-nanobot check; run in tmux, with builds stopped.

Uses an isolated session in the normal session store. Records exact native
input counts, actual saved answers, and content-free server telemetry.
"""
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import time
import tomllib
import urllib.request
from pathlib import Path

from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "context-recovery"))
from memory_probe import memory_sample


# The production CLI is singleton-scoped. Never let a test launch terminate
# an interactive agent: refuse to start while its recorded PID is live.
pid_file = Path.home() / ".nanobot/agent.pid"
if pid_file.exists():
    try:
        os.kill(int(pid_file.read_text().strip()), 0)
    except (ProcessLookupError, ValueError):
        pass
    else:
        raise SystemExit("An interactive nanobot is running; do not replace it with a test.")

root = Path(sys.argv[1]).resolve()
root.mkdir(parents=True, exist_ok=False)
session = "cli:kiss-long-" + str(int(time.time()))
binary = Path.home() / ".local/bin/nanobot"
config = tomllib.loads((Path.home() / ".config/higgs/config.toml").read_text())
tokenizer = Tokenizer.from_file(str(Path.home() / ".cache/lm-studio/models/EschaLabs/Qwen3.6-35B-A3B-Escha-W2/tokenizer.json"))
pid = int(subprocess.check_output(["lsof", "-tiTCP:9000", "-sTCP:LISTEN"], text=True).strip().splitlines()[0])
metadata = {"higgs_pid": pid, "session": session, "nanobot_sha256": hashlib.sha256(binary.read_bytes()).hexdigest()}
(root / "metadata.json").write_text(json.dumps(metadata, indent=2))


def telemetry():
    request = urllib.request.Request("http://127.0.0.1:9000/metrics", headers={"Authorization": "Bearer " + config["server"]["api_key"]})
    with urllib.request.urlopen(request, timeout=10) as response:
        data = json.load(response)
    return {"time": time.time(), "memory": memory_sample(pid), "capacity": data["capacity"], "cache": data["cache"]}


def saved_answer():
    with sqlite3.connect("file:" + str(Path.home() / ".nanobot/sessions.db") + "?mode=ro", uri=True) as db:
        row = db.execute("SELECT m.content,m.id FROM messages m JOIN sessions s ON s.id=m.session_id WHERE s.session_key=? AND m.role='assistant' AND m.content IS NOT NULL AND m.content!='' ORDER BY m.id DESC LIMIT 1", (session,)).fetchone()
        counts = dict(db.execute("SELECT json_extract(e.payload_json,'$.purpose'),count(*) FROM session_events e JOIN sessions s ON s.id=e.session_id WHERE s.session_key=? AND e.event_kind='model_request' GROUP BY 1", (session,)).fetchall())
    return (row[0] if row else ""), counts, (row[1] if row else 0)


def turn(number, prompt, expected):
    (root / f"{number:02d}-prompt.txt").write_text(prompt)
    _, _, prior_answer_id = saved_answer()
    start = time.time()
    with (root / f"{number:02d}.log").open("w") as log, (root / f"{number:02d}-telemetry.jsonl").open("w") as samples:
        process = subprocess.Popen([str(binary), "agent", "--local", "--session", session, "-m", prompt], stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            samples.write(json.dumps(telemetry()) + "\n")
            samples.flush()
            if time.time() - start > 600:
                process.terminate()
                process.wait(timeout=30)
                raise RuntimeError(f"turn {number} timed out")
            time.sleep(5)
    answer, counts, answer_id = saved_answer()
    passed = process.returncode == 0 and answer_id > prior_answer_id and all(literal in answer for literal in expected)
    result = {"turn": number, "seconds": time.time() - start, "exit": process.returncode, "user_tokens": len(tokenizer.encode(prompt).ids), "answer": answer, "expected": expected, "passed": passed, "request_purposes": counts, "after": telemetry()}
    (root / f"{number:02d}-result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("turn", "seconds", "user_tokens", "passed", "answer", "request_purposes")}), flush=True)
    if not passed:
        raise RuntimeError(f"turn {number} did not retain the expected task/fact")


for batch in range(1, 9):
    header = f"This is batch {batch} of an ongoing archive review. Keep the archive facts in conversation; no tools are needed. After reading, reply only BATCH_{batch}_OK. Later I will ask for exact archive keys.\n"
    if batch == 1:
        header += "The launch key is LANTERN-4729-QZ. The project is called Alder.\n"
    if batch == 5:
        header += "The recovery key is COPPER-8163-VX. The project remains Alder.\n"
    lines = []
    for row in range(1000):
        lines.append(f"Archive batch {batch}, record {row}: reviewed inventory, access log and maintenance schedule; no change to project ownership or launch authorization.\n")
        if len(tokenizer.encode(header + "".join(lines)).ids) >= 3900:
            break
    turn(batch, header + "".join(lines), [f"BATCH_{batch}_OK"])

turn(9, "Now complete the archive review. Give the exact launch key from batch 1, recovery key from batch 5, and project name. Use the original facts; do not invent replacements or ask me for a new task.", ["LANTERN-4729-QZ", "COPPER-8163-VX", "Alder"])
(root / "COMPLETE").write_text("Nine turns passed exact response/fact checks. Inspect prompt traces and compaction counts before drawing endurance conclusions.\n")
