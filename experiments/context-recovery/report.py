#!/usr/bin/env python3
"""Summarize completed live trials using SQLite's recorded model usage."""
import json
import sqlite3
import sys
from pathlib import Path

root = Path(sys.argv[1])
trials = []
for trial_file in sorted(root.glob('*/trial.json')):
    trial = json.loads(trial_file.read_text())
    conn = sqlite3.connect(f'file:{trial_file.parent / "sessions.db"}?mode=ro', uri=True)
    artifacts = {(sid, digest): content for sid, digest, content in conn.execute(
        'select session_id,digest,content from session_replay_artifacts')}
    calls = {}
    usage = {'preparation': {}, 'recovery': {}}
    requests = {'preparation': 0, 'recovery': 0}
    failed = []
    tool_failures = 0
    for sid, kind, raw in conn.execute(
            'select session_id,event_kind,payload_json from session_events order by id'):
        event = json.loads(raw)
        stage = ('recovery' if sid == trial['recovery']['session_id']
                 and event.get('purpose') != 'compaction' else 'preparation')
        if kind == 'model_request':
            request = json.loads(artifacts[sid, event['request_digest']])
            # Compaction shares the original session with A's recovery.
            stage = 'preparation' if event['purpose'] == 'compaction' else stage
            calls[event['call_id']] = stage
            requests[stage] += 1
        elif kind == 'model_response':
            stage = calls[event['call_id']]
            response = json.loads(artifacts[sid, event['response_digest']])
            for key, value in response.get('usage', {}).items():
                usage[stage][key] = usage[stage].get(key, 0) + value
        elif kind == 'tool_execute' and not event['ok']:
            tool_failures += 1
        elif kind == 'model_failed':
            failed.append(bytes(artifacts[sid, event['error_digest']]).decode())
    actions_file = trial_file.parent / 'actions.jsonl'
    actions = [json.loads(line) for line in actions_file.read_text().splitlines()] if actions_file.exists() else []
    with (trial_file.parent / 'replay.txt').open() as replay:
        replay_header = replay.read(128)
    trial['replay_availability'] = 'Partial' if 'availability: Partial' in replay_header else 'Exact' if 'availability: Exact' in replay_header else 'Unknown'
    trial['usage'] = usage
    trial['requests'] = requests
    trial['provider_failures'] = failed
    trial['tool_failures'] = tool_failures
    trial['tool_sequence'] = [action['tool'] for action in actions]
    trial['submit_count'] = trial['tool_sequence'].count('submit_result')
    trial['strict_pass'] = (trial['pass'] and trial['submit_count'] == 1
                            and trial['recovery']['outcome'] == 'Finished')
    trials.append(trial)
(root / 'summary.json').write_text(json.dumps(trials, indent=2) + '\n')
print('| Case | Arm | Pass | Prep s | Recovery s | Total s | Input tokens | Output tokens |')
print('|---|---|---|---:|---:|---:|---:|---:|')
for t in trials:
    inp = sum(u.get('prompt_tokens', 0) for u in t['usage'].values())
    out = sum(u.get('completion_tokens', 0) for u in t['usage'].values())
    print(f"| {t['case']} | {t['arm']} | {t['strict_pass']} | {t['preparation']['seconds']:.1f} | "
          f"{t['recovery']['seconds']:.1f} | {t['total_seconds']:.1f} | {inp} | {out} |")
