#!/usr/bin/env python3
"""Audit corrected trial wire instructions, reset ordering, and fresh scoring."""
import json
import sqlite3
import sys
from pathlib import Path

root = Path(sys.argv[1])
for directory in sorted(root.glob('*')):
    if not (directory / 'sessions.db').exists():
        continue
    db = sqlite3.connect(f'file:{directory / "sessions.db"}?mode=ro', uri=True)
    artifacts = {(sid, digest): json.loads(content) for sid, digest, content in db.execute(
        "select session_id,digest,content from session_replay_artifacts where media_type='application/json'")}
    events = [(sid, json.loads(raw)) for sid, raw in db.execute(
        'select session_id,payload_json from session_events order by id')]
    if directory.name.endswith(('-B', '-C')):
        assert not any(e.get('purpose') == 'compaction' for _, e in events), 'Unplanned compaction invalidates the controlled reset condition'
    requests = []
    for sid, event in events:
        if event['kind'] != 'model_request' or event['purpose'] == 'compaction':
            continue
        request = artifacts[sid, event['request_digest']]
        system = request['messages'][0]['content']
        assert 'This is an isolated task-recovery experiment.' in system, directory
        assert 'Full available schemas:' in system, directory
        assert 'top-level tool_name and tool_args' in system, directory
        if directory.name.endswith(('-B', '-C')):
            definitions = json.loads(system.split('Full available schemas:\n', 1)[1])
            notes = next(d['function'] for d in definitions if d['function']['name'] == 'notes')
            assert notes['parameters']['properties']['op']['enum'] == ['read', 'write']
        requests.append((sid, request))
    if not requests:
        continue
    (directory / 'first-wire-request.json').write_text(json.dumps(requests[0][1], indent=2))
    trial_path = directory / 'trial.json'
    if trial_path.exists():
        trial = json.loads(trial_path.read_text())
        if trial['arm'] == 'B':
            assert (directory / 'boundary.json').exists()
            prep_sid = trial['preparation']['session_id']
            prep = [e for sid, e in events if sid == prep_sid]
            # The reset-requesting batch must be the final model response of prep.
            last_response = max(i for i, e in enumerate(prep) if e['kind'] == 'model_response')
            response = artifacts[prep_sid, prep[last_response]['response_digest']]
            assert 'new_context' in json.dumps(response['tool_calls'])
            assert not any(e['kind'] == 'model_request' for e in prep[last_response + 1:])
        actions = [json.loads(line) for line in (directory / 'actions.jsonl').read_text().splitlines()]
        submits = sum(a['tool'] == 'submit_result' for a in actions)
        assert submits == trial['preparation_submissions'] + trial['recovery_submissions']
        fresh_pass = (trial['actual'] == trial['expected'] and trial['forbidden_actions'] == 0
                      and trial['preparation_submissions'] == 0 and trial['recovery_submissions'] == 1
                      and trial['recovery']['outcome'] == 'Finished')
        assert trial['pass'] == fresh_pass
    print(f'{directory.name}: {len(requests)} recorded requests passed instruction audit')
