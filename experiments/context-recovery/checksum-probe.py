#!/usr/bin/env python3
"""Frozen decision ablations; no returned tools are executed."""
import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time
import urllib.request

root = Path(sys.argv[1]); root.mkdir(parents=True, exist_ok=False)
evidence = json.loads(Path('experiments/context-recovery/feasibility-first-error.json').read_text())
db = sqlite3.connect('file:' + evidence['source_db'] + '?mode=ro', uri=True)
revision = int(sys.argv[2]) if len(sys.argv) > 2 else 8
if revision != 8:
    turns = json.loads(Path('experiments/context-recovery/endurance-feasibility-scheduled/B/turns.json').read_text())
    evidence['session_id'] = turns[revision]['session_id']
    events = [json.loads(row[0]) for row in db.execute('select payload_json from session_events where session_id=? order by id', (evidence['session_id'],))]
    evidence['requests'][1]['digest'] = [e for e in events if e['kind'] == 'model_request'][1]['request_digest']
record = json.loads(db.execute('select content from session_replay_artifacts where session_id=? and digest=?', (evidence['session_id'], evidence['requests'][1]['digest'])).fetchone()[0])
# Match local blocking OpenAICompatProvider translation, with retained route
# removed for every ablation. This is a cold decision probe, not session replay.
base = {k: copy.deepcopy(record[k]) for k in ('model', 'messages', 'tools', 'max_tokens', 'tool_choice')}
base.update(stream=False, chat_template_kwargs={'enable_thinking': False}, repeat_penalty=1.1, max_prompt_tokens=10240)
for m in base['messages']:
    for key in list(m):
        if key.startswith('_nanobot_'): del m[key]
# Provider normalizes all object schemas, including nested objects.
def normalize(value):
    if isinstance(value, dict):
        if value.get('type') == 'object': value.setdefault('required', [])
        for child in value.values(): normalize(child)
    elif isinstance(value, list):
        for child in value: normalize(child)
normalize(base['tools'])
expected = json.loads(Path('experiments/context-recovery/endurance-feasibility-scheduled/B/fixture.json').read_text())['expected'][revision]
results = []
order = ['original', 'neutral_penalty', 'field_list', 'copy_instruction']
if revision == 8: order += list(reversed(order))
for i, condition in enumerate(order):
    q = copy.deepcopy(base)
    if condition == 'neutral_penalty': q['repeat_penalty'] = 1.0
    elif condition == 'field_list':
        receipt = json.loads(q['messages'][-1]['content'])
        import re
        match = re.search(r'checksum[ =]([A-Za-z0-9-]+)', receipt['content'])
        assert match, 'checkpoint must already contain checksum; never inject expected-state facts'
        value = match[1]
        receipt['content'] = re.sub(r'authoritative_update applied with checksum[ =][A-Za-z0-9-]+[.,]', 'authoritative_update applied.', receipt['content'])
        receipt['content'] = receipt['content'].replace('Snapshot submitted: ', 'Snapshot submitted: checksum=' + value + ', ')
        q['messages'][-1]['content'] = json.dumps(receipt, separators=(',', ':'))
    elif condition == 'copy_instruction':
        q['messages'][0]['content'] += '\nCopy opaque identifiers byte-for-byte from evidence. If authoritative_update does not change a checksum, copy the checksum from notes exactly; do not generate a replacement.'
    label = f'{i}-{condition}'
    (root/(label+'-request.json')).write_text(json.dumps(q,indent=2))
    started=time.monotonic()
    request=urllib.request.Request('http://127.0.0.1:9000/v1/chat/completions', data=json.dumps(q).encode(), headers={'Authorization':'Bearer higgs','Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=180) as response: raw=response.read()
    (root/(label+'-response.json')).write_bytes(raw)
    reply=json.loads(raw);message=reply['choices'][0]['message'];calls=message.get('tool_calls',[]);actual=None
    for call in calls:
        if call['function']['name']=='submit_result': actual=json.loads(call['function']['arguments']).get('result')
    row={'label':label,'condition':condition,'seconds':time.monotonic()-started,'actual':actual,'exact':actual==expected,'usage':reply.get('usage'),'finish_reason':reply['choices'][0]['finish_reason'],'message':message}
    results.append(row);(root/'summary.json').write_text(json.dumps(results,indent=2));print(json.dumps({k:v for k,v in row.items() if k!='message'}),flush=True)
(root/'provenance.json').write_text(json.dumps({'revision':revision,'record_digest':evidence['requests'][1]['digest'],'expected':expected,'higgs_sha256':hashlib.sha256(Path('/Users/peppi/.local/bin/higgs').read_bytes()).hexdigest(),'scope':'Stateless frozen decision; no tool execution; server default temperature 0.0; local provider repetition control reproduced.'},indent=2))
