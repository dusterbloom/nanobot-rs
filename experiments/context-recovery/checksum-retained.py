#!/usr/bin/env python3
"""Reconstruct notes-read → retained continuation; never execute returned tools."""
import copy
import json
from pathlib import Path
import sys
import time
import urllib.request

root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=False)
results=[]
for revision,source in [(8,'endurance-checksum-probe'),(14,'endurance-checksum-rev14')]:
    expected=json.loads(Path('experiments/context-recovery/endurance-feasibility-scheduled/B/fixture.json').read_text())['expected'][revision]
    for condition,index in [('original',0),('copy_instruction',3)]:
        label=f'{revision}-{condition}'
        q=json.loads(Path('experiments/context-recovery',source,f'{index}-{condition}-request.json').read_text())
        q.update(session_id=time.time_ns()//1000,session_cache_policy='best_effort')
        seed=copy.deepcopy(q);seed['messages']=seed['messages'][:2]
        def send(body,stage):
            (root/f'{label}-{stage}-request.json').write_text(json.dumps(body,indent=2))
            request=urllib.request.Request('http://127.0.0.1:9000/v1/chat/completions',data=json.dumps(body).encode(),headers={'Authorization':'Bearer higgs','Content-Type':'application/json'})
            with urllib.request.urlopen(request,timeout=180) as r:response=json.load(r)
            (root/f'{label}-{stage}-response.json').write_text(json.dumps(response,indent=2));return response
        started=time.monotonic();first=send(seed,'seed');message=first['choices'][0]['message'];calls=message.get('tool_calls',[])
        assert len(calls)==1 and calls[0]['function']['name']=='notes' and json.loads(calls[0]['function']['arguments'])=={'op':'read'}, 'seed did not request the recorded notes'
        receipt=copy.deepcopy(q['messages'][-1]);receipt['tool_call_id']=calls[0]['id']
        q['messages']=seed['messages']+[{k:message[k] for k in ('role','content','tool_calls') if k in message},receipt]
        response=send(q,'decision');m=response['choices'][0]['message'];calls=m.get('tool_calls',[]);actual=None
        if len(calls)==1 and calls[0]['function']['name']=='submit_result':actual=json.loads(calls[0]['function']['arguments']).get('result')
        row={'revision':revision,'condition':condition,'seconds':time.monotonic()-started,'exact':actual==expected,'actual':actual,'usage':response.get('usage'),'session_id':q['session_id']}
        results.append(row);(root/'summary.json').write_text(json.dumps(results,indent=2));print(json.dumps(row),flush=True)
