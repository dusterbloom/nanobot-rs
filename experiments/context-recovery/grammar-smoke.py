#!/usr/bin/env python3
"""Exercise the live Higgs constrained tool-call wire contract (no tools executed)."""
import json
from pathlib import Path
import sys
import time
import urllib.request

out = Path(sys.argv[1]).resolve()
out.mkdir(parents=True, exist_ok=False)
tool = {'type': 'function', 'function': {'name': 'notes', 'description': 'Save checkpoint',
    'parameters': {'type': 'object', 'properties': {
        'op': {'type': 'string', 'enum': ['write']},
        'content': {'type': 'string', 'enum': ['revision=4; receipt=rcpt_Q8n3_L04']}},
        'required': ['op', 'content'], 'additionalProperties': False}}}
results = []
for streaming, choice in [(False, "auto"), (False, "required"), (True, "required")]:
    request = {'model': 'escha-35b-a3b', 'messages': [{'role': 'user', 'content':
        'Do not call any tools. Reply with exactly UNCONSTRAINED_SENTINEL and nothing else.'}],
        'tools': [tool], 'tool_choice': choice, 'max_tokens': 160, 'temperature': 0,
        'enable_thinking': choice == 'required', 'stream': streaming}
    label = 'auto' if choice == 'auto' else ('stream' if streaming else 'blocking')
    (out / (label + '-request.json')).write_text(json.dumps(request, indent=2))
    started = time.monotonic()
    req = urllib.request.Request('http://127.0.0.1:9000/v1/chat/completions',
        data=json.dumps(request).encode(), headers={'Authorization': 'Bearer higgs', 'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=180) as response:
        raw = response.read().decode()
    (out / (label + '-response.txt')).write_text(raw)
    if streaming:
        calls = {}
        content = ''
        for line in raw.splitlines():
            if not line.startswith('data: ') or line == 'data: [DONE]':
                continue
            event = json.loads(line[6:])
            for choice in event.get('choices', []):
                delta = choice.get('delta', {})
                content += delta.get('content') or ''
                for call in delta.get('tool_calls', []):
                    saved = calls.setdefault(call['index'], {'name': '', 'arguments': ''})
                    function = call.get('function', {})
                    saved['name'] += function.get('name') or ''
                    saved['arguments'] += function.get('arguments') or ''
        calls = list(calls.values())
    else:
        message = json.loads(raw)['choices'][0]['message']
        content = message.get('content') or ''
        calls = [call['function'] for call in message.get('tool_calls', [])]
    passed = (not content.strip() and len(calls) == 1 and calls[0]['name'] == 'notes'
              and json.loads(calls[0]['arguments']) == {'op': 'write', 'content': 'revision=4; receipt=rcpt_Q8n3_L04'})
    if choice == 'auto':
        passed = not calls and content.strip() == 'UNCONSTRAINED_SENTINEL'
    result = {'case': label, 'seconds': time.monotonic() - started, 'pass': passed,
              'calls': calls, 'content': content}
    results.append(result)
    (out / 'summary.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(result), flush=True)
    if not passed:
        raise SystemExit('Live required-tool wire contract failed')
