#!/usr/bin/env python3
"""Replay one frozen post-reset delta, OFF/ON/OFF/ON, with fresh Higgs profiles."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from run import start_higgs

root = Path(sys.argv[1]).resolve()
root.mkdir(parents=True, exist_ok=False)
config = Path('experiments/context-recovery/higgs-endurance.toml').read_text()
binary = Path('target/release/deps/nanobot-67848bc0f9b02956').resolve()
(root / 'provenance.json').write_text(json.dumps({
    'harness_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
    'higgs_sha256': hashlib.sha256(Path('/private/tmp/higgs-recovery/higgs-hardened').read_bytes()).hexdigest(),
    'order': ['off1', 'on1', 'off2', 'on2'],
    'purpose': 'Recovery ability at frozen revision 3, not autonomous reset timing',
}, indent=2))
results = []
for label, enabled in [('off1', '0'), ('on1', '1'), ('off2', '0'), ('on2', '1')]:
    server = root / ('server-' + label)
    server.mkdir()
    config_path = server / 'higgs.toml'
    config_path.write_text(config)
    os.environ['HIGGS_EVAL_CONFIG'] = str(config_path)
    capacity = start_higgs(root, label, 12288)
    output = root / label
    env = dict(os.environ, HIGGS_EVAL_URL='http://127.0.0.1:9000/v1',
               ENDURANCE_RESET_HANDOFF=enabled, ENDURANCE_REPLAY_OUT=str(output))
    print('START', label, capacity['bootId'], flush=True)
    with (root / (label + '.log')).open('w') as log:
        completed = subprocess.run([str(binary), 'endurance_reset_handoff_replay_live',
                                    '--ignored', '--nocapture', '--test-threads=1'],
                                   env=env, stdout=log, stderr=subprocess.STDOUT, timeout=300)
    if completed.returncode:
        raise SystemExit(f'{label}: harness exit {completed.returncode}; inspect log')
    result = json.loads((output / 'replay.json').read_text())
    result['label'] = label
    result['boot_id'] = capacity['bootId']
    results.append(result)
    (root / 'summary.json').write_text(json.dumps(results, indent=2))
    print('RESULT', label, 'pass=', result['pass'], 'actual=', result['actual'], flush=True)
(root / 'COMPLETE').write_text('Four frozen replays completed; inspect behavioral scores.\n')
