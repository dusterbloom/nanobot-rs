#!/usr/bin/env python3
"""Run the predeclared paired trials sequentially; launch this driver in tmux."""
import os
from pathlib import Path
import subprocess
import sys
import json
import shlex
import time
import urllib.request
import urllib.error


def start_higgs(root, label, min_prompt=16384):
    higgs_pane = os.environ.get('HIGGS_EVAL_TMUX_PANE') or subprocess.check_output(['tmux', 'display-message', '-p', '-t', 'recovery-higgs:0.0', '#{pane_id}'], text=True).strip()
    request = urllib.request.Request('http://127.0.0.1:9000/v1/capacity?model=escha-35b-a3b', headers={'Authorization': 'Bearer higgs'})
    subprocess.run(['tmux', 'send-keys', '-t', higgs_pane, 'C-c'], check=True)
    for _ in range(30):
        try:
            urllib.request.urlopen(request, timeout=2).close()
        except (OSError, urllib.error.URLError):
            break
        time.sleep(1)
    else:
        raise SystemExit('Higgs did not stop; refusing overlapping servers')
    server_log = root / f'{label}-higgs.log'
    config = os.environ.get('HIGGS_EVAL_CONFIG', '/private/tmp/recovery-higgs-corrected.toml')
    command = 'cd /private/tmp/higgs-recovery && HIGGS_ESCHA_NATIVE=1 RUST_LOG=info ./higgs-hardened serve -c ' + shlex.quote(config) + ' --mlx-profile throughput > ' + shlex.quote(str(server_log)) + ' 2>&1'
    subprocess.run(['tmux', 'send-keys', '-t', higgs_pane, command, 'Enter'], check=True)
    for _ in range(90):
        try:
            with urllib.request.urlopen(request, timeout=2) as response:
                envelope = json.load(response)
            if envelope['maxPromptTokens'] >= min_prompt:
                (root / f'{label}-capacity.json').write_text(json.dumps(envelope, indent=2))
                return envelope
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(1)
    else:
        raise SystemExit('Fresh Higgs lacks required prompt headroom')


if __name__ == "__main__":
    root = Path(sys.argv[1]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    cases = ['changed_requirement', 'failed_action', 'completed_action', 'buried_identifier', 'superseded_state']
    for case, arm in [(c, a) for c in cases for a in ['A', 'B']] + [('failed_action', 'C'), ('buried_identifier', 'C')]:
        directory = root / f'{case}-{arm}'
        if (directory / 'trial.json').exists():
            print(f'Already completed: {directory.name}', flush=True)
            continue
        start_higgs(root, f'{case}-{arm}')
        env = dict(os.environ, RECOVERY_EVAL_OUT=str(root), RECOVERY_EVAL_CASE=case, RECOVERY_EVAL_ARMS=arm)
        print(f'Running {directory.name}', flush=True)
        with (root / f'{case}-{arm}.log').open('w') as log:
            result = subprocess.run(['target/release/deps/nanobot-67848bc0f9b02956', 'recovery_eval_live', '--ignored', '--nocapture', '--test-threads=1'], env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1200)
        if result.returncode:
            raise SystemExit(f'Harness failed; inspect {log.name}')
        subprocess.run([sys.executable, 'experiments/context-recovery/verify.py', str(root)], check=True)
        subprocess.run([sys.executable, 'experiments/context-recovery/report.py', str(root)], check=True)
    (root / 'COMPLETE').write_text('All 12 predeclared trials ran; inspect summary.json for behavioral scores.\n')
