"""Small sequential EvalScope pilot; invoke inside tmux after endurance finishes."""
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'context-recovery'))
from memory_probe import memory_sample

root = Path(sys.argv[1]).resolve()
root.mkdir(parents=True, exist_ok=False)
inputs = Path(__file__).resolve().parent
pid = int(subprocess.check_output(['lsof', '-tiTCP:9000', '-sTCP:LISTEN'], text=True).strip())
for length in (1024, 4096, 8192):
    output = root / str(length)
    command = ['/private/tmp/local-bench-venv/bin/evalscope', 'perf',
               '--url', 'http://127.0.0.1:9000/v1/chat/completions', '--api', 'openai',
               '--api-key', 'higgs', '--model', 'escha-35b-a3b', '--parallel', '1',
               '--number', '3', '--warmup-num', '0', '--stream', '--max-tokens', '256',
               '--prompt', '@' + str(inputs / f'prompt-{length}.txt'),
               '--query-template', '@' + str(inputs / 'perf-query.json'),
               '--outputs-dir', str(output), '--no-timestamp',
               '--read-timeout', '180', '--total-timeout', '300']
    (root / f'{length}-command.json').write_text(json.dumps(command, indent=2) + '\n')
    with (root / f'{length}.log').open('w') as log, (root / f'{length}-memory.jsonl').open('w') as memory:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            try:
                sample = {'time': time.time(), **memory_sample(pid)}
            except Exception as error:
                sample = {'time': time.time(), 'error': str(error)}
            memory.write(json.dumps(sample) + '\n')
            memory.flush()
            time.sleep(5)
        (root / f'{length}-exit.txt').write_text(str(process.returncode) + '\n')
        if process.returncode:
            raise SystemExit(f'EvalScope failed at {length}; inspect {log.name}')
(root / 'COMPLETE').write_text('Pilot requests completed; inspect errors and actual token counts before interpreting performance.\n')
