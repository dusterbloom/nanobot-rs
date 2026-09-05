#!/usr/bin/env python3
"""Content-free 1 Hz OS/MLX capture and spaced cold requests; run in tmux."""
import argparse
import json
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


def api(path, payload=None, timeout=600):
    request = urllib.request.Request(
        'http://127.0.0.1:9000' + path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={'Authorization': 'Bearer higgs', 'Content-Type': 'application/json'},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--count', type=int, default=3)
    parser.add_argument('--interval', type=float, default=165)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    stop = threading.Event()

    def monitor():
        with (args.output / 'telemetry.jsonl').open('w') as out:
            while not stop.is_set():
                sample = {'unix_seconds': time.time(), 'monotonic_seconds': time.monotonic()}
                try:
                    sample['metrics_status'], sample['metrics'] = api('/metrics', timeout=2)
                    raw = subprocess.check_output(['/usr/bin/vm_stat'], text=True)
                    page = re.search(r'page size of (\d+) bytes', raw)
                    sample['vm_page_bytes'] = int(page.group(1)) if page else None
                    sample['vm_counters'] = {k.strip(): int(v) for k, v in re.findall(r'^([^:\n]+):\s+(\d+)\.?$', raw, re.M)}
                    sample['swap_usage'] = subprocess.check_output(['/usr/sbin/sysctl', '-n', 'vm.swapusage'], text=True).strip()
                except Exception as error:
                    sample['error'] = str(error)
                out.write(json.dumps(sample) + '\n')
                out.flush()
                stop.wait(1)

    thread = threading.Thread(target=monitor)
    thread.start()
    results = []
    started = time.monotonic()
    try:
        for number in range(args.count):
            # Different first token content and session identity prevent counting
            # a retained continuation as cold qualification evidence. The engine
            # receipt remains the authority for actual reuse/path classification.
            prefix = f'Isolated allocation sample {number}: '
            body = '\n'.join(f'Record {i}: status recorded, checksum unchanged.' for i in range(220))
            payload = {'model': 'escha-35b-a3b', 'session_id': 8900000 + number,
                       'messages': [{'role': 'user', 'content': prefix + body + '\nReply exactly READY.'}],
                       'max_tokens': 32, 'temperature': 0, 'stream': False}
            before = time.monotonic()
            status, response = api('/v1/chat/completions', payload)
            after_status, metrics = api('/metrics')
            row = {'sample': number, 'started_seconds': before - started,
                   'elapsed_seconds': time.monotonic() - before, 'status': status,
                   'response': response, 'metrics_status': after_status, 'metrics': metrics}
            results.append(row)
            (args.output / 'requests.json').write_text(json.dumps(results, indent=2) + '\n')
            print(json.dumps({'sample': number, 'status': status, 'elapsed_seconds': row['elapsed_seconds']}), flush=True)
            if status != 200:
                break
            if number + 1 < args.count:
                time.sleep(max(0, started + (number + 1) * args.interval - time.monotonic()))
    finally:
        stop.set()
        thread.join()
    (args.output / 'COMPLETE').write_text('Probe finished; inspect receipts and pressure validity before interpreting learning.\n')


if __name__ == '__main__':
    main()
