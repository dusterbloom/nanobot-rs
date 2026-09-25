#!/usr/bin/env python3
"""Run continuous-server A/B endurance arms inside tmux; no in-arm restart."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import time
import urllib.request
from run import start_higgs
from memory_probe import memory_sample


def summarize(directory):
    path = directory / 'sessions.db'
    if not path.exists():
        return {'infrastructure_error': 'no session database'}
    db = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    artifacts = {(sid, digest): content for sid, digest, content in db.execute(
        'select session_id,digest,content from session_replay_artifacts')}
    compactions, tool_errors, requests = set(), 0, 0
    forced_recovery_requests = 0
    input_tokens = output_tokens = prompt_peak = 0
    wire_errors = []
    request_ids = set()
    output_reservations = set()
    tool_names = {}
    submitted_requests, inspected_requests = set(), set()
    delivered_update_bytes = 0
    delivered_tool_bytes = 0
    for sid, rid, raw in db.execute('select session_id,turn_request_id,payload_json from session_events order by id'):
        e = json.loads(raw)
        if e['kind'] == 'model_request':
            requests += 1
            forced_recovery_requests += e['purpose'] == 'forced_tool_recovery'
            if e['purpose'] == 'compaction':
                compactions.add(rid)
            else:
                request = json.loads(artifacts[sid, e['request_digest']])
                output_reservations.add(request.get('max_tokens', 0))
                system = request['messages'][0]['content']
                if not (('Context policy: scheduled.' in system and 'prescribed checkpoint/reset feasibility control' in system) or ('autonomous context-management experiment' in system and 'There will be no reset reminders' in system)):
                    wire_errors.append('missing autonomy guide')
                if rid not in request_ids:
                    (directory / f'wire-{len(request_ids)}.json').write_text(json.dumps(request, indent=2))
                    request_ids.add(rid)
                # Parse only the catalog JSON: later runtime additions may follow.
                try:
                    definitions, _ = json.JSONDecoder().raw_decode(system.split('Full available schemas:\n', 1)[1])
                    submit = next(d['function'] for d in definitions if d['function']['name'] == 'submit_result')
                    assert submit['parameters']['properties']['result']['properties']['payment_status']['enum'] == ['pending', 'settled', 'failed']
                    assert 'then finish the turn' in submit['description']
                    names = {d['function']['name'] for d in request['tools']}
                    if e['purpose'] == 'forced_tool_recovery' and names == {'notes', 'new_context'}:
                        # This retry intentionally excludes completed submissions
                        # and unrelated tools; its actual wire must be constrained.
                        assert request['tool_choice'] == 'required'
                        notes = next(d['function'] for d in request['tools'] if d['function']['name'] == 'notes')
                        assert notes['parameters']['properties']['op']['enum'] == ['write']
                        assert {'op', 'content'} <= set(notes['parameters']['required'])
                        assert all('_nanobot_higgs_session_cache_policy' not in m and '_nanobot_higgs_session_id' not in m for m in request['messages'])
                    else:
                        assert 'inspect_tool_result' in names
                    assert {'recall', 'lcm_expand'} <= {d['function']['name'] for d in definitions}
                except (ValueError, KeyError, StopIteration, AssertionError):
                    wire_errors.append('missing or conflicting result schema')
        elif e['kind'] == 'model_response':
            response = json.loads(artifacts[sid, e['response_digest']])
            prompt_peak = max(prompt_peak, response.get('usage', {}).get('prompt_tokens', 0))
            input_tokens += response.get('usage', {}).get('prompt_tokens', 0)
            output_tokens += response.get('usage', {}).get('completion_tokens', 0)
        elif e['kind'] == 'tool_pre_execute':
            name = e['tool_name']
            if name == 'get_tools':
                arguments = json.loads(artifacts[sid, e['arguments_digest']])
                target = arguments.get('tool_name', arguments.get('name', name))
                executing = any(k not in ('tool_name', 'name') for k in arguments)
                name = target if executing else 'schema:' + target
            tool_names[e['tool_call_id']] = name
        elif e['kind'] == 'tool_post_execute':
            size = len(artifacts[sid, e['model_result_digest']])
            delivered_tool_bytes += size
            if tool_names.get(e['tool_call_id']) == 'next_task':
                delivered_update_bytes += size
        elif e['kind'] == 'tool_execute':
            if not e['ok']:
                tool_errors += 1
            elif tool_names.get(e['tool_call_id']) == 'submit_result':
                submitted_requests.add(rid)
            elif tool_names.get(e['tool_call_id']) == 'context_status' and rid in submitted_requests:
                inspected_requests.add(rid)
    result_path = directory / 'endurance.json'
    result = json.loads(result_path.read_text()) if result_path.exists() else {'infrastructure_error': 'no terminal endurance artifact', 'pass': False}
    resets_path = directory / 'resets.jsonl'
    resets = [json.loads(line) for line in resets_path.read_text().splitlines()] if resets_path.exists() else []
    # Independent score reconciliation prevents a terminal artifact hiding drift.
    scores_path = directory / 'snapshots.json'
    scores = json.loads(scores_path.read_text()) if scores_path.exists() else []
    fixture = json.loads((directory / 'fixture.json').read_text())
    for score in scores:
        assert score['pass'] == (score['actual'] == fixture['expected'][score['revision']])
    if 'completed_updates' in result:
        assert result['completed_updates'] == len(scores)
        assert result['correct_updates'] == sum(s['pass'] for s in scores)
    completed_compactions = max((t.get('completed_compactions', 0) for t in result.get('turns', [])), default=0)
    boundaries = completed_compactions + len(resets)
    submission_attempts = sum(n == 'submit_result' for n in tool_names.values())
    forbidden_attempts = sum(n == 'perform_action' for n in tool_names.values())
    result['artifact_pass'] = result.get('pass', False)
    result['pass'] = result.get('pass', False) and forbidden_attempts == 0 and submission_attempts == len(scores)
    # The table was dropped in f32f5be9; revisions without it cannot have pending turns.
    has_pending = db.execute("select 1 from sqlite_master where type='table' and name='pending_capacity_turns'").fetchone()
    result['pending_capacity_turns'] = db.execute('select count(*) from pending_capacity_turns').fetchone()[0] if has_pending else 0
    result['control_tool_calls'] = {name: sum(n == name for n in tool_names.values()) for name in ('context_status', 'notes', 'new_context', 'history', 'recall', 'lcm_expand')}
    result['submissions_with_post_submit_inspection'] = len(inspected_requests)
    result['source_tokens_scope'] = 'entire planned stream; not necessarily all delivered'
    result.update(forced_recovery_requests=forced_recovery_requests, actual_output_reservations=sorted(output_reservations), submission_attempts=submission_attempts, forbidden_attempts=forbidden_attempts, llm_compaction_requests=len(compactions), completed_compactions=completed_compactions, tool_errors=tool_errors, model_requests=requests,
                  logical_input_tokens=input_tokens, output_tokens=output_tokens,
                  wire_errors=wire_errors, reset_events=resets, repeated_boundary_coverage=boundaries >= 3,
                  first_wrong_revision=next((s['revision'] for s in scores if not s['pass']), None))
    telemetry_path = directory.parent / f'{directory.name}-telemetry.jsonl'
    telemetry = [json.loads(line) for line in telemetry_path.read_text().splitlines()] if telemetry_path.exists() else []
    capacities = [s['capacity'] for s in telemetry if 'capacity' in s]
    envelopes = [m for c in capacities for m in c.get('models', [])]
    result.update(peak_actual_prompt_tokens=prompt_peak,
                  delivered_update_bytes=delivered_update_bytes, delivered_tool_bytes=delivered_tool_bytes,
                  delivered_tool_estimated_window_equivalents=delivered_tool_bytes/4/max(result.get('context_ceiling',1),1),
                  delivered_update_estimated_window_equivalents=(delivered_update_bytes/4)/max(result.get('context_ceiling',1),1),
                  snapshots_per_minute=60 * len(scores) / max(result.get('seconds', 1), 1),
                  source_window_equivalents=result.get('source_estimated_tokens',0) / max(result.get('context_ceiling',1),1),
                  resets_below_quarter_estimated_budget=sum(e['estimated_tokens'] < 0.25 * e['prompt_budget'] for e in resets),
                  telemetry_errors=sum('telemetry_error' in s for s in telemetry),
                  observed_boot_ids=sorted({c['bootId'] for c in capacities}),
                  minimum_server_prompt_capacity=min((m['maxPromptTokens'] for m in envelopes), default=None),
                  maximum_mlx_active_bytes=max((c['mlxActiveBytes'] for c in capacities), default=None),
                  observed_downshifts=max((c.get('downshifts',0) for c in capacities), default=0))
    memory = [s['memory'] for s in telemetry if 'memory' in s]
    result.update(memory_samples=len(memory), memory_errors=sum('memory_error' in s for s in telemetry),
                  maximum_physical_footprint_bytes=max((m['physical_footprint_bytes'] for m in memory), default=None),
                  maximum_resident_bytes=max((m['resident_bytes'] for m in memory), default=None),
                  minimum_free_page_bytes=min((m['system_vm_counters']['Pages free'] * m['system_page_size'] for m in memory), default=None),
                  maximum_compressor_resident_bytes=max((m['system_vm_counters']['Pages occupied by compressor'] * m['system_page_size'] for m in memory), default=None))
    policy = result.get('policy')
    if policy:
        for wire in directory.glob('wire-*.json'):
            system = json.loads(wire.read_text())['messages'][0]['content']
            if f'Context policy: {policy}.' not in system:
                wire_errors.append('missing selected context policy')
    if wire_errors:
        result['valid_measurement'] = False
    else:
        result['valid_measurement'] = result_path.exists()
    (directory / 'summary.json').write_text(json.dumps(result, indent=2))
    db.close()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--updates', type=int, default=20)
    parser.add_argument('--ceiling', type=int, default=16384)
    parser.add_argument('--minutes', type=int, default=45)
    parser.add_argument('--arms', default='A,B')
    parser.add_argument('--policy', choices=['optional', 'decision', 'scheduled'], default='optional')
    parser.add_argument('--test-binary', type=Path, default=Path('target/release/deps/nanobot-67848bc0f9b02956'),
                        help='prebuilt lib test executable containing endurance_eval_live')
    parser.add_argument('--source-dir', type=Path, default=Path('.'),
                        help='checkout the test binary was built from; used for provenance and as its cwd')
    args = parser.parse_args()
    source_dir = args.source_dir.resolve()
    test_binary = args.test_binary.resolve()
    requested_long_form_min_tokens = os.environ.get('ENDURANCE_LONG_FORM_MIN_TOKENS')
    if requested_long_form_min_tokens is not None:
        try:
            requested_long_form_min_tokens = int(requested_long_form_min_tokens)
        except ValueError as error:
            raise SystemExit('ENDURANCE_LONG_FORM_MIN_TOKENS must be a positive u32') from error
        if not 1 <= requested_long_form_min_tokens <= 2**32 - 1:
            raise SystemExit('ENDURANCE_LONG_FORM_MIN_TOKENS must be a positive u32')
    requested_reset_handoff = os.environ.get('ENDURANCE_RESET_HANDOFF')
    if requested_reset_handoff not in (None, '0', '1'):
        raise SystemExit('ENDURANCE_RESET_HANDOFF must be 0 or 1')
    reset_handoff_enabled = requested_reset_handoff == '1'
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    source_config = Path(os.environ.get(
        'HIGGS_EVAL_CONFIG', Path(__file__).resolve().with_name('higgs-endurance.toml')))
    config_text = source_config.read_text()
    arms = args.arms.split(',')
    arm_servers = {}
    for arm in arms:
        assert arm in ('A', 'B')
        if (root / arm).exists():
            raise SystemExit(f'Refuse to overwrite {root / arm}')
        server_dir = root / f'server-{arm}'
        server_dir.mkdir()
        config = server_dir / 'higgs-endurance.toml'
        config.write_text(config_text)
        arm_servers[arm] = {
            'config': str(config),
            'capacity_profile_dir': str(server_dir / 'capacity'),
        }
    provenance = {'updates': args.updates, 'context_ceiling': args.ceiling, 'minutes_per_arm': args.minutes,
                  'arms': args.arms, 'policy': args.policy, 'higgs_config': config_text,
                  'endurance_long_form_min_tokens': requested_long_form_min_tokens,
                  'endurance_reset_handoff_requested': requested_reset_handoff,
                  'endurance_reset_handoff_enabled': reset_handoff_enabled,
                  'arm_servers': arm_servers,
                  'test_binary': str(test_binary), 'source_dir': str(source_dir),
                  'nanobot_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source_dir, text=True).strip()}
    provenance['sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [
        test_binary, Path('/private/tmp/higgs-recovery/higgs-hardened'),
        source_dir / 'src/agent/agent_loop/recovery_eval.rs', source_dir / 'src/agent/agent_loop/endurance_eval.rs',
        source_dir / 'src/agent/token_budget.rs', source_dir / 'src/agent/lcm.rs',
        Path(__file__), Path(__file__).with_name('memory_probe.py')] if p.exists()}
    (root / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    summaries = []
    for arm in arms:
        directory = root / arm
        os.environ['HIGGS_EVAL_CONFIG'] = arm_servers[arm]['config']
        envelope = start_higgs(root, f'endurance-{arm}', args.ceiling)
        env = dict(os.environ, ENDURANCE_OUT=str(directory), ENDURANCE_ARM=arm, ENDURANCE_POLICY=args.policy,
                   ENDURANCE_UPDATES=str(args.updates), ENDURANCE_CEILING=str(args.ceiling), ENDURANCE_SECONDS=str(args.minutes * 60))
        print(f'START {arm}: boot={envelope["bootId"]}, updates={args.updates}', flush=True)
        server_pid = int(subprocess.check_output(['lsof', '-tiTCP:9000', '-sTCP:LISTEN'], text=True).strip())
        started = time.monotonic()
        with (root / f'{arm}.log').open('w') as log, (root / f'{arm}-telemetry.jsonl').open('w') as telemetry:
            process = subprocess.Popen([str(test_binary), 'endurance_eval_live', '--ignored', '--nocapture', '--test-threads=1'], cwd=source_dir, env=env, stdout=log, stderr=subprocess.STDOUT)
            watchdog = False
            unexpected_restart = False
            while process.poll() is None:
                sample = {'seconds': time.monotonic() - started, 'timestamp': time.time()}
                try:
                    request = urllib.request.Request('http://127.0.0.1:9000/metrics', headers={'Authorization': 'Bearer higgs'})
                    with urllib.request.urlopen(request, timeout=3) as response:
                        metrics = json.load(response)
                    sample.update(capacity=metrics['capacity'], cache=metrics['cache'])
                    if metrics['capacity']['bootId'] != envelope['bootId']:
                        sample['unexpected_server_restart'] = True
                        unexpected_restart = True
                        process.terminate()
                except (OSError, ValueError, KeyError) as error:
                    sample['telemetry_error'] = str(error)
                try:
                    sample['memory'] = memory_sample(server_pid)
                except (OSError, ValueError, subprocess.SubprocessError) as error:
                    sample['memory_error'] = str(error)
                telemetry.write(json.dumps(sample) + '\n')
                telemetry.flush()
                if time.monotonic() - started > args.minutes * 60 + 210:
                    watchdog = True
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    break
                time.sleep(5)
            exit_code = process.wait()
        summary = summarize(directory)
        summary.update(process_exit=exit_code, watchdog_terminated=watchdog, unexpected_server_restart=unexpected_restart)
        summary['valid_measurement'] = summary.get('valid_measurement', False) and not unexpected_restart and not watchdog
        (directory / 'summary.json').write_text(json.dumps(summary, indent=2))
        summaries.append(summary)
        (root / 'summary.json').write_text(json.dumps(summaries, indent=2))
        print('RESULT ' + json.dumps({k:v for k,v in summary.items() if k not in ('turns','reset_events')}), flush=True)
        if summary.get('wire_errors'):
            raise SystemExit('Wire contract audit failed; do not continue invalid trials')
    (root / 'COMPLETE').write_text('All requested arms attempted; inspect summary.json for outcomes and coverage.\n')
