#!/usr/bin/env python3
"""A/B the LCM checkpoint writer on local Higgs: model handoff (base) vs mechanical fold (head).

Reuses the endurance arm-A stream (existing LCM + retrieval, no notes/reset
controls) from experiments/context-recovery. Each revision is built once in a
sibling git worktree (so Cargo's ../jack-voice path patch resolves), then the
repeats run interleaved ABBA... on a freshly booted Higgs per run so neither
arm inherits the other's KV cache or thermal state.

Run from the repository root, inside tmux, with the recovery-higgs pane set up
as for endurance.py:

    python3 experiments/local-bench/checkpoint_ab.py /private/tmp/checkpoint-ab-1 \
        --base 139170f~1 --head 139170f --repeats 3

Re-running with the same output directory skips completed runs and only
re-renders the comparison. No cloud model, judge or external action is used.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ENDURANCE = REPO / 'experiments' / 'context-recovery' / 'endurance.py'

# (summary key, label, lower_is_better); None means descriptive only.
METRICS = [
    ('correct_updates', 'correct snapshots', False),
    ('completed_updates', 'completed updates', False),
    ('seconds', 'wall-clock s', True),
    ('snapshots_per_minute', 'snapshots/min', False),
    ('model_requests', 'model requests (all)', True),
    ('llm_compaction_requests', 'compaction model requests', True),
    ('completed_compactions', 'completed compactions', None),
    ('refetch_calls', 'lcm_expand+recall+history calls', True),
    ('tool_errors', 'tool errors', True),
    ('logical_input_tokens', 'prompt tokens (sum)', True),
    ('output_tokens', 'output tokens (sum)', True),
    ('peak_actual_prompt_tokens', 'peak prompt tokens', None),
]


def git(*args, cwd=REPO):
    return subprocess.check_output(['git', *args], cwd=cwd, text=True).strip()


def worktree_for(label, rev):
    """Check out `rev` next to the repo so ../jack-voice resolves like the main checkout."""
    sha = git('rev-parse', '--verify', f'{rev}^{{commit}}')
    path = REPO.parent / f'{REPO.name}-ab-{label}-{sha[:10]}'
    if not path.exists():
        git('worktree', 'add', '--detach', str(path), sha)
    elif git('rev-parse', 'HEAD', cwd=path) != sha:
        raise SystemExit(f'{path} exists at a different commit; remove it or pick another label')
    return sha, path


def build_test_binary(source):
    """Release-build the lib test harness and return the executable Cargo reports."""
    command = ['cargo', 'test', '--release', '--lib', '--no-run', '--message-format=json']
    process = subprocess.run(command, cwd=source, stdout=subprocess.PIPE, text=True)
    if process.returncode:
        raise SystemExit(f'build failed in {source}')
    executables = []
    for line in process.stdout.splitlines():
        try:
            message = json.loads(line)
        except ValueError:
            continue
        if message.get('reason') == 'compiler-artifact' and message.get('executable') \
                and message['target'].get('kind') == ['lib']:
            executables.append(message['executable'])
    if len(executables) != 1:
        raise SystemExit(f'expected one lib test executable in {source}, got {executables}')
    return Path(executables[0])


def schedule(repeats):
    """ABBA ordering so drift over the session cancels between arms."""
    order = []
    for i in range(repeats):
        order += [('base', i), ('head', i)] if i % 2 == 0 else [('head', i), ('base', i)]
    return order


def run_one(root, label, index, binary, source, args):
    out = root / f'{label}-r{index}'
    summary_path = out / 'summary.json'
    if summary_path.exists():
        print(f'skip {out.name}: already complete', flush=True)
    else:
        if out.exists():
            raise SystemExit(f'{out} is incomplete; inspect and remove it before re-running')
        command = [sys.executable, str(ENDURANCE), str(out), '--arms', 'A',
                   '--updates', str(args.updates), '--ceiling', str(args.ceiling),
                   '--minutes', str(args.minutes), '--test-binary', str(binary),
                   '--source-dir', str(source)]
        print(f'run {out.name}: {" ".join(command)}', flush=True)
        # endurance.py imports run.py/memory_probe.py as siblings.
        subprocess.run(command, cwd=REPO, check=False,
                       env=dict(os.environ, PYTHONPATH=str(ENDURANCE.parent)))
    if not summary_path.exists():
        return None
    arms = json.loads(summary_path.read_text())
    return arms[0] if isinstance(arms, list) else arms


def derive(summary):
    calls = summary.get('control_tool_calls', {})
    summary['refetch_calls'] = sum(calls.get(name, 0) for name in ('lcm_expand', 'recall', 'history'))
    return summary


def fmt(values):
    if not values:
        return 'n/a'
    if len(values) == 1:
        return f'{values[0]:.1f}'
    return f'{statistics.median(values):.1f} [{min(values):.1f}-{max(values):.1f}]'


def compare(root, results, provenance):
    lines = [f'# Checkpoint writer A/B ({root.name})', '',
             f'- base (model handoff): `{provenance["base"]["sha"]}`',
             f'- head (mechanical fold): `{provenance["head"]["sha"]}`',
             f'- stream: endurance arm A, {provenance["updates"]} updates, '
             f'ceiling {provenance["ceiling"]}, {provenance["minutes"]} min/run', '']
    valid = {label: [r for r in runs if r.get('valid_measurement')] for label, runs in results.items()}
    for label, runs in results.items():
        lines.append(f'- {label}: {len(valid[label])}/{len(runs)} valid runs, '
                     f'{sum(bool(r.get("pass")) for r in valid[label])} full passes')
    lines += ['', 'Median [min-max] over valid runs.', '',
              '| metric | base | head | better |', '|---|---|---|---|']
    for key, label, lower in METRICS:
        base = [float(r[key]) for r in valid['base'] if r.get(key) is not None]
        head = [float(r[key]) for r in valid['head'] if r.get(key) is not None]
        verdict = ''
        if lower is not None and base and head and statistics.median(base) != statistics.median(head):
            head_wins = (statistics.median(head) < statistics.median(base)) == lower
            verdict = 'head' if head_wins else 'base'
        lines.append(f'| {label} | {fmt(base)} | {fmt(head)} | {verdict} |')
    lines += ['', 'Invalid runs (wire errors, watchdog, server restart) are excluded above; '
              'inspect their summary.json before drawing conclusions. With fewer than 3 valid '
              'runs per arm treat differences as anecdotal.', '']
    report = '\n'.join(lines)
    (root / 'COMPARISON.md').write_text(report)
    print(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('output', type=Path)
    parser.add_argument('--base', default='139170f~1', help='revision with the model-handoff writer')
    parser.add_argument('--head', default='HEAD', help='revision with the mechanical writer')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--updates', type=int, default=20)
    parser.add_argument('--ceiling', type=int, default=16384)
    parser.add_argument('--minutes', type=int, default=45)
    parser.add_argument('--dry-run', action='store_true', help='build and print the schedule only')
    args = parser.parse_args()
    if args.repeats < 1:
        raise SystemExit('--repeats must be positive')

    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    builds = {}
    for label, rev in (('base', args.base), ('head', args.head)):
        sha, source = worktree_for(label, rev)
        print(f'build {label} {sha[:10]} in {source}', flush=True)
        builds[label] = {'sha': sha, 'source': source, 'binary': build_test_binary(source)}
    provenance = {'updates': args.updates, 'ceiling': args.ceiling, 'minutes': args.minutes,
                  'repeats': args.repeats, 'order': schedule(args.repeats),
                  **{label: {k: str(v) for k, v in b.items()} for label, b in builds.items()}}
    (root / 'ab-provenance.json').write_text(json.dumps(provenance, indent=2))
    if args.dry_run:
        print(json.dumps(provenance, indent=2))
        return

    results = {'base': [], 'head': []}
    for label, index in provenance['order']:
        build = builds[label]
        summary = run_one(root, label, index, build['binary'], build['source'], args)
        if summary is not None:
            results[label].append(derive(summary))
    compare(root, results, provenance)


if __name__ == '__main__':
    main()
