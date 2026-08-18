#!/usr/bin/env python
"""Summarize a CAST checkpoint-degradation sweep into a markdown report.

Pairs with ``cast_ckpt_eval_watcher.sh``. That script launches one inference-only rollout per HL
checkpoint of a live CAST training run, each into its own W&B group
``<prefix>_ckpt<step>``. This reads those groups back and emits the per-checkpoint table.

Episode-level rows are identified by the presence of ``rollout/episode_length``, which
``main_carla`` logs only at episode end. Filtering on ``rollout/episodes`` instead counts *steps*
in runs that log it every step -- the mistake called out in ``md_results/
CAST_checkpoint_rollout_eval.md``.

Degenerate chain-of-thought rate is counted from the job logs rather than W&B: it comes from the
``Reason text:`` lines the actor prints, using the same regex as the earlier CAST evals so the
numbers are comparable across reports.

Usage::

    python cast_ckpt_degradation_report.py \\
        --entity catglossop --project OGBench-CARLA \\
        --group-prefix CastWall1095Deg500_eval \\
        --out md_results/CAST_wall1095_ckpt500_degradation.md
"""

from __future__ import annotations

import argparse
import re
import statistics
from collections import Counter
from pathlib import Path

# Same measure as CAST_checkpoint_rollout_eval.md / CAST_checkpoint_sweep.md: an internal
# capital run (camelCase splice) or any non-ASCII codepoint inside a reasoning trace.
DEGENERATE_RE = re.compile(r'[A-Za-z]+[A-Z][a-z]+[A-Z]|[^\x00-\x7F]')
# Not anchored: the actor prints these as "[DEBUG - steervla] Reason text: ...", so a start-of-line
# anchor matches nothing and the column silently reads as "no data" rather than as an error.
REASON_RE = re.compile(r'Reason text:\s*(.*)$')

# Verified against a live run's W&B summary rather than read off main_carla source -- the module
# also defines a 'rollout/episode_length' key on a path this stack does not take, and keying on it
# silently yields zero episodes.
EPISODE_KEY = 'rollout/episode_steps'
KEYS = [
    EPISODE_KEY,
    'rollout/episode_return',
    'rollout/episode_route_completed',
    'rollout/driving_score',
    'rollout/episode_route_progress_pct',
    'rollout/episode_collision_count',
    'rollout/episode_termination_reason',
]


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def _fmt(x, spec='.1f'):
    return '—' if x is None else format(x, spec)


def collect_group(api, entity: str, project: str, group: str) -> dict | None:
    """Pull every episode-end row from every run in one W&B group."""
    runs = list(api.runs(f'{entity}/{project}', filters={'group': group}))
    if not runs:
        return None

    episodes: list[dict] = []
    for run in runs:
        # Full unfiltered scan, then filter locally. Do NOT pass keys=: scan_history(keys=[...])
        # does not return "rows containing these keys" -- it returns one row per logged step with
        # the requested columns mostly null, and it behaves differently for running vs finished
        # runs. That silently reported 21 episodes for a run mid-flight and 0 for the same run once
        # it finished. The unfiltered scan is slower but returns exactly the episode-end rows, with
        # every field populated.
        for row in run.scan_history(page_size=2000):
            if row.get(EPISODE_KEY) is None:
                continue
            episodes.append(row)

    return {
        'group': group,
        'run_ids': [r.id for r in runs],
        'run_states': [r.state for r in runs],
        'episodes': episodes,
    }


def summarize(entry: dict) -> dict:
    eps = entry['episodes']
    terms = Counter(
        e.get('rollout/episode_termination_reason') for e in eps if e.get('rollout/episode_termination_reason')
    )
    completed = sum(1 for e in eps if e.get('rollout/episode_route_completed'))
    return {
        'n': len(eps),
        'completed': completed,
        'progress': _mean([e.get('rollout/episode_route_progress_pct') for e in eps]),
        'driving_score': _mean([e.get('rollout/driving_score') for e in eps]),
        'ret': _mean([e.get('rollout/episode_return') for e in eps]),
        'steps': _mean([e.get(EPISODE_KEY) for e in eps]),
        'collisions': _mean([e.get('rollout/episode_collision_count') for e in eps]),
        'termination': ', '.join(f'{k} ×{v}' for k, v in terms.most_common(3)) or '—',
    }


def degenerate_rate(log_path: Path) -> tuple[int, int] | None:
    """(degenerate reasoning traces, total traces) parsed out of a job log."""
    if not log_path.exists():
        return None
    total = bad = 0
    with log_path.open('r', errors='ignore') as fh:
        for line in fh:
            m = REASON_RE.search(line)
            if not m:
                continue
            total += 1
            if DEGENERATE_RE.search(m.group(1)):
                bad += 1
    return (bad, total) if total else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--entity', default='catglossop')
    ap.add_argument('--project', default='OGBench-CARLA')
    ap.add_argument('--group-prefix', required=True, help='e.g. CastWall1095Deg500_eval')
    ap.add_argument('--steps', default='', help='comma list of checkpoint steps; default = discover')
    ap.add_argument('--ckpt-every', type=int, default=500)
    ap.add_argument('--max-step', type=int, default=8000)
    ap.add_argument('--job-base', type=int, default=160, help='job index = job_base + step/ckpt_every')
    ap.add_argument('--jobs-dir', default='.run_carla/jobs')
    ap.add_argument('--baseline-job', type=int, default=None, help='job index of the step-0 arm')
    ap.add_argument('--out', default='')
    ap.add_argument(
        '--inject',
        default='',
        help='markdown file to splice the table into, between the BEGIN/END RESULTS TABLE markers',
    )
    args = ap.parse_args()

    import wandb

    api = wandb.Api(timeout=60)

    if args.steps:
        steps = [int(s) for s in args.steps.split(',') if s.strip()]
    else:
        steps = list(range(0, args.max_step + 1, args.ckpt_every))

    jobs_dir = Path(args.jobs_dir)
    rows = []
    for step in steps:
        group = f'{args.group_prefix}_ckpt{step}'
        entry = collect_group(api, args.entity, args.project, group)
        if entry is None:
            rows.append({'step': step, 'group': group, 'missing': True})
            continue
        job = args.baseline_job if (step == 0 and args.baseline_job is not None) else args.job_base + step // args.ckpt_every
        summary = summarize(entry)
        summary.update(
            step=step,
            group=group,
            job=job,
            wandb=', '.join(f'`{i}`' for i in entry['run_ids']),
            state=', '.join(sorted(set(entry['run_states']))),
            degen=degenerate_rate(jobs_dir / f'job-{job}.log'),
            missing=False,
        )
        rows.append(summary)

    lines = [
        '| Ckpt step | Job / W&B | State | Eps | Completed | Progress % | Driving score | Return | Steps | Collisions/ep | Termination | Degenerate CoT |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|',
    ]
    for r in rows:
        if r['missing']:
            lines.append(f'| {r["step"]} | — | not started | — | — | — | — | — | — | — | — | — |')
            continue
        if r['degen']:
            bad, total = r['degen']
            degen = f'{100.0 * bad / total:.1f} % ({bad}/{total})'
        else:
            degen = '—'
        lines.append(
            f'| {r["step"]} | {r["job"]} / {r["wandb"]} | {r["state"]} | {r["n"]} | '
            f'{r["completed"]}/{r["n"]} | {_fmt(r["progress"])} | {_fmt(r["driving_score"])} | '
            f'{_fmt(r["ret"])} | {_fmt(r["steps"], ".0f")} | {_fmt(r["collisions"], ".2f")} | '
            f'{r["termination"]} | {degen} |'
        )

    table = '\n'.join(lines)
    if args.out:
        Path(args.out).write_text(table + '\n')
        print(f'wrote {args.out}')
    if args.inject:
        # Rewrite in place between the markers so refreshing the report is one command and can
        # never disturb the prose around it.
        md = Path(args.inject)
        text = md.read_text()
        begin, end = '<!-- BEGIN RESULTS TABLE -->', '<!-- END RESULTS TABLE -->'
        i, j = text.find(begin), text.find(end)
        if i < 0 or j < 0:
            raise SystemExit(f'{md}: missing {begin} / {end} markers')
        md.write_text(text[: i + len(begin)] + '\n' + table + '\n' + text[j:])
        print(f'injected table into {md}')
    print(table)


if __name__ == '__main__':
    main()
