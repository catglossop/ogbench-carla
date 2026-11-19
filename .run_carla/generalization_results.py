#!/usr/bin/env python3
"""generalization_results.py RESULTS_DIR GEN_CKPTS "CARLA_SEEDS"

Cross-route matrix for a generalization sweep (qwen_zs_bon_sweep.sh with GEN_CKPTS): rows are the
training route whose checkpoint was deployed, columns the route it was deployed on, each entry the
mean driving score over the CARLA seeds finished so far (n in brackets when not all are in). The
diagonal is never run here. Writes RESULTS_DIR/results.md and RESULTS_DIR/summary.csv (one row per
episode) and prints the markdown.
"""
import csv
import json
import sys
from pathlib import Path


def main():
    res, ckpts, seeds = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3].split()
    rows = [line.split('\t') for line in ckpts.read_text().splitlines() if line and not line.startswith('#')]
    routes = [r[0] for r in rows]
    tags = {r[0]: r[1] for r in rows}
    cells, episodes = {}, []
    for train in routes:
        for ev in routes:
            if ev == train:
                continue
            scores = []
            for s in seeds:
                f = res / f'{ev}__from-{tags[train]}' / f'carla_seed_{s}' / 'run_summary_frozen_eval.json'
                if not f.exists():
                    continue
                for e in json.loads(f.read_text()).get('eval') or []:
                    scores.append(e['driving_score'])
                    episodes.append(dict(train_route=train, eval_route=ev, carla_seed=s,
                                         eval_seed=e['eval_seed'], driving_score=e['driving_score']))
            if scores:
                cells[train, ev] = scores

    def fmt(v, full):
        if not v:
            return '-'
        m = sum(v) / len(v)
        return f'{m:.1f}' if len(v) >= full else f'{m:.1f} ({len(v)})'

    full = len(seeds)
    out = [f'# {res.name}', '',
           f'Rows: checkpoint trained on route (tag); columns: route deployed on (tag). Entry = mean driving '
           f'score over CARLA seeds {" ".join(seeds)}; "(n)" = only n episodes so far. Diagonal not run.', '',
           f'Cells with any episode: {len(cells)} / {len(routes) * (len(routes) - 1)}; '
           f'episodes: {len(episodes)} / {len(routes) * (len(routes) - 1) * full}.', '']
    out.append('| trained on \\ deployed on | ' + ' | '.join(tags[r] for r in routes) + ' | row mean |')
    out.append('|---|' + '---:|' * (len(routes) + 1))
    for train in routes:
        vals = [v for ev in routes if (train, ev) in cells for v in cells[train, ev]]
        line = [f'`{tags[train]}`'] + [
            '—' if ev == train else fmt(cells.get((train, ev)), full) for ev in routes]
        line.append(f'{sum(vals) / len(vals):.1f}' if vals else '-')
        out.append('| ' + ' | '.join(line) + ' |')
    colm = []
    for ev in routes:
        vals = [v for tr in routes if (tr, ev) in cells for v in cells[tr, ev]]
        colm.append(f'{sum(vals) / len(vals):.1f}' if vals else '-')
    out.append('| **col mean** | ' + ' | '.join(colm) + ' | |')
    allv = [e['driving_score'] for e in episodes]
    if allv:
        succ = sum(1 for v in allv if v >= 100.0)
        out += ['', f'**Overall off-route mean DS: {sum(allv) / len(allv):.2f}** over {len(allv)} episodes; '
                    f'success (DS = 100) {succ}/{len(allv)} = {100 * succ / len(allv):.1f}%.']
    out += ['', '| tag | route |', '|---|---|'] + [f'| `{tags[r]}` | {r} |' for r in routes]
    text = '\n'.join(out) + '\n'
    (res / 'results.md').write_text(text)
    with open(res / 'summary.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['train_route', 'eval_route', 'carla_seed', 'eval_seed', 'driving_score'])
        w.writeheader()
        w.writerows(episodes)
    print(text)


if __name__ == '__main__':
    main()
