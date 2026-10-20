#!/usr/bin/env python3
"""generalization_csv.py RESULTS_DIR GEN_CKPTS OUT_CSV [CARLA_SEEDS...]

One row per (training route, deployment route) pair of a generalization sweep, in ckpts.tsv order,
with the driving score of each CARLA seed: "Seed 1" = CARLA seed 0, "Seed 2" = 1, "Seed 3" = 2 by
default. Pairs and seeds not finished yet are left blank, so re-running fills the sheet in.
"""
import csv
import json
import sys
from pathlib import Path


def main():
    res, ckpts, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    seeds = sys.argv[4:] or ['0', '1', '2']
    rows = [line.split('\t') for line in ckpts.read_text().splitlines() if line and not line.startswith('#')]
    routes = [r[0] for r in rows]
    tags = {r[0]: r[1] for r in rows}
    header = ['Training route', 'Deployment route'] + [f'Seed {i + 1}' for i in range(len(seeds))]
    done = 0
    with open(out, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for train in routes:
            for dep in routes:
                if dep == train:
                    continue
                vals = []
                for s in seeds:
                    f = res / f'{dep}__from-{tags[train]}' / f'carla_seed_{s}' / 'run_summary_frozen_eval.json'
                    ev = json.loads(f.read_text()).get('eval') if f.exists() else None
                    vals.append(f'{ev[0]["driving_score"]:.2f}' if ev else '')
                    done += bool(ev)
                w.writerow([train, dep] + vals)
    print(f'{out}: {done} / {len(routes) * (len(routes) - 1) * len(seeds)} episodes filled')


if __name__ == '__main__':
    main()
