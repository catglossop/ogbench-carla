#!/usr/bin/env python3
"""generalization_ckpts.py SOURCE_ROOT ROUTES_FILE > ckpts.tsv

Checkpoint table for the generalization experiments: for every training route in ROUTES_FILE, the
LAST checkpoint in the checkpoints/ folder of that route's training run under SOURCE_ROOT (the
newest run that wrote any checkpoint, when a route was restarted). Columns:

    train_route <TAB> tag <TAB> checkpoint <TAB> step <TAB> run_dir

`tag` is a short code for the training route (word initials + route number, e.g.
non-signalized-junction-left-turn-enter-flow-002 -> nsjltef002). It names the cross-route cells
(<eval_route>__from-<tag>) and goes into the W&B run name, whose id main_carla cuts at 128 chars --
two full route names plus a timestamp would not fit.
"""
import re
import sys
from pathlib import Path

RUN_RE = re.compile(r'_upd-hl_(?P<route>.+)_seed_0_(?P<ts>\d{8}_\d{6})$')


def tag_for(route):
    words = route.split('-')
    num = words[-1] if words[-1].isdigit() else ''
    body = words[:-1] if num else words
    return ''.join(w.rstrip('_')[0] for w in body if w.rstrip('_')) + num


def main():
    root, routes = Path(sys.argv[1]), [r for r in Path(sys.argv[2]).read_text().split() if r]
    runs = {}
    for d in root.rglob('*_seed_0_*'):
        m = RUN_RE.search(d.name)
        if not d.is_dir() or not m:
            continue
        steps = sorted(int(c.name) for c in (d / 'checkpoints').glob('*')
                       if c.name.isdigit() and (c / 'pytorch_model.bin').is_file())
        if steps:
            runs.setdefault(m['route'], []).append((m['ts'], d, steps[-1]))
    tags, missing = {}, []
    for r in routes:
        if r not in runs:
            missing.append(r)
            continue
        _, d, step = max(runs[r])
        t = tag_for(r)
        if t in tags:
            sys.exit(f'tag collision: {r} and {tags[t]} both -> {t}')
        tags[t] = r
        print(f'{r}\t{t}\t{d / "checkpoints" / str(step)}\t{step}\t{d}')
    if missing:
        print(f'no checkpoint for: {" ".join(missing)}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
