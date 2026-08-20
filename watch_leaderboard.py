#!/usr/bin/env python3
"""Read-only live dashboard for a running (or finished) ``run_leaderboard.py`` job.

``run_leaderboard.py`` draws its own ``rich`` dashboard, but only when it owns a TTY --
a run started with ``--no-ui`` under ``nohup`` has nowhere to draw it. This script
reconstructs the same view from the files that run already writes:

* ``<out-dir>/leaderboard_summary.json`` -- rewritten after every completed route
  (aggregate DS/RC/IP/success + per-route rows + the ``pending`` list),
* the orchestrator log (``--log``) -- the 30 s ``s0:<route>(<phase>,<ticks>t)`` lines,
* ``<out-dir>/logs/<route>.log`` -- the live worker, tailed for game/system time and
  the last ``[RC-PID]`` control line.

It opens nothing else and writes nothing, so it is safe to start and stop at will
against a live run; several copies can watch the same run at once.

Usage::

    ./watch_leaderboard.py leaderboard_runs/b2d_ckpt6000_seed0 --log /tmp/lb_ckpt6000.log

Ctrl-C exits the viewer only -- the run is a separate detached process and is untouched.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

# Orchestrator progress line:
#   [leaderboard] 3/220 done, 216 queued, elapsed 0:41:00 | DS=.. | s0:accident-004(driving,470t)
_RE_PROGRESS = re.compile(
    r"\[leaderboard\]\s+(\d+)/(\d+) done,\s+(\d+) queued,\s+elapsed\s+(\S+)"
)
_RE_SLOT = re.compile(r"s(\d+):([\w\-.]+)\(([^,]+),(\d+)t\)")
# Worker log:
_RE_AGENT = re.compile(r"System time = ([\d.]+) -- Game time = ([\d.]+) -- Ratio = ([\d.]+)x")
_RE_SPEED = re.compile(r"\[RC-PID\] Desired speed:\s*([-\d.]+)\s+Current speed:\s*([-\d.]+)")
_RE_CTRL = re.compile(r"\[RC-PID\] Steer:\s*([-\d.]+)\s+Throttle:\s*([-\d.]+)\s+Brake:\s*(\w+)")


def _hms(seconds: float) -> str:
    return str(timedelta(seconds=int(max(0.0, seconds))))


def _tail(path: Path, nbytes: int = 65536) -> str:
    try:
        with path.open("rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - nbytes))
            return fh.read().decode("utf-8", "replace")
    except OSError:
        return ""


def _last(pattern: re.Pattern, text: str):
    m = None
    for m in pattern.finditer(text):
        pass
    return m


def read_summary(out_dir: Path) -> dict[str, Any]:
    path = out_dir / "leaderboard_summary.json"
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        # Rewritten in place after every route, so a read can land mid-write.
        return {}


def read_orchestrator(log: Path | None) -> dict[str, Any]:
    if not log or not log.exists():
        return {}
    text = _tail(log)
    prog = _last(_RE_PROGRESS, text)
    out: dict[str, Any] = {}
    if prog:
        out["done"], out["total"] = int(prog.group(1)), int(prog.group(2))
        out["queued"], out["elapsed"] = int(prog.group(3)), prog.group(4)
        out["slots"] = [
            {"slot": int(s.group(1)), "route": s.group(2), "phase": s.group(3), "ticks": int(s.group(4))}
            for s in _RE_SLOT.finditer(text[prog.end():])
        ]
    out["tail"] = [ln for ln in text.strip().splitlines()[-3:]]
    return out


def read_worker(out_dir: Path, route: str) -> dict[str, Any]:
    log = out_dir / "logs" / f"{route}.log"
    if not log.exists():
        return {}
    text = _tail(log)
    info: dict[str, Any] = {"mtime": log.stat().st_mtime}
    if (m := _last(_RE_AGENT, text)):
        info["system_s"], info["game_s"], info["ratio"] = float(m.group(1)), float(m.group(2)), float(m.group(3))
    if (m := _last(_RE_SPEED, text)):
        info["desired"], info["current"] = float(m.group(1)), float(m.group(2))
    if (m := _last(_RE_CTRL, text)):
        info["steer"], info["throttle"], info["brake"] = float(m.group(1)), float(m.group(2)), m.group(3)
    return info


def render(out_dir: Path, log: Path | None):
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table

    summary = read_summary(out_dir)
    orch = read_orchestrator(log)
    agg = summary.get("aggregate") or {}
    routes = summary.get("routes") or []

    done = summary.get("n_completed", orch.get("done", 0))
    total = orch.get("total") or (done + summary.get("n_pending", 0))
    ckpt = summary.get("steervla_checkpoint") or "<from agent config>"

    head = Table.grid(expand=True, padding=(0, 2))
    head.add_column(ratio=1)
    head.add_column(justify="right")
    head.add_row(f"[bold]{out_dir.name}[/]  {Path(summary.get('agent_config', '?')).name}",
                 f"routes [bold]{done}[/]/{total}   queued {orch.get('queued', '?')}")
    # ETA from the mean wall-clock of completed routes; blank until one lands.
    walls = [r.get("wall_s", 0.0) for r in routes if r.get("wall_s")]
    eta = ""
    if walls and total and done < total:
        eta = f"   ETA {_hms(sum(walls) / len(walls) * (total - done))}"
    head.add_row(f"[dim]{ckpt}[/]", f"elapsed {orch.get('elapsed', '?')}{eta}")

    scores = Table.grid(expand=True, padding=(0, 2))
    for _ in range(4):
        scores.add_column(justify="center", ratio=1)
    scores.add_row(
        f"Driving Score [bold green]{agg.get('driving_score', 0.0):6.2f}[/]",
        f"Route Completion [bold]{agg.get('route_completion', 0.0):6.2f}[/]",
        f"Infraction Penalty [bold]{agg.get('infraction_penalty', 0.0):5.3f}[/]",
        f"Success [bold]{agg.get('success_rate', 0.0):5.1f}%[/]  ({agg.get('total_km', 0.0):.1f} km)",
    )

    workers = Table(expand=True, box=None, pad_edge=False)
    for col in ("slot", "route", "phase", "ticks", "game s", "ratio", "speed m/s", "steer", "thr"):
        workers.add_column(col, justify="right" if col != "route" else "left")
    now = time.time()
    for slot in orch.get("slots", []):
        w = read_worker(out_dir, slot["route"])
        stale = w.get("mtime") and now - w["mtime"] > 60
        speed = f"{w.get('current', 0.0):.1f}/{w.get('desired', 0.0):.1f}" if "current" in w else "-"
        workers.add_row(
            str(slot["slot"]),
            f"[yellow]{slot['route']}[/]" if stale else slot["route"],
            ("[red]stalled?[/]" if stale else slot["phase"]),
            str(slot["ticks"]),
            f"{w.get('game_s', 0.0):.0f}",
            f"{w.get('ratio', 0.0):.2f}x",
            speed,
            f"{w.get('steer', 0.0):+.3f}" if "steer" in w else "-",
            f"{w.get('throttle', 0.0):.2f}" if "throttle" in w else "-",
        )
    if not orch.get("slots"):
        workers.add_row(*(["-"] * 9))

    recent = Table(expand=True, box=None, pad_edge=False)
    for col, just in (("route", "left"), ("status", "left"), ("DS", "right"), ("RC", "right"),
                      ("IP", "right"), ("infr", "right"), ("wall", "right")):
        recent.add_column(col, justify=just)
    for r in routes[-12:]:
        ds = r.get("driving_score", 0.0)
        color = "green" if ds >= 80 else ("yellow" if ds >= 40 else "red")
        recent.add_row(r.get("route", "?"), str(r.get("status", ""))[:28],
                       f"[{color}]{ds:6.2f}[/]", f"{r.get('route_completion', 0.0):6.2f}",
                       f"{r.get('infraction_penalty', 0.0):5.3f}", str(r.get("num_infractions", 0)),
                       _hms(r.get("wall_s", 0.0)))

    infr = agg.get("infraction_totals") or {}
    infr_line = "  ".join(f"{k}=[bold]{v}[/]" for k, v in sorted(infr.items(), key=lambda kv: -kv[1])) or "[dim]none[/]"

    return Group(
        Panel(head, border_style="cyan"),
        Panel(scores, border_style="green"),
        Panel(workers, title="workers", border_style="blue"),
        Panel(recent, title=f"last {min(12, len(routes))} of {len(routes)} scored", border_style="magenta"),
        Panel(infr_line, title="infractions", border_style="yellow"),
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out_dir", help="the run's --out-dir")
    p.add_argument("--log", default=None, help="orchestrator log (the nohup file); "
                                               "without it, progress/elapsed/worker rows are blank")
    p.add_argument("--interval", type=float, default=2.0, help="refresh seconds (default 2)")
    p.add_argument("--once", action="store_true", help="render a single frame and exit")
    args = p.parse_args()

    out_dir = Path(args.out_dir).resolve()
    if not out_dir.is_dir():
        print(f"no such run directory: {out_dir}")
        return 1
    log = Path(args.log).resolve() if args.log else None

    from rich.console import Console
    from rich.live import Live

    console = Console()
    if args.once:
        console.print(render(out_dir, log))
        return 0
    with Live(console=console, refresh_per_second=4, screen=False) as live:
        try:
            while True:
                live.update(render(out_dir, log))
                time.sleep(args.interval)
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
