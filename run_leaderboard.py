#!/usr/bin/env python3
"""Faithful Bench2Drive / Fail2Drive leaderboard evaluation, parallelised over GPUs.

This is the scoring counterpart to ``run_carla.sh``. Where that script is built for
*training* runs (short episodes, shaped reward, early termination so the RL loop keeps
moving), this one reproduces what the Bench2Drive leaderboard actually measures:

* ``crash_stuck_steps`` is raised out of the way so the leaderboard's own
  ``AgentBlockedTest`` (min_speed=0.1 m/s, max_time=60 s, terminate_on_failure=True)
  is what stops a wedged agent -- not the wrapper's 1-second post-collision cutoff.
* ``max_episode_steps = 0`` removes the wrapper's step cap; the route runs until the
  scenario tree ends or Bench2Drive's own ``tick_count > 4000`` guard fires.
* ``terminate_on_infraction = False`` so a collision or a red light does not end the
  route early -- it is scored as an infraction penalty, exactly like the leaderboard.

Scores are **not** recomputed here. Each worker's ``CarlaBench2DriveWrapper`` calls the
real ``leaderboard.utils.statistics_manager.StatisticsManager`` on route end, which
writes a standard leaderboard checkpoint JSON. This script only points each route's
``checkpoint`` at its own file, reads them back, and aggregates.

Usage
-----
    # Both GPUs, all 220 Bench2Drive routes, live dashboard
    ./run_leaderboard.py --slots 0:1,1:0 --routes bench2drive

    # A named subset, custom checkpoint, no UI (e.g. under nohup)
    ./run_leaderboard.py --slots 0:1 --routes parking-cut-in-001,merging-002 \
        --steervla-checkpoint gs://cat-logs/.../6000 --no-ui

    # See the plan without launching anything
    ./run_leaderboard.py --slots 0:1,1:0 --routes all --dry-run

``--slots`` is a comma-separated list of ``TRAIN_GPU:RENDER_ADAPTER`` pairs, one per
concurrent worker. ``TRAIN_GPU`` indexes ``jax.devices()`` (pins the JAX device via the
agent config's ``training_gpu_rank``); ``RENDER_ADAPTER`` is CARLA's ``-graphicsadapter``,
whose ordering is **swapped** relative to ``nvidia-smi`` -- see the comment in
``impls/configs/carla_config.yaml``. Ports and X displays are derived from the slot index
using the same scheme as ``carla_job.sh`` (rpc 12000+100k, tm 18000+100k, display 30+k),
so slots can never collide with each other or with a running ``carla_job.sh`` job at a
different index.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from ogbench.carla.route_registry import find_route, list_routes

# --------------------------------------------------------------------------- #
# Leaderboard-faithful CARLA config                                            #
# --------------------------------------------------------------------------- #

# NOTE on crash_stuck_steps: the check in carla_utils.py is
# ``self._crash_stuck_ticks >= self._crash_stuck_steps``. Setting it to 0 does NOT
# disable it -- it makes ``0 >= 0`` true and terminates the route on the very first
# step. A sentinel far above AgentBlockedTest's 60 s (=1200 ticks at 20 Hz) is the
# correct way to hand the decision back to the leaderboard criteria.
CRASH_STUCK_DISABLED = 10**9

FAITHFUL_CARLA_OVERRIDES: dict[str, Any] = {
    "crash_stuck_steps": CRASH_STUCK_DISABLED,
    "terminate_on_infraction": False,
    "max_episode_steps": 0,  # 0 = no wrapper cap
    "repetitions": 1,
    "repetition_index": 0,
    "resume": False,
    "record": "",
    "track": "SENSORS",
}

# carla_job.sh's port scheme, so the two never collide at the same index.
RPC_BASE = int(os.environ.get("CARLA_RPC_BASE", "12000"))
TM_BASE = int(os.environ.get("CARLA_TM_BASE", "18000"))
DISPLAY_BASE = int(os.environ.get("CARLA_DISPLAY_BASE", "30"))
PORT_STRIDE = int(os.environ.get("CARLA_PORT_STRIDE", "100"))

# Worker stdout is dominated by the per-tick PID trace; we use it as a free tick counter
# and speed readout instead of asking the env for a heartbeat.
_RE_PID_SPEED = re.compile(r"\[RC-PID\] Desired speed:\s*([-\d.]+)\s+Current speed:\s*([-\d.]+)")
_RE_PID_TICK = re.compile(r"\[RC-PID\] Steer:")
# Emitted by _finalize_route / _stop_active_scenario.
_RE_STOPPING = re.compile(r"Stopping the route")
_NOISE = re.compile(r"\[RC-PID\]|\[DEBUG - steervla\]")

STATUS_SUCCESS = ("Completed", "Perfect")


# --------------------------------------------------------------------------- #
# Slots                                                                        #
# --------------------------------------------------------------------------- #


@dataclass
class Slot:
    """One concurrent worker: a GPU pair plus an isolated port/display block."""

    index: int
    train_gpu: int
    render_gpu: int
    agent_cfg: Path | None = None

    # Live state, mutated by the reader thread and the main loop.
    route: str | None = None
    proc: subprocess.Popen | None = None
    log_path: Path | None = None
    started_at: float = 0.0
    ticks: int = 0
    speed: float = 0.0
    phase: str = "idle"
    reader: threading.Thread | None = None

    @property
    def rpc_port(self) -> int:
        return RPC_BASE + PORT_STRIDE * self.index

    @property
    def streaming_port(self) -> int:
        return self.rpc_port + 1

    @property
    def tm_port(self) -> int:
        return TM_BASE + PORT_STRIDE * self.index

    @property
    def display(self) -> int:
        return DISPLAY_BASE + self.index

    @property
    def busy(self) -> bool:
        return self.proc is not None

    def reset(self) -> None:
        self.route = None
        self.proc = None
        self.log_path = None
        self.started_at = 0.0
        self.ticks = 0
        self.speed = 0.0
        self.phase = "idle"
        self.reader = None


def parse_slots(spec: str) -> list[Slot]:
    slots: list[Slot] = []
    for i, pair in enumerate(s.strip() for s in spec.split(",") if s.strip()):
        if ":" not in pair:
            raise SystemExit(
                f"--slots entry {pair!r} must be TRAIN_GPU:RENDER_ADAPTER (e.g. '0:1,1:0')"
            )
        train_s, render_s = pair.split(":", 1)
        try:
            slots.append(Slot(index=i, train_gpu=int(train_s), render_gpu=int(render_s)))
        except ValueError:
            raise SystemExit(f"--slots entry {pair!r} has non-integer GPU indices")
    if not slots:
        raise SystemExit("--slots must name at least one TRAIN_GPU:RENDER_ADAPTER pair")
    return slots


# --------------------------------------------------------------------------- #
# Route selection                                                              #
# --------------------------------------------------------------------------- #


def resolve_routes(spec: str) -> list[str]:
    """``all`` / ``bench2drive`` / ``fail2drive`` / comma-list / ``@file`` -> route names."""
    spec = spec.strip()
    if spec in ("all", "bench2drive", "fail2drive"):
        source = None if spec == "all" else spec
        return [e.scenario_name for e in list_routes(source=source)]

    if spec.startswith("@"):
        path = Path(spec[1:]).expanduser()
        if not path.is_file():
            raise SystemExit(f"--routes {spec}: no such file {path}")
        raw = [
            line.strip()
            for line in path.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    else:
        raw = [r.strip() for r in spec.split(",") if r.strip()]

    # Normalise every alias (file basename, route id, f2d:N) to the canonical name so
    # the record filenames and the resume check agree.
    out: list[str] = []
    for name in raw:
        try:
            out.append(find_route(name).scenario_name)
        except Exception as exc:
            raise SystemExit(f"--routes: cannot resolve {name!r}: {exc}")
    return out


# --------------------------------------------------------------------------- #
# Generated per-run configs                                                    #
# --------------------------------------------------------------------------- #


AGENT_CFG_TEMPLATE = '''\
# Generated by run_leaderboard.py -- do not edit; regenerated on every run.
import runpy
import sys

# The base configs import from impls/ as top-level modules (``from jax_agents import ...``).
# main_carla.py puts impls/ on sys.path at import time, but do it here too so this file is
# loadable standalone.
_IMPLS = r"{impls}"
if _IMPLS not in sys.path:
    sys.path.insert(0, _IMPLS)

_BASE = runpy.run_path(r"{base}")["get_config"]


def get_config():
    config = _BASE()
    config.training_gpu_rank = {train_gpu}
    config.siglip_device = "cuda:{train_gpu}"
    # Scoring run: never take a gradient step, whatever the base config says.
    config.enable_updates = False
    for _k in ("enable_updates_rl", "enable_updates_bc", "enable_updates_bc_hl"):
        if _k in config:
            config[_k] = False
    # debug_task swaps env reward for -ego_speed; irrelevant to leaderboard scoring but
    # it would corrupt the returns we log alongside it.
    if "debug_task" in config:
        config.debug_task = False
    # debug_noise draws N extra noise vectors per VLA query purely to log a diagnostic plot
    # (steervla_dsrl_config.py ships it True with log_every_n_steps=1). In a scoring run it
    # is N-times wasted inference, and its _sample_best_of_random_noises path is where an
    # intermittent openpi RMSNorm TypeError has been observed killing a route outright.
    if "debug_noise" in config:
        config.debug_noise = False
    if "steervla" in config:
{steervla_overrides}
        # Greedy CoT: a scoring run must not be stochastic in the reasoning trace.
        config.steervla.cot_temperature = {cot_temperature}
    return config
'''


def write_agent_config(dest: Path, base: Path, slot: Slot, args: argparse.Namespace) -> Path:
    lines: list[str] = []
    if args.steervla_checkpoint:
        lines.append(f'        config.steervla.checkpoint = r"{args.steervla_checkpoint}"')
    if args.steervla_actor_config:
        lines.append(f'        config.steervla.actor_config = r"{args.steervla_actor_config}"')
    if not lines:
        lines.append("        pass")
    dest.write_text(
        AGENT_CFG_TEMPLATE.format(
            impls=str(REPO_ROOT / "impls"),
            base=str(base),
            train_gpu=slot.train_gpu,
            steervla_overrides="\n".join(lines),
            cot_temperature=args.cot_temperature,
        )
    )
    return dest


def write_carla_config(
    dest: Path, base_cfg: dict[str, Any], slot: Slot, record_path: Path, live_path: Path,
    args: argparse.Namespace,
) -> Path:
    cfg = dict(base_cfg)
    cfg.update(FAITHFUL_CARLA_OVERRIDES)
    cfg["crash_stuck_steps"] = args.crash_stuck_steps
    cfg["host"] = args.host
    cfg["port"] = slot.rpc_port
    cfg["streaming_port"] = slot.streaming_port
    cfg["traffic_manager_port"] = slot.tm_port
    cfg["gpu_rank"] = slot.render_gpu
    cfg["x_display_num"] = slot.display
    cfg["traffic_manager_seed"] = args.seed
    cfg["timeout"] = float(args.carla_timeout)
    # The leaderboard StatisticsManager writes here on route end -- this file IS the result.
    cfg["checkpoint"] = str(record_path)
    # Per-route so concurrent slots don't clobber a shared ./live_results.txt. Nothing is
    # written to it at debug=0; StatisticsManager just requires the path.
    cfg["debug_checkpoint"] = str(live_path)
    # Deliberately pinned to 0. debug>1 would give us per-tick live scores to display, but
    # the same value is passed to RouteScenario(debug_mode=...), which calls _draw_waypoints
    # -> world.debug.draw_point over the whole route. Those debug primitives are rendered
    # into the scene and land in the RGB camera the policy reads, so a "nicer dashboard"
    # would silently change the observations being scored. Progress in the UI comes from
    # counting the worker's per-tick PID lines instead.
    cfg["debug"] = 0
    dest.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return dest


# --------------------------------------------------------------------------- #
# Worker process                                                               #
# --------------------------------------------------------------------------- #


def launch(slot: Slot, route: str, run_dir: Path, base_cfg: dict[str, Any],
           args: argparse.Namespace) -> None:
    cfg_dir = run_dir / "configs"
    record_path = run_dir / "records" / f"{route}.json"
    live_path = run_dir / "records" / f"{route}.live.txt"
    carla_cfg = write_carla_config(
        cfg_dir / f"slot{slot.index}_carla.yaml", base_cfg, slot, record_path, live_path, args
    )

    save_dir = run_dir / "runs" / route
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "logs" / f"{route}.log"

    cmd = [
        args.python,
        "impls/main_carla.py",
        f"--agent={slot.agent_cfg}",
        f"--carla_config={carla_cfg}",
        f"--route={route}",
        "--eval_only=true",
        "--max_episodes=1",
        f"--online_steps={args.online_steps}",
        f"--save_dir={save_dir}",
        f"--seed={args.seed}",
        f"--run_group={args.run_group}",
        f"--save_video_local={'true' if args.save_video else 'false'}",
    ]
    cmd.extend(args.extra)

    env = os.environ.copy()
    env["WANDB_MODE"] = args.wandb_mode
    if args.carla_root:
        env["CARLA_ROOT"] = args.carla_root
    # OpenPI's get_cache_dir() mkdirs its cache eagerly at actor startup. Several configs in
    # this repo point at /raid, which is not writable here, so the worker dies with
    # PermissionError: '/raid' before CARLA is ever contacted.
    env["OPENPI_DATA_HOME"] = args.openpi_data_home
    # JAX preallocates ~75% of the card by default. Each worker also puts torch/SigLIP on
    # the same train GPU (~2 GB) and, depending on the slot's render adapter, may share the
    # card with a CarlaUE4 renderer (~7 GB observed). Unfractioned, the OpenPI param restore
    # dies with RESOURCE_EXHAUSTED. Same knob run_cast_pool.sh uses for its workers.
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(args.xla_mem_fraction)
    # Each route is its own process, so without a persistent cache every worker re-runs the
    # Pi0-CoT XLA compile from scratch -- measured at ~17 min on this box, which over 220
    # routes is more wall-clock than the driving itself. The cache is keyed on the HLO, so
    # it is shared safely across slots and across runs of the same checkpoint.
    if args.jax_cache_dir:
        Path(args.jax_cache_dir).expanduser().mkdir(parents=True, exist_ok=True)
        env["JAX_COMPILATION_CACHE_DIR"] = str(Path(args.jax_cache_dir).expanduser())
        env["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "1.0"
        # Default (0) lets JAX pick a filesystem-dependent floor that can skip entries; -1
        # disables the size restriction so the big Pi0 modules are definitely cached.
        env["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"] = "-1"
    # training_gpu_rank indexes jax.devices(), so every GPU must stay visible.
    env.pop("CUDA_VISIBLE_DEVICES", None)

    log_f = open(log_path, "w", buffering=1)
    log_f.write(f"$ {' '.join(cmd)}\n\n")
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),  # get_weather_id() reads data/weather.xml relative to CWD
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,  # own process group, so we can kill the whole tree
    )

    slot.route = route
    slot.proc = proc
    slot.log_path = log_path
    slot.started_at = time.time()
    slot.ticks = 0
    slot.speed = 0.0
    slot.phase = "booting CARLA"

    def _pump() -> None:
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                log_f.write(line)
                if _RE_PID_TICK.search(line):
                    slot.ticks += 1
                    slot.phase = "driving"
                    continue
                m = _RE_PID_SPEED.search(line)
                if m:
                    slot.speed = float(m.group(2))
                    continue
                if _RE_STOPPING.search(line):
                    slot.phase = "scoring"
                elif "RPC ready" in line:
                    slot.phase = "loading route"
                elif "Running the route" in line or "load_world success" in line:
                    slot.phase = "driving"
        except Exception:
            pass
        finally:
            try:
                log_f.close()
            except Exception:
                pass

    slot.reader = threading.Thread(target=_pump, daemon=True)
    slot.reader.start()


def cleanup_slot_carla(slot: Slot) -> None:
    """Scoped kill of the CARLA server + Xvfb this slot owns.

    ``CarlaBench2DriveWrapper._setup_simulation`` launches CarlaUE4 and Xvfb with
    ``preexec_fn=_child_process_setup``, which calls ``setsid`` -- so they live in their own
    process groups and a ``killpg`` on the worker's group does **not** reach them. A worker
    that is killed (timeout, Ctrl-C) or that dies before its atexit hook therefore leaks a
    CARLA server holding ~7 GB of VRAM. Both patterns below are slot-specific (rpc port /
    display number), so this mirrors ``carla_job.sh stop`` and never touches a sibling slot
    or an unrelated ``carla_job.sh`` job.
    """
    patterns = [
        f"carla-rpc-port={slot.rpc_port}",
        f"Xvfb :{slot.display} -screen",
    ]
    for pattern in patterns:
        try:
            subprocess.run(
                ["pkill", "-9", "-f", pattern],
                check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10,
            )
        except Exception:
            pass


def kill_slot(slot: Slot) -> None:
    proc = slot.proc
    if proc is None:
        return
    for sig, wait in ((signal.SIGINT, 10.0), (signal.SIGTERM, 5.0), (signal.SIGKILL, 3.0)):
        if proc.poll() is not None:
            break
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.send_signal(sig)
            except Exception:
                break
        try:
            proc.wait(timeout=wait)
        except subprocess.TimeoutExpired:
            continue
    cleanup_slot_carla(slot)


# --------------------------------------------------------------------------- #
# Result harvesting                                                            #
# --------------------------------------------------------------------------- #


@dataclass
class RouteResult:
    route: str
    status: str
    score_composed: float = 0.0
    score_route: float = 0.0
    score_penalty: float = 0.0
    num_infractions: int = 0
    infractions: dict[str, list[str]] = field(default_factory=dict)
    route_length: float = 0.0
    duration_game: float = 0.0
    duration_system: float = 0.0
    wall_seconds: float = 0.0
    returncode: int | None = None
    attempts: int = 1

    @property
    def success(self) -> bool:
        return self.status in STATUS_SUCCESS


def read_record(path: Path, route: str) -> RouteResult | None:
    """Parse a leaderboard StatisticsManager checkpoint written by one worker."""
    if not path.is_file():
        return None
    try:
        blob = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    records = (blob.get("_checkpoint") or {}).get("records") or []
    if not records:
        return None
    rec = records[-1]
    scores = rec.get("scores") or {}
    meta = rec.get("meta") or {}
    infractions = {k: v for k, v in (rec.get("infractions") or {}).items() if v}
    return RouteResult(
        route=route,
        status=str(rec.get("status", "Unknown")),
        score_composed=float(scores.get("score_composed", 0.0)),
        score_route=float(scores.get("score_route", 0.0)),
        score_penalty=float(scores.get("score_penalty", 0.0)),
        num_infractions=int(rec.get("num_infractions", 0)),
        infractions=infractions,
        route_length=float(meta.get("route_length", 0.0)),
        duration_game=float(meta.get("duration_game", 0.0)),
        duration_system=float(meta.get("duration_system", 0.0)),
    )


def aggregate(results: list[RouteResult]) -> dict[str, Any]:
    if not results:
        return {"n": 0}
    n = len(results)
    total_km = sum(r.route_length for r in results) / 1000.0
    infraction_totals: dict[str, int] = {}
    for r in results:
        for k, v in r.infractions.items():
            infraction_totals[k] = infraction_totals.get(k, 0) + len(v)
    return {
        "n": n,
        "driving_score": sum(r.score_composed for r in results) / n,
        "route_completion": sum(r.score_route for r in results) / n,
        "infraction_penalty": sum(r.score_penalty for r in results) / n,
        "success_rate": 100.0 * sum(r.success for r in results) / n,
        "total_km": total_km,
        "infraction_totals": infraction_totals,
        "infractions_per_km": {
            k: (v / total_km if total_km > 0 else 0.0) for k, v in infraction_totals.items()
        },
    }


def write_summary(run_dir: Path, results: list[RouteResult], pending: list[str],
                  args: argparse.Namespace) -> dict[str, Any]:
    agg = aggregate(results)
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "routes_spec": args.routes,
        "agent_config": str(args.agent_config),
        "steervla_checkpoint": args.steervla_checkpoint,
        "n_completed": len(results),
        "n_pending": len(pending),
        "pending": pending,
        "aggregate": agg,
        "routes": [
            {
                "route": r.route,
                "status": r.status,
                "driving_score": r.score_composed,
                "route_completion": r.score_route,
                "infraction_penalty": r.score_penalty,
                "num_infractions": r.num_infractions,
                "infractions": r.infractions,
                "route_length_m": r.route_length,
                "duration_game_s": r.duration_game,
                "wall_s": round(r.wall_seconds, 1),
                "returncode": r.returncode,
                "attempts": r.attempts,
            }
            for r in sorted(results, key=lambda x: x.route)
        ],
    }
    (run_dir / "leaderboard_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# --------------------------------------------------------------------------- #
# UI                                                                           #
# --------------------------------------------------------------------------- #


def _hms(seconds: float) -> str:
    return str(timedelta(seconds=int(max(0.0, seconds))))


class Dashboard:
    """rich Live dashboard; degrades to periodic plain-text lines when unavailable."""

    def __init__(self, enabled: bool, total: int, run_dir: Path) -> None:
        self.total = total
        self.run_dir = run_dir
        self.started = time.time()
        self.live = None
        self._last_plain = 0.0
        if not enabled:
            return
        try:
            from rich.console import Console
            from rich.live import Live

            self.console = Console()
            if not self.console.is_terminal:
                return
            self.live = Live(console=self.console, refresh_per_second=4, screen=False)
        except ImportError:
            self.live = None

    def __enter__(self) -> Dashboard:
        if self.live is not None:
            self.live.__enter__()
        return self

    def __exit__(self, *exc: object) -> None:
        if self.live is not None:
            self.live.__exit__(*exc)

    def update(self, slots: list[Slot], results: list[RouteResult], queued: int) -> None:
        if self.live is None:
            self._plain(slots, results, queued)
            return
        self.live.update(self._render(slots, results, queued))

    # -- plain fallback ----------------------------------------------------- #
    def _plain(self, slots: list[Slot], results: list[RouteResult], queued: int) -> None:
        now = time.time()
        if now - self._last_plain < 30.0:
            return
        self._last_plain = now
        agg = aggregate(results)
        active = ", ".join(
            f"s{s.index}:{s.route}({s.phase},{s.ticks}t)" for s in slots if s.busy
        ) or "-"
        print(
            f"[leaderboard] {len(results)}/{self.total} done, {queued} queued, "
            f"elapsed {_hms(now - self.started)} | "
            f"DS={agg.get('driving_score', 0.0):.2f} "
            f"RC={agg.get('route_completion', 0.0):.2f} "
            f"IP={agg.get('infraction_penalty', 0.0):.3f} | {active}",
            flush=True,
        )

    # -- rich --------------------------------------------------------------- #
    def _render(self, slots: list[Slot], results: list[RouteResult], queued: int):
        from rich.console import Group
        from rich.panel import Panel
        from rich.table import Table

        elapsed = time.time() - self.started
        done = len(results)
        rate = done / elapsed if elapsed > 0 and done else 0.0
        eta = (self.total - done) / rate if rate > 0 else 0.0
        agg = aggregate(results)

        head = Table.grid(expand=True, padding=(0, 2))
        for _ in range(5):
            head.add_column(justify="left")
        head.add_row(
            f"[bold]{done}[/]/{self.total} routes",
            f"queued [bold]{queued}[/]",
            f"elapsed [bold]{_hms(elapsed)}[/]",
            f"ETA [bold]{_hms(eta) if rate > 0 else '--'}[/]",
            f"[dim]{self.run_dir}[/]",
        )
        scores = Table.grid(expand=True, padding=(0, 2))
        for _ in range(4):
            scores.add_column(justify="left")
        scores.add_row(
            f"Driving Score [bold cyan]{agg.get('driving_score', 0.0):6.2f}[/]",
            f"Route Completion [bold]{agg.get('route_completion', 0.0):6.2f}[/]",
            f"Infraction Penalty [bold]{agg.get('infraction_penalty', 0.0):5.3f}[/]",
            f"Success [bold]{agg.get('success_rate', 0.0):5.1f}%[/]",
        )

        workers = Table(expand=True, box=None, pad_edge=False)
        workers.add_column("slot", width=4)
        workers.add_column("gpu t/r", width=8)
        workers.add_column("rpc", width=6)
        workers.add_column("route", ratio=1, no_wrap=True)
        workers.add_column("phase", width=14)
        workers.add_column("ticks", width=7, justify="right")
        workers.add_column("m/s", width=6, justify="right")
        workers.add_column("elapsed", width=9, justify="right")
        for s in slots:
            if s.busy:
                workers.add_row(
                    str(s.index),
                    f"{s.train_gpu}/{s.render_gpu}",
                    str(s.rpc_port),
                    s.route or "",
                    s.phase,
                    str(s.ticks),
                    f"{s.speed:.1f}",
                    _hms(time.time() - s.started_at),
                )
            else:
                workers.add_row(
                    str(s.index), f"{s.train_gpu}/{s.render_gpu}", str(s.rpc_port),
                    "[dim]idle[/]", "", "", "", "",
                )

        recent = Table(expand=True, box=None, pad_edge=False)
        recent.add_column("route", ratio=1, no_wrap=True)
        recent.add_column("status", width=22, no_wrap=True)
        recent.add_column("DS", width=7, justify="right")
        recent.add_column("RC", width=7, justify="right")
        recent.add_column("IP", width=7, justify="right")
        recent.add_column("infr", width=5, justify="right")
        recent.add_column("game s", width=8, justify="right")
        for r in results[-12:]:
            colour = "green" if r.success else ("yellow" if r.score_route > 1.0 else "red")
            recent.add_row(
                r.route,
                f"[{colour}]{r.status[:22]}[/]",
                f"{r.score_composed:.2f}",
                f"{r.score_route:.2f}",
                f"{r.score_penalty:.3f}",
                str(r.num_infractions),
                f"{r.duration_game:.0f}",
            )

        infr = agg.get("infraction_totals") or {}
        infr_line = "  ".join(
            f"{k.replace('_', ' ')}: [bold]{v}[/]"
            for k, v in sorted(infr.items(), key=lambda kv: -kv[1])[:6]
        ) or "[dim]none recorded[/]"

        return Group(
            Panel(Group(head, scores), title="Bench2Drive leaderboard", border_style="cyan"),
            Panel(workers, title="workers", border_style="blue"),
            Panel(recent, title=f"recent results (last {min(12, len(results))})",
                  border_style="magenta"),
            Panel(infr_line, title="infractions", border_style="yellow"),
        )


def print_final(summary: dict[str, Any], run_dir: Path) -> None:
    agg = summary["aggregate"]
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        t = Table(title=f"Leaderboard results ({summary['n_completed']} routes)")
        t.add_column("metric")
        t.add_column("value", justify="right")
        t.add_row("Driving Score", f"{agg.get('driving_score', 0.0):.3f}")
        t.add_row("Route Completion", f"{agg.get('route_completion', 0.0):.3f}")
        t.add_row("Infraction Penalty", f"{agg.get('infraction_penalty', 0.0):.4f}")
        t.add_row("Success rate", f"{agg.get('success_rate', 0.0):.1f}%")
        t.add_row("Total distance", f"{agg.get('total_km', 0.0):.2f} km")
        console.print(t)
        if agg.get("infractions_per_km"):
            it = Table(title="Infractions per km")
            it.add_column("type")
            it.add_column("total", justify="right")
            it.add_column("per km", justify="right")
            for k, v in sorted(agg["infractions_per_km"].items(), key=lambda kv: -kv[1]):
                it.add_row(k, str(agg["infraction_totals"][k]), f"{v:.3f}")
            console.print(it)
    except ImportError:
        print(json.dumps(agg, indent=2))
    print(f"\nSummary: {run_dir / 'leaderboard_summary.json'}")
    print(f"Records: {run_dir / 'records'}")
    print(f"Logs:    {run_dir / 'logs'}")
    if summary["n_pending"]:
        print(f"\n{summary['n_pending']} route(s) did not finish; re-run with --resume "
              f"--out-dir {run_dir} to pick them up.")


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--slots", default="0:1",
                   help="Comma-separated TRAIN_GPU:RENDER_ADAPTER pairs, one per worker "
                        "(e.g. '0:1,1:0'). Default: 0:1")
    p.add_argument("--routes", default="bench2drive",
                   help="'all' | 'bench2drive' | 'fail2drive' | comma-separated route names "
                        "| @file-with-one-name-per-line. Default: bench2drive")
    p.add_argument("--out-dir", default=None,
                   help="Run directory. Default: leaderboard_runs/<timestamp>")
    p.add_argument("--resume", action="store_true",
                   help="Skip routes that already have a record in <out-dir>/records.")
    p.add_argument("--seed", type=int, default=0,
                   help="Traffic-manager seed and agent seed. Default: 0")

    p.add_argument("--agent-config", default="impls/configs/steervla_dsrl_config.py",
                   help="Base agent config; training_gpu_rank and the update switches are "
                        "overridden per slot.")
    p.add_argument("--steervla-checkpoint", default=None,
                   help="Override config.steervla.checkpoint (gs:// or local).")
    p.add_argument("--steervla-actor-config", default=None,
                   help="Override config.steervla.actor_config.")
    p.add_argument("--cot-temperature", type=float, default=0.0,
                   help="config.steervla.cot_temperature for the run. Default: 0.0 (greedy)")

    p.add_argument("--carla-config", default="impls/configs/carla_config.yaml",
                   help="Base CARLA yaml; the leaderboard-faithful keys are overridden.")
    p.add_argument("--carla-root", default=os.environ.get("CARLA_ROOT"),
                   help="CARLA_ROOT for the workers. Default: inherited from the environment.")
    p.add_argument("--host", default="localhost")
    p.add_argument("--carla-timeout", type=float, default=7200.0,
                   help="CARLA client / watchdog timeout in seconds. Default: 7200")

    p.add_argument("--crash-stuck-steps", type=int, default=CRASH_STUCK_DISABLED,
                   help="Wrapper post-collision stuck cutoff, in ticks. The default is a "
                        "sentinel that hands the decision to the leaderboard's own "
                        "AgentBlockedTest (60 s). NOTE: 0 does not disable it -- it "
                        "terminates every route on step 1.")
    p.add_argument("--online-steps", type=int, default=6000,
                   help="Worker env-step budget. Must exceed Bench2Drive's 4000-tick guard. "
                        "Default: 6000")
    p.add_argument("--route-timeout", type=float, default=5400.0,
                   help="Wall-clock seconds before a hung worker is killed. Default: 5400")
    p.add_argument("--retries", type=int, default=1,
                   help="Re-queue a route this many times if the worker dies without writing a "
                        "record (transient CARLA/JAX crashes are common over 220 routes). A "
                        "route that DID produce a record is never retried. Default: 1")
    p.add_argument("--jax-cache-dir", default=str(Path.home() / ".cache/jax_leaderboard"),
                   help="JAX persistent compilation cache, shared by all workers. The first "
                        "route pays the full Pi0-CoT compile (~17 min); the rest reuse it. "
                        "Pass '' to disable. Default: ~/.cache/jax_leaderboard")
    p.add_argument("--xla-mem-fraction", type=float, default=0.60,
                   help="XLA_PYTHON_CLIENT_MEM_FRACTION per worker. JAX's ~0.75 default "
                        "leaves too little for SigLIP + a co-resident CarlaUE4 renderer on a "
                        "24 GB card. Measured on 2x RTX 4090: 0.45 OOMs during the OpenPI "
                        "restore (it needs one 4.83 GB allocation), 0.60 works. Default: 0.60")
    p.add_argument("--openpi-data-home",
                   default=os.environ.get("OPENPI_DATA_HOME", str(Path.home() / ".cache/openpi")),
                   help="OPENPI_DATA_HOME for the workers -- the OpenPI checkpoint cache. "
                        "Several agent configs default it to /raid, which is not writable on "
                        "every box. Default: $OPENPI_DATA_HOME or ~/.cache/openpi")

    p.add_argument("--save-video", action="store_true",
                   help="Keep each route's rollout MP4 (off by default; 220 routes is a lot).")
    p.add_argument("--wandb-mode", default="disabled", choices=["online", "offline", "disabled"])
    p.add_argument("--run-group", default="leaderboard")
    p.add_argument("--python", default=sys.executable,
                   help="Interpreter for the workers. Default: this interpreter.")
    p.add_argument("--no-ui", action="store_true", help="Plain periodic log lines instead of the "
                                                        "live dashboard.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan (slots, ports, routes, config overrides) and exit.")
    p.add_argument("extra", nargs="*",
                   help="Extra flags forwarded verbatim to impls/main_carla.py "
                        "(put them after a bare --).")
    return p


def main() -> int:
    args = build_parser().parse_args()

    slots = parse_slots(args.slots)
    routes = resolve_routes(args.routes)

    agent_base = (REPO_ROOT / args.agent_config).resolve()
    if not agent_base.is_file():
        raise SystemExit(f"--agent-config not found: {agent_base}")
    carla_base_path = (REPO_ROOT / args.carla_config).resolve()
    if not carla_base_path.is_file():
        raise SystemExit(f"--carla-config not found: {carla_base_path}")
    base_cfg = yaml.safe_load(carla_base_path.read_text())

    if args.out_dir:
        run_dir = Path(args.out_dir).expanduser().resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = REPO_ROOT / "leaderboard_runs" / stamp
    for sub in ("configs", "records", "logs", "runs"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    # Resume: harvest anything already scored, and drop it from the queue.
    results: list[RouteResult] = []
    if args.resume:
        kept = []
        for route in routes:
            existing = read_record(run_dir / "records" / f"{route}.json", route)
            if existing is not None:
                results.append(existing)
            else:
                kept.append(route)
        routes = kept

    if args.crash_stuck_steps == 0:
        print(
            "\033[91m[leaderboard] --crash-stuck-steps=0 terminates every route on its first "
            "step (the check is `ticks >= steps`). Use the default sentinel to defer to "
            "AgentBlockedTest.\033[0m",
            file=sys.stderr,
        )
        return 2

    if args.dry_run:
        print(f"run dir      : {run_dir}")
        print(f"agent config : {agent_base}")
        print(f"carla config : {carla_base_path}")
        print(f"checkpoint   : {args.steervla_checkpoint or '<from agent config>'}")
        print(f"seed         : {args.seed}")
        print("\nleaderboard-faithful overrides:")
        for k, v in {**FAITHFUL_CARLA_OVERRIDES,
                     "crash_stuck_steps": args.crash_stuck_steps}.items():
            print(f"  {k:26s} = {v}")
        print("\nslots:")
        for s in slots:
            print(f"  slot {s.index}: train_gpu={s.train_gpu} render_adapter={s.render_gpu} "
                  f"rpc={s.rpc_port} stream={s.streaming_port} tm={s.tm_port} "
                  f"display=:{s.display}")
        print(f"\n{len(results)} already scored, {len(routes)} queued:")
        for r in routes[:20]:
            print(f"  {r}")
        if len(routes) > 20:
            print(f"  ... and {len(routes) - 20} more")
        return 0

    if not routes:
        print("Nothing to do (all requested routes already have records).")
        if results:
            print_final(write_summary(run_dir, results, [], args), run_dir)
        return 0

    for slot in slots:
        slot.agent_cfg = write_agent_config(
            run_dir / "configs" / f"slot{slot.index}_agent.py", agent_base, slot, args
        )

    total = len(routes) + len(results)
    queue = list(routes)
    attempts: dict[str, int] = {}
    stopping = False

    def _on_sigint(_signum: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, _on_sigint)
    signal.signal(signal.SIGTERM, _on_sigint)

    print(f"[leaderboard] {len(queue)} route(s) over {len(slots)} slot(s) -> {run_dir}", flush=True)

    try:
        with Dashboard(not args.no_ui, total, run_dir) as ui:
            while (queue and not stopping) or any(s.busy for s in slots):
                if stopping:
                    # Tear down immediately rather than waiting out the in-flight routes --
                    # a leaderboard route can run for the better part of an hour, and a
                    # Ctrl-C that appears to hang invites a SIGKILL that orphans CARLA.
                    for slot in slots:
                        if slot.busy:
                            slot.phase = "interrupt-kill"
                            kill_slot(slot)
                for slot in slots:
                    if slot.busy:
                        assert slot.proc is not None
                        wall = time.time() - slot.started_at
                        timed_out = wall > args.route_timeout
                        if timed_out and slot.proc.poll() is None:
                            slot.phase = "timeout-kill"
                            kill_slot(slot)
                        if slot.proc.poll() is None:
                            continue

                        rc = slot.proc.returncode
                        route = slot.route or "?"
                        if slot.reader is not None:
                            slot.reader.join(timeout=5.0)
                        res = read_record(run_dir / "records" / f"{route}.json", route)

                        if res is None and attempts.get(route, 1) <= args.retries and not stopping:
                            # No record written at all -> the worker died before scoring
                            # (segfault at import, CARLA boot failure, OOM). Worth one more go.
                            # A route that DID score is never retried, however bad the score.
                            attempts[route] = attempts.get(route, 1) + 1
                            failed_log = run_dir / "logs" / f"{route}.attempt{attempts[route] - 1}.log"
                            if slot.log_path and slot.log_path.exists():
                                slot.log_path.rename(failed_log)
                            print(
                                f"[leaderboard] {route}: no record "
                                f"({'timeout' if timed_out else f'rc={rc}'}); "
                                f"retry {attempts[route] - 1}/{args.retries} "
                                f"(previous log: {failed_log.name})",
                                flush=True,
                            )
                            queue.append(route)
                            cleanup_slot_carla(slot)
                            slot.reset()
                            continue

                        if res is None and stopping:
                            # Killed by our own Ctrl-C, not by a failure. Leave it unscored so
                            # it lands in `pending` for --resume instead of dragging the mean
                            # down with a bogus 0.
                            queue.append(route)
                            cleanup_slot_carla(slot)
                            slot.reset()
                            continue

                        if res is None:
                            res = RouteResult(
                                route=route,
                                status="Timeout (killed)" if timed_out else f"NoRecord (rc={rc})",
                            )
                        res.wall_seconds = wall
                        res.returncode = rc
                        res.attempts = attempts.get(route, 1)
                        results.append(res)
                        write_summary(run_dir, results, queue + [
                            s.route for s in slots if s.busy and s.route and s.route != route
                        ], args)
                        cleanup_slot_carla(slot)
                        slot.reset()

                    if not slot.busy and queue and not stopping:
                        nxt = queue.pop(0)
                        attempts.setdefault(nxt, 1)
                        launch(slot, nxt, run_dir, base_cfg, args)

                ui.update(slots, results, len(queue))
                time.sleep(0.25)
    finally:
        if stopping:
            print("\n[leaderboard] interrupted; stopping workers...", flush=True)
        for slot in slots:
            if slot.busy:
                kill_slot(slot)

    pending = queue + [s.route for s in slots if s.route]
    summary = write_summary(run_dir, results, [p for p in pending if p], args)
    print_final(summary, run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
