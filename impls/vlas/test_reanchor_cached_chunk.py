"""Standalone check for ``SteerVLAActor._reanchor_route_to_current_pose``.

Not a pytest suite (this repo has none) -- run it directly, like
``impls/coaches/test_action_chunk_feedback_integration.py``::

    .venv/bin/python impls/vlas/test_reanchor_cached_chunk.py

Two parts:

1. **Correctness** -- the SE(2) transform is checked against an independent world-frame
   round-trip (ego frame -> world -> new ego frame) over random poses, and the speed columns
   are asserted untouched.
2. **Effect** -- the real ``SimlingoStyleWaypointDecoder`` is driven over a 5-tick hold with
   re-anchoring off vs on, showing what the fix actually buys: a lateral loop that stays closed
   while a chunk is being replayed.

Neither part needs CARLA, jax or openpi: ``carla`` is stubbed (only ``VehicleControl`` is
referenced, and nothing here calls it) and the methods under test are lifted out of
``steervla.py`` by AST so importing the full actor module is unnecessary.
"""

from __future__ import annotations

import ast
import contextlib
import io
import sys
import types
from pathlib import Path
from typing import ClassVar

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# The decoder imports ``carla`` at module scope for VehicleControl, which no test here calls.
sys.modules.setdefault("carla", types.ModuleType("carla")).VehicleControl = object

from ogbench.carla.steervla_simlingo_control import SimlingoStyleWaypointDecoder

_METHODS = (
    "_ego_pose_from_state",
    "_current_ego_pose",
    "_reanchor_disabled",
    "_reanchor_route_to_current_pose",
)
H, D = 10, 4
FORMAT = "DELTA_XY_T_DELTA_XY_SPACE"


def _load_actor_stub():
    """Return a stub class carrying the real re-anchor methods, no jax/openpi import."""
    src = (_REPO_ROOT / "impls" / "vlas" / "steervla.py").read_text()
    cls = next(
        n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "SteerVLAActor"
    )
    body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in _METHODS]
    assert len(body) == len(_METHODS), f"expected {_METHODS} in SteerVLAActor, found {len(body)}"
    ns: dict = {"np": np, "Any": object}
    exec(  # noqa: S102 -- lifting real methods out of steervla.py beats duplicating them here
        compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), "<ast>", "exec"), ns
    )

    class Actor:
        _EGO_STATE_IDX_X, _EGO_STATE_IDX_Y, _EGO_STATE_IDX_YAW_DEG = 0, 1, 5
        _ROUTE_XY_FORMATS = frozenset({"delta_xy_t_delta_xy_space"})
        reanchor_cached_chunk = True
        action_horizon, action_dim = H, D
        output_action_format = FORMAT
        _reanchor_disabled_reason = None
        last_reanchor: ClassVar[dict] = {}
        raw_obs_holder = None

    for name in _METHODS:
        setattr(Actor, name, ns[name])
    return Actor


Actor = _load_actor_stub()


def _world_to_ego(p, origin, yaw):
    """Matches ``carla_utils._compute_target_point_ego``: R(yaw)^T (p - origin), x forward."""
    c, s = np.cos(yaw), np.sin(yaw)
    d = np.asarray(p) - np.asarray(origin)
    return np.array([d[0] * c + d[1] * s, -d[0] * s + d[1] * c])


def _ego_to_world(p, origin, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.asarray(origin) + np.array([p[0] * c - p[1] * s, p[0] * s + p[1] * c])


def _state_vec(pos, yaw_rad, speed):
    """Minimal ``obs['state']``: x, y at 0/1, yaw (deg) at 5, speed at 15."""
    v = np.zeros(25, dtype=np.float32)
    v[0], v[1] = pos
    v[5] = np.degrees(yaw_rad)
    v[15] = speed
    return v


def _make_actor(pose_query, state_now, *, reanchor):
    a = Actor()
    a.reanchor_cached_chunk = reanchor
    a._cached_action_pose = np.asarray(pose_query, dtype=np.float64)
    a.raw_obs_holder = {"obs": {"state": state_now}}
    return a


def test_transform_matches_world_frame(trials: int = 200) -> None:
    """Re-anchored waypoints must equal the same world points seen from the new ego pose."""
    rng = np.random.default_rng(0)
    for trial in range(trials):
        deltas = np.stack([rng.uniform(0.8, 1.2, H), rng.uniform(-0.25, 0.25, H)], axis=1)
        chunk = np.zeros((1, H, D), dtype=np.float32)
        chunk[0, :, 2:4] = deltas
        chunk[0, :, 0:2] = rng.normal(size=(H, 2))  # speed columns must survive untouched
        flat = chunk.reshape(1, H * D)

        origin_q = rng.uniform(-200, 200, 2)
        yaw_q = rng.uniform(-np.pi, np.pi)
        origin_n = _ego_to_world([rng.uniform(0.0, 1.5), rng.uniform(-0.05, 0.05)], origin_q, yaw_q)
        yaw_n = yaw_q + rng.uniform(-0.15, 0.15)

        actor = _make_actor([*origin_q, yaw_q], _state_vec(origin_n, yaw_n, 10.0), reanchor=True)
        with contextlib.redirect_stdout(io.StringIO()):
            out = np.asarray(actor._reanchor_route_to_current_pose(flat.copy(), flat)).reshape(1, H, D)

        assert np.allclose(out[0, :, 0:2], chunk[0, :, 0:2]), f"trial {trial}: speed columns changed"

        pts_q = np.cumsum(np.vstack([np.zeros(2), deltas]), axis=0)
        truth = np.array([_world_to_ego(_ego_to_world(p, origin_q, yaw_q), origin_n, yaw_n) for p in pts_q])
        # What the decoder reconstructs from the re-anchored deltas.
        got = np.cumsum(np.vstack([np.zeros(2), out[0, :, 2:4]]), axis=0)[1:]

        dropped = int(actor.last_reanchor.get("dropped", 0))
        padded = int(actor.last_reanchor.get("padded", 0))
        kept = H - padded
        ref = truth[dropped : dropped + kept]
        err = float(np.abs(got[:kept] - ref).max())
        assert err < 1e-4, f"trial {trial}: max waypoint error {err}"
        assert (ref[:, 0] > 0).all(), f"trial {trial}: kept a waypoint behind the ego"
    print(f"[ok] SE(2) re-anchor matches world-frame ground truth over {trials} random poses")


def _arc_chunk(radius: float, target_speed: float):
    """A constant-curvature route chunk, 1 m waypoint spacing, plus matching speed columns."""
    pts = np.array([[radius * np.sin(k / radius), radius * (1 - np.cos(k / radius))] for k in range(H + 1)])
    chunk = np.zeros((H, D), dtype=np.float32)
    chunk[:, 2:4] = np.diff(pts, axis=0)
    # desired_speed = |wp[2] - wp[0]| * 2 with the *7 denorm on the speed columns => 28 * delta.
    chunk[:, 0] = target_speed / 28.0
    return chunk.reshape(1, H * D), pts


def test_hold_stays_closed_loop(radius: float = 30.0, speed: float = 10.0, dt: float = 0.05) -> None:
    """Drive a 5-tick hold with the ego pushed off the plan; off = blind, on = corrects."""
    flat, _ = _arc_chunk(radius, speed)

    def sweep(drift_per_tick: float, yaw_err_deg_per_tick: float, label: str):
        decoders = {m: SimlingoStyleWaypointDecoder() for m in ("off", "on")}
        rows = []
        for tick in range(5):
            s = speed * dt * tick
            yaw = s / radius
            pos = np.array([radius * np.sin(yaw), radius * (1 - np.cos(yaw))])
            pos = pos + np.array([-np.sin(yaw), np.cos(yaw)]) * (drift_per_tick * tick)
            yaw_err = np.deg2rad(yaw_err_deg_per_tick * tick)
            state = _state_vec(pos, yaw + yaw_err, speed)
            row = {}
            for mode in ("off", "on"):
                actor = _make_actor([0.0, 0.0, 0.0], state, reanchor=(mode == "on"))
                with contextlib.redirect_stdout(io.StringIO()):
                    served = (
                        flat
                        if tick == 0
                        else np.asarray(actor._reanchor_route_to_current_pose(flat.copy(), flat))
                    )
                    decoders[mode]._flat_action_to_pid(
                        served,
                        state_vec=state,
                        output_action_format=FORMAT,
                        action_horizon=H,
                        action_dim=D,
                        action_input_space="normalized",
                    )
                row[mode] = decoders[mode].last_debug
            rows.append(row)

        print(f"\n{label}")
        print(f"{'tick':>4} | {'OFF steer':>10} {'OFF herr':>9} | {'ON steer':>10} {'ON herr':>9}")
        for tick, row in enumerate(rows):
            print(
                f"{tick:>4} | {row['off']['steer']:>10.3f} {row['off']['heading_error']:>9.4f} | "
                f"{row['on']['steer']:>10.3f} {row['on']['heading_error']:>9.4f}"
            )
        return rows

    # On-plan: re-anchoring must be a no-op, since there is no tracking error to remove.
    rows = sweep(0.0, 0.0, "A) ego tracks the plan exactly -- re-anchoring changes nothing")
    for tick, row in enumerate(rows):
        assert abs(row["off"]["steer"] - row["on"]["steer"]) < 5e-3, f"tick {tick}: on-plan mismatch"

    # Off-plan: the un-re-anchored chunk is frozen, the re-anchored one steers back.
    for drift, yaw_rate, label, expect in (
        (0.10, 0.0, "B) ego drifts 0.1 m/tick off the plan laterally", "lateral"),
        (0.0, -1.0, "C) ego yaw lags the plan by 1 deg/tick", "yaw"),
    ):
        rows = sweep(drift, yaw_rate, label)
        off = [r["off"]["steer"] for r in rows]
        on = [r["on"]["steer"] for r in rows]
        assert max(off) - min(off) < 1e-3, f"{expect}: un-re-anchored steer should be frozen, got {off}"
        assert abs(on[-1] - on[0]) > 0.1, f"{expect}: re-anchored steer should respond, got {on}"
        direction = -1.0 if expect == "lateral" else 1.0
        assert direction * (on[-1] - on[0]) > 0, f"{expect}: correction has the wrong sign: {on}"
    print("\n[ok] re-anchored replay keeps the lateral loop closed; un-re-anchored replay is blind")


if __name__ == "__main__":
    test_transform_matches_world_frame()
    test_hold_stays_closed_loop()
    print("\nAll checks passed.")
