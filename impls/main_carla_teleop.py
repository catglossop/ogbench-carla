"""Manual/semi-manual CARLA driving on top of the SteerVLA + SAC-residual policy.

Two modes, both built on candidate sampling shared with the closed-loop policy
(`main_carla.py:_sample_agent_action`), executed against the same CARLA env
plumbing (`main_carla._make_carla_env` / `main_carla.build_carla_session`):

  --mode=interactive  At each decision point, sample --num_candidates policy
                       rollouts (different SteerVLA subtask + action chunk
                       each), overlay their predicted trajectories (as projected
                       waypoints) on the first-person camera frame in distinct
                       colors, and let a human pick one via a small local web UI
                       (reach it over SSH with `ssh -L <port>:localhost:<port>`;
                       reject the whole batch and resample instead of picking).
                       Only the chosen candidate is ever executed -- no residual
                       actor, straight from the base policy, same as bestofn --
                       for --rollout_horizon ticks (its full action chunk by
                       default 10, matching bestofn) before the next decision.

  --mode=bestofn       At each decision point, actually roll each candidate
                       forward in the real simulator for --rollout_horizon
                       steps, sum its reward, then use CarlaBench2DriveWrapper's
                       checkpoint()/restore() (or the CarlaEnvSubprocess wire
                       equivalent) to teleport back and try the next one --
                       no episode reset, no learned critic. Commits whichever
                       candidate scored highest.

Run via run_carla_teleop.sh, which mirrors run_carla.sh's CARLA 0.9.15
subprocess / port setup.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import cv2
import jax
import numpy as np
import wandb
from absl import app, flags

import main_carla
from carla_teleop_server import REJECT, TeleopServer
from ogbench.carla.waypoint_viz import annotate_waypoints_on_frame
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, setup_wandb

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    "mode", "interactive", ["interactive", "bestofn", "auto_then_manual", "wall_snapshot", "wall_video"],
    "interactive: human picks a candidate via the web UI. "
    "bestofn: literally roll out + checkpoint/restore each candidate, commit the best. "
    "auto_then_manual: closed-loop autonomous rollout (single candidate, no picking) "
    "until the car gets stuck, then falls through to interactive-style manual picking "
    "for the rest of the episode. "
    "wall_snapshot: debug-only -- drive straight (raw throttle) to within "
    "--obstacle_offset_m of the nearest scenario obstacle prop, sample --num_candidates "
    "candidates, overlay them + a color-coded caption on one frame, save it as a PNG, "
    "and exit (no episode rollout). "
    "wall_video: debug-only -- same straight-line drive, but continuously (every "
    "--video_sample_every_ticks) samples --num_candidates candidates from the policy "
    "and records an annotated frame, saving the whole approach as an MP4 instead of "
    "one final PNG.",
)
flags.DEFINE_integer("num_candidates", 4, "Number of policy candidates sampled per decision point.")
flags.DEFINE_integer(
    "rollout_horizon", 1,
    "bestofn only: env steps to execute (and sum reward over) per candidate trial.",
)
flags.DEFINE_float(
    "decision_timeout_sec", 8.0,
    "interactive only: seconds to wait for a human click before auto-picking candidate 0.",
)
flags.DEFINE_integer("web_port", 8000, "interactive only: local web UI port.")
flags.DEFINE_integer(
    "num_decisions", 4000,
    "bestofn only: safety cap on total decision points across the whole job (all episodes "
    "combined), in case an episode never terminates. interactive only: number of decision "
    "points to run before exiting (interactive mode has no episode-count flag).",
)
flags.DEFINE_integer("num_episodes", 10, "bestofn only: number of full episodes to run before exiting.")
flags.DEFINE_float(
    "cot_temperature", 1.0,
    "Overrides the SteerVLA actor's cot_temperature (default 0.0 = greedy/deterministic "
    "subtask decoding, meaning every candidate at a decision point gets the *same* subtask "
    "text no matter how many times it's resampled -- only the action-flow noise would vary). "
    "A nonzero value here is required for candidate subtasks to actually differ.",
)
flags.DEFINE_integer(
    "stuck_decisions_trigger", 3,
    "auto_then_manual only: consecutive decision points (each --rollout_horizon ticks) "
    "ending below --stuck_speed_threshold during the autonomous phase before switching "
    "to manual (interactive-style) mode. Trigger is speed-based, not wall-specific, so "
    "it fires for any real obstacle/blockage the base policy can't get past.",
)
flags.DEFINE_float(
    "stuck_speed_threshold", 0.3,
    "auto_then_manual only: speed (m/s) below which a decision counts toward "
    "stuck_decisions_trigger.",
)
flags.DEFINE_float(
    "obstacle_trigger_distance_m", 15.0,
    "auto_then_manual only: hand off to manual mode once "
    "info['nearest_obstacle_distance_m'] (ground-truth distance to a scenario "
    "obstacle prop, e.g. Fail2Drive's brickwall -- see "
    "CarlaBench2DriveWrapper.nearest_obstacle_distance_m) drops below this. Takes "
    "priority over the speed-based stuck fallback whenever such a prop is present.",
)
flags.DEFINE_float(
    "obstacle_offset_m", 10.0,
    "wall_snapshot only: XY distance (m) to the nearest obstacle prop at which the "
    "straight-line drive-forward phase stops and the candidate snapshot is taken. "
    "Driving (not teleporting) all the way there is what makes the obstacle actually "
    "spawn/render -- Bench2Drive stages some scenario props off-map until the "
    "leaderboard's own route-progress trigger fires from real driving progress.",
)
flags.DEFINE_float(
    "drive_forward_throttle", 0.4,
    "wall_snapshot only: constant raw VehicleControl throttle (steer=0) used for the "
    "straight-line drive-forward phase, bypassing the policy (which is cautious "
    "around traffic/junctions and can get stuck well short of a distant obstacle).",
)
flags.DEFINE_integer(
    "drive_forward_max_ticks", 3000,
    "wall_snapshot only: safety cap on CARLA ticks for the straight-line "
    "drive-forward phase, in case the obstacle is never reached (e.g. blocked by "
    "traffic or a collision brings the ego to a stop short of --obstacle_offset_m).",
)
flags.DEFINE_string(
    "snapshot_path", "",
    "wall_snapshot only: output PNG path. Defaults to <out_dir>/wall_snapshot.png.",
)
flags.DEFINE_integer(
    "video_sample_every_ticks", 10,
    "wall_video only, --drive_mode=straight: sample --num_candidates candidates from "
    "the policy and record an annotated frame every this many driving ticks (full "
    "16-candidate sampling is too slow to do every single tick).",
)
flags.DEFINE_enum(
    "drive_mode", "straight", ["straight", "policy", "expert"],
    "wall_video only. straight: raw full-throttle straight-line driving toward the "
    "nearest obstacle prop (see run_wall_video) -- only sensible when the route is "
    "known to run straight to the target, e.g. a Fail2Drive wall route. policy: "
    "drive via the base policy's own single-candidate action each decision (like "
    "auto_then_manual's AUTO phase) -- for routes with turns/junctions/traffic where "
    "a raw straight line wouldn't track the road, e.g. enter-actor-flow-004. expert: "
    "drive via the live PDM-Lite/SimLingo autopilot (env.step_expert()) instead of "
    "either the policy or raw controls -- the policy is still sampled periodically "
    "(--num_candidates every --video_sample_every_decisions decisions) purely for "
    "visualization/recording, never applied to driving.",
)
flags.DEFINE_integer(
    "video_sample_every_decisions", 3,
    "wall_video only, --drive_mode=policy: sample --num_candidates candidates and "
    "record an annotated frame every this many decisions (each --rollout_horizon "
    "ticks); the executed action each decision is always a single fresh sample "
    "(cheap), independent of whether that decision is recorded.",
)
flags.DEFINE_integer(
    "policy_video_max_decisions", 60,
    "wall_video only, --drive_mode=policy: safety cap on decisions (each "
    "--rollout_horizon ticks) before stopping, since there's no obstacle-distance "
    "stop condition for a general route.",
)
flags.DEFINE_float(
    "drive_forward_slowdown_distance_m", 30.0,
    "wall_video only, --drive_mode=straight: throttle drops to "
    "--drive_forward_slow_throttle once within this XY distance of the obstacle, "
    "to avoid overshooting the stop window. Set to 0 to disable and drive at "
    "--drive_forward_throttle the whole way.",
)
flags.DEFINE_float(
    "drive_forward_slow_throttle", 0.12,
    "wall_video only, --drive_mode=straight: throttle used once within "
    "--drive_forward_slowdown_distance_m of the obstacle.",
)
flags.DEFINE_boolean(
    "drive_through", False,
    "wall_video only, --drive_mode=straight: keep driving straight through "
    "--obstacle_offset_m and any collision instead of stopping there -- only "
    "stops on natural episode termination or --drive_forward_max_ticks. Useful "
    "when recording every --video_sample_every_ticks frames anyway and a crash "
    "partway through is an acceptable outcome, not something to avoid.",
)
flags.DEFINE_integer("video_fps", 4, "wall_video only: output MP4 playback frame rate.")
flags.DEFINE_string(
    "video_path", "",
    "wall_video only: output MP4 path. Defaults to <out_dir>/wall_approach.mp4.",
)

_PALETTE = [
    (255, 0, 0), (0, 200, 0), (30, 144, 255), (255, 140, 0),
    (200, 0, 200), (0, 220, 220),
]


def _rgb_to_hex(color: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*color)


def _sanitize_csv_field(text: str) -> str:
    return text.replace(",", ";").replace("\n", " ").strip()


def _sample_base_action(agent, obs_batch, rng) -> np.ndarray:
    """Sample directly from the frozen Pi0/SteerVLA base policy -- no residual actor.

    Matches main_carla.py:_sample_agent_action's warmup-path base rollout
    (``base = agent._clip_actions_to_env(agent.vla_sample_fn(obs, noise))``): the
    full predicted waypoint chunk goes straight to ``env.step()``, which decodes it
    via its own persistent per-episode PID controller
    (``CarlaBench2DriveWrapper._steervla_decoder`` / ``maybe_steervla_vehicle_control``)
    regardless of the agent config's ``residual_action_space`` -- so there's no need
    to PID-decode client-side here even when the attached agent happens to be a
    SAC-residual one; we simply never call its residual sampling methods.
    """
    noise = jax.random.normal(rng, (obs_batch.shape[0], agent._flat_noise_dim()))
    chunk_jax = agent._clip_actions_to_env(jax.numpy.asarray(agent.vla_sample_fn(obs_batch, noise)))
    return np.asarray(jax.device_get(chunk_jax[0]), dtype=np.float32)


# Keyword categories used to tell candidates apart at a glance: "accelerate" vs
# "decelerate" vs "turn right" vs "adjust left", etc. Deliberately coarse (substring
# match on the VLA's own subtask text) -- good enough to reject near-duplicate
# candidates without needing real NLP.
_SUBTASK_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "accelerate": ("accelerat",),
    "decelerate": ("decelerat", "brak", "slow"),
    "stop": ("stop", "stopped", "remain"),
    "turn_right": ("turn right", "turns right", "right turn", "rightward turn"),
    "turn_left": ("turn left", "turns left", "left turn", "leftward turn"),
    "adjust_right": ("adjustment to the right", "right adjustment", "adjusts right", "rightward adjustment"),
    "adjust_left": ("adjustment to the left", "left adjustment", "adjusts left", "leftward adjustment"),
    "maintain": ("maintain", "steady course"),
    "reverse": ("revers",),
}
_MAX_SAMPLE_ATTEMPTS_PER_CANDIDATE = 6
# A candidate must share less than this fraction of its category tags (Jaccard) with
# every already-accepted candidate to count as "diverse enough" and stop early.
_DIVERSITY_JACCARD_THRESHOLD = 0.5


def _subtask_categories(text: str) -> frozenset[str]:
    t = text.lower()
    return frozenset(
        cat for cat, keywords in _SUBTASK_CATEGORY_KEYWORDS.items() if any(kw in t for kw in keywords)
    )


def _category_jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 1.0  # can't tell them apart from keywords alone -> treat as maximally similar
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def _diversity_score(cats: frozenset[str], accepted_cats: list[frozenset[str]]) -> float:
    """Higher is better: how different ``cats`` is from every already-accepted candidate."""
    if not accepted_cats:
        return 1.0
    return 1.0 - max(_category_jaccard(cats, existing) for existing in accepted_cats)


def _sample_candidates(
    session: "main_carla.CarlaSession", k: int, rng: jax.Array,
) -> tuple[list[tuple[np.ndarray, str]], jax.Array]:
    """Sample ``k`` (base_action_chunk, subtask_text) candidates for the current obs.

    For each slot beyond the first, resamples up to _MAX_SAMPLE_ATTEMPTS_PER_CANDIDATE
    times and keeps whichever draw is most different (by subtask keyword category) from
    the candidates already accepted, stopping early once one clears
    _DIVERSITY_JACCARD_THRESHOLD. Some states genuinely only afford one sensible action
    (e.g. stopped at a red light) -- in that case this just falls back to the best of
    the attempted draws rather than forcing artificial diversity that isn't there.
    """
    raw_obs = session.raw_carla_holder["obs"]
    agent_obs = main_carla._extract_agent_obs(
        session.env, raw_obs, session.obs_mode,
        image_encoder=session.image_encoder,
        siglip_encoder=session.siglip_encoder,
        siglip_include_prompt_subtask=session.siglip_include_prompt_subtask,
        steervla_actor=session.steervla_actor,
    )
    obs_batch = agent_obs[None]

    def _draw(rng_key):
        if session.steervla_actor is not None:
            # Force a fresh CoT + action sample per draw; the actor otherwise caches
            # one action chunk per env-step cadence (chunk_size=1 convention), which
            # would make every draw identical.
            session.steervla_actor.reset_action_cache()
        action = _sample_base_action(session.agent, obs_batch, rng_key)
        _prompt, subtask = main_carla._steervla_prompt_subtask_strings(raw_obs, session.steervla_actor)
        return action, subtask, _subtask_categories(subtask)

    candidates: list[tuple[np.ndarray, str]] = []
    accepted_cats: list[frozenset[str]] = []
    for slot in range(k):
        best = None
        best_score = -1.0
        attempts = 1 if slot == 0 else _MAX_SAMPLE_ATTEMPTS_PER_CANDIDATE
        for _attempt in range(attempts):
            rng, sub = jax.random.split(rng)
            action, subtask, cats = _draw(sub)
            score = _diversity_score(cats, accepted_cats)
            if score > best_score:
                best, best_score = (action, subtask, cats), score
            if score >= (1.0 - _DIVERSITY_JACCARD_THRESHOLD):
                break
        action, subtask, cats = best
        candidates.append((action, subtask))
        accepted_cats.append(cats)
    return candidates, rng


def _current_frame(session: "main_carla.CarlaSession") -> tuple[np.ndarray, Optional[np.ndarray]]:
    raw = session.raw_carla_holder["obs"]
    frame = raw.get("image_viz")
    if frame is None:
        frame = raw["image"]
    return np.asarray(frame), raw.get("target_points")


def _overlay_candidates(
    frame: np.ndarray, target_points, candidates: list[tuple[np.ndarray, str]], exec_cfg,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    annotated = frame
    legend = []
    for i, (action, subtask) in enumerate(candidates):
        color = _PALETTE[i % len(_PALETTE)]
        annotated = annotate_waypoints_on_frame(
            annotated, action_flat=action, exec_cfg=exec_cfg,
            target_points=target_points if i == 0 else None,
            route_color=color, speed_color=color, label=str(i + 1),
        )
        legend.append({"index": i, "color": _rgb_to_hex(color), "subtask": subtask})
    return annotated, legend


def _annotate_reward_corner(frame: np.ndarray, reward_value: float) -> np.ndarray:
    """Small reward readout in the top-left corner. Mirrors main_carla.py:_annotate_reward_corner."""
    return _annotate_label_corner(frame, f"r={reward_value:+.3f}")


def _annotate_label_corner(frame: np.ndarray, label: str) -> np.ndarray:
    """Small text readout in the top-left corner (same box style as _annotate_reward_corner)."""
    annotated = np.array(frame, copy=True)
    font_scale = 0.38
    thickness = 1
    pad = 4
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x0, y0 = 6, 6
    x1, y1 = x0 + tw + 2 * pad, y0 + th + baseline + 2 * pad
    cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 0, 0), thickness=-1)
    cv2.rectangle(annotated, (x0, y0), (x1, y1), (255, 255, 255), thickness=1)
    cv2.putText(
        annotated, label, (x0 + pad, y1 - baseline - pad),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
    )
    return annotated


def _clip_text(txt: str, max_chars: int = 120) -> str:
    return txt if len(txt) <= max_chars else (txt[: max_chars - 3] + "...")


def _write_video_mp4(path: str, frames: list[np.ndarray], fps: int = 10) -> None:
    """Saves annotated frames as a local .mp4, independent of wandb (works even with
    --wandb_mode=disabled, which is the common case for an interactive session)."""
    if not frames:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    print(f"[main_carla_teleop] wrote {len(frames)} frames to {path}", flush=True)


def _annotate_rollout_frame(
    frame: np.ndarray,
    raw_obs: dict[str, Any],
    *,
    action_flat: np.ndarray,
    exec_cfg,
    reward: float,
    subtask: str,
    decision: int,
    candidate_index: int,
    num_candidates: int,
    role: str,
) -> np.ndarray:
    """Waypoint overlay + reward/speed/subtask text panel for one env tick.

    Same visual language as main_carla.py's episode-video path
    (_annotate_waypoints + _annotate_reward_corner + _annotate_text_panel), trimmed
    to what applies here: no residual/critic fields, since main_carla_teleop.py
    samples straight from the base policy. ``role`` is "COMMIT" for the actually-
    executed winner (the main episode video) or "TRIAL" for a candidate's
    speculative scoring rollout (the debug video) -- burned into the panel so the
    two are never confused when scrubbing through footage.
    """
    annotated = annotate_waypoints_on_frame(
        frame, action_flat=action_flat, exec_cfg=exec_cfg,
        target_points=raw_obs.get("target_points"),
    )
    annotated = _annotate_reward_corner(annotated, reward)

    h, w = annotated.shape[:2]
    font_scale = 0.26
    line_h = 13
    panel_h = line_h * 3 + 8
    # H.264 (what wandb.Video encodes to) needs even width/height for 4:2:0 chroma
    # subsampling. An odd total height decodes fine in permissive readers (OpenCV)
    # but can fail to render in stricter players, including wandb's web player.
    if (h + panel_h) % 2 != 0:
        panel_h += 1
    panel = np.zeros((h + panel_h, w, 3), dtype=np.uint8)
    panel[:h, :, :] = annotated
    cv2.line(panel, (0, h), (w - 1, h), (255, 255, 255), 1)

    state = np.asarray(raw_obs.get("state", []), dtype=np.float32).reshape(-1)
    speed_idx = main_carla._EGO_STATE_IDX_SPEED
    speed = float(state[speed_idx]) if state.size > speed_idx else 0.0
    routing = str(raw_obs.get("routing_command", "") or "").strip()

    lines = [
        f"[{role}] decision={decision}  candidate {candidate_index + 1}/{num_candidates}  "
        f"spd={speed:.2f} m/s  {routing or 'Follow the route.'}",
        f"Subtask: {_clip_text(subtask)}",
    ]
    y = h + line_h
    for line in lines:
        cv2.putText(
            panel, line, (4, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA,
        )
        y += line_h
    return panel


def _validate_session(session: "main_carla.CarlaSession") -> None:
    if session.agent is None or session.raw_carla_holder is None or session.steervla_actor is None:
        raise ValueError(
            "main_carla_teleop.py requires a SteerVLA-rollout agent config "
            "(steervla.enabled=True, e.g. impls/configs/pi0_residual_sac_config.py)."
        )
    if getattr(session.agent, "vla_sample_fn", None) is None:
        raise ValueError("main_carla_teleop.py requires agent.vla_sample_fn (frozen Pi0/SteerVLA base policy).")


def _manual_pick(
    session: "main_carla.CarlaSession", server: TeleopServer, rng: jax.Array, decision: int,
) -> tuple[list[tuple[np.ndarray, str]], int, jax.Array]:
    """Blocks until the human picks a candidate; REJECT loops for a fresh resample."""
    rejects = 0
    while True:
        candidates, rng = _sample_candidates(session, FLAGS.num_candidates, rng)
        frame, target_points = _current_frame(session)
        annotated, legend = _overlay_candidates(frame, target_points, candidates, session.exec_cfg)
        server.publish(annotated, legend, FLAGS.decision_timeout_sec)

        choice = server.wait_for_choice(FLAGS.decision_timeout_sec, default_index=0)
        if choice == REJECT:
            rejects += 1
            print(f"[main_carla_teleop] decision {decision}: batch rejected (#{rejects}), resampling...", flush=True)
            continue
        return candidates, max(0, min(choice, len(candidates) - 1)), rng


def _execute_chosen(
    session: "main_carla.CarlaSession",
    action: np.ndarray,
    subtask: str,
    *,
    decision: int,
    choice: int,
    num_candidates: int,
    horizon: int,
    episode_frames: list[np.ndarray],
    role: str,
) -> tuple[float, bool, bool, Optional[dict[str, Any]], dict[str, Any]]:
    """Steps env for ``horizon`` ticks with ``action``, capturing annotated frames.

    Same PID-tracked chunk execution as bestofn's commit loop: the full predicted
    waypoint chunk gets fed to env.step() for ``horizon`` ticks (its own action_horizon
    by default, e.g. 10), not just one tick, before the next decision point resamples.
    Returns ``(total_reward, terminated, truncated, obs_dict, last_info)``.
    """
    total_reward = 0.0
    terminated = truncated = False
    obs_dict = None
    info: dict[str, Any] = {}
    for _h in range(horizon):
        obs_dict, reward, terminated, truncated, info = session.env.step(action)
        total_reward += float(reward)
        cam_frame = obs_dict.get("image_viz")
        if cam_frame is None:
            cam_frame = obs_dict.get("image")
        if cam_frame is not None:
            episode_frames.append(
                _annotate_rollout_frame(
                    np.asarray(cam_frame), obs_dict,
                    action_flat=action, exec_cfg=session.exec_cfg,
                    reward=float(reward), subtask=subtask,
                    decision=decision, candidate_index=choice,
                    num_candidates=num_candidates, role=role,
                )
            )
        if terminated or truncated:
            break
    return total_reward, terminated, truncated, obs_dict, info


def run_interactive(session: "main_carla.CarlaSession", out_dir: str) -> None:
    server = TeleopServer(port=FLAGS.web_port)
    server.start()
    rng = jax.random.PRNGKey(FLAGS.seed)
    horizon = max(1, int(FLAGS.rollout_horizon))
    csv_logger = CsvLogger(os.path.join(out_dir, "teleop_interactive.csv"))
    episode_index = 0
    episode_frames: list[np.ndarray] = []
    try:
        for decision in range(FLAGS.num_decisions):
            candidates, choice, rng = _manual_pick(session, server, rng, decision)
            action, subtask = candidates[choice]

            total_reward, terminated, truncated, obs_dict, _info = _execute_chosen(
                session, action, subtask,
                decision=decision, choice=choice, num_candidates=len(candidates),
                horizon=horizon, episode_frames=episode_frames, role="COMMIT",
            )
            session.raw_carla_holder["obs"] = obs_dict
            session.raw_carla_holder["next_obs"] = obs_dict

            csv_logger.log(
                {
                    "decision": decision,
                    "chosen_index": choice,
                    "num_candidates": len(candidates),
                    "subtask": _sanitize_csv_field(subtask),
                    "reward": total_reward,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                },
                step=decision,
            )
            print(
                f"[main_carla_teleop] decision {decision}: human chose candidate {choice} "
                f"reward={total_reward:.3f} subtask={subtask!r}",
                flush=True,
            )

            if terminated or truncated:
                _write_video_mp4(
                    os.path.join(out_dir, "videos", f"episode_{episode_index}.mp4"), episode_frames,
                )
                episode_frames = []
                episode_index += 1
                obs_dict, _info = session.env.reset(seed=FLAGS.seed)
                session.raw_carla_holder["obs"] = obs_dict
                session.raw_carla_holder["next_obs"] = obs_dict
    finally:
        # Flush whatever's left of the current (unterminated) episode too, e.g. if
        # the job hit --num_decisions or the session was stopped mid-episode.
        _write_video_mp4(
            os.path.join(out_dir, "videos", f"episode_{episode_index}_partial.mp4"), episode_frames,
        )
        csv_logger.close()
        server.stop()


def run_auto_then_manual(session: "main_carla.CarlaSession", out_dir: str) -> None:
    """Closed-loop autonomous rollout (single candidate, no picking) until the car gets
    stuck, then falls through to interactive-style manual picking for the rest of the
    episode. Two handoff triggers, checked in this order:

    1. Ground truth: env.step()'s info["nearest_obstacle_distance_m"] (see
       CarlaBench2DriveWrapper.nearest_obstacle_distance_m) drops below
       --stuck_speed_threshold-independent --obstacle_trigger_distance_m -- the real
       CARLA position of a scenario obstacle prop (e.g. Fail2Drive's brickwall), not a
       proxy. Only available when such a prop exists in the world (-1.0 otherwise).
    2. Fallback: --stuck_decisions_trigger consecutive decisions ending below
       --stuck_speed_threshold m/s, for routes/scenarios with no matching prop.
    """
    server = TeleopServer(port=FLAGS.web_port)
    server.start()
    rng = jax.random.PRNGKey(FLAGS.seed)
    horizon = max(1, int(FLAGS.rollout_horizon))
    stuck_trigger = max(1, int(FLAGS.stuck_decisions_trigger))
    stuck_speed_threshold = float(FLAGS.stuck_speed_threshold)
    obstacle_trigger_m = float(FLAGS.obstacle_trigger_distance_m)
    speed_idx = main_carla._EGO_STATE_IDX_SPEED
    csv_logger = CsvLogger(os.path.join(out_dir, "teleop_auto_then_manual.csv"))
    episode_index = 0
    episode_frames: list[np.ndarray] = []
    manual = False
    stuck_decisions = 0
    try:
        for decision in range(FLAGS.num_decisions):
            if manual:
                candidates, choice, rng = _manual_pick(session, server, rng, decision)
            else:
                candidates, rng = _sample_candidates(session, 1, rng)
                choice = 0
                # Read-only preview during autonomous driving (nothing waited on).
                frame, target_points = _current_frame(session)
                annotated, legend = _overlay_candidates(frame, target_points, candidates, session.exec_cfg)
                server.publish(annotated, legend, 0.0)
            action, subtask = candidates[choice]

            role = "MANUAL" if manual else "AUTO"
            total_reward, terminated, truncated, obs_dict, info = _execute_chosen(
                session, action, subtask,
                decision=decision, choice=choice, num_candidates=len(candidates),
                horizon=horizon, episode_frames=episode_frames, role=role,
            )
            session.raw_carla_holder["obs"] = obs_dict
            session.raw_carla_holder["next_obs"] = obs_dict

            obstacle_dist = float(info.get("nearest_obstacle_distance_m", -1.0))
            if not manual:
                state = np.asarray(obs_dict.get("state", []), dtype=np.float32).reshape(-1)
                speed = float(state[speed_idx]) if state.size > speed_idx else 0.0
                if obstacle_dist >= 0.0:
                    trigger_reason = f"obstacle {obstacle_dist:.1f}m away" if obstacle_dist <= obstacle_trigger_m else None
                else:
                    stuck_decisions = stuck_decisions + 1 if speed < stuck_speed_threshold else 0
                    trigger_reason = (
                        f"stuck (speed<{stuck_speed_threshold} m/s) for {stuck_decisions} decisions"
                        if stuck_decisions >= stuck_trigger else None
                    )
                if trigger_reason is not None:
                    manual = True
                    stuck_decisions = 0
                    print(
                        f"[main_carla_teleop] decision {decision}: autonomous rollout handed off to "
                        f"MANUAL mode ({trigger_reason})",
                        flush=True,
                    )

            csv_logger.log(
                {
                    "decision": decision,
                    "mode": "manual" if manual else "auto",
                    "chosen_index": choice,
                    "num_candidates": len(candidates),
                    "subtask": _sanitize_csv_field(subtask),
                    "reward": total_reward,
                    "nearest_obstacle_distance_m": obstacle_dist,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                },
                step=decision,
            )
            print(
                f"[main_carla_teleop] decision {decision} [{'MANUAL' if manual else 'AUTO'}]: "
                f"{'human' if manual else 'policy'} chose candidate {choice} "
                f"reward={total_reward:.3f} obstacle_dist={obstacle_dist:.1f}m subtask={subtask!r}",
                flush=True,
            )

            if terminated or truncated:
                _write_video_mp4(
                    os.path.join(out_dir, "videos", f"episode_{episode_index}.mp4"), episode_frames,
                )
                episode_frames = []
                episode_index += 1
                manual = False
                stuck_decisions = 0
                obs_dict, _info = session.env.reset(seed=FLAGS.seed + episode_index)
                session.raw_carla_holder["obs"] = obs_dict
                session.raw_carla_holder["next_obs"] = obs_dict
    finally:
        _write_video_mp4(
            os.path.join(out_dir, "videos", f"episode_{episode_index}_partial.mp4"), episode_frames,
        )
        csv_logger.close()
        server.stop()


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _render_caption_block(
    width: int, legend: list[dict[str, Any]], *, line_h: int = 16, pad: int = 6,
) -> np.ndarray:
    """One color-coded caption line per candidate ("N: subtask text"), stacked
    below the image. Text is clipped to fit the image width; wraps to a second
    line if still too long for one line at this font size."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.34
    thickness = 1
    max_chars_per_line = max(20, int(width / 6.2))
    lines: list[tuple[str, tuple[int, int, int]]] = []
    for c in legend:
        color = _hex_to_rgb(c["color"])
        text = f"{c['index'] + 1}: {c['subtask'] or '(no subtask)'}"
        while len(text) > max_chars_per_line:
            lines.append((text[:max_chars_per_line], color))
            text = "   " + text[max_chars_per_line:]
        lines.append((text, color))

    block_h = pad * 2 + line_h * len(lines)
    block = np.full((block_h, width, 3), 20, dtype=np.uint8)
    for i, (text, color) in enumerate(lines):
        y = pad + line_h * i + line_h - 4
        cv2.putText(block, text, (pad, y), font, font_scale, color, thickness, cv2.LINE_AA)
    return block


def run_wall_snapshot(session: "main_carla.CarlaSession", out_dir: str) -> None:
    """Debug-only: drive straight forward (raw throttle, bypassing the policy) until
    close to the nearest obstacle prop, then snapshot --num_candidates candidate
    overlays on one frame with a color-coded subtask caption below it. No episode
    rollout, no wandb logging.

    Driving there for real (not teleporting) matters: Bench2Drive stages some
    scenario props (e.g. Fail2Drive's brickwall) off the road at an unrelated Z
    until the leaderboard's own route-progress trigger fires from genuine driving
    progress -- a raw actor teleport doesn't fire that trigger, so the obstacle
    stays invisible/uncollidable even though the ego's XY position is correct.
    Driving via the policy instead is cautious around traffic/junctions and can
    take hundreds of decisions (or get stuck) before covering real distance, so
    this uses a raw straight-line VehicleControl to get there quickly.
    """
    if not hasattr(session.env, "drive_straight_until_close"):
        raise RuntimeError("wall_snapshot requires an env with drive_straight_until_close() (CarlaEnvSubprocess).")

    result = session.env.drive_straight_until_close(
        target_distance_m=FLAGS.obstacle_offset_m,
        max_ticks=FLAGS.drive_forward_max_ticks,
        throttle=FLAGS.drive_forward_throttle,
    )
    print(f"[main_carla_teleop] wall_snapshot: drive_straight_until_close reached={result is not None}", flush=True)
    if result is None:
        print(
            f"[main_carla_teleop] wall_snapshot: did not get within "
            f"{FLAGS.obstacle_offset_m}m of an obstacle prop within "
            f"{FLAGS.drive_forward_max_ticks} ticks (no obstacle in this world, or "
            f"blocked by traffic along the way) -- is --fail2drive true and the "
            f"route correct?",
            flush=True,
        )
        return

    # Use the observation read directly server-side at the moment the drive loop
    # stopped -- no extra env.step()/tick here, since some obstacle props get
    # removed from the world shortly after a collision resolves.
    session.raw_carla_holder["obs"] = result["obs"]
    session.raw_carla_holder["next_obs"] = result["obs"]
    obstacle_dist = float(result["nearest_obstacle_distance_m"])
    collision_count = int(result["collision_count"])
    print(
        f"[main_carla_teleop] wall_snapshot: post-drive obstacle_dist={obstacle_dist:.2f}m "
        f"collision_count={collision_count}",
        flush=True,
    )

    rng = jax.random.PRNGKey(FLAGS.seed)
    candidates, rng = _sample_candidates(session, FLAGS.num_candidates, rng)
    frame, target_points = _current_frame(session)
    annotated, legend = _overlay_candidates(frame, target_points, candidates, session.exec_cfg)
    caption = _render_caption_block(annotated.shape[1], legend)
    composed = np.vstack([np.ascontiguousarray(annotated), caption])

    path = FLAGS.snapshot_path or os.path.join(out_dir, "wall_snapshot.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, cv2.cvtColor(composed, cv2.COLOR_RGB2BGR))
    print(
        f"[main_carla_teleop] wall_snapshot: wrote {len(candidates)} candidates "
        f"(obstacle_dist={obstacle_dist:.2f}m, collision_count={collision_count}) to {path}",
        flush=True,
    )
    for c in legend:
        print(f"  candidate {c['index'] + 1} [{c['color']}]: {c['subtask']!r}", flush=True)


# _ego_state_vector's layout (ogbench/carla/carla_utils.py) puts rot.yaw at index 5:
# [loc.x, loc.y, loc.z, rot.roll, rot.pitch, rot.yaw, vel.x, ...].
_EGO_STATE_IDX_YAW = 5


def run_wall_video(session: "main_carla.CarlaSession", out_dir: str) -> None:
    """Dispatches on --drive_mode: 'straight' (default) for routes known to run
    straight to a target obstacle (see _run_wall_video_straight); 'policy' for
    general routes with turns/junctions/traffic where a raw straight line wouldn't
    track the road (see _run_wall_video_policy); 'expert' to drive via the live
    PDM-Lite/SimLingo autopilot instead of the policy, while still sampling the
    policy periodically purely for visualization (see _run_wall_video_expert)."""
    if FLAGS.drive_mode == "policy":
        _run_wall_video_policy(session, out_dir)
    elif FLAGS.drive_mode == "expert":
        _run_wall_video_expert(session, out_dir)
    else:
        _run_wall_video_straight(session, out_dir)


def _run_wall_video_expert(session: "main_carla.CarlaSession", out_dir: str) -> None:
    """Debug-only: drive via the live PDM-Lite/SimLingo autopilot (env.step_expert()),
    while periodically (every --video_sample_every_decisions ticks) sampling
    --num_candidates candidates from the policy purely for visualization -- never
    applied to driving. Requires the env to have been built with
    extra_carla_config={"expert_controller": "simlingo_autopilot"} (done automatically
    in main() when --mode=wall_video --drive_mode=expert), otherwise step_expert()
    silently falls back to a degenerate zero action. For isolating "what would the
    policy propose here" from "how do we actually get the car around this route" on
    general routes (turns, junctions, traffic), e.g. enter-actor-flow-004. Stops at
    --policy_video_max_decisions * --rollout_horizon ticks or episode end. Each
    recorded frame is also written immediately as its own PNG under
    <out_dir>/wall_video_frames/.
    """
    if not hasattr(session.env, "step_expert"):
        raise RuntimeError("wall_video --drive_mode=expert requires an env with step_expert() (CarlaEnvSubprocess).")

    frames_dir = os.path.join(out_dir, "wall_video_frames")
    os.makedirs(frames_dir, exist_ok=True)
    print(f"[main_carla_teleop] wall_video (expert): saving frames incrementally to {frames_dir}", flush=True)

    max_ticks = int(FLAGS.policy_video_max_decisions) * max(1, int(FLAGS.rollout_horizon))
    sample_every = max(1, int(FLAGS.video_sample_every_decisions))
    rng = jax.random.PRNGKey(FLAGS.seed)
    frames: list[np.ndarray] = []
    stop_reason = "max_ticks"
    for tick in range(max_ticks):
        obs_dict, reward, terminated, truncated, info = session.env.step_expert()
        session.raw_carla_holder["obs"] = obs_dict
        session.raw_carla_holder["next_obs"] = obs_dict

        record = tick % sample_every == 0
        if record:
            candidates, rng = _sample_candidates(session, FLAGS.num_candidates, rng)
            frame, target_points = _current_frame(session)
            annotated, legend = _overlay_candidates(frame, target_points, candidates, session.exec_cfg)
            annotated = _annotate_label_corner(annotated, f"expert tick {tick} r={reward:+.2f}")
            caption = _render_caption_block(annotated.shape[1], legend)
            composed = np.vstack([np.ascontiguousarray(annotated), caption])
            frames.append(composed)
            frame_path = os.path.join(frames_dir, f"frame_{len(frames) - 1:04d}_tick{tick:04d}.png")
            cv2.imwrite(frame_path, cv2.cvtColor(composed, cv2.COLOR_RGB2BGR))

        print(
            f"[main_carla_teleop] wall_video (expert): tick {tick} reward={reward:.3f} recorded={record}",
            flush=True,
        )
        if terminated or truncated:
            stop_reason = "episode_ended"
            break

    print(
        f"[main_carla_teleop] wall_video (expert): stopped ({stop_reason}) after "
        f"{len(frames)} recorded frames (saved incrementally to {frames_dir})",
        flush=True,
    )
    path = FLAGS.video_path or os.path.join(out_dir, "wall_approach.mp4")
    _write_video_mp4(path, frames, fps=int(FLAGS.video_fps))
    print(f"[main_carla_teleop] wall_video (expert): wrote {len(frames)} frames to {path}", flush=True)


def _run_wall_video_policy(session: "main_carla.CarlaSession", out_dir: str) -> None:
    """Debug-only: drive via the base policy's own single-candidate action each
    decision (like auto_then_manual's AUTO phase -- proper env.step(), so lane-
    following/turns/traffic are handled the same as any other closed-loop run),
    periodically (every --video_sample_every_decisions decisions) sampling
    --num_candidates candidates and recording an annotated frame, saving the whole
    rollout as an MP4. For general routes (turns, junctions, traffic) where
    _run_wall_video_straight's raw straight-line drive wouldn't track the road.
    Stops at --policy_video_max_decisions or episode end. Each recorded frame is
    also written immediately as its own PNG under <out_dir>/wall_video_frames/.
    """
    frames_dir = os.path.join(out_dir, "wall_video_frames")
    os.makedirs(frames_dir, exist_ok=True)
    print(f"[main_carla_teleop] wall_video (policy): saving frames incrementally to {frames_dir}", flush=True)

    horizon = max(1, int(FLAGS.rollout_horizon))
    sample_every = max(1, int(FLAGS.video_sample_every_decisions))
    rng = jax.random.PRNGKey(FLAGS.seed)
    frames: list[np.ndarray] = []
    stop_reason = "max_decisions"
    for decision in range(int(FLAGS.policy_video_max_decisions)):
        record = decision % sample_every == 0
        num_sample = FLAGS.num_candidates if record else 1
        candidates, rng = _sample_candidates(session, num_sample, rng)
        action, subtask = candidates[0]

        total_reward, terminated, truncated, obs_dict, info = _execute_chosen(
            session, action, subtask, decision=decision, choice=0,
            num_candidates=len(candidates), horizon=horizon, episode_frames=[], role="AUTO",
        )
        session.raw_carla_holder["obs"] = obs_dict
        session.raw_carla_holder["next_obs"] = obs_dict

        if record:
            frame, target_points = _current_frame(session)
            annotated, legend = _overlay_candidates(frame, target_points, candidates, session.exec_cfg)
            annotated = _annotate_label_corner(annotated, f"decision {decision} r={total_reward:+.2f}")
            caption = _render_caption_block(annotated.shape[1], legend)
            composed = np.vstack([np.ascontiguousarray(annotated), caption])
            frames.append(composed)
            frame_path = os.path.join(frames_dir, f"frame_{len(frames) - 1:04d}_decision{decision:04d}.png")
            cv2.imwrite(frame_path, cv2.cvtColor(composed, cv2.COLOR_RGB2BGR))

        print(
            f"[main_carla_teleop] wall_video (policy): decision {decision} reward={total_reward:.3f} "
            f"subtask={subtask!r} recorded={record}",
            flush=True,
        )
        if terminated or truncated:
            stop_reason = "episode_ended"
            break

    print(
        f"[main_carla_teleop] wall_video (policy): stopped ({stop_reason}) after "
        f"{len(frames)} recorded frames (saved incrementally to {frames_dir})",
        flush=True,
    )
    path = FLAGS.video_path or os.path.join(out_dir, "wall_approach.mp4")
    _write_video_mp4(path, frames, fps=int(FLAGS.video_fps))
    print(f"[main_carla_teleop] wall_video (policy): wrote {len(frames)} frames to {path}", flush=True)


def _run_wall_video_straight(session: "main_carla.CarlaSession", out_dir: str) -> None:
    """Debug-only: drive straight forward (raw throttle, bypassing the policy) toward
    the nearest obstacle prop, continuously (client-side, every
    --video_sample_every_ticks) sampling --num_candidates candidates from the policy
    and recording an annotated frame, saving the whole approach as an MP4. Unlike
    wall_snapshot's single blocking drive_straight_until_close() call, this drives
    tick-by-tick from the client so the (client-side, JAX) policy can be queried
    throughout the approach, not just once at the end. Each recorded frame is also
    written immediately as its own PNG under <out_dir>/wall_video_frames/, so
    progress can be inspected while the job is still running (the MP4 itself is
    only written once, at the very end). No episode rollout beyond this one drive,
    no wandb logging. See run_wall_snapshot's docstring for why a raw straight-line
    drive (not a teleport, not the cautious policy) is used to actually reach the
    obstacle.
    """
    if not hasattr(session.env, "step_raw_control"):
        raise RuntimeError("wall_video requires an env with step_raw_control() (CarlaEnvSubprocess).")

    target_distance_m = float(FLAGS.obstacle_offset_m)
    slowdown_distance_m = float(FLAGS.drive_forward_slowdown_distance_m)
    throttle = float(FLAGS.drive_forward_throttle)
    slow_throttle = float(FLAGS.drive_forward_slow_throttle)
    sample_every = max(1, int(FLAGS.video_sample_every_ticks))

    obs_dict = session.raw_carla_holder["obs"]
    state = np.asarray(obs_dict["state"], dtype=np.float32).reshape(-1)
    initial_yaw = float(state[_EGO_STATE_IDX_YAW]) if state.size > _EGO_STATE_IDX_YAW else 0.0

    frames_dir = os.path.join(out_dir, "wall_video_frames")
    os.makedirs(frames_dir, exist_ok=True)
    print(f"[main_carla_teleop] wall_video (straight): saving frames incrementally to {frames_dir}", flush=True)

    rng = jax.random.PRNGKey(FLAGS.seed)
    frames: list[np.ndarray] = []
    obstacle_dist = -1.0
    collision_count = 0
    stop_reason = "max_ticks"
    for tick in range(int(FLAGS.drive_forward_max_ticks)):
        state = np.asarray(session.raw_carla_holder["obs"]["state"], dtype=np.float32).reshape(-1)
        current_yaw = float(state[_EGO_STATE_IDX_YAW]) if state.size > _EGO_STATE_IDX_YAW else 0.0
        yaw_error = ((current_yaw - initial_yaw + 180.0) % 360.0) - 180.0
        steer = float(np.clip(-0.02 * yaw_error, -0.3, 0.3))
        near = obstacle_dist >= 0.0 and obstacle_dist <= slowdown_distance_m
        cur_throttle = slow_throttle if near else throttle

        result = session.env.step_raw_control(cur_throttle, steer, brake=0.0)
        session.raw_carla_holder["obs"] = result["obs"]
        session.raw_carla_holder["next_obs"] = result["obs"]
        obstacle_dist = float(result["nearest_obstacle_distance_m"])
        new_collision_count = int(result["collision_count"])
        collided = new_collision_count > collision_count
        collision_count = new_collision_count
        running = bool(result["running"])

        record = (tick % sample_every == 0) or collided or not running
        if record:
            candidates, rng = _sample_candidates(session, FLAGS.num_candidates, rng)
            frame, target_points = _current_frame(session)
            annotated, legend = _overlay_candidates(frame, target_points, candidates, session.exec_cfg)
            annotated = _annotate_label_corner(annotated, f"dist={obstacle_dist:.1f}m")
            caption = _render_caption_block(annotated.shape[1], legend)
            composed = np.vstack([np.ascontiguousarray(annotated), caption])
            frames.append(composed)
            frame_path = os.path.join(frames_dir, f"frame_{len(frames) - 1:04d}_tick{tick:05d}.png")
            cv2.imwrite(frame_path, cv2.cvtColor(composed, cv2.COLOR_RGB2BGR))

        if not running:
            stop_reason = "episode_ended"
            break
        if not FLAGS.drive_through:
            if collided:
                stop_reason = "collision"
                break
            if obstacle_dist >= 0.0 and obstacle_dist <= target_distance_m:
                stop_reason = "reached_target_distance"
                break

    print(
        f"[main_carla_teleop] wall_video (straight): stopped ({stop_reason}) after {len(frames)} recorded frames "
        f"(saved incrementally to {frames_dir}), obstacle_dist={obstacle_dist:.2f}m "
        f"collision_count={collision_count}",
        flush=True,
    )
    path = FLAGS.video_path or os.path.join(out_dir, "wall_approach.mp4")
    _write_video_mp4(path, frames, fps=int(FLAGS.video_fps))
    print(f"[main_carla_teleop] wall_video (straight): wrote {len(frames)} frames to {path}", flush=True)


def run_bestofn(session: "main_carla.CarlaSession", out_dir: str) -> None:
    """Runs --num_episodes full episodes of literal-rollback best-of-N, logging to wandb.

    An "episode" here means: keep making decisions until the env reports
    terminated/truncated, then reset and start the next one. --num_decisions acts
    only as a total safety cap across the whole job, in case an episode never ends.
    """
    rng = jax.random.PRNGKey(FLAGS.seed)
    csv_logger = CsvLogger(os.path.join(out_dir, "teleop_bestofn.csv"))
    horizon = max(1, int(FLAGS.rollout_horizon))
    num_episodes = max(1, int(FLAGS.num_episodes))
    max_decisions = max(1, int(FLAGS.num_decisions))
    decision = 0
    try:
        for episode in range(num_episodes):
            episode_return = 0.0
            episode_decisions = 0
            terminated = truncated = False
            info: dict = {}
            episode_frames: list[np.ndarray] = []
            episode_debug_frames: list[np.ndarray] = []
            while not (terminated or truncated):
                if decision >= max_decisions:
                    print(
                        f"[main_carla_teleop] hit --num_decisions safety cap ({max_decisions}) "
                        f"mid-episode {episode}; stopping the job.",
                        flush=True,
                    )
                    wandb.log(
                        {
                            "episode/return": episode_return,
                            "episode/length": episode_decisions,
                            "episode/index": episode,
                            "episode/hit_safety_cap": True,
                        },
                        step=decision,
                    )
                    return

                candidates, rng = _sample_candidates(session, FLAGS.num_candidates, rng)
                ckpt = session.env.checkpoint()

                scored = []
                for i, (action, subtask) in enumerate(candidates):
                    session.env.restore(ckpt)
                    total_reward = 0.0
                    cand_terminated = cand_truncated = False
                    for _h in range(horizon):
                        _obs_dict, reward, cand_terminated, cand_truncated, _info = session.env.step(action)
                        total_reward += float(reward)
                        frame = _obs_dict.get("image_viz")
                        if frame is None:
                            frame = _obs_dict.get("image")
                        if frame is not None:
                            episode_debug_frames.append(
                                _annotate_rollout_frame(
                                    np.asarray(frame), _obs_dict,
                                    action_flat=action, exec_cfg=session.exec_cfg,
                                    reward=float(reward), subtask=subtask,
                                    decision=decision, candidate_index=i,
                                    num_candidates=len(candidates), role="TRIAL",
                                )
                            )
                        if cand_terminated or cand_truncated:
                            break
                    scored.append(
                        {"index": i, "action": action, "subtask": subtask, "reward": total_reward}
                    )

                best = max(scored, key=lambda c: c["reward"])
                session.env.restore(ckpt)
                terminated = truncated = False
                obs_dict = None
                committed_reward = 0.0
                for _h in range(horizon):
                    obs_dict, reward, terminated, truncated, info = session.env.step(best["action"])
                    committed_reward += float(reward)
                    frame = obs_dict.get("image_viz")
                    if frame is None:
                        frame = obs_dict.get("image")
                    if frame is not None:
                        episode_frames.append(
                            _annotate_rollout_frame(
                                np.asarray(frame), obs_dict,
                                action_flat=best["action"], exec_cfg=session.exec_cfg,
                                reward=float(reward), subtask=best["subtask"],
                                decision=decision, candidate_index=best["index"],
                                num_candidates=len(candidates), role="COMMIT",
                            )
                        )
                    if terminated or truncated:
                        break
                session.raw_carla_holder["obs"] = obs_dict
                session.raw_carla_holder["next_obs"] = obs_dict
                episode_return += committed_reward
                episode_decisions += 1
                decision += 1

                candidate_scores = [round(c["reward"], 3) for c in scored]
                print(
                    f"[main_carla_teleop] episode {episode} decision {episode_decisions}: "
                    f"chose candidate {best['index']} reward={committed_reward:.3f} "
                    f"subtask={best['subtask']!r} (scores={candidate_scores})",
                    flush=True,
                )
                for i, c in enumerate(scored):
                    csv_logger.log(
                        {
                            "episode": episode,
                            "decision": decision,
                            "candidate": i,
                            "subtask": _sanitize_csv_field(c["subtask"]),
                            "reward": c["reward"],
                            "chosen": i == best["index"],
                        },
                        step=decision * FLAGS.num_candidates + i,
                    )
                wandb.log(
                    {
                        "decision/committed_reward": committed_reward,
                        "decision/chosen_index": best["index"],
                        "decision/score_max": max(candidate_scores),
                        "decision/score_min": min(candidate_scores),
                        "decision/score_spread": max(candidate_scores) - min(candidate_scores),
                        "episode/index": episode,
                        "episode/running_return": episode_return,
                    },
                    step=decision,
                )

            for key, value in info.items():
                if isinstance(value, (int, float, bool)) and not isinstance(value, str):
                    wandb.log({f"episode_end/{key}": value}, step=decision)
            episode_log = {
                "episode/return": episode_return,
                "episode/length": episode_decisions,
                "episode/index": episode,
            }
            if episode_frames:
                # Committed-frames-only video: every real (executed, non-speculative) tick
                # of this episode, in order -- the candidate-scoring trial frames never
                # appear here since they're rolled back before the video ever sees them.
                _write_video_mp4(
                    os.path.join(out_dir, "videos", f"episode_{episode}.mp4"), episode_frames,
                )
                video = np.stack(episode_frames, axis=0)
                if video.ndim == 4:
                    video = np.transpose(video, (0, 3, 1, 2))  # W&B expects (T, C, H, W)
                episode_log["episode/video"] = wandb.Video(video, fps=10, format="mp4")
            if episode_debug_frames:
                # Debug: every candidate's speculative scoring rollout too (all of them,
                # not just the winner), in decision -> candidate order. Frames are labeled
                # "[TRIAL] candidate i/N" so they're never mistaken for the committed video.
                _write_video_mp4(
                    os.path.join(out_dir, "videos", f"episode_{episode}_debug_rollouts.mp4"),
                    episode_debug_frames,
                )
                debug_video = np.stack(episode_debug_frames, axis=0)
                if debug_video.ndim == 4:
                    debug_video = np.transpose(debug_video, (0, 3, 1, 2))
                episode_log["episode/debug_rollouts_video"] = wandb.Video(debug_video, fps=10, format="mp4")
            wandb.log(episode_log, step=decision)
            print(
                f"[main_carla_teleop] === episode {episode} finished: "
                f"return={episode_return:.3f} length={episode_decisions} ===",
                flush=True,
            )

            if episode < num_episodes - 1:
                obs_dict, _info = session.env.reset(seed=FLAGS.seed + episode + 1)
                session.raw_carla_holder["obs"] = obs_dict
                session.raw_carla_holder["next_obs"] = obs_dict
    finally:
        csv_logger.close()


def main(_):
    if FLAGS.list_routes:
        main_carla._list_routes_and_exit()
        return

    config = FLAGS.agent

    wandb_mode = main_carla._resolve_wandb_mode()
    exp_name = f"{FLAGS.mode}_{get_exp_name(FLAGS.seed)}"
    if FLAGS.route:
        exp_name = f"{exp_name}_{FLAGS.route}"
    setup_wandb(project="OGBench-CARLA", group=FLAGS.run_group, name=exp_name, mode=wandb_mode)
    out_dir = os.path.join(FLAGS.save_dir, "teleop", wandb.run.project, FLAGS.run_group, exp_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "flags.json"), "w") as f:
        json.dump(get_flag_dict(), f)

    carla_yaml, extra_carla, exec_cfg = main_carla._resolve_carla_env_config(config)
    if FLAGS.mode == "wall_video" and FLAGS.drive_mode == "expert":
        # Makes env.step_expert() drive via the live PDM-Lite/SimLingo autopilot
        # (carla_utils.py:CarlaBench2DriveWrapper.step_expert Path A) instead of its
        # degenerate zero-action fallback.
        extra_carla["expert_controller"] = "simlingo_autopilot"
    env = main_carla._make_carla_env(carla_yaml, FLAGS.route, extra_carla_config=extra_carla)
    try:
        session = main_carla.build_carla_session(config, env, exec_cfg=exec_cfg)
        _validate_session(session)
        if session.steervla_actor is not None:
            # Candidate diversity depends on this: the agent config's default
            # cot_temperature=0.0 means greedy/deterministic subtask decoding, so every
            # candidate at a decision point would get the identical subtask text
            # regardless of how many times _sample_candidates resamples.
            session.steervla_actor.cot_temperature = float(FLAGS.cot_temperature)
            print(f"[main_carla_teleop] cot_temperature={FLAGS.cot_temperature}", flush=True)

        if FLAGS.mode == "interactive":
            run_interactive(session, out_dir)
        elif FLAGS.mode == "auto_then_manual":
            run_auto_then_manual(session, out_dir)
        elif FLAGS.mode == "wall_snapshot":
            run_wall_snapshot(session, out_dir)
        elif FLAGS.mode == "wall_video":
            run_wall_video(session, out_dir)
        else:
            run_bestofn(session, out_dir)
    finally:
        try:
            env.close()
        except Exception:
            pass
        wandb.finish()


if __name__ == "__main__":
    app.run(main)
