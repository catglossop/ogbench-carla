"""Parse CARLA rollout trajectory JSON and render time-series plots for VLM coaching."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TRAJECTORY_JSON_DESCRIPTION = """
Trajectory metadata is logged during CARLA online rollouts (one record per env step, after the
policy applies controls). Times below align with the rollout video when ``in_video`` is true.

Episode-level fields:
  episode              — episode index within the training run
  route                — Bench2Drive route / scenario name
  episode_steps        — number of env steps in the episode
  video_fps            — frames per second of the logged episode video
  video_frame_stride   — env steps between consecutive video frames (e.g. 2 = every other step)
  success              — whether the route finished successfully
  termination_reason   — why the episode ended (if applicable)
  scenario_tree_status — leaderboard scenario tree status at episode end

Per-step fields (``steps`` list):
  step                 — global training step counter
  episode_step         — step index within this episode (starts at 1)
  ego_speed_mps        — ego speed in m/s
  control_throttle     — last applied CARLA throttle in [0, 1]
  control_steer        — last applied CARLA steer in [-1, 1]
  control_brake        — last applied CARLA brake in [0, 1]
  collision            — true if a new collision occurred on this step
  route_progress_pct   — route completion from RouteCompletionTest (0–100)
  in_video             — whether this step corresponds to a frame in the rollout video
  video_frame_index    — index into the episode video (null if not in video)
  video_timestamp_sec  — time in seconds from the start of the video (null if not in video)
""".strip()


def is_trajectory_metadata(data: dict[str, Any]) -> bool:
    """True when ``data`` matches the rollout trajectory JSON schema."""
    return isinstance(data.get("steps"), list)


def load_trajectory_metadata(metadata_path: str | Path) -> dict[str, Any]:
    """Load trajectory JSON; raise if the file is missing or not a trajectory object."""
    path = Path(metadata_path)
    if not path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Metadata file must contain a JSON object, got {type(data).__name__}.")
    if not is_trajectory_metadata(data):
        raise ValueError(
            "Metadata file must contain a trajectory object with a top-level 'steps' list."
        )
    return data


def step_time_sec(step: dict[str, Any], metadata: dict[str, Any]) -> float:
    """Map a step record to seconds on the rollout video timeline."""
    ts = step.get("video_timestamp_sec")
    if ts is not None:
        return float(ts)
    stride = int(metadata.get("video_frame_stride", 2))
    fps = float(metadata.get("video_fps", 10.0))
    ep_step = int(step["episode_step"])
    return max(0.0, (ep_step - 1) * stride / fps)


def compact_metadata_for_prompt(metadata: dict[str, Any]) -> dict[str, Any]:
    """Episode summary plus video-aligned steps to keep prompts within context limits."""
    steps = metadata.get("steps", [])
    video_steps = [s for s in steps if s.get("in_video")]
    if not video_steps:
        video_steps = steps
    header = {k: v for k, v in metadata.items() if k != "steps"}
    header["steps_in_prompt"] = len(video_steps)
    header["steps_total"] = len(steps)
    header["note"] = (
        "The steps array below lists video-aligned timesteps only; "
        "the full trajectory JSON contains every env step."
    )
    return {**header, "steps": video_steps}


def generate_trajectory_plots(
    metadata: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write a combined 2×2 figure (and individual panels) of trajectory signals vs video time."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Trajectory plots require matplotlib.") from exc

    if not is_trajectory_metadata(metadata):
        raise ValueError("Metadata does not contain trajectory 'steps'.")

    steps: list[dict[str, Any]] = metadata["steps"]
    if not steps:
        raise ValueError("Trajectory metadata contains no steps.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    times = [step_time_sec(s, metadata) for s in steps]
    speed = [float(s["ego_speed_mps"]) for s in steps]
    throttle = [float(s["control_throttle"]) for s in steps]
    steer = [float(s["control_steer"]) for s in steps]
    brake = [float(s["control_brake"]) for s in steps]
    progress = [float(s["route_progress_pct"]) for s in steps]
    collision_times = [t for t, s in zip(times, steps, strict=True) if s.get("collision")]

    episode = metadata.get("episode", "?")
    route = metadata.get("route", "?")
    title = f"Episode {episode} — {route}"

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    fig.suptitle(title, fontsize=12)

    ax_speed = axes[0, 0]
    ax_speed.plot(times, speed, color="#4C9AFF", linewidth=1.5)
    ax_speed.set_ylabel("Speed (m/s)")
    ax_speed.set_title("Ego speed")
    ax_speed.grid(True, alpha=0.3)

    ax_ctrl = axes[0, 1]
    ax_ctrl.plot(times, throttle, label="throttle", color="#2ECC71", linewidth=1.2)
    ax_ctrl.plot(times, brake, label="brake", color="#E74C3C", linewidth=1.2)
    ax_ctrl.plot(times, steer, label="steer", color="#F1C40F", linewidth=1.2)
    ax_ctrl.set_ylabel("Control")
    ax_ctrl.set_title("Throttle / brake / steer")
    ax_ctrl.set_ylim(-1.05, 1.05)
    ax_ctrl.legend(loc="upper right", fontsize=8)
    ax_ctrl.grid(True, alpha=0.3)

    ax_coll = axes[1, 0]
    ax_coll.set_ylim(0, 1)
    ax_coll.set_yticks([0, 1])
    ax_coll.set_yticklabels(["no", "collision"])
    ax_coll.set_title("Collisions")
    for ct in collision_times:
        ax_coll.axvline(ct, color="#E74C3C", linewidth=1.5, alpha=0.85)
    if collision_times:
        ax_coll.scatter(collision_times, [1] * len(collision_times), color="#E74C3C", s=36, zorder=3)
    ax_coll.grid(True, alpha=0.3)

    ax_prog = axes[1, 1]
    ax_prog.plot(times, progress, color="#9B59B6", linewidth=1.5)
    ax_prog.set_ylabel("Route progress (%)")
    ax_prog.set_title("Route completion")
    ax_prog.set_xlabel("Video time (s)")
    ax_prog.grid(True, alpha=0.3)

    axes[1, 0].set_xlabel("Video time (s)")

    combined = output_dir / "trajectory_plots.png"
    fig.tight_layout()
    fig.savefig(combined, dpi=120, bbox_inches="tight")
    plt.close(fig)

    paths: dict[str, Path] = {"combined": combined}

    def _save_single(name: str, plot_fn) -> None:
        single_fig, single_ax = plt.subplots(figsize=(8, 3))
        plot_fn(single_ax)
        single_ax.set_xlabel("Video time (s)")
        single_ax.grid(True, alpha=0.3)
        out = output_dir / f"trajectory_{name}.png"
        single_fig.tight_layout()
        single_fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(single_fig)
        paths[name] = out

    _save_single(
        "speed",
        lambda ax: (
            ax.plot(times, speed, color="#4C9AFF", linewidth=1.5),
            ax.set_ylabel("Speed (m/s)"),
            ax.set_title(f"{title} — ego speed"),
        ),
    )
    _save_single(
        "controls",
        lambda ax: (
            ax.plot(times, throttle, label="throttle", color="#2ECC71", linewidth=1.2),
            ax.plot(times, brake, label="brake", color="#E74C3C", linewidth=1.2),
            ax.plot(times, steer, label="steer", color="#F1C40F", linewidth=1.2),
            ax.set_ylabel("Control"),
            ax.set_title(f"{title} — controls"),
            ax.set_ylim(-1.05, 1.05),
            ax.legend(loc="upper right", fontsize=8),
        ),
    )
    _save_single(
        "collisions",
        lambda ax: (
            ax.set_ylim(0, 1),
            ax.set_yticks([0, 1]),
            ax.set_yticklabels(["no", "collision"]),
            ax.set_title(f"{title} — collisions"),
            [ax.axvline(ct, color="#E74C3C", linewidth=1.5, alpha=0.85) for ct in collision_times],
            ax.scatter(collision_times, [1] * len(collision_times), color="#E74C3C", s=36, zorder=3)
            if collision_times
            else None,
        ),
    )
    _save_single(
        "route_progress",
        lambda ax: (
            ax.plot(times, progress, color="#9B59B6", linewidth=1.5),
            ax.set_ylabel("Route progress (%)"),
            ax.set_title(f"{title} — route progress"),
        ),
    )

    return paths
