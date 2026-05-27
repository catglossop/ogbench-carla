"""Online VLM coach session for DSRL language feedback during CARLA rollouts."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from coaches.action_chunk_feedback import (
    DEFAULT_ACTION_CHUNK_STEPS,
    DEFAULT_BAD_EVENT_RADIUS_CHUNKS,
    DEFAULT_CHUNK_DURATION_SEC,
    language_label_for_episode_step,
)
from coaches.vlm_feedback import (
    annotate_video,
    create_coach,
    generate_action_chunk_feedback,
)


def write_frames_to_mp4(frames: list[np.ndarray], output_path: Path, *, fps: float = 10.0) -> Path:
    """Write RGB uint8 frames to a temporary MP4 for VLM upload."""
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ImportError("Online VLM coach requires opencv-python.") from exc

    if not frames:
        raise ValueError("Cannot write video from an empty frame list.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames_u8 = [np.asarray(f, dtype=np.uint8) for f in frames]
    h, w = frames_u8[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")
    try:
        for frame in frames_u8:
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if frame.shape[-1] == 3 else frame
            writer.write(bgr)
    finally:
        writer.release()
    return output_path


def _as_config_dict(cfg: Any) -> dict[str, Any]:
    if cfg is None:
        return {}
    if hasattr(cfg, "to_dict"):
        return dict(cfg.to_dict())
    if isinstance(cfg, dict):
        return dict(cfg)
    return dict(cfg)


class OnlineVLMSession:
    """Accumulate rollout video + trajectory; query VLM coach periodically."""

    def __init__(
        self,
        vlm_cfg: Any,
        *,
        save_dir: str | Path,
        action_chunk_steps: int = DEFAULT_ACTION_CHUNK_STEPS,
    ) -> None:
        self.cfg = _as_config_dict(vlm_cfg)
        self.save_dir = Path(save_dir)
        self.artifact_dir = self.save_dir / "coach"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        self.query_every_n_episode_steps = int(self.cfg.get("query_every_n_episode_steps", 128))
        self.query_on_episode_end = bool(self.cfg.get("query_on_episode_end", True))
        self.provider = str(self.cfg.get("provider", "gemini"))
        self.gemini_model = str(self.cfg.get("gemini_model", "gemini-3.5-flash"))
        self.include_plots_in_prompt = bool(self.cfg.get("include_plots_in_prompt", False))
        self.action_chunk_steps = int(self.cfg.get("action_chunk_steps", action_chunk_steps))
        self.action_chunk_duration_sec = float(
            self.cfg.get("action_chunk_duration_sec", DEFAULT_CHUNK_DURATION_SEC)
        )
        self.bad_event_radius_chunks = int(
            self.cfg.get("bad_event_radius_chunks", DEFAULT_BAD_EVENT_RADIUS_CHUNKS)
        )
        self.annotate_video = bool(self.cfg.get("annotate_video", False))
        self.save_artifacts = bool(self.cfg.get("save_artifacts", True))
        self.video_fps = float(self.cfg.get("video_fps", 10.0))
        self.video_frame_stride = int(self.cfg.get("video_frame_stride", 2))

        self._coach = create_coach(
            self.provider,  # type: ignore[arg-type]
            model=self.gemini_model,
        )

        self.reset_episode()

    def reset_episode(self) -> None:
        self.episode_count = 0
        self.route_name = "?"
        self.frames: list[np.ndarray] = []
        self.trajectory_steps: list[dict[str, Any]] = []
        self.episode_buffer_indices: list[int] = []
        self.episode_step_for_buffer: list[int] = []
        self.chunk_feedback_json: dict[str, Any] | None = None
        self.latest_events: list[Any] = []
        self._last_query_episode_step = 0

    def begin_episode(self, *, episode_count: int, route_name: str) -> None:
        self.episode_count = int(episode_count)
        self.route_name = str(route_name)

    def record_frame(self, frame: np.ndarray | None) -> None:
        if frame is not None:
            self.frames.append(np.asarray(frame, dtype=np.uint8))

    def record_trajectory_step(self, step_record: dict[str, Any]) -> None:
        self.trajectory_steps.append(dict(step_record))

    def track_buffer_transition(self, *, buffer_index: int, episode_step: int) -> None:
        self.episode_buffer_indices.append(int(buffer_index))
        self.episode_step_for_buffer.append(int(episode_step))

    def _build_metadata(self, *, done_info: dict[str, Any] | None = None) -> dict[str, Any]:
        done_info = done_info or {}
        return {
            "episode": self.episode_count,
            "route": self.route_name,
            "episode_steps": len(self.trajectory_steps),
            "video_fps": self.video_fps,
            "video_frame_stride": self.video_frame_stride,
            "success": done_info.get("success"),
            "termination_reason": done_info.get("termination_reason"),
            "scenario_tree_status": done_info.get("scenario_tree_status"),
            "steps": self.trajectory_steps,
        }

    def should_query(self, episode_step: int, *, force: bool = False) -> bool:
        if force:
            return self.query_on_episode_end and len(self.frames) > 0
        if self.query_every_n_episode_steps <= 0:
            return False
        if episode_step <= 0 or len(self.frames) == 0:
            return False
        if episode_step % self.query_every_n_episode_steps != 0:
            return False
        return episode_step != self._last_query_episode_step

    def maybe_query(
        self,
        *,
        episode_step: int,
        done_info: dict[str, Any] | None = None,
        force: bool = False,
        global_step: int | None = None,
    ) -> bool:
        if not self.should_query(episode_step, force=force):
            return False
        self._run_coach_query(episode_step=episode_step, done_info=done_info, final=force, global_step=global_step)
        self._last_query_episode_step = episode_step
        return True

    def _run_coach_query(
        self,
        *,
        episode_step: int,
        done_info: dict[str, Any] | None,
        final: bool,
        global_step: int | None = None,
    ) -> None:
        tag = f"ep{self.episode_count:04d}_step{episode_step:04d}{'_final' if final else ''}"
        work_dir = self.artifact_dir / tag
        work_dir.mkdir(parents=True, exist_ok=True)

        video_path = work_dir / "rollout.mp4"
        write_frames_to_mp4(self.frames, video_path, fps=self.video_fps)
        metadata = self._build_metadata(done_info=done_info)
        metadata_path = work_dir / "trajectory.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        print(
            f"[online_vlm_coach] querying {self.provider} on {len(self.frames)} frames "
            f"({len(self.trajectory_steps)} steps) -> {work_dir}",
            flush=True,
        )

        plot_paths = None
        if self.include_plots_in_prompt:
            from coaches.trajectory_plots import generate_trajectory_plots

            plot_map = generate_trajectory_plots(metadata, work_dir / "plots")
            plot_paths = [plot_map["combined"]]

        events = self._coach.analyze(
            video_path,
            metadata,
            plot_paths=plot_paths,
            include_plots_in_prompt=self.include_plots_in_prompt,
        )
        chunk_json = generate_action_chunk_feedback(
            self._coach,
            metadata,
            events,
            steps_per_chunk=self.action_chunk_steps,
            chunk_duration_sec=self.action_chunk_duration_sec,
            bad_event_radius_chunks=self.bad_event_radius_chunks,
        )
        self.chunk_feedback_json = chunk_json
        self.latest_events = events

        chunk_out = work_dir / "chunk_feedback.json"
        chunk_out.write_text(json.dumps(chunk_json, indent=2), encoding="utf-8")

        # Always annotate so wandb gets the labelled video (GOOD/BAD overlays).
        # self.annotate_video controls whether to *also* print the artifact path.
        annotated_path = work_dir / "annotated.mp4"
        annotate_video(
            video_path,
            events,
            annotated_path,
        )

        self._log_to_wandb(
            events=events,
            video_path=annotated_path,
            work_dir=work_dir,
            global_step=global_step,
        )

        if self.save_artifacts:
            print(f"[online_vlm_coach] saved artifacts under {work_dir}", flush=True)

    def _log_to_wandb(
        self,
        *,
        events: list[Any],
        video_path: Path,
        work_dir: Path,
        global_step: int | None,
    ) -> None:
        """Upload coach video + event table to wandb (no-op if wandb is not active)."""
        try:
            import wandb  # type: ignore
        except ImportError:
            return
        if wandb.run is None:
            return

        log: dict[str, Any] = {}

        # ── Video ──────────────────────────────────────────────────────────────
        if video_path.is_file():
            log["coach/video"] = wandb.Video(str(video_path), fps=int(self.video_fps), format="mp4")

        # ── Events table ───────────────────────────────────────────────────────
        if events:
            tbl = wandb.Table(columns=["timestamp_sec", "label", "description"])
            for ev in events:
                tbl.add_data(
                    getattr(ev, "timestamp_sec", None),
                    getattr(ev, "label", None),
                    getattr(ev, "description", None),
                )
            log["coach/events"] = tbl

        # ── Scalar stats ───────────────────────────────────────────────────────
        n_good = sum(1 for ev in events if getattr(ev, "label", None) == "GOOD")
        n_bad = sum(1 for ev in events if getattr(ev, "label", None) == "BAD")
        log["coach/n_events"] = len(events)
        log["coach/n_good"] = n_good
        log["coach/n_bad"] = n_bad
        log["coach/episode"] = self.episode_count

        wandb.log(log, step=global_step)

    def language_label_for_episode_step(self, episode_step: int) -> tuple[str, np.ndarray]:
        return language_label_for_episode_step(
            self.chunk_feedback_json,
            episode_step,
            action_chunk_steps=self.action_chunk_steps,
        )

    def backfill_buffer(self, buffer: Any, *, global_step: int | None = None) -> None:
        """Write coach labels for all transitions recorded this episode."""
        if not self.episode_buffer_indices:
            return
        if self.chunk_feedback_json is None:
            print("[online_vlm_coach] skip backfill: no chunk feedback yet", flush=True)
            return
        for buf_idx, ep_step in zip(self.episode_buffer_indices, self.episode_step_for_buffer):  # strict= requires Python 3.10+
            _text, bow = self.language_label_for_episode_step(ep_step)
            buffer.update_at(int(buf_idx), coach_label=bow)
        n_labeled = sum(
            1
            for ep_step in self.episode_step_for_buffer
            if self.language_label_for_episode_step(ep_step)[1].any()
        )
        n_total = len(self.episode_buffer_indices)
        print(
            f"[online_vlm_coach] backfilled {n_total} transitions ({n_labeled} non-zero labels)",
            flush=True,
        )
        try:
            import wandb  # type: ignore

            if wandb.run is not None:
                wandb.log(
                    {
                        "coach/backfill_transitions": n_total,
                        "coach/backfill_labeled": n_labeled,
                        "coach/backfill_frac_labeled": n_labeled / max(n_total, 1),
                    },
                    step=global_step,
                )
        except ImportError:
            pass
