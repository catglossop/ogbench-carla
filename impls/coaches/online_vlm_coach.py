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
    raw_text_for_chunk_index,
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
        text_encoder: Any = None,
        use_raw_description: bool = False,
        plot_embeddings: bool = False,
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
        # Optional callable (str) -> np.ndarray for VLM-embedding the feedback text.
        # When None, the 17-dim delta-commentary BoW is used instead.
        self._text_encoder = text_encoder
        # When True, encode raw description+correction from the nearest BAD event
        # instead of the structured lateral/longitudinal/detail phrase.
        self._use_raw_description = use_raw_description and (text_encoder is not None)
        self._plot_embeddings = plot_embeddings

        self.reset_episode()

    def reset_episode(self) -> None:
        self.episode_count = 0
        self.route_name = "?"
        self.frames: list[np.ndarray] = []
        self.trajectory_steps: list[dict[str, Any]] = []
        self.episode_buffer_indices: list[int] = []
        self.episode_step_for_buffer: list[int] = []
        self.chunk_feedback_json: dict[str, Any] | None = None
        # Per-window feedback: list of (step_offset, chunk_feedback_json) where
        # step_offset is the number of trajectory steps already queried before this window.
        self._chunk_feedback_windows: list[tuple[int, dict[str, Any]]] = []
        self.latest_events: list[Any] = []
        self._last_query_episode_step = 0
        self._backfill_cursor = 0
        self._frames_cursor = 0
        self._traj_cursor = 0

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

    def _build_metadata(
        self,
        traj_steps: list[dict[str, Any]],
        *,
        done_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        done_info = done_info or {}

        # Compact collision event list so Gemini sees it without parsing all steps.
        # Each entry marks a timestamp where either a new collision fired or bounding-box
        # contact was actively ongoing.
        collision_events: list[dict[str, Any]] = []
        for s in traj_steps:
            if s.get("collision") or s.get("collision_active"):
                collision_events.append({
                    "video_timestamp_sec": s.get("video_timestamp_sec"),
                    "episode_step": s.get("episode_step"),
                    "new_event": bool(s.get("collision")),
                    "contact_active": bool(s.get("collision_active")),
                })

        return {
            "episode": self.episode_count,
            "route": self.route_name,
            "episode_steps": len(traj_steps),
            "video_fps": self.video_fps,
            "video_frame_stride": self.video_frame_stride,
            "success": done_info.get("success"),
            "termination_reason": done_info.get("termination_reason"),
            "scenario_tree_status": done_info.get("scenario_tree_status"),
            "collision_events": collision_events,
            "steps": traj_steps,
        }

    def should_query(self, episode_step: int, *, force: bool = False) -> bool:
        if force:
            # Only query if there are unqueried frames remaining.
            return self.query_on_episode_end and len(self.frames) > self._frames_cursor
        if self.query_every_n_episode_steps <= 0:
            return False
        if episode_step <= 0 or len(self.frames) <= self._frames_cursor:
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
        # Slice to only the frames and steps not yet sent to Gemini.
        frames_window = self.frames[self._frames_cursor:]
        traj_window = self.trajectory_steps[self._traj_cursor:]
        step_offset = self._traj_cursor  # absolute episode steps before this window

        tag = f"ep{self.episode_count:04d}_step{episode_step:04d}{'_final' if final else ''}"
        work_dir = self.artifact_dir / tag
        work_dir.mkdir(parents=True, exist_ok=True)

        video_path = work_dir / "rollout.mp4"
        write_frames_to_mp4(frames_window, video_path, fps=self.video_fps)
        metadata = self._build_metadata(traj_window, done_info=done_info)
        metadata_path = work_dir / "trajectory.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        print(
            f"[online_vlm_coach] querying {self.provider} on {len(frames_window)} frames "
            f"({len(traj_window)} steps, offset={step_offset}) -> {work_dir}",
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

        # Store window with its step offset for windowed lookup in backfill.
        self._chunk_feedback_windows.append((step_offset, chunk_json))
        self.chunk_feedback_json = chunk_json  # keep latest for logging/compat
        self.latest_events = events

        # Advance cursors so the next query starts from here.
        self._frames_cursor = len(self.frames)
        self._traj_cursor = len(self.trajectory_steps)

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
        # Search windows in order; convert to window-relative step for lookup.
        # chunk_feedback_json stores chunk indices relative to the window start,
        # so we subtract the window's step_offset before calling the lookup.
        for step_offset, cfj in self._chunk_feedback_windows:
            window_size = int(cfj.get("episode_steps", 0))
            window_step = episode_step - step_offset
            if 1 <= window_step <= window_size:
                if self._use_raw_description:
                    chunk_index = max(0, (window_step - 1) // self.action_chunk_steps)
                    text = raw_text_for_chunk_index(
                        cfj, chunk_index, bad_event_radius_chunks=self.bad_event_radius_chunks
                    )
                    return text, self._text_encoder(text)
                text, bow = language_label_for_episode_step(
                    cfj, window_step, action_chunk_steps=self.action_chunk_steps
                )
                if self._text_encoder is not None:
                    return text, self._text_encoder(text)
                return text, bow
        # No window matched: no feedback for this step.
        if self._text_encoder is not None:
            return "", self._text_encoder("")
        text, bow = language_label_for_episode_step(
            None, episode_step, action_chunk_steps=self.action_chunk_steps
        )
        return text, bow

    def backfill_buffer(self, buffer: Any, *, global_step: int | None = None) -> None:
        """Write coach labels for transitions recorded since the last backfill call.

        Uses a cursor so mid-episode calls only label new transitions, and a
        final episode-end call labels any remainder without re-writing earlier slots.
        """
        new_indices = self.episode_buffer_indices[self._backfill_cursor:]
        new_steps = self.episode_step_for_buffer[self._backfill_cursor:]
        if not new_indices:
            return
        if self.chunk_feedback_json is None:
            print("[online_vlm_coach] skip backfill: no chunk feedback yet", flush=True)
            return
        for buf_idx, ep_step in zip(new_indices, new_steps):
            _text, bow = self.language_label_for_episode_step(ep_step)
            buffer.update_at(int(buf_idx), coach_label=bow)
        self._backfill_cursor += len(new_indices)
        n_labeled = sum(
            1
            for ep_step in new_steps
            if self.language_label_for_episode_step(ep_step)[1].any()
        )
        n_total = len(new_indices)
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

        if self._plot_embeddings:
            self._log_embedding_plot(global_step=global_step)

    def _log_embedding_plot(self, *, global_step: int | None = None) -> None:
        """PCA scatter of all episode label vectors logged to wandb.

        Collects every transition's label embedding, projects to 2D via PCA
        (numpy SVD, no sklearn required), and uploads as a wandb image.
        Points near a BAD event are coloured by episode step; steps with no
        coaching signal are shown in muted blue.
        """
        try:
            import wandb  # type: ignore
            import matplotlib  # type: ignore
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # type: ignore
        except ImportError:
            return
        if wandb.run is None:
            return

        if not self.episode_buffer_indices:
            return

        # Re-derive all labels for the full episode (cheap — just array lookup).
        vecs = []
        steps = []
        for ep_step in self.episode_step_for_buffer:
            _text, label = self.language_label_for_episode_step(ep_step)
            vecs.append(label)
            steps.append(ep_step)

        vecs_arr = np.stack(vecs)       # (N, D)
        steps_arr = np.array(steps)     # (N,)
        is_nonzero = np.any(vecs_arr != 0, axis=1)

        if vecs_arr.shape[0] < 2 or vecs_arr.shape[1] < 2:
            return

        # PCA to 2D (numpy-only: centre → SVD → project).
        centred = vecs_arr - vecs_arr.mean(axis=0)
        try:
            _, _, Vt = np.linalg.svd(centred, full_matrices=False)
            proj = centred @ Vt[:2].T   # (N, 2)
        except np.linalg.LinAlgError:
            return

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(
            proj[~is_nonzero, 0], proj[~is_nonzero, 1],
            c="steelblue", s=15, alpha=0.35, label="no event",
        )
        if is_nonzero.any():
            sc = ax.scatter(
                proj[is_nonzero, 0], proj[is_nonzero, 1],
                c=steps_arr[is_nonzero], cmap="Reds", s=40, zorder=2,
                label="near BAD event",
            )
            plt.colorbar(sc, ax=ax, label="episode step")
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        mode = "raw" if self._use_raw_description else "structured"
        ax.set_title(
            f"Coach label embeddings — ep {self.episode_count} ({mode}, "
            f"D={vecs_arr.shape[1]}, N={vecs_arr.shape[0]})"
        )
        ax.legend(fontsize=8)
        plt.tight_layout()

        wandb.log({"coach/embedding_pca": wandb.Image(fig)}, step=global_step)
        plt.close(fig)
