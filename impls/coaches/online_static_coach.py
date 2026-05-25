"""Online static meta-action coach session for CARLA DSRL rollouts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from coaches.action_chunk_feedback import DEFAULT_ACTION_CHUNK_STEPS, DEFAULT_CHUNK_DURATION_SEC
from coaches.online_vlm_coach import write_frames_to_mp4
from coaches.static_coach import (
    DEFAULT_META_ACTIONS_FILE,
    MetaActionRecommendation,
    StaticMetaActionCoach,
    language_label_for_episode_step,
    previous_chunk_index_for_episode_step,
    resolve_meta_actions,
    slice_metadata_to_episode_step,
)


def _as_config_dict(cfg: Any) -> dict[str, Any]:
    if cfg is None:
        return {}
    if hasattr(cfg, "to_dict"):
        return dict(cfg.to_dict())
    if isinstance(cfg, dict):
        return dict(cfg)
    return dict(cfg)


class OnlineStaticCoachSession:
    """Accumulate rollout video + trajectory; query meta-action coach at chunk boundaries."""

    def __init__(
        self,
        coach_cfg: Any,
        *,
        save_dir: str | Path,
        action_chunk_steps: int = DEFAULT_ACTION_CHUNK_STEPS,
    ) -> None:
        self.cfg = _as_config_dict(coach_cfg)
        self.save_dir = Path(save_dir)
        self.artifact_dir = self.save_dir / "static_coach"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        self.meta_actions_file = Path(
            self.cfg.get("meta_actions_file", DEFAULT_META_ACTIONS_FILE)
        )
        self.meta_actions = resolve_meta_actions(
            self.cfg.get("meta_actions"),
            self.meta_actions_file,
        )
        self.gemini_model = str(self.cfg.get("gemini_model", "gemini-3.5-flash"))
        self.include_plots_in_prompt = bool(self.cfg.get("include_plots_in_prompt", True))
        self.action_chunk_steps = int(self.cfg.get("action_chunk_steps", action_chunk_steps))
        self.action_chunk_duration_sec = float(
            self.cfg.get("action_chunk_duration_sec", DEFAULT_CHUNK_DURATION_SEC)
        )
        self.save_artifacts = bool(self.cfg.get("save_artifacts", True))
        self.video_fps = float(self.cfg.get("video_fps", 10.0))
        self.video_frame_stride = int(self.cfg.get("video_frame_stride", 2))

        self._coach = StaticMetaActionCoach(
            self.meta_actions,
            meta_actions_file=self.meta_actions_file,
            model=self.gemini_model,
            action_chunk_steps=self.action_chunk_steps,
            chunk_duration_sec=self.action_chunk_duration_sec,
        )

        self.reset_episode()

    def reset_episode(self) -> None:
        self.episode_count = 0
        self.route_name = "?"
        self.frames: list[np.ndarray] = []
        self.trajectory_steps: list[dict[str, Any]] = []
        self.episode_buffer_indices: list[int] = []
        self.episode_step_for_buffer: list[int] = []
        self.chunk_recommendations: dict[int, MetaActionRecommendation] = {}
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

    def _build_metadata(self, *, episode_step: int, done_info: dict[str, Any] | None = None) -> dict[str, Any]:
        done_info = done_info or {}
        metadata = {
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
        return slice_metadata_to_episode_step(metadata, episode_step)

    def should_query(self, episode_step: int) -> bool:
        """Query after each completed action chunk (when a previous chunk exists)."""
        if previous_chunk_index_for_episode_step(
            episode_step,
            action_chunk_steps=self.action_chunk_steps,
        ) is None:
            return False
        if len(self.frames) == 0:
            return False
        if episode_step == self._last_query_episode_step:
            return False
        return True

    def maybe_query(
        self,
        *,
        episode_step: int,
        done_info: dict[str, Any] | None = None,
        force: bool = False,
    ) -> bool:
        if not force and not self.should_query(episode_step):
            return False
        if force and len(self.frames) == 0:
            return False
        self._run_coach_query(episode_step=episode_step, done_info=done_info)
        self._last_query_episode_step = episode_step
        return True

    def _run_coach_query(
        self,
        *,
        episode_step: int,
        done_info: dict[str, Any] | None,
    ) -> None:
        prev_chunk = previous_chunk_index_for_episode_step(
            episode_step,
            action_chunk_steps=self.action_chunk_steps,
        )
        assert prev_chunk is not None

        tag = f"ep{self.episode_count:04d}_step{episode_step:04d}_chunk{prev_chunk:03d}"
        work_dir = self.artifact_dir / tag
        work_dir.mkdir(parents=True, exist_ok=True)

        video_path = work_dir / "rollout_prefix.mp4"
        write_frames_to_mp4(self.frames, video_path, fps=self.video_fps)
        metadata = self._build_metadata(episode_step=episode_step, done_info=done_info)
        metadata_path = work_dir / "trajectory_prefix.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        plot_paths = None
        if self.include_plots_in_prompt:
            from coaches.trajectory_plots import generate_trajectory_plots

            plot_map = generate_trajectory_plots(metadata, work_dir / "plots")
            plot_paths = [plot_map["combined"]]

        print(
            f"[online_static_coach] querying gemini on {len(self.frames)} frames "
            f"(episode_step={episode_step}, previous_chunk={prev_chunk}) -> {work_dir}",
            flush=True,
        )

        rec = self._coach.recommend_for_previous_chunk(
            video_path,
            metadata,
            current_episode_step=episode_step,
            plot_paths=plot_paths,
            include_plots_in_prompt=self.include_plots_in_prompt,
        )
        self.chunk_recommendations[prev_chunk] = rec

        out = work_dir / "meta_action.json"
        out.write_text(
            json.dumps(
                {
                    "meta_action": rec.meta_action,
                    "previous_chunk_index": rec.previous_chunk_index,
                    "behavior_summary": rec.behavior_summary,
                    "rationale": rec.rationale,
                    "current_episode_step": episode_step,
                    "meta_actions_file": str(self.meta_actions_file),
                    "meta_actions": list(self.meta_actions),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        if self.save_artifacts:
            print(
                f"[online_static_coach] chunk {prev_chunk} -> {rec.meta_action!r} "
                f"({rec.behavior_summary[:80]}...)",
                flush=True,
            )

    def language_label_for_episode_step(self, episode_step: int) -> tuple[str, np.ndarray]:
        return language_label_for_episode_step(
            self.chunk_recommendations,
            episode_step,
            meta_actions=self.meta_actions,
            action_chunk_steps=self.action_chunk_steps,
        )

    def backfill_buffer(self, buffer: Any) -> None:
        if not self.episode_buffer_indices:
            return
        if not self.chunk_recommendations:
            print("[online_static_coach] skip backfill: no meta-action recommendations yet", flush=True)
            return
        for buf_idx, ep_step in zip(self.episode_buffer_indices, self.episode_step_for_buffer, strict=True):
            _text, label = self.language_label_for_episode_step(ep_step)
            buffer.update_at(int(buf_idx), language_label=label)
        n_labeled = sum(
            1
            for ep_step in self.episode_step_for_buffer
            if self.language_label_for_episode_step(ep_step)[1].any()
        )
        print(
            f"[online_static_coach] backfilled {len(self.episode_buffer_indices)} transitions "
            f"({n_labeled} non-zero labels)",
            flush=True,
        )
