"""Static meta-action coach: review the previous action chunk and pick a corrective meta-action.

Uses Google Gemini (``gemini-3.5-flash`` by default) with rollout video and trajectory plots
**up to the current env step**. The coach summarizes behavior during the just-finished action
chunk and selects one meta-action from a text file (one subtask description per line).

Default meta-actions file: ``coaches/metadata/example_subtasks.txt``.

Run from ``impls/``::

    python -m coaches.static_coach \\
        --video /path/to/rollout_prefix.mp4 \\
        --metadata /path/to/trajectory.json \\
        --current-episode-step 20 \\
        --meta-actions-file coaches/metadata/example_subtasks.txt \\
        --include-plots-in-prompt

For online CARLA training, use :class:`coaches.online_static_coach.OnlineStaticCoachSession`.
"""

from __future__ import annotations

import json
import os
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from coaches.action_chunk_feedback import (
    DEFAULT_ACTION_CHUNK_STEPS,
    DEFAULT_CHUNK_DURATION_SEC,
    episode_step_to_chunk_index,
)
from coaches.trajectory_plots import (
    TRAJECTORY_JSON_DESCRIPTION,
    compact_metadata_for_prompt,
    generate_trajectory_plots,
    is_trajectory_metadata,
    load_trajectory_metadata,
)
from coaches.vlm_feedback import _extract_json_payload

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
DEFAULT_META_ACTIONS_FILE = Path(__file__).resolve().parent / "metadata" / "example_subtasks.txt"


@dataclass(frozen=True)
class MetaActionRecommendation:
    """Coach output for one completed action chunk."""

    meta_action: str
    previous_chunk_index: int
    behavior_summary: str
    rationale: str

    @classmethod
    def from_dict(
        cls,
        raw: dict[str, Any],
        *,
        meta_actions: tuple[str, ...],
        previous_chunk_index: int,
    ) -> MetaActionRecommendation:
        action = str(raw.get("meta_action", "")).strip()
        if action not in meta_actions:
            raise ValueError(
                f"meta_action must be one of {meta_actions}, got {raw.get('meta_action')!r}."
            )
        return cls(
            meta_action=action,
            previous_chunk_index=int(previous_chunk_index),
            behavior_summary=str(raw.get("behavior_summary", "")).strip(),
            rationale=str(raw.get("rationale", "")).strip(),
        )


def load_meta_actions_from_file(path: str | Path) -> tuple[str, ...]:
    """Load one meta-action per non-empty line from a text file."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Meta-actions file not found: {file_path}")
    actions = tuple(
        line.strip()
        for line in file_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not actions:
        raise ValueError(f"Meta-actions file is empty: {file_path}")
    if len(set(actions)) != len(actions):
        raise ValueError(f"Meta-actions file must contain unique lines: {file_path}")
    return actions


def normalize_meta_actions(meta_actions: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    out = tuple(str(a).strip() for a in meta_actions if str(a).strip())
    if not out:
        raise ValueError("meta_actions must contain at least one non-empty label.")
    if len(set(out)) != len(out):
        raise ValueError(f"meta_actions must be unique, got {meta_actions!r}.")
    return out


def resolve_meta_actions(
    meta_actions: list[str] | tuple[str, ...] | None = None,
    meta_actions_file: str | Path | None = None,
) -> tuple[str, ...]:
    """Resolve meta-actions from an inline list or a text file (one label per line)."""
    if meta_actions:
        return normalize_meta_actions(meta_actions)
    return load_meta_actions_from_file(meta_actions_file or DEFAULT_META_ACTIONS_FILE)


def previous_chunk_index_for_episode_step(
    episode_step: int,
    *,
    action_chunk_steps: int = DEFAULT_ACTION_CHUNK_STEPS,
) -> int | None:
    """Chunk index that just finished when ``episode_step`` completes a chunk boundary."""
    step = int(episode_step)
    if step < int(action_chunk_steps):
        return None
    if step % int(action_chunk_steps) != 0:
        return None
    return episode_step_to_chunk_index(step, steps_per_chunk=action_chunk_steps) - 1


def slice_metadata_to_episode_step(
    metadata: dict[str, Any],
    episode_step: int,
) -> dict[str, Any]:
    """Truncate trajectory metadata to env steps ``<= episode_step``."""
    steps = metadata.get("steps", [])
    if not isinstance(steps, list):
        return dict(metadata)
    kept = [s for s in steps if int(s.get("episode_step", 0)) <= int(episode_step)]
    out = {k: v for k, v in metadata.items() if k != "steps"}
    out["steps"] = kept
    out["episode_steps"] = len(kept)
    out["truncated_to_episode_step"] = int(episode_step)
    return out


def slice_steps_for_chunk(
    metadata: dict[str, Any],
    chunk_index: int,
    *,
    action_chunk_steps: int = DEFAULT_ACTION_CHUNK_STEPS,
) -> list[dict[str, Any]]:
    """Return trajectory step records belonging to ``chunk_index``."""
    ep_start = chunk_index * action_chunk_steps + 1
    ep_end = (chunk_index + 1) * action_chunk_steps
    steps = metadata.get("steps", [])
    return [
        s
        for s in steps
        if ep_start <= int(s.get("episode_step", 0)) <= ep_end
    ]


def build_meta_action_prompt(
    metadata: dict[str, Any],
    *,
    previous_chunk_index: int,
    meta_actions: tuple[str, ...],
    action_chunk_steps: int = DEFAULT_ACTION_CHUNK_STEPS,
    chunk_duration_sec: float = DEFAULT_CHUNK_DURATION_SEC,
    include_plots: bool = False,
) -> str:
    """Prompt Gemini to review the previous chunk and pick one meta-action."""
    chunk_steps = slice_steps_for_chunk(
        metadata,
        previous_chunk_index,
        action_chunk_steps=action_chunk_steps,
    )
    ep_start = previous_chunk_index * action_chunk_steps + 1
    ep_end = min((previous_chunk_index + 1) * action_chunk_steps, int(metadata.get("episode_steps", 0)))
    t0 = previous_chunk_index * chunk_duration_sec
    t1 = (previous_chunk_index + 1) * chunk_duration_sec

    chunk_context = {
        "previous_chunk_index": previous_chunk_index,
        "episode_step_start": ep_start,
        "episode_step_end": ep_end,
        "video_time_start_sec": t0,
        "video_time_end_sec": t1,
        "steps_in_chunk": chunk_steps,
    }

    if is_trajectory_metadata(metadata):
        prompt_metadata = compact_metadata_for_prompt(metadata)
        metadata_block = json.dumps(prompt_metadata, indent=2)
        metadata_intro = TRAJECTORY_JSON_DESCRIPTION
        if include_plots:
            metadata_intro += (
                "\n\nA combined trajectory plot image is attached (speed, controls, collisions, "
                "route progress vs video time in seconds). The video clip ends at the current "
                "rollout time; focus on the previous action chunk window."
            )
        else:
            metadata_intro += (
                "\n\nUse the JSON below as structured context. The attached video covers the "
                "rollout from the start through the current step only."
            )
    else:
        metadata_block = json.dumps(metadata, indent=2)
        metadata_intro = "Optional metadata (may be empty):"

    actions_block = json.dumps(list(meta_actions), indent=2)
    chunk_block = json.dumps(chunk_context, indent=2)

    return textwrap.dedent(
        f"""
        You are a driving coach for an autonomous vehicle policy that executes fixed-length
        action chunks ({action_chunk_steps} env steps ≈ {chunk_duration_sec}s each).

        Review the **previous** action chunk (index {previous_chunk_index}, episode steps
        {ep_start}–{ep_end}, video ~{t0:.1f}s–{t1:.1f}s). The attached video and metadata
        cover the rollout **up to the current step** — do not assume future behavior.

        {metadata_intro}
        ```json
        {metadata_block}
        ```

        Previous chunk focus (steps and timing for the chunk you are judging):
        ```json
        {chunk_block}
        ```

        Choose exactly ONE corrective meta-action for the **next** chunk from this list:
        ```json
        {actions_block}
        ```

        Return ONLY valid JSON (no markdown fences):
        {{
          "meta_action": "<one label from the list>",
          "behavior_summary": "1–2 sentences describing what the vehicle did during the previous chunk.",
          "rationale": "1–2 sentences explaining why the chosen meta-action corrects that behavior."
        }}

        Rules:
        - ``meta_action`` must match one list entry exactly (case-sensitive, including punctuation).
        - Base your judgment on lane keeping, speed, steering, collisions/near-misses, stops, and turns
          visible in the video during the previous chunk window.
        - Cross-check with trajectory JSON: speed, throttle/steer/brake, collisions, route progress.
        - If behavior was acceptable, pick the subtask that best describes continuing that behavior.
        - When correction is needed, pick the subtask that best describes the desired next behavior.
        """
    ).strip()


def parse_meta_action_response(
    text: str,
    *,
    meta_actions: tuple[str, ...],
    previous_chunk_index: int,
) -> MetaActionRecommendation:
    payload = _extract_json_payload(text)
    return MetaActionRecommendation.from_dict(
        payload,
        meta_actions=meta_actions,
        previous_chunk_index=previous_chunk_index,
    )


def meta_action_to_language_label(
    meta_action: str | None,
    meta_actions: tuple[str, ...],
) -> tuple[str, np.ndarray]:
    """One-hot encode the selected meta-action for DSRL critic conditioning."""
    dim = len(meta_actions)
    label = np.zeros(dim, dtype=np.float32)
    action = str(meta_action or "").strip()
    if action in meta_actions:
        label[meta_actions.index(action)] = 1.0
    return action, label


def language_label_for_episode_step(
    chunk_recommendations: dict[int, MetaActionRecommendation] | None,
    episode_step: int,
    *,
    meta_actions: tuple[str, ...],
    action_chunk_steps: int = DEFAULT_ACTION_CHUNK_STEPS,
) -> tuple[str, np.ndarray]:
    """Look up the coach label for an env step from per-chunk recommendations."""
    if not chunk_recommendations:
        return meta_action_to_language_label(None, meta_actions)
    chunk_index = max(0, (int(episode_step) - 1) // int(action_chunk_steps))
    rec = chunk_recommendations.get(chunk_index)
    if rec is None:
        return meta_action_to_language_label(None, meta_actions)
    return meta_action_to_language_label(rec.meta_action, meta_actions)


class StaticMetaActionCoach:
    """Gemini coach that selects a meta-action after reviewing the previous chunk."""

    def __init__(
        self,
        meta_actions: list[str] | tuple[str, ...] | None = None,
        *,
        meta_actions_file: str | Path | None = None,
        api_key: str = GEMINI_API_KEY,
        model: str = DEFAULT_GEMINI_MODEL,
        action_chunk_steps: int = DEFAULT_ACTION_CHUNK_STEPS,
        chunk_duration_sec: float = DEFAULT_CHUNK_DURATION_SEC,
    ) -> None:
        self.meta_actions_file = (
            Path(meta_actions_file) if meta_actions_file else DEFAULT_META_ACTIONS_FILE
        )
        self.meta_actions = resolve_meta_actions(meta_actions, meta_actions_file)
        self.api_key = api_key
        self.model = model
        self.action_chunk_steps = int(action_chunk_steps)
        self.chunk_duration_sec = float(chunk_duration_sec)

    def recommend_for_previous_chunk(
        self,
        video_path: str | Path,
        metadata: dict[str, Any],
        *,
        current_episode_step: int,
        plot_paths: list[Path] | None = None,
        include_plots_in_prompt: bool = True,
    ) -> MetaActionRecommendation:
        """Review behavior in the chunk before ``current_episode_step`` and pick a meta-action."""
        prev_chunk = previous_chunk_index_for_episode_step(
            current_episode_step,
            action_chunk_steps=self.action_chunk_steps,
        )
        if prev_chunk is None:
            raise ValueError(
                f"No previous action chunk at episode_step={current_episode_step} "
                f"(need step >= {self.action_chunk_steps} on a chunk boundary)."
            )

        truncated = slice_metadata_to_episode_step(metadata, current_episode_step)
        prompt = build_meta_action_prompt(
            truncated,
            previous_chunk_index=prev_chunk,
            meta_actions=self.meta_actions,
            action_chunk_steps=self.action_chunk_steps,
            chunk_duration_sec=self.chunk_duration_sec,
            include_plots=include_plots_in_prompt,
        )
        text = self._generate_with_video(
            video_path,
            prompt,
            plot_paths=plot_paths if include_plots_in_prompt else None,
        )
        return parse_meta_action_response(
            text,
            meta_actions=self.meta_actions,
            previous_chunk_index=prev_chunk,
        )

    def _generate_with_video(
        self,
        video_path: str | Path,
        prompt: str,
        *,
        plot_paths: list[Path] | None = None,
    ) -> str:
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "Static meta-action coach requires google-genai. Install with: pip install google-genai"
            ) from exc

        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(f"Video not found: {path}")

        client = genai.Client(api_key=self.api_key)
        uploaded = client.files.upload(file=str(path))
        while uploaded.state.name == "PROCESSING":
            time.sleep(1.0)
            uploaded = client.files.get(name=uploaded.name)
        if uploaded.state.name != "ACTIVE":
            raise RuntimeError(f"Gemini file upload failed with state={uploaded.state.name!r}.")

        contents: list[Any] = [uploaded]
        for plot_path in plot_paths or []:
            plot_upload = client.files.upload(file=str(plot_path))
            while plot_upload.state.name == "PROCESSING":
                time.sleep(0.5)
                plot_upload = client.files.get(name=plot_upload.name)
            if plot_upload.state.name != "ACTIVE":
                raise RuntimeError(
                    f"Gemini plot upload failed for {plot_path} "
                    f"with state={plot_upload.state.name!r}."
                )
            contents.append(plot_upload)

        contents.append(prompt)
        response = client.models.generate_content(model=self.model, contents=contents)
        return getattr(response, "text", None) or str(response)


def load_metadata(metadata_path: str | Path) -> dict[str, Any]:
    return load_trajectory_metadata(metadata_path)


def main(
    video: Path,
    metadata: Path,
    current_episode_step: int,
    meta_actions_file: Path = DEFAULT_META_ACTIONS_FILE,
    meta_actions: list[str] | None = None,
    gemini_model: str = DEFAULT_GEMINI_MODEL,
    include_plots_in_prompt: bool = True,
    plots_dir: Path | None = None,
    action_chunk_steps: int = DEFAULT_ACTION_CHUNK_STEPS,
    chunk_duration_sec: float = DEFAULT_CHUNK_DURATION_SEC,
    output: Path | None = None,
) -> MetaActionRecommendation:
    """Run the static meta-action coach on a prefix rollout video + trajectory JSON."""
    actions = resolve_meta_actions(meta_actions, meta_actions_file)
    metadata_dict = load_metadata(metadata)
    truncated = slice_metadata_to_episode_step(metadata_dict, current_episode_step)

    plot_paths: list[Path] = []
    if include_plots_in_prompt and is_trajectory_metadata(truncated):
        resolved_plots_dir = plots_dir
        if resolved_plots_dir is None:
            stem = metadata.stem if metadata else video.stem
            resolved_plots_dir = (output.parent if output else video.parent) / f"{stem}_plots"
        plot_map = generate_trajectory_plots(truncated, resolved_plots_dir)
        plot_paths = [plot_map["combined"]]
        print(f"[static_coach] wrote trajectory plots to {resolved_plots_dir}", flush=True)

    coach = StaticMetaActionCoach(
        actions,
        meta_actions_file=meta_actions_file,
        model=gemini_model,
        action_chunk_steps=action_chunk_steps,
        chunk_duration_sec=chunk_duration_sec,
    )
    print(
        f"[static_coach] model={gemini_model} episode_step={current_episode_step} "
        f"meta_actions={len(actions)} from {meta_actions_file}",
        flush=True,
    )
    rec = coach.recommend_for_previous_chunk(
        video,
        metadata_dict,
        current_episode_step=current_episode_step,
        plot_paths=plot_paths,
        include_plots_in_prompt=include_plots_in_prompt,
    )
    print(
        f"[static_coach] chunk={rec.previous_chunk_index} meta_action={rec.meta_action!r}\n"
        f"  behavior: {rec.behavior_summary}\n"
        f"  rationale: {rec.rationale}",
        flush=True,
    )

    if output is None:
        output = metadata.with_name(
            f"{metadata.stem}_meta_action_step{current_episode_step:04d}.json"
        )
    payload = {
        "meta_action": rec.meta_action,
        "previous_chunk_index": rec.previous_chunk_index,
        "behavior_summary": rec.behavior_summary,
        "rationale": rec.rationale,
        "current_episode_step": int(current_episode_step),
        "meta_actions_file": str(meta_actions_file),
        "meta_actions": list(actions),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[static_coach] wrote {output}", flush=True)
    return rec


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
