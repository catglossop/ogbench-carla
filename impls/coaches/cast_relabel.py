"""CAST relabel: window rollout -> VLM review -> per-chunk credit -> suggested subtasks.

This mirrors :mod:`coaches.action_chunk_feedback` but changes the *output* from
corrective steering commentary to **suggested subtasks** for each action chunk, with
an explicit credit-assignment step in between.

Pipeline (see ``OnlineCastRelabelSession`` for the online wiring):

1. Roll the agent out for a window of env steps (rounded to whole action chunks; a
   window can be anything from one action chunk up to a whole episode).
2. A VLM reviews the *entire window* video and returns what the agent did **well**
   (``GOOD``) and **poorly** (``BAD``) as timestamped events (``coach.analyze``).
3. **Credit assignment** — a second VLM call maps those GOOD/BAD moments onto the
   specific action chunks in the window. Credit is *causal*, not merely temporal: a chunk is
   BAD either because a BAD event overlaps it (``credit_source="direct"``) or because it is
   part of the lead-up that made a later BAD event hard to avoid (``credit_source="precursor"``),
   so the corrected subtask can pre-empt the failure rather than only react to it.
4. For each chunk, the VLM suggests several subtasks that would improve the behavior,
   seeded (open-vocab) by :data:`SEED_SUBTASKS`, **and** (for chunks whose subtask it
   changes) a fresh chain-of-thought reasoning trace that justifies the corrected subtask.

Consumption:

- **Artifacts + wandb** — each window is written to a ``cast_relabel.json`` and, when
  ``debug`` is enabled, an annotated video (original subtask + waypoints/actions already
  drawn upstream, plus per-chunk GOOD/BAD labels and suggested subtasks) is logged to W&B.
- **High-level (VLM backbone) dataset** — every BAD/relabeled chunk is written out as a SteerVLA
  *high-level* training sample (image + ego state + prompt + corrected subtask + new reasoning
  trace + the executed action chunk with ``action_loss_mask`` all-``False``). When
  ``store_good_chunks`` is set (default), GOOD and unlabeled chunks are also stored, but with the
  **original** subtask/reasoning the model produced (reinforcing good behavior rather than
  correcting it) instead of a VLM-suggested target.
  This mirrors OpenPI's ``steervla_hl_datasets`` / ``steervla_hl_dataset_format`` path (see
  ``openpi.training.steervla_rlds_dataset``), where ``action_supervision=False`` zeroes the
  action-flow loss so only the CoT/VLM backbone is supervised. These samples are consumed
  online: ``SteerVLAActor.update_hl`` (``impls/vlas/steervla.py``) loads them, tokenizes the
  subtask/reasoning as CoT targets with ``action_loss_mask`` all-``False``, and runs an OpenPI
  gradient step on the trainable SteerVLA train state (``steervla.load_trainable_params``). It is
  driven from ``DSRLAgent.update_with_vla(..., run_hl=True)`` (gated by ``enable_updates_bc_hl``).
"""

from __future__ import annotations

import json
import os
import re
import textwrap
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from coaches.action_chunk_feedback import (
    ActionChunkSpec,
    DEFAULT_ACTION_CHUNK_STEPS,
    DEFAULT_CHUNK_DURATION_SEC,
    build_action_chunk_specs,
)
from coaches.correction_memory import DEFAULT_MAX_WORDS as DEFAULT_MEMORY_WORDS
from coaches.correction_memory import CorrectionMemory
from coaches.online_vlm_coach import write_frames_to_mp4

# Saved/logged video only -- never the frames a model consumes. See main_carla for the rationale.
_SAVED_VIDEO_MAX_WIDTH = 512


def _shrink_frames_for_saving(frames, max_width: int = _SAVED_VIDEO_MAX_WIDTH):
    """Aspect-preserving downscale for W&B/debug video; no-op if already narrower."""
    if not frames:
        return frames
    import numpy as _np

    h, w = _np.asarray(frames[0]).shape[:2]
    if w <= max_width:
        return frames
    new_wh = (max_width, max(1, int(round(h * (max_width / float(w))))))
    try:
        import cv2

        return [cv2.resize(_np.asarray(f), new_wh, interpolation=cv2.INTER_AREA) for f in frames]
    except Exception:  # noqa: BLE001 - never cost the video.
        return frames
import concurrent.futures
import threading

from coaches.vlm_feedback import CoachEvent, build_coaching_prompt, create_coach

# Default number of subtask suggestions produced per chunk that needs improvement.
DEFAULT_NUM_SUBTASK_SUGGESTIONS = 3

# Default env action dim for the stored high-level (VLM-backbone) samples. The action head
# is masked out for HL samples (``action_loss_mask`` all-False), so this only fixes the shape
# of the stored (unsupervised) action chunk; it should match ``steervla.action_dim``.
DEFAULT_HL_ACTION_DIM = 4

# Number of ``[speed_mps, course_deg]`` pairs stored per HL sample as ``ego_hist`` (oldest first,
# one pair per env step, last pair = the step the sample's action was taken from). Matches the
# SimLingo RLDS ``observation/ego_hist`` history length: the OpenPI loader uses all 4 pairs
# (8 proprio dims) when ``include_ego_history=True`` and only the last pair when it is False.
# Only the last pair is used by the online HL update, which rebuilds proprio from ``state``;
# the history exists so :mod:`vlas.cast_hl_to_rlds` can emit either flavor of RLDS dataset.
DEFAULT_EGO_HISTORY_LEN = 4

# Indices into the raw CARLA ego-state vector (``ogbench.carla.carla_utils`` STATE_DIM layout),
# the same two ``steervla.carla_state_vec_to_steervla_state`` reads. Duplicated here rather than
# imported so this module stays importable without CARLA on ``sys.path``. Index 11 is ``avel.z``,
# the yaw rate in deg/s; scaling it by :data:`SIMLINGO_FRAME_DT` gives the per-frame heading delta
# that SimLingo's ``ego_hist[..., 1]`` holds. Absolute ``rot.yaw`` (index 5) is NOT that quantity --
# see ``vlas.steervla.carla_yaw_rate_to_simlingo_course``. Keep all three copies in sync.
EGO_STATE_IDX_YAW_RATE = 11
EGO_STATE_IDX_SPEED = 15
SIMLINGO_FRAME_DT = 0.25

# PaliGemma location sentinels (``<loc0000>``..``<loc1023>``) that the CoT decode emits verbatim, so
# every subtask/reasoning string captured at rollout arrives as
# ``'<loc1022>The vehicle remained stopped.;<loc1021>'``. ``vlas.steervla`` strips these before they
# become HL training targets; the same has to happen before they go into a VLM prompt, or the coach
# is asked to reason about — and imitate the phrasing of — tokens that are not language. Duplicated
# from ``vlas.steervla.strip_cot_sentinels`` rather than imported so this module stays importable
# without JAX / OpenPI on the path (same reasoning as the ego-state indices above).
_LOC_SENTINEL_RE = re.compile(r"<loc\d+>")


def strip_cot_sentinels(text: Any) -> str:
    """``'<loc1022>The vehicle accelerates.;<loc1021>'`` -> ``'The vehicle accelerates.'``"""
    s = _LOC_SENTINEL_RE.sub(" ", str(text or ""))
    s = re.sub(r"\s+", " ", s).strip()
    return s.rstrip(" ;").strip()


# ``hl_samples.json`` schema version. 1 = the original online-only manifest; 2 adds the fields the
# offline RLDS converter needs (``ego_hist`` in the npz, ``routing_command`` / ``original_subtask`` /
# ``original_reasoning`` / ``route`` / ``global_step`` / ``current_speed`` in the manifest).
HL_SCHEMA_VERSION = 2

# Seed phrases for the open-vocabulary subtask generation in step 4. The VLM is told it
# MAY reuse these verbatim or propose new phrases in the same style. Provided by the user;
# edit here (or override via ``cast_relabel.seed_subtasks`` in the agent config) to retune.
SEED_SUBTASKS: tuple[str, ...] = (
    "The vehicle cautiously follows the route, maintaining steady lane keeping behind the front car.",
    "The vehicle smoothly adjusts course to the right while cautiously maintaining speed behind the police car.",
    "The vehicle maintains a steady course, following the black car ahead at a normal pace.",
    "The vehicle remains stopped normally at the red traffic light.",
    "The vehicle accelerates behind the black car at 8.8 meters, following it through the green traffic light.",
    "The vehicle accelerated normally through the green traffic light.",
    "The vehicle accelerates normally through the green traffic light in 4.8 meters.",
    "The vehicle smoothly decelerates to a stop, cautiously adjusting course to the left due to a pedestrian.",
    "The vehicle maintains a steady speed while following the route, staying behind the dark green SUV.",
    "The vehicle smoothly follows the route, cautiously reducing speed through the junction with steady lane keeping.",
    "The vehicle normally follows the route, maintaining reduced speed behind the front black car.",
    "The vehicle accelerates steadily to follow the front black car, maintaining a straight course.",
    "The vehicle turns right aggressively while accelerating through the clear junction.",
    "The vehicle accelerates normally and makes a sharp left adjustment following the route.",
    "The vehicle cautiously waits for a gap before normally changing lanes.",
    "The vehicle turns right smoothly while accelerating to follow the maroon car.",
    "The vehicle follows the route, accelerating normally and then making a sharp right adjustment.",
    "The vehicle cautiously adjusts its course to follow the black car, braking if necessary for the walker.",
    "The vehicle follows the route, maintaining reduced speed cautiously with slight adjustments to the left.",
    "The vehicle steadily decelerates to a stop while maintaining a straight course.",
    "The vehicle cautiously decelerates and adjusts course to follow the yellow car ahead.",
    "The vehicle remains stopped, patiently waiting behind the stationary maroon car.",
    "The vehicle remained stopped behind the maroon car 2.8 meters ahead, maintaining a steady course.",
    "The vehicle decelerates smoothly to stay behind the black car ahead, driving cautiously.",
    "The vehicle accelerates through a wide left turn, cautiously monitoring oncoming traffic.",
    "Cautiously adjusting course, the vehicle waits for a gap before changing lanes to the lane with oncoming traffic.",
    "The vehicle remains stopped behind the black car at 8.8 meters, obeying the red traffic light.",
    "The vehicle proceeds through the junction normally due to a green traffic light.",
    "The vehicle accelerates behind the maroon car, normally proceeding on the green traffic light.",
    "The vehicle accelerates steadily as the queue ahead discharges through the green traffic light.",
    "The vehicle follows the route and maintains reduced speed with a steady course.",
    "The vehicle cautiously follows the route and remains stopped behind the black car to the front right.",
    "The vehicle remains stopped behind the front black car due to a red traffic light, normally maintaining its position.",
    "The vehicle follows the front black car forward through the green traffic light, normally maintaining its position.",
    "The vehicle accelerates normally at a green traffic light.",
    "The vehicle remained stopped, steadily maintaining its position behind the maroon SUV.",
    "The vehicle normally follows the route, accelerating and adjusting right.",
    "The vehicle cautiously waits for a gap in traffic before changing lanes.",
    "The vehicle remains stopped normally due to a red traffic light.",
    "The vehicle remained stopped normally due to the red traffic light.",
    "The vehicle follows the route, maintaining reduced speed with a smooth rightward course adjustment.",
    "The vehicle stayed stopped, waiting behind the navy car.",
    "The vehicle follows the dark green car, accelerating and then sharply adjusting course to the left.",
    "The vehicle stops steadily behind the dark green car.",
    "The vehicle accelerates to follow the dark green SUV at 17.4 meters, normally maintaining its lane.",
    "The vehicle follows the route with a smooth rightward adjustment, then a gradual leftward adjustment, normally maintaining its speed before decelerating.",
    "The vehicle turns right normally, accelerating to follow the maroon car 14.6 meters ahead.",
    "The vehicle remains stopped to stay behind the black car that is to the front in 8.8 meters, driving normally.",
    "The vehicle accelerates normally to follow the black car in 9.5 meters.",
    "The vehicle follows the route with steady lane keeping and normal acceleration.",
    "Following the route, the vehicle smoothly adjusts left while maintaining reduced speed behind the yellow car.",
    "The vehicle accelerates normally to follow the gray car ahead, maintaining a steady course.",
    "The vehicle accelerates and makes a wide right adjustment through the junction.",
    "The vehicle smoothly follows the route, normally decelerating and accelerating to match target speed.",
    "The vehicle accelerates normally, maintaining a steady course before making a slight adjustment to the left.",
    "The vehicle accelerates and then adjusts its course smoothly while following the route.",
    "The vehicle steadily follows the route, maintaining a reduced speed behind the front black car.",
    "The vehicle decelerates steadily behind the maroon car, maintaining a straight course.",
    "The vehicle remained stopped behind the front black car at 9.2 meters, normally.",
    "The vehicle remains stopped, cautiously following the black car ahead.",
    "The vehicle remains stopped, then cautiously accelerates slightly forward to maintain position behind the maroon car.",
    "The vehicle remained stopped behind the maroon car to the front right in 8.4 meters.",
    "The vehicle steadily accelerates normally and maintains course along the route.",
    "The car is in a driving scenario.",
    "The vehicle cautiously accelerates to reach the speed limit with steady lane keeping.",
    "The vehicle cautiously decelerates and makes a slight rightward adjustment to follow the black car.",
    "The vehicle remains stopped behind the maroon car, normally adhering to the red traffic light.",
    "The vehicle cautiously stopped, then smoothly accelerated forward, maintaining its course.",
    "The vehicle cautiously waits for a gap before changing lanes.",
    "The car follows the route and maintains reduced speed, normally staying behind the orange car at 10.3 meters.",
    "The vehicle returns to its route with steady lane keeping, normally.",
    "The vehicle remains stopped, steadily maintaining its course behind the maroon car.",
    "The vehicle decelerates smoothly to follow the front car, then cautiously accelerates normally.",
    "The vehicle follows the route, accelerating normally and making a slight right course adjustment.",
    "The vehicle accelerates to follow the dark green SUV ahead, maintaining a steady lane keeping.",
    "The vehicle normally follows the route, steadily maintaining speed while making slight right adjustments.",
    "The vehicle normally accelerates to follow the front car, then abruptly decelerates to a stop.",
    "The vehicle remains stopped, following the route cautiously behind the orange car in 3.1 meters.",
    "The vehicle smoothly decelerates and then accelerates to follow the navy car ahead.",
    "The vehicle follows the black car, accelerating and then braking cautiously for the pedestrian.",
    "The vehicle follows the route, maintaining reduced speed cautiously behind the black SUV at 11.7 meters.",
    "The vehicle smoothly decelerates to follow the dark green SUV ahead, then normally accelerates forward.",
    "The vehicle remained stopped, cautiously preparing to follow the maroon car ahead.",
    "The vehicle follows the route, steadily maintaining speed behind the black car 11.0 meters ahead.",
    "The vehicle accelerates normally into the oncoming lane, then makes a slight, smooth course adjustment.",
    "The vehicle follows the route, maintaining reduced speed with slight course adjustments.",
    "The vehicle turns right smoothly, accelerating to stay behind the silver SUV ahead.",
    "The vehicle follows the route with steady lane keeping and normally maintained reduced speed.",
    "The vehicle remained stopped, waiting behind the navy car.",
    "The vehicle accelerates normally while making a slight adjustment to the left.",
    "The vehicle cautiously waits for a gap in traffic before changing lanes.",
    "The vehicle normally follows the route, maintaining reduced speed behind the navy SUV ahead with slight right adjustments.",
    "The vehicle smoothly adjusts its course leftward while steadily maintaining speed behind the maroon car.",
    "The vehicle smoothly adjusts course right while normally maintaining speed behind the dark green car at 10.9 meters.",
    "The vehicle cautiously decelerates to stay behind the front black car at 23.6 meters.",
    "The vehicle remained stopped, following the black car ahead.",
    "The vehicle cautiously makes a left turn after stopping at a stop sign.",
    "The vehicle cautiously decelerates to follow the route, making slight adjustments.",
    "The vehicle remains stopped normally due to the red traffic light.",
    "The vehicle remains stopped normally at a red traffic light.",
    "The vehicle normally makes a wide right adjustment to change lanes.",
    "The vehicle normally follows the route, maintaining a reduced speed behind the navy car to the front right.",
    "The vehicle cautiously waits for a gap in traffic before changing lanes.",
    "The vehicle normally decelerates to stay behind the front right navy car, then accelerates.",
    "The vehicle maintains steady lane keeping, cautiously following the silver car ahead at 19.2 meters.",
    "The vehicle follows the route, accelerating normally to keep pace with the dark green SUV ahead.",
    "The vehicle remains stopped normally due to the red traffic light in 4.8 meters.",
    "The vehicle accelerates normally at the green traffic light.",
    "The vehicle moves off smoothly as the traffic light turns green, maintaining steady lane keeping.",
)

SEED_REASONING: tuple[str, ...] = (
    "Follow the route.",
    "Follow the route. Remain stopped to stay behind the maroon car that is to the front that is stopped because of a red traffic light.",
    "Follow the route. Accelerate to stay behind the maroon car that is to the front that is pulling away because of a green traffic light.",
    "Follow the route. Accelerate to stay behind the black car that is to the front at 8.0 meters that is moving off because of a green traffic light.",
    "Follow the route. Maintain the speed to drive through the junction because the traffic light is green.",
    "Follow the route. Maintain the reduced speed to stay behind the black car that is to the front.",
    "Follow the route. Decelerate due to the stop sign in 1.0 meters.",
    "Follow the route. Decelerate to stay behind the black bicycle that is to the front.",
    "Steer clear of the parked vehicle. Wait for a gap in the traffic before changing lanes to the lane with oncoming traffic.",
    "Turn left. Remain stopped to stay behind the navy car that is to the front left in 21.5 meters.",
    "Follow the route. Decelerate to stay behind the maroon car that is to the front left.",
    "Follow the route. Accelerate to follow the teal car that is to the front in 9.9 meters.",
    "Follow the route. Decelerate to stay behind the black car that is to the front.",
    "Follow the route. Decelerate to drive with the target speed.",
    "Follow the route. Remain stopped to stay behind the black car that is to the front that is stopped because of a red traffic light.",
    "Follow the route. Accelerate to stay behind the black car that is to the front that is moving off because of a green traffic light.",
    "Follow the route. Accelerate to stay behind the gray car that is to the front that is moving off because of a green traffic light.",
    "Follow the route. Remain stopped to stay behind the maroon car that is to the front in 4.7 meters.",
    "Turn right. Remain stopped due to the red traffic light in 1.5 meters.",
    "Turn right. Accelerate due to the green traffic light in 1.5 meters.",
    "Turn left. Accelerate because the traffic light is green and the oncoming vehicles are stopped.",
    "Follow the route. Accelerate because the traffic light is green but pay attention to the vehicle coming towards the junction.",
    "Follow the route. Accelerate to follow the black car that is to the front.",
    "Follow the route. Maintain your current speed.",
    "Follow the route. Accelerate because the traffic light is green and the other vehicles are stopped at the junction and the vehicle in the junction is moving away.",
    "Stay on your current lane to overtake the bikes on your lane. Remain stopped to stay behind the black SUV that is to the front left.",
    "Follow the route. Maintain the reduced speed to stay behind the silver car that is to the front.",
    "Follow the route. Remain stopped to stay behind the black car that is to the front at 8.0 meters that is stopped because of a red traffic light.",
    "Follow the route. Remain stopped to stay behind the gray car that is to the front that is stopped because of a red traffic light.",
    "Turn left. Maintain your current speed to drive through the junction because the other vehicles are stopped at the junction and the junction is clear.",
    "Follow the route. Remain stopped to stay behind the black car that is to the front that is stopped because of a red traffic light.",
    "Follow the route. Remain stopped to stay behind the dark green SUV that is to the front.",
    "Follow the route. Decelerate due to the pedestrian intersecting your path.",
    "Follow the route. Accelerate since you cleared the stop sign but pay attention to the vehicle in the junction.",
    "Follow the route. Accelerate to drive through the junction.",
    "Follow the route. Maintain the reduced speed to stay behind the navy car that is to the front.",
    "Follow the route. Accelerate to follow the black car that is to the front.",
    "Follow the route. Accelerate to drive with the target speed.",
    "Follow the route. Accelerate to drive through the junction.",
    "Follow the route. Maintain the reduced speed to stay behind the dark blue car that is to the front in 12.5 meters.",
    "Follow the route. Maintain the reduced speed to stay behind the yellow car that is to the front in 25.0 meters.",
    "Turn right. Accelerate to drive with the target speed because the other vehicles are stopped at the junction and the junction is clear.",
    "Shift a bit to the right to make space for the traffic that invades the lane because of the traffic cones. Maintain your current speed.",
    "Shift a bit to the right to make space for the traffic that invades the lane because of the traffic cones. Maintain your current speed.",
    "Follow the route. Maintain the reduced speed to stay behind the navy SUV that is to the front in 10.1 meters.",
    "Follow the route. Decelerate to stay behind the maroon car that is to the front right.",
    "Follow the route. Maintain the reduced speed to stay behind the dark green car that is to the front in 12.2 meters.",
    "Follow the route. Remain stopped to stay behind the black car that is to the front that is stopped because of a red traffic light.",
    "Follow the route. Decelerate to stay behind the maroon car that is to the front left.",
    "Follow the route. Accelerate to follow the navy car that is to the front in 8.8 meters.",
    "Follow the route. Accelerate to follow the maroon car that is to the front.",
    "Follow the route. Accelerate since you cleared the stop sign and the other vehicles are stopped at the junction and the junction is clear.",
    "Follow the route. Maintain the reduced speed to drive through the junction.",
    "Return to your original route after avoiding the obstacle. Maintain the reduced speed.",
    "Avoid the accident on your lane. Wait for a gap in the traffic before changing lanes to the lane with oncoming traffic.",
)



# ── Structured per-chunk credit + subtask suggestions ────────────────────────────────


def _is_json_null(value: Any) -> bool:
    """Whether a parsed JSON field means "absent" — including the *stringly-typed* nulls.

    The credit prompt asks for ``label: "GOOD", "BAD", or null``, and the VLM sometimes emits the
    string ``"null"`` (or ``"none"``/``"n/a"``) instead of a real JSON ``null``. Before this, such a
    value raised out of :meth:`ChunkCredit.from_dict`, and because the parse is per-window that one
    malformed chunk discarded the **entire** window's credit assignment — ~15 HL samples lost over a
    difference in quoting. Observed once in 215 windows.
    """
    if value is None:
        return True
    text = str(value).strip().lower()
    return text in ("", "null", "none", "nil", "n/a", "na")


@dataclass(frozen=True)
class ChunkCredit:
    """Credit assignment + suggested subtasks (and a new reasoning trace) for one action chunk."""

    chunk_index: int
    label: str | None  # "GOOD" | "BAD" | None (no clear signal)
    rationale: str = ""
    suggested_subtasks: tuple[str, ...] = ()
    # Fresh chain-of-thought reasoning that justifies the corrected subtask. Only populated for
    # chunks whose subtask the VLM changes (BAD chunks); empty otherwise. Used as the ``reasoning``
    # target of the stored high-level SteerVLA sample.
    suggested_reasoning: str = ""
    # Why a BAD chunk is BAD: "direct" when a BAD event overlaps the chunk's own time range,
    # "precursor" when the chunk itself looks unremarkable but set up a later BAD event and a
    # different subtask here would have pre-empted it. Empty for GOOD/unlabeled chunks. Both
    # kinds are corrected identically downstream (see :func:`_resolve_hl_targets`); this only
    # records provenance for artifacts/metrics.
    credit_source: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "chunk_index": self.chunk_index,
            "label": self.label,
            "rationale": self.rationale,
            "suggested_subtasks": list(self.suggested_subtasks),
            "suggested_reasoning": self.suggested_reasoning,
            "credit_source": self.credit_source,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ChunkCredit":
        idx = int(raw["chunk_index"])
        label_raw = raw.get("label")
        label: str | None = None
        if not _is_json_null(label_raw):
            label_val = str(label_raw).strip().upper()
            if label_val not in ("GOOD", "BAD"):
                raise ValueError(f"chunk label must be GOOD or BAD, got {label_raw!r}.")
            label = label_val
        subtasks_raw = raw.get("suggested_subtasks", []) or []
        if not isinstance(subtasks_raw, list):
            raise ValueError("suggested_subtasks must be a list of strings.")
        suggested = tuple(str(s).strip() for s in subtasks_raw if str(s).strip())
        source_raw = raw.get("credit_source")
        credit_source = "" if _is_json_null(source_raw) else str(source_raw).strip().lower()
        if credit_source not in ("", "direct", "precursor"):
            raise ValueError(f"credit_source must be direct or precursor, got {source_raw!r}.")
        if label == "BAD" and not credit_source:
            # Older/looser responses omit the field; a BAD chunk with no stated source is the
            # original overlap-based credit.
            credit_source = "direct"
        if label != "BAD":
            credit_source = ""
        return cls(
            chunk_index=idx,
            label=label,
            rationale=str(raw.get("rationale", "")).strip(),
            suggested_subtasks=suggested,
            suggested_reasoning=str(raw.get("suggested_reasoning", "")).strip(),
            credit_source=credit_source,
        )


def _extract_json_payload(text: str) -> dict[str, Any]:
    """Parse model output that may include markdown fences or extra prose."""
    stripped = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL)
    if fence_match:
        stripped = fence_match.group(1)
    else:
        brace_match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if brace_match:
            stripped = brace_match.group(0)
    payload = json.loads(stripped)
    if not isinstance(payload, dict):
        raise ValueError("CAST relabel response must be a JSON object.")
    return payload


def retime_chunk_specs(
    chunk_specs: list[ActionChunkSpec],
    metadata: dict[str, Any],
) -> list[ActionChunkSpec]:
    """Snap each chunk's video time range onto the frames actually recorded for its steps.

    :func:`build_action_chunk_specs` lays chunks out on a uniform clock
    (``chunk_index * chunk_duration_sec``), which assumes exactly
    ``action_chunk_steps / video_frame_stride`` frames per chunk. That is not what the rollout
    records: ``main_carla`` samples a frame every ``video_frame_stride`` env steps **and** on every
    collision / traffic-violation step, so an eventful chunk contributes extra frames and every
    later chunk's true position in the video slides earlier than the uniform grid says. Since the
    VLM reads its event timestamps off the video and we then map those timestamps onto this table,
    the drift lands directly on the credit assignment.

    Here each chunk is re-timed from ``metadata["steps"]`` — whose ``video_timestamp_sec`` values
    are the window-relative frame times computed by
    :meth:`OnlineCastRelabelSession._build_metadata` — so the table describes the video as encoded.
    Chunks with no recorded frame keep their uniform-grid estimate.
    """
    steps = metadata.get("steps") or []
    fps = float(metadata.get("video_fps") or 0.0)
    if not steps or fps <= 0.0:
        return list(chunk_specs)
    frame_dt = 1.0 / fps
    out: list[ActionChunkSpec] = []
    for spec in chunk_specs:
        # ``episode_step_start/end`` are window-relative and 1-based; ``steps`` is the window.
        span = steps[max(0, int(spec.episode_step_start) - 1): int(spec.episode_step_end)]
        times = [
            float(s["video_timestamp_sec"])
            for s in span
            if isinstance(s, dict) and s.get("video_timestamp_sec") is not None
        ]
        if not times:
            out.append(spec)
            continue
        out.append(
            replace(
                spec,
                video_time_start_sec=round(min(times), 3),
                # The last frame of the chunk is on screen until the next one replaces it.
                video_time_end_sec=round(max(times) + frame_dt, 3),
            )
        )
    return out


def build_chunks_payload(
    chunk_specs: list[ActionChunkSpec],
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    """Chunk timing table for the credit prompt, with the env reward earned over each chunk.

    The credit-assignment call gets neither the video nor the per-step trajectory data — only the
    prose event list from the window review. Folding the per-step ``reward_total`` into each chunk
    gives it the one hard, per-chunk signal it can otherwise not see: which chunks the environment
    itself paid for and which it penalized. Reward keys are omitted for chunks whose steps carry no
    reward (older trajectories), so the prompt never shows misleading zeros.

    ``original_subtask`` is the subtask the policy actually executed over the chunk. Without it the
    prompt's "keep the original subtask for GOOD chunks" and "smooth the subtasks relative to the
    adjacent subtasks" instructions are unactionable — the credit call is a fresh, text-only
    request (:meth:`vlm_feedback.GeminiVLMCOach.complete_text` carries no history from the review
    call), so anything not serialized here is invisible to it.
    """
    # ``metadata["steps"]`` is the window in order, so position ``i`` is window-relative step
    # ``i + 1`` — the numbering ``chunk_specs`` uses. Keying by the record's ``episode_step``
    # instead (absolute: 301-450 by the third window of an episode, against spec ranges of 1-150)
    # matched nothing for every window after the first, so the reward columns the prompt then
    # spends a paragraph explaining how to weigh were simply absent from all of them.
    step_rewards: dict[int, float] = {}
    for idx, s in enumerate(metadata.get("steps", []) or []):
        if not isinstance(s, dict) or s.get("reward_total") is None:
            continue
        step_rewards[idx + 1] = float(s["reward_total"])

    originals = metadata.get("chunk_original_subtask", {}) or {}

    payload: list[dict[str, Any]] = []
    for spec in chunk_specs:
        entry: dict[str, Any] = {
            "chunk_index": spec.chunk_index,
            "episode_step_start": spec.episode_step_start,
            "episode_step_end": spec.episode_step_end,
            "video_time_start_sec": round(spec.video_time_start_sec, 3),
            "video_time_end_sec": round(spec.video_time_end_sec, 3),
        }
        # Stripped again here so the prompt is clean even for metadata built elsewhere.
        original = strip_cot_sentinels(originals.get(str(spec.chunk_index), ""))
        if original:
            entry["original_subtask"] = original
        span = [
            step_rewards[st]
            for st in range(int(spec.episode_step_start), int(spec.episode_step_end) + 1)
            if st in step_rewards
        ]
        if span:
            entry["reward_total_sum"] = round(sum(span), 4)
            entry["reward_total_mean"] = round(sum(span) / len(span), 4)
            entry["reward_total_min"] = round(min(span), 4)
        payload.append(entry)
    return payload


_DEFAULT_SCORE_OBJECTIVE = (
    "how well it advances the route safely and efficiently from the current scene (avoid collisions "
    "and stalls; make forward progress toward route completion; prefer candidates consistent with the "
    "reward context)"
)


def build_candidate_score_prompt(
    context: dict[str, Any],
    candidate_subtasks: list[str],
    *,
    objective: str | None = None,
) -> str:
    """Prompt the VLM critic to score K candidate next-subtasks for the current driving scene.

    Used by the GRPO HL path: ``context`` carries the env signals the critic should weigh (speed,
    route progress, cumulative + recent reward, collisions), and the current frame is attached as an
    image part by :meth:`GeminiVLMCOach.complete_image_text`. ``objective`` overrides the scoring
    criterion (e.g. the debug stop task passes a "prefer stopping" objective that matches a -speed
    reward); when None the default route-progress criterion is used.
    """
    cand_block = "\n".join(f"{i}: {s}" for i, s in enumerate(candidate_subtasks))
    n = len(candidate_subtasks)
    return textwrap.dedent(
        f"""
        You are grading candidate next-actions for an autonomous vehicle. The attached image is the
        current front-camera view. Below is the driving context (env reward signals included) followed
        by {n} candidate next-subtasks the policy is considering from THIS state.

        Driving context:
        ```json
        {json.dumps(context, indent=2)}
        ```

        Candidate next-subtasks (index: text):
        {cand_block}

        Score each candidate in [0, 1] for {objective or _DEFAULT_SCORE_OBJECTIVE}. Respond with ONE
        line of raw JSON and nothing else -- no markdown fences, no prose, no trailing commas:
        {{"scores": [s_0, ..., s_{n - 1}]}} -- exactly {n} decimals in candidate order, comma-separated.
        """
    ).strip()


def parse_candidate_scores(text: str, *, num: int) -> list[float]:
    """Parse ``{"scores": [...]}`` into ``num`` floats in [0, 1].

    Tries strict JSON, then a regex salvage tolerant of the VLM's usual malformations (missing/
    trailing commas, stray prose). Raises only when neither yields exactly ``num`` numbers, so a
    genuinely unparseable reply still fails loudly instead of training on a silently degraded group.
    """
    try:
        raw = _extract_json_payload(text).get("scores")
        if isinstance(raw, list) and len(raw) == num:
            return [min(1.0, max(0.0, float(v))) for v in raw]
    except (ValueError, json.JSONDecodeError):
        pass
    m = re.search(r'"?scores"?\s*:\s*\[([^\]]*)\]', text, flags=re.DOTALL)
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", m.group(1) if m else text)
    if len(nums) != num:
        raise ValueError(f"VLM candidate scoring could not be parsed into {num} scores; raw reply: {text!r}")
    return [min(1.0, max(0.0, float(v))) for v in nums]


def build_debug_task_prompt(
    *,
    events: list[CoachEvent],
    chunk_specs: list[ActionChunkSpec],
    metadata: dict[str, Any],
    seed_subtasks: tuple[str, ...] = SEED_SUBTASKS,
    seed_reasonings: tuple[str, ...] = SEED_REASONING,
    num_suggestions: int = DEFAULT_NUM_SUBTASK_SUGGESTIONS,
    max_seed_examples: int = 40,
) -> str:
    """Prompt that does credit assignment (step 3) + subtask suggestion (step 4)."""
    events_payload = [
        {
            "timestamp_sec": e.timestamp_sec,
            "label": e.label,
            "description": e.description,
            "correction": e.correction,
        }
        for e in events
    ]
    chunks_payload = build_chunks_payload(chunk_specs, metadata)
    # Present a bounded sample of seed subtasks to keep the prompt small.
    seeds_subtasks = list(seed_subtasks)[:max_seed_examples]
    seed_subtask_block = "\n".join(f"- {s}" for s in seeds_subtasks)
    seeds_reasonings = list(seed_reasonings)[:max_seed_examples]
    seed_reasoning_block = "\n".join(f"- {s}" for s in seeds_reasonings)

    return textwrap.dedent(
        f"""
        You previously reviewed a short driving rollout window and flagged GOOD and BAD
        moments (below). For this debug task, relabel all the segments to subtasks that result in the vehicle remaining stopped or slowing down.

        Each action chunk spans a fixed number of env steps; the chunk timing (in the same
        video seconds as the events) is given below.

        Window summary (``route_progress_end_pct`` < 100 means the route is unfinished — making
        forward progress toward completion is a primary objective):
        ```json
        {json.dumps({k: metadata.get(k) for k in ("episode", "route", "episode_steps", "window_index", "success", "route_progress_start_pct", "route_progress_end_pct", "route_completed", "mean_end_speed_mps")}, indent=2)}
        ```

        GOOD/BAD events from the window review (timestamps are video seconds):
        ```json
        {json.dumps(events_payload, indent=2)}
        ```

        Action chunks in this window:
        ```json
        {json.dumps(chunks_payload, indent=2)}
        ```

        Example subtask phrasings (open vocabulary — reuse verbatim OR write new phrases in
        the SAME concise style; describe what the vehicle should do, not meta commentary). For this debug task, the subtasks should be the ones that result in the vehicle remaining stopped or slowing down.:
        {seed_subtask_block}

        For EVERY chunk_index above, return:
        - label: "GOOD", "BAD", or null. Does not matter for this debug task.
        - rationale: one short sentence describing the slow down or stop behavior.
        - suggested_subtasks: up to {num_suggestions} subtask phrases that would result in the vehicle remaining stopped or slowing down.

        Return ONLY valid JSON (no markdown fences):
        {{
          "chunk_credits": [
            {{
              "chunk_index": 0,
              "label": "BAD",
              "rationale": "The vehicle must slow down.",
              "suggested_subtasks": [
                "The vehicle smoothly decelerates to a stop."
              ]
            }}
          ]
        }}

        Rules:
        - Include exactly one entry per chunk_index listed above.
        - Use null (not "none") for label when no event applies.
        - Keep rationale under 50 words.
        - Never exceed {num_suggestions} suggested subtasks per chunk.
        """
    ).strip()


def build_credit_relabel_prompt(
    *,
    events: list[CoachEvent],
    chunk_specs: list[ActionChunkSpec],
    metadata: dict[str, Any],
    seed_subtasks: tuple[str, ...] = SEED_SUBTASKS,
    seed_reasonings: tuple[str, ...] = SEED_REASONING,
    num_suggestions: int = DEFAULT_NUM_SUBTASK_SUGGESTIONS,
    max_seed_examples: int = 40,
) -> str:
    """Prompt that does credit assignment (step 3) + subtask suggestion (step 4)."""
    events_payload = [
        {
            "timestamp_sec": e.timestamp_sec,
            "label": e.label,
            "description": e.description,
            "correction": e.correction,
        }
        for e in events
    ]
    chunks_payload = build_chunks_payload(chunk_specs, metadata)
    # Present a bounded sample of seed subtasks to keep the prompt small.
    seeds_subtasks = list(seed_subtasks)[:max_seed_examples]
    seed_subtask_block = "\n".join(f"- {s}" for s in seeds_subtasks)
    seeds_reasonings = list(seed_reasonings)[:max_seed_examples]
    seed_reasoning_block = "\n".join(f"- {s}" for s in seeds_reasonings)
    # What earlier windows of this run already corrected (coaches.correction_memory). This call is
    # stateless, so without it every window re-decides the same trade-off from scratch and the HL
    # dataset can end up teaching both directions of one decision.
    memory_block = str(metadata.get("correction_memory") or "")

    return textwrap.dedent(
        f"""
        You previously reviewed a short driving rollout window and flagged GOOD and BAD
        moments (below). Now assign that credit to the specific action chunks the policy
        executed, and for each chunk propose subtasks that would improve the behavior.

        PRIORITY. **Completing the route is the primary objective.** Safety and traffic rules are
        constraints on how the vehicle makes progress, not goals in their own right — a chunk in
        which the vehicle sits still, having broken no rule but advanced the route not at all, is a
        FAILING chunk, not a neutral one. Over-conservatism is as real a defect as recklessness and
        must be labeled BAD with the same willingness: stopping or crawling with a clear path,
        waiting out a gap that was plainly takeable, hesitating at or stalling inside a junction
        instead of completing the turn, braking for something not on the vehicle's path, or
        creeping far below the speed limit on an open road. Whenever you relabel such a chunk, the
        corrected subtask must be one that MOVES the vehicle along the route (take the gap,
        complete the turn, resume speed) — never a further-slowing subtask. Do not relabel a chunk
        toward stopping unless you can name the specific visible hazard or signal that required it.

        Credit is CAUSAL, not just temporal. A bad driving outcome is usually already decided
        several chunks before it becomes visible: the vehicle carried too much speed into the
        approach, drifted toward the wrong lane, started braking too late, or committed to a gap
        that was never there. Those earlier chunks are ALSO at fault, and fixing them is the only
        way to prevent the failure — by the time the BAD event is on screen it is often too late.
        So for each BAD event, walk BACKWARD through the preceding chunks and blame the ones that
        set it up, giving each a subtask that pre-empts the failure.

        Each action chunk spans a fixed number of env steps; the chunk timing (in the same
        video seconds as the events) is given below.

        Window summary (``route_progress_end_pct`` < 100 means the route is unfinished — making
        forward progress toward completion is a primary objective; ``window_reward_total`` is the
        environment reward summed over the window):
        ```json
        {json.dumps({k: metadata.get(k) for k in ("episode", "route", "episode_steps", "window_index", "success", "route_progress_start_pct", "route_progress_end_pct", "route_completed", "mean_end_speed_mps", "window_reward_total", "window_reward_mean")}, indent=2)}
        ```

        GOOD/BAD events from the window review (timestamps are video seconds):
        ```json
        {json.dumps(events_payload, indent=2)}
        ```

        Action chunks in this window. ``original_subtask`` is the subtask the policy ACTUALLY
        executed over that chunk — it is what you are deciding whether to keep or replace, and the
        sequence of them across chunks is what you are smoothing. It is the policy's OWN output
        and is frequently WRONG — most commonly it claims a red traffic light when the light was
        green, or names a hazard that is not present. Never treat it as evidence about the scene:
        the events and established scene above are what settle that. Where an ``original_subtask``
        contradicts them, that chunk is BAD and its subtask MUST be replaced rather than kept, and
        the replacement must describe what the scene actually showed (for a green light, a subtask
        about proceeding or accelerating — not a stopped-at-red one). ``reward_total_sum`` /
        ``reward_total_mean`` /
        ``reward_total_min`` are the environment reward earned over each chunk's steps — the
        objective the policy is trained on. It rises with route progress and falls with collisions,
        route/traffic infractions, and harsh steering or braking. Treat a chunk with a low or
        negative reward as corroborating evidence for BAD, and a precursor chunk that still looks
        well-paid as a reason to check whether the blame really belongs there. Reward is NOT the
        verdict on its own: it is per-step and myopic, so a chunk can be well-paid and still be the
        precursor that made a later failure unavoidable (e.g. carrying speed toward an occlusion
        earns progress reward right up to the near-miss). The event list remains the primary
        evidence:
        ```json
        {json.dumps(chunks_payload, indent=2)}
        ```

        {memory_block}
        Example subtask phrasings (open vocabulary — reuse verbatim OR write new phrases in
        the SAME concise style; describe what the vehicle should do, not meta commentary). HOWEVER
        the new subtasks should NOT reference objects in the scene and should focus on the driving behavior of
        the vehicle to prevent out of distribution objects from destroying the training signal.:
        {seed_subtask_block}
        
        Example reasoning phrasings (open vocabulary — reuse verbatim OR write new phrases in
        the SAME concise style; describe what the driver would think before arriving at the corrected subtask.
        The new reasoning should CAN reference objects in the scene and the driving behavior of the vehicle.):
        {seed_reasoning_block}

        For EVERY chunk_index above, return:
        - label: "GOOD", "BAD", or null.
          * BAD (direct) — a BAD event overlaps this chunk's time range.
          * BAD (precursor) — no BAD event overlaps this chunk, but it is part of the lead-up
            that made a later BAD event unavoidable or hard to avoid. Test each preceding chunk
            with: "if the policy had executed a different subtask HERE, would the later event
            have been prevented or clearly reduced?" If yes, this chunk is BAD/precursor.
          * GOOD — a GOOD event overlaps the chunk AND it is not a precursor to any BAD event.
            A chunk that looks smooth in isolation is NOT good if it set up a later failure;
            prefer BAD/precursor in that case.
          * null — no event applies and the chunk is not a precursor.
        - credit_source: "direct" or "precursor" for BAD chunks (per above); "" otherwise.
        - rationale: one short sentence tying the chunk to its event. For precursor chunks, say
          which later event it leads to and what about this chunk causes it (e.g. "Still at
          11 m/s approaching the occluded crosswalk that produces the 8.0s near-miss.").
        - suggested_subtasks: up to {num_suggestions} subtask phrases that would improve or
          reinforce this chunk. For BAD/direct chunks, suggest corrective subtasks. For
          BAD/precursor chunks, suggest the subtask that should have been executed AT THIS
          MOMENT to pre-empt the later event — the earlier, gentler action (start scrubbing
          speed, hold the lane, wait for a real gap), NOT the emergency reaction that the
          direct chunk needs. The correction must be justifiable from what is observable in
          THIS chunk; do not suggest a subtask that only makes sense with hindsight of the
          event. For GOOD chunks, keep the original subtask. If a chunk is BAD because the
          vehicle stopped or crawled prematurely while the route is unfinished and the way
          ahead is clear (no red light, stop sign, close leading vehicle, or pedestrian/yield),
          suggest a subtask that has it accelerate and make forward progress along the route.
          The same applies to a chunk that waits out a usable gap at a junction, or that
          hesitates or stalls part-way through a turn: the corrective subtask is to take the gap
          and carry the turn through, NOT to keep waiting. Prefer the progress-making correction
          whenever both a progress-making and a further-slowing subtask would be defensible.
          For GOOD or null chunks, keep the chunk's ``original_subtask`` verbatim.
          DO NOT USE "reverse" or "back up" or "backwards" OR ANYTHING LIKE THIS IN THE SUBTASK.
          This will not be understood by the model and will destroy the training signal.
        - suggested_reasoning: for BAD chunks whose subtask you are changing (direct AND
          precursor), write a fresh, concise chain-of-thought (1-3 sentences, present tense)
          that a driver would think BEFORE arriving at the corrected subtask — describe what is
          observed in the scene and why the corrected action is right. For precursor chunks this
          must read as anticipation from the current scene ("the crosswalk ahead is occluded by
          the parked van, so ease off now"), never as knowledge of the future event. This becomes
          the reasoning that leads to the first corrected subtask. Leave it as "" for GOOD or
          null chunks (no change needed). The suggested reasoning can reference objects in the scene, but the subtasks should not.
        
        **IMPORTANT:** Also try to smooth the subtasks relative to the adjacent chunks' subtasks
        (compare against each chunk's ``original_subtask`` above, and against the subtasks you are
        assigning to its neighbours). If there are rapid changes of the subtask, you can suggest a
        subtask that smooths the transition.

        Return ONLY valid JSON (no markdown fences):
        {{
          "chunk_credits": [
            {{
              "chunk_index": 3,
              "label": "BAD",
              "credit_source": "precursor",
              "rationale": "Holds full speed toward the crosswalk occluded by the parked van, leaving no room to stop for the 2.5s near-miss.",
              "suggested_subtasks": [
                "The vehicle cautiously reduces speed, maintaining steady lane keeping."
              ],
              "suggested_reasoning": "A parked van hides the crosswalk on the right, so the vehicle eases off the throttle now to keep a stopping margin if someone steps out."
            }},
            {{
              "chunk_index": 4,
              "label": "BAD",
              "credit_source": "direct",
              "rationale": "Overlaps the 2.5s near-miss with the pedestrian.",
              "suggested_subtasks": [
                "The vehicle smoothly decelerates to a stop, cautiously adjusting course to the left."
              ],
              "suggested_reasoning": "A pedestrian is stepping into the lane ahead on the left, so the vehicle must brake now and ease left to keep clear rather than continue at speed."
            }}
          ]
        }}

        Rules:
        - Include exactly one entry per chunk_index listed above.
        - Use null (not "none") for label when no event applies.
        - Keep rationale under 50 words.
        - Never exceed {num_suggestions} suggested subtasks per chunk.
        - suggested_reasoning must be "" unless the chunk is BAD and you changed its subtask.
        - credit_source must be "" unless the chunk is BAD.
        - A chunk over which route progress did not increase, while the vehicle was free to move,
          is BAD (direct) — not null. Treat a flat stretch of route progress with no visible
          hazard as an event in its own right, with a subtask that gets the vehicle moving again.
        - Only walk blame back as far as it is genuinely actionable: stop once the hazard was not
          yet observable from the vehicle's viewpoint, or the situation was still comfortably
          recoverable by the chunks that follow. Do not blanket every preceding chunk in the
          window — an unrelated chunk that merely happens to come earlier is null, not precursor.
        """
    ).strip()

def parse_credit_relabel_response(
    text: str,
    *,
    num_chunks: int,
    num_suggestions: int = DEFAULT_NUM_SUBTASK_SUGGESTIONS,
) -> list[ChunkCredit]:
    """Convert the credit/subtask JSON into one ``ChunkCredit`` per chunk (in order)."""
    payload = _extract_json_payload(text)
    raw_items = payload.get("chunk_credits", [])
    if not isinstance(raw_items, list):
        raise ValueError("CAST relabel JSON must contain a 'chunk_credits' list.")

    by_index: dict[int, ChunkCredit] = {}
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        credit = ChunkCredit.from_dict(item)
        # Clamp suggestion count defensively.
        if len(credit.suggested_subtasks) > num_suggestions:
            credit = ChunkCredit(
                chunk_index=credit.chunk_index,
                label=credit.label,
                rationale=credit.rationale,
                suggested_subtasks=credit.suggested_subtasks[:num_suggestions],
                suggested_reasoning=credit.suggested_reasoning,
                credit_source=credit.credit_source,
            )
        by_index[credit.chunk_index] = credit

    out: list[ChunkCredit] = []
    for chunk_index in range(num_chunks):
        out.append(by_index.get(chunk_index, ChunkCredit(chunk_index=chunk_index, label=None)))
    return out


def assemble_cast_relabel_json(
    metadata: dict[str, Any],
    *,
    events: list[CoachEvent],
    credits: list[ChunkCredit],
    chunk_specs: list[ActionChunkSpec],
    seed_subtasks: tuple[str, ...] = SEED_SUBTASKS,
    num_suggestions: int = DEFAULT_NUM_SUBTASK_SUGGESTIONS,
) -> dict[str, Any]:
    """Assemble the per-window artifact JSON."""
    events_out = [
        {
            "timestamp_sec": float(e.timestamp_sec),
            "label": e.label,
            "description": e.description,
            "correction": e.correction,
        }
        for e in events
    ]
    chunks_out: list[dict[str, Any]] = []
    credits_by_index = {c.chunk_index: c for c in credits}
    for spec in chunk_specs:
        credit = credits_by_index.get(spec.chunk_index, ChunkCredit(chunk_index=spec.chunk_index, label=None))
        chunks_out.append(
            {
                "chunk_index": spec.chunk_index,
                "episode_step_start": spec.episode_step_start,
                "episode_step_end": spec.episode_step_end,
                "video_time_start_sec": spec.video_time_start_sec,
                "video_time_end_sec": spec.video_time_end_sec,
                "original_subtask": metadata.get("chunk_original_subtask", {}).get(str(spec.chunk_index)),
                "label": credit.label,
                "credit_source": credit.credit_source,
                "rationale": credit.rationale,
                "suggested_subtasks": list(credit.suggested_subtasks),
                "suggested_reasoning": credit.suggested_reasoning,
            }
        )
    return {
        "episode": metadata.get("episode"),
        "route": metadata.get("route"),
        "window_index": metadata.get("window_index"),
        "episode_steps": metadata.get("episode_steps"),
        "action_chunk_steps": metadata.get("action_chunk_steps"),
        "action_chunk_duration_sec": metadata.get("action_chunk_duration_sec"),
        "num_subtask_suggestions": num_suggestions,
        "events": events_out,
        "action_chunks": chunks_out,
    }


def generate_cast_relabel(
    coach: Any,
    video_path: str | Path,
    metadata: dict[str, Any],
    *,
    steps_per_chunk: int = DEFAULT_ACTION_CHUNK_STEPS,
    chunk_duration_sec: float = DEFAULT_CHUNK_DURATION_SEC,
    seed_subtasks: tuple[str, ...] = SEED_SUBTASKS,
    num_suggestions: int = DEFAULT_NUM_SUBTASK_SUGGESTIONS,
    plot_paths: list[Path] | None = None,
    include_plots_in_prompt: bool = False,
    debug_task: bool = False,
    two_stage: bool = False,
    out_calls: dict[str, Any] | None = None,
) -> tuple[list[CoachEvent], dict[str, Any]]:
    """Full pipeline for one window: review -> credit assignment -> subtask suggestion.

    Returns ``(events, cast_relabel_json)``.

    ``two_stage`` splits the review into two calls over the same video — Step 1 (scene +
    traffic-flow state) answered as prose, then Step 2 (GOOD/BAD events) with that answer given
    as established context. Costs one extra generate_content per window (the video is uploaded
    once either way) and makes the scene reasoning an inspectable artifact rather than hidden
    intermediate tokens.

    ``out_calls``, when given, is filled in place with every prompt and raw response this window
    produced (review + credit). Nothing else reads it; it exists so the caller can persist the
    transcript for the review viewer.
    """
    calls: dict[str, Any] = out_calls if out_calls is not None else {}
    chunk_specs = build_action_chunk_specs(
        metadata,
        steps_per_chunk=steps_per_chunk,
        chunk_duration_sec=chunk_duration_sec,
    )
    # The uniform grid above assumes a fixed number of frames per chunk; the recorded video does
    # not have one (see retime_chunk_specs). Snap the table to the frames that were actually
    # encoded, so the chunk time ranges are on the same clock as the events the VLM reads off it.
    chunk_specs = retime_chunk_specs(chunk_specs, metadata)
    # Step 2: VLM watches the window and flags what went well (GOOD) / poorly (BAD).
    calls["two_stage"] = bool(two_stage)
    if two_stage:
        staged = coach.analyze_two_stage(
            video_path,
            metadata,
            plot_paths=plot_paths,
            include_plots_in_prompt=include_plots_in_prompt,
        )
        events = staged["events"]
        calls["scene_analysis"] = staged.get("scene_analysis", "")
        calls["review_stage1_prompt"] = staged.get("stage1_prompt", "")
        calls["review_stage1_response"] = staged.get("stage1_response", "")
        calls["review_stage2_prompt"] = staged.get("stage2_prompt", "")
        calls["review_stage2_response"] = staged.get("stage2_response", "")
        divergence = staged.get("route_divergence") or {"diverged": False}
    else:
        review_out: dict[str, Any] = {}
        events = coach.analyze(
            video_path,
            metadata,
            plot_paths=plot_paths,
            include_plots_in_prompt=include_plots_in_prompt,
            out=review_out,
        )
        # Single-call path: record the prompt that was sent. The raw response is not returned by
        # ``analyze`` (it parses in place), so it is recovered through ``out=`` above -- which is
        # also how the route_divergence verdict, which rides in that same JSON, gets here.
        calls["review_prompt"] = build_coaching_prompt(
            metadata, include_plots=bool(include_plots_in_prompt and plot_paths)
        )
        calls["review_response"] = review_out.get("response", "")
        divergence = review_out.get("route_divergence") or {"diverged": False}

    # Route divergence: the ego took a branch the routing command did not ask for, so from this
    # moment on it is solving a task nobody set it. Everything after the divergence is cut BEFORE
    # credit assignment runs -- relabeling off-route chunks would spend VLM calls inventing
    # corrections for a situation the policy should never have been in, and would then train on
    # them. The chunk that straddles the divergence goes too: it contains the wrong turn itself.
    calls["route_divergence"] = divergence
    divergence_ts = divergence.get("timestamp_sec") if divergence.get("diverged") else None
    if divergence_ts is not None:
        kept_specs = [c for c in chunk_specs if float(c.video_time_end_sec) <= float(divergence_ts)]
        dropped = len(chunk_specs) - len(kept_specs)
        # Events after the divergence describe off-route driving; leaving them in would let the
        # credit pass attribute post-divergence blame onto the on-route chunks we are keeping.
        kept_events = [e for e in events if float(e.timestamp_sec) <= float(divergence_ts)]
        print(
            f"[cast_relabel] route divergence at t={float(divergence_ts):.2f}s "
            f"({divergence.get('reason', '')!r}): dropping {dropped} of {len(chunk_specs)} chunks "
            f"and {len(events) - len(kept_events)} of {len(events)} events before credit "
            f"assignment.",
            flush=True,
        )
        calls["route_divergence_dropped_chunks"] = dropped
        # Chunk index, NOT an episode step: ActionChunkSpec.episode_step_start is window-relative
        # and 1-based (see retime_chunk_specs), while the session's HL cutoff is compared against
        # absolute episode steps. Mixing the two would cut at the wrong place, so the caller
        # derives its absolute cutoff from the divergence timestamp instead.
        cut_chunks = [
            int(c.chunk_index)
            for c in chunk_specs
            if float(c.video_time_end_sec) > float(divergence_ts)
        ]
        calls["route_divergence_first_dropped_chunk"] = min(cut_chunks) if cut_chunks else None
        chunk_specs, events = kept_specs, kept_events

    calls["events"] = [
        {"timestamp_sec": e.timestamp_sec, "label": e.label,
         "description": e.description, "correction": e.correction}
        for e in events
    ]
    # Steps 3 + 4: credit assignment onto chunks + suggested subtasks per chunk.
    if not hasattr(coach, "complete_text"):
        raise RuntimeError(f"Coach {type(coach).__name__} does not support text completion.")
        
    prompt = build_credit_relabel_prompt(
        events=events,
        chunk_specs=chunk_specs,
        metadata=metadata,
        seed_subtasks=seed_subtasks,
        num_suggestions=num_suggestions,
    )
    
    if debug_task:
        prompt = build_debug_task_prompt(
            events=events,
            chunk_specs=chunk_specs,
            metadata=metadata,
            seed_subtasks=seed_subtasks,
            num_suggestions=num_suggestions,
        )
    calls["credit_prompt"] = prompt
    response_text = coach.complete_text(prompt)
    calls["credit_response"] = response_text
    creds = parse_credit_relabel_response(
        response_text, num_chunks=len(chunk_specs), num_suggestions=num_suggestions
    )
    cast_json = assemble_cast_relabel_json(
        metadata,
        events=events,
        credits=creds,
        chunk_specs=chunk_specs,
        seed_subtasks=seed_subtasks,
        num_suggestions=num_suggestions,
    )
        
    return events, cast_json


# ── High-level (VLM-backbone) dataset in steervla_hl_dataset_format ───────────────────


@dataclass
class HLSample:
    """One SteerVLA high-level training sample (``steervla_hl_dataset_format`` schema).

    Mirrors the per-frame dict the OpenPI SteerVLA RLDS loader emits for an HL dataset
    (``openpi.training.steervla_rlds_dataset`` SIMLINGO restructure + ``enable_cot``):
    an image + ego state + prompt, plus the CoT ``subtask`` / ``reasoning`` targets, and an
    action chunk whose ``action_loss_mask`` is all-``False`` (``action_supervision=False``) so
    only the VLM backbone is supervised.

    ``state`` is stored as the **raw CARLA ego-state vector** (index 15 = speed m/s, index 5 =
    yaw deg) rather than the loader's pre-normalized proprio, so the downstream SteerVLA input
    transform can normalize it exactly like the online actor does from the same raw obs.
    """

    image: np.ndarray  # (H, W, 3) uint8
    state: np.ndarray  # raw CARLA ego-state vector, float32
    current_speed: float
    prompt: str
    subtask: str
    reasoning: str
    actions: np.ndarray  # (action_chunk_steps, action_dim) float32 (unsupervised placeholder ok)
    action_loss_mask: np.ndarray  # (action_chunk_steps,) bool — all False for HL
    # Provenance (kept in the manifest, not fed to the model).
    episode: int
    window_index: int
    chunk_index: int
    episode_step: int
    label: str | None
    # "direct" | "precursor" | "" — see ``ChunkCredit.credit_source``. Lets a later analysis
    # separate reactive corrections from pre-emptive ones.
    credit_source: str = ""
    # True only when ``subtask``/``reasoning`` are the original CoT the model produced for ``actions``
    # (the reinforce path), so the action is a valid FAST-supervision target. False on the correct/
    # relabel path, where the subtask was replaced but the action is still the uncorrected one.
    action_matches_subtask: bool = False
    # ── offline-conversion extras (ignored by the online ``update_hl`` path) ──────────────
    # ``(ego_history_len, 2)`` float32 of ``[speed_mps, course_deg]`` over the env steps ending at
    # this sample's step (oldest first). Feeds the RLDS ``observation/ego_hist`` field; ``None``
    # when no ego history was captured (the converter then tiles the current pair).
    ego_hist: np.ndarray | None = None
    # Bare routing instruction ("Turn right in 20 meter.") *without* the "The current speed is
    # X m/s. " prefix that ``prompt`` carries. The RLDS loader rebuilds the prompt from
    # ``routing_command`` + ``speed``, so storing it verbatim avoids a lossy prefix strip.
    routing_command: str = ""
    # The CoT the model actually produced at rollout, kept even on the corrective path (where
    # ``subtask``/``reasoning`` hold the VLM's replacement) so a converted dataset can report /
    # filter on what was changed.
    original_subtask: str = ""
    original_reasoning: str = ""
    route: str = ""
    global_step: int = -1
    # Pooled runs (impls/cast_pool.py): which published policy version produced this sample.
    # Captured at chunk-start, not at window flush, because workers hot-reload mid-episode -- a
    # single window can straddle two versions. -1 outside a pooled run.
    policy_version: int = -1

    def manifest_entry(self, sample_file: str) -> dict[str, Any]:
        return {
            "sample_file": sample_file,
            "prompt": self.prompt,
            "subtask": self.subtask,
            "reasoning": self.reasoning,
            "episode": self.episode,
            "window_index": self.window_index,
            "chunk_index": self.chunk_index,
            "episode_step": self.episode_step,
            "label": self.label,
            "credit_source": self.credit_source,
            "action_matches_subtask": bool(self.action_matches_subtask),
            "image_shape": [int(x) for x in self.image.shape],
            "state_dim": int(self.state.shape[-1]),
            "action_chunk_steps": int(self.actions.shape[0]),
            "action_dim": int(self.actions.shape[-1]),
            "action_supervision": False,
            # Offline-conversion extras (see the field docs above).
            "routing_command": self.routing_command,
            "original_subtask": self.original_subtask,
            "original_reasoning": self.original_reasoning,
            "route": self.route,
            "global_step": int(self.global_step),
            "current_speed": float(self.current_speed),
            "ego_history_len": 0 if self.ego_hist is None else int(np.asarray(self.ego_hist).shape[0]),
            "policy_version": int(self.policy_version),
        }


def _resolve_hl_targets(
    chunk: dict[str, Any],
    model_input: dict[str, Any],
    *,
    relabel_all: bool = False,
) -> tuple[str, str, bool] | None:
    """Pick the ``(subtask, reasoning, action_matches_subtask)`` HL targets for one chunk, or ``None``.

    ``action_matches_subtask`` is ``True`` only on the *reinforce* path, where the stored subtask is
    the original CoT the model produced for the executed action chunk — so that chunk's action and
    subtask are consistent and its FAST action tokens are a valid supervision target. On the *correct*
    path (BAD or ``relabel_all``) the subtask is replaced but the stored action is still the original
    (uncorrected) one, so the pair is inconsistent and the action must NOT be FAST-supervised under
    the new subtask.

    - **BAD** chunks are *corrected*: the first suggested subtask becomes the target and the
      freshly-generated ``suggested_reasoning`` (falling back to the credit rationale) the
      reasoning. Skipped when the VLM offered no suggestion (nothing to correct toward). This
      covers both ``credit_source`` kinds — a "precursor" chunk (blamed for setting up a later
      BAD event rather than overlapping one) is corrected exactly like a "direct" chunk, which
      is the point: it teaches the pre-emptive subtask at the moment it was still actionable.
    - **GOOD / unlabeled** chunks are *reinforced as-is*: the original subtask + reasoning the
      model produced at rollout (stashed on ``model_input``) become the targets, so good
      high-level behavior is imitated rather than corrected. Skipped when no original subtask
      was captured (nothing to reinforce).

    ``relabel_all`` (the cast_relabel **debug task**) forces *every* chunk to use its suggested
    subtask regardless of ``label``. The debug prompt corrects every chunk toward "remain stopped
    / slow down" but leaves ``label`` null, so without this the label gate would fall through to
    "reinforce original" and store the model's original subtask instead of the relabel. When a
    chunk has no suggestion, we fall back to reinforcing the original.
    """
    label = chunk.get("label")
    label_str = str(label).strip().upper() if label is not None else ""
    if relabel_all or label_str == "BAD":
        suggested = chunk.get("suggested_subtasks") or []
        subtask_target = str(suggested[0]).strip() if suggested else ""
        if subtask_target:
            reasoning_target = str(chunk.get("suggested_reasoning") or "").strip()
            if not reasoning_target:
                reasoning_target = str(chunk.get("rationale") or "").strip()
            # Corrected subtask: the executed action no longer matches it.
            return subtask_target, reasoning_target, False
        if label_str == "BAD" and not relabel_all:
            # BAD with nothing suggested: nothing to correct toward.
            return None
        # relabel_all but no suggestion for this chunk -> fall back to reinforcing the original.

    # GOOD or no label -> reinforce the original CoT the model produced for this chunk. The stored
    # action was taken under exactly this subtask, so it is a valid FAST-supervision target.
    subtask_target = str(model_input.get("subtask") or "").strip()
    if not subtask_target:
        return None
    reasoning_target = str(model_input.get("reasoning") or "").strip()
    return subtask_target, reasoning_target, True


def build_hl_samples_from_window(
    cast_json: dict[str, Any],
    chunk_specs: list[ActionChunkSpec],
    traj_window: list[dict[str, Any]],
    model_inputs: dict[int, dict[str, Any]],
    *,
    action_chunk_steps: int,
    action_dim: int,
    episode: int,
    window_index: int,
    store_good_chunks: bool = True,
    relabel_all: bool = False,
    route: str = "",
    ego_history_len: int = DEFAULT_EGO_HISTORY_LEN,
) -> list[HLSample]:
    """Turn a window's chunks into ``steervla_hl_dataset_format`` samples.

    BAD/relabeled chunks are stored as *corrective* targets. When ``store_good_chunks`` is set,
    GOOD and unlabeled chunks are also stored, *reinforcing* the original subtask/reasoning the
    model produced (see :func:`_resolve_hl_targets`). The model input (image / raw state / speed /
    prompt / executed chunk) is looked up by the chunk's absolute start episode step. Every HL
    sample keeps ``action_loss_mask`` all-``False`` (``action_supervision=False``): ``update_hl``
    supervises only the VLM backbone, so GOOD and BAD chunks alike train the CoT, not the action.

    ``relabel_all`` (the cast_relabel debug task) uses the suggested subtask for *every* chunk
    regardless of ``label`` and stores all chunks (the store_good_chunks gate is bypassed), so the
    whole window is relabeled toward the debug target rather than only the BAD chunks.
    """
    specs_by_index = {int(s.chunk_index): s for s in chunk_specs}
    samples: list[HLSample] = []
    for chunk in cast_json.get("action_chunks", []):
        label = chunk.get("label")
        label_str = str(label).strip().upper() if label is not None else ""
        is_bad = label_str == "BAD"
        if not is_bad and not store_good_chunks and not relabel_all:
            continue
        chunk_index = int(chunk.get("chunk_index", -1))
        spec = specs_by_index.get(chunk_index)
        if spec is None:
            continue
        # Window-relative 1-based start step -> the recorded trajectory step -> absolute episode step.
        traj_idx = int(spec.episode_step_start) - 1
        if traj_idx < 0 or traj_idx >= len(traj_window):
            continue
        abs_episode_step = int(traj_window[traj_idx].get("episode_step", -1))
        model_input = model_inputs.get(abs_episode_step)
        if model_input is None or model_input.get("image") is None:
            # No captured model input for this chunk start (e.g. state-only obs); skip.
            continue

        targets = _resolve_hl_targets(chunk, model_input, relabel_all=relabel_all)
        if targets is None:
            continue
        subtask_target, reasoning_target, action_matches_subtask = targets

        prompt = str(model_input.get("prompt") or "").strip()
        actions_src = model_input.get("action_chunk")
        actions = _shape_hl_action_chunk(actions_src, action_chunk_steps, action_dim)
        action_loss_mask = np.zeros((int(action_chunk_steps),), dtype=bool)

        state_vec = np.asarray(model_input.get("state"), dtype=np.float32).reshape(-1)
        ego_hist = model_input.get("ego_hist")
        if ego_hist is None:
            ego_hist = _ego_hist_from_state(
                state_vec,
                current_speed=float(model_input.get("current_speed", 0.0)),
                ego_history_len=ego_history_len,
            )
        samples.append(
            HLSample(
                image=np.asarray(model_input["image"], dtype=np.uint8),
                state=state_vec,
                current_speed=float(model_input.get("current_speed", 0.0)),
                prompt=prompt,
                subtask=subtask_target,
                reasoning=reasoning_target,
                actions=actions,
                action_loss_mask=action_loss_mask,
                episode=int(episode),
                window_index=int(window_index),
                chunk_index=chunk_index,
                episode_step=abs_episode_step,
                label=(label_str or None),
                credit_source=str(chunk.get("credit_source") or ""),
                action_matches_subtask=bool(action_matches_subtask),
                ego_hist=np.asarray(ego_hist, dtype=np.float32),
                routing_command=str(model_input.get("routing_command") or ""),
                original_subtask=str(model_input.get("subtask") or ""),
                original_reasoning=str(model_input.get("reasoning") or ""),
                route=str(route),
                global_step=int(model_input.get("global_step", -1)),
                policy_version=int(model_input.get("policy_version", -1)),
            )
        )
    return samples


def _ego_hist_from_state(
    state_vec: np.ndarray | None,
    *,
    current_speed: float,
    ego_history_len: int,
) -> np.ndarray:
    """``(ego_history_len, 2)`` of ``[speed, course]`` built by tiling the *current* pair.

    Fallback for samples with no captured per-step history (and the reader-side fallback for
    datasets written before ``ego_hist`` existed). Only the last pair is used when the OpenPI
    loader runs with ``include_ego_history=False``, so tiling is exact for that configuration
    and a constant-history approximation for the ego-history one.
    """
    speed = float(current_speed)
    course = 0.0
    if state_vec is not None:
        flat = np.asarray(state_vec, dtype=np.float32).reshape(-1)
        if flat.size > EGO_STATE_IDX_SPEED:
            speed = float(flat[EGO_STATE_IDX_SPEED])
        if flat.size > EGO_STATE_IDX_YAW_RATE:
            course = float(flat[EGO_STATE_IDX_YAW_RATE]) * SIMLINGO_FRAME_DT
    return np.tile(np.array([speed, course], dtype=np.float32), (max(1, int(ego_history_len)), 1))


def _shape_hl_action_chunk(
    actions_src: Any,
    action_chunk_steps: int,
    action_dim: int,
) -> np.ndarray:
    """Coerce a (possibly flattened / missing) executed chunk to ``(action_chunk_steps, action_dim)``.

    The stored chunk is unsupervised for HL samples (``action_loss_mask`` all-False), so a
    zero placeholder is fine when the executed action is unavailable or mis-shaped.
    """
    steps = int(action_chunk_steps)
    dim = int(action_dim)
    out = np.zeros((steps, dim), dtype=np.float32)
    if actions_src is None:
        return out
    arr = np.asarray(actions_src, dtype=np.float32).reshape(-1)
    if arr.size == steps * dim:
        return arr.reshape(steps, dim)
    if arr.size == dim:  # single-step action; place it in the first row.
        out[0, :] = arr
    return out


def write_hl_samples(samples: list[HLSample], out_dir: Path) -> list[dict[str, Any]]:
    """Persist HL samples: one ``.npz`` (arrays) per sample + a ``hl_samples.json`` manifest.

    Layout (per window ``tag`` dir)::

        <out_dir>/sample_0000.npz   # image, state, ego_hist, current_speed, actions, action_loss_mask
        <out_dir>/hl_samples.json   # dataset_format + per-sample text targets + provenance

    ``ego_hist`` and the extra manifest fields (``routing_command`` / ``original_*`` / ``route`` /
    ``global_step``) exist for the offline RLDS conversion (:mod:`vlas.cast_hl_to_rlds`); the online
    ``SteerVLAActor.update_hl`` reader ignores every key it does not know, so old and new pools
    interleave freely.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for i, s in enumerate(samples):
        sample_file = f"sample_{i:04d}.npz"
        arrays: dict[str, Any] = {
            "image": s.image,
            "state": s.state,
            "current_speed": np.float32(s.current_speed),
            "actions": s.actions,
            "action_loss_mask": s.action_loss_mask,
        }
        if s.ego_hist is not None:
            arrays["ego_hist"] = np.asarray(s.ego_hist, dtype=np.float32)
        np.savez_compressed(out_dir / sample_file, **arrays)
        manifest.append(s.manifest_entry(sample_file))
    (out_dir / "hl_samples.json").write_text(
        json.dumps(
            {
                "dataset_format": "steervla_hl_dataset_format",
                # Bumped to 2 when ego_hist / routing_command / original_* / route were added.
                "schema_version": HL_SCHEMA_VERSION,
                "action_supervision": False,
                "num_samples": len(manifest),
                "samples": manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest


# ── Debug annotation (GOOD/BAD + suggested subtasks side panel) ───────────────────────


def _wrap(text: str, width: int) -> list[str]:
    cleaned = " ".join(str(text).split())
    return textwrap.wrap(cleaned, width=width) if cleaned else []


def annotate_cast_relabel_frames(
    frames: list[np.ndarray],
    frame_chunk_indices: list[int],
    cast_json: dict[str, Any],
    *,
    dropped_chunk_indices: "set[int] | frozenset[int]" = frozenset(),
    drop_reason: str = "",
) -> list[np.ndarray]:
    """Double each frame's width and draw the active chunk's GOOD/BAD label + subtasks.

    Input ``frames`` already carry the original subtask text panel and waypoint/action
    overlays (drawn upstream in ``main_carla``); this only adds the credit-assignment
    side panel so the debug video shows everything the user asked for.

    The panel shows the chunk's ``original_subtask`` (what the policy actually said at that
    moment) directly above the suggested replacements, so the correction is legible as a
    before/after rather than as a list of suggestions with nothing to compare against.

    ``dropped_chunk_indices`` are chunks whose data is NOT used as high-level training data --
    either truncated out before credit assignment by a route divergence, or falling at/after the
    episode's failure cutoff. Their frames get a red banner naming ``drop_reason``, so a stretch
    of video that looks reviewed but is actually discarded cannot be mistaken for supervision.
    """
    import cv2  # type: ignore

    chunks_by_index = {int(c["chunk_index"]): c for c in cast_json.get("action_chunks", [])}
    out_frames: list[np.ndarray] = []
    font = cv2.FONT_HERSHEY_SIMPLEX

    for frame, chunk_idx in zip(frames, frame_chunk_indices):
        frame = np.asarray(frame, dtype=np.uint8)
        h, w = frame.shape[:2]
        canvas = np.zeros((h, w * 2, 3), dtype=np.uint8)
        canvas[:, :w, :] = frame

        chunk = chunks_by_index.get(int(chunk_idx))
        panel_x = w + 12
        wrap_width = max(24, w // 9)
        lines: list[tuple[str, tuple[int, int, int], int]] = []
        is_dropped = int(chunk_idx) in dropped_chunk_indices
        if is_dropped:
            # A dropped chunk is usually absent from ``action_chunks`` entirely (divergence
            # truncates it before credit assignment), so there is no label or rationale to draw --
            # the banner and the chunk id are the whole story for these frames.
            lines.append((f"Chunk {int(chunk_idx)}", (255, 200, 80), 2))
            lines.append(("NOT USED FOR TRAINING", (60, 60, 255), 2))
            for part in _wrap(drop_reason or "dropped after the episode went off the rails",
                              wrap_width):
                lines.append((part, (170, 170, 255), 1))
            if chunk is not None:
                lines.append(("", (0, 0, 0), 1))
        if chunk is not None and not is_dropped:
            label = chunk.get("label")
            t0 = float(chunk.get("video_time_start_sec", 0.0))
            t1 = float(chunk.get("video_time_end_sec", 0.0))
            lines.append((f"Chunk {chunk.get('chunk_index', '?')} ({t0:.1f}-{t1:.1f}s)", (255, 200, 80), 2))
            if label == "GOOD":
                lines.append(("GOOD", (0, 220, 0), 2))
            elif label == "BAD":
                source = str(chunk.get("credit_source") or "").strip()
                if source == "precursor":
                    # Distinct colour: this chunk looks fine on screen, it is blamed for what
                    # comes later, so a reviewer needs to see why it is flagged.
                    lines.append(("BAD (precursor)", (0, 140, 255), 2))
                else:
                    lines.append(("BAD", (0, 0, 255), 2))
            else:
                lines.append(("(no signal)", (160, 160, 160), 1))
            rationale = str(chunk.get("rationale", "")).strip()
            if rationale:
                lines.append(("Why:", (255, 255, 255), 1))
                for part in _wrap(rationale, wrap_width):
                    lines.append((part, (220, 220, 220), 1))
            original = str(chunk.get("original_subtask") or "").strip()
            if original:
                lines.append(("Current subtask (what the policy said):", (255, 255, 255), 1))
                for part in _wrap(original, wrap_width):
                    lines.append((part, (170, 170, 170), 1))
            subtasks = chunk.get("suggested_subtasks") or []
            if subtasks:
                lines.append(
                    ("Corrected to:" if original else "Suggested subtasks:", (255, 255, 255), 1)
                )
                for i, st in enumerate(subtasks):
                    for j, part in enumerate(_wrap(st, wrap_width)):
                        prefix = f"{i + 1}. " if j == 0 else "   "
                        lines.append((prefix + part, (120, 210, 255), 1))

        y = 24
        if is_dropped:
            # Full-width banner over the camera image too, not just the text column: at a glance,
            # scrubbing the video should show exactly which stretch was thrown away.
            cv2.rectangle(canvas, (0, 0), (w * 2, 30), (0, 0, 170), thickness=-1)
            banner = "DROPPED - NOT USED FOR TRAINING"
            if drop_reason:
                # Fit the reason to the composite's real width rather than a fixed cut, which
                # chopped it mid-word on the 2048px debug video. ~12px per char at scale 0.55.
                room = max(20, ((w * 2 - 24) // 12) - len(banner) - 4)
                shown = drop_reason if len(drop_reason) <= room else drop_reason[: room - 1] + "\u2026"
                banner += f"  ({shown})"
            cv2.putText(canvas, banner, (10, 21), font, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.rectangle(canvas, (0, 0), (w * 2 - 1, h - 1), (0, 0, 170), thickness=3)
            y = 30 + 24

        line_h = 17
        for line, color, weight in lines:
            if y + line_h > h - 6:
                break
            cv2.putText(canvas, line, (panel_x, y), font, 0.42, color, weight, cv2.LINE_AA)
            y += line_h
        out_frames.append(canvas)

    return out_frames


# ── Online session (wired into main_carla) ───────────────────────────────────────────


def _pad_frames_to_common_shape(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Zero-pad ``frames`` (H, W, 3) to the max height/width across them, content top-left.

    ``annotate_cast_relabel_frames`` sizes each frame's text column to the text it must fit, so one
    window's composites can disagree in height/width and ``np.stack`` then raises. Returns the input
    untouched when the frames already agree, so the common case costs one comparison and no copy.
    """
    shapes = {f.shape[:2] for f in frames}
    if len(shapes) <= 1:
        return frames
    max_h = max(f.shape[0] for f in frames)
    max_w = max(f.shape[1] for f in frames)
    out: list[np.ndarray] = []
    for f in frames:
        h, w = f.shape[:2]
        if (h, w) == (max_h, max_w):
            out.append(f)
            continue
        buf = np.zeros((max_h, max_w, f.shape[2]), dtype=f.dtype)
        buf[:h, :w] = f
        out.append(buf)
    return out


def _as_config_dict(cfg: Any) -> dict[str, Any]:
    if cfg is None:
        return {}
    if hasattr(cfg, "to_dict"):
        return dict(cfg.to_dict())
    return dict(cfg)


class OnlineCastRelabelSession:
    """Accumulate rollout frames/trajectory; run the CAST relabel pipeline per window.

    Windows are a fixed env-step count (``query_every_n_episode_steps``) rounded down to a
    whole number of action chunks. Consumption is artifacts + wandb only — nothing is
    written back to the replay buffer.
    """

    def __init__(
        self,
        cast_cfg: Any,
        *,
        save_dir: str | Path,
        action_chunk_steps: int = DEFAULT_ACTION_CHUNK_STEPS,
        video_frame_stride: int = 2,
        run_tag: str = "",
    ) -> None:
        self.cfg = _as_config_dict(cast_cfg)
        self.save_dir = Path(save_dir)
        self.artifact_dir = self.save_dir / "cast_relabel"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        self.provider = str(self.cfg.get("provider", "gemini"))
        self.gemini_model = str(self.cfg.get("gemini_model", "gemini-3.5-flash"))
        self.action_chunk_steps = int(self.cfg.get("action_chunk_steps", action_chunk_steps))
        self.video_fps = float(self.cfg.get("video_fps", 10.0))
        self.video_frame_stride = int(self.cfg.get("video_frame_stride", video_frame_stride))
        self.num_subtask_suggestions = int(
            self.cfg.get("num_subtask_suggestions", DEFAULT_NUM_SUBTASK_SUGGESTIONS)
        )
        self.save_artifacts = bool(self.cfg.get("save_artifacts", True))
        self.debug = bool(self.cfg.get("debug", False))
        # Debug relabel task: swap the credit/subtask prompt for build_debug_task_prompt, which
        # forces every chunk's suggested subtasks toward "remain stopped / slow down".
        self.debug_task = bool(self.cfg.get("debug_task", False))
        self.query_on_episode_end = bool(self.cfg.get("query_on_episode_end", True))
        # Minimum env steps a forced end-of-episode window must span to be worth reviewing.
        # Defaults to one action chunk — below that there is no chunk to assign credit to, and the
        # sub-second video is rejected by the VLM API anyway (see should_query).
        self.min_final_window_steps = int(
            self.cfg.get("min_final_window_steps", self.action_chunk_steps)
        )
        # High-level (VLM-backbone) dataset storage: persist BAD/relabeled chunks as SteerVLA
        # ``steervla_hl_dataset_format`` samples for a later VLM-backbone fine-tuning step.
        self.store_hl_dataset = bool(self.cfg.get("store_hl_dataset", True))
        # Also reinforce GOOD/unlabeled chunks by storing their original (uncorrected) subtask +
        # reasoning as HL samples. Set False to keep the dataset corrective (BAD chunks only).
        self.store_good_chunks = bool(self.cfg.get("store_good_chunks", True))
        # Stop supervising the high-level policy for the REST of an episode once the run has
        # seriously failed -- a counted collision, an outside-route event, or a crash_stuck
        # termination. After such a failure the ego is somewhere the policy will never legitimately
        # be (wedged against a wall, facing backwards, off the drivable surface), so those chunks
        # are far out of distribution and carry no signal about strategy; relabeling them mostly
        # teaches the CoT to narrate wreckage. Windows already stored before the cutoff are kept:
        # the lead-up to the failure is exactly the part worth learning from.
        # Split the window review into two VLM calls over the same video: Step 1 (scene +
        # traffic-flow state) as prose, then Step 2 (GOOD/BAD events) with that answer supplied as
        # established context. Costs one extra generate_content per window -- the video is uploaded
        # once regardless -- and makes the scene reasoning an artifact you can actually read.
        self.two_stage_review = bool(self.cfg.get("two_stage_review", False))
        # Write every prompt + raw response per window to ``vlm_calls.json`` and refresh the live
        # HTML review viewer. Independent of two_stage_review: useful on the single-call path too.
        self.save_vlm_calls = bool(self.cfg.get("save_vlm_calls", True))
        # One-shot guard so the viewer link is announced to W&B once, not once per window.
        self._viewer_logged = False
        self.review_viewer_path: Path | None = None
        self.hl_stop_after_failure = bool(self.cfg.get("hl_stop_after_failure", True))
        # Which failures trip the cutoff. Off by one -> that class no longer ends supervision.
        self.hl_stop_on_collision = bool(self.cfg.get("hl_stop_on_collision", True))
        self.hl_stop_on_off_route = bool(self.cfg.get("hl_stop_on_off_route", True))
        self.hl_action_dim = int(self.cfg.get("hl_action_dim", DEFAULT_HL_ACTION_DIM))
        # Per-step ego history stored with each HL sample (see DEFAULT_EGO_HISTORY_LEN).
        self.ego_history_len = max(1, int(self.cfg.get("ego_history_len", DEFAULT_EGO_HISTORY_LEN)))
        # Where the HL samples land. Default: ``<save_dir>/<hl_dataset_subdir>`` (per-run, which is
        # what the online update wants). ``hl_dataset_root`` overrides that with an absolute path
        # shared across runs — the offline-collection mode, where many routes/seeds accumulate into
        # one corpus that :mod:`vlas.cast_hl_to_rlds` later converts in a single pass. The run tag
        # keeps one level of nesting under the root so window dirs from different runs can't collide
        # *and* the layout the actor globs (``<hl_dataset_dir>/<window>/hl_samples.json``) is
        # preserved, so an offline-collection dir is still directly loadable by ``update_hl``.
        # Live policy version for a pooled run: bumped by main_carla's checkpoint watcher each time
        # this worker hot-reloads a newly published checkpoint, and stamped onto every HL sample so
        # the trainer can age out supervision from superseded policies. Stays 0 in a solo run.
        self.policy_version = 0
        self.run_tag = str(run_tag or Path(self.save_dir).name or "run")
        hl_root = str(self.cfg.get("hl_dataset_root", "") or "").strip()
        if hl_root:
            self.hl_dataset_dir = Path(hl_root).expanduser() / self.run_tag
        else:
            self.hl_dataset_dir = self.save_dir / str(
                self.cfg.get("hl_dataset_subdir", "cast_relabel_hl_dataset")
            )
        if self.store_hl_dataset:
            self.hl_dataset_dir.mkdir(parents=True, exist_ok=True)
        # Render the env-reward + route-progress graph per window and attach it to the review
        # call alongside the video. On by default: these are the two signals the per-timestamp
        # block keeps dense, and their shape over the window is the thing a table hides. Set
        # ``include_plots_in_prompt=False`` for a video-and-text-only review.
        self.include_plots_in_prompt = bool(self.cfg.get("include_plots_in_prompt", True))

        raw_seeds = self.cfg.get("seed_subtasks")
        self.seed_subtasks: tuple[str, ...] = (
            tuple(str(s) for s in raw_seeds) if raw_seeds else SEED_SUBTASKS
        )

        # One recorded video frame == ``video_frame_stride`` env steps, so a chunk spans
        # ``action_chunk_steps / video_frame_stride`` frames. This keeps the VLM's video
        # timestamps aligned with the chunk time ranges we hand it.
        frames_per_chunk = max(1, self.action_chunk_steps // max(1, self.video_frame_stride))
        self.chunk_duration_sec = frames_per_chunk / self.video_fps

        # Window size in env steps, rounded down to a whole number of chunks (>= 1 chunk).
        requested = int(self.cfg.get("query_every_n_episode_steps", 128))
        n_chunks = max(1, requested // self.action_chunk_steps)
        self.window_env_steps = n_chunks * self.action_chunk_steps

        self._coach = create_coach(self.provider, model=self.gemini_model)
        # Bounded cross-window memory of corrections already made, injected into both prompts so
        # later windows don't reverse earlier ones. ``correction_memory_words`` caps the whole
        # rendered block; 0 disables the cache entirely.
        memory_words = int(self.cfg.get("correction_memory_words", DEFAULT_MEMORY_WORDS))
        self._memory: CorrectionMemory | None = (
            CorrectionMemory(
                self.artifact_dir / "correction_memory.json",
                max_words=memory_words,
                coach=self._coach,
            )
            if memory_words > 0
            else None
        )
        if self.provider == "gemini":
            _key = os.environ.get("GEMINI_API_KEY", "")
            if not _key or _key.startswith("YOUR_"):
                print(
                    "[cast_relabel] WARNING: GEMINI_API_KEY is unset (or placeholder). Every "
                    "window review will fail (handled non-fatally) and produce no feedback. "
                    "Export GEMINI_API_KEY before launching.",
                    flush=True,
                )
        self.window_count = 0
        # ── Asynchronous window review ────────────────────────────────────────────────
        # The VLM review is pure hindsight: it reads a window that has already been driven,
        # so nothing in the rollout depends on its result before the next window. Running it
        # inline therefore costs wall-clock for no correctness benefit (measured ~30% of a
        # 4000-step run waiting on Gemini). With ``async_review`` the window is snapshotted and
        # handed to a single background worker; the rollout continues immediately.
        #
        # ONE worker on purpose: windows stay strictly ordered, and ``_memory`` /
        # ``_store_hl_samples`` / ``window_count`` are then only ever touched by one thread.
        # Sample discovery is already safe against a racing reader -- ``_scan_pool`` keys off
        # ``hl_samples.json``, which is written after every ``.npz``.
        # Feed the reviewer the un-annotated camera frame (1024x512) rather than the
        # composited one (1024x1292: waypoints + reward badge + a fixed-height text panel,
        # of which the bottom ~28% is blank). The overlays duplicate data the prompt already
        # carries as structured text while consuming the VLM's fixed pixel budget -- measured
        # on ep0002_win0023, the scene was only 40% of each frame, which puts a ~10x25 px
        # traffic-light head at ~12 px after the model's downsample.
        self.raw_video = bool(self.cfg.get("raw_video", True))
        self.async_review = bool(self.cfg.get("async_review", False))
        self._review_executor: concurrent.futures.ThreadPoolExecutor | None = (
            concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="cast-review")
            if self.async_review
            else None
        )
        self._review_future: concurrent.futures.Future | None = None
        # wandb is logged from the MAIN thread only. A worker finishing at wall-clock time T
        # holds a ``global_step`` from when its window closed, which is now behind the run's
        # current step; ``wandb.log(step=stale)`` is dropped. The worker parks its payload here
        # and ``drain_wandb()`` (called from the env loop) logs it at the current step instead.
        self._wandb_queue: list[dict[str, Any]] = []
        self._wandb_lock = threading.Lock()
        self.hl_sample_count = 0
        self.reset_episode()

    # ── recording ────────────────────────────────────────────────────────────────
    def reset_episode(self) -> None:
        self.episode_count = 0
        self.route_name = "?"
        self.route_id = "?"
        self.frames: list[np.ndarray] = []
        # Overlaid copies, same indices as ``frames``; used only for the W&B debug video.
        self.frames_annotated: list[np.ndarray] = []
        self.frame_subtasks: list[str] = []
        self.frame_episode_steps: list[int] = []
        self.trajectory_steps: list[dict[str, Any]] = []
        self._frames_cursor = 0
        self._traj_cursor = 0
        self._last_query_episode_step = 0
        self.route_command_plan: list[dict[str, Any]] = []
        # Raw model inputs captured at chunk-start steps (abs episode step -> dict), used to build
        # high-level dataset samples at window review time. Cleared per window.
        self._model_inputs: dict[int, dict[str, Any]] = {}
        # Rolling ``[speed, course]`` over the last ``ego_history_len`` env steps of THIS episode, so a
        # chunk-start sample can carry a real ego history rather than a tiled current pair.
        self._ego_hist: deque[tuple[float, float]] = deque(maxlen=self.ego_history_len)
        # Episode step at which a serious failure first occurred; HL samples at or after it are
        # dropped. ``None`` while the episode is still clean. See ``hl_stop_after_failure``.
        self._hl_cutoff_step: int | None = None
        # Why the cutoff fired ("collision" / "off_route" / "crash_stuck" / "route
        # divergence ..."), purely so the debug video's banner can say which one it was.
        self._hl_cutoff_reason: str = ""

    def begin_episode(
        self,
        *,
        episode_count: int,
        route_name: str,
        route_command_plan: list[dict[str, Any]] | None = None,
        route_id: str = "",
    ) -> None:
        self.episode_count = int(episode_count)
        # NOTE: ``route_name`` is what main_carla passes as scene context for the VLM prompt, and it
        # is the current *routing command* ("follow the road."), not the scenario name. ``route_id``
        # is the actual route (``--route``), used only for dataset provenance/grouping. Keeping them
        # separate leaves the VLM prompt byte-identical to previous runs.
        self.route_name = str(route_name)
        self.route_id = str(route_id or route_name)
        # Full ordered maneuver plan for the episode (constant), surfaced to the VLM coach as
        # the overall "task" context. May be empty if the route planner was unavailable.
        self.route_command_plan = list(route_command_plan) if route_command_plan else []

    def record_frame(
        self,
        frame: np.ndarray | None,
        *,
        subtask_text: str = "",
        episode_step: int = 0,
        annotated: np.ndarray | None = None,
    ) -> None:
        """Record one frame. ``frame`` is what the VLM reviews; ``annotated`` (optional) is the
        overlaid copy kept only for the W&B debug video.

        The two are separate on purpose: the reviewer should spend its fixed pixel budget on the
        scene, while a human scrubbing the debug video wants the waypoints/subtask/reward burned
        in. When ``annotated`` is omitted the same frame serves both.
        """
        if frame is None:
            return
        self.frames.append(np.asarray(frame, dtype=np.uint8))
        self.frames_annotated.append(
            np.asarray(annotated if annotated is not None else frame, dtype=np.uint8)
        )
        self.frame_subtasks.append(str(subtask_text or ""))
        self.frame_episode_steps.append(int(episode_step))

    def record_trajectory_step(self, step_record: dict[str, Any]) -> None:
        self.trajectory_steps.append(dict(step_record))
        self._maybe_trip_hl_cutoff(step_record)

    def _maybe_trip_hl_cutoff(self, step_record: dict[str, Any]) -> None:
        """Latch ``_hl_cutoff_step`` at the first serious failure of the episode.

        Serious means the environment counted an infraction, not that a sensor grazed something:
        ``collision_delta``/``outside_route_delta`` are per-step increments of the leaderboard's
        own counters, so they fire once per event rather than for every tick of sustained contact.
        ``crash_stuck`` is included because it means the ego collided and then never got moving
        again -- the canonical "rest of the episode is garbage" case.

        Latched, never cleared mid-episode: once the run is off the rails it does not come back,
        and re-arming would let a brief recovery re-open supervision on a wrecked trajectory.
        """
        if not (self.hl_stop_after_failure and self.store_hl_dataset):
            return
        if self._hl_cutoff_step is not None:
            return
        reason = None
        if self.hl_stop_on_collision and float(step_record.get("collision_delta", 0.0)) > 0.0:
            reason = "collision"
        elif self.hl_stop_on_off_route and float(step_record.get("outside_route_delta", 0.0)) > 0.0:
            reason = "off_route"
        elif step_record.get("termination_reason") == "crash_stuck":
            reason = "crash_stuck"
        if reason is None:
            return
        self._hl_cutoff_step = int(step_record.get("episode_step", 0))
        self._hl_cutoff_reason = reason
        print(
            f"[cast_relabel] HL supervision cutoff at episode step {self._hl_cutoff_step} "
            f"({reason}); the rest of this episode will be reviewed but not stored as "
            f"high-level training data.",
            flush=True,
        )

    def note_route_divergence(
        self, divergence: dict[str, Any] | None, traj_window: list[dict[str, Any]]
    ) -> None:
        """Latch the HL cutoff at a VLM-reported route divergence.

        ``divergence["timestamp_sec"]`` is a *video* time; ``_hl_cutoff_step`` is an *absolute*
        episode step. ``traj_window`` carries both on every step it recorded, so the mapping is
        read off the window rather than recomputed from the chunk table (whose episode steps are
        window-relative and 1-based -- a mismatch there would silently cut in the wrong place).

        The cutoff lands one step past the last frame shown strictly BEFORE the divergence, so the
        moment of the wrong turn is itself excluded. Steps with no recorded frame carry no video
        timestamp and cannot be placed on the video clock; erring one frame early is the safe
        direction, since the cost is a few dropped good samples rather than trained-on garbage.

        Same latching rule as ``_maybe_trip_hl_cutoff``: first cutoff of the episode wins, so a
        later, looser verdict can never re-open supervision on an already-condemned trajectory.
        """
        if not (self.hl_stop_after_failure and self.store_hl_dataset):
            return
        if not divergence or not divergence.get("diverged"):
            return
        ts = divergence.get("timestamp_sec")
        if ts is None:
            return
        last_before: int | None = None
        for step in traj_window:
            t = step.get("video_timestamp_sec")
            if t is None:
                continue
            if float(t) < float(ts):
                last_before = int(step.get("episode_step", 0))
            else:
                break
        if last_before is not None:
            cutoff = last_before + 1
        elif traj_window:
            # Diverged before the first recorded frame of this window: nothing here is usable.
            cutoff = int(traj_window[0].get("episode_step", 0))
        else:
            return
        if self._hl_cutoff_step is not None and self._hl_cutoff_step <= cutoff:
            return
        self._hl_cutoff_step = cutoff
        self._hl_cutoff_reason = (
            f"off route at t={float(ts):.1f}s: {divergence.get('reason', '')}".strip()
        )
        print(
            f"[cast_relabel] HL supervision cutoff at episode step {cutoff} (route divergence at "
            f"t={float(ts):.2f}s: {divergence.get('reason', '')}); the rest of this episode will "
            f"be reviewed but not stored as high-level training data.",
            flush=True,
        )

    def record_model_input(
        self,
        *,
        episode_step: int,
        image: np.ndarray | None,
        state: np.ndarray | None,
        current_speed: float = 0.0,
        prompt: str = "",
        subtask: str = "",
        reasoning: str = "",
        action_chunk: np.ndarray | None = None,
        routing_command: str = "",
        global_step: int = -1,
    ) -> None:
        """Stash the raw SteerVLA model input for the high-level dataset (chunk-start steps only).

        Called every env step; only the first step of each action chunk is retained (that's the
        one frame we need per chunk). ``image``/``state`` are the raw pre-step obs the action was
        taken from (``obs["image"]`` / ``obs["state"]``), not the annotated video frame. ``subtask``
        / ``reasoning`` are the **original** CoT targets the model produced for this chunk — used as
        the HL supervision targets when a GOOD/unlabeled chunk is reinforced as-is.

        The ego-history ring is pushed on **every** call (not just chunk starts) so the retained
        sample carries the ``ego_history_len`` env steps leading up to it; ``routing_command`` is the
        bare instruction behind ``prompt``'s "The current speed is X m/s. " prefix, kept verbatim for
        the RLDS conversion.
        """
        self._push_ego_history(state, current_speed)
        if not self.store_hl_dataset or image is None:
            return
        # Chunk starts are every ``action_chunk_steps`` env steps from the (1-based) episode start.
        if (int(episode_step) - 1) % max(1, self.action_chunk_steps) != 0:
            return
        self._model_inputs[int(episode_step)] = {
            "image": np.asarray(image, dtype=np.uint8),
            "state": None if state is None else np.asarray(state, dtype=np.float32).reshape(-1),
            "ego_hist": self._ego_history_array(),
            "current_speed": float(current_speed),
            "prompt": str(prompt or ""),
            "subtask": str(subtask or ""),
            "reasoning": str(reasoning or ""),
            "routing_command": str(routing_command or ""),
            "global_step": int(global_step),
            "action_chunk": None if action_chunk is None else np.asarray(action_chunk, dtype=np.float32),
            # Snapshot the live policy version here rather than at window flush: a pooled worker
            # hot-reloads mid-episode, so chunks in one window can come from different versions.
            "policy_version": int(self.policy_version),
        }

    def _push_ego_history(self, state: np.ndarray | None, current_speed: float) -> None:
        """Append this env step's ``[speed_mps, course_deg]`` to the rolling ego history.

        ``course_deg`` is the SimLingo per-frame heading delta, not absolute yaw. Note the pairs are
        one *env* step apart (0.05 s) where SimLingo's are one dataset frame apart (0.25 s); that
        only matters if the actor ever runs with ``include_ego_history=True``, since the active
        config consumes the last pair alone.
        """
        speed = float(current_speed)
        course = 0.0
        if state is not None:
            flat = np.asarray(state, dtype=np.float32).reshape(-1)
            if flat.size > EGO_STATE_IDX_SPEED:
                speed = float(flat[EGO_STATE_IDX_SPEED])
            if flat.size > EGO_STATE_IDX_YAW_RATE:
                course = float(flat[EGO_STATE_IDX_YAW_RATE]) * SIMLINGO_FRAME_DT
        self._ego_hist.append((speed, course))

    def _ego_history_array(self) -> np.ndarray:
        """``(ego_history_len, 2)`` float32, oldest first, left-padded with the oldest pair.

        Padding only happens in the first few steps of an episode, where fewer than
        ``ego_history_len`` steps exist yet.
        """
        hist = list(self._ego_hist)
        if not hist:
            return np.zeros((self.ego_history_len, 2), dtype=np.float32)
        while len(hist) < self.ego_history_len:
            hist.insert(0, hist[0])
        return np.asarray(hist[-self.ego_history_len:], dtype=np.float32)

    # ── querying ─────────────────────────────────────────────────────────────────
    def should_query(self, episode_step: int, *, force: bool = False) -> bool:
        if len(self.frames) <= self._frames_cursor:
            return False
        if force:
            if not self.query_on_episode_end:
                return False
            # An episode that ends just past a window boundary leaves a remainder of a few env
            # steps. Reviewing it uploads a sub-second video (1-4 frames), which Gemini rejects
            # with a 400 — every time, ~2-5% of all windows. Such a window spans less than one
            # action chunk, so it could not have produced a usable credit assignment anyway.
            # Require at least one whole chunk before spending the call.
            pending_steps = len(self.trajectory_steps) - self._traj_cursor
            if pending_steps < self.min_final_window_steps:
                print(
                    f"[cast_relabel] skipping end-of-episode window: only {pending_steps} env "
                    f"steps pending (< {self.min_final_window_steps} = one action chunk).",
                    flush=True,
                )
                self._frames_cursor = len(self.frames)
                self._traj_cursor = len(self.trajectory_steps)
                self._model_inputs = {}
                return False
            return True
        if self.window_env_steps <= 0 or episode_step <= 0:
            return False
        if episode_step % self.window_env_steps != 0:
            return False
        return episode_step != self._last_query_episode_step

    def _run_window_async(self, snap, episode_step, done_info, global_step) -> None:
        """Worker entry point: same review, on the snapshot, never fatal to the rollout."""
        try:
            self._run_window(
                episode_step=episode_step,
                done_info=done_info,
                final=False,
                global_step=global_step,
                snapshot=snap,
            )
        except Exception as exc:  # noqa: BLE001 - a VLM failure must not kill the route
            import traceback

            print(f"[cast_relabel] async window query failed (non-fatal): {exc}", flush=True)
            traceback.print_exc()

    def _snapshot_window(self) -> dict[str, Any] | None:
        """Slice the pending window and advance the cursors, on the MAIN thread.

        Advancing here (not in the worker) is what makes the async path correct: the rollout
        keeps appending to ``self.frames`` / ``self.trajectory_steps`` during the review, and
        those rows belong to the *next* window.
        """
        frames_window = list(self.frames[self._frames_cursor:])
        if not frames_window:
            return None
        snap = {
            "frames": frames_window,
            "frames_annotated": list(self.frames_annotated[self._frames_cursor:]),
            "subtasks": list(self.frame_subtasks[self._frames_cursor:]),
            "frame_steps": list(self.frame_episode_steps[self._frames_cursor:]),
            "traj": [dict(r) for r in self.trajectory_steps[self._traj_cursor:]],
            "step_offset": self._traj_cursor,
            "model_inputs": dict(self._model_inputs),
        }
        self.window_count += 1
        snap["window_index"] = self.window_count
        self._frames_cursor = len(self.frames)
        self._traj_cursor = len(self.trajectory_steps)
        self._model_inputs = {}
        return snap

    def _refresh_review_viewer(self, cast_dir: Path) -> None:
        """Rebuild the live HTML transcript viewer, and register its link with W&B once.

        Called after every window. Cheap by construction: the page links videos by relative path
        instead of embedding them (see coaches/cast_review_viewer.py), so cost is independent of
        how many windows have accumulated.
        """
        try:
            from coaches.cast_review_viewer import refresh as _refresh_viewer
        except ImportError:
            from impls.coaches.cast_review_viewer import refresh as _refresh_viewer

        out = _refresh_viewer(cast_dir, title=f"{self.route_name} · {Path(self.save_dir).name}")
        if self._viewer_logged:
            return
        self._viewer_logged = True
        self.review_viewer_path = out
        print(f"[cast_relabel] live review viewer -> {out}", flush=True)
        try:
            import wandb

            if wandb.run is not None:
                url = out.resolve().as_uri()
                link = (
                    f'<p style="font:14px system-ui">CAST relabel review transcripts '
                    f'(regenerated after every window, auto-refreshing):<br>'
                    f'<a href="{url}">{out}</a></p>'
                )
                self._emit_wandb({"cast_relabel/review_viewer": wandb.Html(link)}, None)
                # Also as a plain summary field, which survives in the run overview and is easier
                # to copy than a media panel.
                wandb.run.summary["cast_relabel_review_viewer"] = str(out.resolve())
        except Exception as exc:  # noqa: BLE001 - the link is a convenience, never fatal
            print(f"[cast_relabel] could not register viewer link with wandb: {exc}", flush=True)

    def _emit_wandb(self, payload: dict[str, Any], step: int | None) -> None:
        """Log now (sync) or park for the main thread to log (async)."""
        if not self.async_review:
            import wandb

            wandb.log(payload, step=step) if step is not None else wandb.log(payload)
            return
        with self._wandb_lock:
            self._wandb_queue.append(payload)

    def drain_wandb(self) -> int:
        """Log anything a review worker parked. Call from the env loop (main thread)."""
        if not self.async_review:
            return 0
        with self._wandb_lock:
            pending, self._wandb_queue = self._wandb_queue, []
        if not pending:
            return 0
        import wandb

        if wandb.run is not None:
            for payload in pending:
                try:
                    wandb.log(payload)  # no step= : attribute to completion time
                except Exception as exc:  # noqa: BLE001 - logging must never kill a run
                    print(f"[cast_relabel] wandb drain failed (non-fatal): {exc}", flush=True)
        return len(pending)

    def wait_for_reviews(self, timeout: float | None = None) -> None:
        """Block until the in-flight review finishes (episode end / shutdown)."""
        fut = self._review_future
        if fut is not None and not fut.done():
            try:
                fut.result(timeout=timeout)
            except Exception as exc:  # noqa: BLE001
                print(f"[cast_relabel] async review failed (non-fatal): {exc}", flush=True)
        self.drain_wandb()

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
        # A coach/VLM failure (bad model id, auth, network, quota, parse error) must never
        # tear down the CARLA route or the training run. Catch everything, log it, and
        # advance the cursors so the failed window is skipped rather than retried forever.
        # NOTE: only Exception is caught here; a watchdog timeout raises KeyboardInterrupt,
        # which is why main_carla pauses the leaderboard watchdogs around this call.
        if self.async_review and self._review_executor is not None and not force:
            # Keep at most one review in flight so windows stay ordered and the worker-owned
            # state (_memory, window_count, HL sample dir) is never touched concurrently. A
            # review normally finishes well inside one window, so this rarely blocks; when it
            # does, the wait is still strictly less than running it inline.
            if self._review_future is not None and not self._review_future.done():
                try:
                    self._review_future.result()
                except Exception as exc:  # noqa: BLE001
                    print(f"[cast_relabel] async review failed (non-fatal): {exc}", flush=True)
            snap = self._snapshot_window()
            self.drain_wandb()
            if snap is not None:
                self._review_future = self._review_executor.submit(
                    self._run_window_async, snap, episode_step, done_info, global_step
                )
            self._last_query_episode_step = episode_step
            return True
        try:
            self._run_window(
                episode_step=episode_step, done_info=done_info, final=force, global_step=global_step
            )
        except Exception as exc:  # noqa: BLE001 - deliberately non-fatal
            import traceback

            print(f"[cast_relabel] window query failed (non-fatal): {exc}", flush=True)
            traceback.print_exc()
            self._frames_cursor = len(self.frames)
            self._traj_cursor = len(self.trajectory_steps)
            self._model_inputs = {}
        self._last_query_episode_step = episode_step
        return True

    def _build_metadata(
        self,
        traj_window: list[dict[str, Any]],
        frames_window_subtasks: list[str],
        *,
        frame_episode_steps: list[int],
        step_offset: int,
        done_info: dict[str, Any] | None,
    ) -> dict[str, Any]:
        done_info = done_info or {}
        # ── Window-relative video clock ────────────────────────────────────────────────
        # The ``video_timestamp_sec`` main_carla records counts frames from the start of the RUN
        # (``episode_video_frame_index`` is never reset, not per window and not per episode), but
        # the video handed to the VLM covers only THIS window and starts at t=0. Left alone, the
        # per-timestamp block in the review prompt is keyed to a clock the VLM cannot see —
        # window 3 of a 150-step cadence describes seconds 15.0-22.4 of a 7.5 s video, and it only
        # gets worse as the run goes on. Re-derive the frame index and timestamp from the frames
        # this session actually recorded for the window, keyed by episode step.
        #
        # Deriving them from the recorded frames (rather than ``episode_step * stride``) also
        # absorbs the non-uniform cadence: main_carla samples a frame every ``video_frame_stride``
        # steps AND on every collision / traffic-violation step, so eventful windows carry extra
        # frames and a stride-based mapping mis-times everything after the first one.
        step_to_frame: dict[int, int] = {}
        for idx, ep_step in enumerate(frame_episode_steps or []):
            step_to_frame.setdefault(int(ep_step), int(idx))
        fps = float(self.video_fps) if float(self.video_fps) > 0.0 else 1.0
        steps_out: list[dict[str, Any]] = []
        for s in traj_window:
            rec = dict(s)
            # These strings go straight into the review prompt's per-timestamp block.
            for key in ("subtask", "reasoning"):
                if rec.get(key):
                    rec[key] = strip_cot_sentinels(rec[key])
            frame_idx = step_to_frame.get(int(rec.get("episode_step", -1)))
            if frame_idx is None:
                rec["video_frame_index"] = None
                rec["video_timestamp_sec"] = None
                rec["in_video"] = False
            else:
                rec["video_frame_index"] = frame_idx
                rec["video_timestamp_sec"] = round(frame_idx / fps, 3)
                rec["in_video"] = True
            steps_out.append(rec)
        traj_window = steps_out

        collision_events: list[dict[str, Any]] = []
        for s in traj_window:
            if s.get("collision") or s.get("collision_active"):
                collision_events.append(
                    {
                        "video_timestamp_sec": s.get("video_timestamp_sec"),
                        "episode_step": s.get("episode_step"),
                        "new_event": bool(s.get("collision")),
                    }
                )
        # Original subtask per (window-relative) chunk index: first subtask seen in the chunk.
        # Mapped through each frame's recorded episode step for the same reason as the clock
        # above — ``offset * stride`` drifts as soon as one off-cadence frame is captured.
        first_abs_step = int(traj_window[0].get("episode_step", 1)) if traj_window else 1
        frame_chunk_indices: list[int] = []
        chunk_original_subtask: dict[str, str] = {}
        for offset, ep_step in enumerate(frame_episode_steps or []):
            window_step = int(ep_step) - first_abs_step + 1
            chunk_index = max(0, (window_step - 1) // self.action_chunk_steps)
            frame_chunk_indices.append(chunk_index)
            subtask = strip_cot_sentinels(
                frames_window_subtasks[offset] if offset < len(frames_window_subtasks) else ""
            )
            key = str(chunk_index)
            if subtask and key not in chunk_original_subtask:
                chunk_original_subtask[key] = subtask
        # Route-completion context: how far along the route the ego got over this window, so
        # the VLM can penalize failing to make progress (e.g. stopping prematurely with a
        # clear gap ahead while route progress is well short of 100%).
        progresses = [
            float(s["route_progress_pct"])
            for s in traj_window
            if s.get("route_progress_pct") is not None
        ]
        route_progress_start_pct = round(progresses[0], 2) if progresses else None
        route_progress_end_pct = round(progresses[-1], 2) if progresses else None
        route_progress_delta_pct = (
            round(route_progress_end_pct - route_progress_start_pct, 2)
            if route_progress_start_pct is not None and route_progress_end_pct is not None
            else None
        )
        route_completed = route_progress_end_pct is not None and route_progress_end_pct >= 99.5
        # Absolute position along the plan, in metres. The routing-command plan keys each maneuver
        # by ``start_distance_m``, so percentages alone cannot be aligned with it; giving the VLM
        # "84 m of 210 m" lets it say which plan entry is in force and whether it was executed.
        distances = [
            float(s["route_distance_m"])
            for s in traj_window
            if s.get("route_distance_m") is not None
        ]
        route_distance_start_m = round(distances[0], 1) if distances else None
        route_distance_end_m = round(distances[-1], 1) if distances else None
        totals = [
            float(s["route_total_distance_m"])
            for s in traj_window
            if s.get("route_total_distance_m")
        ]
        route_total_distance_m = round(totals[-1], 1) if totals else None
        # The routing command in force over this window. ``current`` is the one active at the end
        # of the window (what the ego is being asked to do now); the sequence records every change
        # seen, with the video timestamp, so a divergence can be attributed to the exact command
        # that was disobeyed rather than to the window as a whole.
        commands_seen: list[dict[str, Any]] = []
        for s in traj_window:
            cmd = str(s.get("routing_command") or "").strip()
            if not cmd:
                continue
            if not commands_seen or commands_seen[-1]["command"] != cmd:
                commands_seen.append(
                    {
                        "command": cmd,
                        "episode_step": s.get("episode_step"),
                        "video_timestamp_sec": s.get("video_timestamp_sec"),
                        "route_distance_m": (
                            round(float(s["route_distance_m"]), 1)
                            if s.get("route_distance_m") is not None
                            else None
                        ),
                    }
                )
        current_routing_command = commands_seen[-1]["command"] if commands_seen else ""
        # Mean ego speed over the last few steps — a near-zero value with the route
        # unfinished is a strong hint the ego stopped prematurely.
        end_speeds = [
            float(s["ego_speed_mps"])
            for s in traj_window[-5:]
            if s.get("ego_speed_mps") is not None
        ]
        mean_end_speed_mps = round(sum(end_speeds) / len(end_speeds), 3) if end_speeds else None
        # Env reward over the window. The per-step values go into the review prompt's
        # per-timestamp block; these summaries let the credit pass (which sees no per-step data)
        # reason about the window as a whole.
        rewards = [
            float(s["reward_total"]) for s in traj_window if s.get("reward_total") is not None
        ]
        window_reward_total = round(sum(rewards), 4) if rewards else None
        window_reward_mean = round(sum(rewards) / len(rewards), 4) if rewards else None
        return {
            "episode": self.episode_count,
            "route": self.route_name,
            "route_command_plan": self.route_command_plan,
            "window_index": self.window_count,
            "episode_steps": len(traj_window),
            "action_chunk_steps": self.action_chunk_steps,
            "action_chunk_duration_sec": self.chunk_duration_sec,
            "video_fps": self.video_fps,
            "video_frame_stride": self.video_frame_stride,
            "step_offset": step_offset,
            "success": done_info.get("success"),
            "termination_reason": done_info.get("termination_reason"),
            "route_progress_start_pct": route_progress_start_pct,
            "route_progress_end_pct": route_progress_end_pct,
            "route_progress_delta_pct": route_progress_delta_pct,
            "route_completed": route_completed,
            "route_distance_start_m": route_distance_start_m,
            "route_distance_end_m": route_distance_end_m,
            "route_total_distance_m": route_total_distance_m,
            "current_routing_command": current_routing_command,
            "routing_commands_in_window": commands_seen,
            "mean_end_speed_mps": mean_end_speed_mps,
            "window_reward_total": window_reward_total,
            "window_reward_mean": window_reward_mean,
            "collision_events": collision_events,
            # Rendered cross-window correction memory, picked up by BOTH prompt builders straight
            # off the metadata (so no signature threading) and recorded in the window artifact, so
            # every window says exactly which memory was in play when it was reviewed.
            "correction_memory": self._memory.render() if self._memory else "",
            "chunk_original_subtask": chunk_original_subtask,
            # Window-relative chunk index of each recorded frame, so the debug-video annotation
            # (and any offline viewer) can line frames up with chunks without re-deriving the
            # non-uniform frame cadence.
            "frame_chunk_indices": frame_chunk_indices,
            "steps": traj_window,
        }

    def _run_window(
        self,
        *,
        episode_step: int,
        done_info: dict[str, Any] | None,
        final: bool,
        global_step: int | None,
        snapshot: dict[str, Any] | None = None,
    ) -> None:
        """Review one window. ``snapshot`` (async path) supplies the already-sliced window and
        its ``window_index``; the cursors were advanced by ``_snapshot_window`` at submit time,
        so re-deriving them here would swallow everything driven during the review."""
        if snapshot is None:
            frames_window = self.frames[self._frames_cursor:]
            subtasks_window = self.frame_subtasks[self._frames_cursor:]
            frame_steps_window = self.frame_episode_steps[self._frames_cursor:]
            traj_window = self.trajectory_steps[self._traj_cursor:]
            annotated_window = self.frames_annotated[self._frames_cursor:]
            step_offset = self._traj_cursor
            self.window_count += 1
        else:
            frames_window = snapshot["frames"]
            subtasks_window = snapshot["subtasks"]
            frame_steps_window = snapshot["frame_steps"]
            traj_window = snapshot["traj"]
            annotated_window = snapshot.get("frames_annotated") or snapshot["frames"]
            step_offset = snapshot["step_offset"]
            self.window_count = snapshot["window_index"]

        tag = f"ep{self.episode_count:04d}_win{self.window_count:04d}{'_final' if final else ''}"
        work_dir = self.artifact_dir / tag
        work_dir.mkdir(parents=True, exist_ok=True)

        video_path = work_dir / "rollout.mp4"
        write_frames_to_mp4(frames_window, video_path, fps=self.video_fps)
        metadata = self._build_metadata(
            traj_window,
            subtasks_window,
            frame_episode_steps=frame_steps_window,
            step_offset=step_offset,
            done_info=done_info,
        )
        (work_dir / "trajectory.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        print(
            f"[cast_relabel] window {self.window_count}: reviewing {len(frames_window)} frames "
            f"({len(traj_window)} env steps, offset={step_offset}) -> {work_dir}",
            flush=True,
        )

        # Attach the two dense signals (env reward + route progress) to the review call as an
        # image. The per-timestamp block thins speed/controls but keeps these two on every row;
        # the plot gives the coach their shape at a glance, on the same clock as the video.
        # Non-fatal: a missing matplotlib or a bad window must not cost us the review.
        plot_paths: list[Path] | None = None
        if self.include_plots_in_prompt:
            try:
                from coaches.trajectory_plots import generate_reward_progress_plot

                plot_paths = [
                    generate_reward_progress_plot(
                        metadata,
                        work_dir / "plots" / "reward_progress.png",
                        chunk_steps=self.action_chunk_steps,
                    )
                ]
            except Exception as exc:  # noqa: BLE001 - the plot is an extra, not a prerequisite
                print(f"[cast_relabel] reward/progress plot failed (non-fatal): {exc}", flush=True)
                plot_paths = None

        vlm_calls: dict[str, Any] = {}
        events, cast_json = generate_cast_relabel(
            self._coach,
            video_path,
            metadata,
            steps_per_chunk=self.action_chunk_steps,
            chunk_duration_sec=self.chunk_duration_sec,
            seed_subtasks=self.seed_subtasks,
            num_suggestions=self.num_subtask_suggestions,
            plot_paths=plot_paths,
            include_plots_in_prompt=self.include_plots_in_prompt,
            debug_task=self.debug_task,
            two_stage=self.two_stage_review,
            out_calls=vlm_calls,
        )
        (work_dir / "cast_relabel.json").write_text(json.dumps(cast_json, indent=2), encoding="utf-8")
        if self.save_vlm_calls:
            # The transcript is the point of the review viewer: prompts and raw responses are built
            # on the fly and otherwise never touch disk. Non-fatal -- a failure here must not cost
            # the window's HL samples.
            try:
                vlm_calls["route"] = self.route_name
                vlm_calls["episode"] = self.episode_count
                vlm_calls["window_index"] = self.window_count
                vlm_calls["video"] = video_path.name if hasattr(video_path, "name") else str(video_path)
                (work_dir / "vlm_calls.json").write_text(
                    json.dumps(vlm_calls, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                self._refresh_review_viewer(work_dir.parent)
            except Exception as exc:  # noqa: BLE001 - diagnostic artifact, never fatal
                print(f"[cast_relabel] vlm_calls/viewer write failed (non-fatal): {exc}", flush=True)

        # Fold this window's corrections in *after* the prompts were built, so a window is never
        # shown its own outcome — only what came before it.
        if self._memory is not None:
            try:
                self._memory.observe_window(
                    cast_json, window_index=self.window_count, route=self.route_id
                )
            except Exception as exc:  # noqa: BLE001 - the memory is an aid, not a prerequisite
                print(f"[cast_relabel] correction memory update failed (non-fatal): {exc}", flush=True)

        # Route divergence latches the episode's HL cutoff. generate_cast_relabel has already
        # kept the off-route chunks out of credit assignment; this additionally stops every LATER
        # window of the same episode from contributing, since once the ego is off-route it stays
        # off-route -- the same "never comes back" reasoning as the collision cutoff.
        # NOTE: metadata["steps"], not traj_window. _build_metadata re-derives
        # ``video_timestamp_sec`` onto COPIES of each step (window-relative, matching the video the
        # VLM actually watched) and then rebinds only its own local name -- the caller's
        # traj_window still carries main_carla's RUN-relative timestamps, which are never reset and
        # so are all far larger than any window-relative divergence time. Passing that list made
        # the lookup find no frame before the divergence, fall back to "cut at the window's first
        # step", and silently discard the whole episode's supervision instead of just the
        # post-divergence part. Observed live on 2026-09-03 (cutoff printed as step 1).
        self.note_route_divergence(
            vlm_calls.get("route_divergence"), metadata.get("steps") or traj_window
        )

        # Persist BAD/relabeled chunks as high-level (VLM-backbone) SteerVLA samples.
        n_hl = self._store_hl_samples(cast_json, metadata, traj_window, tag)

        if self.debug:
            # Guarded like the correction-memory update above: the debug video is an aid, and an
            # exception here used to propagate out of _run_window and skip BOTH _log_scalars and
            # the cursor advance below. With the cursors left un-advanced the next window re-reviews
            # every frame from the start, so windows grow without bound and re-emit duplicate HL
            # samples -- which silently inflates the pool and the sample-reuse telemetry.
            try:
                _frame_chunks = metadata.get("frame_chunk_indices") or []
                _dropped, _drop_reason = self._debug_dropped_chunks(
                    cast_json, traj_window, _frame_chunks, vlm_calls.get("route_divergence")
                )
                self._log_debug_video(
                    frames_window,
                    cast_json,
                    frame_chunk_indices=_frame_chunks,
                    global_step=global_step,
                    dropped_chunk_indices=_dropped,
                    drop_reason=_drop_reason,
                )
            except Exception as exc:  # noqa: BLE001 - the debug video is never a prerequisite.
                print(f"[cast_relabel] debug video failed (non-fatal): {exc}", flush=True)
        self._log_scalars(events, cast_json, n_hl_samples=n_hl, global_step=global_step)

        # Advance cursors so the next window starts fresh. On the async path this already
        # happened at submit time -- redoing it here would discard every frame the rollout
        # collected while this review was in flight.
        if snapshot is None:
            self._frames_cursor = len(self.frames)
            self._traj_cursor = len(self.trajectory_steps)
            self._model_inputs = {}

        if self.save_artifacts:
            print(f"[cast_relabel] saved artifacts under {work_dir}", flush=True)

    def _debug_dropped_chunks(
        self,
        cast_json: dict[str, Any],
        traj_window: list[dict[str, Any]],
        frame_chunk_indices: list[int],
        divergence: dict[str, Any] | None,
    ) -> tuple[set[int], str]:
        """Which chunk indices in this window contribute NO high-level training data, and why.

        Two independent mechanisms put a chunk here, and the debug video should not care which:

        * **Route divergence** truncated the chunk out before credit assignment, so it is absent
          from ``cast_json["action_chunks"]`` entirely -- it shows up as a frame whose chunk index
          has no entry.
        * **The episode's failure cutoff** (collision / off-route / crash_stuck / divergence) sits
          at an absolute episode step; a chunk starting at or after it is reviewed but never
          stored. ``episode_step_start`` is window-relative and 1-based, so it is mapped through
          ``traj_window`` rather than compared directly -- the same trap as ``note_route_divergence``.

        Returns ``(indices, reason)``; an empty set when the window is entirely clean.
        """
        diverged = bool((divergence or {}).get("diverged"))
        cutoff = self._hl_cutoff_step
        if not diverged and cutoff is None:
            return set(), ""

        kept = {int(c.get("chunk_index", -1)) for c in cast_json.get("action_chunks", [])}
        dropped: set[int] = set()
        if diverged:
            # Frames whose chunk was truncated away before credit assignment.
            dropped |= {int(i) for i in frame_chunk_indices if int(i) not in kept}
        if cutoff is not None:
            for c in cast_json.get("action_chunks", []):
                start = int(c.get("episode_step_start", 0)) - 1
                if 0 <= start < len(traj_window):
                    if int(traj_window[start].get("episode_step", 0)) >= cutoff:
                        dropped.add(int(c.get("chunk_index", -1)))
        reason = self._hl_cutoff_reason or (
            f"off route at t={float((divergence or {}).get('timestamp_sec') or 0.0):.1f}s"
            if diverged else ""
        )
        return dropped, reason

    def _store_hl_samples(
        self,
        cast_json: dict[str, Any],
        metadata: dict[str, Any],
        traj_window: list[dict[str, Any]],
        tag: str,
    ) -> int:
        """Build + persist high-level dataset samples for this window; return the count stored."""
        if not self.store_hl_dataset:
            return 0
        try:
            chunk_specs = build_action_chunk_specs(
                metadata,
                steps_per_chunk=self.action_chunk_steps,
                chunk_duration_sec=self.chunk_duration_sec,
            )
            hl_samples = build_hl_samples_from_window(
                cast_json,
                chunk_specs,
                traj_window,
                self._model_inputs,
                action_chunk_steps=self.action_chunk_steps,
                action_dim=self.hl_action_dim,
                episode=self.episode_count,
                window_index=self.window_count,
                store_good_chunks=self.store_good_chunks,
                relabel_all=self.debug_task,
                route=self.route_id,
                ego_history_len=self.ego_history_len,
            )
            if self._hl_cutoff_step is not None:
                cutoff = self._hl_cutoff_step
                kept = [s for s in hl_samples if int(s.episode_step) < cutoff]
                if len(kept) != len(hl_samples):
                    print(
                        f"[cast_relabel] dropped {len(hl_samples) - len(kept)} of "
                        f"{len(hl_samples)} high-level samples at/after the failure cutoff "
                        f"(episode step {cutoff}).",
                        flush=True,
                    )
                hl_samples = kept
            if not hl_samples:
                return 0
            hl_dir = self.hl_dataset_dir / tag
            # Build under a ``.tmp-`` name and rename into place. In a pooled run the trainer globs
            # this tree continuously, and a window is many MB of .npz written over several seconds --
            # the rename makes it appear atomically, complete or not at all. See impls/cast_pool.py.
            try:
                from cast_pool import commit_staged_dir, staging_dir_for
            except ImportError:  # imported as ``impls.coaches.cast_relabel`` without impls/ on path
                from impls.cast_pool import commit_staged_dir, staging_dir_for

            staged = staging_dir_for(hl_dir)
            if staged.exists():
                import shutil

                shutil.rmtree(staged, ignore_errors=True)
            write_hl_samples(hl_samples, staged)
            # May land at ``<tag>__r2`` if a crash-restart regenerated a tag this run_tag already
            # used -- commit_staged_dir never overwrites a complete window. Log/index the real name.
            written_dir = commit_staged_dir(staged, hl_dir)
            self.hl_sample_count += len(hl_samples)
            self._append_window_index(written_dir.name, hl_samples)
            print(
                f"[cast_relabel] wrote {len(hl_samples)} high-level samples "
                f"(total {self.hl_sample_count}) -> {written_dir}",
                flush=True,
            )
            return len(hl_samples)
        except Exception as exc:  # noqa: BLE001 - HL storage must never break the run.
            import traceback

            print(f"[cast_relabel] HL sample storage failed (non-fatal): {exc}", flush=True)
            traceback.print_exc()
            return 0

    def _append_window_index(self, tag: str, hl_samples: list[HLSample]) -> None:
        """Append one line per stored window to ``<hl_dataset_dir>/windows.jsonl``.

        A cheap running index of the corpus (route / episode / window / label counts) so an
        offline-collection run can be inspected — and its conversion planned — without walking
        every per-window ``hl_samples.json``. Non-fatal: a failure here loses only the index.
        """
        try:
            n_bad = sum(1 for s in hl_samples if (s.label or "") == "BAD")
            n_precursor = sum(1 for s in hl_samples if s.credit_source == "precursor")
            line = {
                "tag": tag,
                "run_tag": self.run_tag,
                "route": self.route_id,
                "episode": int(self.episode_count),
                "window_index": int(self.window_count),
                "num_samples": len(hl_samples),
                "num_bad": int(n_bad),
                "num_precursor": int(n_precursor),
                "num_good_or_unlabeled": len(hl_samples) - int(n_bad),
            }
            with (self.hl_dataset_dir / "windows.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(line) + "\n")
        except Exception as exc:  # noqa: BLE001 - index is best-effort.
            print(f"[cast_relabel] window index append failed (non-fatal): {exc}", flush=True)

    # ── logging ──────────────────────────────────────────────────────────────────
    def _log_debug_video(
        self,
        frames_window: list[np.ndarray],
        cast_json: dict[str, Any],
        *,
        frame_chunk_indices: list[int],
        global_step: int | None,
        dropped_chunk_indices: "set[int] | None" = None,
        drop_reason: str = "",
    ) -> None:
        try:
            import wandb  # type: ignore
        except ImportError:
            return
        if wandb.run is None or not frames_window:
            return
        # Window-relative chunk index per recorded frame, derived in ``_build_metadata`` from each
        # frame's episode step (a stride-based estimate drifts on off-cadence frames).
        if len(frame_chunk_indices) < len(frames_window):
            frame_chunk_indices = list(frame_chunk_indices) + [
                (offset * self.video_frame_stride) // self.action_chunk_steps
                for offset in range(len(frame_chunk_indices), len(frames_window))
            ]
        annotated = annotate_cast_relabel_frames(
            frames_window,
            frame_chunk_indices,
            cast_json,
            dropped_chunk_indices=frozenset(dropped_chunk_indices or ()),
            drop_reason=drop_reason,
        )
        if not annotated:
            return
        # NOT shrunk. ``annotate_cast_relabel_frames`` composites the camera frame beside an
        # equal-width text column, so scaling the composite down shrinks the GOOD/BAD rationale and
        # subtask text with it and the panel stops being readable -- which is the only reason this
        # video exists. Camera frames therefore stay at the native 1024x512 and the composite is
        # ~2048px wide. Plain camera-only video (episode clips, local .mp4 dumps) still gets
        # downscaled in main_carla, where there is no text to lose.
        # ``annotate_cast_relabel_frames`` sizes each frame's text column to the rationale it has
        # to fit, so composites in one window can differ in height (and, for a wrapped subtask, in
        # width). np.stack requires exact agreement, so pad each frame out to the window maximum
        # with black rather than dropping the odd-sized ones -- content stays top-left aligned and
        # every frame survives.
        annotated = _pad_frames_to_common_shape(annotated)
        video = np.stack(annotated, axis=0)  # (T, H, W, 3)
        video = np.transpose(video, (0, 3, 1, 2))  # (T, C, H, W) for W&B
        self._emit_wandb(
            {"cast_relabel/debug_video": wandb.Video(video, fps=int(self.video_fps), format="mp4")},
            step=global_step,
        )

    def _log_scalars(
        self,
        events: list[CoachEvent],
        cast_json: dict[str, Any],
        *,
        n_hl_samples: int = 0,
        global_step: int | None,
    ) -> None:
        try:
            import wandb  # type: ignore
        except ImportError:
            return
        if wandb.run is None:
            return
        chunks = cast_json.get("action_chunks", [])
        n_good = sum(1 for c in chunks if c.get("label") == "GOOD")
        n_bad = sum(1 for c in chunks if c.get("label") == "BAD")
        n_bad_direct = sum(
            1 for c in chunks if c.get("label") == "BAD" and c.get("credit_source") != "precursor"
        )
        n_bad_precursor = sum(
            1 for c in chunks if c.get("label") == "BAD" and c.get("credit_source") == "precursor"
        )
        n_relabeled = sum(1 for c in chunks if c.get("suggested_subtasks"))
        log: dict[str, Any] = {
            "cast_relabel/window_index": self.window_count,
            "cast_relabel/n_events": len(events),
            "cast_relabel/n_chunks": len(chunks),
            "cast_relabel/n_good_chunks": n_good,
            "cast_relabel/n_bad_chunks": n_bad,
            "cast_relabel/n_bad_direct_chunks": n_bad_direct,
            "cast_relabel/n_bad_precursor_chunks": n_bad_precursor,
            "cast_relabel/n_relabeled_chunks": n_relabeled,
            "cast_relabel/n_hl_samples_window": int(n_hl_samples),
            "cast_relabel/hl_samples_total": int(self.hl_sample_count),
        }
        # Cross-window correction memory (coaches/correction_memory.py). This block is injected into
        # BOTH the window-review and credit prompts, so it silently steers every later window -- but
        # until now it only existed in correction_memory.json on disk. Logged here as a Table (one
        # row per window, so the whole history is scrubbable in the W&B UI) plus a word count for
        # charting, since the block is pruned oldest-note-first once it exceeds
        # ``correction_memory_words`` and it is useful to see when that pruning kicks in.
        if self._memory is not None:
            try:
                block = self._memory.render() or ""
                log["cast_relabel/correction_memory_words"] = len(block.split())
                log["cast_relabel/correction_memory_chars"] = len(block)
                mem_tbl = wandb.Table(columns=["window_index", "episode", "n_words", "memory_block"])
                mem_tbl.add_data(
                    int(self.window_count), int(self.episode_count), len(block.split()), block
                )
                log["cast_relabel/correction_memory"] = mem_tbl
            except Exception as exc:  # noqa: BLE001 - logging must never break the run.
                print(f"[cast_relabel] correction-memory logging failed (non-fatal): {exc}", flush=True)
        if self.debug and chunks:
            tbl = wandb.Table(
                columns=[
                    "chunk_index",
                    "label",
                    "credit_source",
                    "rationale",
                    "suggested_subtasks",
                    "suggested_reasoning",
                ]
            )
            for c in chunks:
                tbl.add_data(
                    c.get("chunk_index"),
                    c.get("label"),
                    c.get("credit_source"),
                    c.get("rationale"),
                    " | ".join(c.get("suggested_subtasks") or []),
                    c.get("suggested_reasoning"),
                )
            log["cast_relabel/chunks"] = tbl
        self._emit_wandb(log, global_step)
