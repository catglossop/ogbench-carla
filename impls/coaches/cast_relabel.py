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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from coaches.action_chunk_feedback import (
    ActionChunkSpec,
    DEFAULT_ACTION_CHUNK_STEPS,
    DEFAULT_CHUNK_DURATION_SEC,
    build_action_chunk_specs,
)
from coaches.online_vlm_coach import write_frames_to_mp4
from coaches.vlm_feedback import CoachEvent, create_coach

# Default number of subtask suggestions produced per chunk that needs improvement.
DEFAULT_NUM_SUBTASK_SUGGESTIONS = 3

# Default env action dim for the stored high-level (VLM-backbone) samples. The action head
# is masked out for HL samples (``action_loss_mask`` all-False), so this only fixes the shape
# of the stored (unsupervised) action chunk; it should match ``steervla.action_dim``.
DEFAULT_HL_ACTION_DIM = 4

# Seed phrases for the open-vocabulary subtask generation in step 4. The VLM is told it
# MAY reuse these verbatim or propose new phrases in the same style. Provided by the user;
# edit here (or override via ``cast_relabel.seed_subtasks`` in the agent config) to retune.
SEED_SUBTASKS: tuple[str, ...] = (
    "The vehicle cautiously follows the route, maintaining steady lane keeping behind the front car.",
    "The vehicle smoothly adjusts course to the right while cautiously maintaining speed behind the police car.",
    "The vehicle maintains a steady course, following the black car ahead at a normal pace.",
    "The vehicle remains stopped normally at the red traffic light.",
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
    "The vehicle follows the route and maintains reduced speed with a steady course.",
    "The vehicle cautiously follows the route and remains stopped behind the black car to the front right.",
    "The vehicle remains stopped behind the front black car due to a red traffic light, normally maintaining its position.",
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
)

SEED_REASONING: tuple[str, ...] = (
    "Follow the route.",
    "Follow the route. Remain stopped to stay behind the maroon car that is to the front that is stopped because of a red traffic light.",
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
    "Follow the route. Remain stopped to stay behind the maroon car that is to the front in 4.7 meters.",
    "Turn right. Remain stopped due to the red traffic light in 1.5 meters.",
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
        if label_raw is not None and str(label_raw).strip():
            label_val = str(label_raw).strip().upper()
            if label_val not in ("GOOD", "BAD"):
                raise ValueError(f"chunk label must be GOOD or BAD, got {label_raw!r}.")
            label = label_val
        subtasks_raw = raw.get("suggested_subtasks", []) or []
        if not isinstance(subtasks_raw, list):
            raise ValueError("suggested_subtasks must be a list of strings.")
        suggested = tuple(str(s).strip() for s in subtasks_raw if str(s).strip())
        source_raw = raw.get("credit_source")
        credit_source = str(source_raw).strip().lower() if source_raw is not None else ""
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
    """
    step_rewards: dict[int, float] = {}
    for s in metadata.get("steps", []) or []:
        if not isinstance(s, dict) or s.get("reward_total") is None or s.get("episode_step") is None:
            continue
        step_rewards[int(s["episode_step"])] = float(s["reward_total"])

    payload: list[dict[str, Any]] = []
    for spec in chunk_specs:
        entry: dict[str, Any] = {
            "chunk_index": spec.chunk_index,
            "episode_step_start": spec.episode_step_start,
            "episode_step_end": spec.episode_step_end,
            "video_time_start_sec": round(spec.video_time_start_sec, 3),
            "video_time_end_sec": round(spec.video_time_end_sec, 3),
        }
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
        {seed_block}

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

    return textwrap.dedent(
        f"""
        You previously reviewed a short driving rollout window and flagged GOOD and BAD
        moments (below). Now assign that credit to the specific action chunks the policy
        executed, and for each chunk propose subtasks that would improve the behavior.

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

        Action chunks in this window. ``reward_total_sum`` / ``reward_total_mean`` /
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
        
        **IMPORTANT:** Also try to smooth the subtasks relative to the adjacent subtasks in the chunk. If there is rapid changes of the subtask, you can suggest a subtask that smooths the transition.

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
) -> tuple[list[CoachEvent], dict[str, Any]]:
    """Full pipeline for one window: review -> credit assignment -> subtask suggestion.

    Returns ``(events, cast_relabel_json)``.
    """
    chunk_specs = build_action_chunk_specs(
        metadata,
        steps_per_chunk=steps_per_chunk,
        chunk_duration_sec=chunk_duration_sec,
    )
    # Step 2: VLM watches the window and flags what went well (GOOD) / poorly (BAD).
    events = coach.analyze(
        video_path,
        metadata,
        plot_paths=plot_paths,
        include_plots_in_prompt=include_plots_in_prompt,
    )
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
    response_text = coach.complete_text(prompt)
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

        samples.append(
            HLSample(
                image=np.asarray(model_input["image"], dtype=np.uint8),
                state=np.asarray(model_input.get("state"), dtype=np.float32).reshape(-1),
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
            )
        )
    return samples


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

        <out_dir>/sample_0000.npz   # image, state, current_speed, actions, action_loss_mask
        <out_dir>/hl_samples.json   # dataset_format + per-sample text targets + provenance
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for i, s in enumerate(samples):
        sample_file = f"sample_{i:04d}.npz"
        np.savez_compressed(
            out_dir / sample_file,
            image=s.image,
            state=s.state,
            current_speed=np.float32(s.current_speed),
            actions=s.actions,
            action_loss_mask=s.action_loss_mask,
        )
        manifest.append(s.manifest_entry(sample_file))
    (out_dir / "hl_samples.json").write_text(
        json.dumps(
            {
                "dataset_format": "steervla_hl_dataset_format",
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
) -> list[np.ndarray]:
    """Double each frame's width and draw the active chunk's GOOD/BAD label + subtasks.

    Input ``frames`` already carry the original subtask text panel and waypoint/action
    overlays (drawn upstream in ``main_carla``); this only adds the credit-assignment
    side panel so the debug video shows everything the user asked for.
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
        if chunk is not None:
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
            subtasks = chunk.get("suggested_subtasks") or []
            if subtasks:
                lines.append(("Suggested subtasks:", (255, 255, 255), 1))
                for i, st in enumerate(subtasks):
                    for j, part in enumerate(_wrap(st, wrap_width)):
                        prefix = f"{i + 1}. " if j == 0 else "   "
                        lines.append((prefix + part, (120, 210, 255), 1))

        y = 24
        line_h = 17
        for line, color, weight in lines:
            if y + line_h > h - 6:
                break
            cv2.putText(canvas, line, (panel_x, y), font, 0.42, color, weight, cv2.LINE_AA)
            y += line_h
        out_frames.append(canvas)

    return out_frames


# ── Online session (wired into main_carla) ───────────────────────────────────────────


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
        # High-level (VLM-backbone) dataset storage: persist BAD/relabeled chunks as SteerVLA
        # ``steervla_hl_dataset_format`` samples for a later VLM-backbone fine-tuning step.
        self.store_hl_dataset = bool(self.cfg.get("store_hl_dataset", True))
        # Also reinforce GOOD/unlabeled chunks by storing their original (uncorrected) subtask +
        # reasoning as HL samples. Set False to keep the dataset corrective (BAD chunks only).
        self.store_good_chunks = bool(self.cfg.get("store_good_chunks", True))
        self.hl_action_dim = int(self.cfg.get("hl_action_dim", DEFAULT_HL_ACTION_DIM))
        self.hl_dataset_dir = self.save_dir / str(self.cfg.get("hl_dataset_subdir", "cast_relabel_hl_dataset"))
        if self.store_hl_dataset:
            self.hl_dataset_dir.mkdir(parents=True, exist_ok=True)
        # When set, render the trajectory graphs (speed / throttle / steer / brake / route
        # progress vs video time) per window and attach them to the VLM coach prompt.
        self.include_plots_in_prompt = bool(self.cfg.get("include_plots_in_prompt", False))

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
        self.hl_sample_count = 0
        self.reset_episode()

    # ── recording ────────────────────────────────────────────────────────────────
    def reset_episode(self) -> None:
        self.episode_count = 0
        self.route_name = "?"
        self.frames: list[np.ndarray] = []
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

    def begin_episode(
        self,
        *,
        episode_count: int,
        route_name: str,
        route_command_plan: list[dict[str, Any]] | None = None,
    ) -> None:
        self.episode_count = int(episode_count)
        self.route_name = str(route_name)
        # Full ordered maneuver plan for the episode (constant), surfaced to the VLM coach as
        # the overall "task" context. May be empty if the route planner was unavailable.
        self.route_command_plan = list(route_command_plan) if route_command_plan else []

    def record_frame(self, frame: np.ndarray | None, *, subtask_text: str = "", episode_step: int = 0) -> None:
        if frame is None:
            return
        self.frames.append(np.asarray(frame, dtype=np.uint8))
        self.frame_subtasks.append(str(subtask_text or ""))
        self.frame_episode_steps.append(int(episode_step))

    def record_trajectory_step(self, step_record: dict[str, Any]) -> None:
        self.trajectory_steps.append(dict(step_record))

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
    ) -> None:
        """Stash the raw SteerVLA model input for the high-level dataset (chunk-start steps only).

        Called every env step; only the first step of each action chunk is retained (that's the
        one frame we need per chunk). ``image``/``state`` are the raw pre-step obs the action was
        taken from (``obs["image"]`` / ``obs["state"]``), not the annotated video frame. ``subtask``
        / ``reasoning`` are the **original** CoT targets the model produced for this chunk — used as
        the HL supervision targets when a GOOD/unlabeled chunk is reinforced as-is.
        """
        if not self.store_hl_dataset or image is None:
            return
        # Chunk starts are every ``action_chunk_steps`` env steps from the (1-based) episode start.
        if (int(episode_step) - 1) % max(1, self.action_chunk_steps) != 0:
            return
        self._model_inputs[int(episode_step)] = {
            "image": np.asarray(image, dtype=np.uint8),
            "state": None if state is None else np.asarray(state, dtype=np.float32).reshape(-1),
            "current_speed": float(current_speed),
            "prompt": str(prompt or ""),
            "subtask": str(subtask or ""),
            "reasoning": str(reasoning or ""),
            "action_chunk": None if action_chunk is None else np.asarray(action_chunk, dtype=np.float32),
        }

    # ── querying ─────────────────────────────────────────────────────────────────
    def should_query(self, episode_step: int, *, force: bool = False) -> bool:
        if len(self.frames) <= self._frames_cursor:
            return False
        if force:
            return self.query_on_episode_end
        if self.window_env_steps <= 0 or episode_step <= 0:
            return False
        if episode_step % self.window_env_steps != 0:
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
        # A coach/VLM failure (bad model id, auth, network, quota, parse error) must never
        # tear down the CARLA route or the training run. Catch everything, log it, and
        # advance the cursors so the failed window is skipped rather than retried forever.
        # NOTE: only Exception is caught here; a watchdog timeout raises KeyboardInterrupt,
        # which is why main_carla pauses the leaderboard watchdogs around this call.
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
        step_offset: int,
        done_info: dict[str, Any] | None,
    ) -> dict[str, Any]:
        done_info = done_info or {}
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
        chunk_original_subtask: dict[str, str] = {}
        for offset, subtask in enumerate(frames_window_subtasks):
            # Recorded frame ``offset`` maps to window env step ``offset * stride + 1``.
            window_step = offset * self.video_frame_stride + 1
            chunk_index = (window_step - 1) // self.action_chunk_steps
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
            "mean_end_speed_mps": mean_end_speed_mps,
            "window_reward_total": window_reward_total,
            "window_reward_mean": window_reward_mean,
            "collision_events": collision_events,
            "chunk_original_subtask": chunk_original_subtask,
            "steps": traj_window,
        }

    def _run_window(
        self,
        *,
        episode_step: int,
        done_info: dict[str, Any] | None,
        final: bool,
        global_step: int | None,
    ) -> None:
        frames_window = self.frames[self._frames_cursor:]
        subtasks_window = self.frame_subtasks[self._frames_cursor:]
        traj_window = self.trajectory_steps[self._traj_cursor:]
        step_offset = self._traj_cursor
        self.window_count += 1

        tag = f"ep{self.episode_count:04d}_win{self.window_count:04d}{'_final' if final else ''}"
        work_dir = self.artifact_dir / tag
        work_dir.mkdir(parents=True, exist_ok=True)

        video_path = work_dir / "rollout.mp4"
        write_frames_to_mp4(frames_window, video_path, fps=self.video_fps)
        metadata = self._build_metadata(
            traj_window, subtasks_window, step_offset=step_offset, done_info=done_info
        )
        (work_dir / "trajectory.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        print(
            f"[cast_relabel] window {self.window_count}: reviewing {len(frames_window)} frames "
            f"({len(traj_window)} env steps, offset={step_offset}) -> {work_dir}",
            flush=True,
        )

        plot_paths: list[Path] | None = None
        if self.include_plots_in_prompt:
            from coaches.trajectory_plots import generate_trajectory_plots

            plot_map = generate_trajectory_plots(metadata, work_dir / "plots")
            plot_paths = [plot_map["combined"]]

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
        )
        (work_dir / "cast_relabel.json").write_text(json.dumps(cast_json, indent=2), encoding="utf-8")

        # Persist BAD/relabeled chunks as high-level (VLM-backbone) SteerVLA samples.
        n_hl = self._store_hl_samples(cast_json, metadata, traj_window, tag)

        if self.debug:
            self._log_debug_video(frames_window, subtasks_window, cast_json, global_step=global_step)
        self._log_scalars(events, cast_json, n_hl_samples=n_hl, global_step=global_step)

        # Advance cursors so the next window starts fresh.
        self._frames_cursor = len(self.frames)
        self._traj_cursor = len(self.trajectory_steps)
        self._model_inputs = {}

        if self.save_artifacts:
            print(f"[cast_relabel] saved artifacts under {work_dir}", flush=True)

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
            )
            if not hl_samples:
                return 0
            hl_dir = self.hl_dataset_dir / tag
            write_hl_samples(hl_samples, hl_dir)
            self.hl_sample_count += len(hl_samples)
            print(
                f"[cast_relabel] wrote {len(hl_samples)} high-level samples "
                f"(total {self.hl_sample_count}) -> {hl_dir}",
                flush=True,
            )
            return len(hl_samples)
        except Exception as exc:  # noqa: BLE001 - HL storage must never break the run.
            import traceback

            print(f"[cast_relabel] HL sample storage failed (non-fatal): {exc}", flush=True)
            traceback.print_exc()
            return 0

    # ── logging ──────────────────────────────────────────────────────────────────
    def _log_debug_video(
        self,
        frames_window: list[np.ndarray],
        subtasks_window: list[str],
        cast_json: dict[str, Any],
        *,
        global_step: int | None,
    ) -> None:
        try:
            import wandb  # type: ignore
        except ImportError:
            return
        if wandb.run is None or not frames_window:
            return
        # Window-relative chunk index per recorded frame.
        frame_chunk_indices = [
            (offset * self.video_frame_stride) // self.action_chunk_steps
            for offset in range(len(frames_window))
        ]
        annotated = annotate_cast_relabel_frames(frames_window, frame_chunk_indices, cast_json)
        if not annotated:
            return
        video = np.stack(annotated, axis=0)  # (T, H, W, 3)
        video = np.transpose(video, (0, 3, 1, 2))  # (T, C, H, W) for W&B
        wandb.log(
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
        wandb.log(log, step=global_step)
