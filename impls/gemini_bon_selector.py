"""Gemini-based best-of-N action selection.

Alternative to a learned critic (see qgf_guidance.py's make_q_fn_batched): each
candidate action chunk is projected onto the first-person camera frame as its
route/speed waypoints (ogbench.carla.waypoint_viz.annotate_waypoints_on_frame),
paired with its subtask text, and all N candidates are sent together in a
single multiple-choice prompt asking Gemini to pick the single best one.
"""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
from PIL import Image

_PROMPT_PREAMBLE = (
    "You are choosing the best candidate short-horizon driving trajectory for an "
    "autonomous vehicle. Below are {n} candidate images, each labeled with its index "
    "(also stamped on the image itself, near the end of its plotted path), showing the "
    "SAME forward-facing camera view from the vehicle with a DIFFERENT candidate "
    "trajectory projected onto it as colored dots:\n"
    "- RED dots: the candidate's planned route/position waypoints (where the vehicle "
    "will be).\n"
    "- GREEN dots: the candidate's planned speed waypoints (spacing between "
    "consecutive dots indicates speed -- farther apart means faster).\n"
    "Each candidate also has a short text description of the driving subtask it is "
    "attempting.\n\n"
    "First, identify the CRITICAL AGENTS in the scene -- other vehicles, pedestrians, "
    "cyclists, and relevant traffic controls (lights, signs) -- and how they constrain "
    "safe driving right now (e.g. a pedestrian about to cross, a car merging, a red "
    "light ahead). Then pick the SINGLE best candidate: the one whose trajectory is "
    "safest with respect to those critical agents, most on-route, and most "
    "appropriately paced for the situation. Respond with strict JSON only: "
    '{{"critical_agents": "<brief description of the key agents/hazards you identified>", '
    '"choice": <candidate index, 0 to {max_idx}>, '
    '"reasoning": "<one short sentence on why this candidate best handles those agents>"}}.'
)


class GeminiActionSelector:
    def __init__(self, model: str = "gemini-2.5-flash", api_key: str | None = None):
        from google import genai

        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "GEMINI_API_KEY not set in the environment. Export it (e.g. from "
                "~/.bashrc) before launching -- it is not sourced automatically for "
                "non-interactive/background shells."
            )
        self.model = model
        self._client = genai.Client(api_key=key)

    def select_candidate(
        self, frames: list[np.ndarray], subtasks: list[str]
    ) -> tuple[int, str, str, np.ndarray]:
        """One multiple-choice API call covering all N candidates at once.

        Returns ``(choice_idx, critical_agents, reasoning, one_hot_scores)`` -- the
        one-hot array lets this slot into the existing critic-style ``{"q_vals": ...}``
        candidates dict (argmax-selection and the candidates panel are value-agnostic)
        without needing a separate code path. On any API/parse failure, falls back to
        choice 0 (the first sampled candidate) with an error message, rather than
        raising and killing the whole rollout.
        """
        from google.genai import types

        n = len(frames)
        parts: list[Any] = []
        for i, (frame, subtask) in enumerate(zip(frames, subtasks)):
            parts.append(f"Candidate {i} (subtask: {(subtask or '').strip() or '(none given)'}):")
            parts.append(Image.fromarray(np.asarray(frame, dtype=np.uint8)))
        prompt_text = _PROMPT_PREAMBLE.format(n=n, max_idx=n - 1)
        contents = [prompt_text, *parts]
        try:
            resp = self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0,
                ),
            )
            data = json.loads(resp.text)
            choice = int(data["choice"])
            if not (0 <= choice < n):
                raise ValueError(f"choice {choice} out of range [0, {n})")
            critical_agents = str(data.get("critical_agents", ""))
            reasoning = str(data.get("reasoning", ""))
        except Exception as exc:  # noqa: BLE001
            choice = 0
            critical_agents = ""
            reasoning = f"<gemini error, defaulted to candidate 0: {exc}>"
        scores = np.zeros(n, dtype=np.float32)
        scores[choice] = 1.0
        return choice, critical_agents, reasoning, scores
