"""Smoke tests for VLM action-chunk feedback → DSRL language_label wiring."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_IMPLS = Path(__file__).resolve().parents[1]
if str(_IMPLS) not in sys.path:
    sys.path.insert(0, str(_IMPLS))

from coaches.action_chunk_feedback import (  # noqa: E402
    chunk_feedback_to_language_label,
    language_label_for_episode_step,
    parse_chunk_feedback_response,
)
from coaches.expert_label import NUM_DELTA_COMMENTARY_WORDS  # noqa: E402
from utils.datasets import ReplayBuffer  # noqa: E402


def test_parse_and_label_lookup() -> None:
    affected = {2, 3}
    response = json.dumps(
        {
            "chunk_feedback": [
                {
                    "chunk_index": 2,
                    "lateral": "adjust right",
                    "longitudinal": "decelerate",
                    "detail": "Slow before the junction.",
                },
                {
                    "chunk_index": 3,
                    "lateral": None,
                    "longitudinal": "accelerate",
                    "detail": "",
                },
            ]
        }
    )
    parsed = parse_chunk_feedback_response(response, num_chunks=5, affected_chunk_indices=affected)
    assert parsed[2] is not None
    assert parsed[2]["lateral"] == "adjust right"
    assert parsed[3] is not None
    assert parsed[0] is None

    chunk_json = {
        "episode_steps": 50,
        "action_chunks": [
            {
                "chunk_index": i,
                "feedback": parsed[i],
            }
            for i in range(5)
        ],
    }
    text, bow = language_label_for_episode_step(chunk_json, episode_step=25, action_chunk_steps=10)
    assert "adjust right" in text or "decelerate" in text
    assert bow.shape == (NUM_DELTA_COMMENTARY_WORDS,)
    assert bow.any()

    text0, bow0 = language_label_for_episode_step(chunk_json, episode_step=5, action_chunk_steps=10)
    assert text0 == ""
    assert not bow0.any()


def test_backfill_writes_language_label() -> None:
    """OnlineVLMSession must backfill DSRL's ``language_label`` field, not ``coach_label``."""
    from coaches.online_vlm_coach import OnlineVLMSession

    example = dict(
        observations=np.zeros(4, dtype=np.float32),
        actions=np.zeros(2, dtype=np.float32),
        rewards=np.float32(0.0),
        next_observations=np.zeros(4, dtype=np.float32),
        masks=np.float32(1.0),
        terminals=np.float32(0.0),
        language_label=np.zeros(NUM_DELTA_COMMENTARY_WORDS, dtype=np.float32),
        next_language_label=np.zeros(NUM_DELTA_COMMENTARY_WORDS, dtype=np.float32),
    )
    buffer = ReplayBuffer.create(example, size=8)
    idx = buffer.add_transition(dict(example))

    session = OnlineVLMSession({"query_every_n_episode_steps": 0}, save_dir="/tmp/vlm_coach_test")
    session._chunk_feedback_windows = [
        (
            0,
            {
                "episode_steps": 10,
                "action_chunks": [
                    {
                        "chunk_index": 0,
                        "feedback": {
                            "lateral": "adjust left",
                            "longitudinal": None,
                            "detail": "",
                        },
                    }
                ],
            },
        )
    ]
    session.episode_buffer_indices = [idx]
    session.episode_step_for_buffer = [1]
    session.chunk_feedback_json = session._chunk_feedback_windows[0][1]

    session.backfill_buffer(buffer)

    stored = np.asarray(buffer._dict["language_label"][idx])
    assert stored.any(), "backfill should write non-zero language_label"
    assert "coach_label" not in buffer._dict


if __name__ == "__main__":
    test_parse_and_label_lookup()
    test_backfill_writes_language_label()
    print("action_chunk_feedback integration OK")
