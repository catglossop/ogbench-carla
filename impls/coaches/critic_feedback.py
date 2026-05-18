from __future__ import annotations

from typing import Any

import numpy as np

from coaches.expert_label import (
    NUM_COMMENTARY_WORDS,
    NUM_DELTA_COMMENTARY_WORDS,
    delta_commentary_from_critic_actions,
)


def critic_language_dim(agent_config: Any) -> int:
    """Return replay-buffer language-label width for the configured critic mode."""
    mode = str(agent_config.get("critic_feedback_mode", "commentary_bow"))
    if mode == "none":
        return 0
    if mode == "action_delta":
        return int(agent_config.get("critic_action_dim", 4))
    if mode == "delta_commentary_bow":
        return NUM_DELTA_COMMENTARY_WORDS
    return NUM_COMMENTARY_WORDS


def compute_action_delta(
    obs_raw: dict,
    action_flat: np.ndarray,
    agent,
    agent_config,
) -> np.ndarray:
    """Compute ``first_step(expert_action) - first_step(agent_action)`` in critic action space."""
    critic_action_dim = int(agent_config.get("critic_action_dim", 4))
    zeros = np.zeros(critic_action_dim, dtype=np.float32)
    expert_raw = obs_raw.get("expert_action")
    if expert_raw is None:
        return zeros
    try:
        import jax.numpy as jnp

        expert_first = np.asarray(agent._as_critic_actions(jnp.array(np.asarray(expert_raw).reshape(1, -1))))[0]
        agent_first = np.asarray(agent._as_critic_actions(jnp.array(action_flat.reshape(1, -1))))[0]
        return (expert_first - agent_first).astype(np.float32)
    except Exception as e:
        print(f"[action_delta] failed: {e}", flush=True)
        return zeros


def compute_action_delta_commentary(
    obs_raw: dict,
    action_flat: np.ndarray,
    agent,
) -> tuple[str, np.ndarray]:
    """Compute a corrective BoW language label from expert-vs-agent action delta."""
    zeros = np.zeros(NUM_DELTA_COMMENTARY_WORDS, dtype=np.float32)
    expert_raw = obs_raw.get("expert_action")
    if expert_raw is None:
        return "", zeros
    try:
        import jax.numpy as jnp

        expert_first = np.asarray(
            agent._as_critic_actions(jnp.array(np.asarray(expert_raw).reshape(1, -1)))
        )[0]
        agent_first = np.asarray(agent._as_critic_actions(jnp.array(action_flat.reshape(1, -1))))[0]
        text, bow = delta_commentary_from_critic_actions(expert_first, agent_first)
        return text, bow.astype(np.float32)
    except Exception as e:
        print(f"[action_delta_commentary] failed: {e}", flush=True)
        return "", zeros
