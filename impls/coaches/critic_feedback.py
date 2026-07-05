from __future__ import annotations

from typing import Any

import numpy as np

from coaches.expert_label import (
    NUM_COMMENTARY_WORDS,
    NUM_DELTA_COMMENTARY_WORDS,
    collision_override_delta_commentary,
    delta_commentary_from_critic_actions,
)


def critic_language_dim(agent_config: Any) -> int:
    """Return replay-buffer language-label width for the configured critic mode."""
    mode = resolve_critic_feedback_mode(agent_config)
    if mode == "none":
        return 0
    if mode == "expert_action":
        explicit = agent_config.get("language_label_dim")
        if explicit is not None:
            return int(explicit)
        return int(agent_config.get("critic_action_dim", 4)) + 1
    if mode == "action_delta":
        return int(agent_config.get("critic_action_dim", 4))
    if mode in ("delta_commentary_bow", "vlm_chunk_bow"):
        return NUM_DELTA_COMMENTARY_WORDS
    return NUM_COMMENTARY_WORDS


_EXPLICIT_MODES = frozenset({"none", "expert_action", "action_delta", "delta_commentary_bow", "vlm_chunk_bow"})


def resolve_critic_feedback_mode(agent_config: Any) -> str:
    """Map ``language_feedback`` config (or legacy ``critic_feedback_mode``) to a mode string."""
    # An explicit critic_feedback_mode (any non-commentary-bow value) always wins over
    # language_feedback settings, so that run_carla.sh --critic-mode flags take effect.
    cfm = agent_config.get("critic_feedback_mode")
    if cfm is not None and str(cfm).strip().lower() in _EXPLICIT_MODES:
        return str(cfm).strip().lower()
    lang_fb = agent_config.get("language_feedback")
    if lang_fb is not None:
        src = str(lang_fb.get("source", "expert")).strip().lower()
        if src == "vlm":
            return "vlm_chunk_bow"
        expert_mode = lang_fb.get("expert_mode")
        if expert_mode is not None:
            return str(expert_mode)
    return str(agent_config.get("critic_feedback_mode", "commentary_bow"))


def _expert_first_step(obs_raw: dict, agent, expert_first=None):
    """Expert action in critic action space, or None when unavailable.

    ``expert_first`` overrides the default waypoint-chunk first-step extraction —
    used by the accel_steer residual mode to provide the PID-decoded
    ``[accel, steer]`` controls instead of waypoint displacements.
    """
    if expert_first is not None:
        return np.asarray(expert_first, dtype=np.float32).reshape(-1)
    expert_raw = obs_raw.get("expert_action")
    if expert_raw is None:
        return None
    import jax.numpy as jnp

    return np.asarray(
        agent._env_action_first_step(jnp.array(np.asarray(expert_raw).reshape(1, -1)))
    )[0]


def compute_expert_target(
    obs_raw: dict,
    agent,
    agent_config,
    expert_first=None,
) -> np.ndarray:
    """Privileged critic input: ``[first_step(expert_action), validity]`` — state-only.

    Unlike ``action_delta`` (expert − agent), this label does not depend on the
    agent's action, so it stays consistent across the Bellman recursion (the
    bootstrap conditions Q(s', ·, a') on the expert target *at s'*) and across the
    actor update (freshly sampled actions pair correctly with a state-only label).

    When collision contact is active the target is overridden to zeros ("stop").
    The trailing validity flag is 1.0 when the expert action was available, so an
    all-zero target is distinguishable from "no expert this step".
    """
    critic_action_dim = int(agent_config.get("critic_action_dim", 4))
    out = np.zeros(critic_action_dim + 1, dtype=np.float32)
    try:
        target = _expert_first_step(obs_raw, agent, expert_first)
        if target is None:
            return out
        if bool(obs_raw.get("_collision_active_private", False)):
            target = np.zeros_like(target)
        n = min(critic_action_dim, int(target.shape[0]))
        out[:n] = target[:n]
        out[-1] = 1.0
        return out
    except Exception as e:
        print(f"[expert_target] failed: {e}", flush=True)
        return out


def compute_action_delta(
    obs_raw: dict,
    action_flat: np.ndarray,
    agent,
    agent_config,
    expert_first=None,
    agent_first=None,
) -> np.ndarray:
    """Compute ``first_step(expert_action) - first_step(agent_action)`` in critic action space.

    ``expert_first`` / ``agent_first`` override the waypoint-chunk first-step
    extraction (accel_steer residual mode passes PID-decoded 2-D controls).

    Note: this label depends on the logged action, so the bootstrap label must be
    backfilled one step later (main_carla.py) and the actor's Q-query still pairs
    fresh actions with the logged action's delta — prefer ``expert_action`` for
    residual SAC.
    """
    critic_action_dim = int(agent_config.get("critic_action_dim", 4))
    zeros = np.zeros(critic_action_dim, dtype=np.float32)
    try:
        expert_vec = _expert_first_step(obs_raw, agent, expert_first)
        if expert_vec is None:
            return zeros
        if agent_first is not None:
            agent_vec = np.asarray(agent_first, dtype=np.float32).reshape(-1)
        else:
            import jax.numpy as jnp

            agent_vec = np.asarray(
                agent._env_action_first_step(jnp.array(action_flat.reshape(1, -1)))
            )[0]
        collision_active = bool(obs_raw.get("_collision_active_private", False))
        agent_forward = float(agent_vec[0]) if agent_vec.shape[0] > 0 else 0.0
        agent_speed = float(np.linalg.norm(agent_vec[:2])) if agent_vec.shape[0] >= 2 else 0.0
        if collision_active and (agent_forward > 0.05 or agent_speed > 0.10):
            # Override the nominal expert target with a "stop pushing" target.
            target = np.zeros_like(agent_vec, dtype=np.float32)
            return (target - agent_vec).astype(np.float32)
        return (expert_vec - agent_vec).astype(np.float32)
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
            agent._env_action_first_step(jnp.array(np.asarray(expert_raw).reshape(1, -1)))
        )[0]
        agent_first = np.asarray(agent._env_action_first_step(jnp.array(action_flat.reshape(1, -1))))[0]
        collision_active = bool(obs_raw.get("_collision_active_private", False))
        # Heuristic "throttling" test in critic-action space: positive forward
        # speed target means the policy is trying to keep moving into contact.
        agent_forward = float(agent_first[0]) if agent_first.shape[0] > 0 else 0.0
        agent_speed = float(np.linalg.norm(agent_first[:2])) if agent_first.shape[0] >= 2 else 0.0
        if collision_active and (agent_forward > 0.05 or agent_speed > 0.10):
            text, bow = collision_override_delta_commentary()
            return text, bow.astype(np.float32)
        text, bow = delta_commentary_from_critic_actions(expert_first, agent_first)
        return text, bow.astype(np.float32)
    except Exception as e:
        print(f"[action_delta_commentary] failed: {e}", flush=True)
        return "", zeros
