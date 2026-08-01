"""Q-Guided Flow (QGF) inference-time guidance for pi0/SteerVLA on CARLA.

Applies Q-gradient guidance at each denoising step of pi0's flow-matching policy
using a separately pretrained offline critic (from pretrain_critic.py checkpoints).

Reference (QGF): https://arxiv.org/pdf/2606.11087

The pretrained critic (from pretrain_critic.py) takes:
  obs_enc:     (B, 1152)  SigLIP image-only embedding
  action_flat: (B, 40)    flattened 10-step DELTA_XY_T_DELTA_XY_SPACE waypoints

No RLDS normalization is used. The model outputs raw physical-space actions, so
a_approx_critic is already in the same space the critic was trained on (~0.5 m/step
for all 4 dims). No conversion is needed at inference time.

Guidance (pi0 convention: t goes 1→0, so t=1 is noise, t=0 is clean):
  a_approx  = clip(x_t + t * v_bc, -1, 1)         one-Euler denoised estimate
  qgrad     = ∇_{a_approx} min Q(obs_enc, a_approx)  gradient in model space
  v_guided  = v_bc + guidance_weight * qgrad
"""

from __future__ import annotations

import pickle
from typing import Callable

import jax
import jax.numpy as jnp


def model_to_physical_flat(
    a_model_flat: jnp.ndarray,
    action_horizon: int = 10,
    action_dim: int = 4,
) -> jnp.ndarray:
    """Identity: critic is trained in model space, no conversion needed."""
    return a_model_flat


def qgrad_physical_to_model_flat(
    qgrad_phys_flat: jnp.ndarray,
    action_horizon: int = 10,
    action_dim: int = 4,
) -> jnp.ndarray:
    """Identity: chain rule is diag(1,...,1) since critic space == model space."""
    return qgrad_phys_flat


def load_pretrained_critic(ckpt_path: str) -> tuple:
    """Load a pretrain_critic.py checkpoint.

    Returns (critic_def, critic_params) where:
      critic_def    – Flax Linen Critic module expecting (obs_enc, action_flat)
      critic_params – 'modules_critic' sub-params from the .pkl file
    """
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from jax_agents.dsrl import Critic

    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)

    all_params = ckpt["params"]
    if "modules_critic" not in all_params:
        raise ValueError(
            f"No 'modules_critic' in {ckpt_path}. Found: {list(all_params.keys())}"
        )
    critic_params = all_params["modules_critic"]

    # Architecture matches pretrain_critic.py defaults (hidden_dims=(256,256), layer_norm=True, ensemble=2).
    critic_def = Critic(hidden_dims=(256, 256), layer_norm=True, ensemble_size=2)
    return critic_def, critic_params


def make_q_fn(critic_def, critic_params: dict) -> Callable:
    """Return jitted f(obs_enc_1152, action_model_flat) → scalar min-ensemble Q value."""

    @jax.jit
    def q_fn(obs_enc: jnp.ndarray, action_model_flat: jnp.ndarray) -> jnp.ndarray:
        qs = critic_def.apply({"params": critic_params}, obs_enc, action_model_flat)  # (2, B)
        return jnp.min(qs, axis=0).mean()

    return q_fn


def make_q_fn_batched(critic_def, critic_params: dict) -> Callable:
    """Return jitted f(obs_enc_1152, action_model_flat) -> per-row min-ensemble Q, shape (B,).

    Unlike :func:`make_q_fn` (which reduces to a scalar mean, for use as a gradient
    target), this keeps the batch dimension so callers can rank/argmax candidates
    against each other (e.g. best-of-N action selection).
    """

    @jax.jit
    def q_fn_batched(obs_enc: jnp.ndarray, action_model_flat: jnp.ndarray) -> jnp.ndarray:
        qs = critic_def.apply({"params": critic_params}, obs_enc, action_model_flat)  # (2, B)
        return jnp.min(qs, axis=0)  # (B,)

    return q_fn_batched


def make_q_grad_fn(critic_def, critic_params: dict) -> Callable:
    """Return jitted f(obs_enc_1152, action_flat_40) → qgrad_flat_40.

    The gradient is ∂(min_ensemble Q)/∂action_flat in model space.
    """

    @jax.jit
    def q_grad_fn(obs_enc: jnp.ndarray, action_model_flat: jnp.ndarray) -> jnp.ndarray:
        def q_scalar(a: jnp.ndarray) -> jnp.ndarray:
            qs = critic_def.apply({"params": critic_params}, obs_enc, a)  # (2, B)
            return jnp.min(qs, axis=0).mean()  # scalar

        return jax.grad(q_scalar)(action_model_flat)

    return q_grad_fn
