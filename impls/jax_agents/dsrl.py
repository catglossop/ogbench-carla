"""Minimal JAX implementation of DSRL (Diffusion Steering RL).

Reference: https://arxiv.org/abs/2506.15799 .

This implementation keeps the core idea -- a stochastic *noise* policy that is
pushed through a behaviour-cloning **flow** to produce environment actions, with
a SAC-style actor / critic update -- but trims the paper's full algorithm to
something easy to read, run online, and extend:

* BC **flow actor** ``v_phi(s, a, t)`` trained with flow-matching loss on stored
  ``(s, a)`` pairs in the replay buffer. Sampling integrates Euler from
  ``x_0 ~ N(0, I) * noise_scale`` to ``x_1`` over ``flow_steps``.
* **Stochastic noise actor** ``pi_psi(z|s)`` -- a tanh-squashed diagonal Gaussian
  whose samples ``z`` seed the flow.
* **Critic ensemble** ``Q_theta(s, a)`` (size 2 by default) with a target copy.
* Optional **rollout SteerVLA**: pass ``vla_sample_fn`` ``(obs_batch, noise_batch) -> action_batch`` into
  :meth:`DSRLAgent.create`; :meth:`DSRLAgent.sample_actions_with_vla` uses it instead of the BC flow.
* Joint **VLA + DSRL training** (when ``vla_train_state`` and ``openpi_train_config`` are set):
  :meth:`DSRLAgent.update_with_vla` updates the Flax stack with ``critic_loss_vla`` + ``actor_loss_vla``
  (both use :func:`vlas.steervla.flow_sample_with_vla` instead of :meth:`_flow_sample`), and updates OpenPI
  with :func:`vlas.steervla.vla_flow_only_train_step` (Pi0-CoT **action** flow-matching only). Batches must
  include ``openpi_observation``, ``next_openpi_observation``, and ``openpi_actions`` (model-normalized chunks).

Things deliberately omitted vs. the paper that you can add later:

* z-critic distillation (we sample noises through the flow each gradient step).
* best-of-n action selection.
* learnable temperature alpha (we use a fixed ``alpha`` for now; trivial to add).
* target BC flow (we always sample with the live flow params).

The module-level ``get_config()`` makes this discoverable to
``--agent=jax_agents/dsrl.py``.
"""

from __future__ import annotations

import copy
import functools
import time
from typing import Any, Callable, Optional

import distrax
import flax
import flax.linen as nn
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import ImpalaEncoder
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP, default_init, ensemblize

PI05_ACTION_DIM = 32 

@functools.lru_cache(maxsize=1)
def _steervla():
    """Deferred import (cached once): ``vlas.steervla`` imports this module."""
    import vlas.steervla as steervla_mod

    return steervla_mod


# def dsrl_encode_obs_sample_noise_logprob(
#     network: TrainState,
#     noise_scale: jnp.ndarray | float | int,
#     observations: jnp.ndarray,
#     rng: jnp.ndarray,
# ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
#     """Encode observations and sample noise + log-prob from the **noise actor** (DSRL).

#     DSRL params are treated as constants (no gradients). Matches :meth:`DSRLAgent.actor_loss`
#     sampling, without integrating the BC flow.
#     """
#     params = jax.tree.map(jax.lax.stop_gradient, network.params)
#     obs_e = network.select("obs_encoder")(observations, params=params)
#     dist = network.select("noise_actor")(obs_e, params=params)
#     noise, log_prob = dist.sample_and_log_prob(seed=rng)
#     ns = jnp.asarray(noise_scale, dtype=noise.dtype)
#     return obs_e, noise * ns, log_prob


def dsrl_critic_min_q(network: TrainState, obs_e: jnp.ndarray, actions: jnp.ndarray, language_label=None) -> jnp.ndarray:
    """``min_k Q_k(obs_e, actions)`` with critic weights frozen; gradients flow through ``actions`` only."""
    params = jax.tree.map(jax.lax.stop_gradient, network.params)
    obs_e_sg = jax.lax.stop_gradient(obs_e)
    if language_label is not None:
        obs_e_sg = jnp.concatenate([obs_e_sg, jnp.asarray(language_label, dtype=jnp.float32)], axis=-1)
    qs = network.select("noise_critic")(obs_e_sg, actions, params=params)
    return jnp.min(qs, axis=0)


def _critic_obs_e(obs_e: jnp.ndarray, batch: dict, key: str) -> jnp.ndarray:
    """Append language label from ``batch[key]`` to encoded observation for critic-only consumption."""
    lang = batch.get(key)
    if lang is None:
        return obs_e
    return jnp.concatenate([obs_e, jnp.asarray(lang, dtype=jnp.float32)], axis=-1)


def _batch_obs_embed_or_encoder(
    network: TrainState,
    batch: dict,
    *,
    obs_key: str,
    embed_key: str,
    params,
) -> jnp.ndarray:
    """Use precomputed critic embeddings from ``batch`` when present, else ``obs_encoder``."""
    if embed_key in batch:
        return jnp.asarray(batch[embed_key], dtype=jnp.float32)
    return network.select("obs_encoder")(batch[obs_key], params=params)

@jax.jit
def _critic_loss_vla_pure_math(
    network: TrainState,
    batch: dict,
    next_actions_critic: jnp.ndarray,
    critic_actions: jnp.ndarray,
    grad_params,
    discount: jnp.ndarray,
    next_log_pi=None,
    alpha=None,
):
    """Pure DSRL critic math: target-Q bootstrap + critic MSE, no VLA model access.

    ``next_actions_critic`` and ``critic_actions`` must already be clipped to the env
    action layout (e.g. via :meth:`DSRLAgent._clip_actions_to_env`).

    When ``next_log_pi`` and ``alpha`` are provided (residual SAC path), the soft
    Bellman target subtracts the entropy bonus: ``min_Q(s',a') - alpha * log_pi(a'|s')``.
    """
    next_obs_e = _batch_obs_embed_or_encoder(
        network, batch, obs_key="next_observations", embed_key="critic_next_obs_e", params=network.params,
    )
    _next_critic_obs_e = _critic_obs_e(next_obs_e, batch, "next_language_label")
    if "base_next_actions" in batch:
        _next_critic_obs_e = jnp.concatenate(
            [_next_critic_obs_e, jnp.asarray(batch["base_next_actions"], dtype=jnp.float32)], axis=-1
        )
    next_qs = network.select("target_critic")(_next_critic_obs_e, next_actions_critic)
    next_q = jnp.min(next_qs, axis=0)
    if next_log_pi is not None and alpha is not None:
        next_q = next_q - alpha * next_log_pi
    target_q = batch["rewards"] + discount * batch["masks"] * next_q

    obs_e = _batch_obs_embed_or_encoder(
        network, batch, obs_key="observations", embed_key="critic_obs_e", params=grad_params,
    )
    _curr_critic_obs_e = _critic_obs_e(obs_e, batch, "language_label")
    if "base_actions" in batch:
        _curr_critic_obs_e = jnp.concatenate(
            [_curr_critic_obs_e, jnp.asarray(batch["base_actions"], dtype=jnp.float32)], axis=-1
        )
    qs = network.select("critic")(_curr_critic_obs_e, critic_actions, params=grad_params)
    critic_loss = jnp.square(qs - target_q[None]).mean()
    return critic_loss, {
        "critic_loss": critic_loss,
        "q_mean": qs.mean(),
        "q_max": qs.max(),
        "q_min": qs.min(),
        "target_q": target_q.mean(),
    }


@jax.jit
def _noise_critic_loss_vla_pure_math(
    network: TrainState,
    batch: dict,
    vla_actions: jnp.ndarray,
    noise_actions: jnp.ndarray,
    grad_params,
):
    """Distill ``noise_critic(s, z)`` toward ``critic(s, VLA(s, z))`` at the same state."""
    obs_e = network.select("obs_encoder")(batch["observations"], params=grad_params)
    noise_qs = network.select("noise_critic")(
        _critic_obs_e(obs_e, batch, "language_label"), noise_actions, params=grad_params,
    )
    actions_sg = jax.lax.stop_gradient(vla_actions)
    target_qs = network.select("critic")(
        _critic_obs_e(obs_e, batch, "language_label"), actions_sg,
    )
    noise_q = jnp.min(noise_qs, axis=0)
    target_q = jnp.min(target_qs, axis=0)
    noise_critic_loss = jnp.square(noise_q - jax.lax.stop_gradient(target_q)).mean()
    return noise_critic_loss, {
        "noise_critic_loss": noise_critic_loss,
        "noise_q_values": noise_q.mean(),
        "q_values": target_q.mean(),
    }


@jax.jit
def _noise_actor_loss_vla_pure_math(
    network: TrainState,
    batch: dict,
    grad_params,
    rng_noise: jnp.ndarray,
    alpha: jnp.ndarray,
    noise_scale: jnp.ndarray,
):
    """SAC-style noise actor loss using ``noise_critic`` Q-values."""
    noise_actions = _vla_forward_prepare_actor_noise(
        network, batch["observations"], rng_noise, noise_scale,
    )
    obs_e = network.select("obs_encoder")(batch["observations"], params=grad_params)
    dist = network.select("noise_actor")(obs_e, params=grad_params)
    log_prob = dist.log_prob(noise_actions / noise_scale)
    qs = network.select("noise_critic")(
        _critic_obs_e(obs_e, batch, "language_label"), noise_actions,
    )
    q = jnp.min(qs, axis=0)
    actor_loss = (alpha * log_prob - q).mean()
    return actor_loss, {
        "noise_actor_loss": actor_loss,
        "noise_log_prob": log_prob.mean(),
        "q_for_noise_actor": q.mean(),
    }


@jax.jit
def _apply_grads_pure(network: TrainState, grads):
    """Apply ``grads`` to ``network`` via optax and return the new ``TrainState`` + grad stats."""
    grad_max = jax.tree_util.tree_map(jnp.max, grads)
    grad_min = jax.tree_util.tree_map(jnp.min, grads)
    grad_norm = jax.tree_util.tree_map(jnp.linalg.norm, grads)
    grad_max_flat = jnp.concatenate([jnp.reshape(x, -1) for x in jax.tree_util.tree_leaves(grad_max)], axis=0)
    grad_min_flat = jnp.concatenate([jnp.reshape(x, -1) for x in jax.tree_util.tree_leaves(grad_min)], axis=0)
    grad_norm_flat = jnp.concatenate([jnp.reshape(x, -1) for x in jax.tree_util.tree_leaves(grad_norm)], axis=0)
    stats = {
        "grad/max": jnp.max(grad_max_flat),
        "grad/min": jnp.min(grad_min_flat),
        "grad/norm": jnp.linalg.norm(grad_norm_flat, ord=1),
    }

    updates, new_opt_state = network.tx.update(grads, network.opt_state, network.params)
    new_params = optax.apply_updates(network.params, updates)
    new_network = network.replace(
        step=network.step + 1,
        params=new_params,
        opt_state=new_opt_state,
    )
    return new_network, stats

@jax.jit
def _vla_forward_prepare_actor_noise(
    network: TrainState,
    observations: jnp.ndarray,
    rng_noise: jnp.ndarray,
    noise_scale: jnp.ndarray,
) -> jnp.ndarray:
    """Pure JAX prep for actor VLA forward: sample frozen policy noise on current observations."""
    obs_e_frozen = network.select("obs_encoder")(observations, params=network.params)
    dist_frozen = network.select("noise_actor")(obs_e_frozen)
    return dist_frozen.sample(seed=rng_noise) * noise_scale


# --------------------------------------------------------------------------- #
# Networks                                                                    #
# --------------------------------------------------------------------------- #


class CarlaObservationEncoder(nn.Module):
    """Encode CARLA observations: vector state, precomputed SigLIP/policy embeds, or RGB via IMPALA."""

    observation_mode: str
    image_encoder: str = "impala"
    impala_width: int = 1
    impala_stack_sizes: tuple = (16, 32, 32)
    impala_num_blocks: int = 2
    image_mlp_hidden_dims: tuple = (512,)
    layer_norm: bool = False

    @nn.compact
    def __call__(self, observations):
        if self.observation_mode in ("state", "policy_embed"):
            return observations.astype(jnp.float32)
        if self.image_encoder == "siglip":
            # Precomputed float32 SigLIP embeddings — pass through unchanged.
            return observations.astype(jnp.float32)
        return ImpalaEncoder(
            width=self.impala_width,
            stack_sizes=self.impala_stack_sizes,
            num_blocks=self.impala_num_blocks,
            mlp_hidden_dims=self.image_mlp_hidden_dims,
            layer_norm=self.layer_norm,
        )(observations)


class FlowActor(nn.Module):
    """Predicts the velocity ``v(s, a, t)`` for a flow from noise to action."""

    hidden_dims: tuple
    action_dim: int
    layer_norm: bool = False
    time_embed_dim: int = 16

    @nn.compact
    def __call__(self, observations, actions, times):
        # times has shape (..., 1); broadcast a Fourier embedding.
        if times.ndim == observations.ndim - 1:
            times = times[..., None]
        freqs = jnp.exp(jnp.linspace(0.0, 4.0, self.time_embed_dim // 2))
        sinusoid = jnp.concatenate(
            [jnp.sin(times * freqs), jnp.cos(times * freqs)], axis=-1
        )
        x = jnp.concatenate([observations, actions, sinusoid], axis=-1)
        x = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)(x)
        velocity = nn.Dense(self.action_dim, kernel_init=default_init(0.01))(x)
        return velocity


class NoiseActor(nn.Module):
    """Tanh-squashed diagonal Gaussian over the *noise* space."""

    hidden_dims: tuple
    action_dim: int
    log_std_min: float = -5.0
    log_std_max: float = 2.0

    @nn.compact
    def __call__(self, observations, temperature: float = 1.0):
        x = MLP(self.hidden_dims, activate_final=True)(observations)
        mean = nn.Dense(self.action_dim, kernel_init=default_init(0.01))(x)
        log_std = nn.Dense(self.action_dim, kernel_init=default_init(0.01))(x)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        base = distrax.MultivariateNormalDiag(
            loc=mean, scale_diag=jnp.exp(log_std) * temperature
        )
        return distrax.Transformed(base, distrax.Block(distrax.Tanh(), ndims=1))


class Critic(nn.Module):
    """Q(s, a) with optional ensemble; layer-norm by default."""

    hidden_dims: tuple
    layer_norm: bool = True
    ensemble_size: int = 2

    def setup(self):
        mlp = ensemblize(MLP, self.ensemble_size)
        self.value_net = mlp(
            (*self.hidden_dims, 1), activate_final=False, layer_norm=self.layer_norm
        )

    def __call__(self, observations, actions):
        inputs = jnp.concatenate([observations, actions], axis=-1)
        return self.value_net(inputs).squeeze(-1)


# --------------------------------------------------------------------------- #
# Agent                                                                       #
# --------------------------------------------------------------------------- #
class DSRLAgent(flax.struct.PyTreeNode):
    """Minimal JAX DSRL agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()
    vla_sample_fn: Any = nonpytree_field()  # Optional callable: (obs, noise) -> action
    # vla_train_state: Any = nonpytree_field()  # OpenPI ``training_utils.TrainState`` for :meth:`update_with_vla`
    openpi_train_config: Any = nonpytree_field()  # ``openpi.training.config.TrainConfig`` for VLA flow step
    steervla_actor: Any = nonpytree_field()  # Optional attached local ``SteerVLAActor`` instance
    sac_residual_agent: Any = None  # Optional ``jax_agents.sac_residual.SACResidualAgent`` (pytree)

    @staticmethod
    def _as_jax_pytree(x):
        """Convert leaves to JAX arrays (no-op for existing JAX arrays)."""
        return jax.tree.map(lambda y: jnp.asarray(y), x)

    def _vla_flow_sample_kwargs(self) -> dict[str, Any]:
        c = self.config
        return dict(
            cot_temperature=float(c.get("vla_cot_temperature", 0.0)),
            cot_replay_reasoning=bool(c.get("vla_cot_replay_reasoning", True)),
            sample_actions_num_steps=int(c.get("vla_sample_actions_num_steps", 10)),
            sample_actions_low_memory=bool(c.get("vla_sample_actions_low_memory", True)),
            sample_actions_jit_denoise_steps=bool(c.get("vla_sample_actions_jit_denoise_steps", False)),
            action_horizon=int(c.get("vla_action_horizon", 10)),
            action_dim=int(c.get("vla_action_dim", 4)),
            output_action_format=c.get("vla_output_action_format") or "DELTA_XY_T_DELTA_XY_SPACE",
        )

    def _model_action_horizon(self) -> int:
        """Horizon for Pi0 / noise actor (``action_horizon``)."""
        return int(self.config.get("action_horizon", self.config.get("vla_action_horizon", 10)))

    def _model_action_dim(self) -> int:
        """Per-step dim for noise / Pi0 actions (``actor_action_dim``)."""
        return int(self.config.get("actor_action_dim", PI05_ACTION_DIM))

    def _env_action_horizon(self) -> int:
        """Horizon for env rollout and action critic (``vla_action_horizon``)."""
        return int(self.config.get("vla_action_horizon", self.config.get("action_horizon", 10)))

    def _env_action_dim(self) -> int:
        """Per-step dim for env rollout and action critic (``vla_action_dim``)."""
        return int(self.config.get("vla_action_dim", 4))

    def _flat_noise_dim(self) -> int:
        return self._model_action_horizon() * self._model_action_dim()

    def _flat_env_action_dim(self) -> int:
        return self._env_action_horizon() * self._env_action_dim()

    def _as_flat_noise(self, noise: jnp.ndarray) -> jnp.ndarray:
        """Flatten noise to ``(B, action_horizon * actor_action_dim)``."""
        x = jnp.asarray(noise, dtype=jnp.float32)
        flat_dim = self._flat_noise_dim()
        if x.ndim == 3:
            ah = self._model_action_horizon()
            ad = self._model_action_dim()
            return x[:, :ah, :ad].reshape(x.shape[0], flat_dim)
        if x.ndim == 2 and x.shape[-1] == flat_dim:
            return x
        raise ValueError(
            f"Cannot map noise shape {tuple(x.shape)} to flat noise dim {flat_dim} "
            f"(action_horizon={self._model_action_horizon()}, actor_action_dim={self._model_action_dim()})."
        )

    def _as_noise_chunk(self, noise: jnp.ndarray) -> jnp.ndarray:
        """Reshape noise to ``(B, action_horizon, actor_action_dim)``."""
        flat = self._as_flat_noise(noise)
        return flat.reshape(flat.shape[0], self._model_action_horizon(), self._model_action_dim())

    def _clip_actions_to_env(self, actions: jnp.ndarray) -> jnp.ndarray:
        """Clip model / Pi0 actions to flat env layout ``(B, vla_action_horizon * vla_action_dim)``."""
        x = jnp.asarray(actions, dtype=jnp.float32)
        ah = self._env_action_horizon()
        ad = self._env_action_dim()
        flat_env = ah * ad

        if x.ndim == 3:
            return x[:, :ah, :ad].reshape(x.shape[0], flat_env)
        if x.ndim == 2:
            if x.shape[-1] == flat_env:
                return x
            model_flat = self._flat_noise_dim()
            if x.shape[-1] == model_flat:
                chunk = x.reshape(x.shape[0], self._model_action_horizon(), self._model_action_dim())
                return chunk[:, :ah, :ad].reshape(x.shape[0], flat_env)
            if x.shape[-1] == ah * self._model_action_dim():
                chunk = x.reshape(x.shape[0], ah, self._model_action_dim())
                return chunk[:, :, :ad].reshape(x.shape[0], flat_env)

        raise ValueError(
            f"Cannot clip actions shape {tuple(x.shape)} to env flat dim {flat_env} "
            f"(vla_action_horizon={ah}, vla_action_dim={ad})."
        )

    def _env_action_first_step(self, actions: jnp.ndarray) -> jnp.ndarray:
        """First env step ``(B, vla_action_dim)`` for action-delta critic feedback."""
        flat = self._clip_actions_to_env(actions)
        ah = self._env_action_horizon()
        ad = self._env_action_dim()
        return flat.reshape(flat.shape[0], ah, ad)[:, 0, :]

    def _as_critic_actions(self, actions: jnp.ndarray) -> jnp.ndarray:
        """Alias for :meth:`_clip_actions_to_env` (full env chunk, not first step only)."""
        return self._clip_actions_to_env(actions)

    def _as_openpi_actions(self, actions: jnp.ndarray) -> jnp.ndarray:
        """Convert replay/env action tensors to OpenPI action shape ``[B, H, D]``."""
        x = jnp.asarray(actions, dtype=jnp.float32)
        ah = int(self.config.get("vla_action_horizon", 10))
        ad = int(self.config.get("vla_action_dim", 4))
        if x.ndim == 3:
            actions = x[:, :ah, :ad]
        elif x.ndim == 2 and x.shape[-1] == ah * ad:
            actions = x.reshape(x.shape[0], ah, ad)
        else:
            raise ValueError(
                f"Cannot map actions shape {tuple(x.shape)} to OpenPI actions [B,{ah},{ad}]."
            )
        
        # Then we need to expand the actions to the dimensions of the base pi05 model
        # Pad the action dimension (last dim) to get (1,10,32) with zeros
        pad_dim = PI05_ACTION_DIM - actions.shape[-1]
        if pad_dim > 0:
            pad_widths = [(0, 0)] * (actions.ndim - 1) + [(0, pad_dim)]
            actions = jnp.pad(actions, pad_widths)
 
        return actions

    # ----- sampling ------------------------------------------------------- #

    def _encode_obs(self, params, observations):
        return self.network.select("obs_encoder")(observations, params=params)

    def _flow_sample(self, params, observations, noises):
        """Euler integrate the BC flow from ``noises`` (t=0) to ``actions`` (t=1)."""
        flow_steps = self.config["flow_steps"]
        observations = self._encode_obs(params, observations)
        action = noises
        for i in range(flow_steps):
            t = jnp.full(observations.shape[:-1] + (1,), i / flow_steps)
            v = self.network.select("flow")(observations, action, t, params=params)
            action = action + v / flow_steps
        return jnp.clip(action, -1.0, 1.0)

    @jax.jit
    def sample_actions(self, observations, seed=None, temperature=1.0):
        seed = seed if seed is not None else self.rng
        seed, sub = jax.random.split(seed)
        enc = self._encode_obs(self.network.params, observations)
        dist = self.network.select("noise_actor")(enc, temperature=temperature)
        noise = dist.sample(seed=sub)
        noise = noise * self.config["noise_scale"]
        actions = self._flow_sample(self.network.params, observations, noise)
        return actions

    def sample_actions_with_vla(self, observations, seed=None, temperature=1.0):
        """Rollout path only: when ``vla_sample_fn`` is set, map noise through VLA instead of BC flow."""
        if self.vla_sample_fn is None:
            return self.sample_actions(observations, seed=seed, temperature=temperature)
        seed = seed if seed is not None else self.rng
        seed, sub = jax.random.split(seed)
        obs_mode = str(self.config.get("observation_mode", "state"))
        if obs_mode == "policy_embed":
            if self.steervla_actor is None:
                raise RuntimeError("observation_mode='policy_embed' requires an attached SteerVLAActor.")
            embed = self.steervla_actor.ensure_policy_embedding(int(observations.shape[0]))
            enc = jnp.asarray(embed, dtype=jnp.float32)
        else:
            enc = self._encode_obs(self.network.params, observations)
        dist = self.network.select("noise_actor")(enc, temperature=temperature)
        noise = dist.sample(seed=sub)
        noise = noise * self.config["noise_scale"]
        return jnp.asarray(self.vla_sample_fn(observations, noise))

    @jax.jit
    def sample_actions_dagger(self, observations):
        """Deterministic rollout policy for DAgger: BC flow from zero noise."""
        batch = observations.shape[0]
        noise = jnp.zeros((batch, self._flat_env_action_dim()), dtype=jnp.float32)
        return self._flow_sample(self.network.params, observations, noise)

    # ----- losses --------------------------------------------------------- #

    # Noise critic loss (distill Q(s, z) toward Q(s, a_env) at the same state).
    def noise_critic_loss(self, batch, grad_params, rng):
        obs_e = self._encode_obs(grad_params, batch["observations"])
        dist = self.network.select("noise_actor")(obs_e)
        noise_flat = self._as_flat_noise(dist.sample(seed=rng) * self.config["noise_scale"])
        env_actions = self._clip_actions_to_env(batch["actions"])
        target_qs = self.network.select("critic")(
            _critic_obs_e(obs_e, batch, "language_label"), env_actions,
        )
        target_q = jnp.min(target_qs, axis=0)

        qs = self.network.select("noise_critic")(
            _critic_obs_e(obs_e, batch, "language_label"), noise_flat, params=grad_params
        )
        critic_loss = jnp.square(qs - target_q[None]).mean()
        return critic_loss, {
            "critic_loss": critic_loss,
            "q_mean": qs.mean(),
            "q_max": qs.max(),
            "q_min": qs.min(),
            "target_q": target_q.mean(),
        }
        
    def flow_loss(self, batch, grad_params, rng):
        actions = self._clip_actions_to_env(batch["actions"])
        x_0 = jax.random.normal(rng, actions.shape) * self.config["noise_scale"]
        rng_t = jax.random.fold_in(rng, 1)
        t = jax.random.uniform(rng_t, actions.shape[:-1] + (1,))
        x_t = (1.0 - t) * x_0 + t * actions
        target_v = actions - x_0
        obs_e = self._encode_obs(grad_params, batch["observations"])
        pred_v = self.network.select("flow")(obs_e, x_t, t, params=grad_params)
        loss = jnp.square(pred_v - target_v).mean()
        return loss, {"flow_loss": loss}

    def actor_loss(self, batch, grad_params, rng):
        obs_e = self._encode_obs(grad_params, batch["observations"])
        dist = self.network.select("noise_actor")(obs_e, params=grad_params)
        noise, log_prob = dist.sample_and_log_prob(seed=rng)
        noise = noise * self.config["noise_scale"]
        noise_flat = self._as_flat_noise(noise)
        qs = self.network.select("noise_critic")(
            _critic_obs_e(obs_e, batch, "language_label"), noise_flat,
        )
        q = jnp.min(qs, axis=0)
        actor_loss = (self.config["alpha"] * log_prob - q).mean()
        return actor_loss, {
            "actor_loss": actor_loss,
            "noise_log_prob": log_prob.mean(),
            "q_for_actor": q.mean(),
        }
        
    def _vla_forward(self, observations, openpi_observations, rng, noise=None):
        rng_n, rng_act = jax.random.split(rng)
        if noise is None:
            noise_scale = jnp.asarray(self.config["noise_scale"], dtype=jnp.float32)
            noise = _vla_forward_prepare_actor_noise(
                self.network, observations, rng_n, noise_scale,
            )
        noise_chunk = self._as_noise_chunk(noise)
        next_openpi_observation = self._as_jax_pytree(openpi_observations)
        sample_actions_time = time.time()
        next_actions = self.steervla_actor._sample_actions(
            rng_act,
            next_openpi_observation,
            noise=noise_chunk,
            image_keys=tuple(self.config.get("image_keys", ("base_0_rgb",))),
            num_steps=int(
                self.config.get(
                    "vla_update_flow_steps",
                    self.config.get("flow_steps", 10),
                )
            ),
        )
        next_actions = self.steervla_actor.postprocess_sampled_trajectory(
            next_actions,
            observation_state=next_openpi_observation.state,
        )
        jax.block_until_ready(next_actions)
        sample_actions_time = time.time() - sample_actions_time
        print(f"[DEBUG - dsrl] Sample actions time: {sample_actions_time} seconds")
        return jax.lax.stop_gradient(self._clip_actions_to_env(next_actions))

    # def _vla_forward_for_actor(self, batch, rng):
    #     rng_a, rng_act = jax.random.split(rng)
    #     noise_scale = jnp.asarray(self.config["noise_scale"], dtype=jnp.float32)
    #     noise = _vla_forward_prepare_actor_noise(
    #         self.network, batch["observations"], rng_a, noise_scale,
    #     )
    #     noise_flat = self._as_flat_noise(noise)
    #     openpi_observation = self._as_jax_pytree(batch["openpi_observation"])

    #     actions = self.steervla_actor._sample_actions(
    #         rng_act,
    #         openpi_observation,
    #         noise=self._as_noise_chunk(noise_flat),
    #         image_keys=tuple(self.config.get("image_keys", ("base_0_rgb",))),
    #         num_steps=int(self.config.get("flow_steps", 10)),
    #     )
    #     return self._clip_actions_to_env(actions), jax.lax.stop_gradient(noise_flat)

    def noise_critic_loss_vla(self, batch, grad_params, rng):
        """Distill ``noise_critic(s, z)`` toward ``critic(s, VLA(s, z))``."""
        rng, noise_rng, vla_rng = jax.random.split(rng, 3)
        batch_size = batch["observations"].shape[0]
        noise_scale = jnp.asarray(self.config["noise_scale"], dtype=jnp.float32)
        noise_actions = (
            jax.random.normal(noise_rng, (batch_size, self._flat_noise_dim()))
            * noise_scale
        )
        vla_actions = self._vla_forward(
            batch["observations"], batch["openpi_observation"], vla_rng, noise=noise_actions,
        )
        return _noise_critic_loss_vla_pure_math(
            self.network, batch, vla_actions, noise_actions, grad_params,
        )

    def noise_actor_loss_vla(self, batch, grad_params, rng):
        """SAC-style noise actor loss using ``noise_critic`` Q-values."""
        alpha = jnp.asarray(self.config["alpha"], dtype=jnp.float32)
        noise_scale = jnp.asarray(self.config["noise_scale"], dtype=jnp.float32)
        return _noise_actor_loss_vla_pure_math(
            self.network, batch, grad_params, rng, alpha, noise_scale,
        )

    def critic_loss_vla(self, batch, grad_params, rng):
        """Critic loss with bootstrap target from a frozen VLA action sample.

        VLA forward (CoT + flow-matching) is executed eagerly; only the surrounding DSRL math
        (target-Q, MSE) is jitted via :func:`_critic_loss_vla_pure_math`.
        """
        next_actions = self._vla_forward(
            batch["next_observations"], batch["next_openpi_observation"], rng,
        )
        critic_actions = self._clip_actions_to_env(batch["actions"])
        discount = jnp.asarray(self.config["discount"], dtype=jnp.float32)
        return _critic_loss_vla_pure_math(
            self.network, batch, next_actions, critic_actions, grad_params, discount,
        )

    def total_loss_vla(self, batch, grad_params, rng=None, vla_cache=None):
        """Sum of VLA-path losses; ``update_with_vla`` passes precomputed ``vla_cache``."""
        rng = rng if rng is not None else self.rng
        rng, nc_rng, na_rng, c_rng = jax.random.split(rng, 4)
        discount = jnp.asarray(self.config["discount"], dtype=jnp.float32)
        alpha = jnp.asarray(self.config["alpha"], dtype=jnp.float32)
        noise_scale = jnp.asarray(self.config["noise_scale"], dtype=jnp.float32)
        critic_actions = self._clip_actions_to_env(batch["actions"])

        noise_total_loss_vla_time = time.time()
        if vla_cache is not None:
            noise_critic_loss_vla_time = time.time()
            nc_loss, nc_info = _noise_critic_loss_vla_pure_math(
                self.network,
                batch,
                vla_cache["vla_actions_nc"],
                vla_cache["noise_actions"],
                grad_params,
            )
            na_loss, na_info = _noise_actor_loss_vla_pure_math(
                self.network, batch, grad_params, na_rng, alpha, noise_scale,
            )
            c_loss, c_info = _critic_loss_vla_pure_math(
                self.network,
                batch,
                vla_cache["next_actions_critic"],
                critic_actions,
                grad_params,
                discount,
            )
            noise_critic_loss_vla_time = time.time() - noise_critic_loss_vla_time
            print(f"[DEBUG - dsrl] Noise critic loss vla time: {noise_critic_loss_vla_time} seconds")
        else:
            noise_critic_loss_vla_time = time.time()
            nc_loss, nc_info = self.noise_critic_loss_vla(batch, grad_params, nc_rng)
            na_loss, na_info = self.noise_actor_loss_vla(batch, grad_params, na_rng)
            c_loss, c_info = self.critic_loss_vla(batch, grad_params, c_rng)
            noise_critic_loss_vla_time = time.time() - noise_critic_loss_vla_time
            print(f"[DEBUG - dsrl] Noise critic loss vla time: {noise_critic_loss_vla_time} seconds")
        total_loss_vla_time = time.time() - noise_total_loss_vla_time
        print(f"[DEBUG - dsrl] Total loss vla time: {total_loss_vla_time} seconds")
        info = {}
        for k, v in nc_info.items():
            info[f"noise_critic_vla/{k}"] = v
        for k, v in na_info.items():
            info[f"noise_actor_vla/{k}"] = v
        for k, v in c_info.items():
            info[f"critic/{k}"] = v
        combined = nc_loss + na_loss + c_loss
        info["total_loss_vla"] = combined
        info["flax_loss_vla"] = nc_loss + na_loss + c_loss
        info["noise_critic_loss_vla"] = nc_loss
        info["noise_actor_loss_vla"] = na_loss
        info["critic_loss_vla"] = c_loss
        return combined, info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng
        rng, nc_rng, a_rng = jax.random.split(rng, 3)

        nc_loss, nc_info = self.noise_critic_loss(batch, grad_params, nc_rng)
        for k, v in nc_info.items():
            info[f"noise_critic/{k}"] = v
        a_loss, a_info = self.actor_loss(batch, grad_params, a_rng)
        for k, v in a_info.items():
            info[f"noise_actor/{k}"] = v

        return nc_loss + a_loss, info

    def target_update(self, network):
        new_target = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["tau"] + tp * (1 - self.config["tau"]),
            network.params["modules_critic"],
            network.params["modules_target_critic"],
        )
        network.params["modules_target_critic"] = new_target

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network)
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update_dagger(self, batch):
        """Online imitation update for DAgger: optimize only the BC flow loss."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            loss, info = self.flow_loss(batch, grad_params, rng)
            return loss, {f"dagger/{k}": v for k, v in info.items()}

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(network=new_network, rng=new_rng), info

    def _prepare_vla_batch(self, batch):
        """Attach ``openpi_observation`` / ``next_openpi_observation`` to a replay batch."""
        sv = _steervla()
        if "openpi_state" in batch and "next_openpi_state" in batch:
            openpi_observation = sv.openpi_observation_from_replay_batch(batch)
            next_openpi_observation = sv.openpi_observation_from_replay_batch(
                batch, prefix="next_",
            )
            if self.steervla_actor is not None:
                openpi_observation = self.steervla_actor.attach_replay_tokens(
                    openpi_observation, batch,
                )
                next_openpi_observation = self.steervla_actor.attach_replay_tokens(
                    next_openpi_observation, batch, prefix="next_",
                )
        else:
            raw_obs = None
            raw_next = None
            if self.steervla_actor is not None and getattr(self.steervla_actor, "raw_obs_holder", None) is not None:
                raw_obs = self.steervla_actor.raw_obs_holder.get("obs")
                raw_next = self.steervla_actor.raw_obs_holder.get("next_obs")
            openpi_observation = self.steervla_actor.build_observation_batch_numpy(
                batch_size=batch["observations"].shape[0], raw=raw_obs,
            )
            next_openpi_observation = self.steervla_actor.build_observation_batch_numpy(
                batch_size=batch["next_observations"].shape[0], raw=raw_next,
            )
            openpi_observation = sv.with_replay_cot_tokens(openpi_observation, batch)
            next_openpi_observation = sv.with_replay_cot_tokens(
                next_openpi_observation, batch, prefix="next_",
            )
            if self.steervla_actor is not None:
                openpi_observation = self.steervla_actor.attach_replay_tokens(
                    openpi_observation, batch,
                )
                next_openpi_observation = self.steervla_actor.attach_replay_tokens(
                    next_openpi_observation, batch, prefix="next_",
                )
        batch = dict(batch)
        batch["openpi_observation"] = self._as_jax_pytree(openpi_observation)
        batch["next_openpi_observation"] = self._as_jax_pytree(next_openpi_observation)
        return batch

    def _precompute_vla_loss_cache(self, batch, rng):
        """Eager VLA forwards consumed by :meth:`total_loss_vla` inside ``update_with_vla``."""
        rng, nc_rng, c_rng = jax.random.split(rng, 3)
        batch_size = batch["observations"].shape[0]
        noise_scale = jnp.asarray(self.config["noise_scale"], dtype=jnp.float32)

        next_actions = self._vla_forward(
            batch["next_observations"], batch["next_openpi_observation"], c_rng,
        )

        nc_rng, vla_rng = jax.random.split(nc_rng)
        noise_actions = (
            jax.random.normal(nc_rng, (batch_size, self._flat_noise_dim()))
            * noise_scale
        )
        vla_actions_nc = self._vla_forward(
            batch["observations"], batch["openpi_observation"], vla_rng, noise=noise_actions,
        )

        return {
            "next_actions_critic": next_actions,
            "noise_actions": noise_actions,
            "vla_actions_nc": vla_actions_nc,
        }

    # ----- residual SAC helpers ------------------------------------------- #

    def _critic_uses_pi_prefix_features(self) -> bool:
        return bool(self.config.get("critic_use_pi_prefix_features", False))

    def _residual_uses_pi_image_features(self) -> bool:
        return bool(self.config.get("residual_use_pi_image_features", False))

    def _residual_append_state(self) -> bool:
        """When True, the raw CARLA state vector is appended to the residual actor obs_e."""
        return bool(self.config.get("residual_append_state", False))

    def _residual_append_base_action(self) -> bool:
        """When True, the base Pi0 action is appended to the residual actor obs_e (before state)."""
        return bool(self.config.get("residual_append_base_action", False))

    def _residual_uses_accel_steer(self) -> bool:
        """True when the residual acts on the 2-D PID-decoded [accel, steer] action
        (torch ``residual_sac.py`` parity) instead of the raw waypoint chunk."""
        return str(self.config.get("residual_action_space", "waypoint_chunk")).strip().lower() == "accel_steer"

    def _residual_env_actions(self, actions: jnp.ndarray) -> jnp.ndarray:
        """Map stored residual-mode actions to the critic layout.

        In ``accel_steer`` mode actions are already 2-D and must NOT go through
        :meth:`_clip_actions_to_env` (which expects the waypoint-chunk layout).
        """
        if self._residual_uses_accel_steer():
            return jnp.asarray(actions, dtype=jnp.float32)
        return self._clip_actions_to_env(actions)

    def _get_live_state(self, batch_size: int) -> jnp.ndarray:
        """Read the current CARLA state vector from ``raw_obs_holder`` (rollout path only)."""
        import numpy as _np
        raw = None
        if self.steervla_actor is not None and getattr(self.steervla_actor, "raw_obs_holder", None) is not None:
            raw = self.steervla_actor.raw_obs_holder.get("obs")
        if raw is not None and isinstance(raw, dict) and "state" in raw:
            st = _np.asarray(raw["state"], dtype=_np.float32)[6:]
            return jnp.asarray(_np.broadcast_to(st, (batch_size, st.shape[0])))
        state_dim = int(self.config.get("residual_obs_dim", 25))
        return jnp.zeros((batch_size, state_dim), dtype=jnp.float32)

    def _residual_pi_feature_source(self) -> str:
        return str(self.config.get("residual_pi_feature_source", "prefix")).strip().lower()

    def _residual_uses_pi_prefix_features(self) -> bool:
        return self._residual_uses_pi_image_features() and self._residual_pi_feature_source() == "prefix"

    def _prepare_pi_prefix_feature_batch(self, batch):
        """Attach frozen Pi prefix embeddings once so critic/residual can share them."""
        if not (self._critic_uses_pi_prefix_features() or self._residual_uses_pi_prefix_features()):
            return batch
        if self.steervla_actor is None:
            raise RuntimeError("Pi prefix features require an attached steervla_actor.")
        if "openpi_observation" not in batch or "next_openpi_observation" not in batch:
            batch = self._prepare_vla_batch(batch)
        if "pi_prefix_obs_e" in batch and "pi_prefix_next_obs_e" in batch:
            return batch
        batch = dict(batch)
        batch["pi_prefix_obs_e"] = jax.lax.stop_gradient(
            self.steervla_actor.encode_prefix_features(batch["openpi_observation"])
        )
        batch["pi_prefix_next_obs_e"] = jax.lax.stop_gradient(
            self.steervla_actor.encode_prefix_features(batch["next_openpi_observation"])
        )
        return batch

    def _prepare_critic_feature_batch(self, batch):
        """Attach frozen Pi prefix embeddings for critic/noise_critic when enabled."""
        if not self._critic_uses_pi_prefix_features():
            return batch
        batch = self._prepare_pi_prefix_feature_batch(batch)
        batch = dict(batch)
        batch["critic_obs_e"] = batch["pi_prefix_obs_e"]
        batch["critic_next_obs_e"] = batch["pi_prefix_next_obs_e"]
        return batch

    def _update_critic_only(self, batch, next_actions, next_log_pi=None, alpha=None):
        """Critic-only update for residual SAC / shared VLA-backed actor paths.

        ``next_actions`` is the bootstrap action a' at s' (env-layout flat); it must
        be computed eagerly outside this jitted core because it may come from a
        heavy VLA-backed policy path that is not jit-friendly.

        ``next_log_pi`` and ``alpha`` are optional; when provided (residual SAC path)
        the soft Bellman target includes the entropy bonus.
        """
        new_rng, rng = jax.random.split(self.rng)
        critic_actions = self._residual_env_actions(batch["actions"])
        discount = jnp.asarray(self.config["discount"], dtype=jnp.float32)

        def loss_fn(grad_params):
            loss, info = _critic_loss_vla_pure_math(
                self.network, batch, next_actions, critic_actions, grad_params, discount,
                next_log_pi=next_log_pi, alpha=alpha,
            )
            return loss, {f"critic/{k}": v for k, v in info.items()}

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network)
        return self.replace(network=new_network, rng=new_rng), info

    # ----- sac_residual --------------------------------------------------- #

    def attach_sac_residual(self, sac_residual_agent):
        """Attach a :class:`jax_agents.sac_residual.SACResidualAgent` sub-agent."""
        return self.replace(sac_residual_agent=sac_residual_agent)

    def _critic_obs_e_with_lang(self, obs_e, lang):
        """Append ``lang`` to ``obs_e`` if present (matches ``_critic_obs_e``)."""
        if lang is None:
            return obs_e
        return jnp.concatenate([obs_e, jnp.asarray(lang, dtype=jnp.float32)], axis=-1)

    def _residual_obs_features(
        self,
        observations,
        openpi_observation=None,
        base_action=None,
        precomputed_prefix_features=None,
        residual_obs=None,
    ):
        """Feature source for the residual actor.

        Returns DSRL's ``obs_encoder`` output (or a frozen Pi feature when
        ``residual_use_pi_image_features=True``).  When ``residual_append_state=True``
        the raw CARLA state vector (25-dim) is concatenated to the end, giving the
        residual MLP explicit access to ego kinematics alongside the semantic VLM
        features.  ``residual_obs`` carries the pre-fetched state at training time
        (from ``batch["residual_obs"]``); at rollout it is read live from
        ``steervla_actor.raw_obs_holder["obs"]["state"]``.
        """
        if not self._residual_uses_pi_image_features():
            obs_e = self.network.select("obs_encoder")(observations)
        else:
            if self.steervla_actor is None:
                raise RuntimeError("Pi residual features require an attached steervla_actor.")
            if openpi_observation is None:
                raw_obs = None
                if getattr(self.steervla_actor, "raw_obs_holder", None) is not None:
                    raw_obs = self.steervla_actor.raw_obs_holder.get("obs")
                openpi_observation = self.steervla_actor.build_observation_batch_numpy(
                    batch_size=observations.shape[0],
                    raw=raw_obs,
                )
            source = self._residual_pi_feature_source()
            if source == "prefix":
                if precomputed_prefix_features is not None:
                    obs_e = jnp.asarray(precomputed_prefix_features, dtype=jnp.float32)
                else:
                    obs_e = self.steervla_actor.encode_prefix_features(openpi_observation)
            elif source == "suffix":
                if base_action is None:
                    raise RuntimeError("Pi suffix residual features require base_action.")
                obs_e = self.steervla_actor.encode_suffix_features(openpi_observation, base_action)
            else:
                raise ValueError(
                    f"Unsupported residual_pi_feature_source={source!r}; expected 'prefix' or 'suffix'."
                )

        if self._residual_append_base_action() and base_action is not None:
            obs_e = jnp.concatenate([obs_e, jnp.asarray(base_action, dtype=jnp.float32)], axis=-1)

        if self._residual_append_state():
            if residual_obs is not None:
                state = jnp.asarray(residual_obs, dtype=jnp.float32)
            else:
                state = self._get_live_state(int(observations.shape[0]))
            obs_e = jnp.concatenate([obs_e, state], axis=-1)

        return obs_e

    def sample_actions_sac_residual(self, observations, seed=None, temperature=1.0, base_action=None):
        """Rollout for SAC-residual / DAgger-residual: ``base + residual * scale``.

        Returns ``(action, base_action)`` so the caller can store the base in
        the replay buffer alongside the executed action.

        ``base_action``: optional precomputed base (e.g. the 2-D PID-decoded
        [accel, steer] in ``accel_steer`` mode, where the caller runs
        ``vla_sample_fn`` + waypoint decode itself). When omitted, the raw
        Pi0 waypoint chunk is sampled here.
        """
        if self.sac_residual_agent is None:
            raise RuntimeError(
                "sample_actions_sac_residual requires an attached sac_residual_agent."
            )
        seed = seed if seed is not None else self.rng
        seed_b, seed_r = jax.random.split(seed)
        if base_action is not None:
            base_action = jnp.asarray(base_action, dtype=jnp.float32)
        else:
            if self.vla_sample_fn is None:
                raise RuntimeError(
                    "sample_actions_sac_residual requires a SteerVLA vla_sample_fn."
                )
            noise = jax.random.normal(seed_b, (observations.shape[0], self._flat_noise_dim()))
            base_action = jnp.asarray(self.vla_sample_fn(observations, noise))
            base_action = self._clip_actions_to_env(base_action)
        # Get obs_e for the residual actor.  For policy_embed mode, use the fresh embedding
        # stashed by vla_sample_fn so it matches what main_carla.py stores in the buffer.
        # The state append (when residual_append_state=True) reads from raw_obs_holder, which
        # is already pointing at the current step's obs (set by main_carla.py before this call).
        if (
            not self._residual_uses_pi_image_features()
            and str(self.config.get("observation_mode", "state")) == "policy_embed"
            and self.steervla_actor is not None
        ):
            embed = self.steervla_actor.ensure_policy_embedding(int(observations.shape[0]))
            obs_e = jnp.asarray(embed, dtype=jnp.float32)
            if self._residual_append_base_action():
                obs_e = jnp.concatenate([obs_e, base_action], axis=-1)
            if self._residual_append_state():
                obs_e = jnp.concatenate([obs_e, self._get_live_state(int(observations.shape[0]))], axis=-1)
        else:
            obs_e = self._residual_obs_features(observations, base_action=base_action)
        action, _residual = self.sac_residual_agent.sample_actions_residual(
            obs_e,
            base_action,
            seed=seed_r,
            temperature=temperature,
        )
        return action, base_action

    def update_sac_residual(self, batch):
        """SAC update with a residual actor on top of the frozen Pi0 base.

        Steps:

        1. Compute ``base_next = Pi0(s')`` eagerly (or use cached ``base_next_actions``).
        2. Compute ``next_action = clip(base_next + residual(obs_e', base_next) * scale, -1, 1)``.
        3. Update the DSRL critic with TD target bootstrapped from ``next_action``.
        4. Update the residual MLP by maximizing ``Q(s, base_stored + residual(obs_e, base_stored) * scale)``.

        Requires ``batch`` to contain ``base_actions`` — the base Pi0 action that
        was used during rollout (stored separately by ``main_carla.py``).
        """
        new_rng, rng = jax.random.split(self.rng)
        if self.sac_residual_agent is None:
            return self.replace(rng=new_rng), {"sac_residual/skipped_no_subagent": 1.0}
        if "base_actions" not in batch:
            return self.replace(rng=new_rng), {"sac_residual/skipped_no_base_actions": 1.0}

        # Only prepare VLA batch when Pi prefix features are needed or a live VLA
        # forward is required for the bootstrap target.
        _needs_vla_batch = (
            self._critic_uses_pi_prefix_features()
            or self._residual_uses_pi_prefix_features()
            or "base_next_actions" not in batch
        )
        if _needs_vla_batch and self.steervla_actor is not None:
            batch = self._prepare_vla_batch(batch)
        batch = self._prepare_pi_prefix_feature_batch(batch)
        if self._critic_uses_pi_prefix_features():
            batch = self._prepare_critic_feature_batch(batch)

        rng_base, rng_res = jax.random.split(rng)

        # 1-2. Bootstrap action + log_prob for the critic TD target.
        # Use pre-stored base_next_actions from the replay buffer when available (zero
        # Pi0 overhead at training time); fall back to a live _vla_forward if not cached.
        if "base_next_actions" in batch:
            base_next = jax.lax.stop_gradient(
                self._residual_env_actions(jnp.asarray(batch["base_next_actions"], dtype=jnp.float32))
            )
        else:
            if self._residual_uses_accel_steer():
                raise RuntimeError(
                    "accel_steer residual mode requires base_next_actions in the replay "
                    "buffer (the PID decode is rollout-side only)."
                )
            base_next = self._vla_forward(
                batch["next_observations"], batch["next_openpi_observation"], rng_base,
            )
        next_obs_e_sg = jax.lax.stop_gradient(
            self._residual_obs_features(
                batch["next_observations"],
                batch.get("next_openpi_observation"),
                base_next,
                precomputed_prefix_features=batch.get("pi_prefix_next_obs_e"),
                residual_obs=batch.get("next_residual_obs"),
            )
        )
        next_action, _, next_log_pi = self.sac_residual_agent.sample_actions_and_log_prob_residual(
            next_obs_e_sg, base_next, seed=rng_res,
        )
        next_action = jax.lax.stop_gradient(self._residual_env_actions(next_action))
        next_log_pi = jax.lax.stop_gradient(next_log_pi)
        alpha = jnp.asarray(self.sac_residual_agent._alpha(), dtype=jnp.float32)

        # 3. Critic update with soft Bellman target (subtracts entropy bonus).
        new_self, critic_info = self._update_critic_only(batch, next_action, next_log_pi=next_log_pi, alpha=alpha)

        # 4. Residual actor update: maximize Q(s, base + residual * scale).
        base_action = new_self._residual_env_actions(
            jnp.asarray(batch["base_actions"], dtype=jnp.float32)
        )
        obs_e_sg = jax.lax.stop_gradient(
            new_self._residual_obs_features(
                batch["observations"],
                batch.get("openpi_observation"),
                base_action,
                precomputed_prefix_features=batch.get("pi_prefix_obs_e"),
                residual_obs=batch.get("residual_obs"),
            )
        )
        if "critic_obs_e" in batch:
            critic_encoder_obs_e_sg = jax.lax.stop_gradient(
                jnp.asarray(batch["critic_obs_e"], dtype=jnp.float32)
            )
        else:
            critic_encoder_obs_e_sg = jax.lax.stop_gradient(
                new_self.network.select("obs_encoder")(batch["observations"])
            )
        critic_obs_e_sg = new_self._critic_obs_e_with_lang(
            critic_encoder_obs_e_sg, batch.get("language_label")
        )
        if "base_actions" in batch:
            critic_obs_e_sg = jnp.concatenate(
                [critic_obs_e_sg, jax.lax.stop_gradient(jnp.asarray(batch["base_actions"], dtype=jnp.float32))],
                axis=-1,
            )
        new_residual, residual_info = new_self.sac_residual_agent.update_actor(
            obs_e_sg=obs_e_sg,
            base_action=base_action,
            critic_obs_e_sg=critic_obs_e_sg,
            dsrl_network=new_self.network,
        )
        new_self = new_self.replace(sac_residual_agent=new_residual, rng=new_rng)
        return new_self, {
            **critic_info,
            **{f"sac_residual/{k}": v for k, v in residual_info.items()},
        }

    def update_dagger_residual(self, batch):
        """DAgger update for the residual actor: MSE toward the expert.

        No critic update — only MSE between ``base + residual * scale`` and the
        expert action stored as ``batch["actions"]``. The base Pi0 action lives in
        ``batch["base_actions"]``.

        Config flag ``dagger_residual_train_obs_encoder`` (default False): when True,
        the MSE loss also drives DSRL's ``obs_encoder`` so the image CNN learns
        features useful for predicting the residual.
        """
        new_rng, _ = jax.random.split(self.rng)
        if self.sac_residual_agent is None:
            return self.replace(rng=new_rng), {"dagger_residual/skipped_no_subagent": 1.0}
        if "base_actions" not in batch:
            return self.replace(rng=new_rng), {"dagger_residual/skipped_no_base_actions": 1.0}

        if self._residual_uses_pi_image_features():
            if bool(self.config.get("dagger_residual_train_obs_encoder", False)):
                raise ValueError(
                    "dagger_residual_train_obs_encoder=True is incompatible with "
                    "residual_use_pi_image_features=True."
                )
            if self.steervla_actor is not None:
                batch = self._prepare_vla_batch(batch)

        base_action = self._residual_env_actions(
            jnp.asarray(batch["base_actions"], dtype=jnp.float32)
        )
        expert_action = self._residual_env_actions(
            jnp.asarray(batch["actions"], dtype=jnp.float32)
        )

        if bool(self.config.get("dagger_residual_train_obs_encoder", False)):
            from jax_agents.sac_residual import _joint_dagger_apply_step

            scale = jnp.asarray(
                self.sac_residual_agent._scale(), dtype=jnp.float32,
            )
            clip_bound = jnp.asarray(
                float(self.sac_residual_agent._clip_bound()), dtype=jnp.float32,
            )
            new_dsrl_network, new_residual_network, info = _joint_dagger_apply_step(
                self.network,
                self.sac_residual_agent.network,
                batch["observations"],
                base_action,
                expert_action,
                scale,
                clip_bound,
            )
            new_residual = self.sac_residual_agent.replace(network=new_residual_network)
            new_self = self.replace(
                network=new_dsrl_network,
                sac_residual_agent=new_residual,
                rng=new_rng,
            )
            return new_self, {f"dagger_residual/{k}": v for k, v in info.items()}

        obs_e_sg = jax.lax.stop_gradient(
            self._residual_obs_features(
                batch["observations"],
                batch.get("openpi_observation"),
                base_action,
                precomputed_prefix_features=batch.get("pi_prefix_obs_e"),
                residual_obs=batch.get("residual_obs"),
            )
        )
        new_residual, residual_info = self.sac_residual_agent.update_actor_dagger(
            obs_e_sg=obs_e_sg,
            base_action=base_action,
            expert_action=expert_action,
        )
        new_self = self.replace(sac_residual_agent=new_residual, rng=new_rng)
        return new_self, {f"dagger_residual/{k}": v for k, v in residual_info.items()}

    def update_with_vla(self, batch):
        """Flax update via :meth:`total_loss_vla` with eager VLA forwards and a jitted gradient core."""
        new_rng, rng = jax.random.split(self.rng)
        batch = self._prepare_vla_batch(batch)
        vla_cache = self._precompute_vla_loss_cache(batch, rng)

        def loss_fn(grad_params):
            return self.total_loss_vla(batch, grad_params, rng=rng, vla_cache=vla_cache)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network)
        return self.replace(network=new_network, rng=new_rng), info

    # ----- construction --------------------------------------------------- #

    @classmethod
    def create(
        cls,
        seed: int,
        ex_observations,
        ex_actions,
        config,
        vla_sample_fn: Optional[Callable] = None,
        openpi_train_config: Any = None,
        steervla_actor: Any = None,
    ):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        obs_mode = str(config.get("observation_mode", "state"))
        if obs_mode not in ("state", "image", "policy_embed"):
            raise ValueError(
                f"observation_mode must be 'state', 'image', or 'policy_embed', got {obs_mode!r}"
            )
        image_encoder = str(config.get("image_encoder", "impala")).lower()

        if vla_sample_fn is not None:
            env_action_dim = int(config.get("vla_action_dim", 4)) * int(
                config.get("vla_action_horizon", config.get("action_horizon", 10))
            )
            noise_action_dim = int(config["actor_action_dim"]) * int(config["action_horizon"])
        else:
            env_action_dim = ex_actions.shape[-1]
            noise_action_dim = env_action_dim
        if (
            str(config.get("online_training_mode", "")) in ("sac_residual", "dagger_residual")
            and str(config.get("residual_action_space", "waypoint_chunk")) == "accel_steer"
        ):
            # Residual acts on the 2-D PID-decoded [accel, steer]; the replay buffer
            # and critic see 2-D actions, not the waypoint chunk.
            env_action_dim = 2
        obs_encoder_def = CarlaObservationEncoder(
            observation_mode=obs_mode,
            image_encoder=image_encoder,
            impala_width=int(config.get("image_impala_width", 1)),
            impala_stack_sizes=tuple(config.get("image_impala_stack_sizes", (16, 32, 32))),
            impala_num_blocks=int(config.get("image_impala_num_blocks", 2)),
            image_mlp_hidden_dims=tuple(config.get("image_mlp_hidden_dims", (512,))),
            layer_norm=config["layer_norm"],
        )
        if obs_mode in ("state", "policy_embed"):
            embed_dim = int(ex_observations.shape[-1])
        elif image_encoder == "siglip":
            embed_dim = int(ex_observations.shape[-1])  # precomputed SigLIP embedding
        else:
            embed_dim = int(tuple(config.get("image_mlp_hidden_dims", (512,)))[-1])
        critic_feedback_mode = str(config.get("critic_feedback_mode", "commentary_bow"))
        if critic_feedback_mode == "action_delta":
            lang_dim = int(config.get("critic_action_dim", 4))
        else:
            lang_dim = int(config.get("language_label_dim", 119))
        batch_shape = ex_observations.shape[:1]
        base_action_critic_dim = (
            env_action_dim if str(config.get("online_training_mode", "")) == "sac_residual" else 0
        )
        ex_embedded = jnp.zeros(batch_shape + (embed_dim,), dtype=jnp.float32)
        ex_critic_embedded = jnp.zeros(batch_shape + (embed_dim + lang_dim + base_action_critic_dim,), dtype=jnp.float32)
        ex_env_actions = jnp.zeros(batch_shape + (env_action_dim,), dtype=jnp.float32)
        ex_noise_actions = jnp.zeros(batch_shape + (noise_action_dim,), dtype=jnp.float32)
        ex_t = jnp.zeros(batch_shape + (1,), dtype=jnp.float32)

        # BC flow-matching head must stay a Flax module: init calls ``flow(enc, x_t, t)``.
        # OpenPI SteerVLA hooks only via ``vla_sample_fn`` (see ``sample_actions_with_vla``).
        flow_def = FlowActor(
            hidden_dims=tuple(config["flow_hidden_dims"]),
            action_dim=env_action_dim,
            layer_norm=config["layer_norm"],
        )
        actor_def = NoiseActor(
            hidden_dims=tuple(config["actor_hidden_dims"]),
            action_dim=noise_action_dim,
        )
        noise_critic_def = Critic(
            hidden_dims=tuple(config["critic_hidden_dims"]),
            layer_norm=config["layer_norm"],
            ensemble_size=config["critic_ensemble"],
        )
        critic_def = Critic(
            hidden_dims=tuple(config["critic_hidden_dims"]),
            layer_norm=config["layer_norm"],
            ensemble_size=config["critic_ensemble"],
        )
        networks = {
            "obs_encoder": (obs_encoder_def, (ex_observations,)),
            "flow": (flow_def, (ex_embedded, ex_env_actions, ex_t)),
            "noise_actor": (actor_def, (ex_embedded,)),
            "noise_critic": (noise_critic_def, (ex_critic_embedded, ex_noise_actions)),

            "critic": (critic_def, (ex_critic_embedded, ex_env_actions)),
            "target_critic": (
                copy.deepcopy(critic_def),
                (ex_critic_embedded, ex_env_actions),
            ),
        }
        defs = {k: v[0] for k, v in networks.items()}
        args = {k: v[1] for k, v in networks.items()}

        network_def = ModuleDict(defs)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)
        network.params["modules_target_critic"] = network.params["modules_critic"]

        return cls(
            rng=rng,
            network=network,
            config=flax.core.FrozenDict(**config),
            vla_sample_fn=vla_sample_fn,
            openpi_train_config=openpi_train_config,
            steervla_actor=steervla_actor,
        )


# --------------------------------------------------------------------------- #
# Config                                                                      #
# --------------------------------------------------------------------------- #


def get_config():
    """ml_collections.ConfigDict consumed by ``main_carla.py``."""

    config = ml_collections.ConfigDict(
        dict(
            agent_name="dsrl",
            lr=3e-4,
            batch_size=256,
            actor_hidden_dims=(256, 256),
            critic_hidden_dims=(256, 256),
            flow_hidden_dims=(256, 256),
            critic_ensemble=2,
            layer_norm=True,
            discount=0.99,
            tau=0.005,
            alpha=0.1,
            noise_scale=1.0,
            flow_steps=5,
            observation_mode="state",
            image_impala_width=1,
            image_impala_stack_sizes=(16, 32, 32),
            image_impala_num_blocks=2,
            image_mlp_hidden_dims=(512,),
            # Online-loop knobs (used by main_carla.py, not by the agent itself).
            # Env steps with policy rollouts but no RL updates (see main_carla.run_online_carla).
            warmup_steps=1000,
            updates_per_step=1,
            # If false, main_carla collects rollouts but skips RL gradient updates.
            enable_updates=True,
            buffer_capacity=100_000,
            # W&B: when ``observation_mode`` is ``image``, log ``rollout/curr_obs`` every N env steps.
            image_log_curr_interval=1000,
            # Pin JAX/XLA default GPU for RL (CARLA sim uses ``gpu_rank`` in carla_config.yaml).
            # ``-1`` = do not override (JAX picks as usual).
            training_gpu_rank=-1,
            # Ignored by DSRL but populated for symmetry with other agents.
            frame_stack=ml_collections.config_dict.placeholder(int),
            dataset_class="GCDataset",
            # ``flow_sample_with_vla`` / ``update_with_vla`` (Pi0-CoT denoise path).
            vla_cot_temperature=0.0,
            vla_cot_replay_reasoning=True,
            vla_sample_actions_num_steps=10,
            # Pi0 flow-matching denoise steps in ``update_with_vla`` (rollout uses ``steervla.sample_actions_num_steps``).
            vla_update_flow_steps=10,
            vla_sample_actions_low_memory=True,
            vla_sample_actions_jit_denoise_steps=False,
            vla_action_horizon=10,
            vla_action_dim=4,
            vla_output_action_format="DELTA_XY_T_DELTA_XY_SPACE",
            # Pi0 / noise-actor space (``actor_action_dim`` × ``action_horizon``).
            actor_action_dim=32,
            action_horizon=10,
            # Env rollout + action critic space (``vla_action_dim`` × ``vla_action_horizon``).
            # ``critic_action_dim`` is the per-step env dim (used by action-delta feedback).
            critic_action_dim=4,
            # Expert / delta language feedback appended to obs_e ONLY for the critic.
            # For commentary_bow this is 119 (= COMMENTARY_VOCAB).
            # For delta_commentary_bow it is auto-set in main_carla.py to the corrective BOW size.
            # Ignored when critic_feedback_mode="action_delta" (lang_dim = critic_action_dim instead).
            language_label_dim=119,
            # "commentary_bow": expert commentary BOW.
            # "action_delta": (critic_action_dim)-dim vector = expert first step − agent first step.
            # "delta_commentary_bow": corrective language BOW from expert-vs-agent action delta.
            critic_feedback_mode="commentary_bow",
            # Online training regime.
            # "rl": standard DSRL online RL updates.
            # "dagger": collect on-policy states, store expert actions, and train with flow imitation only.
            # "sac_residual": Pi0 frozen; small residual MLP trained via Q-gradient from DSRL critic.
            # "dagger_residual": Pi0 frozen; small residual MLP supervised via MSE toward expert action.
            online_training_mode="rl",
            # When True, critic and noise_critic consume frozen Pi prefix features
            # instead of DSRL.obs_encoder features.
            critic_use_pi_prefix_features=False,
            # When True, the residual actor consumes a frozen Pi feature instead of
            # DSRL's obs_encoder output.
            residual_use_pi_image_features=False,
            # "prefix": pooled frozen Pi prefix hidden states
            # "suffix": pooled frozen Pi action-suffix hidden states
            residual_pi_feature_source="prefix",
            # When ``online_training_mode="dagger_residual"`` and True, the DAgger MSE
            # gradient also updates DSRL's ``obs_encoder`` (image CNN).
            dagger_residual_train_obs_encoder=False,
            # Action space the residual acts on:
            # "waypoint_chunk": flat Pi0 chunk (physical units; env PID-decodes it).
            # "accel_steer":    2-D [accel, steer] in [-1, 1]; the waypoint chunk is
            #                   PID-decoded in the rollout process BEFORE the residual
            #                   (torch ``residual_sac.py`` / run_simlingo parity).
            residual_action_space="waypoint_chunk",
            # Env steps to run pure Pi0 (residual zeroed) before applying the residual MLP.
            residual_warmup_steps=500,
            # Residual actor hyperparameters.
            residual_actor_hidden_dims=(256, 256),
            residual_action_scale=0.3,
            # Composed-action clip bound (None = no clip; required for physical
            # DELTA_XY actions — see jax_agents/sac_residual.py docstring).
            residual_action_clip=None,
            residual_alpha=0.1,
            residual_lr=3e-4,
            residual_log_std_min=-5.0,
            residual_log_std_max=2.0,
            residual_layer_norm=False,
        )
    )
    return config
