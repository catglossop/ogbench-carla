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


def dsrl_encode_obs_sample_noise_logprob(
    network: TrainState,
    noise_scale: jnp.ndarray | float | int,
    observations: jnp.ndarray,
    rng: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Encode observations and sample noise + log-prob from the **noise actor** (DSRL).

    DSRL params are treated as constants (no gradients). Matches :meth:`DSRLAgent.actor_loss`
    sampling, without integrating the BC flow.
    """
    params = jax.tree.map(jax.lax.stop_gradient, network.params)
    obs_e = network.select("obs_encoder")(observations, params=params)
    dist = network.select("noise_actor")(obs_e, params=params)
    noise, log_prob = dist.sample_and_log_prob(seed=rng)
    ns = jnp.asarray(noise_scale, dtype=noise.dtype)
    return obs_e, noise * ns, log_prob


def dsrl_critic_min_q(network: TrainState, obs_e: jnp.ndarray, actions: jnp.ndarray, language_label=None) -> jnp.ndarray:
    """``min_k Q_k(obs_e, actions)`` with critic weights frozen; gradients flow through ``actions`` only."""
    params = jax.tree.map(jax.lax.stop_gradient, network.params)
    obs_e_sg = jax.lax.stop_gradient(obs_e)
    if language_label is not None:
        obs_e_sg = jnp.concatenate([obs_e_sg, jnp.asarray(language_label, dtype=jnp.float32)], axis=-1)
    qs = network.select("critic")(obs_e_sg, actions, params=params)
    return jnp.min(qs, axis=0)


def _critic_obs_e(obs_e: jnp.ndarray, batch: dict, key: str) -> jnp.ndarray:
    """Append language label from ``batch[key]`` to encoded observation for critic-only consumption."""
    lang = batch.get(key)
    if lang is None:
        return obs_e
    return jnp.concatenate([obs_e, jnp.asarray(lang, dtype=jnp.float32)], axis=-1)

@jax.jit
def _critic_loss_vla_pure_math(
    network: TrainState,
    batch: dict,
    next_actions_critic: jnp.ndarray,
    critic_actions: jnp.ndarray,
    grad_params,
    discount: jnp.ndarray,
):
    """Pure DSRL critic math: target-Q bootstrap + critic MSE, no VLA model access.

    ``next_actions_critic`` and ``critic_actions`` must already be projected to the
    critic action representation (e.g. via :meth:`DSRLAgent._as_critic_actions`).
    """
    next_obs_e = network.select("obs_encoder")(batch["next_observations"], params=network.params)
    next_qs = network.select("target_critic")(_critic_obs_e(next_obs_e, batch, "next_language_label"), next_actions_critic)
    next_q = jnp.min(next_qs, axis=0)
    target_q = batch["rewards"] + discount * batch["masks"] * next_q

    obs_e = network.select("obs_encoder")(batch["observations"], params=grad_params)
    qs = network.select("critic")(_critic_obs_e(obs_e, batch, "language_label"), critic_actions, params=grad_params)
    critic_loss = jnp.square(qs - target_q[None]).mean()
    return critic_loss, {
        "critic_loss": critic_loss,
        "q_mean": qs.mean(),
        "q_max": qs.max(),
        "q_min": qs.min(),
        "target_q": target_q.mean(),
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
def _actor_loss_vla_pure_math(
    network: TrainState,
    batch: dict,
    actions_vla: jnp.ndarray,
    noise_for_logprob: jnp.ndarray,
    grad_params,
    alpha: jnp.ndarray,
    noise_scale: jnp.ndarray,
):
    """Pure DSRL actor math given a precomputed VLA action.

    ``actions_vla`` is treated as a constant input (stop_gradient): the gradient through the VLA
    is dropped on purpose so we can keep the heavy Pi0-CoT forward eager. ``noise_for_logprob``
    is the noise that the VLA was conditioned on; we re-evaluate ``log_prob`` here under
    ``grad_params`` so the noise actor still gets a gradient for the entropy term.
    """
    obs_e = network.select("obs_encoder")(batch["observations"], params=grad_params)
    dist = network.select("noise_actor")(obs_e, params=grad_params)
    log_prob = dist.log_prob(noise_for_logprob / noise_scale)

    actions_sg = jax.lax.stop_gradient(actions_vla)
    qs = network.select("critic")(_critic_obs_e(obs_e, batch, "language_label"), actions_sg)
    q = jnp.min(qs, axis=0)
    actor_loss = (alpha * log_prob - q).mean()
    return actor_loss, {
        "actor_loss": actor_loss,
        "noise_log_prob": log_prob.mean(),
        "q_for_actor": q.mean(),
    }


@jax.jit
def _vla_forward_prepare_critic_noise(
    network: TrainState,
    batch: dict,
    rng_noise: jnp.ndarray,
    noise_scale: jnp.ndarray,
) -> jnp.ndarray:
    """Pure JAX prep for critic VLA forward: sample frozen policy noise on next observations."""
    next_obs_e_frozen = network.select("obs_encoder")(batch["next_observations"], params=network.params)
    next_dist = network.select("noise_actor")(next_obs_e_frozen)
    return next_dist.sample(seed=rng_noise) * noise_scale


@jax.jit
def _vla_forward_prepare_actor_noise(
    network: TrainState,
    batch: dict,
    rng_noise: jnp.ndarray,
    noise_scale: jnp.ndarray,
) -> jnp.ndarray:
    """Pure JAX prep for actor VLA forward: sample frozen policy noise on current observations."""
    obs_e_frozen = network.select("obs_encoder")(batch["observations"], params=network.params)
    dist_frozen = network.select("noise_actor")(obs_e_frozen)
    return dist_frozen.sample(seed=rng_noise) * noise_scale


# --------------------------------------------------------------------------- #
# Networks                                                                    #
# --------------------------------------------------------------------------- #


class CarlaObservationEncoder(nn.Module):
    """Encode CARLA observations: vector state as float, or RGB uint8 via IMPALA."""

    observation_mode: str
    impala_width: int = 1
    impala_stack_sizes: tuple = (16, 32, 32)
    impala_num_blocks: int = 2
    image_mlp_hidden_dims: tuple = (512,)
    layer_norm: bool = False

    @nn.compact
    def __call__(self, observations):
        if self.observation_mode == "state":
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

    def _as_critic_actions(self, actions: jnp.ndarray) -> jnp.ndarray:
        """Project actions into the critic representation (default: first-step 4-D VLA action)."""
        x = jnp.asarray(actions, dtype=jnp.float32)
        target_dim = int(self.config.get("critic_action_dim", x.shape[-1]))
        if x.shape[-1] == target_dim:
            return x

        # Replay may store flattened chunk actions (H * D). Project to first-step D-dim action for critic.
        if x.ndim == 2:
            ah = int(self.config.get("vla_action_horizon", 10))
            ad = int(self.config.get("vla_action_dim", 4))
            if x.shape[-1] == ah * ad:
                chunk = x.reshape(x.shape[0], ah, ad)
                return chunk[:, 0, :target_dim]
        elif x.ndim == 3:
            return x[:, 0, :target_dim]

        raise ValueError(
            f"Cannot map actions shape {tuple(x.shape)} to critic_action_dim={target_dim}. "
            "Check vla_action_horizon/vla_action_dim or disable chunk actions for env."
        )

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
        enc = self._encode_obs(self.network.params, observations)
        dist = self.network.select("noise_actor")(enc, temperature=temperature)
        noise = dist.sample(seed=sub)
        noise = noise * self.config["noise_scale"]
        return jnp.asarray(self.vla_sample_fn(observations, noise))

    @jax.jit
    def sample_actions_dagger(self, observations):
        """Deterministic rollout policy for DAgger: BC flow from zero noise."""
        batch = observations.shape[0]
        flow_action_dim = int(self.config.get("critic_action_dim", self.config["actor_action_dim"]))
        noise = jnp.zeros((batch, flow_action_dim), dtype=jnp.float32)
        return self._flow_sample(self.network.params, observations, noise)

    # ----- losses --------------------------------------------------------- #

    def critic_loss(self, batch, grad_params, rng):
        next_obs_e = self._encode_obs(self.network.params, batch["next_observations"])
        next_dist = self.network.select("noise_actor")(next_obs_e)
        next_noise = next_dist.sample(seed=rng) * self.config["noise_scale"]
        next_actions = self._flow_sample(
            self.network.params, batch["next_observations"], next_noise
        )
        next_actions = self._as_critic_actions(next_actions)
        next_qs = self.network.select("target_critic")(_critic_obs_e(next_obs_e, batch, "next_language_label"), next_actions)
        next_q = jnp.min(next_qs, axis=0)

        target_q = batch["rewards"] + self.config["discount"] * batch["masks"] * next_q

        obs_e = self._encode_obs(grad_params, batch["observations"])
        critic_actions = self._as_critic_actions(batch["actions"])
        qs = self.network.select("critic")(
            _critic_obs_e(obs_e, batch, "language_label"), critic_actions, params=grad_params
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
        actions = self._as_critic_actions(batch["actions"])
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
        # We do not want gradient through the (frozen) flow weights.
        actions = self._flow_sample(self.network.params, batch["observations"], noise)
        qs = self.network.select("critic")(_critic_obs_e(obs_e, batch, "language_label"), actions)
        q = jnp.min(qs, axis=0)
        actor_loss = (self.config["alpha"] * log_prob - q).mean()
        return actor_loss, {
            "actor_loss": actor_loss,
            "noise_log_prob": log_prob.mean(),
            "q_for_actor": q.mean(),
        }
        
    def _vla_forward_for_critic(self, batch, rng):
        """Eager VLA forward used by the critic bootstrap target. No gradient w.r.t. ``grad_params``."""
        sv = _steervla()
        rng_n, rng_act = jax.random.split(rng)
        noise_scale = jnp.asarray(self.config["noise_scale"], dtype=jnp.float32)
        next_noise = _vla_forward_prepare_critic_noise(
            self.network, batch, rng_n, noise_scale,
        )
        next_noise = next_noise.reshape(batch["observations"].shape[0], self.config["action_horizon"], self.config["actor_action_dim"])
        next_openpi_observation = self._as_jax_pytree(batch["next_openpi_observation"])
        # We need to increase the dimension of the noise to match the action dimension
        next_actions = self.steervla_actor._sample_actions(
            rng_act,
            next_openpi_observation,
            noise=next_noise,
            image_keys=tuple(self.config.get("image_keys", ("base_0_rgb",))),
            num_steps=int(self.config.get("flow_steps", 10)),
        )

        return jax.lax.stop_gradient(self._as_critic_actions(next_actions))

    def _vla_forward_for_actor(self, batch, rng):
        """Eager VLA forward used by the actor loss; returns ``(actions_for_critic, noise_used)``."""
        sv = _steervla()
        rng_a, rng_act = jax.random.split(rng)
        noise_scale = jnp.asarray(self.config["noise_scale"], dtype=jnp.float32)
        t0 = time.perf_counter()
        noise = _vla_forward_prepare_actor_noise(
            self.network, batch, rng_a, noise_scale,
        )
        noise = noise.reshape(batch["observations"].shape[0], self.config["action_horizon"], self.config["actor_action_dim"])
        openpi_observation = self._as_jax_pytree(batch["openpi_observation"])

        actions = self.steervla_actor._sample_actions(
            rng_act,
            openpi_observation,
            noise=noise,
            image_keys=tuple(self.config.get("image_keys", ("base_0_rgb",))),
            num_steps=int(self.config.get("flow_steps", 10)),
        )
        return self._as_critic_actions(actions), jax.lax.stop_gradient(noise)

    def critic_loss_vla(self, batch, grad_params, rng):
        """Critic loss with bootstrap target from a frozen VLA action sample.

        VLA forward (CoT + flow-matching) is executed eagerly; only the surrounding DSRL math
        (target-Q, MSE) is jitted via :func:`_critic_loss_vla_pure_math`. ``grad_params`` does
        not propagate through the VLA in the original code either (frozen action mapper for the
        target), so this is a behavior-preserving split.
        """
        next_actions_critic = self._vla_forward_for_critic(batch, rng)
        critic_actions = self._as_critic_actions(batch["actions"])
        discount = jnp.asarray(self.config["discount"], dtype=jnp.float32)
        return _critic_loss_vla_pure_math(
            self.network,
            batch,
            next_actions_critic,
            critic_actions,
            grad_params,
            discount,
        )

    def actor_loss_vla(self, batch, grad_params, rng):
        """Actor loss with a VLA action sampled eagerly outside the gradient trace.

        Trade-off vs. the original implementation: gradient does NOT flow through the VLA back
        into the noise actor (i.e., the reparameterization ``∂Q/∂a · ∂a/∂z`` term is dropped).
        The noise actor still gets gradient via (i) the entropy term ``alpha * log_prob`` and
        (ii) the encoder→critic path. This is the standard way to integrate a frozen,
        non-fully-differentiable action mapper (Pi0-CoT has non-traceable CoT decoding).
        """
        actions_critic, noise = self._vla_forward_for_actor(batch, rng)
        alpha = jnp.asarray(self.config["alpha"], dtype=jnp.float32)
        noise_scale = jnp.asarray(self.config["noise_scale"], dtype=jnp.float32)
        return _actor_loss_vla_pure_math(
            self.network,
            batch,
            actions_critic,
            noise,
            grad_params,
            alpha,
            noise_scale,
        )

    def total_loss_vla(self, batch, grad_params, rng=None):
        """Sum of VLA-path losses (logging); optimization uses :meth:`update_with_vla` (two optimizers)."""
        rng = rng if rng is not None else self.rng
        rng, c_rng, a_rng, f_rng = jax.random.split(rng, 4)

        c_loss, c_info = self.critic_loss_vla(batch, grad_params, c_rng)
        a_loss, a_info = self.actor_loss_vla(batch, grad_params, a_rng)
        
        info = {}
        for k, v in c_info.items():
            info[f"critic_vla/{k}"] = v
        for k, v in a_info.items():
            info[f"actor_vla/{k}"] = v
        combined = c_loss + a_loss
        info["total_loss_vla"] = combined
        info["flax_loss_vla"] = c_loss + a_loss
        return combined, info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng
        rng, c_rng, f_rng, a_rng = jax.random.split(rng, 4)

        c_loss, c_info = self.critic_loss(batch, grad_params, c_rng)
        for k, v in c_info.items():
            info[f"critic/{k}"] = v
        f_loss, f_info = self.flow_loss(batch, grad_params, f_rng)
        for k, v in f_info.items():
            info[f"flow/{k}"] = v
        a_loss, a_info = self.actor_loss(batch, grad_params, a_rng)
        for k, v in a_info.items():
            info[f"actor/{k}"] = v

        return c_loss + f_loss + a_loss, info

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

    def update_with_vla(self, batch):
        """Flax critic + noise actor update with eager VLA forwards and a jitted gradient core.

        Profile metrics returned in ``info`` (in ms, summed across the whole update step):
        """
        new_rng, rng = jax.random.split(self.rng)
        rng, c_rng, a_rng, _vla_rng = jax.random.split(rng, 4)
        
        sv = _steervla()
        if "openpi_state" in batch and "next_openpi_state" in batch:
            openpi_observation = sv.openpi_observation_from_replay_batch(batch)
            next_openpi_observation = sv.openpi_observation_from_replay_batch(
                batch, prefix="next_",
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
        batch = dict(batch)
        batch["openpi_observation"] = self._as_jax_pytree(openpi_observation)
        batch["next_openpi_observation"] = self._as_jax_pytree(next_openpi_observation)

        next_actions_critic = self._vla_forward_for_critic(batch, c_rng)
        actions_critic, noise_actor_sample = self._vla_forward_for_actor(batch, a_rng)

        discount = jnp.asarray(self.config["discount"], dtype=jnp.float32)
        alpha = jnp.asarray(self.config["alpha"], dtype=jnp.float32)
        noise_scale = jnp.asarray(self.config["noise_scale"], dtype=jnp.float32)
        critic_actions = self._as_critic_actions(batch["actions"])

        def loss_pure(grad_params):
            c, ci = _critic_loss_vla_pure_math(
                self.network, batch, next_actions_critic, critic_actions, grad_params, discount,
            )
            noise_actor_sample_reshaped = noise_actor_sample.reshape(batch["observations"].shape[0], self.config["action_horizon"]*self.config["actor_action_dim"])
            a, ai = _actor_loss_vla_pure_math(
                self.network, batch, actions_critic, noise_actor_sample_reshaped, grad_params, alpha, noise_scale,
            )
            aux = {
                **{f"critic/{k}": v for k, v in ci.items()},
                **{f"actor/{k}": v for k, v in ai.items()},
            }
            return c + a, aux

        (loss_value, info), grads = jax.value_and_grad(loss_pure, has_aux=True)(self.network.params)
        
        # Force compute to finish so we measure compute, not dispatch.
        jax.tree_util.tree_map(
            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, grads,
        )

        new_network, grad_stats = _apply_grads_pure(self.network, grads)
        jax.tree_util.tree_map(
            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, new_network.params,
        )
        info = {**info, **grad_stats}

        self.target_update(new_network)
        # target_update mutates the dict; force a sync via params again.
        jax.tree_util.tree_map(
            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, new_network.params,
        )


        new_agent = self.replace(network=new_network, rng=new_rng)
        return new_agent, info

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
        if obs_mode not in ("state", "image"):
            raise ValueError(f"observation_mode must be 'state' or 'image', got {obs_mode!r}")

        if vla_sample_fn is not None:
            action_dim = int(config.get("critic_action_dim", 2))
        else:
            action_dim = ex_actions.shape[-1]
        obs_encoder_def = CarlaObservationEncoder(
            observation_mode=obs_mode,
            impala_width=int(config.get("image_impala_width", 1)),
            impala_stack_sizes=tuple(config.get("image_impala_stack_sizes", (16, 32, 32))),
            impala_num_blocks=int(config.get("image_impala_num_blocks", 2)),
            image_mlp_hidden_dims=tuple(config.get("image_mlp_hidden_dims", (512,))),
            layer_norm=config["layer_norm"],
        )
        if obs_mode == "state":
            embed_dim = int(ex_observations.shape[-1])
        else:
            embed_dim = int(tuple(config.get("image_mlp_hidden_dims", (512,)))[-1])
        critic_feedback_mode = str(config.get("critic_feedback_mode", "commentary_bow"))
        if critic_feedback_mode == "action_delta":
            lang_dim = int(config.get("critic_action_dim", 4))
        else:
            lang_dim = int(config.get("language_label_dim", 119))
        batch_shape = ex_observations.shape[:1]
        ex_embedded = jnp.zeros(batch_shape + (embed_dim,), dtype=jnp.float32)
        ex_critic_embedded = jnp.zeros(batch_shape + (embed_dim + lang_dim,), dtype=jnp.float32)
        ex_model_actions = jnp.zeros(batch_shape + (action_dim,), dtype=jnp.float32)
        ex_t = jnp.zeros(batch_shape + (1,), dtype=jnp.float32)

        # BC flow-matching head must stay a Flax module: init calls ``flow(enc, x_t, t)``.
        # OpenPI SteerVLA hooks only via ``vla_sample_fn`` (see ``sample_actions_with_vla``).
        flow_def = FlowActor(
            hidden_dims=tuple(config["flow_hidden_dims"]),
            action_dim=action_dim,
            layer_norm=config["layer_norm"],
        )
        actor_def = NoiseActor(
            hidden_dims=tuple(config["actor_hidden_dims"]),
            action_dim=config["actor_action_dim"]*config["action_horizon"],
        )
        critic_def = Critic(
            hidden_dims=tuple(config["critic_hidden_dims"]),
            layer_norm=config["layer_norm"],
            ensemble_size=config["critic_ensemble"],
        )

        networks = {
            "obs_encoder": (obs_encoder_def, (ex_observations,)),
            "flow": (flow_def, (ex_embedded, ex_model_actions, ex_t)),
            "noise_actor": (actor_def, (ex_embedded,)),
            "critic": (critic_def, (ex_critic_embedded, ex_model_actions)),
            "target_critic": (
                copy.deepcopy(critic_def),
                (ex_critic_embedded, ex_model_actions),
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
            warmup_steps=1000,
            updates_per_step=1,
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
            vla_sample_actions_low_memory=True,
            vla_sample_actions_jit_denoise_steps=False,
            vla_action_horizon=10,
            vla_action_dim=4,
            vla_output_action_format="DELTA_XY_T_DELTA_XY_SPACE",
            # Critic/action/flow representation used inside DSRL when VLA is enabled.
            # ``4`` matches SteerVLA DELTA_XY_T_DELTA_XY_SPACE first-step action.
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
            online_training_mode="rl",
        )
    )
    return config
