"""Minimal JAX implementation of TMRL (Timestep- / context-smoothing steering RL).

TMRL extends :class:`jax_agents.dsrl.DSRLAgent` with **one additional network** whose job is
to select, per state, *how much to noise the model context* in addition to the DSRL steering
vector. Concretely, the SAC noise actor grows two extra heads on top of the usual steering-noise
head ``z``:

* a **timestep** head ``t`` -> a scalar context-noise level ``t_context in [0, 1]`` (how strongly
  to smooth / corrupt the context), and
* a **context-noise** head ``c`` -> the ``context_dim`` noise realization used to noise the context.

The reference torch implementation lives in ``/home/carla/tmrl`` (``sac_ogbench.py``,
``models/sac_agent.py`` :class:`Actor` / :class:`SoftQNetwork`, and ``models/flow.py``
:class:`ContextSmoothedFlowPolicy`). This port keeps that structure but plugs into the CARLA /
SteerVLA stack the same way DSRL does.

Two execution paths, mirroring DSRL:

* **SteerVLA path** (``vla_sample_fn`` / ``steervla_actor`` set): the steering vector ``z`` seeds the
  frozen Pi0-CoT action flow, and the learned ``t_context`` is pushed into the SteerVLA actor's
  Context-Smoothed Pre-training (CSP) hook (:meth:`vlas.steervla.SteerVLAActor.set_t_context` at
  rollout; the ``t_context`` kwarg of ``_sample_actions`` during RL updates). This realizes exactly
  what ``steervla.py`` calls "what a TMRL high-level policy would select". Requires a checkpoint
  trained with ``Pi0CoTConfig.context_smoothing`` (e.g. ``*_csp``); otherwise TMRL degrades to DSRL
  steering-only behaviour (see ``use_vla_context_smoothing``).
* **Standalone BC-flow path** (no VLA): a :class:`ContextSmoothedFlowActor` conditions the flow on a
  diffusion-noised encoded observation at level ``t_context`` (matching ``ContextSmoothedFlowPolicy``).
  Used for ``sample_actions`` / ``update`` / DAgger without a SteerVLA attached.

The noise critic ``Q(s, z, t_context, c)`` conditions on the two extra actor outputs (matching the
reference ``SoftQNetwork``); the env-action critic ``Q(s, a_env)`` is unchanged from DSRL.

Deliberately kept simple (same spirit as :mod:`jax_agents.dsrl`): fixed SAC temperatures
``alpha`` / ``alpha_timestep`` / ``alpha_context`` (autotuned temperatures as in the torch reference
are a trivial extension), and the context noise realization ``c`` is fed to the VLA only implicitly
(the Pi0-CoT model samples its own context noise given ``t_context``); ``c`` conditions the noise
critic and drives the standalone flow.

``get_config()`` makes this discoverable to ``--agent=jax_agents/tmrl.py`` and to
``impls/configs/steervla_tmrl_config.py``.
"""

from __future__ import annotations

import copy
import time
from typing import Any, Callable, Optional

import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState
from utils.networks import MLP, default_init

from jax_agents.dsrl import (
    CarlaObservationEncoder,
    Critic,
    DSRLAgent,
    _apply_debug_stop_reward_relabel,
    _critic_obs_e,
)


# --------------------------------------------------------------------------- #
# Diffusion / time helpers                                                    #
# --------------------------------------------------------------------------- #


def _fourier_time_embed(t: jnp.ndarray, dim: int) -> jnp.ndarray:
    """Sinusoidal embedding of a batch of scalar times ``(B,)`` or ``(B, 1)`` -> ``(B, dim)``.

    Normalizes to a trailing singleton first so ``B == 1`` (rollout) keeps its batch axis.
    """
    t = jnp.asarray(t, dtype=jnp.float32)
    if t.ndim == 0:
        t = t[None]
    if t.ndim == 1:
        t = t[:, None]
    elif t.shape[-1] != 1:
        t = t[..., None]
    freqs = jnp.exp(jnp.linspace(0.0, 4.0, dim // 2))
    return jnp.concatenate([jnp.sin(t * freqs), jnp.cos(t * freqs)], axis=-1)


def _diffusion_noise(
    x: jnp.ndarray,
    t_context: jnp.ndarray,
    context_noise: jnp.ndarray,
    num_train_steps: int,
) -> jnp.ndarray:
    """Forward diffusion of ``x`` at level ``t_context in [0, 1]``: ``sqrt(ab) x + sqrt(1-ab) eps``.

    ``ab`` is the cumulative product of ``alphas`` from a linear ``beta`` schedule, matching
    ``ContextSmoothedFlowPolicy._apply_diffusion_noise`` in the torch reference.
    """
    betas = jnp.linspace(1e-4, 0.02, num_train_steps, dtype=jnp.float32)
    alpha_bars = jnp.cumprod(1.0 - betas)
    idx = jnp.clip(
        (jnp.asarray(t_context, dtype=jnp.float32) * num_train_steps).astype(jnp.int32),
        0,
        num_train_steps - 1,
    )
    ab = alpha_bars[idx]  # (B,)
    view = ab.reshape(ab.shape[:1] + (1,) * (x.ndim - 1))
    return jnp.sqrt(view) * x + jnp.sqrt(1.0 - view) * context_noise


# --------------------------------------------------------------------------- #
# Networks                                                                     #
# --------------------------------------------------------------------------- #


class TMRLNoiseActor(nn.Module):
    """DSRL steering-noise actor plus two heads: context-noise level ``t`` and context noise ``c``.

    ``__call__`` returns a tuple ``(dist_z, dist_t, dist_c)`` of independent tanh-squashed diagonal
    Gaussians (as in :class:`jax_agents.dsrl.NoiseActor`), over the steering vector, the (1-d)
    timestep, and the ``context_dim`` context noise respectively. The agent maps ``t in [-1, 1]``
    to ``t_context in [0, 1]`` and scales ``z`` / ``c`` by the configured bounds.
    """

    hidden_dims: tuple
    noise_dim: int
    context_dim: int
    log_std_min: float = -5.0
    log_std_max: float = 2.0

    @nn.compact
    def __call__(self, observations, temperature: float = 1.0):
        trunk = MLP(self.hidden_dims, activate_final=True)(observations)

        def _head(dim):
            mean = nn.Dense(dim, kernel_init=default_init(0.01))(trunk)
            log_std = nn.Dense(dim, kernel_init=default_init(0.01))(trunk)
            log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
            base = distrax.MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std) * temperature)
            return distrax.Transformed(base, distrax.Block(distrax.Tanh(), ndims=1))

        return _head(self.noise_dim), _head(1), _head(self.context_dim)


class ContextSmoothedFlowActor(nn.Module):
    """Flow velocity ``v(context, a, t_action, t_context)`` conditioned on the context-noise level.

    Matches :class:`jax_agents.dsrl.FlowActor` but adds a Fourier embedding of ``t_context`` so the
    BC flow is trained (and sampled) aware of how strongly the conditioning context was smoothed --
    the standalone analogue of Pi0-CoT context smoothing.
    """

    hidden_dims: tuple
    action_dim: int
    layer_norm: bool = False
    time_embed_dim: int = 16

    @nn.compact
    def __call__(self, context, actions, t_action, t_context):
        ta = _fourier_time_embed(t_action, self.time_embed_dim)
        tc = _fourier_time_embed(t_context, self.time_embed_dim)
        x = jnp.concatenate([context, actions, ta, tc], axis=-1)
        x = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)(x)
        return nn.Dense(self.action_dim, kernel_init=default_init(0.01))(x)


# --------------------------------------------------------------------------- #
# Agent                                                                        #
# --------------------------------------------------------------------------- #


class TMRLAgent(DSRLAgent):
    """DSRL + a learned context-noise-level (and context-noise) policy.

    Reuses DSRL's action-shape / VLA-batch plumbing verbatim; overrides everything that touches the
    (now three-headed) noise actor, the ``(z, t_context, c)``-conditioned noise critic, and the
    context-smoothed flow.
    """

    # ----- config-derived scalars --------------------------------------- #

    def _timestep_max(self) -> float:
        return float(self.config.get("timestep_max", 1.0))

    def _context_noise_bound(self) -> float:
        return float(self.config.get("context_noise_bound", 1.0))

    def _context_dim(self) -> int:
        return int(self.config["context_dim"])

    def _num_train_steps(self) -> int:
        return int(self.config.get("context_num_train_steps", 100))

    def _timestep_embed_dim(self) -> int:
        return int(self.config.get("timestep_embed_dim", 16))

    def _t_to_context(self, t: jnp.ndarray) -> jnp.ndarray:
        """Map a tanh sample ``t in [-1, 1]`` (shape ``(B, 1)``) to ``t_context in [0, 1]`` ``(B,)``."""
        t = jnp.asarray(t, dtype=jnp.float32)
        tc = (t + 1.0) * 0.5 * self._timestep_max()
        tc = jnp.clip(tc, 0.0, 1.0)
        return tc[..., 0] if tc.ndim >= 1 and tc.shape[-1] == 1 else tc

    def _vla_context_smoothing_on(self) -> bool:
        """True iff configured on and the attached SteerVLA checkpoint supports CSP ``t_context``."""
        if not bool(self.config.get("use_vla_context_smoothing", True)):
            return False
        actor = self.steervla_actor
        if actor is None:
            return False
        supported = getattr(actor, "_context_smoothing_supported", None)
        try:
            return bool(supported()) if callable(supported) else False
        except Exception:
            return False

    # ----- noise-critic input assembly ---------------------------------- #

    def _noise_critic_obs(self, obs_e, batch, key, t_context, context_noise) -> jnp.ndarray:
        """Concatenate ``[obs_e, lang_label, embed(t_context), context_noise]`` for the noise critic."""
        base = _critic_obs_e(obs_e, batch, key)
        t_emb = _fourier_time_embed(jnp.asarray(t_context, dtype=jnp.float32), self._timestep_embed_dim())
        c = jnp.asarray(context_noise, dtype=jnp.float32)
        return jnp.concatenate([base, t_emb, c], axis=-1)

    def _sample_actor(self, params, obs_e, rng, temperature: float = 1.0):
        """Sample ``(z, t_context, context_noise)`` (+ log-probs) from the three-headed actor.

        ``z`` is the (scaled) steering vector at the actor's ``noise_dim`` -- for a VLA that equals
        ``action_horizon * actor_action_dim`` (reshape to a chunk with :meth:`_as_noise_chunk`); for
        the standalone flow it equals the env action dim. ``t_context in [0, 1]`` is the CSP level and
        ``context_noise`` the (scaled) context-noise realization.
        """
        dist_z, dist_t, dist_c = self.network.select("noise_actor")(obs_e, temperature=temperature, params=params)
        rz, rt, rc = jax.random.split(rng, 3)
        z, logp_z = dist_z.sample_and_log_prob(seed=rz)
        t, logp_t = dist_t.sample_and_log_prob(seed=rt)
        c, logp_c = dist_c.sample_and_log_prob(seed=rc)
        z = z * self.config["noise_scale"]
        t_context = self._t_to_context(t)
        c = c * self._context_noise_bound()
        return (z, t_context, c), (logp_z, logp_t, logp_c)

    # ----- sampling (standalone BC-flow path) --------------------------- #

    def _flow_sample_ctx(self, params, observations, noises, t_context, context_noise):
        """Euler-integrate the context-smoothed BC flow from ``noises`` (t=0) to actions (t=1)."""
        flow_steps = self.config["flow_steps"]
        obs_e = self._encode_obs(params, observations)
        context = _diffusion_noise(obs_e, t_context, context_noise, self._num_train_steps())
        action = noises
        for i in range(flow_steps):
            ta = jnp.full(context.shape[:-1] + (1,), i / flow_steps)
            v = self.network.select("flow")(context, action, ta, t_context, params=params)
            action = action + v / flow_steps
        return jnp.clip(action, -1.0, 1.0)

    @jax.jit
    def sample_actions(self, observations, seed=None, temperature=1.0):
        """Standalone (no-VLA) rollout: steering noise ``z`` seeds the context-smoothed BC flow.

        Requires the actor's ``noise_dim`` to match the env action dim (the standalone create path);
        with a VLA attached use :meth:`sample_actions_with_vla` instead.
        """
        seed = seed if seed is not None else self.rng
        seed, sub = jax.random.split(seed)
        obs_e = self._encode_obs(self.network.params, observations)
        (z, t_context, c), _ = self._sample_actor(self.network.params, obs_e, sub, temperature=temperature)
        return self._flow_sample_ctx(self.network.params, observations, z, t_context, c)

    @jax.jit
    def sample_actions_dagger(self, observations):
        """Deterministic BC rollout: clean context (t_context=0), zero steering noise."""
        batch = observations.shape[0]
        noise = jnp.zeros((batch, self._flat_env_action_dim()), dtype=jnp.float32)
        t_context = jnp.zeros((batch,), dtype=jnp.float32)
        context_noise = jnp.zeros((batch, self._context_dim()), dtype=jnp.float32)
        return self._flow_sample_ctx(self.network.params, observations, noise, t_context, context_noise)

    def sample_actions_with_vla(self, observations, seed=None, temperature=1.0):
        """Rollout: steer the VLA with ``z`` and drive its CSP level with the learned ``t_context``.

        The actor is queried on the current observation; ``z`` is passed to ``vla_sample_fn`` as the
        steering vector, and (when the checkpoint supports CSP) ``t_context`` is pushed to the
        SteerVLA actor via :meth:`vlas.steervla.SteerVLAActor.set_t_context` before the query.
        """
        if self.vla_sample_fn is None:
            return self.sample_actions(observations, seed=seed, temperature=temperature)
        seed = seed if seed is not None else self.rng
        seed, sub = jax.random.split(seed)
        obs_e = self._encode_obs(self.network.params, observations)
        (z, t_context, _), _ = self._sample_actor(self.network.params, obs_e, sub, temperature=temperature)
        if self._vla_context_smoothing_on():
            tc = float(jax.device_get(jnp.mean(t_context)))
            self.steervla_actor.set_t_context(tc, sample=False)
        return jnp.asarray(self.vla_sample_fn(observations, self._as_noise_chunk(z)))

    # ----- losses (standalone BC-flow path) ----------------------------- #

    def flow_loss(self, batch, grad_params, rng):
        """Context-smoothed flow-matching loss (mirrors ``ContextSmoothedFlowPolicy.forward``)."""
        actions = self._clip_actions_to_env(batch["actions"])
        rng, r_x0, r_ta, r_tc, r_cn = jax.random.split(rng, 5)
        x_0 = jax.random.normal(r_x0, actions.shape) * self.config["noise_scale"]
        t = jax.random.uniform(r_ta, actions.shape[:-1] + (1,))
        x_t = (1.0 - t) * x_0 + t * actions
        target_v = actions - x_0

        obs_e = self._encode_obs(grad_params, batch["observations"])
        t_context = jax.random.uniform(r_tc, (obs_e.shape[0],))
        context_noise = jax.random.normal(r_cn, obs_e.shape)
        context = _diffusion_noise(obs_e, t_context, context_noise, self._num_train_steps())
        pred_v = self.network.select("flow")(context, x_t, t, t_context, params=grad_params)
        loss = jnp.square(pred_v - target_v).mean()
        return loss, {"flow_loss": loss}

    def noise_critic_loss(self, batch, grad_params, rng):
        """Distill ``noise_critic(s, z, t, c)`` toward env ``critic(s, a_env_stored)`` (as in DSRL)."""
        obs_e = self._encode_obs(grad_params, batch["observations"])
        (z, t_context, c), _ = self._sample_actor(grad_params, obs_e, rng)
        env_actions = self._clip_actions_to_env(batch["actions"])
        target_qs = self.network.select("critic")(_critic_obs_e(obs_e, batch, "language_label"), env_actions)
        target_q = jnp.min(target_qs, axis=0)
        qs = self.network.select("noise_critic")(
            self._noise_critic_obs(obs_e, batch, "language_label", t_context, c), z, params=grad_params,
        )
        critic_loss = jnp.square(qs - target_q[None]).mean()
        return critic_loss, {
            "critic_loss": critic_loss,
            "q_mean": qs.mean(),
            "q_max": qs.max(),
            "q_min": qs.min(),
            "target_q": target_q.mean(),
        }

    def actor_loss(self, batch, grad_params, rng):
        """SAC actor loss summing entropy over the steering, timestep, and context-noise heads."""
        obs_e = self._encode_obs(grad_params, batch["observations"])
        (z, t_context, c), (logp_z, logp_t, logp_c) = self._sample_actor(grad_params, obs_e, rng)
        qs = self.network.select("noise_critic")(
            self._noise_critic_obs(obs_e, batch, "language_label", t_context, c), z,
        )
        q = jnp.min(qs, axis=0)
        alpha = self.config["alpha"]
        alpha_t = float(self.config.get("alpha_timestep", self.config["alpha"]))
        alpha_c = float(self.config.get("alpha_context", self.config["alpha"]))
        actor_loss = (alpha * logp_z + alpha_t * logp_t + alpha_c * logp_c - q).mean()
        return actor_loss, {
            "actor_loss": actor_loss,
            "noise_log_prob": logp_z.mean(),
            "timestep_log_prob": logp_t.mean(),
            "context_log_prob": logp_c.mean(),
            "t_context_mean": jnp.mean(t_context),
            "q_for_actor": q.mean(),
        }

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

    # ----- VLA path ----------------------------------------------------- #

    def _vla_forward(self, observations, openpi_observations, rng, noise=None, t_context=None):
        """Frozen VLA action sample; injects the actor-selected (or given) ``t_context`` as CSP level.

        ``noise`` (steering) and ``t_context`` default to a *frozen*-policy sample on ``observations``.
        Returns env-clipped actions (stop-gradient), matching :meth:`DSRLAgent._vla_forward`.
        """
        rng_n, rng_act = jax.random.split(rng)
        if noise is None or t_context is None:
            obs_e = self._encode_obs(self.network.params, observations)
            (z_s, t_s, _), _ = self._sample_actor(self.network.params, obs_e, rng_n)
            if noise is None:
                noise = z_s
            if t_context is None:
                t_context = t_s
        noise_chunk = self._as_noise_chunk(noise)
        next_openpi_observation = self._as_jax_pytree(openpi_observations)

        extra = {}
        if self._vla_context_smoothing_on():
            extra["t_context"] = jnp.asarray(t_context, dtype=jnp.float32)

        sample_actions_time = time.time()
        next_actions = self.steervla_actor._sample_actions(
            rng_act,
            next_openpi_observation,
            noise=noise_chunk,
            image_keys=tuple(self.config.get("image_keys", ("base_0_rgb",))),
            num_steps=int(self.config.get("vla_update_flow_steps", self.config.get("flow_steps", 10))),
            **extra,
        )
        next_actions = self.steervla_actor.postprocess_sampled_trajectory(
            next_actions, observation_state=next_openpi_observation.state,
        )
        jax.block_until_ready(next_actions)
        print(f"[DEBUG - tmrl] Sample actions time: {time.time() - sample_actions_time} seconds")
        return jax.lax.stop_gradient(self._clip_actions_to_env(next_actions))

    def critic_loss_vla(self, batch, grad_params, rng):
        """Env-critic Bellman with a bootstrap from a frozen VLA action at the actor-selected level."""
        next_actions = self._vla_forward(
            batch["next_observations"], batch["next_openpi_observation"], rng,
        )
        next_obs_e = self.network.select("obs_encoder")(batch["next_observations"], params=self.network.params)
        next_qs = self.network.select("target_critic")(
            _critic_obs_e(next_obs_e, batch, "next_language_label"), next_actions,
        )
        next_q = jnp.min(next_qs, axis=0)
        discount = jnp.asarray(self.config["discount"], dtype=jnp.float32)
        target_q = batch["rewards"] + discount * batch["masks"] * next_q

        obs_e = self.network.select("obs_encoder")(batch["observations"], params=grad_params)
        critic_actions = self._clip_actions_to_env(batch["actions"])
        qs = self.network.select("critic")(
            _critic_obs_e(obs_e, batch, "language_label"), critic_actions, params=grad_params,
        )
        critic_loss = jnp.square(qs - target_q[None]).mean()
        return critic_loss, {
            "critic_loss": critic_loss,
            "q_mean": qs.mean(),
            "q_max": qs.max(),
            "q_min": qs.min(),
            "target_q": target_q.mean(),
        }

    def noise_critic_loss_vla(self, batch, grad_params, rng):
        """Distill ``noise_critic(s, z, t, c)`` toward env ``critic(s, VLA(s, z, t_context))``.

        ``z`` (steering) and ``t_context`` are the *same* actor sample used for the VLA forward, so
        the noise critic learns the value of steering the VLA at that context-noise level.
        """
        rng, act_rng, vla_rng = jax.random.split(rng, 3)
        obs_e = self._encode_obs(grad_params, batch["observations"])
        (z, t_context, c), _ = self._sample_actor(grad_params, obs_e, act_rng)
        vla_actions = self._vla_forward(
            batch["observations"], batch["openpi_observation"], vla_rng,
            noise=jax.lax.stop_gradient(z), t_context=jax.lax.stop_gradient(t_context),
        )
        target_qs = self.network.select("critic")(_critic_obs_e(obs_e, batch, "language_label"), vla_actions)
        target_q = jnp.min(target_qs, axis=0)
        qs = self.network.select("noise_critic")(
            self._noise_critic_obs(obs_e, batch, "language_label", t_context, c), z, params=grad_params,
        )
        noise_q = jnp.min(qs, axis=0)
        noise_critic_loss = jnp.square(noise_q - jax.lax.stop_gradient(target_q)).mean()
        return noise_critic_loss, {
            "noise_critic_loss": noise_critic_loss,
            "noise_q_values": noise_q.mean(),
            "q_values": target_q.mean(),
        }

    def noise_actor_loss_vla(self, batch, grad_params, rng):
        """SAC actor loss (VLA path) over the steering, timestep, and context-noise heads."""
        return self.actor_loss(batch, grad_params, rng)

    def total_loss_vla(self, batch, grad_params, rng=None, vla_cache=None):
        """Sum of VLA-path losses. ``vla_cache`` (precomputed eager VLA forwards) is honored."""
        if bool(self.config.get("debug_task", False)):
            batch = _apply_debug_stop_reward_relabel(batch)
        rng = rng if rng is not None else self.rng
        rng, nc_rng, na_rng, c_rng = jax.random.split(rng, 4)
        discount = jnp.asarray(self.config["discount"], dtype=jnp.float32)
        critic_actions = self._clip_actions_to_env(batch["actions"])

        if vla_cache is not None:
            # Noise-critic distillation from cached VLA forward at the cached (z, t_context).
            obs_e = self._encode_obs(grad_params, batch["observations"])
            z = vla_cache["noise_actions"]
            t_context = vla_cache["t_context"]
            c = vla_cache["context_noise"]
            target_qs = self.network.select("critic")(_critic_obs_e(obs_e, batch, "language_label"), vla_cache["vla_actions_nc"])
            target_q = jnp.min(target_qs, axis=0)
            nc_qs = self.network.select("noise_critic")(
                self._noise_critic_obs(obs_e, batch, "language_label", t_context, c), z, params=grad_params,
            )
            nc_q = jnp.min(nc_qs, axis=0)
            nc_loss = jnp.square(nc_q - jax.lax.stop_gradient(target_q)).mean()
            nc_info = {"noise_critic_loss": nc_loss, "noise_q_values": nc_q.mean(), "q_values": target_q.mean()}

            # Actor loss (fresh sample; noise critic is frozen wrt actor grads inside).
            na_loss, na_info = self.actor_loss(batch, grad_params, na_rng)

            # Env-critic Bellman from cached next action.
            next_obs_e = self.network.select("obs_encoder")(batch["next_observations"], params=self.network.params)
            next_qs = self.network.select("target_critic")(
                _critic_obs_e(next_obs_e, batch, "next_language_label"), vla_cache["next_actions_critic"],
            )
            target_c = batch["rewards"] + discount * batch["masks"] * jnp.min(next_qs, axis=0)
            obs_e_c = self.network.select("obs_encoder")(batch["observations"], params=grad_params)
            c_qs = self.network.select("critic")(_critic_obs_e(obs_e_c, batch, "language_label"), critic_actions, params=grad_params)
            c_loss = jnp.square(c_qs - target_c[None]).mean()
            c_info = {"critic_loss": c_loss, "q_mean": c_qs.mean(), "q_max": c_qs.max(), "q_min": c_qs.min(), "target_q": target_c.mean()}
        else:
            nc_loss, nc_info = self.noise_critic_loss_vla(batch, grad_params, nc_rng)
            na_loss, na_info = self.noise_actor_loss_vla(batch, grad_params, na_rng)
            c_loss, c_info = self.critic_loss_vla(batch, grad_params, c_rng)

        info = {}
        for k, v in nc_info.items():
            info[f"noise_critic_vla/{k}"] = v
        for k, v in na_info.items():
            info[f"noise_actor_vla/{k}"] = v
        for k, v in c_info.items():
            info[f"critic/{k}"] = v
        combined = nc_loss + na_loss + c_loss
        info["total_loss_vla"] = combined
        info["noise_critic_loss_vla"] = nc_loss
        info["noise_actor_loss_vla"] = na_loss
        info["critic_loss_vla"] = c_loss
        return combined, info

    def _precompute_vla_loss_cache(self, batch, rng):
        """Eager VLA forwards (next action + noise-critic action) consumed by :meth:`total_loss_vla`."""
        rng, act_rng, nc_rng, c_rng = jax.random.split(rng, 4)

        next_actions = self._vla_forward(
            batch["next_observations"], batch["next_openpi_observation"], c_rng,
        )

        obs_e = self._encode_obs(self.network.params, batch["observations"])
        (z, t_context, c), _ = self._sample_actor(self.network.params, obs_e, act_rng)
        z = jax.lax.stop_gradient(z)
        t_context = jax.lax.stop_gradient(t_context)
        c = jax.lax.stop_gradient(c)
        vla_actions_nc = self._vla_forward(
            batch["observations"], batch["openpi_observation"], nc_rng, noise=z, t_context=t_context,
        )
        return {
            "next_actions_critic": next_actions,
            "noise_actions": z,
            "t_context": t_context,
            "context_noise": c,
            "vla_actions_nc": vla_actions_nc,
        }

    # ----- construction ------------------------------------------------- #

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
        image_encoder = str(config.get("image_encoder", "impala")).lower()
        if image_encoder not in ("impala", "siglip"):
            raise ValueError(f"image_encoder must be 'impala' or 'siglip', got {image_encoder!r}")
        if obs_mode != "image" and image_encoder == "siglip":
            raise ValueError("image_encoder='siglip' requires observation_mode='image'.")

        if vla_sample_fn is not None:
            env_action_dim = int(config.get("vla_action_dim", 4)) * int(
                config.get("vla_action_horizon", config.get("action_horizon", 10))
            )
            noise_action_dim = int(config["actor_action_dim"]) * int(config["action_horizon"])
        else:
            env_action_dim = ex_actions.shape[-1]
            noise_action_dim = env_action_dim

        obs_encoder_def = CarlaObservationEncoder(
            observation_mode=obs_mode,
            image_encoder=image_encoder,
            impala_width=int(config.get("image_impala_width", 1)),
            impala_stack_sizes=tuple(config.get("image_impala_stack_sizes", (16, 32, 32))),
            impala_num_blocks=int(config.get("image_impala_num_blocks", 2)),
            image_mlp_hidden_dims=tuple(config.get("image_mlp_hidden_dims", (512,))),
            layer_norm=config["layer_norm"],
        )
        if obs_mode == "state":
            embed_dim = int(ex_observations.shape[-1])
        elif image_encoder == "siglip":
            single = int(config.get("siglip_embed_dim", ex_observations.shape[-1]))
            embed_dim = single * 3 if bool(config.get("siglip_include_prompt_subtask", False)) else single
        else:
            embed_dim = int(tuple(config.get("image_mlp_hidden_dims", (512,)))[-1])

        # Context that gets smoothed is the encoded observation (matching ``noise_state``).
        context_dim = int(config.get("context_dim") or embed_dim)
        config = dict(config)
        config["context_dim"] = context_dim

        critic_feedback_mode = str(config.get("critic_feedback_mode", "commentary_bow"))
        if critic_feedback_mode == "action_delta":
            lang_dim = int(config.get("critic_action_dim", 4))
        else:
            lang_dim = int(config.get("language_label_dim", 119))
        t_embed_dim = int(config.get("timestep_embed_dim", 16))

        batch_shape = ex_observations.shape[:1]
        ex_embedded = jnp.zeros(batch_shape + (embed_dim,), dtype=jnp.float32)
        ex_critic_embedded = jnp.zeros(batch_shape + (embed_dim + lang_dim,), dtype=jnp.float32)
        # Noise critic sees [obs_e, lang, t_embed, context_noise] on the observation side.
        ex_noise_critic_obs = jnp.zeros(batch_shape + (embed_dim + lang_dim + t_embed_dim + context_dim,), dtype=jnp.float32)
        ex_env_actions = jnp.zeros(batch_shape + (env_action_dim,), dtype=jnp.float32)
        ex_noise_actions = jnp.zeros(batch_shape + (noise_action_dim,), dtype=jnp.float32)
        ex_context = jnp.zeros(batch_shape + (context_dim,), dtype=jnp.float32)
        ex_t = jnp.zeros(batch_shape + (1,), dtype=jnp.float32)
        ex_tc = jnp.zeros(batch_shape, dtype=jnp.float32)

        flow_def = ContextSmoothedFlowActor(
            hidden_dims=tuple(config["flow_hidden_dims"]),
            action_dim=env_action_dim,
            layer_norm=config["layer_norm"],
            time_embed_dim=t_embed_dim,
        )
        actor_def = TMRLNoiseActor(
            hidden_dims=tuple(config["actor_hidden_dims"]),
            noise_dim=noise_action_dim,
            context_dim=context_dim,
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
            "flow": (flow_def, (ex_context, ex_env_actions, ex_t, ex_tc)),
            "noise_actor": (actor_def, (ex_embedded,)),
            "noise_critic": (noise_critic_def, (ex_noise_critic_obs, ex_noise_actions)),
            "critic": (critic_def, (ex_critic_embedded, ex_env_actions)),
            "target_critic": (copy.deepcopy(critic_def), (ex_critic_embedded, ex_env_actions)),
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
# Config                                                                       #
# --------------------------------------------------------------------------- #


def get_config():
    """ml_collections.ConfigDict consumed by ``main_carla.py`` (superset of DSRL's)."""
    from jax_agents.dsrl import get_config as _dsrl_get_config

    config = _dsrl_get_config()
    config.agent_name = "tmrl"

    # --- TMRL: context-noise-level policy knobs ---
    # Context that gets smoothed (encoded observation). ``None`` -> auto-set to the encoder width.
    config.context_dim = ml_collections.config_dict.placeholder(int)
    # Actor timestep head maps tanh in [-1, 1] to t_context in [0, timestep_max].
    config.timestep_max = 1.0
    # Bound on the (tanh-squashed) context-noise realization ``c``.
    config.context_noise_bound = 1.0
    # SAC temperatures for the two extra heads (default to the steering-noise ``alpha``).
    config.alpha_timestep = 0.1
    config.alpha_context = 0.1
    # Diffusion schedule length for the standalone context-smoothed BC flow.
    config.context_num_train_steps = 100
    # Fourier embedding width for t_context (flow + noise critic).
    config.timestep_embed_dim = 16
    # Push the learned t_context into the SteerVLA CSP hook (requires a context_smoothing checkpoint).
    config.use_vla_context_smoothing = True

    return config
