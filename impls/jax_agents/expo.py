"""Minimal JAX implementation of EXPO (Edit Policy Online).

Reference: https://arxiv.org/abs/2507.07986 .

This variant freezes the base actor as a SteerVLA / Pi0-CoT model and only
learns:

* **Edit actor** ``pi_psi(delta | s, a_base)`` -- a tanh-squashed diagonal
  Gaussian over an additive edit, scaled by ``edit_action_scale`` and applied
  on top of the VLA's ``a_base``.
* **Critic ensemble** ``Q_theta(s, a)`` (size 2 by default) with a Polyak
  target copy.
* **Temperature** ``alpha`` (optional, ``learn_temperature=True``) following
  SAC's log-temperature parameterization (see :class:`Temperature`).

Action selection at rollout is **best-of-K**:

  1. Sample ``N`` base candidates from the VLA at ``s``.
  2. For the first ``n_edit_samples`` of them, sample an edit and form
     ``clip(a_base + edit_action_scale * delta)``.
  3. Pick ``argmax_k Q_target(s, a_k)`` over the resulting ``N + n_edit_samples``
     candidates.

Differences vs. the EXPO paper that are intentional simplifications:

* The base actor is a frozen VLA: no BC / IL / DDPM update on it.
* No target edit actor (edit actor params used as-is in the bootstrap target).
* Critic bootstrap target uses a single (base, edit) pair by default; set
  ``train_best_of_n=True`` to also score K candidates at ``s'`` (expensive: K x VLA forwards).

The module-level ``get_config()`` makes this discoverable to
``--agent=jax_agents/expo.py``.
"""

from __future__ import annotations

import copy
import functools
import time
from typing import Any, Callable, Optional

import distrax
import flax
import flax.linen as nn
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


def _critic_obs_e(obs_e: jnp.ndarray, batch: dict, key: str) -> jnp.ndarray:
    """Append language label from ``batch[key]`` to encoded observation for critic-only consumption."""
    lang = batch.get(key)
    if lang is None:
        return obs_e
    return jnp.concatenate([obs_e, jnp.asarray(lang, dtype=jnp.float32)], axis=-1)


def _apply_debug_stop_reward_relabel(batch: dict) -> dict:
    """Debug task: replace env reward with ``-ego_speed`` (m/s) to encourage stopping."""
    if "ego_speed" not in batch:
        raise KeyError(
            "debug_task requires replay field 'ego_speed' (store CARLA speed m/s in main_carla)."
        )
    out = dict(batch)
    out["rewards"] = -jnp.asarray(out["ego_speed"], dtype=jnp.float32)
    return out


# --------------------------------------------------------------------------- #
# Pure-math JIT cores                                                         #
# --------------------------------------------------------------------------- #


@jax.jit
def _edit_apply(
    base_actions: jnp.ndarray,
    raw_delta: jnp.ndarray,
    edit_action_scale: jnp.ndarray,
) -> jnp.ndarray:
    """Compose final env action from VLA base + scaled tanh-Gaussian edit, clipped to [-1, 1]."""
    return jnp.clip(base_actions + edit_action_scale * raw_delta, -1.0, 1.0)


@jax.jit
def _edit_actor_loss_pure_math(
    network: TrainState,
    batch: dict,
    base_actions: jnp.ndarray,
    grad_params,
    rng: jnp.ndarray,
    alpha: jnp.ndarray,
    edit_action_scale: jnp.ndarray,
):
    """SAC-style loss for the edit actor.

    ``base_actions`` are clipped to the env action layout and come from a frozen VLA
    forward (executed eagerly outside JIT). The critic params are stop-gradient'd so
    only the edit actor params receive gradient.
    """
    obs_e = network.select("obs_encoder")(batch["observations"], params=grad_params)
    edit_input = jnp.concatenate([obs_e, base_actions], axis=-1)
    dist = network.select("edit_actor")(edit_input, params=grad_params)
    raw_delta, log_prob = dist.sample_and_log_prob(seed=rng)
    # Jacobian of the scaling ``raw_delta -> edit_action_scale * raw_delta``.
    action_dim = base_actions.shape[-1]
    log_prob = log_prob - action_dim * jnp.log(edit_action_scale)

    actions = _edit_apply(base_actions, raw_delta, edit_action_scale)
    qs = network.select("critic")(
        _critic_obs_e(obs_e, batch, "language_label"), actions,
    )
    q = jnp.min(qs, axis=0)
    actor_loss = (alpha * log_prob - q).mean()
    return actor_loss, {
        "edit_actor_loss": actor_loss,
        "edit_log_prob": log_prob.mean(),
        "q_for_edit_actor": q.mean(),
        "edit_delta_abs_mean": jnp.abs(edit_action_scale * raw_delta).mean(),
    }


@jax.jit
def _critic_loss_vla_pure_math(
    network: TrainState,
    batch: dict,
    next_actions_critic: jnp.ndarray,
    next_log_prob: jnp.ndarray,
    critic_actions: jnp.ndarray,
    grad_params,
    discount: jnp.ndarray,
    alpha: jnp.ndarray,
    backup_entropy: jnp.ndarray,
):
    """Bellman target + critic MSE.

    ``next_actions_critic`` is the env-layout action sampled as
    ``base' + edit_action_scale * delta'`` (clipped), executed eagerly outside JIT.
    ``next_log_prob`` is the edit actor log-prob at ``s'`` (already adjusted for the
    scaling Jacobian); ``backup_entropy`` is a 0/1 scalar that switches the SAC
    soft-target on/off.
    """
    next_obs_e = network.select("obs_encoder")(
        batch["next_observations"], params=network.params,
    )
    next_qs = network.select("target_critic")(
        _critic_obs_e(next_obs_e, batch, "next_language_label"), next_actions_critic,
    )
    next_q = jnp.min(next_qs, axis=0)
    next_q = next_q - backup_entropy * alpha * next_log_prob
    target_q = batch["rewards"] + discount * batch["masks"] * next_q

    obs_e = network.select("obs_encoder")(batch["observations"], params=grad_params)
    qs = network.select("critic")(
        _critic_obs_e(obs_e, batch, "language_label"), critic_actions, params=grad_params,
    )
    critic_loss = jnp.square(qs - target_q[None]).mean()
    return critic_loss, {
        "critic_loss": critic_loss,
        "q_mean": qs.mean(),
        "q_max": qs.max(),
        "q_min": qs.min(),
        "target_q": target_q.mean(),
        "next_log_prob": next_log_prob.mean(),
    }


@jax.jit
def _sample_edit_at_state(
    network: TrainState,
    observations: jnp.ndarray,
    base_actions: jnp.ndarray,
    edit_action_scale: jnp.ndarray,
    rng: jnp.ndarray,
):
    """Sample one ``(action, adjusted_log_prob)`` pair from the (live) edit actor.

    Used to materialize the bootstrap target for the critic loss; params are NOT
    stop-gradient'd here because the call sites either pass live params (target
    construction) or rely on ``jax.lax.stop_gradient`` on the outputs as needed.
    """
    obs_e = network.select("obs_encoder")(observations, params=network.params)
    edit_input = jnp.concatenate([obs_e, base_actions], axis=-1)
    dist = network.select("edit_actor")(edit_input, params=network.params)
    raw_delta, log_prob = dist.sample_and_log_prob(seed=rng)
    action_dim = base_actions.shape[-1]
    log_prob = log_prob - action_dim * jnp.log(edit_action_scale)
    actions = _edit_apply(base_actions, raw_delta, edit_action_scale)
    return actions, log_prob


# --------------------------------------------------------------------------- #
# Networks                                                                    #
# --------------------------------------------------------------------------- #


class CarlaObservationEncoder(nn.Module):
    """Encode CARLA observations: vector state or precomputed image/SigLIP embeddings."""

    observation_mode: str
    image_encoder: str = "impala"
    impala_width: int = 1
    impala_stack_sizes: tuple = (16, 32, 32)
    impala_num_blocks: int = 2
    image_mlp_hidden_dims: tuple = (512,)
    layer_norm: bool = False

    @nn.compact
    def __call__(self, observations):
        if self.observation_mode == "state":
            return observations.astype(jnp.float32)
        if self.image_encoder == "siglip":
            return observations.astype(jnp.float32)
        return ImpalaEncoder(
            width=self.impala_width,
            stack_sizes=self.impala_stack_sizes,
            num_blocks=self.impala_num_blocks,
            mlp_hidden_dims=self.image_mlp_hidden_dims,
            layer_norm=self.layer_norm,
        )(observations)


class EditActor(nn.Module):
    """Tanh-squashed diagonal Gaussian over the *edit* (delta-action) space.

    Input is ``concat([obs_e, a_base])``; output is a distribution over a
    pre-scaled edit ``raw_delta``. The caller multiplies by ``edit_action_scale``
    and adds it to ``a_base``.
    """

    hidden_dims: tuple
    action_dim: int
    log_std_min: float = -5.0
    log_std_max: float = 2.0

    @nn.compact
    def __call__(self, inputs, temperature: float = 1.0):
        x = MLP(self.hidden_dims, activate_final=True)(inputs)
        mean = nn.Dense(self.action_dim, kernel_init=default_init(0.01))(x)
        log_std = nn.Dense(self.action_dim, kernel_init=default_init(0.01))(x)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        base = distrax.MultivariateNormalDiag(
            loc=mean, scale_diag=jnp.exp(log_std) * temperature,
        )
        return distrax.Transformed(base, distrax.Block(distrax.Tanh(), ndims=1))


class Temperature(nn.Module):
    """SAC log-temperature: ``alpha = exp(log_temp)``, a single trainable scalar."""

    initial_temperature: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param(
            "log_temp",
            init_fn=lambda key: jnp.full((), jnp.log(self.initial_temperature)),
        )
        return jnp.exp(log_temp)


class Critic(nn.Module):
    """Q(s, a) with optional ensemble; layer-norm by default."""

    hidden_dims: tuple
    layer_norm: bool = True
    ensemble_size: int = 2

    def setup(self):
        mlp = ensemblize(MLP, self.ensemble_size)
        self.value_net = mlp(
            (*self.hidden_dims, 1), activate_final=False, layer_norm=self.layer_norm,
        )

    def __call__(self, observations, actions):
        inputs = jnp.concatenate([observations, actions], axis=-1)
        return self.value_net(inputs).squeeze(-1)


# --------------------------------------------------------------------------- #
# Agent                                                                       #
# --------------------------------------------------------------------------- #
class EXPOAgent(flax.struct.PyTreeNode):
    """Minimal JAX EXPO agent (frozen VLA base + learned edit actor + critic)."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()
    vla_sample_fn: Any = nonpytree_field()  # (obs, noise) -> base actions (env layout)
    openpi_train_config: Any = nonpytree_field()
    steervla_actor: Any = nonpytree_field()
    use_edit_actor_at_rollout: bool = nonpytree_field(default=True)

    def set_edit_actor_rollout_enabled(self, enabled: bool) -> None:
        """Toggle whether rollout uses the edit actor (VLA-only when False)."""
        object.__setattr__(self, "use_edit_actor_at_rollout", bool(enabled))

    @staticmethod
    def _as_jax_pytree(x):
        """Convert leaves to JAX arrays (no-op for existing JAX arrays)."""
        return jax.tree.map(lambda y: jnp.asarray(y), x)

    # ----- shape helpers (mirror dsrl.py) --------------------------------- #

    def _model_action_horizon(self) -> int:
        return int(self.config.get("action_horizon", self.config.get("vla_action_horizon", 10)))

    def _model_action_dim(self) -> int:
        return int(self.config.get("actor_action_dim", PI05_ACTION_DIM))

    def _env_action_horizon(self) -> int:
        return int(self.config.get("vla_action_horizon", self.config.get("action_horizon", 10)))

    def _env_action_dim(self) -> int:
        return int(self.config.get("vla_action_dim", 4))

    def _flat_noise_dim(self) -> int:
        return self._model_action_horizon() * self._model_action_dim()

    def _flat_env_action_dim(self) -> int:
        return self._env_action_horizon() * self._env_action_dim()

    def _as_flat_noise(self, noise: jnp.ndarray) -> jnp.ndarray:
        x = jnp.asarray(noise, dtype=jnp.float32)
        flat_dim = self._flat_noise_dim()
        if x.ndim == 3:
            ah = self._model_action_horizon()
            ad = self._model_action_dim()
            return x[:, :ah, :ad].reshape(x.shape[0], flat_dim)
        if x.ndim == 2 and x.shape[-1] == flat_dim:
            return x
        raise ValueError(
            f"Cannot map noise shape {tuple(x.shape)} to flat noise dim {flat_dim}."
        )

    def _as_noise_chunk(self, noise: jnp.ndarray) -> jnp.ndarray:
        flat = self._as_flat_noise(noise)
        return flat.reshape(flat.shape[0], self._model_action_horizon(), self._model_action_dim())

    def _clip_actions_to_env(self, actions: jnp.ndarray) -> jnp.ndarray:
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
            f"Cannot clip actions shape {tuple(x.shape)} to env flat dim {flat_env}."
        )

    def _as_critic_actions(self, actions: jnp.ndarray) -> jnp.ndarray:
        return self._clip_actions_to_env(actions)

    # ----- VLA forward (eager) -------------------------------------------- #

    def _vla_base_sample(
        self,
        observations: jnp.ndarray,
        rng: jnp.ndarray,
    ) -> jnp.ndarray:
        """Sample a fresh base action chunk from the (frozen) VLA at ``observations``.

        ``vla_sample_fn`` expects ``(observations, noise)``; we feed iid Gaussian
        noise with shape ``(B, flat_noise_dim)`` (the SteerVLAActor server / local
        wrapper unflattens internally).
        """
        if self.vla_sample_fn is None:
            raise RuntimeError("EXPO requires ``vla_sample_fn`` to be set (frozen VLA base actor).")
        batch_size = observations.shape[0]
        noise = jax.random.normal(rng, (batch_size, self._flat_noise_dim()), dtype=jnp.float32)
        t0 = time.time()
        base = self.vla_sample_fn(observations, noise)
        base = jnp.asarray(base, dtype=jnp.float32)
        jax.block_until_ready(base)
        print(f"[DEBUG - expo] vla_sample_fn time: {time.time() - t0:.3f}s")
        return jax.lax.stop_gradient(self._clip_actions_to_env(base))

    def _vla_base_sample_openpi(
        self,
        observations: jnp.ndarray,
        openpi_observations,
        rng: jnp.ndarray,
    ) -> jnp.ndarray:
        """Same as :meth:`_vla_base_sample`, but go through the attached ``steervla_actor``.

        This path matches dsrl.py's ``_vla_forward`` (uses the OpenPI observation
        struct already attached to the replay batch). Falls back to
        :meth:`_vla_base_sample` when no ``steervla_actor`` is attached.
        """
        if self.steervla_actor is None:
            return self._vla_base_sample(observations, rng)

        rng_noise, rng_act = jax.random.split(rng)
        batch_size = observations.shape[0]
        noise = jax.random.normal(
            rng_noise, (batch_size, self._flat_noise_dim()), dtype=jnp.float32,
        )
        noise_chunk = self._as_noise_chunk(noise)
        openpi_obs = self._as_jax_pytree(openpi_observations)
        t0 = time.time()
        actions = self.steervla_actor._sample_actions(
            rng_act,
            openpi_obs,
            noise=noise_chunk,
            image_keys=tuple(self.config.get("image_keys", ("base_0_rgb",))),
            num_steps=int(
                self.config.get(
                    "vla_update_flow_steps", self.config.get("flow_steps", 10),
                )
            ),
        )
        actions = self.steervla_actor.postprocess_sampled_trajectory(
            actions, observation_state=openpi_obs.state,
        )
        jax.block_until_ready(actions)
        print(f"[DEBUG - expo] steervla_actor sample time: {time.time() - t0:.3f}s")
        return jax.lax.stop_gradient(self._clip_actions_to_env(actions))

    # ----- sampling -------------------------------------------------------- #

    def _encode_obs(self, params, observations):
        return self.network.select("obs_encoder")(observations, params=params)

    @jax.jit
    def _edit_step(
        self,
        observations: jnp.ndarray,
        base_actions: jnp.ndarray,
        seed: jnp.ndarray,
        temperature: float,
    ) -> jnp.ndarray:
        """JIT inner core of rollout sampling: encode obs, draw edit, compose action."""
        obs_e = self._encode_obs(self.network.params, observations)
        edit_input = jnp.concatenate([obs_e, base_actions], axis=-1)
        dist = self.network.select("edit_actor")(edit_input, temperature=temperature)
        raw_delta = dist.sample(seed=seed)
        edit_action_scale = jnp.asarray(self.config["edit_action_scale"], dtype=jnp.float32)
        return _edit_apply(base_actions, raw_delta, edit_action_scale)

    def sample_actions(self, observations, seed=None, temperature=1.0):
        """Fallback when no VLA is attached: edit on a zero base action.

        EXPO doesn't define a meaningful policy without the frozen VLA, so this
        path effectively returns the edit actor's standalone sample. Use
        :meth:`sample_actions_with_vla` for the actual EXPO rollout policy.
        """
        seed = seed if seed is not None else self.rng
        seed, sub = jax.random.split(seed)
        batch_size = observations.shape[0]
        base_actions = jnp.zeros(
            (batch_size, self._flat_env_action_dim()), dtype=jnp.float32,
        )
        return self._edit_step(observations, base_actions, sub, temperature)

    def _critic_lang_dim(self) -> int:
        """Width of the language-feedback vector the critic was sized for."""
        mode = str(self.config.get("critic_feedback_mode", "commentary_bow"))
        if mode == "none":
            return 0
        if mode == "action_delta":
            return int(self.config.get("critic_action_dim", 4))
        return int(self.config.get("language_label_dim", 119))

    def _critic_obs_at_rollout(self, observations: jnp.ndarray) -> jnp.ndarray:
        """Encode obs and append zero language label so the critic sees the trained width."""
        obs_e = self._encode_obs(self.network.params, observations)
        lang_dim = self._critic_lang_dim()
        if lang_dim <= 0:
            return obs_e
        zero_lang = jnp.zeros((observations.shape[0], lang_dim), dtype=obs_e.dtype)
        return jnp.concatenate([obs_e, zero_lang], axis=-1)

    @jax.jit
    def _score_target_q(
        self,
        critic_obs: jnp.ndarray,
        candidates: jnp.ndarray,
    ) -> jnp.ndarray:
        """Score ``candidates`` (B, K, env_dim) under target_critic; return per-row argmax index."""
        B, K, env_dim = candidates.shape
        critic_obs_rep = jnp.repeat(critic_obs, K, axis=0)
        cand_flat = candidates.reshape(B * K, env_dim)
        qs = self.network.select("target_critic")(critic_obs_rep, cand_flat)
        q = jnp.min(qs, axis=0).reshape(B, K)
        return jnp.argmax(q, axis=1)

    def sample_actions_with_vla(self, observations, seed=None, temperature=1.0):
        """EXPO rollout policy with best-of-K candidate selection.

        Generates ``N`` base actions from the frozen VLA at ``s``, applies an edit
        to the first ``n_edit_samples`` of them, and picks
        ``argmax_k Q_target(s, a_k)`` over the resulting ``N + n_edit_samples``
        candidates. ``N=1, n_edit_samples=1`` reduces to "one base vs. one edit,
        pick best Q"; ``N=1, n_edit_samples=0`` reduces to "VLA only".
        """
        if self.vla_sample_fn is None:
            return self.sample_actions(observations, seed=seed, temperature=temperature)

        seed = seed if seed is not None else self.rng
        seed, vla_seed = jax.random.split(seed)
        if not self.use_edit_actor_at_rollout:
            return self._vla_base_sample(observations, vla_seed)

        B = observations.shape[0]
        N = max(1, int(self.config.get("N", 1)))
        n_edit = max(0, int(self.config.get("n_edit_samples", 1)))
        n_edit = min(n_edit, N)

        seed, edit_seed = jax.random.split(seed)

        # === Sample N base candidates per state (single batched VLA call). === #
        if N > 1:
            obs_for_vla = jnp.repeat(observations, N, axis=0)
        else:
            obs_for_vla = observations
        base = self._vla_base_sample(obs_for_vla, vla_seed)
        env_dim = base.shape[-1]
        base = base.reshape(B, N, env_dim)

        # === Edit the first n_edit base candidates per state. === #
        if n_edit > 0:
            base_to_edit = base[:, :n_edit, :].reshape(B * n_edit, env_dim)
            obs_for_edit = jnp.repeat(observations, n_edit, axis=0)
            edited = self._edit_step(obs_for_edit, base_to_edit, edit_seed, temperature)
            edited = edited.reshape(B, n_edit, env_dim)
            candidates = jnp.concatenate([base, edited], axis=1)
        else:
            candidates = base

        K = candidates.shape[1]
        if K <= 1:
            return candidates[:, 0]

        critic_obs = self._critic_obs_at_rollout(observations)
        best_idx = self._score_target_q(critic_obs, candidates)
        batch_idx = jnp.arange(B)
        return candidates[batch_idx, best_idx]

    @jax.jit
    def sample_actions_dagger(self, observations):
        """DAgger is not implemented for EXPO (no BC update on the base actor)."""
        return self.sample_actions(observations)

    # ----- losses --------------------------------------------------------- #

    def edit_actor_loss_vla(self, batch, grad_params, rng, base_actions):
        """SAC-style edit actor loss; ``base_actions`` is the eager VLA forward at ``s``."""
        alpha = jnp.asarray(self.config["alpha"], dtype=jnp.float32)
        edit_action_scale = jnp.asarray(self.config["edit_action_scale"], dtype=jnp.float32)
        return _edit_actor_loss_pure_math(
            self.network, batch, base_actions, grad_params, rng, alpha, edit_action_scale,
        )

    def critic_loss_vla(self, batch, grad_params, rng, next_actions, next_log_prob):
        """Critic loss with bootstrap target action = ``clip(VLA(s') + edit_actor(s', .))``."""
        critic_actions = self._clip_actions_to_env(batch["actions"])
        discount = jnp.asarray(self.config["discount"], dtype=jnp.float32)
        alpha = jnp.asarray(self.config["alpha"], dtype=jnp.float32)
        backup_entropy = jnp.asarray(
            float(bool(self.config.get("backup_entropy", True))), dtype=jnp.float32,
        )
        return _critic_loss_vla_pure_math(
            self.network,
            batch,
            next_actions,
            next_log_prob,
            critic_actions,
            grad_params,
            discount,
            alpha,
            backup_entropy,
        )

    def _current_alpha(self, grad_params=None):
        """Return ``(alpha_grad, alpha_sg)``: shared current α with and without gradient flow.

        With ``learn_temperature=False`` both are the fixed config ``alpha``.
        """
        if bool(self.config.get("learn_temperature", False)):
            alpha_grad = self.network.select("temperature")(params=grad_params)
            alpha_sg = jax.lax.stop_gradient(alpha_grad)
            return alpha_grad, alpha_sg
        a = jnp.asarray(self.config["alpha"], dtype=jnp.float32)
        return a, a

    def total_loss_vla(self, batch, grad_params, rng=None, vla_cache=None):
        """Sum of EXPO losses; ``update_with_vla`` passes a precomputed ``vla_cache``.

        Inlines the edit-actor sample so its ``log_prob`` can be shared between the
        edit-actor loss (entropy-regularized Q maximization) and the temperature
        loss (constrain entropy toward ``target_entropy``).
        """
        if bool(self.config.get("debug_task", False)):
            batch = _apply_debug_stop_reward_relabel(batch)
        rng = rng if rng is not None else self.rng
        rng, ea_rng = jax.random.split(rng, 2)

        t0 = time.time()
        edit_action_scale = jnp.asarray(self.config["edit_action_scale"], dtype=jnp.float32)
        discount = jnp.asarray(self.config["discount"], dtype=jnp.float32)
        backup_entropy = jnp.asarray(
            float(bool(self.config.get("backup_entropy", True))), dtype=jnp.float32,
        )

        alpha_grad, alpha_sg = self._current_alpha(grad_params=grad_params)

        # --- Edit actor forward (shared) --- #
        base_actions = vla_cache["base_actions"]
        action_dim = base_actions.shape[-1]
        obs_e = self.network.select("obs_encoder")(batch["observations"], params=grad_params)
        edit_input = jnp.concatenate([obs_e, base_actions], axis=-1)
        dist = self.network.select("edit_actor")(edit_input, params=grad_params)
        raw_delta, raw_log_prob = dist.sample_and_log_prob(seed=ea_rng)
        log_prob = raw_log_prob - action_dim * jnp.log(edit_action_scale)
        actions = _edit_apply(base_actions, raw_delta, edit_action_scale)
        qs_actor = self.network.select("critic")(
            _critic_obs_e(obs_e, batch, "language_label"), actions,
        )
        q_actor = jnp.min(qs_actor, axis=0)
        ea_loss = (alpha_sg * log_prob - q_actor).mean()

        # --- Critic loss --- #
        next_obs_e = self.network.select("obs_encoder")(
            batch["next_observations"], params=self.network.params,
        )
        next_qs = self.network.select("target_critic")(
            _critic_obs_e(next_obs_e, batch, "next_language_label"),
            vla_cache["next_actions_critic"],
        )
        next_q = jnp.min(next_qs, axis=0)
        next_q = next_q - backup_entropy * alpha_sg * vla_cache["next_log_prob"]
        target_q = batch["rewards"] + discount * batch["masks"] * next_q

        critic_actions = self._clip_actions_to_env(batch["actions"])
        qs_train = self.network.select("critic")(
            _critic_obs_e(obs_e, batch, "language_label"), critic_actions, params=grad_params,
        )
        c_loss = jnp.square(qs_train - target_q[None]).mean()

        # --- Temperature loss --- #
        if bool(self.config.get("learn_temperature", False)):
            target_entropy = jnp.asarray(self.config["target_entropy"], dtype=jnp.float32)
            entropy_sg = jax.lax.stop_gradient(-log_prob.mean())
            t_loss = alpha_grad * (entropy_sg - target_entropy)
        else:
            t_loss = jnp.asarray(0.0, dtype=jnp.float32)

        print(f"[DEBUG - expo] Total loss vla time: {time.time() - t0:.3f}s")

        info = {
            "edit_actor_vla/edit_actor_loss": ea_loss,
            "edit_actor_vla/edit_log_prob": log_prob.mean(),
            "edit_actor_vla/q_for_edit_actor": q_actor.mean(),
            "edit_actor_vla/edit_delta_abs_mean": jnp.abs(edit_action_scale * raw_delta).mean(),
            "critic/critic_loss": c_loss,
            "critic/q_mean": qs_train.mean(),
            "critic/q_max": qs_train.max(),
            "critic/q_min": qs_train.min(),
            "critic/target_q": target_q.mean(),
            "critic/next_log_prob": vla_cache["next_log_prob"].mean(),
            "temp/alpha": alpha_sg,
            "temp/entropy_mean": -log_prob.mean(),
            "temp/temp_loss": t_loss,
            "edit_actor_loss_vla": ea_loss,
            "critic_loss_vla": c_loss,
        }
        combined = ea_loss + c_loss + t_loss
        info["total_loss_vla"] = combined
        return combined, info

    def target_update(self, network):
        new_target = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["tau"] + tp * (1 - self.config["tau"]),
            network.params["modules_critic"],
            network.params["modules_target_critic"],
        )
        network.params["modules_target_critic"] = new_target

    @jax.jit
    def update(self, batch):
        """No-op fallback: EXPO requires the VLA path. ``update_with_vla`` is the real entry."""
        raise NotImplementedError(
            "EXPO requires a frozen VLA base actor; call ``update_with_vla`` (set ``steervla.enabled=True``)."
        )

    @jax.jit
    def update_dagger(self, batch):
        """EXPO has no BC/IL update on the base actor."""
        raise NotImplementedError("EXPO does not learn the base actor; ``update_dagger`` is not supported.")

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
        """Eager VLA forwards for both losses + the edit-actor sample at s'.

        Default: one VLA forward at ``s`` (for the edit-actor loss) and one at
        ``s'`` (for the critic bootstrap), plus one edit-actor sample at ``s'``.

        With ``train_best_of_n=True``: ``s'`` uses the same best-of-K target as
        rollout (sample ``N`` base candidates from VLA, edit the first
        ``n_edit_samples``, pick argmax target_critic). This multiplies the
        per-update VLA cost by ``N``.
        """
        train_best_of_n = bool(self.config.get("train_best_of_n", False))
        N = max(1, int(self.config.get("N", 1)))
        n_edit = min(max(0, int(self.config.get("n_edit_samples", 1))), N)

        rng, base_s_rng, base_sp_rng, edit_sp_rng = jax.random.split(rng, 4)
        edit_action_scale = jnp.asarray(self.config["edit_action_scale"], dtype=jnp.float32)

        # Base actions at current state (for edit-actor loss): single VLA call.
        base_actions = self._vla_base_sample_openpi(
            batch["observations"], batch["openpi_observation"], base_s_rng,
        )

        if not train_best_of_n:
            next_base = self._vla_base_sample_openpi(
                batch["next_observations"], batch["next_openpi_observation"], base_sp_rng,
            )
            next_actions, next_log_prob = _sample_edit_at_state(
                self.network, batch["next_observations"], next_base, edit_action_scale, edit_sp_rng,
            )
        else:
            next_actions, next_log_prob = self._best_of_n_target_at_sprime(
                batch, N, n_edit, edit_action_scale, base_sp_rng, edit_sp_rng,
            )

        next_actions = jax.lax.stop_gradient(next_actions)
        next_log_prob = jax.lax.stop_gradient(next_log_prob)

        return {
            "base_actions": base_actions,
            "next_actions_critic": next_actions,
            "next_log_prob": next_log_prob,
        }

    def _best_of_n_target_at_sprime(
        self,
        batch,
        N: int,
        n_edit: int,
        edit_action_scale: jnp.ndarray,
        base_sp_rng: jnp.ndarray,
        edit_sp_rng: jnp.ndarray,
    ):
        """Best-of-K critic target at ``s'``: sample N VLA bases, edit n_edit, argmax target Q.

        Returns ``(next_actions, next_log_prob)`` where ``next_log_prob`` is the
        adjusted edit-actor log-prob (Jacobian-corrected) for the edited
        candidates and zero for picked pure-VLA candidates. The mask is just an
        approximation -- for the SAC-style soft target we only need an entropy
        estimate; using zero for the pure-VLA branch slightly under-counts the
        bonus there, which is conservative.
        """
        B = batch["next_observations"].shape[0]
        next_obs = batch["next_observations"]

        # Sample N base candidates per next-state via one batched VLA call.
        if N > 1:
            obs_rep = jnp.repeat(next_obs, N, axis=0)
            openpi_rep = jax.tree.map(
                lambda x: jnp.repeat(x, N, axis=0), batch["next_openpi_observation"],
            )
        else:
            obs_rep = next_obs
            openpi_rep = batch["next_openpi_observation"]
        base = self._vla_base_sample_openpi(obs_rep, openpi_rep, base_sp_rng)
        env_dim = base.shape[-1]
        base = base.reshape(B, N, env_dim)

        # Edit the first n_edit base candidates per next-state.
        if n_edit > 0:
            base_to_edit = base[:, :n_edit, :].reshape(B * n_edit, env_dim)
            obs_for_edit = jnp.repeat(next_obs, n_edit, axis=0)
            edited, edit_log_prob = _sample_edit_at_state(
                self.network, obs_for_edit, base_to_edit, edit_action_scale, edit_sp_rng,
            )
            edited = edited.reshape(B, n_edit, env_dim)
            edit_log_prob = edit_log_prob.reshape(B, n_edit)
            candidates = jnp.concatenate([base, edited], axis=1)
            # Per-candidate log_prob: 0 for pure-VLA branches, adjusted log_prob for edited.
            cand_log_prob = jnp.concatenate(
                [jnp.zeros((B, N), dtype=edit_log_prob.dtype), edit_log_prob], axis=1,
            )
        else:
            candidates = base
            cand_log_prob = jnp.zeros((B, N), dtype=base.dtype)

        # Score with target_critic at s' (use next_language_label if available).
        next_obs_e = self._encode_obs(self.network.params, next_obs)
        next_critic_obs = _critic_obs_e(next_obs_e, batch, "next_language_label")
        best_idx = self._score_target_q(next_critic_obs, candidates)
        batch_idx = jnp.arange(B)
        next_actions = candidates[batch_idx, best_idx]
        next_log_prob = cand_log_prob[batch_idx, best_idx]
        return next_actions, next_log_prob

    def update_with_vla(self, batch):
        """Flax update via :meth:`total_loss_vla` with eager VLA forwards + jitted gradient core."""
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
        if obs_mode not in ("state", "image"):
            raise ValueError(
                f"observation_mode must be 'state' or 'image', got {obs_mode!r}"
            )
        image_encoder = str(config.get("image_encoder", "impala")).lower()
        if image_encoder not in ("impala", "siglip"):
            raise ValueError(f"image_encoder must be 'impala' or 'siglip', got {image_encoder!r}")
        if obs_mode != "image" and image_encoder == "siglip":
            raise ValueError("image_encoder='siglip' requires observation_mode='image'.")

        if vla_sample_fn is not None:
            env_action_dim = int(config.get("vla_action_dim", 4)) * int(
                config.get("vla_action_horizon", config.get("action_horizon", 10))
            )
        else:
            env_action_dim = ex_actions.shape[-1]

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
            if bool(config.get("siglip_include_prompt_subtask", False)):
                embed_dim = single * 3
            else:
                embed_dim = single
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
        ex_env_actions = jnp.zeros(batch_shape + (env_action_dim,), dtype=jnp.float32)
        ex_edit_input = jnp.zeros(batch_shape + (embed_dim + env_action_dim,), dtype=jnp.float32)

        edit_actor_def = EditActor(
            hidden_dims=tuple(config["actor_hidden_dims"]),
            action_dim=env_action_dim,
        )
        critic_def = Critic(
            hidden_dims=tuple(config["critic_hidden_dims"]),
            layer_norm=config["layer_norm"],
            ensemble_size=config["critic_ensemble"],
        )
        temperature_def = Temperature(
            initial_temperature=float(config.get("init_temperature", 1.0)),
        )

        # Resolve target_entropy: ``None`` / ``'auto'`` → adjusted SAC default
        # ``-action_dim/2 + action_dim * log(edit_action_scale)`` (matches expo_original).
        edit_action_scale_f = float(config.get("edit_action_scale", 1.0))
        target_entropy_cfg = config.get("target_entropy", None)
        if (
            target_entropy_cfg is None
            or (isinstance(target_entropy_cfg, str) and target_entropy_cfg.lower() == "auto")
        ):
            import math as _math

            target_entropy_val = (
                -env_action_dim / 2.0
                + env_action_dim * _math.log(max(edit_action_scale_f, 1e-8))
            )
        else:
            target_entropy_val = float(target_entropy_cfg)
        config = dict(config)
        config["target_entropy"] = target_entropy_val

        networks = {
            "obs_encoder": (obs_encoder_def, (ex_observations,)),
            "edit_actor": (edit_actor_def, (ex_edit_input,)),
            "critic": (critic_def, (ex_critic_embedded, ex_env_actions)),
            "target_critic": (
                copy.deepcopy(critic_def),
                (ex_critic_embedded, ex_env_actions),
            ),
            "temperature": (temperature_def, ()),
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
            agent_name="expo",
            lr=3e-4,
            batch_size=256,
            actor_hidden_dims=(256, 256),
            critic_hidden_dims=(256, 256),
            critic_ensemble=2,
            layer_norm=True,
            discount=0.99,
            tau=0.005,
            # SAC-style entropy coefficient on edit actor log-prob. Used directly when
            # ``learn_temperature=False``; ignored otherwise (replaced by exp(log_temp)).
            alpha=0.1,
            # Learnable temperature (SAC). When True, ``alpha = exp(log_temp)`` is
            # learned to drive ``entropy`` toward ``target_entropy``.
            learn_temperature=True,
            init_temperature=1.0,
            # Auto-resolved at agent.create from edit_action_scale and env_action_dim:
            #   target_entropy = -dim/2 + dim * log(edit_action_scale)
            # Override with a float, or set "auto" / leave None to keep the default.
            target_entropy=None,
            # Scale on edit actor delta: a = clip(a_base + edit_action_scale * tanh_sample).
            edit_action_scale=0.5,
            # Bellman target uses SAC soft entropy bonus when True.
            backup_entropy=True,
            # Best-of-K action selection (paper EXPO §3.3).
            #   N: number of VLA base candidates per state at rollout / (optional) target.
            #   n_edit_samples: number of those base candidates to additionally edit
            #     (must be <= N). Candidate pool size at selection = N + n_edit_samples.
            N=1,
            n_edit_samples=1,
            # When True, use best-of-K to construct the critic target at s' (paper's
            # ``sample_batch_actions``). Costs N x VLA forwards per gradient step on s'.
            train_best_of_n=False,
            observation_mode="state",
            image_encoder="impala",
            image_impala_width=1,
            image_impala_stack_sizes=(16, 32, 32),
            image_impala_num_blocks=2,
            image_mlp_hidden_dims=(512,),
            siglip_model_id="google/siglip2-so400m-patch14-384",
            siglip_embed_dim=1152,
            siglip_include_prompt_subtask=False,
            siglip_device=None,
            # Online-loop knobs (used by main_carla.py).
            warmup_steps=250,
            # Train edit/critic but keep VLA-only rollout until step > warmup_expo_steps.
            # Must be > warmup_steps.
            warmup_expo_steps=500,
            updates_per_step=1,
            enable_updates=True,
            buffer_capacity=100_000,
            image_log_curr_interval=1000,
            training_gpu_rank=-1,
            frame_stack=ml_collections.config_dict.placeholder(int),
            dataset_class="GCDataset",
            # VLA / Pi0-CoT knobs (shared with dsrl for symmetry).
            vla_cot_temperature=0.0,
            vla_cot_replay_reasoning=True,
            vla_sample_actions_num_steps=10,
            vla_update_flow_steps=10,
            vla_sample_actions_low_memory=True,
            vla_sample_actions_jit_denoise_steps=False,
            vla_action_horizon=10,
            vla_action_dim=4,
            vla_output_action_format="DELTA_XY_T_DELTA_XY_SPACE",
            # Pi0 / model space (used to size the noise feed into ``vla_sample_fn``).
            actor_action_dim=32,
            action_horizon=10,
            critic_action_dim=4,
            # Critic-side language feedback (see dsrl.py for full meaning).
            language_label_dim=119,
            critic_feedback_mode="commentary_bow",
            # EXPO is RL-only; the base actor is frozen so DAgger / pure RL flow paths are not used.
            online_training_mode="rl",
            debug_task=False,
            # SteerVLA debug knobs (forwarded to the actor; harmless if disabled).
            debug_noise=True,
            debug_noise_samples=15,
            debug_noise_log_every_n_steps=5,
            use_best_noise=True,
        )
    )
    return config
