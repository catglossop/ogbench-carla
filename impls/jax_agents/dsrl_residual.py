"""DSRL residual sub-agent (from the routing-commands stack).

NOTE: this is **not** ``jax_agents.sac_residual``. Two independent residual
implementations arrived from different branches and they are not interchangeable:

* ``jax_agents.sac_residual`` (surya/rl-token) is a standalone, self-contained SAC --
  its own critic, ``StateHead``, and on-the-fly candidate selection
  (``select_action_otf``). It is registered in ``jax_agents/__init__.py`` as
  ``agents["sac_residual"]`` and selected with ``agent_name="sac_residual"``
  (see ``impls/configs/steervla_residual_config.py``).

* This module is a residual actor **attached to** :class:`~jax_agents.dsrl.DSRLAgent`
  (``DSRLAgent.sac_residual_agent``). It has no critic of its own -- it scores against
  DSRL's, or a critic pretrained by ``impls/pretrain_critic.py`` -- and adds the DAgger
  paths (``update_actor_dagger``, ``_joint_dagger_apply_step``) and observation
  normalization. It is selected with ``agent_name="dsrl"`` plus
  ``online_training_mode="sac_residual"`` / ``"dagger_residual"``
  (see ``impls/configs/pi0_residual_sac_config.py``).

The class is still called ``SACResidualAgent``; the module name is what distinguishes
them. Import it explicitly as ``from jax_agents.dsrl_residual import SACResidualAgent``.
"""

from __future__ import annotations

from typing import Any

import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP, default_init


# --------------------------------------------------------------------------- #
# Networks                                                                    #
# --------------------------------------------------------------------------- #


class ResidualActor(nn.Module):
    """Tanh-squashed diagonal Gaussian residual on top of a base action.

    Input ``obs_e`` is the **plain** observation embedding (no language label);
    the language label is critic-only in DSRL. ``base_action`` is the flat env
    action chunk that the base policy produced for ``obs``. The tanh-squashed
    sample is scaled by ``residual_action_scale`` *outside* this module so the
    tanh output stays in ``(-1, 1)``.
    """

    hidden_dims: tuple
    action_dim: int
    log_std_min: float = -20.0
    log_std_max: float = 2.0
    layer_norm: bool = False

    @nn.compact
    def __call__(self, obs_e, base_action, temperature: float = 1.0):
        x = jnp.concatenate([obs_e, base_action], axis=-1)
        x = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)(x)
        mean = nn.Dense(self.action_dim, kernel_init=default_init(0.01))(x)
        log_std = nn.Dense(self.action_dim, kernel_init=default_init(0.01))(x)
        # Tanh-based soft clamping (SpinUp / Denis Yarats) — keeps gradients flowing
        # even when log_std is very small; hard clip would kill gradients at the boundary.
        log_std = jnp.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
        base = distrax.MultivariateNormalDiag(
            loc=mean, scale_diag=jnp.exp(log_std) * temperature
        )
        return distrax.Transformed(base, distrax.Block(distrax.Tanh(), ndims=1))


# --------------------------------------------------------------------------- #
# Jit kernels                                                                  #
# --------------------------------------------------------------------------- #


@jax.jit
def _sample_action_jit(
    network: TrainState,
    obs_e: jnp.ndarray,
    base_action: jnp.ndarray,
    seed: jnp.ndarray,
    temperature: jnp.ndarray,
    scale: jnp.ndarray,
    clip_bound: jnp.ndarray,
):
    """Sample ``action = clip(base + residual * scale, ±clip_bound)`` (±inf = no clip)."""
    dist = network.select("residual_actor")(obs_e, base_action, temperature=temperature)
    residual_norm = dist.sample(seed=seed)
    residual = residual_norm * scale
    return jnp.clip(base_action + residual, -clip_bound, clip_bound), residual


@jax.jit
def _sample_and_log_prob_jit(
    network: TrainState,
    obs_e: jnp.ndarray,
    base_action: jnp.ndarray,
    seed: jnp.ndarray,
    temperature: jnp.ndarray,
    scale: jnp.ndarray,
    clip_bound: jnp.ndarray,
):
    """Sample and return ``(action, residual, log_prob)`` for use in SAC TD targets."""
    dist = network.select("residual_actor")(obs_e, base_action, temperature=temperature)
    residual_norm, log_prob = dist.sample_and_log_prob(seed=seed)
    residual = residual_norm * scale
    action = jnp.clip(base_action + residual, -clip_bound, clip_bound)
    return action, residual, log_prob


@jax.jit
def _residual_actor_apply_step(
    network: TrainState,
    obs_e_sg: jnp.ndarray,
    base_action: jnp.ndarray,
    critic_obs_e_sg: jnp.ndarray,
    dsrl_network: TrainState,
    rng: jnp.ndarray,
    alpha: jnp.ndarray,
    scale: jnp.ndarray,
    clip_bound: jnp.ndarray,
):
    """SAC actor loss using the DSRL critic as a frozen Q-function."""
    dsrl_params_sg = jax.tree.map(jax.lax.stop_gradient, dsrl_network.params)

    def loss_fn(grad_params):
        dist = network.select("residual_actor")(obs_e_sg, base_action, params=grad_params)
        residual_norm, log_prob = dist.sample_and_log_prob(seed=rng)
        residual = residual_norm * scale
        action = jnp.clip(base_action + residual, -clip_bound, clip_bound)
        qs = dsrl_network.apply_fn(
            {"params": dsrl_params_sg}, critic_obs_e_sg, action, name="critic"
        )
        q = jnp.min(qs, axis=0)
        loss = (alpha * log_prob - q).mean()
        info = {
            "residual_actor_loss": loss,
            "residual_entropy": -log_prob.mean(),
            "residual_log_prob": log_prob.mean(),
            "residual_q_mean": q.mean(),
            "residual_q_min": q.min(),
            "residual_q_max": q.max(),
            "residual_abs_mean": jnp.abs(residual).mean(),
            "residual_abs_max": jnp.abs(residual).max(),
        }
        return loss, info

    return network.apply_loss_fn(loss_fn=loss_fn)


@jax.jit
def _dagger_actor_apply_step(
    network: TrainState,
    obs_e_sg: jnp.ndarray,
    base_action: jnp.ndarray,
    expert_action: jnp.ndarray,
    scale: jnp.ndarray,
    clip_bound: jnp.ndarray,
):
    """DAgger residual loss: MSE between ``base + tanh(loc) * scale`` and expert.

    Uses the deterministic mean of the tanh-Gaussian (``temperature=0`` → sampling
    collapses to ``tanh(loc)``) so the supervised signal is noise-free.
    """

    def loss_fn(grad_params):
        dist = network.select("residual_actor")(
            obs_e_sg, base_action, temperature=0.0, params=grad_params,
        )
        # With temperature=0 the underlying Gaussian has zero scale, so sample is
        # deterministic: tanh(loc). Any seed works.
        residual_norm = dist.sample(seed=jax.random.PRNGKey(0))
        residual = residual_norm * scale
        predicted = jnp.clip(base_action + residual, -clip_bound, clip_bound)
        diff = predicted - expert_action
        loss = jnp.mean(jnp.square(diff))
        base_diff = base_action - expert_action
        info = {
            "dagger_residual_loss": loss,
            "dagger_residual_predicted_mse": jnp.mean(jnp.square(diff)),
            "dagger_residual_base_mse": jnp.mean(jnp.square(base_diff)),
            "dagger_residual_abs_mean": jnp.abs(residual).mean(),
            "dagger_residual_abs_max": jnp.abs(residual).max(),
            "dagger_base_diff_hist": base_diff.ravel(),
            "dagger_pred_diff_hist": diff.ravel(),
        }
        return loss, info

    return network.apply_loss_fn(loss_fn=loss_fn)


@jax.jit
def _joint_dagger_apply_step(
    dsrl_network: TrainState,
    residual_network: TrainState,
    observations: jnp.ndarray,
    base_action: jnp.ndarray,
    expert_action: jnp.ndarray,
    scale: jnp.ndarray,
    clip_bound: jnp.ndarray,
):
    """Joint DAgger update of DSRL ``obs_encoder`` + the residual MLP.

    Same loss as :func:`_dagger_actor_apply_step`, but the gradient also flows
    through DSRL's observation encoder (so its image CNN learns features useful
    for residual prediction). Non-encoder DSRL params receive zero gradient;
    their Adam moments are zeroed out too so the optimizer state stays clean.
    """

    def loss_fn(params):
        dsrl_params, residual_params = params
        obs_e = dsrl_network.apply_fn(
            {"params": dsrl_params}, observations, name="obs_encoder",
        )
        dist = residual_network.apply_fn(
            {"params": residual_params}, obs_e, base_action,
            temperature=0.0, name="residual_actor",
        )
        residual_norm = dist.sample(seed=jax.random.PRNGKey(0))
        residual = residual_norm * scale
        predicted = jnp.clip(base_action + residual, -clip_bound, clip_bound)
        diff = predicted - expert_action
        loss = jnp.mean(jnp.square(diff))
        base_diff = base_action - expert_action
        info = {
            "dagger_residual_loss": loss,
            "dagger_residual_predicted_mse": loss,
            "dagger_residual_base_mse": jnp.mean(jnp.square(base_diff)),
            "dagger_residual_abs_mean": jnp.abs(residual).mean(),
            "dagger_residual_abs_max": jnp.abs(residual).max(),
            "dagger_base_diff_hist": base_diff.ravel(),
            "dagger_pred_diff_hist": diff.ravel(),
        }
        return loss, info

    (_loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        (dsrl_network.params, residual_network.params)
    )
    g_dsrl, g_res = grads

    # Zero out gradients for all DSRL modules except obs_encoder so Adam's
    # moment estimates for flow / noise_actor / critic stay intact.
    def _mask_encoder_only(path, g):
        key = path[0].key if hasattr(path[0], "key") else str(path[0])
        return g if key == "modules_obs_encoder" else jnp.zeros_like(g)

    g_dsrl_masked = jax.tree_util.tree_map_with_path(_mask_encoder_only, g_dsrl)
    new_dsrl = dsrl_network.apply_gradients(grads=g_dsrl_masked)
    new_residual = residual_network.apply_gradients(grads=g_res)
    return new_dsrl, new_residual, info


# --------------------------------------------------------------------------- #
# Agent                                                                       #
# --------------------------------------------------------------------------- #


class SACResidualAgent(flax.struct.PyTreeNode):
    """Holds the residual actor + its optimizer state.

    Used as a sub-component of :class:`jax_agents.dsrl.DSRLAgent` when
    ``online_training_mode == "sac_residual"``. The DSRL critic provides the
    Q-value; this agent only owns the residual MLP.
    """

    rng: Any
    network: Any  # ModuleDict TrainState containing key "residual_actor"
    config: Any = nonpytree_field()
    obs_mean: Any = nonpytree_field()  # np.float32 array (obs_dim,) set after warmup, or None
    obs_std: Any = nonpytree_field()   # np.float32 array (obs_dim,) set after warmup, or None

    # ----- helpers -------------------------------------------------------- #

    def _scale(self):
        """Residual scale: scalar, or per-dim vector (e.g. ``(accel, steer)`` like the
        torch agent's ``res_scale_accel`` / ``res_scale_steer``)."""
        import numpy as _np
        v = self.config.get("residual_action_scale", 0.1)
        arr = _np.asarray(v, dtype=_np.float32)
        return float(arr) if arr.ndim == 0 else arr

    def _clip_bound(self) -> float:
        """Composed-action clip bound. None/unset → +inf (no clip).

        SteerVLA base actions are physical DELTA_XY meters/degrees; clipping them
        to a normalized range silently destroys the base policy.
        """
        v = self.config.get("residual_action_clip", None)
        return float(v) if v is not None else float("inf")

    def _alpha(self) -> float:
        return float(self.config.get("residual_alpha", self.config.get("alpha", 0.1)))

    def _normalize_obs(self, obs_e: jnp.ndarray) -> jnp.ndarray:
        """Normalize the appended state suffix of obs_e (no-op if normalizer not yet set).

        obs_e = concat([base_embedding, state]) where the last ``len(obs_mean)`` dims are
        the raw state vector.  Only the state suffix is normalized; the base embedding
        (policy_embed or Pi-features) is passed through unchanged.
        """
        if self.obs_mean is None or self.obs_std is None:
            return obs_e
        mean = jnp.asarray(self.obs_mean, dtype=jnp.float32)
        std = jnp.asarray(self.obs_std, dtype=jnp.float32)
        state_dim = mean.shape[0]
        prefix = obs_e[..., :-state_dim]
        state = obs_e[..., -state_dim:]
        return jnp.concatenate([prefix, jnp.clip((state - mean) / std, -5.0, 5.0)], axis=-1)

    def set_obs_norm(self, mean, std):
        """Return a new agent with obs normalizer set to ``mean`` / ``std``."""
        import numpy as _np
        return self.replace(
            obs_mean=_np.asarray(mean, dtype=_np.float32),
            obs_std=_np.asarray(std, dtype=_np.float32),
        )

    # ----- sampling ------------------------------------------------------- #

    def sample_actions_residual(self, obs_e, base_action, seed=None, temperature=1.0):
        """Return ``(action, residual)`` for a batched ``(obs_e, base_action)``.

        ``action = base_action + residual * scale`` (clipped only when
        ``residual_action_clip`` is set).
        """
        seed = seed if seed is not None else self.rng
        scale = jnp.asarray(self._scale(), dtype=jnp.float32)
        temp = jnp.asarray(temperature, dtype=jnp.float32)
        clip_bound = jnp.asarray(self._clip_bound(), dtype=jnp.float32)
        return _sample_action_jit(self.network, self._normalize_obs(obs_e), base_action, seed, temp, scale, clip_bound)

    def sample_actions_and_log_prob_residual(self, obs_e, base_action, seed=None, temperature=1.0):
        """Return ``(action, residual, log_prob)`` — used to compute the entropy term in the SAC TD target."""
        seed = seed if seed is not None else self.rng
        scale = jnp.asarray(self._scale(), dtype=jnp.float32)
        temp = jnp.asarray(temperature, dtype=jnp.float32)
        clip_bound = jnp.asarray(self._clip_bound(), dtype=jnp.float32)
        return _sample_and_log_prob_jit(self.network, self._normalize_obs(obs_e), base_action, seed, temp, scale, clip_bound)

    # ----- training ------------------------------------------------------- #

    def update_actor(
        self,
        obs_e_sg: jnp.ndarray,
        base_action: jnp.ndarray,
        critic_obs_e_sg: jnp.ndarray,
        dsrl_network: TrainState,
    ):
        """SAC actor step on the residual MLP using the DSRL critic.

        Args:
            obs_e_sg: DSRL obs encoder output, **stop-gradient** — input to the
                residual MLP. Shape ``(B, embed_dim)``.
            base_action: Base Pi0 action chunk (flat env layout). Shape
                ``(B, env_action_horizon * env_action_dim)``.
            critic_obs_e_sg: ``concat([obs_e, language_label])`` if a language
                label is used for the critic, otherwise just ``obs_e``. Always
                stop-gradient. Shape ``(B, embed_dim + lang_dim)``.
            dsrl_network: Current DSRL ``TrainState``. Its critic is queried
                via ``apply_fn(name="critic")``; gradients only flow through
                ``action`` back into the residual MLP.
        """
        new_rng, rng = jax.random.split(self.rng)
        alpha = jnp.asarray(self._alpha(), dtype=jnp.float32)
        scale = jnp.asarray(self._scale(), dtype=jnp.float32)
        clip_bound = jnp.asarray(self._clip_bound(), dtype=jnp.float32)
        new_network, info = _residual_actor_apply_step(
            self.network,
            self._normalize_obs(obs_e_sg),
            base_action,
            critic_obs_e_sg,
            dsrl_network,
            rng,
            alpha,
            scale,
            clip_bound,
        )
        return self.replace(network=new_network, rng=new_rng), info

    def update_actor_dagger(
        self,
        obs_e_sg: jnp.ndarray,
        base_action: jnp.ndarray,
        expert_action: jnp.ndarray,
    ):
        """DAgger residual update: MSE-supervise the residual toward the expert.

        Trains the residual MLP so that
        ``base_action + tanh(loc(obs_e, base_action)) * scale``
        matches ``expert_action`` in mean-square. Uses the deterministic mean of
        the tanh-Gaussian; no critic, no exploration noise.

        Args:
            obs_e_sg: DSRL obs encoder output (stop-grad). Shape ``(B, embed_dim)``.
            base_action: Base Pi0 action chunk (flat env layout). Shape
                ``(B, env_action_horizon * env_action_dim)``.
            expert_action: Expert action chunk in the same flat env layout.
        """
        scale = jnp.asarray(self._scale(), dtype=jnp.float32)
        clip_bound = jnp.asarray(self._clip_bound(), dtype=jnp.float32)
        new_network, info = _dagger_actor_apply_step(
            self.network, self._normalize_obs(obs_e_sg), base_action, expert_action, scale, clip_bound,
        )
        return self.replace(network=new_network), info

    # ----- construction --------------------------------------------------- #

    @classmethod
    def create(
        cls,
        seed: int,
        ex_observations,
        ex_actions,
        config,
        embed_dim: int,
    ):
        """Create a SACResidualAgent.

        Args:
            seed: PRNG seed.
            ex_observations: Example obs batch (only its leading batch dim is used).
            ex_actions: Example actions batch (unused — action layout comes from
                config keys ``vla_action_dim`` / ``vla_action_horizon`` to match
                what the DSRL critic was trained on).
            config: Agent config dict (``ml_collections.ConfigDict`` or
                ``FrozenDict``).
            embed_dim: Dimensionality of the DSRL obs encoder output. Required
                so the residual MLP knows its input width.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        if str(config.get("residual_action_space", "waypoint_chunk")) == "accel_steer":
            # 2-D [accel, steer] in [-1, 1] (waypoint chunk is PID-decoded before the
            # residual; matches torch_agents/residual_sac.py used by run_simlingo).
            action_dim = 2
        else:
            action_dim = int(config.get("vla_action_dim", 4)) * int(
                config.get("vla_action_horizon", config.get("action_horizon", 10))
            )
        batch = ex_observations.shape[0]
        ex_obs_e = jnp.zeros((batch, int(embed_dim)), dtype=jnp.float32)
        ex_action = jnp.zeros((batch, action_dim), dtype=jnp.float32)

        actor_def = ResidualActor(
            hidden_dims=tuple(config.get("residual_actor_hidden_dims", (256, 256))),
            action_dim=action_dim,
            layer_norm=bool(
                config.get("residual_layer_norm", config.get("layer_norm", False))
            ),
            log_std_min=float(config.get("residual_log_std_min", -5.0)),
            log_std_max=float(config.get("residual_log_std_max", 2.0)),
        )
        network_def = ModuleDict({"residual_actor": actor_def})
        network_tx = optax.adam(
            learning_rate=float(config.get("residual_lr", config.get("lr", 3e-4)))
        )
        params = network_def.init(init_rng, residual_actor=(ex_obs_e, ex_action))[
            "params"
        ]
        network = TrainState.create(network_def, params, tx=network_tx)
        return cls(rng=rng, network=network, config=flax.core.FrozenDict(**config), obs_mean=None, obs_std=None)


# --------------------------------------------------------------------------- #
# Config                                                                      #
# --------------------------------------------------------------------------- #


def get_config():
    """Standalone config dict for the residual sub-agent.

    In practice the residual agent is instantiated by ``main_carla.py`` from
    the DSRL config when ``online_training_mode == "sac_residual"``; the keys
    below mirror what that path reads (``residual_*``) so the values can also
    be set directly in the DSRL config.
    """
    return ml_collections.ConfigDict(
        dict(
            agent_name="sac_residual",
            lr=3e-4,
            residual_lr=3e-4,
            residual_actor_hidden_dims=(256, 256),
            # Residual magnitude in the base policy's PHYSICAL action units
            # (DELTA_XY meters): ±0.3 m correction per waypoint coordinate.
            residual_action_scale=0.3,
            # Clip bound for the composed action (None = no clip). Leave None for
            # SteerVLA physical actions; see module docstring.
            residual_action_clip=None,
            residual_alpha=0.3,
            # Consistent with create() default (-5.0 prevents extreme log-stds).
            residual_log_std_min=-5.0,
            residual_log_std_max=2.0,
            residual_layer_norm=False,
            layer_norm=False,
            vla_action_dim=4,
            vla_action_horizon=10,
            action_horizon=10,
            # Env steps to execute pure Pi0 (zero residual) before applying the
            # residual MLP. Prevents random-init residual from corrupting the
            # base policy before any useful gradient signal has accumulated.
            residual_warmup_steps=500,
        )
    )
