"""Standalone residual SAC: learns a small additive correction to a frozen base policy.

A frozen base policy (SteerVLA) proposes an action chunk ``base_action``; this
agent learns a residual on top of it::

    final = clip(base_action + residual_scale * tanh_gaussian(actor(x, base_action)), -1, 1)

where ``x`` is the RL state.

NOTE: the critic here is not language-conditioned. While the end goal is to attach the RLT
state representation to a language-conditioned residual critic, this standalone label-free 
critic exists only to test whether RLT is effective in isolation.
"""

from __future__ import annotations

import copy
from typing import Any, Sequence

import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCValue, LogParam, MLP, default_init


def _apply_debug_stop_reward_relabel(batch: dict) -> dict:
    """Debug task: replace env reward with ``-ego_speed`` (m/s) to encourage stopping.

    Requires the replay field ``ego_speed`` (stored by main_carla_residual when
    ``debug_task`` is set). Keeps the stored env reward intact for logging.
    """
    if "ego_speed" not in batch:
        raise KeyError(
            "debug_task requires replay field 'ego_speed' (store CARLA speed m/s in main_carla_residual)."
        )
    out = dict(batch)
    out["rewards"] = -jnp.asarray(out["ego_speed"], dtype=jnp.float32)
    return out


class ResidualActor(nn.Module):
    """Tanh-squashed diagonal-Gaussian residual conditioned on ``(x, base_action)``.

    The tanh-squashed sample lies in ``(-1, 1)`` and is scaled by ``residual_scale``
    outside this module.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    log_std_min: float = -10.0
    log_std_max: float = 2.0
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x, base_action, temperature: float = 1.0):
        h = jnp.concatenate([x, base_action], axis=-1)
        h = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)(h)
        mean = nn.Dense(self.action_dim, kernel_init=default_init(0.01))(h)
        log_std = nn.Dense(self.action_dim, kernel_init=default_init(0.01))(h)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        dist = distrax.MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std) * temperature)
        return distrax.Transformed(dist, distrax.Block(distrax.Tanh(), ndims=1))


class SACResidualAgent(flax.struct.PyTreeNode):
    """Residual soft actor-critic over the flattened action chunk."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def _final_action(self, base_action, residual, scale):
        return jnp.clip(base_action + scale * residual, -1.0, 1.0)

    def critic_loss(self, batch, grad_params, rng, scale):
        next_dist = self.network.select("actor")(batch["next_observations"], batch["next_base_actions"])
        next_residual, next_log_probs = next_dist.sample_and_log_prob(seed=rng)
        next_actions = self._final_action(batch["next_base_actions"], next_residual, scale)

        next_qs = self.network.select("target_critic")(batch["next_observations"], None, actions=next_actions)
        next_q = jnp.min(next_qs, axis=0)

        alpha = self.network.select("alpha")()
        bootstrap = self.config["discount"] * batch["masks"]
        target_q = batch["rewards"] + bootstrap * (next_q - alpha * next_log_probs)
        target_q = jax.lax.stop_gradient(target_q)

        q = self.network.select("critic")(batch["observations"], None, actions=batch["actions"], params=grad_params)
        critic_loss = jnp.square(q - target_q).mean()

        return critic_loss, {
            "critic_loss": critic_loss,
            "q_mean": q.mean(),
            "target_q_mean": target_q.mean(),
        }

    def actor_loss(self, batch, grad_params, rng, scale):
        dist = self.network.select("actor")(batch["observations"], batch["base_actions"], params=grad_params)
        residual, log_probs = dist.sample_and_log_prob(seed=rng)
        actions = self._final_action(batch["base_actions"], residual, scale)

        q = jnp.min(self.network.select("critic")(batch["observations"], None, actions=actions), axis=0)

        alpha = self.network.select("alpha")()
        actor_loss = (alpha * log_probs - q).mean()

        grad_alpha = self.network.select("alpha")(params=grad_params)
        entropy = -jax.lax.stop_gradient(log_probs).mean()
        alpha_loss = (grad_alpha * (entropy - self.config["target_entropy"])).mean()

        return actor_loss + alpha_loss, {
            "actor_loss": actor_loss,
            "alpha_loss": alpha_loss,
            "alpha": alpha,
            "entropy": -log_probs.mean(),
            "residual_scale": scale,
            "residual_abs_mean": jnp.abs(scale * residual).mean(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, scale, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng
        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        if bool(self.config.get("debug_task", False)):
            batch = _apply_debug_stop_reward_relabel(batch)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng, scale)
        for k, v in critic_info.items():
            info[f"critic/{k}"] = v
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng, scale)
        for k, v in actor_info.items():
            info[f"actor/{k}"] = v

        return critic_loss + actor_loss, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["tau"] + tp * (1 - self.config["tau"]),
            self.network.params[f"modules_{module_name}"],
            self.network.params[f"modules_target_{module_name}"],
        )
        network.params[f"modules_target_{module_name}"] = new_target_params

    @jax.jit
    def update(self, batch, scale):
        """One SAC update. ``scale`` is the current residual authority (jax scalar so the
        annealing schedule doesn't trigger a recompile each step)."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, scale, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, "critic")
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, base_actions, scale, seed=None, temperature=1.0):
        """Sample ``(final_action, residual)`` for a batched state + base chunk at ``scale``."""
        dist = self.network.select("actor")(observations, base_actions, temperature=temperature)
        residual = dist.sample(seed=seed)
        return self._final_action(base_actions, residual, scale), residual

    @classmethod
    def create(cls, seed, ex_observations, ex_base_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        action_dim = ex_base_actions.shape[-1]
        target_entropy = config["target_entropy"]
        if target_entropy is None:
            target_entropy = -float(config["target_entropy_multiplier"]) * action_dim

        actor_def = ResidualActor(
            hidden_dims=tuple(config["actor_hidden_dims"]),
            action_dim=action_dim,
            layer_norm=config["layer_norm"],
        )
        critic_def = GCValue(hidden_dims=tuple(config["value_hidden_dims"]), layer_norm=config["layer_norm"], ensemble=True)
        alpha_def = LogParam()

        network_info = dict(
            critic=(critic_def, (ex_observations, None, ex_base_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, None, ex_base_actions)),
            actor=(actor_def, (ex_observations, ex_base_actions)),
            alpha=(alpha_def, ()),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)
        network.params["modules_target_critic"] = network.params["modules_critic"]

        agent_config = dict(
            discount=float(config["discount"]),
            tau=float(config["tau"]),
            target_entropy=float(target_entropy),
            debug_task=bool(config.get("debug_task", False)),
        )
        return cls(rng=rng, network=network, config=flax.core.FrozenDict(agent_config))


def get_config():
    return ml_collections.ConfigDict(
        dict(
            agent_name="sac_residual",
            lr=3e-4,
            batch_size=256,
            actor_hidden_dims=(256, 256),
            value_hidden_dims=(256, 256),
            layer_norm=True,
            discount=0.99,
            tau=0.005,
            residual_scale=0.1,
            # Consumed by main_carla_residual.py (the SAC agent itself is space-agnostic):
            #   "accel_steer" (default) -> base chunk is PID-decoded to a 2-D [accel, steer]
            #                       control first; the residual acts there (waypoints stay the
            #                       base plan). Well-conditioned; typically wants residual_scale
            #                       ~0.5-0.6 since the control is already in [-1, 1].
            #   "waypoint_chunk" -> residual acts on the flattened normalized 40-D chunk
            #                       (reshapes the executed waypoints).
            residual_action_space="accel_steer",
            target_entropy=ml_collections.config_dict.placeholder(float),  # None -> auto.
            target_entropy_multiplier=0.5,
            # Warm-start schedule (applied in main_carla_residual via a step-dependent scale):
            #   step <= residual_warmup_steps           -> scale 0 (pure base policy; also the
            #                                              in-run base baseline + no RL updates)
            #   warmup < step <= warmup + ramp_steps     -> scale ramps 0 -> residual_scale (linear)
            #   step  > warmup + ramp_steps              -> scale = residual_scale (full authority)
            residual_warmup_steps=2000,
            residual_ramp_steps=3000,
            updates_per_step=10,
            # Debug task: RL updates use reward = -ego_speed (m/s) instead of env reward,
            # so the policy should learn to brake to a stop.
            debug_task=False,
        )
    )
