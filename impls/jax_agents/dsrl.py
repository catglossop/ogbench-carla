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
* Optional **SteerVLA hookup**: pass ``vla_actor`` (a callable
  ``(obs_batch, noise_batch) -> action_batch``) into :meth:`DSRLAgent.create`
  and the BC flow integration is replaced by that callable; useful when you have
  a pretrained OpenPI / SteerVLA flow you want to keep frozen.

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
        """When ``vla_sample_fn`` is provided, replace the JAX flow with the VLA call."""
        if self.vla_sample_fn is None:
            return self.sample_actions(observations, seed=seed, temperature=temperature)
        seed = seed if seed is not None else self.rng
        seed, sub = jax.random.split(seed)
        enc = self._encode_obs(self.network.params, observations)
        dist = self.network.select("noise_actor")(enc, temperature=temperature)
        noise = dist.sample(seed=sub)
        noise = noise * self.config["noise_scale"]
        return jnp.asarray(self.vla_sample_fn(observations, noise))

    # ----- losses --------------------------------------------------------- #

    def critic_loss(self, batch, grad_params, rng):
        next_obs_e = self._encode_obs(self.network.params, batch["next_observations"])
        next_dist = self.network.select("noise_actor")(next_obs_e)
        next_noise = next_dist.sample(seed=rng) * self.config["noise_scale"]
        next_actions = self._flow_sample(
            self.network.params, batch["next_observations"], next_noise
        )
        next_qs = self.network.select("target_critic")(next_obs_e, next_actions)
        next_q = jnp.min(next_qs, axis=0)

        target_q = batch["rewards"] + self.config["discount"] * batch["masks"] * next_q

        obs_e = self._encode_obs(grad_params, batch["observations"])
        qs = self.network.select("critic")(
            obs_e, batch["actions"], params=grad_params
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
        actions = batch["actions"]
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
        qs = self.network.select("critic")(obs_e, actions)
        q = jnp.min(qs, axis=0)
        actor_loss = (self.config["alpha"] * log_prob - q).mean()
        return actor_loss, {
            "actor_loss": actor_loss,
            "noise_log_prob": log_prob.mean(),
            "q_for_actor": q.mean(),
        }

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

    # ----- construction --------------------------------------------------- #

    @classmethod
    def create(
        cls,
        seed: int,
        ex_observations,
        ex_actions,
        config,
        vla_sample_fn: Optional[Callable] = None,
    ):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        obs_mode = str(config.get("observation_mode", "state"))
        if obs_mode not in ("state", "image"):
            raise ValueError(f"observation_mode must be 'state' or 'image', got {obs_mode!r}")

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
        batch_shape = ex_observations.shape[:1]
        ex_embedded = jnp.zeros(batch_shape + (embed_dim,), dtype=jnp.float32)
        ex_t = jnp.zeros(batch_shape + (1,), dtype=jnp.float32)

        flow_def = FlowActor(
            hidden_dims=tuple(config["flow_hidden_dims"]),
            action_dim=action_dim,
            layer_norm=config["layer_norm"],
        )
        actor_def = NoiseActor(
            hidden_dims=tuple(config["actor_hidden_dims"]),
            action_dim=action_dim,
        )
        critic_def = Critic(
            hidden_dims=tuple(config["critic_hidden_dims"]),
            layer_norm=config["layer_norm"],
            ensemble_size=config["critic_ensemble"],
        )

        networks = {
            "obs_encoder": (obs_encoder_def, (ex_observations,)),
            "flow": (flow_def, (ex_embedded, ex_actions, ex_t)),
            "noise_actor": (actor_def, (ex_embedded,)),
            "critic": (critic_def, (ex_embedded, ex_actions)),
            "target_critic": (
                copy.deepcopy(critic_def),
                (ex_embedded, ex_actions),
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
            # Ignored by DSRL but populated for symmetry with other agents.
            frame_stack=ml_collections.config_dict.placeholder(int),
            dataset_class="GCDataset",
        )
    )
    return config
