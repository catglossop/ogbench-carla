"""Off-policy Generative Policy Optimization (OGPO) for flow-based control policies.

Reference: https://arxiv.org/abs/2605.03065

First-draft JAX agent aligned with ``jax_agents/dsrl.py`` conventions. OGPO finetunes
the full flow-matching policy (not just initial noise, unlike DSRL) via a PPO-style
objective over denoising trajectories, with off-policy critic learning.

Implemented regularizers from the paper (configurable):

* Success-buffer flow-matching BC (OGPO+, ``bc_coeff``)
* Conservative ensemble advantages (OGPO+CA, ``conservative_advantage``)
* Q-variance-reduced critic targets (``q_variance_reduction``, ``n_vr_samples``)
* Debiased ODE-to-SDE noise injection (``error_correct_sde_to_ode``, ``score`` net)
* Chi-squared / slow-policy drift penalty (OGPO+chi2, ``chi2_reg``)

VLA integration is intentionally omitted in this draft; the flow actor is a local
Flax module (see ``dsrl.py`` for the SteerVLA hook pattern).
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

from jax_agents.dsrl import CarlaObservationEncoder, Critic, FlowActor
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP, default_init


def _steervla():
    """Deferred import: mirrors dsrl.py to avoid circular import at module load."""
    import vlas.steervla as steervla_mod
    return steervla_mod


def _critic_obs_e(obs_e: jnp.ndarray, batch: dict, key: str) -> jnp.ndarray:
    """Append optional language label for critic inputs (CARLA feedback modes)."""
    lang = batch.get(key)
    if lang is None:
        return obs_e
    return jnp.concatenate([obs_e, jnp.asarray(lang, dtype=jnp.float32)], axis=-1)


def _aggregate_q(qs: jnp.ndarray, mode: str, rng: jnp.ndarray) -> jnp.ndarray:
    """Aggregate critic ensemble for ``Q_targ`` (Eq. A.2). ``qs`` shape ``(M, B)``."""
    if mode == "min":
        return jnp.min(qs, axis=0)
    if mode == "mean":
        return jnp.mean(qs, axis=0)
    if mode == "subsample":
        m = qs.shape[0]
        i1, i2 = jax.random.randint(rng, (2,), 0, m)
        return jnp.minimum(qs[i1], qs[i2])
    raise ValueError(f"Unknown critic_agg {mode!r}")


def _conservative_advantage(adv_ensemble: jnp.ndarray) -> jnp.ndarray:
    """Conservative advantage across critic ensemble (Eq. 4.5). ``(M, G)`` -> ``(G,)``."""
    min_adv = jnp.min(adv_ensemble, axis=0)
    max_adv = jnp.max(adv_ensemble, axis=0)
    return jnp.where(min_adv > 0.0, min_adv, jnp.where(max_adv < 0.0, max_adv, 0.0))


class ScoreNetwork(nn.Module):
    """Score / denoiser for debiased ODE-to-SDE conversion (Appendix A.4–A.5)."""

    hidden_dims: tuple
    action_dim: int
    layer_norm: bool = False
    time_embed_dim: int = 16

    @nn.compact
    def __call__(self, observations, actions, times):
        if times.ndim == observations.ndim - 1:
            times = times[..., None]
        freqs = jnp.exp(jnp.linspace(0.0, 4.0, self.time_embed_dim // 2))
        sinusoid = jnp.concatenate(
            [jnp.sin(times * freqs), jnp.cos(times * freqs)], axis=-1
        )
        x = jnp.concatenate([observations, actions, sinusoid], axis=-1)
        x = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)(x)
        return nn.Dense(self.action_dim, kernel_init=default_init(0.01))(x)


class OGPOAgent(flax.struct.PyTreeNode):
    """OGPO agent for flow-based generative control policies."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()
    vla_sample_fn: Any = nonpytree_field()  # Optional callable: (obs, noise) -> action
    steervla_actor: Any = nonpytree_field()  # Optional local SteerVLAActor for update_with_vla

    # ----- helpers -------------------------------------------------------- #

    def _encode_obs(self, params, observations):
        return self.network.select("obs_encoder")(observations, params=params)

    @staticmethod
    def _as_jax_pytree(x):
        return jax.tree.map(lambda y: jnp.asarray(y), x)

    def _clip_actions_to_env(self, actions: jnp.ndarray) -> jnp.ndarray:
        """Flatten/clip VLA output to ``(B, vla_action_horizon * vla_action_dim)``."""
        ah = int(self.config.get("vla_action_horizon", 10))
        ad = int(self.config.get("vla_action_dim", 4))
        flat_env = ah * ad
        x = jnp.asarray(actions, dtype=jnp.float32)
        if x.ndim == 3:
            return x[:, :ah, :ad].reshape(x.shape[0], flat_env)
        if x.ndim == 2 and x.shape[-1] == flat_env:
            return x
        raise ValueError(
            f"Cannot clip actions shape {tuple(x.shape)} to env flat dim {flat_env} "
            f"(vla_action_horizon={ah}, vla_action_dim={ad})."
        )

    def _as_noise_chunk(self, noise: jnp.ndarray) -> jnp.ndarray:
        """Reshape flat noise ``(B, ah*ad)`` to chunk ``(B, ah, ad)`` for VLA."""
        ah = int(self.config.get("vla_action_horizon", 10))
        ad = int(self.config.get("vla_action_dim", 4))
        return noise.reshape(noise.shape[0], ah, ad)

    def _prepare_vla_batch(self, batch):
        """Attach ``openpi_observation`` / ``next_openpi_observation`` to a replay batch."""
        sv = _steervla()
        if "openpi_state" in batch and "next_openpi_state" in batch:
            openpi_obs = sv.openpi_observation_from_replay_batch(batch)
            next_openpi_obs = sv.openpi_observation_from_replay_batch(batch, prefix="next_")
            if self.steervla_actor is not None:
                openpi_obs = self.steervla_actor.attach_replay_tokens(openpi_obs, batch)
                next_openpi_obs = self.steervla_actor.attach_replay_tokens(
                    next_openpi_obs, batch, prefix="next_"
                )
        else:
            raw_obs = raw_next = None
            if self.steervla_actor is not None and getattr(self.steervla_actor, "raw_obs_holder", None) is not None:
                raw_obs = self.steervla_actor.raw_obs_holder.get("obs")
                raw_next = self.steervla_actor.raw_obs_holder.get("next_obs")
            openpi_obs = self.steervla_actor.build_observation_batch_numpy(
                batch_size=batch["observations"].shape[0], raw=raw_obs
            )
            next_openpi_obs = self.steervla_actor.build_observation_batch_numpy(
                batch_size=batch["next_observations"].shape[0], raw=raw_next
            )
            openpi_obs = sv.with_replay_cot_tokens(openpi_obs, batch)
            next_openpi_obs = sv.with_replay_cot_tokens(next_openpi_obs, batch, prefix="next_")
            if self.steervla_actor is not None:
                openpi_obs = self.steervla_actor.attach_replay_tokens(openpi_obs, batch)
                next_openpi_obs = self.steervla_actor.attach_replay_tokens(
                    next_openpi_obs, batch, prefix="next_"
                )
        batch = dict(batch)
        batch["openpi_observation"] = self._as_jax_pytree(openpi_obs)
        batch["next_openpi_observation"] = self._as_jax_pytree(next_openpi_obs)
        return batch

    def _vla_forward(self, observations, openpi_observations, rng, noise=None):
        """Eager VLA forward (PyTorch side); returns stop-gradient env actions."""
        rng_n, rng_act = jax.random.split(rng)
        if noise is None:
            action_dim = int(self.config["action_dim"])
            noise = (
                jax.random.normal(rng_n, (observations.shape[0], action_dim))
                * self.config["noise_scale"]
            )
        noise_chunk = self._as_noise_chunk(noise)
        next_openpi_obs = self._as_jax_pytree(openpi_observations)
        t0 = time.time()
        next_actions = self.steervla_actor._sample_actions(
            rng_act,
            next_openpi_obs,
            noise=noise_chunk,
            image_keys=tuple(self.config.get("image_keys", ("base_0_rgb",))),
            num_steps=int(self.config.get("vla_update_flow_steps", self.config.get("flow_steps", 10))),
        )
        next_actions = self.steervla_actor.postprocess_sampled_trajectory(
            next_actions,
            observation_state=next_openpi_obs.state,
        )
        jax.block_until_ready(next_actions)
        print(f"[DEBUG - ogpo] _vla_forward: {time.time() - t0:.2f}s")
        return jax.lax.stop_gradient(self._clip_actions_to_env(next_actions))

    def _flow_velocity(self, params, obs_e, actions, t):
        return self.network.select("flow")(obs_e, actions, t, params=params)

    def _sde_step(
        self,
        params,
        obs_e,
        action_k,
        next_action,
        step_idx,
    ) -> jnp.ndarray:
        """Log pi(next_action | action_k, s) for one debiased SDE step."""
        flow_steps = int(self.config["flow_steps"])
        dt = 1.0 / flow_steps
        t = jnp.full(action_k.shape[:-1] + (1,), step_idx / flow_steps, dtype=jnp.float32)
        v = self._flow_velocity(params, obs_e, action_k, t)

        sigma = jnp.asarray(self.config["sde_sigma"], dtype=action_k.dtype)
        if self.config["error_correct_sde_to_ode"]:
            z_hat = self.network.select("score")(obs_e, action_k, t, params=params)
            mean = action_k + (v + (sigma / (2.0 * dt)) * z_hat) * dt
        else:
            mean = action_k + v * dt

        dist = distrax.MultivariateNormalDiag(
            loc=mean,
            scale_diag=jnp.full(action_k.shape[-1:], sigma),
        )
        return dist.log_prob(next_action)

    def _integrate_sde_trajectory(
        self,
        flow_params,
        obs_e,
        rng,
        *,
        initial_noise: Optional[jnp.ndarray] = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Sample denoising chain via ``jax.lax.scan``; returns ``(final_action, trajectory, log_prob_sum)``.

        ``trajectory`` has shape ``(K+1, B, A)`` with index 0 the initial noise.
        """
        batch = obs_e.shape[0]
        action_dim = int(self.config["action_dim"])
        flow_steps = int(self.config["flow_steps"])
        sigma = jnp.asarray(self.config["sde_sigma"], dtype=obs_e.dtype)
        dt = 1.0 / flow_steps

        if initial_noise is None:
            rng, noise_rng = jax.random.split(rng)
            action_0 = (
                jax.random.normal(noise_rng, (batch, action_dim))
                * self.config["noise_scale"]
            )
        else:
            action_0 = initial_noise

        def step_fn(carry, i):
            action, rng = carry
            rng, step_rng = jax.random.split(rng)
            t = jnp.full(action.shape[:-1] + (1,), i / flow_steps, dtype=jnp.float32)
            v = self._flow_velocity(flow_params, obs_e, action, t)
            if self.config["error_correct_sde_to_ode"]:
                # flow_params is module-specific (e.g., EMA flow weights) and cannot be
                # passed to the score network which has its own separate weights.
                z_hat = self.network.select("score")(obs_e, action, t)
                mean = action + (v + (sigma / (2.0 * dt)) * z_hat) * dt
            else:
                mean = action + v * dt
            dist = distrax.MultivariateNormalDiag(
                loc=mean,
                scale_diag=jnp.full(action.shape[-1:], sigma),
            )
            new_action, lp = dist.sample_and_log_prob(seed=step_rng)
            return (new_action, rng), (new_action, lp)

        (final_action, _), (action_seq, log_probs_per_step) = jax.lax.scan(
            step_fn, (action_0, rng), jnp.arange(flow_steps)
        )
        trajectory = jnp.concatenate([action_0[None], action_seq], axis=0)
        log_prob_sum = log_probs_per_step.sum(axis=0)
        return jnp.clip(final_action, -1.0, 1.0), trajectory, log_prob_sum

    def _trajectory_log_prob(self, params, obs_e, trajectory):
        """Sum of log pi(a_{i+1}|a_i,s) along a fixed trajectory ``(K+1, B, A)``."""
        flow_steps = int(self.config["flow_steps"])

        def step_fn(_, i):
            lp = self._sde_step(params, obs_e, trajectory[i], trajectory[i + 1], i)
            return None, lp

        _, log_probs = jax.lax.scan(step_fn, None, jnp.arange(flow_steps))
        return log_probs.sum(axis=0)

    def _integrate_ode(self, params, obs_e, initial_noise: Optional[jnp.ndarray], rng):
        """Deterministic flow integration (BC inference / rollout without SDE noise)."""
        batch = obs_e.shape[0]
        action_dim = int(self.config["action_dim"])
        flow_steps = int(self.config["flow_steps"])

        if initial_noise is None:
            rng, noise_rng = jax.random.split(rng)
            action = (
                jax.random.normal(noise_rng, (batch, action_dim))
                * self.config["noise_scale"]
            )
        else:
            action = initial_noise

        def step_fn(action, i):
            t = jnp.full(action.shape[:-1] + (1,), i / flow_steps, dtype=jnp.float32)
            v = self._flow_velocity(params, obs_e, action, t)
            return action + v / flow_steps, None

        final_action, _ = jax.lax.scan(step_fn, action, jnp.arange(flow_steps))
        return jnp.clip(final_action, -1.0, 1.0)

    def _q_all(self, params, batch, obs_e, actions):
        critic_in = _critic_obs_e(obs_e, batch, "language_label")
        return self.network.select("critic")(critic_in, actions, params=params)

    def _q_target(self, batch, obs_e, actions, rng):
        critic_in = _critic_obs_e(obs_e, batch, "language_label")
        qs = self.network.select("target_critic")(critic_in, actions)
        return _aggregate_q(qs, self.config["critic_agg"], rng)

    # ----- losses --------------------------------------------------------- #

    def score_loss(self, batch, grad_params, rng):
        """Denoiser loss for SDE correction (Eq. A.4)."""
        actions = batch["actions"]
        batch_size, action_dim = actions.shape
        rng, z_rng, t_rng = jax.random.split(rng, 3)
        z = jax.random.normal(z_rng, (batch_size, action_dim))
        t = jax.random.uniform(t_rng, (batch_size, 1))
        sigma = jnp.asarray(self.config["sde_sigma"], dtype=actions.dtype)
        x_noisy = actions + sigma * z
        obs_e = self._encode_obs(grad_params, batch["observations"])
        pred_z = self.network.select("score")(obs_e, x_noisy, t, params=grad_params)
        loss = jnp.square(pred_z - z).mean()
        return loss, {"score_loss": loss}

    def flow_bc_loss(self, batch, grad_params, rng):
        """Flow-matching BC loss (success buffer / pretraining)."""
        actions = batch["actions"]
        rng, x_rng, t_rng = jax.random.split(rng, 3)
        x_0 = jax.random.normal(x_rng, actions.shape) * self.config["noise_scale"]
        t = jax.random.uniform(t_rng, actions.shape[:-1] + (1,))
        x_t = (1.0 - t) * x_0 + t * actions
        target_v = actions - x_0
        obs_e = self._encode_obs(grad_params, batch["observations"])
        pred_v = self._flow_velocity(grad_params, obs_e, x_t, t)
        loss = jnp.square(pred_v - target_v).mean()
        return loss, {"flow_bc_loss": loss}

    def critic_loss(self, batch, grad_params, rng):
        """Ensemble critic TD loss with optional Q-variance reduction (Eq. 4.1)."""
        rng, sample_rng, agg_rng = jax.random.split(rng, 3)
        obs_e = self._encode_obs(grad_params, batch["observations"])
        next_obs_e = self._encode_obs(self.network.params, batch["next_observations"])

        ema_params = self.network.params["modules_ema_flow"]
        if self.config["q_variance_reduction"]:
            n_vr = int(self.config["n_vr_samples"])

            def one_next_q(rng_i):
                rng_i, tr_rng = jax.random.split(rng_i)
                na, _, _ = self._integrate_sde_trajectory(
                    ema_params, next_obs_e, tr_rng,
                )
                return self._q_target(
                    batch,
                    next_obs_e,
                    na,
                    jax.random.fold_in(agg_rng, 0),
                )

            vr_rngs = jax.random.split(sample_rng, n_vr)
            next_q = jnp.mean(jax.vmap(one_next_q)(vr_rngs), axis=0)
        else:
            next_actions, _, _ = self._integrate_sde_trajectory(
                ema_params, next_obs_e, sample_rng,
            )
            next_q = self._q_target(batch, next_obs_e, next_actions, agg_rng)

        target_q = batch["rewards"] + self.config["discount"] * batch["masks"] * next_q
        qs = self._q_all(grad_params, batch, obs_e, batch["actions"])
        critic_loss = jnp.square(qs - target_q[None]).mean()

        return critic_loss, {
            "critic_loss": critic_loss,
            "q_mean": qs.mean(),
            "q_max": qs.max(),
            "q_min": qs.min(),
            "target_q": target_q.mean(),
        }

    def ppo_loss(self, batch, grad_params, rng):
        """PPO-style policy extraction over parallel denoising trajectories (Eq. 3.2–3.3)."""
        obs_e = self._encode_obs(grad_params, batch["observations"])
        batch_size = obs_e.shape[0]
        n_group = int(self.config["grpo_num_samples"])
        clip_eps = float(self.config["clip_epsilon"])

        ema_params = self.network.params["modules_ema_flow"]
        if self.config["chi2_reg"]:
            slow_params = self.network.params["modules_slow_flow"]
        else:
            slow_params = ema_params

        rng, group_rng, q_rng = jax.random.split(rng, 3)
        group_rngs = jax.random.split(group_rng, n_group)

        def sample_group(rng_g):
            final_a, traj, lp = self._integrate_sde_trajectory(ema_params, obs_e, rng_g)
            return final_a, traj, lp

        final_actions, trajectories, old_log_probs = jax.vmap(sample_group)(group_rngs)
        # ``final_actions``: (G, B, A), ``trajectories``: (G, K+1, B, A)

        trajectories_sg = jax.lax.stop_gradient(trajectories)
        old_log_probs = jax.lax.stop_gradient(old_log_probs)

        def group_new_logprob(traj_g):
            return self._trajectory_log_prob(grad_params, obs_e, traj_g)

        new_log_probs = jax.vmap(group_new_logprob)(trajectories_sg)
        ratio = jnp.exp(new_log_probs - old_log_probs)

        # Group-wise advantages using target critic (Eq. 3.2)
        critic_in = _critic_obs_e(obs_e, batch, "language_label")

        def q_for_group(actions_g):
            return self.network.select("target_critic")(critic_in, actions_g)

        q_ensemble = jax.vmap(q_for_group)(final_actions)  # (G, M, B)
        q_ensemble = jnp.transpose(q_ensemble, (1, 0, 2))  # (M, G, B)
        baseline = jnp.mean(q_ensemble, axis=1, keepdims=True)
        adv_ensemble = q_ensemble - baseline

        if self.config["conservative_advantage"]:
            advantages = _conservative_advantage(adv_ensemble)
        else:
            advantages = jnp.mean(adv_ensemble, axis=0)

        if self.config["chi2_reg"]:
            slow_log_probs = jax.vmap(
                lambda traj_g: self._trajectory_log_prob(slow_params, obs_e, traj_g)
            )(trajectories_sg)
            beta = float(self.config["chi2_beta_init"]) * jnp.std(q_ensemble)
            advantages = advantages - beta * jnp.exp(old_log_probs - slow_log_probs)

        # Normalize advantages across the group (standard GRPO practice; stabilizes PPO step size).
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        pg1 = -advantages * ratio
        pg2 = -advantages * jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        ppo_loss = jnp.maximum(pg1, pg2).mean()

        approx_kl = (old_log_probs - new_log_probs).mean()
        clip_frac = (jnp.abs(ratio - 1.0) > clip_eps).mean()

        return ppo_loss, {
            "ppo_loss": ppo_loss,
            "advantage_mean": advantages.mean(),
            "approx_kl": approx_kl,
            "clip_frac": clip_frac,
            "ratio_mean": ratio.mean(),
        }

    def total_loss(self, batch, grad_params, rng=None, succ_batch=None):
        info = {}
        rng = rng if rng is not None else self.rng
        rng, critic_rng, ppo_rng, score_rng, bc_rng = jax.random.split(rng, 5)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f"critic/{k}"] = v

        ppo_loss, ppo_info = self.ppo_loss(batch, grad_params, ppo_rng)
        for k, v in ppo_info.items():
            info[f"actor/{k}"] = v

        loss = critic_loss + ppo_loss

        if self.config["train_score"]:
            score_loss, score_info = self.score_loss(batch, grad_params, score_rng)
            for k, v in score_info.items():
                info[f"score/{k}"] = v
            loss = loss + float(self.config["score_coeff"]) * score_loss

        bc_coeff = float(self.config["bc_coeff"])
        if succ_batch is not None and bc_coeff > 0.0:
            bc_loss, bc_info = self.flow_bc_loss(succ_batch, grad_params, bc_rng)
            for k, v in bc_info.items():
                info[f"bc/{k}"] = v
            loss = loss + bc_coeff * bc_loss

        info["total_loss"] = loss
        return loss, info

    def actor_loss(self, batch, grad_params, rng, succ_batch=None):
        """IGP update loss: PPO + optional score denoiser + BC from success buffer (Algorithm 6)."""
        info = {}
        rng, ppo_rng, score_rng, bc_rng = jax.random.split(rng, 4)

        ppo_loss, ppo_info = self.ppo_loss(batch, grad_params, ppo_rng)
        for k, v in ppo_info.items():
            info[f"actor/{k}"] = v
        loss = ppo_loss

        if self.config["train_score"]:
            score_loss, score_info = self.score_loss(batch, grad_params, score_rng)
            for k, v in score_info.items():
                info[f"score/{k}"] = v
            loss = loss + float(self.config["score_coeff"]) * score_loss

        bc_coeff = float(self.config["bc_coeff"])
        if succ_batch is not None and bc_coeff > 0.0:
            bc_loss, bc_info = self.flow_bc_loss(succ_batch, grad_params, bc_rng)
            for k, v in bc_info.items():
                info[f"bc/{k}"] = v
            loss = loss + bc_coeff * bc_loss

        info["actor_total_loss"] = loss
        return loss, info

    # ----- target / EMA updates ------------------------------------------- #

    def target_update(self, network, module_name):
        tau = self.config["tau"]
        new_target = jax.tree_util.tree_map(
            lambda p, tp: p * tau + tp * (1.0 - tau),
            network.params[f"modules_{module_name}"],
            network.params[f"modules_target_{module_name}"],
        )
        network.params[f"modules_target_{module_name}"] = new_target

    def ema_update(self, network, src_name="flow", dst_name="ema_flow"):
        alpha = float(self.config["ema_decay"])
        new_ema = jax.tree_util.tree_map(
            lambda p, ep: alpha * ep + (1.0 - alpha) * p,
            network.params[f"modules_{src_name}"],
            network.params[f"modules_{dst_name}"],
        )
        network.params[f"modules_{dst_name}"] = new_ema

    def slow_ema_update(self, network):
        if not self.config["chi2_reg"]:
            return
        alpha = float(self.config["slow_ema_decay"])
        new_slow = jax.tree_util.tree_map(
            lambda p, sp: alpha * sp + (1.0 - alpha) * p,
            network.params["modules_flow"],
            network.params["modules_slow_flow"],
        )
        network.params["modules_slow_flow"] = new_slow

    @jax.jit
    def update(self, batch, succ_batch=None):
        new_rng, rng = jax.random.split(self.rng)
        rng, critic_rng, actor_rng = jax.random.split(rng, 3)
        info = {}

        # UPDATEQ (Algorithm 5): critic-only backward pass.
        def critic_loss_fn(grad_params):
            return self.critic_loss(batch, grad_params, critic_rng)
        new_network, critic_info = self.network.apply_loss_fn(loss_fn=critic_loss_fn)
        self.target_update(new_network, "critic")
        info.update(critic_info)

        # UPDATEIGP (Algorithm 6): actor-only backward pass.
        def actor_loss_fn(grad_params):
            return self.actor_loss(batch, grad_params, actor_rng, succ_batch=succ_batch)
        new_network, actor_info = new_network.apply_loss_fn(loss_fn=actor_loss_fn)
        info.update(actor_info)

        self.ema_update(new_network, "flow", "ema_flow")
        self.slow_ema_update(new_network)
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update_bc_only(self, batch):
        """Flow-matching pretraining / offline BC before online OGPO."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            loss, bc_info = self.flow_bc_loss(batch, grad_params, rng)
            if self.config["train_score"]:
                rng_s, _ = jax.random.split(rng)
                s_loss, s_info = self.score_loss(batch, grad_params, rng_s)
                loss = loss + float(self.config["score_coeff"]) * s_loss
                bc_info = {**bc_info, **{f"score/{k}": v for k, v in s_info.items()}}
            return loss, {f"bc/{k}": v for k, v in bc_info.items()}

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.ema_update(new_network, "flow", "ema_flow")
        return self.replace(network=new_network, rng=new_rng), info

    # ----- action sampling ------------------------------------------------ #

    @jax.jit
    def sample_actions(self, observations, seed=None, temperature=1.0, use_sde=False):
        """Rollout actions using EMA policy (πθtarg, Algorithm 4). ODE is deterministic."""
        seed = seed if seed is not None else self.rng
        obs_e = self._encode_obs(self.network.params, observations)
        # Algorithm 4 uses πθtarg (EMA) for all rollouts, not the live online policy.
        ema_params = self.network.params["modules_ema_flow"]
        if use_sde or self.config["rollout_use_sde"]:
            actions, _, _ = self._integrate_sde_trajectory(ema_params, obs_e, seed)
        else:
            actions = self._integrate_ode(ema_params, obs_e, None, seed)
        return actions

    def sample_actions_with_vla(self, observations, seed=None, temperature=1.0):
        """Rollout via ``vla_sample_fn`` with random noise (no NoiseActor in OGPO)."""
        if self.vla_sample_fn is None:
            return self.sample_actions(observations, seed=seed, temperature=temperature)
        seed = seed if seed is not None else self.rng
        seed, noise_rng = jax.random.split(seed)
        action_dim = int(self.config["action_dim"])
        noise = (
            jax.random.normal(noise_rng, (observations.shape[0], action_dim))
            * self.config["noise_scale"]
        )
        return jnp.asarray(self.vla_sample_fn(observations, noise))

    @jax.jit
    def sample_actions_best_of_n(self, observations, seed=None, n=None):
        """Optional Best-of-N using target critic (Eq. 4.4)."""
        n = int(n or self.config.get("best_of_n", 4))
        seed = seed if seed is not None else self.rng
        obs_e = self._encode_obs(self.network.params, observations)
        batch = observations.shape[0]
        ema_params = self.network.params["modules_ema_flow"]

        def one_sample(rng_i):
            a, _, _ = self._integrate_sde_trajectory(ema_params, obs_e, rng_i)
            return a

        rngs = jax.random.split(seed, n)
        candidates = jax.vmap(one_sample)(rngs)  # (n, B, A)
        critic_in = jnp.repeat(obs_e[None, ...], n, axis=0).reshape(n * batch, -1)
        qs = self.network.select("target_critic")(
            critic_in,
            candidates.reshape(n * batch, -1),
        )
        q_each = _aggregate_q(qs, self.config["critic_agg"], seed).reshape(n, batch)
        best_idx = jnp.argmax(q_each, axis=0)
        flat = candidates.transpose(1, 0, 2)  # (B, n, A)
        bidx = jnp.arange(batch)
        return flat[bidx, best_idx]

    # ----- VLA update path ----------------------------------------------- #

    def critic_loss_vla(self, batch, grad_params, rng, next_vla_actions):
        """Critic TD loss with VLA-generated next-state actions as the bootstrap target."""
        rng, agg_rng = jax.random.split(rng)
        obs_e = self._encode_obs(grad_params, batch["observations"])
        next_obs_e = self._encode_obs(self.network.params, batch["next_observations"])
        next_q = self._q_target(batch, next_obs_e, next_vla_actions, agg_rng)
        target_q = batch["rewards"] + self.config["discount"] * batch["masks"] * next_q
        qs = self._q_all(grad_params, batch, obs_e, batch["actions"])
        critic_loss = jnp.square(qs - target_q[None]).mean()
        return critic_loss, {
            "critic_loss": critic_loss,
            "q_mean": qs.mean(),
            "q_max": qs.max(),
            "q_min": qs.min(),
            "target_q": target_q.mean(),
        }

    def total_loss_vla(self, batch, grad_params, rng, next_vla_actions, succ_batch=None):
        """total_loss variant that uses VLA next actions for the critic bootstrap."""
        info = {}
        rng, critic_rng, ppo_rng, score_rng, bc_rng = jax.random.split(rng, 5)

        critic_loss, critic_info = self.critic_loss_vla(
            batch, grad_params, critic_rng, next_vla_actions
        )
        for k, v in critic_info.items():
            info[f"critic/{k}"] = v

        ppo_loss, ppo_info = self.ppo_loss(batch, grad_params, ppo_rng)
        for k, v in ppo_info.items():
            info[f"actor/{k}"] = v

        loss = critic_loss + ppo_loss

        if self.config["train_score"]:
            score_loss, score_info = self.score_loss(batch, grad_params, score_rng)
            for k, v in score_info.items():
                info[f"score/{k}"] = v
            loss = loss + float(self.config["score_coeff"]) * score_loss

        bc_coeff = float(self.config["bc_coeff"])
        if succ_batch is not None and bc_coeff > 0.0:
            bc_loss, bc_info = self.flow_bc_loss(succ_batch, grad_params, bc_rng)
            for k, v in bc_info.items():
                info[f"bc/{k}"] = v
            loss = loss + bc_coeff * bc_loss

        info["total_loss"] = loss
        return loss, info

    def update_with_vla(self, batch, succ_batch=None):
        """Update using VLA next-state actions for the critic, PPO over local flow.

        VLA forward runs eagerly (PyTorch); the jitted gradient steps only see the
        precomputed stop-gradient ``next_vla_actions``.
        """
        new_rng, rng = jax.random.split(self.rng)
        batch = self._prepare_vla_batch(batch)
        rng, vla_rng, critic_rng, actor_rng = jax.random.split(rng, 4)
        info = {}

        next_vla_actions = self._vla_forward(
            batch["next_observations"],
            batch["next_openpi_observation"],
            vla_rng,
        )

        # UPDATEQ with VLA bootstrap actions (Algorithm 5).
        def critic_loss_fn(grad_params):
            return self.critic_loss_vla(batch, grad_params, critic_rng, next_vla_actions)
        new_network, critic_info = self.network.apply_loss_fn(loss_fn=critic_loss_fn)
        self.target_update(new_network, "critic")
        info.update(critic_info)

        # UPDATEIGP (Algorithm 6): actor-only backward pass.
        def actor_loss_fn(grad_params):
            return self.actor_loss(batch, grad_params, actor_rng, succ_batch=succ_batch)
        new_network, actor_info = new_network.apply_loss_fn(loss_fn=actor_loss_fn)
        info.update(actor_info)

        self.ema_update(new_network, "flow", "ema_flow")
        self.slow_ema_update(new_network)
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
        action_dim = int(ex_actions.shape[-1])
        config = dict(config)
        config["action_dim"] = action_dim

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
            embed_dim = single * 3 if config.get("siglip_include_prompt_subtask", False) else single
        else:
            embed_dim = int(tuple(config.get("image_mlp_hidden_dims", (512,)))[-1])

        critic_feedback_mode = str(config.get("critic_feedback_mode", "commentary_bow"))
        if critic_feedback_mode == "action_delta":
            lang_dim = int(config.get("critic_action_dim", action_dim))
        else:
            lang_dim = int(config.get("language_label_dim", 119))

        batch_shape = ex_observations.shape[:1]
        ex_embedded = jnp.zeros(batch_shape + (embed_dim,), dtype=jnp.float32)
        ex_critic_embedded = jnp.zeros(batch_shape + (embed_dim + lang_dim,), dtype=jnp.float32)
        ex_t = jnp.zeros(batch_shape + (1,), dtype=jnp.float32)

        flow_def = FlowActor(
            hidden_dims=tuple(config["flow_hidden_dims"]),
            action_dim=action_dim,
            layer_norm=config["layer_norm"],
        )
        score_def = ScoreNetwork(
            hidden_dims=tuple(config.get("score_hidden_dims", config["flow_hidden_dims"])),
            action_dim=action_dim,
            layer_norm=config["layer_norm"],
        )
        critic_def = Critic(
            hidden_dims=tuple(config["critic_hidden_dims"]),
            layer_norm=config["layer_norm"],
            ensemble_size=config["critic_ensemble"],
        )

        networks = {
            "obs_encoder": (obs_encoder_def, (ex_observations,)),
            "flow": (flow_def, (ex_embedded, ex_actions, ex_t)),
            "ema_flow": (copy.deepcopy(flow_def), (ex_embedded, ex_actions, ex_t)),
            "score": (score_def, (ex_embedded, ex_actions, ex_t)),
            "critic": (critic_def, (ex_critic_embedded, ex_actions)),
            "target_critic": (copy.deepcopy(critic_def), (ex_critic_embedded, ex_actions)),
        }
        if config.get("chi2_reg", False):
            networks["slow_flow"] = (copy.deepcopy(flow_def), (ex_embedded, ex_actions, ex_t))

        defs = {k: v[0] for k, v in networks.items()}
        args = {k: v[1] for k, v in networks.items()}

        network_def = ModuleDict(defs)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)
        network.params["modules_target_critic"] = network.params["modules_critic"]
        network.params["modules_ema_flow"] = network.params["modules_flow"]
        if "modules_slow_flow" in network.params:
            network.params["modules_slow_flow"] = network.params["modules_flow"]

        return cls(
            rng=rng,
            network=network,
            config=flax.core.FrozenDict(**config),
            vla_sample_fn=vla_sample_fn,
            steervla_actor=steervla_actor,
        )


def get_config():
    """``ml_collections.ConfigDict`` for ``--agent=jax_agents/ogpo.py``."""
    return ml_collections.ConfigDict(
        dict(
            agent_name="ogpo",
            lr=3e-4,
            batch_size=256,
            flow_hidden_dims=(256, 256),
            score_hidden_dims=(256, 256),
            critic_hidden_dims=(256, 256),
            critic_ensemble=2,
            layer_norm=True,
            discount=0.99,
            tau=0.005,
            # Flow / SDE
            flow_steps=10,
            noise_scale=1.0,
            sde_sigma=0.1,
            error_correct_sde_to_ode=True,
            train_score=True,
            score_coeff=1.0,
            rollout_use_sde=False,
            action_dim=ml_collections.config_dict.placeholder(int),
            # OGPO PPO extraction
            grpo_num_samples=8,
            clip_epsilon=0.01,
            ema_decay=0.995,
            # Regularizers (OGPO+ / +CA / +chi2)
            bc_coeff=1.0,
            conservative_advantage=True,
            chi2_reg=False,
            chi2_beta_init=1.0,
            slow_ema_decay=0.999,
            q_variance_reduction=True,
            n_vr_samples=4,
            critic_agg="subsample",
            best_of_n=4,
            # CARLA / observation (mirrors dsrl)
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
            language_label_dim=119,
            critic_action_dim=4,
            critic_feedback_mode="commentary_bow",
            warmup_steps=1000,
            updates_per_step=1,
            enable_updates=True,
            buffer_capacity=100_000,
            image_log_curr_interval=1000,
            training_gpu_rank=-1,
            frame_stack=ml_collections.config_dict.placeholder(int),
            dataset_class="ReplayBuffer",
        )
    )
