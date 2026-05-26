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
import math
import os
from typing import Any, Callable, Optional

import distrax
import flax
import flax.linen as nn
import flax.nnx as nnx
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


# Pi0-CoT action-expert attribute names at the top level of nnx.State.
# Everything else (PaliGemma.llm + PaliGemma.img) is the frozen backbone.
_VLA_ACTION_HEAD_NAMES: frozenset = frozenset(
    ["action_in_proj", "time_mlp_in", "time_mlp_out", "action_out_proj"]
)


def _vla_state_split(full_state) -> tuple:
    """Split nnx.State into (action_head_state, backbone_state) by top-level key.

    Uses ``raw_mapping`` to avoid the double-wrapping that ``dict(state)``
    produces when ``State.__getitem__`` re-wraps nested dicts in ``State``.
    """
    raw = full_state.raw_mapping
    head = {k: raw[k] for k in raw if k in _VLA_ACTION_HEAD_NAMES}
    backbone = {k: raw[k] for k in raw if k not in _VLA_ACTION_HEAD_NAMES}
    return nnx.State(head), nnx.State(backbone)


def _vla_merge_state(action_head_state, backbone_state) -> "nnx.State":
    """Reconstruct full nnx.State from (action_head_state, backbone_state)."""
    merged = dict(backbone_state.raw_mapping)
    merged.update(action_head_state.raw_mapping)
    return nnx.State(merged)


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


def _maybe_log_jax_memory(tag: str, **extra: Any) -> None:
    """Best-effort device-memory probe, enabled with ``OGPO_JAX_MEM_DEBUG=1``."""
    if os.environ.get("OGPO_JAX_MEM_DEBUG", "").strip().lower() not in ("1", "true", "yes", "on"):
        return
    try:
        live = list(jax.live_arrays())
        total_bytes = 0
        samples: list[str] = []
        for arr in live[:6]:
            nbytes = int(getattr(arr, "nbytes", 0))
            total_bytes += nbytes
            samples.append(f"{tuple(arr.shape)}:{arr.dtype}")
        extra_str = " ".join(f"{k}={v}" for k, v in extra.items())
        print(
            f"[ogpo][jax-mem] {tag} live_arrays={len(live)} "
            f"tracked_bytes={total_bytes / (1024 ** 2):.1f}MiB "
            f"samples={samples} {extra_str}".rstrip(),
            flush=True,
        )
    except Exception as exc:
        print(f"[ogpo][jax-mem] {tag} probe_failed={exc}", flush=True)


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
    vla_sample_fn: Any = nonpytree_field()   # Optional callable: (obs, noise) -> action
    steervla_actor: Any = nonpytree_field()  # Optional local SteerVLAActor for update_with_vla
    # VLA fine-tuning state (OGPO direct VLA PPO).  All None when VLA is frozen.
    vla_graphdef: Any = nonpytree_field()    # nnx.GraphDef — static, not a pytree
    vla_tx: Any = nonpytree_field()          # optax transform — static
    # Action-head-only states (action_in_proj / time_mlp_in / time_mlp_out / action_out_proj).
    # The frozen backbone (PaliGemma) is in vla_backbone_state and never differentiated.
    vla_online_state: Any = None             # nnx.State pytree — live θ (action head)
    vla_ema_state: Any = None               # nnx.State pytree — EMA θ̄ (action head)
    vla_slow_state: Any = None              # nnx.State pytree — slow θ̃ (chi2_reg, action head)
    vla_opt_state: Any = None              # optax optimizer state (action head only)
    vla_backbone_state: Any = None         # nnx.State pytree — frozen PaliGemma params

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
        batch_size = observations.shape[0]
        if noise is None:
            action_dim = int(self.config["action_dim"])
            noise = (
                jax.random.normal(rng_n, (batch_size, action_dim))
                * self.config["noise_scale"]
            )
        # Build full-model-size noise tensor so model.action_in_proj sees the right shape.
        # Mirrors steervla.py::flow_sample: zeros(model_ah, model_ad) then write cfg chunk.
        model_ah = int(self.steervla_actor.model.action_horizon)
        model_ad = int(self.steervla_actor.model.action_dim)
        cfg_ah = min(int(self.config.get("vla_action_horizon", 10)), model_ah)
        cfg_ad = min(int(self.config.get("vla_action_dim", 4)), model_ad)
        noise_chunk = self._as_noise_chunk(noise)[:, :cfg_ah, :cfg_ad]
        noise_full = jnp.zeros((batch_size, model_ah, model_ad), dtype=jnp.float32)
        noise_full = noise_full.at[:, :cfg_ah, :cfg_ad].set(noise_chunk)
        next_openpi_obs = self._as_jax_pytree(openpi_observations)
        next_actions = self.steervla_actor._sample_actions(
            rng_act,
            next_openpi_obs,
            noise=noise_full,
            image_keys=tuple(self.config.get("image_keys", ("base_0_rgb",))),
            num_steps=int(self.config.get("vla_update_flow_steps", self.config.get("flow_steps", 10))),
        )
        next_actions = self.steervla_actor.postprocess_sampled_trajectory(
            next_actions,
            observation_state=next_openpi_obs.state,
        )
        return jax.lax.stop_gradient(self._clip_actions_to_env(next_actions))

    def _eval_vla_traj_log_probs_flat(
        self,
        vla_action_head_state,
        traj_g,
        kv_cache,
        prefix_mask,
        prefix_mask_nr,
        sigma: float,
        num_steps: int,
    ):
        """Evaluate sum of log pi(a_{k+1}|a_k,s) for ONE group's trajectory.

        ``traj_g`` shape ``(K+1, B, model_ah, model_ad)``; ``kv_cache`` is
        B-sized (never tiled to G×B — that would OOM on a single GPU).
        Returns ``(B,)`` log-prob sums.

        Differentiable w.r.t. ``vla_action_head_state`` only — the action-expert
        layers (``action_in_proj``, ``time_mlp_in/out``, ``action_out_proj``).
        The frozen backbone (``vla_backbone_state``) is a closure constant; JAX
        never computes or allocates gradients w.r.t. it, and the Adam optimizer
        state only covers the action head (~few MB vs ~21 GB for the full model).
        The prefix KV cache is stop-gradiented by ``compute_prefix_kv_for_ogpo``.
        """
        import einops as _einops
        from openpi.models import pi0 as _openpi_pi0

        # Merge live action-head params with frozen backbone to get a runnable model.
        full_state = _vla_merge_state(vla_action_head_state, self.vla_backbone_state)
        model = nnx.merge(self.vla_graphdef, full_state)
        B = traj_g.shape[1]
        ah = traj_g.shape[2]
        ad = traj_g.shape[3]
        action_flat = ah * ad
        sigma_sq = float(sigma) ** 2
        log_norm = float(action_flat) * math.log(2.0 * math.pi * sigma_sq)
        dt = jnp.array(-1.0 / num_steps, dtype=jnp.float32)

        # jax.checkpoint: recompute this step during backward instead of storing
        # K copies of LLM activations.  Without it, scan stores K × (full LLM
        # hidden states) = ~10 GB for K=10 steps on a Gemma-2B model.
        @jax.checkpoint
        def step_fn(_, inputs):
            x_t, x_next, k = inputs
            t_b = jnp.broadcast_to(jnp.array(1.0 - k / num_steps, dtype=jnp.float32), (B,))
            sfx_tok, sfx_mask, _, adarms = model._embed_action_suffix(None, x_t, t_b)
            sfx_ar = jnp.array([True] + [False] * (ah - 1))
            sfx_attn = _openpi_pi0.make_attn_mask(sfx_mask, sfx_ar)
            a2p = _einops.repeat(prefix_mask_nr, "b p -> b s p", s=sfx_tok.shape[1])
            full_mask = jnp.concatenate([a2p, sfx_attn], axis=-1)
            pos = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(sfx_mask, axis=-1) - 1
            (_, sfx_out), _ = model.PaliGemma.llm(
                [None, sfx_tok], mask=full_mask, positions=pos,
                kv_cache=kv_cache, adarms_cond=[None, adarms],
            )
            v_t = model.action_out_proj(sfx_out[:, -ah:])
            mean = x_t + dt * v_t
            diff = (x_next - mean).reshape(B, action_flat)
            log_p = -0.5 * (jnp.sum(diff ** 2, axis=-1) / sigma_sq + log_norm)
            return None, log_p

        _, log_probs_per_step = jax.lax.scan(
            step_fn, None,
            (traj_g[:-1], traj_g[1:], jnp.arange(num_steps, dtype=jnp.float32)),
        )
        return log_probs_per_step.sum(axis=0)  # (B,)

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
        """Critic TD loss with VLA bootstrap targets.

        ``next_vla_actions`` shape:
        - ``(B, A_env)`` — standard (single next-state action per transition)
        - ``(n_vr, B, A_env)`` — Q-variance-reduction: average Q over n_vr samples
          per next state (paper Sec. 4, Eq. 4.3).
        """
        rng, agg_rng = jax.random.split(rng)
        obs_e = self._encode_obs(grad_params, batch["observations"])
        next_obs_e = self._encode_obs(self.network.params, batch["next_observations"])
        if next_vla_actions.ndim == 3:
            # VR critic: average Q-targets over n_vr next-state samples
            def q_for_i(na_i):
                return self._q_target(batch, next_obs_e, na_i, jax.random.fold_in(agg_rng, 0))
            next_q = jnp.mean(jax.vmap(q_for_i)(next_vla_actions), axis=0)
        else:
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

    @jax.jit
    def update_with_vla(self, batch, succ_batch=None):
        """OGPO update: VR critic with VLA next-state actions + VLA PPO gradient step.

        When ``vla_online_state`` is initialised (VLA fine-tuning mode):
          1. Sample n_vr VLA next-state actions from EMA VLA state → VR critic.
          2. Update critic (Eq. 4.1 + Eq. 4.3 VR).
          3. Compute prefix KV cache from EMA VLA state (frozen prefix; gradient
             flows only through action-expert layers).
          4. Sample G*B denoising trajectories from EMA VLA state (stop-grad).
          5. Compute Q-advantages from target critic; apply chi2 penalty if enabled.
          6. ``jax.value_and_grad`` PPO loss w.r.t. online VLA state.
          7. Apply optax update; EMA and slow-EMA updates for VLA state.

        Falls back to FlowActor PPO when VLA state is not initialised.
        """
        new_rng, rng = jax.random.split(self.rng)
        batch = self._prepare_vla_batch(batch)
        info = {}
        _maybe_log_jax_memory(
            "update_start",
            batch_size=batch["observations"].shape[0],
            image_shape=tuple(batch["openpi_observation"].images["base_0_rgb"].shape),
        )

        do_vla_ppo = (
            self.vla_graphdef is not None
            and self.vla_online_state is not None
            and self.vla_ema_state is not None
        )

        # ── Step 0: Extract config vars + ema_full_state (shared by steps 1 & 3) #
        if do_vla_ppo:
            image_keys = tuple(self.config.get("image_keys", ("base_0_rgb",)))
            batch_size = batch["observations"].shape[0]
            model_ah = int(self.steervla_actor.model.action_horizon)
            model_ad = int(self.steervla_actor.model.action_dim)
            cfg_ah = int(self.config.get("vla_action_horizon", 10))
            cfg_ad = int(self.config.get("vla_action_dim", 4))
            sigma = float(self.config.get("sde_sigma", 0.1))
            num_steps = int(self.config.get("vla_update_flow_steps", self.config.get("flow_steps", 10)))
            ema_full_state = _vla_merge_state(self.vla_ema_state, self.vla_backbone_state)

        # ── 1. VLA next-state actions for critic bootstrap ─────────────────── #
        rng, vr_rng, critic_rng = jax.random.split(rng, 3)
        if do_vla_ppo and self.config.get("q_variance_reduction", False):
            n_vr = int(self.config.get("n_vr_samples", 4))
            # Compute next-obs KV cache ONCE; reuse across all n_vr VR samples.
            kv_next, mask_next, mask_nr_next = self.steervla_actor.compute_prefix_kv_for_ogpo(
                self.vla_graphdef, ema_full_state, batch["next_openpi_observation"], image_keys,
            )
            _maybe_log_jax_memory("after_next_prefix")
            vr_rngs = jax.random.split(vr_rng, n_vr)
            vr_noises = jax.vmap(
                lambda k: jax.random.normal(k, (batch_size, model_ah, model_ad))
            )(vr_rngs)

            def _sample_one_vr(_, noise_rng):
                noise_i, rng_i = noise_rng
                traj_i, _ = self.steervla_actor.sample_sde_trajectory_for_ppo(
                    self.vla_graphdef, ema_full_state,
                    kv_next, mask_next, mask_nr_next,
                    noise_i, rng=rng_i, sigma=sigma, num_steps=num_steps,
                )
                final = traj_i[-1, :, :cfg_ah, :cfg_ad].reshape(batch_size, cfg_ah * cfg_ad).clip(-1.0, 1.0)
                return None, final

            _, next_vla_actions = jax.lax.scan(_sample_one_vr, None, (vr_noises, vr_rngs))  # (n_vr, B, A_env)
            # Free next-obs KV cache — both KV caches live simultaneously would OOM.
            del kv_next, mask_next, mask_nr_next
            _maybe_log_jax_memory("after_next_action_scan")
        else:
            next_vla_actions = self._vla_forward(
                batch["next_observations"], batch["next_openpi_observation"], vr_rng
            )  # (B, A_env)

        # ── 2. Critic update ────────────────────────────────────────────────── #
        def critic_loss_fn(grad_params):
            return self.critic_loss_vla(batch, grad_params, critic_rng, next_vla_actions)

        new_network, critic_info = self.network.apply_loss_fn(loss_fn=critic_loss_fn)
        self.target_update(new_network, "critic")
        info.update(critic_info)
        _maybe_log_jax_memory("after_critic_update")

        if do_vla_ppo:
            # ── 3. Prefix KV cache (B-sized, never tiled to G×B) ─────────────── #
            rng, traj_rng = jax.random.split(rng)
            n_group = int(self.config["grpo_num_samples"])
            clip_eps = float(self.config["clip_epsilon"])

            # B-sized KV cache — NOT tiled to G×B.  Each group reuses this cache.
            # ema_full_state was already computed in step 0 above.
            kv_cache, prefix_mask, prefix_mask_nr = (
                self.steervla_actor.compute_prefix_kv_for_ogpo(
                    self.vla_graphdef, ema_full_state,
                    batch["openpi_observation"], image_keys,
                )
            )
            _maybe_log_jax_memory("after_current_prefix")

            # ── 4. Sample G trajectories via jax.lax.scan (no Python loop) ────── #
            # Serial over groups (one B-batch at a time) keeps kv_cache at B-size.
            # Using scan instead of a Python for-loop means the XLA runtime executes
            # all G iterations as one continuous GPU program — no Python dispatch
            # gaps between groups.
            g_rngs = jax.random.split(traj_rng, n_group)
            # Use per-group keys for noise to match original per-group sampling behavior.
            all_noises = jax.vmap(
                lambda k: jax.random.normal(k, (batch_size, model_ah, model_ad))
            )(g_rngs)

            def sample_one_group(_, noise_rng):
                noise_g, rng_g = noise_rng
                traj_g, lp_g = self.steervla_actor.sample_sde_trajectory_for_ppo(
                    self.vla_graphdef, ema_full_state,
                    kv_cache, prefix_mask, prefix_mask_nr,
                    noise_g, rng=rng_g, sigma=sigma, num_steps=num_steps,
                )
                return None, (traj_g, lp_g)

            _, (trajs_stacked, old_lps_stacked) = jax.lax.scan(
                sample_one_group, None, (all_noises, g_rngs)
            )
            # trajs_stacked:  (G, K+1, B, model_ah, model_ad)  stop-gradient
            # old_lps_stacked: (G, B)                           stop-gradient

            # ── 5a. Q-advantages from target critic ─────────────────────────── #
            obs_e = jax.lax.stop_gradient(
                self._encode_obs(new_network.params, batch["observations"])
            )
            critic_in_obs = _critic_obs_e(obs_e, batch, "language_label")

            # trajs_stacked[:, -1] is the denoised final action for each group.
            final_env = (
                trajs_stacked[:, -1, :, :cfg_ah, :cfg_ad]
                .reshape(n_group, batch_size, cfg_ah * cfg_ad)
                .clip(-1.0, 1.0)
            )  # (G, B, A_env)

            q_ensemble = jax.vmap(
                lambda ag: new_network.select("target_critic")(critic_in_obs, ag)
            )(final_env)                                  # (G, M, B)
            q_ensemble = jnp.transpose(q_ensemble, (1, 0, 2))  # (M, G, B)
            baseline = jnp.mean(q_ensemble, axis=1, keepdims=True)
            adv_ensemble = q_ensemble - baseline
            if self.config["conservative_advantage"]:
                advantages = _conservative_advantage(adv_ensemble)
            else:
                advantages = jnp.mean(adv_ensemble, axis=0)  # (G, B)

            # ── 5b. Chi2 penalty via jax.lax.scan ───────────────────────────── #
            if self.config.get("chi2_reg", False) and self.vla_slow_state is not None:
                def compute_slow_lp(_, traj_g):
                    lp = self._eval_vla_traj_log_probs_flat(
                        self.vla_slow_state, traj_g,
                        kv_cache, prefix_mask, prefix_mask_nr, sigma, num_steps,
                    )
                    return None, lp

                _, slow_log_probs = jax.lax.scan(
                    compute_slow_lp, None, trajs_stacked
                )  # (G, B)
                beta = float(self.config["chi2_beta_init"]) * jnp.std(q_ensemble)
                advantages = advantages - beta * jnp.exp(
                    old_lps_stacked - slow_log_probs
                )

            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # ── 6. Single value_and_grad over all G groups ───────────────────── #
            # Action head is tiny (~4 Linear layers) so no gradient accumulation
            # is needed.  Compute new log-probs for all G groups via an inner scan
            # (sequential over groups, bounded memory), then backprop once.
            def full_ppo_loss(state):
                def eval_group_lp(_, traj_g):
                    new_lp = self._eval_vla_traj_log_probs_flat(
                        state, traj_g, kv_cache, prefix_mask, prefix_mask_nr,
                        sigma, num_steps,
                    )
                    return None, new_lp

                _, new_lps = jax.lax.scan(eval_group_lp, None, trajs_stacked)  # (G, B)
                ratio = jnp.exp(new_lps - old_lps_stacked)
                pg1 = -advantages * ratio
                pg2 = -advantages * jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
                ppo_loss = jnp.maximum(pg1, pg2).mean()
                return ppo_loss, {
                    "ppo_loss": ppo_loss,
                    "advantage_mean": advantages.mean(),
                    "approx_kl": (old_lps_stacked - new_lps).mean(),
                    "clip_frac": (jnp.abs(ratio - 1.0) > clip_eps).mean(),
                    "ratio_mean": ratio.mean(),
                }

            (_, ppo_info), total_grads = jax.value_and_grad(
                full_ppo_loss, has_aux=True
            )(self.vla_online_state)
            info.update({f"vla/{k}": v for k, v in ppo_info.items()})

            # ── 7. Apply optax update + EMA ──────────────────────────────────── #
            updates, new_vla_opt_state = self.vla_tx.update(total_grads, self.vla_opt_state)
            new_vla_online_state = optax.apply_updates(self.vla_online_state, updates)

            tau_ema = 1.0 - float(self.config["ema_decay"])
            new_vla_ema_state = jax.tree_util.tree_map(
                lambda p, e: tau_ema * p + (1.0 - tau_ema) * e,
                new_vla_online_state, self.vla_ema_state,
            )
            if self.config.get("chi2_reg", False) and self.vla_slow_state is not None:
                tau_slow = 1.0 - float(self.config["slow_ema_decay"])
                new_vla_slow_state = jax.tree_util.tree_map(
                    lambda p, s: tau_slow * p + (1.0 - tau_slow) * s,
                    new_vla_online_state, self.vla_slow_state,
                )
            else:
                new_vla_slow_state = self.vla_slow_state

        else:
            # Fall back to FlowActor PPO when VLA state is not initialised
            rng, actor_rng = jax.random.split(rng)

            def actor_loss_fn(grad_params):
                return self.actor_loss(batch, grad_params, actor_rng, succ_batch=succ_batch)

            new_network, actor_info = new_network.apply_loss_fn(loss_fn=actor_loss_fn)
            info.update(actor_info)
            new_vla_online_state = self.vla_online_state
            new_vla_ema_state = self.vla_ema_state
            new_vla_slow_state = self.vla_slow_state
            new_vla_opt_state = self.vla_opt_state

        self.ema_update(new_network, "flow", "ema_flow")
        self.slow_ema_update(new_network)
        _maybe_log_jax_memory("update_end")
        return self.replace(
            network=new_network,
            rng=new_rng,
            vla_online_state=new_vla_online_state,
            vla_ema_state=new_vla_ema_state,
            vla_slow_state=new_vla_slow_state,
            vla_opt_state=new_vla_opt_state,
        ), info

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

        # ── VLA fine-tuning state ─────────────────────────────────────────── #
        # Only the action expert layers are trained (action_in_proj, time_mlp_in,
        # time_mlp_out, action_out_proj).  The PaliGemma backbone is frozen and
        # stored separately as ``vla_backbone_state`` — it is never differentiated
        # and never included in the optimizer state, saving ~21 GB of Adam m/v.
        if steervla_actor is not None and getattr(steervla_actor, "_local_ready", False):
            vla_graphdef, vla_init_full_state = nnx.split(steervla_actor.model)
            # Split: action-head params (trainable) + backbone params (frozen).
            vla_action_head_state, vla_backbone_state = _vla_state_split(vla_init_full_state)
            # Deep copy via tree_map so ema/slow start at the same weights but
            # are independent pytrees (updates don't alias online_state).
            vla_online_state = vla_action_head_state
            vla_ema_state = jax.tree_util.tree_map(lambda x: x, vla_action_head_state)
            vla_slow_state = (
                jax.tree_util.tree_map(lambda x: x, vla_action_head_state)
                if config.get("chi2_reg", False)
                else None
            )
            vla_lr = float(config.get("vla_lr", config["lr"]))
            vla_tx = optax.adam(learning_rate=vla_lr)
            # Optimizer state only for action-head (~few MB vs ~21 GB for full model).
            vla_opt_state = vla_tx.init(vla_action_head_state)
        else:
            vla_graphdef = None
            vla_tx = None
            vla_online_state = None
            vla_ema_state = None
            vla_slow_state = None
            vla_opt_state = None
            vla_backbone_state = None

        return cls(
            rng=rng,
            network=network,
            config=flax.core.FrozenDict(**config),
            vla_sample_fn=vla_sample_fn,
            steervla_actor=steervla_actor,
            vla_graphdef=vla_graphdef,
            vla_tx=vla_tx,
            vla_online_state=vla_online_state,
            vla_ema_state=vla_ema_state,
            vla_slow_state=vla_slow_state,
            vla_opt_state=vla_opt_state,
            vla_backbone_state=vla_backbone_state,
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
            # VLA fine-tuning (OGPO direct VLA PPO)
            vla_lr=3e-5,           # learning rate for VLA action-expert update
            vla_update_flow_steps=10,  # denoising steps used in PPO trajectory eval
        )
    )
