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

    Requires the replay field ``ego_speed`` (stored by main_carla when
    ``debug_task`` is set). Keeps the stored env reward intact for logging.
    """
    if "ego_speed" not in batch:
        raise KeyError(
            "debug_task requires replay field 'ego_speed' (store CARLA speed m/s in main_carla)."
        )
    out = dict(batch)
    out["rewards"] = -jnp.asarray(out["ego_speed"], dtype=jnp.float32)
    return out


class StateHead(nn.Module):
    """Shared trainable bottleneck on the frozen state feature, consumed by both actor and critic.

    Compresses the high-dim frozen encoder feature into a representation trained end-to-end by the
    critic's TD loss (the actor sees it detached, SAC-AE style, so the actor objective can't collapse
    the rep). ``mlp`` adds one GELU hidden layer -- worthwhile only when the feature exposes structure
    to mix (e.g. pi_prefix_groups); a linear head on a single pooled vector is just a low-rank reparam.
    Only created when ``state_head_dim > 0``.
    """

    out_dim: int
    layer_norm: bool = True
    mlp: bool = False

    @nn.compact
    def __call__(self, x):
        if self.mlp:
            x = nn.Dense(self.out_dim, kernel_init=default_init())(x)
            x = nn.gelu(nn.LayerNorm()(x) if self.layer_norm else x)
        x = nn.Dense(self.out_dim, kernel_init=default_init())(x)
        if self.layer_norm:
            x = nn.LayerNorm()(x)
        return x


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

    def _enc(self, x, grad_params=None, target=False):
        """Map the frozen state feature through the shared trainable head (identity if disabled).

        Pass ``grad_params`` only on the critic's current-Q path so the rep is shaped by TD; the
        actor calls this without ``grad_params`` (detached). ``target`` selects the slow target head,
        paired with the target critic in TD backups.
        """
        if not self.config["use_state_head"]:
            return x
        return self.network.select("target_state_head" if target else "state_head")(x, params=grad_params)

    def _edit(self, z, base_action, scale, rng):
        """EXPO edit on an already head-encoded state ``z``: ã = clip(base + scale * residual)."""
        dist = self.network.select("actor")(z, base_action)
        residual = dist.sample(seed=rng)
        return self._final_action(base_action, residual, scale)

    def _otf_next_q(self, next_obs_cands, next_base_cands, scale, rng):
        """EXPO hard-max TD target: edit each of N base cands 1:1 (online actor over the online-encoded
        state), take max_i min-ensemble Q_target (target critic over the target-encoded state) across
        the 2N base+edited pool. No entropy term (argmax has no log-prob).

        next_base_cands (B, N, adim); next_obs_cands (B, K, embed), K in {1, N} broadcast to N.
        """
        b, n, adim = next_base_cands.shape
        edim = next_obs_cands.shape[-1]
        obs_cands = jnp.broadcast_to(next_obs_cands, (b, n, edim))
        z_on = self._enc(obs_cands).reshape(b * n, -1)
        edited = self._edit(z_on, next_base_cands.reshape(b * n, adim), scale, rng).reshape(b, n, adim)
        z_tg = self._enc(obs_cands, target=True)
        pool_states = jnp.concatenate([z_tg, z_tg], axis=1)
        pool_actions = jnp.concatenate([next_base_cands, edited], axis=1)
        m = 2 * n
        d = pool_states.shape[-1]
        qs = self.network.select("target_critic")(
            pool_states.reshape(b * m, d), None, actions=pool_actions.reshape(b * m, adim)
        )
        q = jnp.min(qs, axis=0).reshape(b, m)
        return jnp.max(q, axis=1)

    def critic_loss(self, batch, grad_params, rng, scale):
        if self.config["expo"] and self.config["otf_td_backup"]:
            next_q = self._otf_next_q(batch["next_obs_cands"], batch["next_base_cands"], scale, rng)
            target_q = batch["rewards"] + self.config["discount"] * batch["masks"] * next_q
            extra = {}
        else:
            # Plain soft SAC backup over the first stored base candidate (regular residual SAC
            # when expo=False, or the rollout-only EXPO ablation when otf_td_backup=False).
            next_obs = batch["next_obs_cands"][:, 0, :]
            next_base = batch["next_base_cands"][:, 0, :]
            next_dist = self.network.select("actor")(self._enc(next_obs), next_base)
            next_residual, next_log_probs = next_dist.sample_and_log_prob(seed=rng)
            next_actions = self._final_action(next_base, next_residual, scale)
            next_qs = self.network.select("target_critic")(self._enc(next_obs, target=True), None, actions=next_actions)
            next_q = jnp.min(next_qs, axis=0)
            alpha = self.network.select("alpha")()
            bootstrap = self.config["discount"] * batch["masks"]
            target_q = batch["rewards"] + bootstrap * (next_q - alpha * next_log_probs)
            extra = {"next_entropy": -next_log_probs.mean()}
        target_q = jax.lax.stop_gradient(target_q)

        q = self.network.select("critic")(
            self._enc(batch["observations"], grad_params=grad_params),
            None,
            actions=batch["actions"],
            params=grad_params,
        )
        critic_loss = jnp.square(q - target_q).mean()

        return critic_loss, {
            "critic_loss": critic_loss,
            "q_mean": q.mean(),
            "target_q_mean": target_q.mean(),
            **extra,
        }

    def actor_loss(self, batch, grad_params, rng, scale):
        # Detached rep (no grad_params): the head is trained by the critic TD loss only.
        z = self._enc(batch["observations"])
        dist = self.network.select("actor")(z, batch["base_actions"], params=grad_params)
        residual, log_probs = dist.sample_and_log_prob(seed=rng)
        actions = self._final_action(batch["base_actions"], residual, scale)

        q = jnp.min(self.network.select("critic")(z, None, actions=actions), axis=0)

        alpha = self.network.select("alpha")()
        actor_loss = (alpha * log_probs - q).mean()

        # O4 actor-side tether (offline-RL constraint ladder); additive, default off -> V0. V3 gate
        # relaxes the tether where editing beats the base (adv>0), stop-grad breaking the res->gate loop.
        scaled_res = scale * residual
        bc_beta = self.config["residual_bc_beta"]
        kl_beta = self.config["residual_kl_beta"]
        tether_info: dict[str, Any] = {}
        if bc_beta > 0.0 or kl_beta > 0.0:
            if self.config["residual_adv_gate"]:
                q_base = jnp.min(self.network.select("critic")(z, None, actions=batch["base_actions"]), axis=0)
                gate = jax.nn.sigmoid(jax.lax.stop_gradient(q - q_base) / self.config["residual_adv_gate_temp"])
                keep = 1.0 - gate
                tether_info["adv_gate_mean"] = gate.mean()
            else:
                keep = jnp.ones_like(log_probs)
            # V1 TD3+BC L2 penalty, Q-normalized (beta_eff = beta / mean|Q|) so beta transfers across routes.
            if bc_beta > 0.0:
                beta_eff = bc_beta
                if self.config["residual_bc_normalize"]:
                    beta_eff = bc_beta / (jax.lax.stop_gradient(jnp.abs(q).mean()) + 1e-6)
                bc_pen = (keep * jnp.mean(jnp.square(scaled_res), axis=-1)).mean()
                actor_loss = actor_loss + beta_eff * bc_pen
                tether_info["bc_penalty"] = bc_pen
                tether_info["bc_beta_eff"] = jnp.asarray(beta_eff, dtype=jnp.float32)
            # V2 KL(pi_res || N(0, sigma0^2 I)) on the pre-tanh Gaussian (closed-form diagonal KL).
            if kl_beta > 0.0:
                base_dist = dist.distribution
                mu, sig = base_dist.mean(), base_dist.stddev()
                sig0 = self.config["residual_kl_sigma0"]
                kl = jnp.sum(
                    jnp.log(sig0) - jnp.log(sig) + (sig**2 + mu**2) / (2.0 * sig0**2) - 0.5, axis=-1
                )
                kl_pen = (keep * kl).mean()
                actor_loss = actor_loss + kl_beta * kl_pen
                tether_info["kl_to_prior"] = kl.mean()

        grad_alpha = self.network.select("alpha")(params=grad_params)
        entropy = -jax.lax.stop_gradient(log_probs).mean()
        alpha_loss = (grad_alpha * (entropy - self.config["target_entropy"])).mean()

        return actor_loss + alpha_loss, {
            "actor_loss": actor_loss,
            "alpha_loss": alpha_loss,
            "alpha": alpha,
            "entropy": -log_probs.mean(),
            "residual_scale": jnp.mean(scale),
            "residual_abs_mean": jnp.abs(scaled_res).mean(),
            **tether_info,
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
        if self.config["use_state_head"]:
            self.target_update(new_network, "state_head")
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, base_actions, scale, seed=None, temperature=1.0):
        """Sample ``(final_action, residual)`` for a batched state + base chunk at ``scale``."""
        dist = self.network.select("actor")(self._enc(observations), base_actions, temperature=temperature)
        residual = dist.sample(seed=seed)
        return self._final_action(base_actions, residual, scale), residual

    @jax.jit
    def score_action_otf(self, x_cands, base_cands, scale, seed):
        """Sample one residual edit per base candidate and score the complete EXPO pool.

        Returns ``(edited, critic_heads)`` where ``edited`` is ``(N, action_dim)`` and
        ``critic_heads`` is ``(ensemble, 2N)`` in ``[base candidates, edited candidates]``
        order. Keeping all heads lets the rollout loop make conservative paired decisions
        or escalate uncertain choices to an external selector such as Gemini.
        """
        n, adim = base_cands.shape
        edim = x_cands.shape[-1]
        x_full = jnp.broadcast_to(x_cands, (n, edim))
        z = self._enc(x_full)
        dist = self.network.select("actor")(z, base_cands)
        residual = dist.sample(seed=seed)
        edited = self._final_action(base_cands, residual, scale)
        pool_states = jnp.concatenate([z, z], axis=0)
        pool_actions = jnp.concatenate([base_cands, edited], axis=0)
        critic_heads = self.network.select("critic")(pool_states, None, actions=pool_actions)
        return edited, critic_heads

    @jax.jit
    def select_action_otf(self, x_cands, base_cands, scale, seed):
        """EXPO rollout: edit each of N base cands 1:1, execute the argmax-min-ensemble-Q of the
        2N base+edited pool.

        base_cands (N, adim); x_cands (K, embed), K in {1, N} broadcast to N. Returns
        (executed, winner_base, winner_x, q_all (2N,), winner_idx); winner_idx >= N -> an edit won.
        """
        n, adim = base_cands.shape
        edim = x_cands.shape[-1]
        x_full = jnp.broadcast_to(x_cands, (n, edim))
        z = self._enc(x_full)
        dist = self.network.select("actor")(z, base_cands)
        residual = dist.sample(seed=seed)
        edited = self._final_action(base_cands, residual, scale)
        pool_states = jnp.concatenate([z, z], axis=0)
        pool_actions = jnp.concatenate([base_cands, edited], axis=0)
        qs = self.network.select("critic")(pool_states, None, actions=pool_actions)
        q = jnp.min(qs, axis=0)
        winner = jnp.argmax(q)
        j = jnp.mod(winner, n)
        # Return the RAW feature x_full[j] as winner_x: the buffer stores frozen features and the
        # head is re-applied inside the agent each update.
        return pool_actions[winner], base_cands[j], x_full[j], q, winner

    @classmethod
    def create(cls, seed, ex_observations, ex_base_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        action_dim = ex_base_actions.shape[-1]
        target_entropy = config["target_entropy"]
        if target_entropy is None:
            target_entropy = -float(config["target_entropy_multiplier"]) * action_dim

        # Optional shared bottleneck: actor/critic then consume the head's output (dim d), not the
        # raw frozen feature. Size them on a d-wide example so their input dims match runtime.
        state_head_dim = int(config.get("state_head_dim", 0))
        use_state_head = state_head_dim > 0
        ex_state = (
            jnp.zeros((*ex_observations.shape[:-1], state_head_dim), dtype=jnp.float32)
            if use_state_head
            else ex_observations
        )

        actor_def = ResidualActor(
            hidden_dims=tuple(config["actor_hidden_dims"]),
            action_dim=action_dim,
            layer_norm=config["layer_norm"],
        )
        critic_def = GCValue(hidden_dims=tuple(config["value_hidden_dims"]), layer_norm=config["layer_norm"], ensemble=True)
        alpha_def = LogParam()

        network_info = dict(
            critic=(critic_def, (ex_state, None, ex_base_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_state, None, ex_base_actions)),
            actor=(actor_def, (ex_state, ex_base_actions)),
            alpha=(alpha_def, ()),
        )
        if use_state_head:
            head_def = StateHead(
                out_dim=state_head_dim,
                layer_norm=config["layer_norm"],
                mlp=bool(config.get("state_head_mlp", False)),
            )
            network_info["state_head"] = (head_def, (ex_observations,))
            network_info["target_state_head"] = (copy.deepcopy(head_def), (ex_observations,))
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)
        network.params["modules_target_critic"] = network.params["modules_critic"]
        if use_state_head:
            network.params["modules_target_state_head"] = network.params["modules_state_head"]

        # O4 ladder is an ablation: at most one tether (V1 XOR V2), V3 only modulates an active tether.
        bc_beta = float(config.get("residual_bc_beta", 0.0))
        kl_beta = float(config.get("residual_kl_beta", 0.0))
        adv_gate = bool(config.get("residual_adv_gate", False))
        if bc_beta > 0.0 and kl_beta > 0.0:
            raise ValueError("Enable at most one of residual_bc_beta (V1) / residual_kl_beta (V2).")
        if adv_gate and bc_beta <= 0.0 and kl_beta <= 0.0:
            raise ValueError("residual_adv_gate (V3) needs an active tether (residual_bc_beta or residual_kl_beta > 0).")

        agent_config = dict(
            discount=float(config["discount"]),
            tau=float(config["tau"]),
            target_entropy=float(target_entropy),
            debug_task=bool(config.get("debug_task", False)),
            expo=bool(config.get("expo", True)),
            otf_td_backup=bool(config.get("otf_td_backup", True)),
            use_state_head=bool(use_state_head),
            # O4 actor-side offline-RL tether (constraint ladder); all default off -> V0 behavior.
            residual_bc_beta=bc_beta,
            residual_bc_normalize=bool(config.get("residual_bc_normalize", False)),
            residual_kl_beta=kl_beta,
            residual_kl_sigma0=float(config.get("residual_kl_sigma0", 1.0)),
            residual_adv_gate=adv_gate,
            residual_adv_gate_temp=float(config.get("residual_adv_gate_temp", 1.0)),
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
            # Shared trainable state bottleneck (Dense(state_head_dim)+LayerNorm on the frozen encoder
            # feature) consumed by both actor and critic and trained by the critic TD loss (actor sees
            # it detached). 0 disables it -> the frozen feature is fed straight to the MLPs.
            state_head_dim=0,
            # Make the head nonlinear (one GELU hidden layer). Only helps when the feature has structure
            # to mix across (e.g. state_encoder="pi_prefix_groups"); pointless on a single pooled vector.
            state_head_mlp=False,
            discount=0.99,
            tau=0.005,
            # Deprecated shared authority. It remains a compatibility fallback for old
            # ``--agent.residual_scale=...`` commands; new experiments should specify
            # both per-control scales below.
            residual_scale=0.1,
            # accel_steer only: explicit per-control residual authority. A negative
            # accel value falls back to the legacy shared ``residual_scale``; a negative
            # steer value reuses the resolved accel authority. This preserves old commands
            # while making throttle/brake and steering sweeps unambiguous.
            residual_accel_scale=-1.0,
            residual_steer_scale=-1.0,
            # Consumed by main_carla.py (the SAC agent itself is space-agnostic):
            #   "accel_steer" (default) -> base chunk is PID-decoded to a 2-D [accel, steer] before residual is applied.
            #   "waypoint_chunk" -> residual acts on the flattened normalized 40-D chunk
            residual_action_space="accel_steer",
            target_entropy=ml_collections.config_dict.placeholder(float),  # None -> auto.
            target_entropy_multiplier=0.5,
            residual_warmup_steps=2000,
            residual_ramp_steps=3000,
            # O4 actor-side tether ladder (default off = V0). V1 L2 penalty with a
            # fixed configured beta; V2 KL to N(0, sigma0^2 I); V3 gate relaxes V1/V2
            # where Q(base+res) > Q(base). Q-normalization remains an explicit legacy option.
            residual_bc_beta=0.0,
            residual_bc_normalize=False,
            residual_kl_beta=0.0,
            residual_kl_sigma0=1.0,
            residual_adv_gate=False,
            residual_adv_gate_temp=1.0,
            expo=False,
            best_of_n=8,
            vla_cot_temperature=1.0,
            otf_td_backup=False,
            # EXPO uses the online critic only. Conservative mode admits an edited
            # candidate only when both critic heads predict a paired improvement.
            expo_conservative=False,
            updates_per_step=10,
            # Debug task: RL updates use reward = -ego_speed (m/s) instead of env reward,
            # so the policy should learn to brake to a stop.
            debug_task=False,
        )
    )
