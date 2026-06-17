"""Minimal JAX implementation of Best of N.
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

    ``next_actions_critic`` and ``critic_actions`` must already be clipped to the env
    action layout (e.g. via :meth:`DSRLAgent._clip_actions_to_env`).
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
class BestOfNAgent(flax.struct.PyTreeNode):
    """Minimal JAX DSRL agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()
    vla_sample_fn: Any = nonpytree_field()  # Optional callable: (obs, noise) -> action
    # vla_train_state: Any = nonpytree_field()  # OpenPI ``training_utils.TrainState`` for :meth:`update_with_vla`
    openpi_train_config: Any = nonpytree_field()  # ``openpi.training.config.TrainConfig`` for VLA flow step
    steervla_actor: Any = nonpytree_field()  # Optional attached local ``SteerVLAActor`` instance
    siglip_encoder: Any = nonpytree_field()  # Optional ``utils.siglip_encoder.SigLIPEncoder`` for subtask labels

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

    def _encode_subtask_labels(self, subtask_texts) -> jnp.ndarray:
        """Encode candidate subtask strings into ``(n, language_label_dim)`` SigLIP text embeddings.

        Returns zeros (no language conditioning) when no SigLIP encoder is attached.
        """
        n = len(subtask_texts)
        if self.siglip_encoder is None:
            dim = int(self.config.get("siglip_embed_dim", self.config.get("language_label_dim", 0)))
            return jnp.zeros((n, dim), dtype=jnp.float32)
        import numpy as np

        vecs = [np.asarray(self.siglip_encoder.encode_text(t), dtype=np.float32) for t in subtask_texts]
        return jnp.asarray(np.stack(vecs, axis=0), dtype=jnp.float32)

    def sample_actions_with_vla(self, observations, seed=None, temperature=1.0):
        """Best-of-N rollout: sample many CoTs, score each chunk with the critic, execute the best.

        The attached :class:`SteerVLAActor` samples ``best_of_n`` chains-of-thought at
        ``vla_cot_temperature`` in one batched forward (each candidate's reasoning + subtask is
        printed for debugging). Each candidate's subtask is embedded with SigLIP and used as the
        critic language label, so the winner is ``argmax_i Q([obs_e; subtask_i], action_i)``. The
        winning CoT tokens and subtask embedding are stashed into the raw obs holder so the replay
        transition (and thus the critic) trains on the executed action **and** its subtask.
        """
        if self.vla_sample_fn is None:
            raise RuntimeError(
                "BestOfNAgent requires a SteerVLA vla_sample_fn; it has no internal BC-flow policy."
            )
        actor = self.steervla_actor
        if actor is None or not hasattr(actor, "sample_candidates"):
            raise RuntimeError(
                "BestOfNAgent.sample_actions_with_vla requires a local SteerVLAActor with "
                "sample_candidates() (remote/best-of-n over CoTs is not supported)."
            )

        n = int(self.config.get("best_of_n", 10))
        cot_temperature = float(self.config.get("vla_cot_temperature", 1.0))
        seed = seed if seed is not None else self.rng
        seed, rng_cand = jax.random.split(seed)

        raw = None
        if getattr(actor, "raw_obs_holder", None) is not None:
            raw = actor.raw_obs_holder.get("obs")

        cands = actor.sample_candidates(n, temperature=cot_temperature, rng=rng_cand, raw=raw)
        actions_flat = jnp.asarray(cands["actions"], dtype=jnp.float32)  # (n, env_flat)
        subtask_texts = cands["subtask_texts"]

        lang = self._encode_subtask_labels(subtask_texts)  # (n, language_label_dim)
        obs_e = self._encode_obs(self.network.params, observations)  # (1, embed)
        obs_e_n = jnp.broadcast_to(obs_e, (actions_flat.shape[0],) + obs_e.shape[1:])
        critic_in = jnp.concatenate([obs_e_n, lang], axis=-1)
        critic_actions = self._clip_actions_to_env(actions_flat)
        qs = self.network.select("critic")(critic_in, critic_actions)  # (ensemble, n)
        q = jnp.min(qs, axis=0)  # (n,)
        best = int(jnp.argmax(q))

        import numpy as np

        print(
            f"[best_of_n] selected candidate {best}/{n} "
            f"q={float(q[best]):.4f} subtask={subtask_texts[best]!r} "
            f"q_all={np.asarray(jax.device_get(q)).round(3).tolist()}",
            flush=True,
        )

        # Persist the winner's CoT + subtask label so replay capture / critic training stay aligned.
        actor.stash_candidate_cot(cands["cot_out"], best, raw)
        if raw is not None:
            raw["language_label"] = np.asarray(jax.device_get(lang[best]), dtype=np.float32)

        return actions_flat[best][None]

    # ----- losses --------------------------------------------------------- #

    def _vla_forward(self, observations, openpi_observations, rng, noise=None):
        rng_n, rng_act = jax.random.split(rng)
        if noise is None:
            # BestOfN has no noise actor; seed the BC flow with random Gaussian noise for
            # the critic bootstrap target (a single VLA sample, as in standard DSRL TD).
            noise_scale = jnp.asarray(self.config["noise_scale"], dtype=jnp.float32)
            noise = jax.random.normal(rng_n, (observations.shape[0], self._flat_noise_dim())) * noise_scale
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
        if bool(self.config.get("debug_task", False)):
            batch = _apply_debug_stop_reward_relabel(batch)
        rng = rng if rng is not None else self.rng
        rng, nc_rng, na_rng, c_rng = jax.random.split(rng, 4)
        discount = jnp.asarray(self.config["discount"], dtype=jnp.float32)
        alpha = jnp.asarray(self.config["alpha"], dtype=jnp.float32)
        noise_scale = jnp.asarray(self.config["noise_scale"], dtype=jnp.float32)
        critic_actions = self._clip_actions_to_env(batch["actions"])
        noise_total_loss_vla_time = time.time()
        if vla_cache is not None:
            c_loss, c_info = _critic_loss_vla_pure_math(
                self.network,
                batch,
                vla_cache["next_actions_critic"],
                critic_actions,
                grad_params,
                discount,
            )
        else:
            c_loss, c_info = self.critic_loss_vla(batch, grad_params, c_rng)
        total_loss_vla_time = time.time() - noise_total_loss_vla_time
        print(f"[DEBUG - dsrl] Total loss vla time: {total_loss_vla_time} seconds")
        info = {}
        for k, v in c_info.items():
            info[f"critic/{k}"] = v
        combined = c_loss
        info["total_loss_vla"] = combined
        info["flax_loss_vla"] = c_loss
        info["critic_loss_vla"] = c_loss
        return combined, info

    def target_update(self, network):
        new_target = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["tau"] + tp * (1 - self.config["tau"]),
            network.params["modules_critic"],
            network.params["modules_target_critic"],
        )
        network.params["modules_target_critic"] = new_target

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
        """Eager VLA forward consumed by :meth:`total_loss_vla` inside ``update_with_vla``.

        Only the critic bootstrap needs a VLA action (next-state); BestOfN has no noise
        critic/actor, so no current-state VLA forward is required.
        """
        rng, c_rng = jax.random.split(rng)
        next_actions = self._vla_forward(
            batch["next_observations"], batch["next_openpi_observation"], c_rng,
        )
        return {"next_actions_critic": next_actions}

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
        siglip_encoder: Any = None,
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
            if bool(config.get("siglip_include_prompt_subtask", False)):
                embed_dim = single * 3
            else:
                embed_dim = single
        else:
            embed_dim = int(tuple(config.get("image_mlp_hidden_dims", (512,)))[-1])
        critic_feedback_mode = str(config.get("critic_feedback_mode", "subtask_siglip"))
        if critic_feedback_mode == "subtask_siglip":
            # Subtask text is embedded with SigLIP and used as the critic language label.
            if siglip_encoder is None:
                from utils.siglip_encoder import SigLIPEncoder

                siglip_encoder = SigLIPEncoder(
                    model_id=str(config.get("siglip_model_id", "google/siglip2-so400m-patch14-384")),
                    device=config.get("siglip_device"),
                )
                siglip_encoder.setup()
            lang_dim = int(siglip_encoder.embedding_dim)
            try:
                config.siglip_embed_dim = lang_dim  # keep main_carla buffer width in sync
            except Exception:
                pass
        elif critic_feedback_mode == "action_delta":
            lang_dim = int(config.get("critic_action_dim", 4))
        else:
            lang_dim = int(config.get("language_label_dim", 119))
        batch_shape = ex_observations.shape[:1]
        ex_critic_embedded = jnp.zeros(batch_shape + (embed_dim + lang_dim,), dtype=jnp.float32)
        ex_env_actions = jnp.zeros(batch_shape + (env_action_dim,), dtype=jnp.float32)

        # BC flow-matching head must stay a Flax module: init calls ``flow(enc, x_t, t)``.
        # OpenPI SteerVLA hooks only via ``vla_sample_fn`` (see ``sample_actions_with_vla``).
        critic_def = Critic(
            hidden_dims=tuple(config["critic_hidden_dims"]),
            layer_norm=config["layer_norm"],
            ensemble_size=config["critic_ensemble"],
        )
        networks = {
            "obs_encoder": (obs_encoder_def, (ex_observations,)),
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
            siglip_encoder=siglip_encoder,
        )


# --------------------------------------------------------------------------- #
# Config                                                                      #
# --------------------------------------------------------------------------- #


def get_config():
    """ml_collections.ConfigDict consumed by ``main_carla.py``."""

    config = ml_collections.ConfigDict(
        dict(
            agent_name="best_of_n",
            # Number of chain-of-thought candidates sampled per env step; the critic picks the best.
            best_of_n=10,
            lr=3e-4,
            batch_size=256,
            critic_hidden_dims=(256, 256),
            critic_ensemble=2,
            layer_norm=True,
            discount=0.99,
            tau=0.005,
            alpha=0.1,
            noise_scale=1.0,
            flow_steps=5,
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
            # SigLIP runs in PyTorch; default CPU avoids competing with JAX GPU memory.
            # Online-loop knobs (used by main_carla.py, not by the agent itself).
            # Env steps with policy rollouts but no RL updates (see main_carla.run_online_carla).
            warmup_steps=1000,
            updates_per_step=1,
            # Run RL gradient updates every N env steps (1 = every step).
            update_interval=1,
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
            # Temperature for the best-of-N CoT sampling (>0 so candidates differ).
            vla_cot_temperature=1.0,
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
            # "subtask_siglip": SigLIP text embedding of the executed candidate's subtask (best-of-N default).
            # "commentary_bow": expert commentary BOW.
            # "action_delta": (critic_action_dim)-dim vector = expert first step − agent first step.
            # "delta_commentary_bow": corrective language BOW from expert-vs-agent action delta.
            critic_feedback_mode="subtask_siglip",
            # Online training regime.
            # "rl": standard DSRL online RL updates.
            # "dagger": collect on-policy states, store expert actions, and train with flow imitation only.
            online_training_mode="rl",
            # When true, RL updates use reward = -ego_speed (m/s) instead of env reward.
            debug_task=False,
            # Rollout-only: sample many random VLA flow noises and pick the slowest chunk.
            debug_noise=True,
            debug_noise_samples=15,
            debug_noise_log_every_n_steps=5,
            use_best_noise=True,
        )
    )
    return config
