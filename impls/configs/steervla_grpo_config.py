"""``get_config()`` for GRPO on the SteerVLA high-level (CoT/subtask) policy.

Use with ``--agent=impls/configs/steervla_grpo_config.py``. ``main_carla`` dispatches to the GRPO
path when ``online_training_mode == "grpo_hl"`` (see :func:`main_carla._run_grpo_entry`). This is a
standalone RL baseline: no cast_relabel, no VLM coach, no DSRL critic. Each group is ``grpo.group_size``
rollouts of the frozen base policy on the SAME route/env seed (only CoT sampling varies), the per-episode
environment return is the reward, and :meth:`vlas.steervla.SteerVLAActor.update_hl_grpo` takes a
group-relative policy-gradient step on the CoT tokens with a KL penalty to a frozen reference. The action
expert is never modified (the CoT cross-entropy is independent of it).

Requires ``steervla.load_trainable_params=True`` (the HL backbone must be a full OpenPI TrainState) and a
positive ``steervla.cot_temperature`` (greedy decoding gives an all-zero-advantage group -> no signal).
"""

import ml_collections

from configs.steervla_dsrl_config import get_config as get_dsrl_config


def get_config():
    config = get_dsrl_config()

    # GRPO runs its own group loop; the DSRL critic/coach are unused. Keep the critic language source
    # off so no VLM coach spins up if some other codepath inspects it.
    config.language_feedback.source = "expert"
    config.language_feedback.expert_mode = "none"
    config.online_training_mode = "grpo_hl"
    config.training_gpu_rank = 0
    config.siglip_device = "cuda:0"

    # Env-step budget (total across all groups); one rollout ends at the route's terminal/truncation.
    config.buffer_capacity = 1_000  # unused by the GRPO loop; kept so shared plumbing stays happy.

    # Rollout video (first rollout of each group is captured and logged to W&B as grpo/rollout_video).
    config.log_episode_video = True
    config.episode_video_fps = 10.0
    config.episode_video_every = 2

    # Trainable HL backbone (full OpenPI TrainState) + the frozen KL reference snapshot are built in
    # SteerVLAActor.setup when load_trainable_params=True.
    config.steervla.load_trainable_params = True
    # Isolate the HL (VLM-backbone) gradient step on its own JAX GPU (index into jax.devices("gpu"),
    # NOT CARLA's -graphicsadapter). -1 keeps it on training_gpu_rank. run_carla.sh --hl-gpu overrides.
    config.steervla.hl_training_gpu_rank = 1
    # Freeze the memory-heavy pretrained subtrees for the HL step (SigLIP tower + tied token embedder);
    # the CoT policy gradient only needs the LLM transformer blocks + CoT heads. [] = full fine-tune.
    config.steervla.hl_freeze_regexes = [".*img.*", ".*embedder.*"]
    # Minibatch size for the GRPO update: a full group of rollouts is far too large for one Pi0-CoT
    # forward+backward, so update_hl_grpo shuffles the pooled CoTs into minibatches of this size.
    # 16 fits an 80 GB card (see steervla_cast_relabel_config.py); drop to 8 if memory is tight.
    config.steervla.hl_update_batch_size = 16
    # GRPO needs exploration: sample CoTs stochastically so the group's rollouts differ. Greedy (0.0)
    # yields identical CoTs -> zero-variance returns -> zero advantages -> no learning signal.
    config.steervla.cot_temperature = 1.0

    config.grpo = ml_collections.ConfigDict(
        dict(
            # Rollouts per group. Advantages are normalized within the group, so >=2 is required; more
            # rollouts give a lower-variance baseline at linear wall-clock cost (each is a full episode).
            group_size=8,
            # KL(πθ‖π_ref) penalty weight against the frozen base HL policy (0 = pure REINFORCE).
            beta_kl=0.01,
            # Passes over the pooled group each update (minibatched inside update_hl_grpo).
            num_update_steps=1,
            advantage_eps=1e-6,
            # Export the fine-tuned HL backbone to <save_dir>/steervla_hl_ckpt/<step> every N groups
            # (0 = only the final export). Redeploy frozen on other routes: steervla.checkpoint=<step>
            # dir, same actor_config, load_trainable_params=False.
            checkpoint_every_groups=5,
        )
    )

    return config
