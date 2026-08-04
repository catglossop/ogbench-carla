"""``get_config()`` for **Best-of-N selection + CAST-relabel high-level training**.

This is the merge of ``steervla_best_of_n_config.py`` (critic-guided selection over CoT
candidates) and ``steervla_cast_relabel_train_config.py`` (VLM window review -> per-chunk
GOOD/BAD credit -> corrected subtasks -> OpenPI VLM-backbone gradient step). The two compose
cleanly because they touch disjoint machinery:

  * Best-of-N owns **action selection**: each env step ``SteerVLAActor.sample_candidates``
    draws ``best_of_n`` CoTs at ``vla_cot_temperature`` and the critic executes the argmax-Q
    chunk. The winning CoT (reasoning + subtask) is stashed onto the raw obs by
    ``stash_candidate_cot``.
  * CAST relabel owns **supervision**: ``OnlineCastRelabelSession`` is a pure observer driven
    from ``main_carla`` off ``agent_config.cast_relabel``; it reads exactly that stashed CoT
    (``record_model_input(subtask=..., reasoning=...)``), so every HL sample it writes is keyed
    to the *executed*, critic-selected candidate — not to an arbitrary sample. GOOD/unlabeled
    chunks stored via ``store_good_chunks`` therefore reinforce the candidate the critic picked;
    BAD chunks replace it with the VLM's corrected subtask.

The RL critic is **frozen** here (``enable_updates_rl=False``): it is warm-started from a
pretrained checkpoint and only used to rank candidates, while the only gradients flowing are
the high-level ones into the VLM backbone (``enable_updates_bc_hl=True``).

Note on ``critic_pretrained_weights``: setting it is a hard override that forces the
no-language critic path everywhere (``resolve_critic_feedback_mode`` -> ``"none"``, and
``BestOfNAgent.create`` mirrors it by dropping the SigLIP encoder and setting ``lang_dim=0``).
So candidate ranking here is ``argmax_i Q(obs_e, action_i)`` — unconditioned on the candidate
subtask — because the pretrained critic was trained without a language label. Drop
``critic_pretrained_weights`` (and let the critic train from scratch) if you want the
``subtask_siglip``-conditioned ``argmax_i Q([obs_e; subtask_i], action_i)`` variant instead.
The checkpoint below has critic input width 3496 = 3*1152 (SigLIP image + prompt + subtask
slots, i.e. ``siglip_include_prompt_subtask=True``) + 40 (10x4 action chunk); keep those knobs
in sync or ``_load_pretrained_critic_params`` will raise on the shape check.

GPUs: ``training_gpu_rank`` holds the RL critic + SteerVLA inference (including the N-candidate
batched forward), ``steervla.hl_training_gpu_rank`` isolates the HL train state on its own card.
Both are JAX device indices, NOT CARLA's ``-graphicsadapter`` rank.
"""

import ml_collections

from configs.steervla_best_of_n_config import get_config as get_best_of_n_config


def get_config():
    config = get_best_of_n_config()

    # ── selection (best-of-N) ────────────────────────────────────────────────────
    # Critic used purely as a ranker over the N CoT candidates. Directory form: the loader
    # resolves it to latest.pkl / the highest step_*.pkl (here step_0012000.pkl).
    config.critic_pretrained_weights = "/raid/users/cglossop/critic_ckpts"
    # Must stay True: the pretrained critic's obs_enc is 3*siglip_embed_dim wide.
    config.siglip_include_prompt_subtask = True
    # CoT sampling temperature for the N candidates. ``vla_cot_temperature`` is the one
    # ``BestOfNAgent.sample_actions_with_vla`` passes to ``sample_candidates``;
    # ``steervla.cot_temperature`` covers the non-candidate single-sample paths. Keep them
    # equal. Must stay > 0 or subtask decoding goes greedy and all N candidates collapse to
    # the same text.
    config.vla_cot_temperature = 0.5
    config.steervla.cot_temperature = 0.5

    # ── training gates ───────────────────────────────────────────────────────────
    # Only the high-level (VLM-backbone) update runs. The critic is frozen at its pretrained
    # weights and best-of-N has no noise actor, so there is nothing else to train.
    config.enable_updates = True
    config.enable_updates_rl = False
    config.enable_updates_bc = False
    config.enable_updates_bc_hl = True
    # main_carla only enters the update block once the buffer holds a full batch and warmup is
    # over, so keep both small — the HL step's own cadence is steervla.hl_update_every.
    config.batch_size = 32
    config.warmup_steps = 50
    config.update_interval = 10
    config.updates_per_step = 1
    config.buffer_capacity = 5_000

    # ── CAST relabel observer (identical pipeline to steervla_cast_relabel_config) ──
    config.cast_relabel = ml_collections.ConfigDict(
        dict(
            enabled=True,
            debug=True,
            debug_task=False,
            # Review one window every N env steps (rounded down to whole action chunks).
            query_every_n_episode_steps=150,
            query_on_episode_end=True,
            provider="gemini",
            gemini_model="gemini-3.5-flash",
            # Must match config.action_horizon (and steervla.action_horizon).
            action_chunk_steps=10,
            num_subtask_suggestions=3,
            video_fps=10.0,
            video_frame_stride=2,
            save_artifacts=True,
            # Persist BAD/relabeled chunks as steervla_hl_dataset_format samples for update_hl.
            store_hl_dataset=True,
            # Reinforce GOOD chunks with the *critic-selected* subtask (see module docstring) —
            # this is the part that makes the merge more than a coincidence of configs.
            store_good_chunks=True,
            hl_dataset_subdir="cast_relabel_hl_dataset",
            hl_action_dim=4,
            seed_subtasks=[],
        )
    )

    # ── SteerVLA: trainable (HL) on top of the best-of-N candidate sampler ───────
    # Load the full OpenPI TrainState so update_hl can take real gradient steps. This is on top
    # of the inference copy used by sample_candidates, so plan VRAM accordingly: keep the HL
    # train state on its own GPU via hl_training_gpu_rank.
    config.steervla.load_trainable_params = True
    config.steervla.hl_training_gpu_rank = 1
    # Freeze the memory-heavy pretrained subtrees (SigLIP tower + tied Gemma embedder); the HL
    # step supervises only CoT/subtask targets, with the action loss masked out.
    config.steervla.hl_freeze_regexes = [".*img.*", ".*embedder.*"]
    # Measured on an 80 GB H100 with the freeze list above: batch 64 needs a 71.65 GiB peak
    # (18.84 GiB fixed for trainable params + Adam mu/nu, ~0.83 GiB of activations per sample
    # even after XLA rematerialization) and OOMs, because JAX preallocates only ~75% of the
    # card. 16 lands near a 32 GiB peak, which leaves real headroom. Raise only with the
    # rematerialization line in the log as evidence -- the sibling cast_relabel config's
    # "start at 128" comment does not hold for this checkpoint.
    config.steervla.hl_update_batch_size = 16
    config.steervla.hl_update_every = 5
    config.steervla.hl_update_num_steps = 2
    config.steervla.hl_min_online_samples = 20
    # Mix pretraining data into every HL batch so the backbone doesn't forget general CoT/FAST.
    config.steervla.hl_replay_root = "/raid/users/cglossop/steervla_hl_pools"
    config.steervla.hl_online_weight = 0.7
    config.steervla.hl_online_bad_fraction = 0.8
    config.steervla.hl_online_precursor_fraction = 0.5
    config.steervla.hl_replay_pools = [
        dict(name="simlingo_dataset_all_img512_1116", weight=0.2),
        dict(name="simplified_reasoning_dataset", weight=0.1),
    ]

    return config
