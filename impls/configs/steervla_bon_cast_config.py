"""``get_config()`` for **static-critic Best-of-N + CAST-relabel HL training** (no residual).

This is the base of the best-of-N + CAST family. ``steervla_bon_cast_residual_config`` inherits
from it and adds *only* the residual actor, so the two differ by exactly the residual and
nothing else.

Two mechanisms:

  1. **Selection** — each env step the frozen Pi0 is sampled ``bon_num_candidates`` times, one
     draw at a time with the actor's action cache reset between draws, so every candidate gets
     its own freshly sampled CoT/subtask and therefore its own action chunk. A **static**
     pretrained critic then executes ``argmax_i Q(obs_e, action_i)``. The subtask's influence
     reaches the critic through the action chunk it produced, which is the per-candidate input
     being scored.
  2. **Supervision** — ``OnlineCastRelabelSession`` reviews rollout windows with a VLM, assigns
     per-chunk GOOD/BAD credit and writes corrected subtasks as high-level samples that
     ``SteerVLAActor.update_hl`` trains the VLM backbone on.

**Nothing is trained except the VLM backbone.** ``enable_updates_rl=False``: the selection
critic is loaded from ``--bon_critic_ckpt`` into a jitted closure over fixed params
(``main_carla._bon_q_fn``) that no gradient can reach, and with RL off there is no other critic
update either.

Why ``agent_name`` stays ``"dsrl"`` and not ``"best_of_n"``
----------------------------------------------------------
``BestOfNAgent`` selects via ``SteerVLAActor.sample_candidates``, which draws all N CoTs in a
single batched forward and never resets the actor's action cache between draws. This config
deliberately uses the flag-driven path in ``main_carla`` instead
(``--bon_critic_ckpt`` -> ``_sample_diverse_candidates`` + ``_score_candidates_with_critic``),
which samples one candidate at a time, resets the cache between draws, and resamples up to
``--bon_max_sample_attempts`` times per slot to keep the subtasks genuinely distinct. That is
the path the residual variant uses and the one that produced sane rollouts;
``steervla_bon_cast_relabel_config`` (the ``agent_name="best_of_n"`` variant) is superseded.

Requires on the command line:
  ``--train-mode rl``            no residual actor (run_carla.sh always writes this into the config)
  ``--bon_critic_ckpt <path>``   the static selection critic
  ``--bon_num_candidates N``     candidates per step (main_carla reads the flag, not a config key)
"""

import ml_collections

from configs.steervla_cast_relabel_train_config import get_config as get_cast_relabel_config


def get_config():
    config = get_cast_relabel_config()

    # ── no residual: best-of-N's pick is executed as-is ──────────────────────────
    config.online_training_mode = "rl"

    # ── training gates: VLM backbone only ────────────────────────────────────────
    config.enable_updates = True
    config.enable_updates_rl = False
    config.enable_updates_bc = False
    config.enable_updates_bc_hl = True

    # ── update sizing ────────────────────────────────────────────────────────────
    # With RL off, batch_size only decides how full the buffer must be before main_carla
    # enters the update block at all; the HL step's own cadence is steervla.hl_update_every.
    # Kept equal to the residual variant so the two configs differ only by the residual.
    config.batch_size = 64
    config.buffer_capacity = 5_000
    config.warmup_steps = 500
    config.update_interval = 10
    config.updates_per_step = 5

    # ── critic observation (must match the static selection critic's input width) ─
    # The checkpoint is 3496 = 3*1152 (SigLIP image + prompt + subtask slots) + 40 (10x4 chunk),
    # so include_prompt_subtask must stay True or _load_pretrained_critic raises on the shape
    # check. Note the prompt/subtask slots are fed the empty-string embedding at scoring time,
    # matching how pretrain_critic.py built this checkpoint's dataset.
    config.observation_mode = "image"
    config.image_encoder = "siglip"
    config.siglip_model_id = "google/siglip2-so400m-patch14-384"
    config.siglip_include_prompt_subtask = True

    # ── CoT sampling ─────────────────────────────────────────────────────────────
    # Must stay > 0 or every candidate decodes the same subtask and selection is a no-op.
    config.vla_cot_temperature = 0.5
    config.steervla.cot_temperature = 0.5

    # cast_relabel / steervla HL knobs are inherited unchanged from
    # steervla_cast_relabel_train_config (window 150, gemini-3.5-flash, store_hl_dataset,
    # store_good_chunks, hl_update_every=8, hl_update_batch_size=2).
    assert isinstance(config.cast_relabel, ml_collections.ConfigDict)
    assert config.cast_relabel.enabled and config.cast_relabel.store_hl_dataset

    return config
