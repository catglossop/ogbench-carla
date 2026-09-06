"""Inference-only rollout of the ``ll_heavy`` UNNORMALIZED run trained with the matched image crop.

Checkpoint family::

    gs://cat-logs/pi05_steervla_cot_simplified_reasoning_ll_heavy
      /ll_heavy_unnormed_matchcrop/ll_heavy_unnormed_matchcrop_20260904_152800/<STEP>

mirrored locally (params + assets only; ``train_state`` is 37 GB and unused because
``rollout_only_config`` sets ``load_trainable_params=False``) at
``/raid/users/cglossop/steervla_pi_ckpts/ll_heavy_unnormed_matchcrop/<STEP>``.

What is different about this run, and why it needs its own eval config:

* **Matched image framing.** It is the first checkpoint trained after
  ``SIMLINGO_FRAMING_CROP`` was applied to ``simplified_reasoning_dataset`` at decode time,
  so *every* training corpus reached the model through the same box. The live CARLA frame
  must therefore be cropped the same way before the squeeze to 224x224 --
  ``carla_utils.downscale_rgb_for_policy`` does that. Verified 2026-09-05: a live-style
  uncropped frame through the CARLA path and through the openpi training path agree to
  0.30/255 (they differed by 32/255 before the crop was added).

* **No norm stats.** ``skip_norm_stats=True``, and the checkpoint ships no
  ``assets/*/norm_stats.json``, so ``resolve_openpi_norm_enabled`` correctly leaves OpenPI
  ``Unnormalize`` off and the fixed ``denormalize_actions`` scaling is the whole decode.
  This is the opposite of ``steervla_norm_ll_heavy_eval_config.py``.

* **``proprio_norm=False``.** ``rollout_only_config`` inherits ``True`` from
  ``steervla_cast_relabel_config``, but every ``pi05_steervla_cot_simplified_reasoning*``
  TrainConfig -- this one included -- sets ``proprio_norm=False``, i.e. raw m/s and degrees.
  Without norm stats nothing enforces the checkpoint's value (see
  ``vlas/steervla.py :: _check_proprio_layout``), so it is pinned here deliberately. Note
  this diverges from the older ckptcmp/base-policy numbers, which were all collected with
  the inherited ``True``.

* **Cadence** matches ``.run_carla/rollout_infer_apmq3_apc5.py`` so the resulting table is
  read against the same agent cadence as the earlier checkpoint comparisons.

``CHECKPOINT`` below is only a default; ``run_leaderboard.py --steervla-checkpoint`` sets
the step per run.
"""

from configs.steervla_rollout_base import rollout_only_config

# Overwritten per slot by run_leaderboard.py's generated wrapper; present so a direct
# main_carla.py invocation still lands on a real card.
DEFAULT_GPU = 5

CKPT_ROOT = "/raid/users/cglossop/steervla_pi_ckpts/ll_heavy_unnormed_matchcrop"
CHECKPOINT = f"{CKPT_ROOT}/6000"


def get_config():
    config = rollout_only_config(
        gpu_rank=DEFAULT_GPU,
        checkpoint=CHECKPOINT,
        actor_config="pi05_steervla_cot_simplified_reasoning_ll_heavy",
        # openpi's pi05_steervla_cot_simplified_reasoning_ll_heavy sets
        # data.include_ego_history=False -> prompt-state width 2.
        include_ego_history=False,
    )
    # See the module docstring: matches the checkpoint's own TrainConfig.
    config.steervla.proprio_norm = False
    # Same cadence as the earlier checkpoint comparisons.
    config.steervla.actions_per_model_query = 3
    config.steervla.actions_per_cot = 5
    if "debug_noise" in config:
        config.debug_noise = False
    return config
