"""Inference-only rollout of the norm-stats checkpoint ``*_norm_ll_heavy``.

This is the first CARLA rollout of a SteerVLA checkpoint trained with
``skip_norm_stats=False``. Unlike every earlier checkpoint, it predicts in the
**quantile-normalized** action space, so inference must run OpenPI ``Unnormalize`` before
the fixed ``denormalize_actions`` scaling. That is detected automatically from
``<checkpoint>/assets/steervla_simlingo_cot_normed/norm_stats.json`` -- see
``vlas/steervla.py :: resolve_openpi_norm_enabled``. Nothing here switches it on; the
checkpoint does.

Two settings below are properties of *this* checkpoint's OpenPI TrainConfig and must match
it, or the model is fed a state it never saw:

* ``include_ego_history=False`` -> prompt-state width 2 (not 8).
* ``proprio_norm=False`` -> raw m/s and degrees, which is what the checkpoint's state norm
  stats were computed on (q99 = [20.24, 179.93]). The actor enforces this whenever norm
  stats are active and warns if the config disagrees; it is set explicitly so the intent is
  visible and the warning stays quiet.

Everything else -- rollout-only gating, CAST observer off, bare-params restore, one GPU per
job -- comes from ``steervla_rollout_base.rollout_only_config``, so this run is directly
comparable to the ``steervla_hl_dagger_base_config.py`` controls.

Launch (ports/display derived from ``carla_job.sh --job k``)::

    ./carla_job.sh start --job 400 --train-gpu 4 --render-adapter 4 \\
        --route signalized-junction-left-turn-001 -- \\
        --agent-config impls/configs/steervla_norm_ll_heavy_eval_config.py \\
        --train-mode rl --critic-mode none --eval-only true \\
        --online-steps 4000 --save-buffer false \\
        --run-group norm_ll_heavy_eval
"""

from configs.steervla_rollout_base import rollout_only_config

# Overwritten per job by run_carla.sh --train-gpu; present so a direct main_carla.py
# invocation still lands on a real card. Also sets siglip_device, which run_carla.sh
# does *not* override.
DEFAULT_GPU = 4

# ll_heavy_bs1152, step 6000. Step 4000 is also available under the same run directory and
# carries the same norm stats.
CHECKPOINT = (
    "gs://cat-logs/pi05_steervla_cot_simplified_reasoning_norm_ll_heavy"
    "/ll_heavy_bs1152/ll_heavy_bs1152_20260826_185850/6000"
)


def get_config():
    config = rollout_only_config(
        gpu_rank=DEFAULT_GPU,
        checkpoint=CHECKPOINT,
        actor_config="pi05_steervla_cot_simplified_reasoning_norm_ll_heavy",
        include_ego_history=False,
    )
    # See the module docstring: matches the checkpoint's data config, which is what the
    # state norm stats were computed against.
    config.steervla.proprio_norm = False
    return config
