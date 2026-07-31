"""Job 2 — rollout-only baseline on ``enter-actor-flow-004``.

The **current** cast_relabel checkpoint (``pi05_steervla_cot_simplified_reasoning_no_ego_history``
v1 @ step 6000) with everything frozen: no updates, no CAST observer. This is the control
that jobs 3 and 4 are compared against on the same route.

GPU 1 for CARLA render + JAX inference + SigLIP. See ``steervla_rollout_base.py`` for
exactly what is switched off relative to ``steervla_cast_relabel_config.py``.

Launch (ports/display come from ``carla_job.sh --job 1``)::

    ./carla_job.sh start --job 1 --train-gpu 1 --render-adapter 1 \\
        --route enter-actor-flow-004 -- \\
        --agent-config impls/configs/steervla_rollout_job2_eaf004_no_ego_history.py \\
        --train-mode rl --critic-mode none --online-steps 8000 --save-buffer false
"""

from configs.steervla_rollout_base import rollout_only_config

JOB_GPU = 1


def get_config():
    return rollout_only_config(
        gpu_rank=JOB_GPU,
        checkpoint=(
            "gs://cat-logs/pi05_steervla_cot_simplified_reasoning_no_ego_history"
            "/pi05_steervla_simplified_reasoning_no_ego_history_v1"
            "/pi05_steervla_simplified_reasoning_no_ego_history_v1_20260718_201640/6000"
        ),
        actor_config="pi05_steervla_cot_simplified_reasoning_no_ego_history",
        # openpi's pi05_steervla_cot_simplified_reasoning_no_ego_history sets
        # data.include_ego_history=False.
        include_ego_history=False,
    )
