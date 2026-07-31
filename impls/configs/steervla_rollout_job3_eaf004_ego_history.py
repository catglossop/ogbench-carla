"""Job 3 — rollout-only eval of the **ego-history** checkpoint on ``enter-actor-flow-004``.

``pi05_steervla_cot_simplified_reasoning_ego_history`` v1 @ step 10000, frozen: no updates,
no CAST observer. Same route as jobs 2 and 4 so the three checkpoints are directly comparable.

Note ``include_ego_history=True``: openpi's ``pi05_steervla_cot_simplified_reasoning_ego_history``
sets ``data.include_ego_history=True``, which widens the actor's prompt state from 2 to 8
(``vlas/steervla.py :: steervla_prompt_state_dim``). Leaving it at the base config's False
would feed this checkpoint a truncated state without erroring.

GPU 2 for CARLA render + JAX inference + SigLIP.

Launch (ports/display come from ``carla_job.sh --job 2``)::

    ./carla_job.sh start --job 2 --train-gpu 2 --render-adapter 2 \\
        --route enter-actor-flow-004 -- \\
        --agent-config impls/configs/steervla_rollout_job3_eaf004_ego_history.py \\
        --train-mode rl --critic-mode none --online-steps 8000 --save-buffer false
"""

from configs.steervla_rollout_base import rollout_only_config

JOB_GPU = 2


def get_config():
    return rollout_only_config(
        gpu_rank=JOB_GPU,
        checkpoint=(
            "gs://cat-logs/pi05_steervla_cot_simplified_reasoning_ego_history"
            "/pi05_steervla_simplified_reasoning_ego_history_v1"
            "/pi05_steervla_simplified_reasoning_ego_history_v1_20260717_183503/10000"
        ),
        actor_config="pi05_steervla_cot_simplified_reasoning_ego_history",
        include_ego_history=True,
    )
