"""Job 4 — rollout-only eval of the base **simplified-reasoning** checkpoint on ``enter-actor-flow-004``.

``pi05_steervla_cot_simplified_reasoning`` @ step 8000, frozen: no updates, no CAST observer.
This is the older/plain simplified-reasoning model (no ego-history variant suffix), evaluated
on the same route as jobs 2 and 3.

``include_ego_history=False`` matches openpi's ``pi05_steervla_cot_simplified_reasoning``
(``data.include_ego_history=False``).

GPU 3 for CARLA render + JAX inference + SigLIP.

Launch (ports/display come from ``carla_job.sh --job 3``)::

    ./carla_job.sh start --job 3 --train-gpu 3 --render-adapter 3 \\
        --route enter-actor-flow-004 -- \\
        --agent-config impls/configs/steervla_rollout_job4_eaf004_simplified_reasoning.py \\
        --train-mode rl --critic-mode none --online-steps 8000 --save-buffer false
"""

from configs.steervla_rollout_base import rollout_only_config

JOB_GPU = 3


def get_config():
    return rollout_only_config(
        gpu_rank=JOB_GPU,
        checkpoint=(
            "gs://cat-logs/pi05_steervla_cot_simplified_reasoning"
            "/pi05_steervla_cot_simplified_reasoning"
            "/pi05_steervla_cot_simplified_reasoning_20260523_222304/8000"
        ),
        actor_config="pi05_steervla_cot_simplified_reasoning",
        include_ego_history=False,
    )
