"""Frozen base-policy control for the HL-DAgger route study.

This is the *first* half of each queued pair in ``run_hl_dagger_queue.sh``: roll the
starting policy out on a route with nothing training and nothing observing, so the
HL-DAgger run that follows has a same-route, same-settings control to be measured
against.

The starting policy is the ``no_ego_history`` v1 checkpoint at **step 6000** — the same
weights ``steervla_cast_relabel_config.py`` loads, so the control and the training run
begin from identical parameters.

Everything except the update/observer switches is inherited from
``steervla_cast_relabel_config.py`` via ``steervla_rollout_base.rollout_only_config``,
which is what keeps the two arms comparable: ``actions_per_model_query=5``,
``actions_per_cot=5``, ``action_horizon=10``, ``action_dim=4`` and the
``pi05_steervla_cot_simplified_reasoning_no_ego_history`` actor config all come along
unchanged. (Note this is *not* the ``actions_per_model_query=3`` setting used for the
220-route leaderboard sweep that picked the routes; that sweep chose the routes, it is
not the control.)

``gpu_rank`` below is only a default — ``run_carla.sh`` overwrites both
``training_gpu_rank`` and ``siglip_device`` from ``--train-gpu``, so the queue assigns
the card per job.

Launch (ports/display derived from ``carla_job.sh --job k``)::

    ./carla_job.sh start --job 200 --train-gpu 3 --render-adapter 3 \\
        --route hazard-at-side-lane-005 -- \\
        --agent-config impls/configs/steervla_hl_dagger_base_config.py \\
        --train-mode rl --critic-mode none --online-steps 8000 \\
        --save-buffer false --run-group bench2drive_hl_dagger_base
"""

from configs.steervla_rollout_base import rollout_only_config

# Overwritten per job by run_carla.sh --train-gpu; present so a direct main_carla.py
# invocation still lands on a real card.
DEFAULT_GPU = 0

CHECKPOINT = (
    "gs://cat-logs/pi05_steervla_cot_simplified_reasoning_no_ego_history"
    "/pi05_steervla_simplified_reasoning_no_ego_history_v1"
    "/pi05_steervla_simplified_reasoning_no_ego_history_v1_20260718_201640/6000"
)


def get_config():
    return rollout_only_config(
        gpu_rank=DEFAULT_GPU,
        checkpoint=CHECKPOINT,
        # openpi's pi05_steervla_cot_simplified_reasoning_no_ego_history sets
        # data.include_ego_history=False; the two must agree or the actor is fed the
        # wrong prompt-state width (2 vs 8) without raising.
        actor_config="pi05_steervla_cot_simplified_reasoning_no_ego_history",
        include_ego_history=False,
    )
