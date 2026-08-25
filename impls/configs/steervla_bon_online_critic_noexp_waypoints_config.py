"""``get_config()`` for Best-of-N with a LIVE online critic, warm-started from a pretrained
no-language ("noexp_waypoints") critic checkpoint, driving the
``pi05_steervla_cot_simplified_reasoning_no_ego_history`` Pi0-CoT checkpoint.

Use with:
  ./run_carla.sh --agent-config impls/configs/steervla_bon_online_critic_noexp_waypoints_config.py \
    --train-mode rl --online-steps 50000 \
    --bon-online-critic true --bon-num-candidates 8 --bon-candidates-log-every 1 \
    --pretrained-critic /path/to/critic_pretrain_noexp_waypoints/run_20260701_233000/step_0012000.pkl \
    --route <ROUTE> ...

Differs from ``steervla_best_of_n_config.py`` in:
  * ``enable_updates_rl=True`` so the critic keeps training online via the normal
    ``update_with_vla()`` Bellman backup on collected rollout transitions --
    ``--bon_online_critic`` reads ``modules_critic`` live every step, not a frozen snapshot.
  * ``critic_pretrained_weights`` cleared: the warm start comes from ``--pretrained_critic``
    (loaded by ``main_carla.py`` *after* agent creation), not this config-time field, whose
    stale ``/raid/...`` default won't exist on most machines and would crash inside
    ``BestOfNAgent.create`` before the intended checkpoint ever loads.
  * ``steervla.actor_config`` / ``checkpoint`` point at the requested no-ego-history
    simplified-reasoning Pi0-CoT checkpoint (``include_ego_history=False`` to match).
  * ``best_of_n`` / ``steervla.action_decode_batch_size`` raised to 8 so both CoT sampling
    (already batched across candidates by ``sample_candidates``) and action-chunk decoding
    for the N candidates batch together instead of decoding in batches of 2.
"""

from configs.steervla_best_of_n_config import get_config as get_base_config


def get_config():
    config = get_base_config()

    # See module docstring: the base config's default points at a path that almost certainly
    # doesn't exist on this machine and would crash BestOfNAgent.create before
    # --pretrained_critic gets a chance to load the real warm-start checkpoint.
    config.critic_pretrained_weights = ""

    # Master switch + RL gate on so the critic actually trains online; nothing else.
    config.enable_updates = True
    config.enable_updates_rl = True
    config.enable_updates_bc = False
    config.enable_updates_bc_hl = False

    config.best_of_n = 8
    config.steervla.action_decode_batch_size = 8

    config.steervla.actor_config = "pi05_steervla_cot_simplified_reasoning_no_ego_history"
    config.steervla.checkpoint = (
        "gs://cat-logs/pi05_steervla_cot_simplified_reasoning_no_ego_history/"
        "pi05_steervla_simplified_reasoning_no_ego_history_v1/"
        "pi05_steervla_simplified_reasoning_no_ego_history_v1_20260718_201640/6000"
    )
    config.steervla.include_ego_history = False

    return config
