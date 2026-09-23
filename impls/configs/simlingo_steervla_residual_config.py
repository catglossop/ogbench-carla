"""``get_config()`` for residual SAC on a frozen SimLingo SteerVLA base (``vlas/simlingo_steervla.py``).

Same residual stack as ``steervla_residual_config.py`` (``agent_name="sac_residual"`` ->
``run_online_residual``) with the OpenPI Pi0-CoT base swapped for the SimLingo HL planner -> LL waypoint
policy via ``steervla.vla``. Both SimLingo models stay frozen; only the residual SAC agent trains.

Supported: ``base_only=True``, or ``state_encoder="siglip_pool"`` with ``expo=False``, in either
``residual_action_space`` (``accel_steer`` / ``waypoint_chunk``). ``pi_prefix`` / ``pi_prefix_groups`` /
``rl_token`` and EXPO best-of-N read Pi0 internals and are rejected at startup.

    ./run_carla.sh --agent-config impls/configs/simlingo_steervla_residual_config.py \\
      --route <route> --online-steps 10000 --train-gpu 0 --sim-gpu 0

A CAST-relabel-trained HL (``<run>/checkpoints/<step>``) deploys with ``--steervla-checkpoint``, which
sets ``steervla.hl_checkpoint`` for this VLA.
"""

from configs.simlingo_steervla_cast_relabel_train_config import apply_simlingo_steervla
from configs.steervla_residual_config import get_config as get_steervla_residual_config


def get_config():
    config = get_steervla_residual_config()
    config.state_encoder = "siglip_pool"
    config.residual_warmup_steps = 1000
    config.residual_ramp_steps = 1500
    config.residual_accel_scale = 0.1
    config.residual_steer_scale = 0.1
    config.residual_bc_beta = 1.0
    config.residual_bc_normalize = False
    config.expo = False
    config.best_of_n = 1
    config.otf_td_backup = False

    s = apply_simlingo_steervla(config.steervla)
    # HL re-plan cadence of the SimLingo CAST runs (HL every 6 env steps with the LL every 3), so a
    # CAST-trained HL is deployed the way it was trained.
    s.actions_per_cot = 5
    s.actions_per_model_query = 3
    s.proprio_norm = False
    return config
