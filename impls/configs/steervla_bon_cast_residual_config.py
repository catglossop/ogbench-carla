"""``get_config()`` for **static-critic Best-of-N + residual RL + CAST-relabel HL training**.

Inherits :mod:`configs.steervla_bon_cast_config` and adds *only* the residual actor, so this
config and its base differ by exactly the residual and nothing else — same selection path, same
static critic, same CAST/HL pipeline, same CoT temperature.

The added mechanism: the best-of-N winner is PID-decoded to ``[accel, steer]`` and a SAC
residual actor perturbs those two numbers, so the executed action is
``clip(base + residual * residual_action_scale)``. The residual trains its own DSRL TD critic —
a **different** critic from the frozen selection critic, which no gradient ever reaches
(``main_carla._bon_q_fn`` is a jitted closure over params loaded from ``--bon_critic_ckpt``).

Selection and correction only compose because ``main_carla._bon_selected_to_action`` hands the
selected chunk to ``sample_actions_sac_residual(base_action=...)`` instead of executing it;
before that the best-of-N branches returned first and the residual actor never ran.

``enable_updates_rl`` is flipped back **True** here: it drives the residual SAC update. It does
not affect selection, which reads the frozen ``_bon_q_fn`` either way.

Requires on the command line:
  ``--train-mode sac_residual``  or run_carla.sh overwrites online_training_mode with "rl"
  ``--bon_critic_ckpt <path>``   the static selection critic
  ``--bon_num_candidates N``     candidates per step (main_carla reads the flag, not a config key)
"""

from configs.steervla_bon_cast_config import get_config as get_bon_cast_config


def get_config():
    config = get_bon_cast_config()

    # ── residual RL on the selected action ───────────────────────────────────────
    config.online_training_mode = "sac_residual"
    # Residual perturbs the PID-decoded [accel, steer] rather than the raw 40-D chunk, so the
    # replay buffer and the residual's critic both see the bounded 2-D control. This is also
    # what _bon_selected_to_action requires: it only routes a selected chunk into the residual
    # when residual_action_space == "accel_steer".
    config.residual_action_space = "accel_steer"
    config.residual_actor_hidden_dims = (256, 256)
    # The base config types residual_action_scale as a float; ml_collections refuses a tuple
    # override on a typed field, so drop it first (same dance as pi0_residual_sac_config).
    del config.residual_action_scale
    config.residual_action_scale = (0.6, 0.6)
    config.residual_action_clip = 1.0
    config.residual_alpha = 0.1
    config.residual_lr = 3e-4
    config.residual_log_std_min = -5.0
    config.residual_log_std_max = 2.0
    config.residual_layer_norm = False
    # Execute the selected candidate unmodified for this many steps, so the residual's critic
    # sees on-distribution base actions before the residual starts steering.
    config.residual_warmup_steps = 500

    # The residual SAC update is the only thing this turns on; the selection critic stays
    # frozen regardless.
    config.enable_updates_rl = True

    return config
