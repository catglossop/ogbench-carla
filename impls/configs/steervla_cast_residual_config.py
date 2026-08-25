"""``get_config()`` for **CAST relabel + residual RL**.

Inherits :mod:`configs.steervla_cast_relabel_config` — the online CAST loop (window rollout ->
VLM GOOD/BAD review -> per-chunk causal credit -> corrected subtask + fresh reasoning) with its
HL batch size, cadence and replay mixture — and adds a residual SAC actor on top.

The added mechanism: the SteerVLA chunk is PID-decoded to ``[accel, steer]`` and the residual
actor perturbs those two numbers, so the executed control is

    clip(base + residual_action_scale * tanh_gaussian(actor(obs, base)), -1, 1)

The residual trains its own DSRL TD critic from env reward.

Deliberately inherits from the *observer* CAST config rather than
``steervla_cast_relabel_train_config``: that variant throttles the HL update to batch 2 / every 8
/ 1 step for VRAM, which is not the reference we want here. The HL knobs below therefore stay at
the CAST-relabel values — batch 64, every 5 update_with_vla calls, 2 gradient steps — along with
the replay mixture (online 0.7 with an 80/20 BAD/GOOD bias and a 50/50 direct/precursor split
inside the corrective share; simlingo 0.2; simplified_reasoning 0.1).

**Both the residual actor and the CoT backbone learn**: the residual from env reward via its own
TD critic, and the VLM backbone from the CAST-relabeled subtasks/reasoning via ``update_hl``.
The plain BC / DAgger path stays off.

No checkpoints are written: ``hl_checkpoint_every_steps = 0``.

Requires on the command line::

    --train-mode sac_residual     # else run_carla.sh overwrites online_training_mode

Usage::

    ./carla_job.sh start --job 140 --train-gpu 1 --hl-gpu 1 --render-adapter 1 \\
      --route opposite-vehicle-running-red-light-004 -- \\
      --agent-config impls/configs/steervla_cast_residual_config.py \\
      --train-mode sac_residual --online-steps 20000 \\
      --run-group CastResidual -- --save_dir=/raid/users/cglossop/exps

``GEMINI_API_KEY`` must be exported — the CAST relabel pass produces the online HL pool that
``update_hl`` consumes, so without it the backbone has nothing to train on.
"""

from configs.steervla_cast_relabel_config import get_config as get_cast_relabel_config


def get_config():
    config = get_cast_relabel_config()

    # ── residual RL on the executed control ──────────────────────────────────────────────
    config.online_training_mode = "sac_residual"
    # Perturb the PID-decoded [accel, steer] rather than the raw 40-D chunk, so the replay
    # buffer and the residual critic both see the bounded 2-D control. Perturbing the chunk
    # directly would mean regressing a residual in physical DELTA_XY meters/degrees.
    config.residual_action_space = "accel_steer"
    config.residual_actor_hidden_dims = (256, 256)
    # ``residual_action_scale`` is typed as a float upstream and ml_collections refuses a tuple
    # override on a typed field, so drop it first (same dance as pi0_residual_sac_config).
    # 0.3 per dim: the residual can move accel or steer by at most 0.3 of full range before the
    # composed action is clipped, so the base policy stays in charge.
    del config.residual_action_scale
    config.residual_action_scale = (0.3, 0.3)
    config.residual_action_clip = 1.0
    config.residual_alpha = 0.1
    config.residual_lr = 3e-4
    config.residual_log_std_min = -5.0
    config.residual_log_std_max = 2.0
    config.residual_layer_norm = False
    # Execute the base chunk unmodified for this many env steps, so the residual's critic sees
    # on-distribution base actions before the residual starts steering.
    config.residual_warmup_steps = 500

    # Transitions per residual SAC gradient step. 32 upstream; matched to the HL batch size.
    config.batch_size = 64

    # ── updates: residual RL + CAST HL ───────────────────────────────────────────────────
    config.enable_updates = True
    config.enable_updates_rl = True
    # Plain BC / DAgger imitation path stays off; the two things learning here are the residual
    # actor (env reward, TD) and the CoT backbone (CAST-relabeled subtasks/reasoning).
    config.enable_updates_bc = False
    # HL (VLM-backbone) update ON, at the CAST-relabel batch/cadence/mixture inherited above.
    config.enable_updates_bc_hl = True

    # ── no checkpoints ───────────────────────────────────────────────────────────────────
    # 0 disables the periodic export *and* the one at exit.
    config.steervla.hl_checkpoint_every_steps = 0
    config.steervla.hl_checkpoint_dir = ""
    config.steervla.hl_checkpoint_keep_last = 0

    return config
