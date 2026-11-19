"""YAY-Robot on the mixed policy: Gemini foresight corrections, online InternVL2 HL training, pi05 LL frozen.

The training recipe is Surya's SimLingo YAY arm (``simlingo_steervla_yay_robot_train_config.py``: correction
cadence, HL update cadence / batch / lr / replay mix, 2k-step HL exports); the actor is the mixed policy
of the HARP evals (``_mixed_policy.py``). ``steervla.hl_trainable`` makes ``MixedSteerVLAActor`` run
``SimLingoSteerVLAActor``'s HL update inside its InternVL2 worker; pi05 stays inference-only, so
``load_trainable_params`` is False. Exports (``<save_dir>/checkpoints/<step>``) are SimLingo HL checkpoints,
deployable as ``steervla.hl_checkpoint`` of any mixed-policy eval.
"""
from configs._mixed_policy import apply_mixed_policy
from configs.simlingo_steervla_yay_robot_train_config import get_config as get_simlingo_yay_config
from configs.steervla_mixed_eval_config import get_config as get_mixed_eval_config

# HL-training / export knobs carried over from the SimLingo YAY arm.
_HL_KEYS = (
    "hl_update_every", "hl_update_num_steps", "hl_update_batch_size", "hl_micro_batch_size", "hl_lr",
    "hl_weight_decay", "hl_grad_clip", "hl_replay_root", "hl_replay_pools", "hl_online_weight",
    "hl_online_bad_fraction", "hl_online_precursor_fraction", "hl_online_backfill_from_replay",
    "use_adaptive_sampling", "adaptive_sampling_weights", "hl_min_online_samples", "hl_keep_last_rounds",
    "hl_kl_coef", "hl_log_kl", "hl_checkpoint_every_steps", "hl_checkpoint_dir", "hl_checkpoint_keep_last",
)


def get_config():
    config = get_simlingo_yay_config()
    simlingo = config.steervla
    s = get_mixed_eval_config().steervla  # pi05 actor block of the mixed policy
    s.unlock() if hasattr(s, "unlock") else None
    for key in _HL_KEYS:
        if key in simlingo:
            s[key] = simlingo[key]
    apply_mixed_policy(s)
    s.hl_trainable = True
    s.load_trainable_params = False
    s.hl_freeze_regexes = None  # OpenPI-only
    # main_carla stores this frame in the YAY HL samples; InternVL2 trains on the native camera.
    s.image_key = "image_viz"
    s.hl_training_gpu_rank = -1  # the worker trains on the policy GPU
    config.steervla = s
    # Correction cadence follows the HL period (5/3 -> a CoT every 6 env steps -> every 4th CoT ~ 24 steps),
    # same as the SimLingo arm; recompute in case the cadence above differs.
    ll_period = max(1, int(s.actions_per_model_query))
    cot_period = -(-max(1, int(s.actions_per_cot)) // ll_period) * ll_period
    config.yay_robot.query_every_n_cot_queries = max(1, round(24 / cot_period))
    return config
