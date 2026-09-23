"""``get_config()`` for CAST-relabel HL training with the MIXED stack: the original SimLingo
InternVL2 high level driving a pi05 low level (``vlas/mixed_steervla.py``).

Same run as ``simlingo_steervla_cast_relabel_train_config.py`` -- DSRL rollout with RL/BC off, CAST
relabel windows writing HL samples, ``update_hl`` on the same cadence, the same HL optimizer recipe
and replay mixture -- with SimLingo's meta-action waypoint LL replaced by the pi05 no-ego-history
action expert. The HL therefore learns against the behaviour of the policy that will be evaluated.

The HL is the *original* SimLingo checkpoint (not a CAST-finetuned export): this sweep is what
produces the finetuned ones. Its torch optimizer lives in the InternVL2 worker process; pi05 is
frozen and never loaded trainable.

    ./run_carla.sh --agent-config impls/configs/mixed_steervla_cast_relabel_train_config.py \\
      --train-gpu 0 --hl-gpu 1 --render-adapter 0 \\
      --steervla-checkpoint <pi05 no_ego_history 6000> \\
      --actor-config pi05_steervla_cot_simplified_reasoning_no_ego_history ...

Machine-specific paths come from the environment (see the constants below) so the committed file
stays valid on any box. Needs ``uv sync --extra all-gpu --extra simlingo``.
"""

import os
from pathlib import Path

from configs.steervla_cast_relabel_train_config import get_config as get_cast_relabel_train_config

_REPO_ROOT = Path(__file__).resolve().parents[2]
# HL update cadence of the b2d/f2d sweeps: 10 gradient steps every 200 env steps.
HL_UPDATE_EVERY_ENV_STEPS = 200
HL_UPDATE_NUM_STEPS = 10
# Low-level (waypoint) query cadence in env steps; chunks are held and re-anchored in between.
ACTIONS_PER_MODEL_QUERY = 3

# The original SimLingo HL (seed1 bellman, epoch 19) -- the same weights the mixed BoN eval loads.
HL_CHECKPOINT = os.environ.get(
    "MIXED_HL_CHECKPOINT",
    "/data/local/cglossop/2026_05_24_06_52_33_simlingo_seed1_bellman/checkpoints/epoch=019.ckpt",
)
SIMLINGO_SOURCE_ROOT = os.environ.get("SIMLINGO_SOURCE_ROOT", "/home/celinet/simlingo-steervla")
# The worker only needs torch + the simlingo extra, which the repo venv has; it runs
# internvl2_hl_worker.py, whose own directory supplies ``simlingo_model``.
HL_PYTHON = os.environ.get("MIXED_HL_PYTHON", str(_REPO_ROOT / ".venv" / "bin" / "python"))
HL_REPLAY_ROOT = os.environ.get("HL_REPLAY_ROOT", "/data/local/cglossop/simlingo_hl_pools")
# Pool directory name under the root. The dgx6 copy bundles its frames as .npz beside the manifest
# ("_img") instead of pointing at the SimLingo database, so the name differs per machine.
HL_REPLAY_POOL = os.environ.get("HL_REPLAY_POOL", "simlingo_hl_simplified")


def apply_mixed_steervla(s):
    """Point a ``config.steervla`` block at the InternVL2 HL -> pi05 LL policy."""
    # Selects MixedSteerVLAActor inside the ordinary "steervla" VLA branch (vlas/steervla.py).
    s.hl_provider = "internvl2"
    s.hl_checkpoint = HL_CHECKPOINT
    s.simlingo_source_root = SIMLINGO_SOURCE_ROOT
    s.hl_python = HL_PYTHON
    # Native 1024x512 rgb_front: what InternVL2 is fed at rollout time (mixed_steervla reads
    # raw['image_viz']), so it must also be the frame cast_relabel stores in HL samples. pi05 keeps
    # its own raw['image'] regardless.
    s.image_key = "image_viz"
    # HL CoT decoding: 0 = greedy (run_carla.sh --cot-temperature overrides; the sweep uses 0.1).
    s.cot_temperature = 0.0
    s.actions_per_model_query = ACTIONS_PER_MODEL_QUERY
    return s


def get_config():
    config = get_cast_relabel_train_config()
    s = apply_mixed_steervla(config.steervla)

    # HL (torch) update. hl_training_gpu_rank (inherited, run_carla.sh --hl-gpu) places the HL
    # worker on its own GPU. ``hl_update_every`` counts update_with_vla calls, not env steps.
    update_interval = max(1, int(config.get("update_interval", 1)))
    updates_per_step = max(1, int(config.get("updates_per_step", 1)))
    s.hl_update_every = max(1, round(HL_UPDATE_EVERY_ENV_STEPS * updates_per_step / update_interval))
    s.hl_update_num_steps = HL_UPDATE_NUM_STEPS
    s.hl_lr = 1e-5
    s.hl_weight_decay = 0.1
    s.hl_grad_clip = 1.0
    # 64 rows, half online / half replay, 90% corrective within the online half (2/3 precursor).
    # Peak memory is set by hl_micro_batch_size, not the batch size.
    s.hl_update_batch_size = 64
    s.hl_micro_batch_size = 8
    s.hl_freeze_regexes = None  # OpenPI-only; the torch HL trains the checkpoint's own trainable set.

    # Replay: SimLingo HL training frames (build once with extract_simlingo_hl_replay.py). A missing
    # pool only warns and falls back to online-only, which silently changes the recipe.
    s.hl_replay_root = HL_REPLAY_ROOT
    s.hl_replay_pools = [dict(name=HL_REPLAY_POOL, weight=0.5)]
    s.hl_online_weight = 0.5
    s.hl_online_bad_fraction = 0.90
    s.hl_online_precursor_fraction = 60.0 / 90.0
    s.use_adaptive_sampling = False

    # The HL (actions_per_cot, inherited 5) is only re-planned on an LL query step once it is that
    # old, so with the LL every 3 steps the HL refreshes every 6.
    return config
