"""Shared base for the rollout-only (no-gradient) SteerVLA evaluation configs.

Jobs 2-4 exist to measure **raw checkpoint performance** on a route: roll the frozen
SteerVLA policy out and log what happens, with nothing training and nothing observing.
Relative to ``steervla_cast_relabel_config.py`` that means turning three things off:

1. **All gradient updates.** ``enable_updates=False`` is the master switch; the per-kind
   switches are set too so the intent survives a CLI override of any single one.
2. **The CAST relabel observer.** No Gemini window reviews, no credit assignment, no
   annotated debug videos, no HL dataset on disk. This is what makes the run cheap and
   keeps it from perturbing rollout timing.
3. **The trainable OpenPI restore.** ``load_trainable_params=False`` restores bare params
   instead of a full ``TrainState``; the optimizer + opt_state are the bulk of the VRAM and
   nothing would consume them here.

Each job then pins itself to one GPU (CARLA renderer, JAX inference and SigLIP all share
it) and picks its own checkpoint / OpenPI actor config.

``include_ego_history`` is **not** defaulted here — it is a property of the checkpoint and
each job must state it. It sets the actor's prompt-state width (8 with history, 2 without;
``vlas/steervla.py :: steervla_prompt_state_dim``), so a mismatch silently feeds the model
the wrong state layout rather than raising.
"""

from configs.steervla_cast_relabel_config import get_config as get_cast_relabel_config


def rollout_only_config(*, gpu_rank: int, checkpoint: str, actor_config: str, include_ego_history: bool):
    """Return a frozen-policy rollout config pinned to ``gpu_rank``."""
    config = get_cast_relabel_config()

    # ── No gradient updates of any kind ──────────────────────────────────────────
    config.enable_updates = False
    config.enable_updates_rl = False
    config.enable_updates_bc = False
    config.enable_updates_bc_hl = False
    # run_carla.sh defaults --train-mode to "dagger" and unconditionally writes
    # online_training_mode into the temp agent config, so pass --train-mode rl on the
    # command line as well. This keeps a direct `main_carla.py` invocation consistent.
    config.online_training_mode = "rl"

    # ── CAST relabel observer off ────────────────────────────────────────────────
    config.cast_relabel.enabled = False
    config.cast_relabel.debug = False
    config.cast_relabel.save_artifacts = False
    config.cast_relabel.store_hl_dataset = False

    # ── Inference-only SteerVLA ──────────────────────────────────────────────────
    config.steervla.load_trainable_params = False
    config.steervla.hl_training_gpu_rank = -1
    # Only read by update_hl, which never runs here; emptied so a stray call can't
    # start scanning the pool directories.
    config.steervla.hl_replay_pools = []

    config.steervla.checkpoint = checkpoint
    config.steervla.actor_config = actor_config
    config.steervla.include_ego_history = include_ego_history

    # ── One GPU per job ──────────────────────────────────────────────────────────
    # training_gpu_rank indexes jax.devices("gpu"); siglip_device is a torch device string.
    # Both are physical GPU order because run_carla.sh forces use_cuda_visible_devices=False.
    # run_carla.sh overwrites training_gpu_rank from --train-gpu but never touches
    # siglip_device, so this assignment is what keeps concurrent jobs off each other's card.
    config.training_gpu_rank = gpu_rank
    config.siglip_device = f"cuda:{gpu_rank}"

    return config
