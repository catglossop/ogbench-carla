"""Inference-only probe run: does CARLA actually render on a card the sweep marked bad?

Same SteerVLA policy as ``steervla_cast_overfit_commentary_config.py`` (the 0823 commentary
checkpoint), stripped down to a pure rollout so the only thing under test is whether the CARLA
renderer survives on the target GPU:

  * ``load_trainable_params=False`` -- restore bare params, not the full OpenPI TrainState. Drops
    the optimizer + Adam mu/nu + backward buffers, so JAX peaks at a fraction of the ~83 GB the
    training runs needed and there is no HL gradient step competing for the card.
  * ``enable_updates=False`` -- no RL, BC, or HL gradients at all.
  * ``cast_relabel.enabled=False`` -- no window videos, no Gemini calls, no HL dataset writes.
    ``main_carla`` skips constructing the session entirely when this is false.
  * checkpointing stays off (inherited 0) -- nothing is training, so there is nothing to export.

Context (2026-08-31): GPUs 0 and 7 fail CARLA world setup with ``VK_ERROR_DEVICE_LOST``, and the
kernel log shows ``Xid 109 CTX SWITCH TIMEOUT`` on exactly those two cards (PCI 1b:00 and df:00)
going back to Aug 26 -- from CarlaUE4 *and* from other tenants' processes. Adapters 1-6 all pass.
This config exists to check whether a lighter, inference-only workload gets through on a card that
a bare renderer probe could not, or whether the Xid condition kills it regardless.

Launch (render on the card under test, JAX on a healthy one)::

    CUDA_VISIBLE_DEVICES=1 ./carla_job.sh start --job 17 --train-gpu 0 --render-adapter 0 \\
        --route signalized-junction-left-turn-001 -- \\
        --hl-gpu 0 --agent-config impls/configs/steervla_gpu0_inference_config.py \\
        --train-mode rl --critic-mode none --enable-updates false --online-steps 1000 \\
        -- --save_dir=/home/cglossop/exps

``--render-adapter`` is a physical nvidia-smi index (the mapping is identity -- verified against
vulkaninfo UUID/PCI order and against the driver's own Xid PCI addresses); ``--train-gpu`` is an
index into ``jax.devices("gpu")``, which the CUDA_VISIBLE_DEVICES mask collapses to 0.
"""

from configs.steervla_cast_overfit_commentary_config import get_config as get_overfit_config


def get_config():
    config = get_overfit_config()

    # Pure rollout: bare params, no train state, no gradients of any kind.
    config.steervla.load_trainable_params = False
    config.enable_updates = False
    config.enable_updates_rl = False
    config.enable_updates_bc = False
    config.enable_updates_bc_hl = False

    # No VLM coach: this run is a renderer health probe, not a labeling run. Skipping it also
    # avoids spending Gemini calls and keeps startup fast.
    config.cast_relabel.enabled = False
    config.cast_relabel.store_hl_dataset = False

    return config
