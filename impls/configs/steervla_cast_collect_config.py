"""``get_config()`` for **offline CAST collection** — roll out, relabel, save; never train.

This is the data-collection half of the "collect-then-finetune" variant of CAST relabel. The
online variant (``steervla_cast_relabel_train_config.py``) interleaves rollout with
``SteerVLAActor.update_hl`` gradient steps on whatever relabeled chunks exist so far. Here we
decouple the two:

  1. **This config** rolls the frozen SteerVLA policy out for a large step budget (~20k env
     steps, i.e. ``--online-steps 20000``, spread over as many episodes as the route needs),
     runs the same CAST relabel pipeline (window video -> VLM GOOD/BAD review -> per-chunk
     causal credit -> corrected subtask + fresh reasoning), and writes every chunk to disk as a
     ``steervla_hl_dataset_format`` sample. **No gradients are taken at any point.**
  2. ``impls/vlas/cast_hl_to_rlds.py`` then converts that corpus into an RLDS/TFDS dataset in
     the SimLingo layout the OpenPI loader expects, and
  3. steervla-pi trains on it offline at a large batch size, as a normal
     ``pi05_steervla_cot_simplified_reasoning_no_ego_history``-style run.

Why a separate config rather than flags on the training one:

- ``steervla.load_trainable_params`` is **False** here. The trainable OpenPI ``TrainState``
  (params + Adam mu/nu + backward activations) is the VRAM hog; collection only needs the
  forward-only inference params, which frees the second GPU entirely and makes the rollout
  meaningfully faster.
- Every update switch is off, so a stray ``enable_updates_rl`` can't quietly start training
  mid-collection and shift the behavior policy underneath the corpus. A collection run must
  keep one fixed policy from the first step to the last, otherwise the dataset mixes samples
  from different policies.
- ``cast_relabel.hl_dataset_root`` is an absolute path shared across runs, so many
  routes/seeds/machines accumulate into one corpus. Each run still gets its own subdirectory
  (named after the run's ``exp_name``), so window tags cannot collide and the per-run layout
  stays exactly what the online ``update_hl`` reader globs.

Usage::

    # one collection run (repeat per route; --job k keeps ports/displays disjoint)
    ./run_carla.sh --agent-config impls/configs/steervla_cast_collect_config.py \\
        --route parking-cut-in-001 --online-steps 20000 \\
        --train-gpu 0 --render-adapter 1 --x-display-num 30

    # ...then, in a TF-equipped env, convert the whole corpus in one pass
    python impls/vlas/cast_hl_to_rlds.py \\
        --hl-root /raid/users/cglossop/cast_collect \\
        --out-dir /raid/datasets/steervla --dataset-name cast_relabel_hl_v1

``GEMINI_API_KEY`` must be exported — the relabel pass is the whole point of the run, and
without it every window review fails (non-fatally) and the corpus stays empty.
"""

from configs.steervla_cast_relabel_config import get_config as get_cast_relabel_config


def get_config():
    config = get_cast_relabel_config()

    # ── (1) Pure rollout: no gradients, anywhere, ever. ──────────────────────────────────
    # The master switch alone is enough (each per-kind switch is ANDed with it), but the
    # per-kind ones are set too so a later edit to the master can't silently enable training.
    config.enable_updates = False
    config.enable_updates_rl = False
    config.enable_updates_bc = False
    config.enable_updates_bc_hl = False
    # Never leave the rollout policy: with updates off this only controls the warmup print,
    # but a huge value also guards against any future path that gates on it.
    config.warmup_steps = 10**9

    # Inference-only SteerVLA: no OpenPI TrainState, no optimizer state, no HL gradient buffers.
    # This is what frees the HL GPU during collection.
    config.steervla.load_trainable_params = False
    config.steervla.hl_training_gpu_rank = -1
    # HL replay pools are a *training*-time mixture knob; collection reads no replay data.
    config.steervla.hl_replay_pools = []
    config.steervla.hl_replay_root = ""
    config.steervla.hl_checkpoint_dir = ""

    # ── (2) Write everything the offline converter needs. ────────────────────────────────
    config.cast_relabel.store_hl_dataset = True
    # Keep GOOD/unlabeled chunks too: offline training wants both the corrective (BAD ->
    # replacement subtask) and the reinforcing (GOOD -> original subtask) halves, and the
    # BAD/GOOD balance is chosen at *training* time by the RLDS mixture weights rather than
    # being baked into what we bothered to save.
    config.cast_relabel.store_good_chunks = True
    config.cast_relabel.hl_action_dim = config.steervla.action_dim
    # Per-step [speed_mps, yaw_deg] history stored with each sample. 4 matches the SimLingo RLDS
    # ``observation/ego_hist`` length, so one corpus can feed either the ego-history or the
    # no-ego-history OpenPI config without recollecting.
    config.cast_relabel.ego_history_len = 4
    # Shared corpus root: <root>/<exp_name>/<window tag>/{sample_*.npz, hl_samples.json}.
    # Point every collection run at the same root and convert them together.
    config.cast_relabel.hl_dataset_root = "/raid/users/cglossop/cast_collect"
    # Debug videos cost a wandb upload per window; keep them on for a collection run since the
    # annotated GOOD/BAD overlays are the only cheap way to audit label quality at scale.
    config.cast_relabel.debug = True
    config.cast_relabel.debug_task = False
    # Review one window per this many env steps. Windows are the unit of VLM cost: at 150 steps
    # a 20k-step budget is ~133 window reviews (~15 chunks each).
    config.cast_relabel.query_every_n_episode_steps = 150
    config.cast_relabel.query_on_episode_end = True

    return config
