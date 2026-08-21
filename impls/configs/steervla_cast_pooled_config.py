"""``get_config()`` for a **pooled** CAST-relabel run: many routes, one policy.

Same per-worker pipeline as ``steervla_cast_relabel_config.py`` (window rollout -> VLM review ->
per-chunk GOOD/BAD credit -> corrected subtask + fresh reasoning), but the training is centralized:

  * N rollout **workers** (one CARLA job per route) write their HL samples into one shared pool and
    take **no** gradient steps. They are inference-only, so each is far lighter than a solo
    cast_relabel worker (no optimizer state, no backward activations).
  * One **trainer** (``impls/train_hl_pooled.py``) owns the only trainable OpenPI ``TrainState``.
    Every ``cast_pool.round_new_samples`` fresh samples it takes one large ``update_hl`` round over
    the pooled corpus -- corrections from every route in the same batch -- then exports and
    publishes a params-only checkpoint.
  * Workers hot-reload the newest published version mid-episode, so all routes converge back onto
    one policy after every round without a CARLA restart.

Nothing blocks: workers keep driving while the trainer trains, and a CARLA crash on one route costs
that route's share of the next round rather than stalling the pool. See ``impls/cast_pool.py`` for
the filesystem protocol and ``run_cast_pool.sh`` for the launcher.

Launch with ``./run_cast_pool.sh --routes a,b,c --worker-gpus 2,3,4 --trainer-gpu 7``; the launcher
fills in ``pool_root`` / ``checkpoint_dir`` / ``role`` per process, so they are left empty here.

**GPU memory.** JAX preallocates ~75% of a card on first use, so an unfractioned worker reserves
~110 GB of an H200 while actually needing far less (an inference-only actor plus a ~7 GB CarlaUE4
renderer) -- exactly one worker fits per GPU, which is what caps a pool at "one route per free
card". ``run_cast_pool.sh --worker-mem-fraction`` sets ``XLA_PYTHON_CLIENT_MEM_FRACTION`` per worker
so several routes can share a card; that flag, not anything in this file, is what makes a 5-6 route
pool fit on 2-3 GPUs. The trainer keeps a whole GPU (``--trainer-mem-fraction``, default 0.90) --
its pooled ``update_hl`` at ``round_batch_size`` is the largest allocation in the run.
"""

import ml_collections

from configs.steervla_cast_relabel_config import get_config as get_cast_relabel_config


def get_config():
    config = get_cast_relabel_config()

    # ── worker training switches ──────────────────────────────────────────────────────
    # A pooled worker is a pure sample producer. All gradient work happens in the trainer process,
    # so that every route trains ONE policy instead of N drifting copies. (The trainer overrides
    # load_trainable_params back to True for itself.)
    config.enable_updates = False
    config.enable_updates_rl = False
    config.enable_updates_bc = False
    config.enable_updates_bc_hl = False
    config.steervla.load_trainable_params = False
    # Irrelevant for an inference-only worker, and actively wrong to leave on: no worker should be
    # exporting policy checkpoints, only the trainer publishes versions.
    config.steervla.hl_checkpoint_every_steps = 0

    # Send this worker's HL samples to the shared pool instead of its own run dir. The session
    # namespaces them as ``<hl_dataset_root>/<run_tag>/<window>/``, which is exactly the layout the
    # trainer's pooled scan expects. Declared (empty) here so run_cast_pool.sh can set it per run.
    config.cast_relabel.hl_dataset_root = ""

    # ── rollout knobs (workers) ───────────────────────────────────────────────────────
    # Pinned explicitly rather than inherited. With actions_per_model_query=5 a chunk is held for
    # five 20 Hz ticks, and interpolate_waypoints re-zeroes arc length at the plan's first point --
    # so without re-anchoring the decoder assumes the ego never moved and re-issues the same steer
    # for the whole hold, leaving the lateral loop open (see commit 69fd7c3). The actor already
    # defaults this to True, but a pooled run holds chunks on every one of its workers, so the
    # dependency is stated here instead of relying on that default.
    # Use the v2 coaching prompts (coaches/cast_relabel_v2.py + coaches/vlm_feedback_v2.py).
    config.cast_relabel.prompt_version = 2

    config.steervla.reanchor_cached_chunk = True
    # Chunk hold / CoT reuse. Decoupled deliberately: the chunk is re-queried every 3 env steps
    # while the CoT reasoning+subtask is reused for 5, so the policy re-plans its trajectory more
    # often than it re-thinks its intent. Re-anchoring above is what makes a >1 hold safe.
    config.steervla.actions_per_model_query = 3
    config.steervla.actions_per_cot = 5

    # ── trainer-side HL knobs ─────────────────────────────────────────────────────────
    # One big round instead of the solo config's small, frequent updates. The trainer has a whole
    # GPU to itself (no inference model sharing it), so the batch can go above the ~64 that fits
    # alongside a rollout -- but not as far as it looks.
    #
    # MEASURED 2026-08-19 on an H200 (143 GB) at mem_fraction 0.90: batch 128 needs ~93 GiB of live
    # activations (XLA rematerialization could not get under its 82 GiB budget) plus a ~55 GB
    # transient. It survived a replay-only smoke batch and then OOM'd on the first REAL round, whose
    # batch was 90 online + 38 replay. So a replay-only trial does not clear batch 128 -- keep 64,
    # which halves the activation footprint, and raise only with a real online-heavy batch to test.
    config.steervla.hl_update_batch_size = 64
    # NOTE: inert in a pooled run. ``train_hl_pooled.py`` forces ``hl_update_every = 1`` because the
    # trainer is driven by ROUNDS, not by a per-call throttle -- ``cast_pool.round_new_samples`` is
    # the pooled equivalent of this knob. Left at the requested 8 so the intent is recorded; to
    # actually train 8x less often, multiply ``round_new_samples`` instead.
    config.steervla.hl_update_every = 8
    # 32 steps per round over-fits a round's worth of fresh corrections; 2 keeps each round a light
    # nudge, closer to the solo cast_relabel runs (which used 1).
    config.steervla.hl_update_num_steps = 2
    # Sliding window over policy versions: drop online samples produced more than N rounds ago. They
    # were corrections for a backbone that has since been replaced, and keeping them indefinitely is
    # what drives the sample-reuse blowup seen in the solo runs. Offline replay pools are exempt
    # (version -1) -- they are pretraining data, not on-policy corrections.
    config.steervla.hl_keep_last_rounds = 3
    # With ~5-6 workers each producing ~15 samples per window, a round's worth of fresh data lands
    # much faster than in a solo run, so the pool is far less starved; keep the start gate low.
    config.steervla.hl_min_online_samples = 20

    # ── pooled-run coordination ───────────────────────────────────────────────────────
    config.cast_pool = ml_collections.ConfigDict(
        dict(
            enabled=True,
            # "worker" (rollout, set by run_cast_pool.sh for each route) or "trainer".
            role="worker",
            # Shared HL sample pool and published-checkpoint dir. Both are filled in per run by
            # run_cast_pool.sh; the worker's own samples land under <pool_root>/<run_tag>/ via
            # cast_relabel.hl_dataset_root, which already namespaces by run tag.
            pool_root="",
            checkpoint_dir="",
            # Trainer: fire a round once this many NEW samples have landed across all workers.
            # ~15 samples per window per worker, so 90 is roughly one window from each of 6 routes.
            round_new_samples=90,
            # Trainer: size of the pooled update. These override the steervla.hl_update_* values
            # above at call time, so a round is a single deliberate unit of training.
            round_batch_size=64,  # see the MEASURED note above before raising this.
            round_num_steps=2,
            # Trainer: seconds between pool polls while waiting for a round to fill.
            poll_interval_sec=15.0,
            # Trainer: published versions to retain (~10 GB each). Must stay comfortably above 1 --
            # pruning a version a worker is mid-load would fail that worker's reload.
            checkpoint_keep_last=5,
            # Trainer: stop after N rounds (0 = run until killed).
            max_rounds=0,
            # Worker: env steps between checkpoint-dir polls. A poll is a couple of stat calls, so
            # this can be tight; the real cost is the reload itself (tens of seconds, off-tick).
            worker_check_every_steps=50,
        )
    )

    return config
