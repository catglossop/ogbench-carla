"""``get_config()`` for DSRL + SteerVLA on CARLA with the YAY-Robot foresight corrector.

Same DSRL/SteerVLA stack as ``steervla_cast_relabel_config.py``, but the ``cast_relabel``
observer is swapped for a ``yay_robot`` block driving
:class:`coaches.yay_robot.OnlineYayRobotSession`. The two are alternatives, not layers:
``main_carla`` raises if both are enabled, because only one can own
``steervla.hl_dataset_dir`` and therefore the high-level update.

The difference from CAST is *when* the VLM speaks, and whether the policy hears it:

  CAST (hindsight)   drive a window -> review the video -> blame chunks -> relabel them.
                     The correction is only ever a training target.
  YAY  (foresight)   every CoT query -> show the VLM this frame and the CoT the model just
                     produced -> KEEP or CORRECT -> the corrected subtask/reasoning is
                     re-tokenized and **conditions the flow/action expert for that query**,
                     so the vehicle drives on it. It is then stored as HL supervision,
                     together with the ``pre_intervention_sec`` of frames leading up to it.

Both write ``steervla_hl_dataset_format`` samples with the same ``BAD``/``GOOD`` and
``direct``/``precursor`` vocabulary, so the existing ``steervla.hl_online_*`` bucketing
applies unchanged.

Cost note: unlike the CAST window review (one call per ~200 env steps, and asynchronous),
this is a **blocking single-image call on the action path**, once per CoT query. With
``actions_per_cot=5`` and ``actions_per_model_query=5`` a CoT is queried roughly every 25
env steps, so an 8k-step run is ~320 calls at ``query_every_n_cot_queries=1``. The CARLA
sim is already paused around the actor forward (``main_carla`` pauses across
``_sample_agent_action``), so the call does not advance sim time -- but it does cost wall
clock, and ``yay_robot.request_timeout_sec`` bounds how much.

Requires ``GEMINI_API_KEY``. Without it the session disables itself after
``max_consecutive_failures`` and the run degrades to a plain rollout with no interventions.

This config trains the high-level backbone (``enable_updates_bc_hl``) but writes **no
policy checkpoints**; use ``steervla_yay_robot_train_config.py`` for the 2k-step
train-and-redeploy loop.
"""

import ml_collections

from configs.steervla_cast_relabel_config import get_config as get_cast_relabel_config


def get_config():
    # Inherit the whole DSRL/SteerVLA stack (checkpoint, actor config, HL update knobs, replay
    # pools, GPU pinning) from the CAST-relabel config, then swap the labeling method.
    config = get_cast_relabel_config()

    # YAY-Robot replaces CAST relabel; main_carla refuses to run both. Turning it off here rather
    # than deleting the block keeps every cast_relabel knob visible for comparison runs.
    config.cast_relabel.enabled = False

    # The high-level (VLM-backbone) update is the point of the method: it is what teaches the
    # policy to produce the corrected language itself, so that the VLM eventually has nothing to
    # correct. RL and low-level DAgger stay off, as in the CAST arm.
    config.enable_updates = True
    config.enable_updates_bc_hl = True
    config.enable_updates_rl = False
    config.enable_updates_bc = False
    config.steervla.load_trainable_params = True

    config.yay_robot = ml_collections.ConfigDict(
        dict(
            enabled=True,
            provider="gemini",
            gemini_model="gemini-3.5-flash",
            # Print each intervention (original -> corrected) as it fires.
            debug=True,
            # Append every query (prompt + raw response + verdict) to yay_robot/queries.jsonl.
            save_artifacts=True,
            # ── when the VLM is asked ──────────────────────────────────────────────────
            # 1 = every CoT query, which is the actual YAY-Robot regime. Raise it to thin the
            # calls (2 = every other query) if the API budget or wall clock demands it.
            query_every_n_cot_queries=1,
            # Wall-clock floor between calls, on top of the count-based thinning. 0 = no floor.
            min_seconds_between_queries=0.0,
            # Hard ceiling on one correction call. vlm_feedback's Gemini client uses a 120 s
            # socket timeout and retries 429/5xx five times with exponential backoff -- worst
            # case over two minutes inside a single call, which is far too long to sit on the
            # action path. A timed-out call is abandoned (it cannot be cancelled) and the next
            # few queries pass through uncorrected until it drains.
            request_timeout_sec=30.0,
            # Give up entirely after this many consecutive failures (dead API key, no network),
            # rather than paying a timeout on every single CoT query for the rest of the run.
            max_consecutive_failures=5,
            # How many recent subtasks the prompt shows as continuity context, so the judge can
            # tell a sensible continuation from an abrupt switch.
            subtask_history_len=4,
            # Cross-query correction memory ("remain stopped -> accelerate: 7x"), injected into
            # the prompt so a per-frame stateless judge does not flip the same decision back and
            # forth and teach the backbone both directions of it. Word budget for the whole
            # rendered block; 0 disables it.
            correction_memory_words=300,
            # ── the 2 s lead-up ────────────────────────────────────────────────────────
            # When a correction fires at step S, the frames covering this many seconds before S
            # are stored with the SAME corrected language, labeled BAD/precursor. That is what
            # teaches the policy to reach the corrected subtask before the VLM has to ask.
            pre_intervention_sec=2.0,
            # CARLA ticks at 20 Hz, so 2.0 s = 40 env steps.
            env_step_dt=0.05,
            # Only every Nth env step is buffered as a lead-up candidate. One action chunk is the
            # same frame spacing cast_relabel stores, and keeps the ring buffer to a handful of
            # full-resolution images instead of 40.
            pre_intervention_stride_steps=10,
            # Ceiling on lead-up samples per intervention (newest first, so the frames closest to
            # the intervention survive the cut).
            max_pre_intervention_samples=8,
            # ── what is stored ─────────────────────────────────────────────────────────
            store_hl_dataset=True,
            # Reinforce path: store KEPT frames with the model's OWN subtask/reasoning, the
            # analogue of cast_relabel.store_good_chunks. False -> corrective samples only.
            store_kept_samples=True,
            # KEPT frames vastly outnumber interventions, so only every Nth is written or the
            # reinforce bucket swamps the corrective one on disk. The per-batch ratio is still
            # governed by steervla.hl_online_bad_fraction; this only bounds what lands.
            kept_sample_stride=4,
            hl_dataset_subdir="yay_robot_hl_dataset",
            # Must match the rollout's action chunk length (config.action_horizon), as
            # cast_relabel.action_chunk_steps does. Overwritten from it just below.
            action_chunk_steps=10,
            # Shape of the (unsupervised) stored action chunk; match config.steervla.action_dim.
            hl_action_dim=4,
            # Leave empty to use coaches.cast_relabel.SEED_SUBTASKS; set a list to override.
            seed_subtasks=[],
        )
    )
    # Keep the two shape knobs pinned to the rollout rather than to a literal, so changing the
    # chunk length in one place cannot silently mis-shape every stored sample.
    config.yay_robot.hl_action_dim = config.steervla.action_dim
    config.yay_robot.action_chunk_steps = config.action_horizon

    return config
