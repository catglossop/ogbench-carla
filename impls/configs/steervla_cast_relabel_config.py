"""``get_config()`` for DSRL + SteerVLA on CARLA with the CAST relabel observer.

Same DSRL/SteerVLA stack as ``steervla_dsrl_config.py``, but adds a ``cast_relabel``
block that drives :class:`coaches.cast_relabel.OnlineCastRelabelSession`. That session
runs *alongside* the rollout as a pure observer: every ``query_every_n_episode_steps``
env steps (rounded down to whole action chunks) it

  1. writes the window video,
  2. asks a VLM what the agent did well / poorly,
  3. assigns that GOOD/BAD credit to the individual action chunks, and
  4. suggests subtasks (open-vocab, seeded by ``cast_relabel.seed_subtasks``) to improve
     each chunk.

Consumption is **artifacts + wandb only** — nothing is written back to the replay buffer
or the DSRL critic. Set ``cast_relabel.debug=True`` to also log annotated debug videos to
Weights & Biases (original subtask + waypoints/actions are already drawn upstream; the
debug pass overlays the per-chunk GOOD/BAD labels and suggested subtasks).

Because CAST relabel is observational, the DSRL critic language feedback is disabled here
(``language_feedback.source='expert'``, ``expert_mode='none'``) so no second VLM coach
spins up. Flip those back on if you also want live critic coaching.
"""

import ml_collections

from configs.steervla_dsrl_config import get_config as get_dsrl_config


def get_config():
    config = get_dsrl_config()

    # CAST relabel is a pure observer; don't also spin up the DSRL VLM critic coach.
    config.lr = 3e-5
    config.language_feedback.source = "expert"
    config.language_feedback.expert_mode = "none"
    config.training_gpu_rank = 0
    config.siglip_device = "cuda:0"
    config.batch_size = 32
    config.warmup_steps = 500
    
    config.enable_updates = True
    # Per-kind switches, each ANDed with ``enable_updates``:
    #   rl    -> critic/actor (RL) updates
    #   bc    -> full BC / DAgger imitation path (``update_dagger``)
    #   bc_hl -> high-level VLM backbone update: DSRLAgent.update_with_vla(run_hl=True) calls
    #            SteerVLAActor.update_hl, which fine-tunes the CoT/VLM backbone on the
    #            cast_relabel HL samples (subtask + reasoning targets, action loss masked out).
    config.enable_updates_rl = False
    config.enable_updates_bc = True
    config.enable_updates_bc_hl = True

    config.cast_relabel = ml_collections.ConfigDict(
        dict(
            enabled=True,
            # Log annotated debug videos (per-chunk GOOD/BAD + suggested subtasks) to wandb.
            debug=True,
            debug_task=True,
            # Review one window every N env steps (rounded down to whole action chunks).
            # A window can be as small as one chunk or as large as a whole episode.
            query_every_n_episode_steps=200,
            query_on_episode_end=True,
            provider="gemini",
            gemini_model="gemini-3.5-flash",
            # Must match the rollout's action chunk length (config.action_horizon).
            action_chunk_steps=10,
            # How many subtasks to suggest per chunk that needs improvement.
            num_subtask_suggestions=3,
            # Video encoding (kept consistent with main_carla's rollout video sampling).
            video_fps=10.0,
            video_frame_stride=2,
            save_artifacts=True,
            # Persist every BAD/relabeled chunk as a SteerVLA high-level (VLM-backbone) training
            # sample in the steervla_hl_dataset_format schema (image + ego state + prompt +
            # corrected subtask + new reasoning trace + masked action chunk). These are consumed
            # online by SteerVLAActor.update_hl (gated by enable_updates_bc_hl above), which runs a
            # real OpenPI gradient step supervising the CoT/VLM backbone on the subtask + reasoning.
            store_hl_dataset=True,
            # Also reinforce GOOD/unlabeled chunks: store their *original* (uncorrected) subtask +
            # reasoning as HL samples so good behavior is imitated. False -> corrective (BAD) only.
            store_good_chunks=True,
            hl_dataset_subdir="cast_relabel_hl_dataset",
            # Shape of the (unsupervised) stored action chunk; match config.steervla.action_dim.
            hl_action_dim=4,
            # Leave empty to use coaches.cast_relabel.SEED_SUBTASKS; set a list to override.
            seed_subtasks=[],
        )
    )
    
    config.steervla = ml_collections.ConfigDict(
        dict(
            enabled=True,
            # Load SteerVLA as a trainable model: build the full OpenPI TrainState
            # (optimizer + opt_state + freeze/trainable filters) like scripts/train.py,
            # instead of an inference-only param restore. Needed for the cast_relabel
            # fine-tuning step. Set False for a frozen inference-only actor.
            load_trainable_params=True,
            # High-level (VLM-backbone) update knobs, consumed by SteerVLAActor.update_hl.
            # This is the gradient step that supervises the CoT/VLM backbone on the cast_relabel
            # subtask + reasoning targets (action loss masked out). Only active when
            # load_trainable_params=True and enable_updates_bc_hl=True.
            #   hl_update_batch_size -> HL samples per gradient step (default 2, too small).
            #     NOTE: this is a full Pi0-CoT forward+backward per sample; 256 may OOM the
            #     training GPU, so start at 128 and bump to 256 if memory allows.
            #   hl_update_every      -> run update_hl once every N update_with_vla calls (1 = every).
            #   hl_update_num_steps  -> gradient steps taken per update_hl call.
            hl_update_batch_size=16,
            hl_update_every=10,
            hl_update_num_steps=1,
            # Local OpenPI inference (ignored when actor_url is set):
            actor_config="pi05_steervla_cot_simplified_reasoning",
            checkpoint="gs://cat-logs/pi05_steervla_cot_simplified_reasoning/pi05_steervla_cot_simplified_reasoning/pi05_steervla_cot_simplified_reasoning_20260523_222304/8000",
            routing_command="Follow the route and stay in lane.",
            # Per-step CoT sampling temperature for the rollout actor. Best-of-N samples
            # ``best_of_n`` CoTs via ``sample_candidates`` at ``vla_cot_temperature`` (set above),
            # so this default only affects any non-candidate single-sample paths.
            cot_temperature=1.0,
            include_ego_history=False,
            proprio_norm=True,
            # Replay buffer + CARLA ``step`` use OpenPI chunk layout (``action_horizon`` × ``action_dim``),
            # executed like ``simlingo/team_code/agent_steervla.py`` (cumsums + PID).
            use_pi_action_chunk_for_env=True,
            action_horizon=10,
            action_dim=4,
            # Query Pi0-CoT once, then execute this many rows before re-querying (1 = every step).
            actions_per_model_query=1,
            # Reuse sampled CoT reasoning/subtask for this many env actions before re-sampling CoT.
            actions_per_cot=1,
            output_action_format="DELTA_XY_T_DELTA_XY_SPACE",
            sample_actions_num_steps=10,
            # Decode action chunks in small micro-batches (CoT stays batched). Large
            # batched _sample_actions forwards can trigger native aborts on some drivers.
            action_decode_batch_size=2,
            # Remote HTTP actor is NOT supported for best-of-N (needs local sample_candidates):
            # actor_url="http://35.186.30.251:8000",
        )
    )

    return config
