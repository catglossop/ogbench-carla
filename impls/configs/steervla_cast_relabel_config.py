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
    config.batch_size = 64
    config.warmup_steps = 50
    
    config.enable_updates = True
    # Per-kind switches, each ANDed with ``enable_updates``:
    #   rl    -> critic/actor (RL) updates
    #   bc    -> full BC / DAgger imitation path (``update_dagger``)
    #   bc_hl -> high-level VLM backbone update: DSRLAgent.update_with_vla(run_hl=True) calls
    #            SteerVLAActor.update_hl, which fine-tunes the CoT/VLM backbone on the
    #            cast_relabel HL samples (subtask + reasoning targets, action loss masked out).
    config.enable_updates_rl = False
    config.enable_updates_bc = False
    config.enable_updates_bc_hl = True

    config.cast_relabel = ml_collections.ConfigDict(
        dict(
            enabled=True,
            # Log annotated debug videos (per-chunk GOOD/BAD + suggested subtasks) to wandb.
            debug=True,
            debug_task=False,
            # Review one window every N env steps (rounded down to whole action chunks).
            # A window can be as small as one chunk or as large as a whole episode.
            query_every_n_episode_steps=150,
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
            # Attach an env-reward + route-progress plot (vs video time, with chunk boundaries and
            # collisions marked) to the window review call alongside the video. These two signals
            # are the ones the per-timestamp block keeps at full density -- speed and the control
            # channels are thinned to every 2nd timestamp -- so the plot is where their shape over
            # the window is actually legible. Set False for a video-and-text-only review.
            include_plots_in_prompt=True,
            # Cross-window correction memory: a bounded record of the longitudinal changes earlier
            # windows already made ("stop -> accelerate: 7x"), injected into BOTH the review and
            # the credit prompt so successive windows don't reverse each other and the HL dataset
            # doesn't end up teaching both directions of the same decision. This is the whole
            # rendered block's word budget -- it is pruned oldest-note-first and, if still over,
            # summarized by the coach. Persisted to cast_relabel/correction_memory.json. 0 = off.
            correction_memory_words=300,
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
            # Put the high-level (VLM-backbone) gradient step on its OWN GPU, separate from the
            # inference model + DSRL RL updates (which sit on training_gpu_rank above). This is a JAX
            # device index into jax.devices("gpu") (same convention as training_gpu_rank), NOT CARLA's
            # -graphicsadapter rank. The HL train state (params + Adam mu/nu + backward activations) is
            # the memory hog, so isolating it frees the whole HL GPU for a much larger
            # hl_update_batch_size; the inference GPU only keeps a forward-only copy of the params.
            # After each HL update the refreshed weights are copied back to the inference GPU. Set to
            # -1 (or the same rank as training_gpu_rank) to disable the split and keep everything on
            # one GPU, exactly as before. Make sure this rank is NOT the CARLA renderer's GPU.
            hl_training_gpu_rank=1,
            # High-level (VLM-backbone) update knobs, consumed by SteerVLAActor.update_hl.
            # This is the gradient step that supervises the CoT/VLM backbone on the cast_relabel
            # subtask + reasoning targets (action loss masked out). Only active when
            # load_trainable_params=True and enable_updates_bc_hl=True.
            #   hl_update_batch_size -> HL samples per gradient step (default 2, too small).
            #     NOTE: this is a full Pi0-CoT forward+backward per sample. With hl_training_gpu_rank
            #     isolating the HL step on its own GPU the whole card is available, so you can push
            #     this much higher (start at 128 and bump toward 256 if memory allows); when the HL
            #     step shares the inference GPU (hl_training_gpu_rank<0) keep it modest to avoid OOM.
            #   hl_update_every      -> run update_hl once every N update_with_vla calls (1 = every).
            #   hl_update_num_steps  -> gradient steps taken per update_hl call.
            hl_update_batch_size=64,
            hl_update_every=5,
            hl_update_num_steps=2,
            # Freeze the memory-heavy pretrained subtrees for the HL (VLM-backbone) update so its
            # grad + Adam(mu/nu) buffers fit. The HL step supervises only the CoT/subtask targets
            # (action loss masked), so the SigLIP vision tower (``.*img.*``) and the tied Gemma token
            # embedder (``.*embedder.*`` -- the ~2.1 GB buffer that OOM'd) don't need to train. Frees
            # ~14 GB and keeps the LLM transformer blocks + action/CoT heads trainable. Set to [] for
            # a full fine-tune. SteerVLAActor.setup prints the resulting trainable-param count.
            hl_freeze_regexes=[".*img.*", ".*embedder.*"],
            # HL replay: mix a small amount of the ORIGINAL pretraining data into every HL update to
            # stabilize the VLM backbone (prevent it forgetting general CoT/FAST as it fine-tunes on
            # relabeled subtasks). Pre-extract the pools once with impls/vlas/extract_hl_replay.py:
            #   python impls/vlas/extract_hl_replay.py \
            #     --data-dir /scratch/current/cglossop/steervla_datasets \
            #     --out-root /scratch/current/cglossop/steervla_hl_pools \
            #     --actor-config pi05_steervla_cot_simplified_reasoning \
            #     --proprio-norm true --include-ego-history false \
            #     --dataset simlingo_dataset_all_img512_1116 --kind regular --supervise-fast true  --n 2000 \
            #     --dataset simplified_reasoning_dataset    --kind hl      --supervise-fast false --n 2000
            # Weights follow the steervla-pi mixture convention (per-source; normalized internally):
            #   online 0.6 / simlingo 0.3 / simplified_reasoning 0.1.
            # Supervision per source is baked into each pool's hl_samples.json by the extractor:
            #   simlingo (kind=regular)          -> flow + FAST + reasoning + subtask (everything).
            #   simplified_reasoning (kind=hl)   -> reasoning + subtask only (flow AND FAST masked).
            # Leave hl_replay_pools empty (or hl_online_weight=1.0) to disable and train online-only.
            hl_replay_root="/raid/users/cglossop/steervla_hl_pools",
            hl_online_weight=0.7,
            # Within the online cast_relabel share of each HL batch, bias the draw toward corrective
            # chunks: 0.8 -> 80% of the online slots come from the BAD / BAD(precursor) bucket
            # (label == "BAD", either credit_source) and 20% from the GOOD / null bucket (label GOOD or
            # absent). Whichever bucket is thin is topped up from the other so the batch still fills.
            # Set < 0 to disable (uniform draw over the whole online pool). Only the online pool is
            # bucketed; the replay pools keep their own weighted share.
            hl_online_bad_fraction=0.8,
            # Within that corrective (BAD) share, balance BAD(precursor) vs direct BAD chunks:
            # 0.5 -> half of the corrective slots are BAD(precursor) (credit_source == "precursor")
            # and half direct BAD, again topping up from whichever sub-bucket is thin. Set < 0 to
            # disable the sub-split (corrective slots drawn uniformly over BAD/precursor). Only takes
            # effect when hl_online_bad_fraction >= 0.
            hl_online_precursor_fraction=0.5,
            # How many online cast_relabel samples must exist before the first HL update fires. The
            # update never waits for the online pool to fill its full hl_online_weight share of
            # hl_update_batch_size: at/above this count it takes every online sample it has and fills
            # the rest of the batch from the offline replay pools (repeating rows only if those are
            # empty too). 1 = start updating as soon as the first relabeled window lands; 0 = allow
            # replay-only updates before any online sample exists.
            hl_min_online_samples=20,
            hl_replay_pools=[
                dict(name="simlingo_dataset_all_img512_1116", weight=0.2),
                dict(name="simplified_reasoning_dataset", weight=0.1),
            ],
            # Policy checkpointing during the run, for later *inference* (not for resuming
            # training): every ``hl_checkpoint_every_steps`` env steps — and once more at exit —
            # the HL-fine-tuned backbone is exported to ``<hl_checkpoint_dir>/<step>/params`` as a
            # params-only OpenPI checkpoint (no optimizer state). Redeploy one frozen with
            #   steervla.checkpoint="<hl_checkpoint_dir>/<step>", load_trainable_params=False
            # (same ``actor_config``). See SteerVLAActor.save_checkpoint.
            #
            # Only fires when the HL update is actually running (``enable_updates_bc_hl`` AND
            # ``load_trainable_params``) — in a rollout-only run the weights never change, so
            # there is nothing to checkpoint and main_carla says so at startup.
            #
            # 0 disables. Empty dir -> ``<save_dir>/checkpoints`` (next to videos/ and trajectories/).
            hl_checkpoint_every_steps=2000,
            hl_checkpoint_dir="",
            # Each checkpoint is ~10 GB, so 2000-step spacing over a 20k-step run is ~100 GB.
            # Keep only the newest N step dirs (pruned after each successful write); 0 = keep all.
            hl_checkpoint_keep_last=3,
            # Local OpenPI inference (ignored when actor_url is set):
            # actor_config="pi05_steervla_cot_simplified_reasoning",
            actor_config="pi05_steervla_cot_simplified_reasoning_no_ego_history",
            # checkpoint="gs://cat-logs/pi05_steervla_cot_simplified_reasoning/pi05_steervla_cot_simplified_reasoning/pi05_steervla_cot_simplified_reasoning_20260523_222304/8000",
            # checkpoint="/scratch/current/cglossop/openpi/cat-logs/pi05_steervla_cot_simplified_reasoning_ego_history/pi05_steervla_cot_simplified_reasoning_ego_history_v1/pi05_steervla_simplified_reasoning_ego_history_v1_20260717_183503/4000",
            # checkpoint="gs://cat-logs/pi05_steervla_cot_simplified_reasoning_ego_history/pi05_steervla_simplified_reasoning_ego_history_v1/pi05_steervla_simplified_reasoning_ego_history_v1_20260717_183503/10000",
            checkpoint="gs://cat-logs/pi05_steervla_cot_simplified_reasoning_no_ego_history/pi05_steervla_simplified_reasoning_no_ego_history_v1/pi05_steervla_simplified_reasoning_no_ego_history_v1_20260718_201640/6000",
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
