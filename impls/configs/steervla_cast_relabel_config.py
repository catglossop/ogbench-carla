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
    config.language_feedback.source = "expert"
    config.language_feedback.expert_mode = "none"

    config.cast_relabel = ml_collections.ConfigDict(
        dict(
            enabled=True,
            # Log annotated debug videos (per-chunk GOOD/BAD + suggested subtasks) to wandb.
            debug=True,
            # Review one window every N env steps (rounded down to whole action chunks).
            # A window can be as small as one chunk or as large as a whole episode.
            query_every_n_episode_steps=128,
            query_on_episode_end=True,
            provider="gemini",
            gemini_model="gemini-3.1-flash",
            # Must match the rollout's action chunk length (config.action_horizon).
            action_chunk_steps=10,
            # How many subtasks to suggest per chunk that needs improvement.
            num_subtask_suggestions=3,
            # Video encoding (kept consistent with main_carla's rollout video sampling).
            video_fps=10.0,
            video_frame_stride=2,
            save_artifacts=True,
            # Leave empty to use coaches.cast_relabel.SEED_SUBTASKS; set a list to override.
            seed_subtasks=[],
        )
    )

    return config
