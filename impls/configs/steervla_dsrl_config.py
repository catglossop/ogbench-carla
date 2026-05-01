"""``get_config()`` for DSRL + (optional) SteerVLA on CARLA Bench2Drive.

Use with ``--agent=impls/configs/steervla_dsrl_config.py``. This is a thin
convenience wrapper around :func:`jax_agents.dsrl.get_config` with sensible
defaults for online single-route runs and a ``steervla`` block describing how
to plug in a frozen SteerVLA flow.

Set ``config.observation_mode`` to ``"state"`` (default) or ``"image"`` so DSRL
reads either the vector ``obs['state']`` or RGB ``obs['image']`` from the CARLA env.

Plugging in SteerVLA is two-line: at agent ``create()`` time, build a callable
``vla_sample_fn(obs_batch, noise_batch) -> action_batch`` that wraps your
loaded ``openpi`` policy and pass it to :meth:`DSRLAgent.create`. ``main_carla``
does this when ``config.steervla.enabled = True`` and a checkpoint is provided.
"""

from __future__ import annotations

import ml_collections

from jax_agents import dsrl as dsrl_agent


def get_config():
    config = dsrl_agent.get_config()

    config.lr = 3e-4
    config.batch_size = 256
    config.flow_steps = 5
    config.noise_scale = 1.0
    config.alpha = 0.1
    config.warmup_steps = 1000
    config.updates_per_step = 1
    config.buffer_capacity = 100_000
    # DSRL trains on ``observation_mode`` only; env step always returns both keys.
    config.observation_mode = "state"
    config.steervla = None

    # config.steervla = ml_collections.ConfigDict(
    #     dict(
    #         enabled=False,
    #         actor_config="pi05_steervla_inference",
    #         checkpoint="gs://cat-logs/pi05_steervla_cot_ki/pi05_steervla_cot_ki/90000",
    #         actor_url=None,
    #     )
    # )

    return config
