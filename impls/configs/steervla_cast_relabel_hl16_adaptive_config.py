"""CAST-relabel / HL-DAgger run: the hl16 / 50-50 / LR 1e-5 / CoT T=0.5 recipe, plus adaptive
severity weighting of the online pool.

Identical to ``steervla_cast_relabel_hl16_mix5050_config.py`` in every other respect -- same
matched-crop 6000 checkpoint, ``proprio_norm=False``, ``hl_update_every=16``,
``hl_update_batch_size=64``, ``hl_lr=1e-5``, ``cot_temperature=0.5``, same 50/40/10 pool mixture --
so a comparison against those runs isolates the sampling change.

What changes: ``use_adaptive_sampling`` draws the online cast_relabel pool by per-sample severity
weight instead of uniformly *within* each bucket. Two layers stay separate:

  * **Coverage.** ``hl_online_bad_fraction=0.9`` still fixes how many corrective rows a batch gets
    against how many reinforce rows -- adaptive sampling does not touch that quota, only which
    rows fill it.
  * **Severity.** Inside the corrective share, a BAD(precursor) chunk that set up a catastrophic
    outcome (a counted collision, or the ego abandoning the routing command) outranks an ordinary
    precursor, which outranks a directly-blamed BAD chunk -- by the time a chunk directly overlaps
    the failure the mistake is usually already unavoidable, so it teaches less than its lead-up.
    Inside the reinforce share, a GOOD chunk from a route-completing episode outranks an ordinary
    GOOD, which outranks an unlabeled chunk. Weights: ``ADAPTIVE_SAMPLING_WEIGHTS`` in
    ``impls/vlas/steervla.py``.

``hl_online_precursor_fraction`` (0.6667 from the parent) is IGNORED under adaptive sampling -- the
precursor/direct balance is what the weights express, and a hard sub-split on top would cap the
very rows the weighting exists to promote. It is left set rather than cleared so the lineage stays
readable; ``update_hl`` prints a NOTE at the first update saying it is being ignored.

The severity categories need an online corpus carrying ``outcome`` tags, which
``coaches/cast_relabel.py`` began writing on 2026-09-07. These runs collect their own pool from
scratch, so they get them; pointing this config at a pre-existing pool would fall back to
precursor-over-direct weighting only, and ``update_hl`` warns once when it sees that.

Launch (both jobs, one card each)::

    CUDA_VISIBLE_DEVICES=5 ./carla_job.sh start --job 4 --train-gpu 0 --render-adapter 5 \\
        --route generalization-wall-1095 -- --hl-gpu 0 \\
        --agent-config impls/configs/steervla_cast_relabel_hl16_adaptive_config.py \\
        --train-mode rl --critic-mode none --online-steps 8000 --save-buffer true \\
        -- --stop_on_driving_score=100 --updates_after_driving_score=10 \\
           --max_hl_updates=150 --post_stop_eval_episodes=3
"""

from configs.steervla_cast_relabel_hl16_mix5050_config import get_config as get_hl16_config

USE_ADAPTIVE_SAMPLING = True


def get_config():
    config = get_hl16_config()
    config.steervla.use_adaptive_sampling = USE_ADAPTIVE_SAMPLING
    # Empty -> the ADAPTIVE_SAMPLING_WEIGHTS defaults. Override individual categories here (e.g.
    # {"bad_direct": 0.25}) if the balance needs tuning; unknown keys raise rather than silently
    # doing nothing, and weights are normalized inside each bucket so only within-bucket ratios
    # matter (``good_success`` competes with ``good``, never with the corrective weights).
    config.steervla.adaptive_sampling_weights = {}
    return config
