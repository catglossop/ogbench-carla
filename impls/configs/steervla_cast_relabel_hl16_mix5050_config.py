"""CAST-relabel / HL-DAgger run: HL update every 16, 50/50 offline-online batch, LR 1e-5, CoT T=0.5.

Successor to ``steervla_cast_relabel_hl8_lr5e5_config.py``. Motivation: with
``hl_update_every=8`` the policy degraded after roughly 1k env steps / 40 HL updates, the point
at which the online cast_relabel pool starts being heavily re-drawn. Two changes attack that:

  * **Half the update rate** (``hl_update_every`` 8 -> 16), so each online sample is reused half
    as often for the same number of env steps.
  * **Half the batch drawn from the frozen pretraining pools** (``hl_online_weight`` 0.7 -> 0.5),
    so the anti-forgetting anchor is twice as strong relative to the online corrective data.

Batch composition. ``_load_hl_batch`` splits the batch across the online pool and the replay
pools by their **jointly normalized** weights, so these three numbers are one mixture summing
to 1.0 (the same convention as steervla-pi's dataset mixture):

    online                            0.5     <- 50% online
    simlingo_dataset_all_img512_1116  0.4     <- 80% of the 50% offline share
    simplified_reasoning_dataset      0.1     <- 20% of the 50% offline share

Online sub-composition. Within the online half, ``hl_online_bad_fraction`` splits corrective
(BAD, either credit_source) against GOOD/unlabeled, and ``hl_online_precursor_fraction`` then
splits that corrective share between BAD(precursor) and direct BAD. To hit
60% precursor / 30% direct / 10% good of the online half:

    hl_online_bad_fraction        = 0.90      # 60 + 30 corrective
    hl_online_precursor_fraction  = 0.6667    # 60 / 90 of the corrective share

Whichever bucket is thin is topped up from the other, so these are targets rather than
guarantees when the online pool is small or skewed.

Everything else -- matched-crop 6000 checkpoint, ``proprio_norm=False``,
``hl_update_batch_size=64``, freeze regexes, single-card pinning -- is unchanged from the hl8
config. Launch identically::

    CUDA_VISIBLE_DEVICES=5 ./carla_job.sh start --job 4 --train-gpu 0 --render-adapter 5 \\
        --route generalization-wall-1095 -- --hl-gpu 0 \\
        --agent-config impls/configs/steervla_cast_relabel_hl16_mix5050_config.py \\
        --train-mode rl --critic-mode none --online-steps 8000 --save-buffer true
"""

from configs.steervla_cast_relabel_hl8_lr5e5_config import get_config as get_hl8_config

HL_UPDATE_EVERY = 16
# Overrides the 5e-5 inherited from the hl8 parent. Degradation was still appearing under the
# hl16 + 50/50 mixture at 5e-5, so the step size comes down as well as the update rate; 1e-5 is
# also what ``steervla_cast_relabel_config`` ships as its own default.
HL_LR = 1e-5
ONLINE_WEIGHT = 0.5
SIMLINGO_WEIGHT = 0.4          # 0.8 of the 0.5 offline share
SIMPLIFIED_REASONING_WEIGHT = 0.1  # 0.2 of the 0.5 offline share
BAD_FRACTION = 0.90            # 60% precursor + 30% direct BAD
PRECURSOR_FRACTION = 60.0 / 90.0   # of the corrective share
# Per-step CoT sampling temperature for the rollout actor (base config ships 1.0). Lowering it
# to 0.5 narrows the reasoning/subtask distribution the online pool is relabeled from, so fewer
# off-format samples enter the corrective batch. NOT the same knob as ``vla_cot_temperature``,
# which only governs best-of-N candidate sampling (unused here: --critic-mode none).
COT_TEMPERATURE = 0.5


def get_config():
    config = get_hl8_config()

    # Smaller step per update, on top of the reduced update rate below.
    config.steervla.hl_lr = HL_LR

    # Narrower CoT sampling for the rollout actor (base config ships 1.0).
    config.steervla.cot_temperature = COT_TEMPERATURE

    # Halve the HL update rate: same env steps, half the re-use of each online sample.
    config.steervla.hl_update_every = HL_UPDATE_EVERY

    # 50/50 online vs frozen pretraining replay. Jointly normalized with the pool weights below.
    config.steervla.hl_online_weight = ONLINE_WEIGHT
    config.steervla.hl_replay_pools = [
        dict(name="simlingo_dataset_all_img512_1116", weight=SIMLINGO_WEIGHT),
        dict(name="simplified_reasoning_dataset", weight=SIMPLIFIED_REASONING_WEIGHT),
    ]

    # Online half: 60% BAD(precursor) / 30% direct BAD / 10% GOOD-or-unlabeled.
    config.steervla.hl_online_bad_fraction = BAD_FRACTION
    config.steervla.hl_online_precursor_fraction = PRECURSOR_FRACTION

    return config
