"""Sanity-check: build the SteerVLA trainable state with the cast_relabel freeze filter.

No CARLA, no HL gradient step, no wandb. Just constructs the actor exactly as
main_carla would (load_trainable_params + hl_freeze_regexes) so setup() prints the
`[steervla] trainable params …M / …M` line and confirms the freeze regexes matched.
"""

import os
import sys

os.environ.setdefault("WANDB_MODE", "disabled")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "impls"))

from configs.steervla_cast_relabel_config import get_config  # noqa: E402
from vlas.steervla import create_steervla_pi0_cot_sample_fn  # noqa: E402

cfg = get_config()
sv = cfg.steervla.to_dict()
sv["hl_dataset_dir"] = None  # not needed for a build check

_, actor = create_steervla_pi0_cot_sample_fn(
    sv, {}, training_gpu_rank=int(cfg.training_gpu_rank)
)
print(
    f"\nBUILD OK. load_trainable_params={actor.load_trainable_params} "
    f"hl_freeze_regexes={actor.hl_freeze_regexes}",
    flush=True,
)
