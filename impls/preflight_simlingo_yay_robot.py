"""Non-CARLA startup gate for the SimLingo SteerVLA YAY-Robot configuration."""
from __future__ import annotations

import argparse
import runpy
import tempfile
from pathlib import Path

import torch

from vlas.simlingo_steervla import create_simlingo_steervla_sample_fn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-config", required=True)
    parser.add_argument("--train-gpu", type=int, required=True)
    parser.add_argument("--hl-gpu", type=int, required=True)
    args = parser.parse_args()

    get_config = runpy.run_path(args.agent_config)["get_config"]
    config = get_config()
    steervla = config.steervla
    steervla.training_gpu_rank = args.train_gpu
    steervla.hl_training_gpu_rank = args.hl_gpu

    # Before rollout there cannot be online corrections. Exercise the identical replay loader,
    # model.forward_loss, backward pass, and optimizer step with one actual offline record.
    with tempfile.TemporaryDirectory(prefix="simlingo_yay_preflight_") as empty_online:
        steervla.hl_dataset_dir = empty_online
        steervla.hl_online_weight = 0.0
        steervla.hl_min_online_samples = 0
        steervla.hl_update_every = 1
        _, actor = create_simlingo_steervla_sample_fn(steervla, {})
        info = actor.update_hl(batch_size=1, num_steps=1, global_step=0)
        if not info or info.get("n_samples") != 1.0:
            raise RuntimeError(f"HL preflight did not apply an update: {info}")
        print(
            "[simlingo_yay_preflight] passed: "
            f"one offline HL forward/backward/update, loss={info['loss']:.4f}",
            flush=True,
        )
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
