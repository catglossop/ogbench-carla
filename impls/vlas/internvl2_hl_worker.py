"""Isolated torch HL inference (and, optionally, online HL training). Communication uses an inherited private socket.

Requests are dicts. A request without ``cmd`` (or ``cmd='generate'``) is one HL sample, as before. With
``--train-config`` the worker also owns a trainable copy of the HL and accepts:

* ``cmd='update_hl'`` -- one throttled HL update on ``hl_dataset_dir`` (+ replay pools), exactly
  ``SimLingoSteerVLAActor.update_hl``; replies ``{'info': {...}}``.
* ``cmd='save_checkpoint'`` -- ``SimLingoSteerVLAActor.save_checkpoint`` (``<out_root>/<step>/pytorch_model.bin``
  + ``.hydra/config.yaml``, the format HARP exports and the mixed actor deploys); replies ``{'path': ...}``.

Training reuses ``SimLingoSteerVLAActor``'s HL-update code unchanged (HL-only: no LL is loaded), so the
mixed actor's YAY-Robot arm trains InternVL2 the same way the SimLingo YAY arm does.
"""
from __future__ import annotations

import argparse
import json
from multiprocessing.connection import Connection
import os
from pathlib import Path
import sys
import traceback

import torch

from simlingo_model import _SimLingoModel, _import_simlingo, split_hl_output


def _build_trainer(checkpoint: str, source_root: Path, cfg: dict):
    """HL-only ``SimLingoSteerVLAActor``: its update_hl / save_checkpoint on a single trainable InternVL2."""
    # simlingo_steervla imports vlas.steervla (JAX/openpi). Keep JAX off the GPU in this process.
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    impls_dir = str(Path(__file__).resolve().parents[1])
    if impls_dir not in sys.path:
        sys.path.insert(0, impls_dir)
    from vlas.simlingo_steervla import SimLingoSteerVLAActor
    from vlas.simlingo_steervla import _SimLingoModel as _TrainModel

    class HLTrainer(SimLingoSteerVLAActor):
        def __init__(self):  # noqa: D401 - deliberately skips the LL / rollout state of the parent
            self.hl_dataset_dir = None
            self.hl_update_every = max(1, int(cfg.get('hl_update_every', 1)))
            self.hl_update_batch_size = max(1, int(cfg.get('hl_update_batch_size', 16)))
            self.hl_update_num_steps = max(1, int(cfg.get('hl_update_num_steps', 1)))
            self.hl_micro_batch_size = max(1, int(cfg.get('hl_micro_batch_size', 8)))
            self.hl_lr = float(cfg.get('hl_lr') or 1e-5)
            self.hl_weight_decay = float(cfg.get('hl_weight_decay', 0.1))
            self.hl_grad_clip = float(cfg.get('hl_grad_clip', 1.0))
            self.hl_kl_coef = float(cfg.get('hl_kl_coef', 0.0))
            self.hl_log_kl = bool(cfg.get('hl_log_kl', True))
            self._hl_ref_model = None
            self._init_hl_pool_sampling(
                hl_replay_root=cfg.get('hl_replay_root'),
                hl_replay_pools=cfg.get('hl_replay_pools'),
                hl_online_weight=float(cfg.get('hl_online_weight', 1.0)),
                hl_online_bad_fraction=float(cfg.get('hl_online_bad_fraction', -1.0)),
                hl_online_precursor_fraction=float(cfg.get('hl_online_precursor_fraction', -1.0)),
                hl_online_backfill_from_replay=bool(cfg.get('hl_online_backfill_from_replay', True)),
                use_adaptive_sampling=bool(cfg.get('use_adaptive_sampling', False)),
                adaptive_sampling_weights=cfg.get('adaptive_sampling_weights') or None,
                hl_min_online_samples=int(cfg.get('hl_min_online_samples', 1)),
                hl_keep_last_rounds=int(cfg.get('hl_keep_last_rounds', 0)),
            )
            self.device = self.hl_device = torch.device('cuda:0')
            self.hl = _TrainModel(checkpoint, source_root=source_root, device=self.hl_device, expect_type='hl')
            self._hl_params, self._hl_master, self._hl_optimizer = [], [], None
            self._setup_hl_training()

    return HLTrainer()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fd', type=int, required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--train-config', default='', help='JSON of steervla.hl_* knobs; enables online HL training')
    args = parser.parse_args()
    conn = Connection(args.fd)
    try:
        root = Path(args.source_root)
        _import_simlingo(root)
        trainer = None
        if args.train_config:
            trainer = _build_trainer(args.checkpoint, root, json.loads(args.train_config))
            hl = trainer.hl  # generation runs on the weights being trained
        else:
            hl = _SimLingoModel(args.checkpoint, source_root=root, device=torch.device('cuda:0'), expect_type='hl')
            hl.model.requires_grad_(False)
        if hl.dataset_cfg.get('use_history_image', False):
            raise ValueError('History images are not supported by this HL bridge.')
        original_sample = hl.model.language_model.greedy_sample
        # Per-request HL sampling temperature (0 = greedy, the default). Best-of-N needs > 0 so each
        # candidate gets its own subtask; a per-request seed keeps those draws reproducible.
        sampling = dict(temperature=0.0)

        def sample(*args, **kwargs):
            kwargs['temperature'] = sampling['temperature']
            return original_sample(*args, **kwargs)

        hl.model.language_model.greedy_sample = sample
        # HL_KV_CACHE=1 swaps in a KV-cached loop instead (hl_kv_cache.py). Same model, same
        # positions -- step 0 is bitwise identical -- but bf16 attention takes a different
        # reduction path once the cache is in play, so a step whose top-1/top-2 margin is inside
        # that noise can flip. Sampled text is therefore a valid draw, not a reproducible one.
        if os.environ.get('HL_KV_CACHE') == '1':
            from hl_kv_cache import install_kv_cache

            install_kv_cache(hl, sampling)
            print('[internvl2-hl] KV-cached greedy_sample enabled (HL_KV_CACHE=1)', flush=True)
        conn.send(dict(ready=True, weights=str(hl.weights), trainable=trainer is not None,
                       use_ego_history=bool(hl.dataset_cfg.get('use_ego_state_history', False)),
                       history_count=int(hl.dataset_cfg.get('ego_state_history_count', 3))))
        vision_cache = None
        scene_id = None
        if os.environ.get('HL_REUSE_VISION') == '1':
            if trainer is not None:
                raise ValueError('HL_REUSE_VISION caches frozen vision features; it cannot be used while training.')
            from frozen_vision_cache import FrozenVisionCache
            vision_cache = FrozenVisionCache(hl.model.vision_model.image_encoder.model)
        while True:
            request = conn.recv()
            if request is None:
                break
            cmd = request.get('cmd', 'generate')
            if cmd == 'update_hl':
                if trainer is None:
                    raise ValueError('update_hl sent to an inference-only HL worker (no --train-config).')
                trainer.hl_dataset_dir = Path(request['hl_dataset_dir']) if request.get('hl_dataset_dir') else None
                info = trainer.update_hl(global_step=request.get('global_step'))
                conn.send(dict(info={k: float(v) for k, v in (info or {}).items()
                                     if isinstance(v, (int, float, bool))}))
                continue
            if cmd == 'save_checkpoint':
                if trainer is None:
                    raise ValueError('save_checkpoint sent to an inference-only HL worker (no --train-config).')
                path = trainer.save_checkpoint(request['out_root'], int(request['step']),
                                               keep_last=int(request.get('keep_last', 0)))
                conn.send(dict(path=None if path is None else str(path)))
                continue
            if cmd != 'generate':
                raise ValueError(f'unknown HL worker command {cmd!r}')
            if vision_cache is not None:
                # A missing ID is deliberately uncacheable (legacy callers).
                incoming_scene = request.get('scene_id')
                if incoming_scene is None or incoming_scene != scene_id:
                    vision_cache.clear()
                scene_id = incoming_scene
            sampling['temperature'] = float(request.get('temperature', 0.0))
            if request.get('seed') is not None:
                torch.manual_seed(int(request['seed']))
            with torch.no_grad():
                # Always the inference-path forward (simlingo_model._SimLingoModel.generate), also on the
                # trainer's model: it is what the mixed evals decode with, and the training class's own
                # HL ``generate`` caps at 100 new tokens, which cuts the subtask off a full scene description.
                _, _, language = _SimLingoModel.generate(hl, request['image'], request['prompt'])
            output = str(language[0] if language else '')
            reasoning, subtask = split_hl_output(output)
            if not subtask.strip():
                raise ValueError(f'InternVL2 returned an empty subtask: {output!r}')
            conn.send(dict(reasoning=reasoning, subtask=subtask, output=output,
                           vision_cache_hits=vision_cache.hits if vision_cache else 0,
                           vision_cache_misses=vision_cache.misses if vision_cache else 0))
    except EOFError:
        pass
    except Exception:
        error = traceback.format_exc()
        print(error, flush=True)
        conn.send(dict(error=error))
        raise
    finally:
        conn.close()


if __name__ == '__main__':
    main()
