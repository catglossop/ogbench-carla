"""InternVL2 text planner + pi05's own vision backbone and action expert.

Inference is the original use. With ``steervla.load_trainable_params=True`` the actor also runs the
CAST-relabel HL update: the InternVL2 optimizer lives in the worker process (see
``internvl2_hl_worker.py``), because that is where the HL model is, and this class only samples the
batch and drives the worker. pi05 stays frozen and inference-only either way -- the low level is not
trained on this path, so ``SteerVLAActor``'s OpenPI train state is deliberately never restored.
"""
from __future__ import annotations

import atexit
from collections import deque
import json
from multiprocessing.connection import Connection
import os
from pathlib import Path
import socket
import subprocess
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from vlas.steervla import SteerVLAActor, strip_cot_sentinels


class MixedSteerVLAActor(SteerVLAActor):
    def __init__(self, *, hl_checkpoint, simlingo_source_root, hl_python,
                 hl_micro_batch_size: int = 8, hl_weight_decay: float = 0.1,
                 hl_grad_clip: float = 1.0, **kwargs):
        if kwargs.get('fixed_subtask_text'):
            raise ValueError('Mixed SteerVLA drives the subtask from the live InternVL2 HL.')
        # The HL trains in the worker; pi05 must stay inference-only, so the flag is kept away from
        # SteerVLAActor (which would restore a full OpenPI TrainState) and re-applied afterwards --
        # main_carla gates the CAST dataset wiring and checkpointing on the attribute.
        trainable = bool(kwargs.pop('load_trainable_params', False))
        self.hl_micro_batch_size = max(1, int(hl_micro_batch_size))
        self.hl_weight_decay = float(hl_weight_decay)
        self.hl_grad_clip = float(hl_grad_clip)
        self._hl_history = deque(maxlen=402)
        super().__init__(**kwargs)
        self.load_trainable_params = trainable
        # The HL update is the only gradient work here, so it may sit on its own card
        # (run_carla.sh --hl-gpu); otherwise it shares pi05's.
        hl_rank = int(kwargs.get('hl_training_gpu_rank', -1))
        if hl_rank < 0:
            hl_rank = int(kwargs['training_gpu_rank'])
        parent, child = socket.socketpair()
        self._hl_conn = Connection(parent.detach())
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(hl_rank)
        env.pop('PYTHONPATH', None)
        argv = [
            hl_python, '-u', str(Path(__file__).with_name('internvl2_hl_worker.py')),
            '--fd', str(child.fileno()), '--checkpoint', hl_checkpoint,
            '--source-root', simlingo_source_root,
        ]
        if trainable:
            if not self.hl_lr:
                raise ValueError('Trainable mixed SteerVLA needs steervla.hl_lr (the torch HL optimizer '
                                 'has no schedule to fall back on).')
            argv += [
                '--trainable',
                '--hl-lr', repr(float(self.hl_lr)),
                '--hl-weight-decay', repr(self.hl_weight_decay),
                '--hl-grad-clip', repr(self.hl_grad_clip),
                '--hl-kl-coef', repr(float(self.hl_kl_coef)),
                '--hl-micro-batch-size', str(self.hl_micro_batch_size),
            ]
        self._hl_proc = subprocess.Popen(argv, pass_fds=(child.fileno(),), env=env)
        child.close()
        atexit.register(self.close)
        ready = self._receive_hl(600)
        self._hl_use_history = ready['use_ego_history']
        self._hl_history_count = ready['history_count']
        self._hl_history = deque(maxlen=40 * self._hl_history_count + 2)
        mode = 'trainable HL' if trainable else 'frozen HL'
        print(f"[mixed-steervla] HL={ready['weights']} (gpu {hl_rank}, {mode}) "
              f"LL={kwargs['checkpoint_path']} pi05 vision preserved; InternVL2 text conditioning only",
              flush=True)

    def close(self):
        self._hl_conn.close()
        if self._hl_proc.poll() is None:
            self._hl_proc.terminate()
            try:
                self._hl_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._hl_proc.kill()
                self._hl_proc.wait()

    def _receive_hl(self, timeout=180):
        if not self._hl_conn.poll(timeout):
            raise TimeoutError(f'InternVL2 worker did not respond in {timeout}s')
        response = self._hl_conn.recv()
        if 'error' in response:
            raise RuntimeError(response['error'])
        return response

    def __call__(self, observations_jax, noise_jax):
        raw = self.raw_obs_holder['obs']
        state = np.asarray(raw['state']).reshape(-1)
        self._hl_history.append((float(state[15]), float(state[5])))
        return super().__call__(observations_jax, noise_jax)

    def reset_action_cache(self):
        super().reset_action_cache()
        self._hl_history.clear()

    # ---- online HL update (the worker owns the optimizer) ----------------------------------- #

    def _read_hl_record(self, e: dict[str, Any]) -> dict[str, Any] | None:
        """One HL sample for ``HLPoolSamplingMixin._load_hl_batch``, as a *path* plus its text.

        The frame stays on disk: the worker decodes it. A 64-record batch of 1024x512 frames is
        ~100 MB, which is not worth pushing through the socket when both processes share the disk.
        Text handling matches ``SimLingoSteerVLAActor._read_hl_record`` so the two stacks build the
        same supervision from the same pools.
        """
        path = Path(e['dir']) / str(e['file'])
        if not path.exists():
            return None
        pool = str(e.get('pool', 'online'))
        prompt = str(e.get('prompt') or '').strip()
        subtask = strip_cot_sentinels(e.get('subtask', ''))
        if not prompt or not subtask:
            return None
        online = pool == 'online'
        # Online relabels keep the rollout reasoning as context and supervise only the subtask;
        # cast_relabel's ``reasoning`` is a SteerVLA-style replacement, not SimLingo's format.
        reasoning = str(e.get('original_reasoning') or '').strip() if online else ''
        if not reasoning:
            reasoning = strip_cot_sentinels(e.get('reasoning', ''))
        return {
            'image_path': str(path),
            'prompt': prompt,
            'subtask': subtask,
            'reasoning': reasoning,
            'loss_scope': 'subtask' if online else 'full',
            'pool': pool,
            'label': e.get('label'),
            'credit_source': e.get('credit_source', ''),
            'policy_version': int(e.get('policy_version', -1)),
            'sample_id': str(path),
        }

    def _dump_hl_batch(self, records: list[dict[str, Any]], *, global_step: int | None) -> None:
        """Text view of the exact batch each update trained on, next to the HL dataset."""
        if self.hl_dataset_dir is None:
            return
        try:
            out_dir = Path(self.hl_dataset_dir).parent / 'mixed_hl_batches'
            out_dir.mkdir(parents=True, exist_ok=True)
            rows = [{k: r.get(k) for k in ('sample_id', 'pool', 'label', 'credit_source',
                                           'loss_scope', 'prompt', 'reasoning', 'subtask')}
                    for r in records]
            name = f'update_{self._hl_updates_applied + 1:06d}_step_{int(global_step or -1)}.json'
            (out_dir / name).write_text(json.dumps(rows, indent=2), encoding='utf-8')
        except Exception as exc:  # noqa: BLE001 - logging must never break the update.
            print(f'[mixed-steervla.update_hl] batch dump failed (non-fatal): {exc}', flush=True)

    def update_hl(self, *, batch_size=None, num_steps=None, rng=None, global_step=None) -> dict[str, float]:
        """Throttled HL gradient steps on the cast_relabel pool (+ replay); same contract as SteerVLA."""
        del rng
        if not self.load_trainable_params:
            return self._hl_skip('actor was not loaded trainable (needs steervla.load_trainable_params=True)')
        if self.hl_dataset_dir is None:
            return self._hl_skip('hl_dataset_dir is unset — main_carla only wires it when cast_relabel.enabled '
                                 'and load_trainable_params are both true')
        self._hl_update_calls += 1
        if self.hl_update_every > 1 and (self._hl_update_calls % self.hl_update_every != 0):
            return {}
        bs = int(batch_size or self.hl_update_batch_size)
        ns = int(num_steps or self.hl_update_num_steps)
        records = self._load_hl_batch(bs)
        if records is None:
            return self._hl_skip(f'waiting for the online HL pool to start: {self._hl_pool_size}/'
                                 f'{self.hl_min_online_samples} samples under {self.hl_dataset_dir}')
        self._last_hl_skip_reason = None
        reuse_info = self._record_hl_sample_uses(records, ns)
        self._dump_hl_batch(records, global_step=global_step)
        started = time.time()
        self._hl_conn.send(dict(op='train', records=records, num_steps=ns))
        # One update is 10 grad steps over a 64-record batch: minutes, not the default 180 s.
        info = self._receive_hl(3600)
        if 'skipped' in info:
            return self._hl_skip(str(info['skipped']))
        self._hl_updates_applied += 1
        self._hl_grad_steps = int(info.get('policy_updates', self._hl_grad_steps + ns))
        out = {k: float(v) for k, v in info.items()}
        out.update(
            n_samples=float(len(records)),
            n_distinct_samples=float(len({r.get('sample_id') for r in records if r.get('sample_id')})),
            n_online=float(sum(1 for r in records if r['pool'] == 'online')),
            updates_applied=float(self._hl_updates_applied),
            update_calls=float(self._hl_update_calls),
            roundtrip_seconds=float(time.time() - started),
        )
        out.update(reuse_info)
        print(f"[mixed-steervla.update_hl] loss={out['loss']:.4f} grad_norm={out['grad_norm']:.3f} "
              f"policy_updates={self._hl_grad_steps} bs={len(records)} (online {int(out['n_online'])}) "
              f"tokens={int(out['supervised_tokens'])} {out['update_seconds']:.1f}s "
              f"peak={out['hl_mem_peak_gb']:.1f}GB", flush=True)
        return out

    def save_checkpoint(self, out_root, step: int, *, keep_last: int = 0):
        """Export the worker's HL as ``<out_root>/<step>/pytorch_model.bin`` + ``.hydra/config.yaml``."""
        if not self.load_trainable_params:
            print('[mixed-steervla.save_checkpoint] skipped: HL not loaded trainable.', flush=True)
            return None
        self._hl_conn.send(dict(op='save', out_root=str(out_root), step=int(step)))
        step_dir = Path(self._receive_hl(1800)['path'])
        print(f'[mixed-steervla.save_checkpoint] wrote HL checkpoint -> {step_dir}', flush=True)
        if int(keep_last) > 0:
            self._prune_checkpoints(out_root, keep_last=int(keep_last))
        return step_dir

    def _sample_cot_checked(self, rng, obs_jax):
        if obs_jax.tokenized_prompt.shape[0] != 1:
            raise ValueError('Mixed HL evaluation expects one live scene at a time.')
        raw = self.raw_obs_holder['obs']
        state = np.asarray(raw['state']).reshape(-1)
        history = ''
        if self._hl_use_history and self._hl_history:
            picked = [self._hl_history[-(1 + i * 40)] if len(self._hl_history) >= 1 + i * 40
                      else self._hl_history[0] for i in range(self._hl_history_count, 0, -1)]
            speeds = ' '.join(f'{round(s, 1)} m/s' for s, _ in picked)
            headings = ' '.join(f'{round(h, 1)} degrees' for _, h in picked)
            history = f'Speed history: {speeds} Heading history: {headings}\n'
        prompt = (f'{history}Current speed: {round(float(state[15]), 1)} m/s\n'
                  f"Command: {raw.get('routing_command') or self.routing_command}")
        # The native camera is for InternVL2 only. The superclass still constructs
        # pi05's Observation from raw['image'] and encodes it with pi05's backbone.
        # Sample the HL at the actor's CoT temperature (0 keeps the original greedy HL). The seed
        # comes from this call's JAX key, so draws are independent yet reproducible.
        seed = int(jax.random.randint(rng, (), 0, np.iinfo(np.int32).max))
        self._hl_conn.send(dict(op='generate', image=np.asarray(raw['image_viz'], dtype=np.uint8),
                                prompt=prompt, temperature=float(self.cot_temperature), seed=seed))
        result = self._receive_hl()
        raw['simlingo_hl_output'] = result['output']
        raw['simlingo_hl_prompt'] = prompt
        print(f"[mixed-steervla] InternVL2 subtask: {result['subtask']}", flush=True)
        out = {}
        for segment in ('reasoning', 'subtask'):
            tokens, mask = getattr(self.tokenizer, f'tokenize_{segment}')(result[segment])
            out[f'tokenized_{segment}'] = jax.device_put(
                jnp.asarray(tokens[None], dtype=obs_jax.tokenized_prompt.dtype), self._jax_device)
            out[f'tokenized_{segment}_mask'] = jax.device_put(
                jnp.asarray(mask[None], dtype=bool), self._jax_device)
        if self.tokenizer.use_fast_tokens:
            # FAST is an auxiliary autoregressive action prediction. This model's
            # action expert explicitly masks FAST columns; external-HL inference
            # uses the same empty FAST segment as SteerVLA's fixed-CoT path.
            shape = (1, int(self.model_cfg.max_fast_len))
            out['tokenized_fast'] = jax.device_put(
                jnp.zeros(shape, dtype=obs_jax.tokenized_prompt.dtype), self._jax_device)
            out['tokenized_fast_mask'] = jax.device_put(jnp.zeros(shape, dtype=bool), self._jax_device)
        return out
