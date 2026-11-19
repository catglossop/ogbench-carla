"""InternVL2 text planner + pi05's own vision backbone and action expert."""
from __future__ import annotations

import atexit
from collections import deque
import json
from multiprocessing.connection import Connection
import os
from pathlib import Path
import socket
import subprocess

import jax
import jax.numpy as jnp
import numpy as np

from vlas.steervla import SteerVLAActor


DRIVING_BEHAVIOR_MARKER = 'Driving Behavior:'


class MixedSteerVLAActor(SteerVLAActor):
    """pi05 (frozen) executes; InternVL2 plans in a torch worker process.

    ``hl_train_config`` (the ``steervla.hl_*`` knobs, from ``steervla.hl_trainable=True``) makes the
    *InternVL2 HL* trainable inside the worker -- pi05 stays frozen and inference-only. The actor then
    reports ``load_trainable_params=True`` to main_carla, takes the YAY-Robot ``cot_intervention_fn``
    exactly like ``SimLingoSteerVLAActor``, and forwards ``update_hl`` / ``save_checkpoint`` to the worker,
    which runs ``SimLingoSteerVLAActor``'s own HL update / export.
    """

    def __init__(self, *, hl_checkpoint, simlingo_source_root, hl_python, hl_train_config=None, **kwargs):
        if kwargs.get('fixed_subtask_text'):
            raise ValueError('Mixed SteerVLA is a live-HL actor.')
        if kwargs.get('load_trainable_params'):
            if hl_train_config is None:
                raise ValueError('Mixed SteerVLA keeps pi05 inference-only; train the HL with steervla.hl_trainable=True.')
            kwargs['load_trainable_params'] = False
        self._hl_trainable = hl_train_config is not None
        self._hl_history = deque(maxlen=402)
        self._candidate_scene_id = 0
        super().__init__(**kwargs)
        parent, child = socket.socketpair()
        self._hl_conn = Connection(parent.detach())
        env = os.environ.copy()
        # training_gpu_rank indexes the devices THIS process sees. If the parent was already pinned
        # (CUDA_VISIBLE_DEVICES=2 with --train-gpu 0), translate the rank to the physical id so the
        # worker lands on the same GPU instead of physical GPU <rank>.
        rank = int(kwargs['training_gpu_rank'])
        parent_visible = [d for d in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if d.strip()]
        env['CUDA_VISIBLE_DEVICES'] = parent_visible[rank].strip() if 0 <= rank < len(parent_visible) else str(rank)
        env.pop('PYTHONPATH', None)
        cmd = [
            hl_python, '-u', str(Path(__file__).with_name('internvl2_hl_worker.py')),
            '--fd', str(child.fileno()), '--checkpoint', hl_checkpoint,
            '--source-root', simlingo_source_root,
        ]
        if self._hl_trainable:
            cmd += ['--train-config', json.dumps(dict(hl_train_config), default=str)]
            env['JAX_PLATFORMS'] = 'cpu'  # the worker imports vlas.steervla for the HL trainer only
        self._hl_proc = subprocess.Popen(cmd, pass_fds=(child.fileno(),), env=env)
        child.close()
        atexit.register(self.close)
        ready = self._receive_hl(600)
        self._hl_use_history = ready['use_ego_history']
        self._hl_history_count = ready['history_count']
        self._hl_history = deque(maxlen=40 * self._hl_history_count + 2)
        print(f"[mixed-steervla] HL={ready['weights']} LL={kwargs['checkpoint_path']} "
              'pi05 vision preserved; InternVL2 text conditioning only', flush=True)

        # YAY-Robot foresight hook (assigned by main_carla), as on SimLingoSteerVLAActor.
        self.cot_intervention_fn = None
        self._cot_interventions = 0
        self._last_cot_intervention = None
        self._hl_text = None
        if self._hl_trainable:
            if not ready.get('trainable'):
                raise RuntimeError('InternVL2 worker did not come up trainable.')
            cfg = dict(hl_train_config)
            # main_carla's "is the HL being trained" gate (dataset wiring, checkpoint cadence).
            self.load_trainable_params = True
            self.hl_dataset_dir = None
            self.hl_update_every = max(1, int(cfg.get('hl_update_every', 1)))
            self.hl_update_batch_size = max(1, int(cfg.get('hl_update_batch_size', 16)))
            self._hl_grad_steps = self._hl_updates_applied = self._hl_update_calls = 0
            print(f'[mixed-steervla] InternVL2 HL trainable in worker: {cfg}', flush=True)

    def update_hl(self, *, batch_size=None, num_steps=None, rng=None, global_step=None):
        del batch_size, num_steps, rng
        if not self._hl_trainable:
            return {}
        self._hl_conn.send(dict(cmd='update_hl', global_step=global_step,
                                hl_dataset_dir=None if self.hl_dataset_dir is None else str(self.hl_dataset_dir)))
        info = self._receive_hl(3600)['info']
        # main_carla's stop rules (--max_hl_updates, DS-streak target) read these off the actor.
        if 'policy_updates' in info:
            self._hl_grad_steps = int(info['policy_updates'])
        if 'updates_applied' in info:
            self._hl_updates_applied = int(info['updates_applied'])
        if 'update_calls' in info:
            self._hl_update_calls = int(info['update_calls'])
        return info

    def save_checkpoint(self, out_root, step, *, keep_last=0):
        if not self._hl_trainable:
            print('[mixed-steervla.save_checkpoint] skipped: HL not trainable.', flush=True)
            return None
        self._hl_conn.send(dict(cmd='save_checkpoint', out_root=str(out_root), step=int(step), keep_last=int(keep_last)))
        path = self._receive_hl(3600)['path']
        return None if path is None else Path(path)

    def _maybe_intervene(self, result, raw, prompt):
        """``SimLingoSteerVLAActor._maybe_intervene_on_cot`` for the InternVL2 text."""
        hook = self.cot_intervention_fn
        self._last_cot_intervention = None
        if hook is None:
            return result
        try:
            state = np.asarray(raw.get('state'), dtype=np.float32).reshape(-1)
            corrected = hook({
                'subtask': result['subtask'],
                'reasoning': result['reasoning'],
                'image': np.asarray(raw['image_viz'], dtype=np.uint8),
                'state': state,
                'current_speed': float(state[15]) if state.size > 15 else 0.0,
                'routing_command': self.routing_command,
                'prompt': prompt,
            })
        except Exception as exc:  # never let a coach failure stop the car
            print(f'[mixed-steervla] cot intervention failed ({exc}); keeping sampled CoT.', flush=True)
            return result
        if not corrected:
            return result
        subtask, reasoning = (str(v).strip() for v in corrected)
        if not subtask or not reasoning:
            return result
        self._cot_interventions += 1
        self._last_cot_intervention = dict(original_subtask=result['subtask'], original_reasoning=result['reasoning'],
                                           subtask=subtask, reasoning=reasoning)
        return {**result, 'subtask': subtask, 'reasoning': reasoning,
                'output': f'{reasoning}\n\n{DRIVING_BEHAVIOR_MARKER} {subtask}'}

    def _stash_hl_text(self, raw):
        """Put the full InternVL2 text where YAY-Robot reads it (as SimLingoSteerVLAActor._stash_cot_in_raw)."""
        if raw is None or self._hl_text is None:
            return
        raw['subtask_text'] = self._hl_text['subtask']
        raw['reasoning_text'] = self._hl_text['reasoning']
        raw['openpi_prompt_raw_text'] = self._hl_text['prompt']  # what update_hl trains on
        raw['simlingo_hl_output'] = self._hl_text['output']
        if self._last_cot_intervention is not None:
            raw['yay_intervention'] = True
            raw['yay_original_subtask'] = self._last_cot_intervention['original_subtask']
            raw['yay_original_reasoning'] = self._last_cot_intervention['original_reasoning']
        else:
            raw['yay_intervention'] = False

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
        out = super().__call__(observations_jax, noise_jax)
        if self._hl_trainable:
            self._stash_hl_text(raw)
        return out

    def reset_action_cache(self):
        super().reset_action_cache()
        self._hl_history.clear()
        self._hl_text = None
        self._last_cot_intervention = None

    def begin_candidate_scene(self):
        self._candidate_scene_id += 1

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
        # Sample the HL at the actor's CoT temperature (Best-of-N sets 1.0 so each candidate slot gets
        # its own subtask; 0 keeps the original greedy HL). The seed comes from this call's JAX key, so
        # every sequential BoN slot draws independently yet reproducibly.
        seed = int(jax.random.randint(rng, (), 0, np.iinfo(np.int32).max))
        self._hl_conn.send(dict(image=np.asarray(raw['image_viz'], dtype=np.uint8), prompt=prompt,
                                temperature=float(self.cot_temperature), seed=seed,
                                scene_id=self._candidate_scene_id))
        result = self._receive_hl()
        result = self._maybe_intervene(result, raw, prompt)
        self._hl_text = dict(prompt=prompt, reasoning=result['reasoning'], subtask=result['subtask'],
                             output=result['output'])
        self.last_hl_vision_cache_hits = result.get('vision_cache_hits', 0)
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
