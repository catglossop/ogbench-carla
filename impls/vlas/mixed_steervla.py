"""InternVL2 text planner + pi05's own vision backbone and action expert."""
from __future__ import annotations

import atexit
from collections import deque
from multiprocessing.connection import Connection
import os
from pathlib import Path
import socket
import subprocess

import jax
import jax.numpy as jnp
import numpy as np

from vlas.steervla import SteerVLAActor


class MixedSteerVLAActor(SteerVLAActor):
    def __init__(self, *, hl_checkpoint, simlingo_source_root, hl_python, **kwargs):
        if kwargs.get('load_trainable_params') or kwargs.get('fixed_subtask_text'):
            raise ValueError('Mixed SteerVLA is an inference-only, live-HL actor.')
        self._hl_history = deque(maxlen=402)
        super().__init__(**kwargs)
        parent, child = socket.socketpair()
        self._hl_conn = Connection(parent.detach())
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(kwargs['training_gpu_rank'])
        env.pop('PYTHONPATH', None)
        self._hl_proc = subprocess.Popen([
            hl_python, '-u', str(Path(__file__).with_name('internvl2_hl_worker.py')),
            '--fd', str(child.fileno()), '--checkpoint', hl_checkpoint,
            '--source-root', simlingo_source_root,
        ], pass_fds=(child.fileno(),), env=env)
        child.close()
        atexit.register(self.close)
        ready = self._receive_hl(600)
        self._hl_use_history = ready['use_ego_history']
        self._hl_history_count = ready['history_count']
        self._hl_history = deque(maxlen=40 * self._hl_history_count + 2)
        print(f"[mixed-steervla] HL={ready['weights']} LL={kwargs['checkpoint_path']} "
              'pi05 vision preserved; InternVL2 text conditioning only', flush=True)

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
                                temperature=float(self.cot_temperature), seed=seed))
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
