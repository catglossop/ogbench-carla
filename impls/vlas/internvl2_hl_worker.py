"""Isolated torch HL inference and (optionally) online HL training.

Communication uses an inherited private socket. Requests are dicts tagged by ``op``; a request with
no ``op`` is a generate call, which is what the inference-only mixed actor has always sent.

``--trainable`` additionally builds the HL optimizer in *this* process, because the HL model lives
here and nowhere else: the mixed actor's own ``update_hl`` would otherwise train pi05's JAX state
(the frozen low level), not InternVL2. The recipe is the one
``vlas/simlingo_steervla.py`` uses for the in-process SimLingo HL -- same trainable set (the
checkpoint's own ``requires_grad`` flags: LoRA + vision tower), bf16 compute with fp32 master
weights, AdamW, per-record loss scope, and the k3 KL penalty against a frozen snapshot -- so a run
here and a SimLingo-stack run train the same parameters the same way.

The actor ships record *paths* rather than pixels: a 64-record batch of 1024x512 frames is ~100 MB,
and both processes can read the same disk.
"""
from __future__ import annotations

import argparse
from multiprocessing.connection import Connection
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch

from simlingo_model import DRIVING_BEHAVIOR_MARKER, _SimLingoModel, _import_simlingo, split_hl_output


def _load_image(path: str) -> np.ndarray | None:
    """Decode one HL sample frame; ``None`` for a half-written or unreadable file."""
    p = Path(path)
    try:
        if p.suffix.lower() in ('.jpg', '.jpeg', '.png'):
            import cv2

            bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if bgr is None:
                return None
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        with np.load(p) as z:
            return np.asarray(z['image'], dtype=np.uint8)
    except Exception:
        return None  # cast_relabel writes concurrently; a torn .npz is retried on the next update.


class _HLTrainer:
    """Online HL updates, mirroring ``SimLingoSteerVLAActor``'s in-process trainer."""

    def __init__(self, hl: _SimLingoModel, *, lr: float, weight_decay: float, grad_clip: float,
                 kl_coef: float, micro_batch_size: int):
        self.hl = hl
        self.lr = float(lr)
        self.grad_clip = float(grad_clip)
        self.kl_coef = float(kl_coef)
        self.micro_batch_size = max(1, int(micro_batch_size))
        named = dict(hl.model.named_parameters())
        self.params = [named[n] for n in hl.trainable_names]
        for p in hl.model.parameters():
            p.requires_grad_(False)
        for p in self.params:
            p.requires_grad_(True)
        # bf16 compute, fp32 master weights + AdamW state (SimLingo trained 16-mixed).
        self.master = [p.detach().float().clone().requires_grad_(True) for p in self.params]
        self.optimizer = torch.optim.AdamW(self.master, lr=self.lr, weight_decay=float(weight_decay),
                                           betas=(0.9, 0.999))
        self.ref_model = None
        if self.kl_coef > 0.0:
            import copy

            # Frozen snapshot of the starting HL: the reference policy for KL(pi_theta || pi_ref).
            self.ref_model = copy.deepcopy(hl.model).eval().requires_grad_(False)
        self.grad_steps = 0
        n = sum(p.numel() for p in self.params)
        print(f'[internvl2-hl] trainable: {len(self.params)} tensors / {n / 1e6:.1f}M params on '
              f'{hl.device} (lr={self.lr}, wd={weight_decay}, kl={self.kl_coef})', flush=True)

    @staticmethod
    def _answer(record: dict) -> str:
        reasoning = str(record.get('reasoning') or '').strip()
        subtask = str(record['subtask']).strip()
        return (f'{reasoning}\n\n{DRIVING_BEHAVIOR_MARKER} {subtask}' if reasoning
                else f'{DRIVING_BEHAVIOR_MARKER} {subtask}')

    def _build_example(self, records: list[dict]):
        """SimLingo DrivingExample with the training chat template and per-record loss scope."""
        from simlingo_training.utils.custom_types import DrivingExample, DrivingLabel, LanguageLabel
        from simlingo_training.utils.internvl2_utils import get_custom_chat_template

        m = self.hl
        tiles = [m.pixel_tiles(r['image']) for r in records]
        num_patches = int(tiles[0].shape[0])
        if any(int(t.shape[0]) != num_patches for t in tiles):
            raise ValueError('HL batch mixes image aspect ratios (different tile counts); frames must share a size.')
        pixels = torch.stack(tiles)[:, None]  # (B, 1, patches, 3, S, S)
        conversations = [
            [
                {'role': 'user', 'content': [{'type': 'text', 'text': r['prompt']}]},
                {'role': 'assistant', 'content': [{'type': 'text', 'text': self._answer(r)}]},
            ]
            for r in records
        ]
        conv, question = get_custom_chat_template(
            conversations, m.tokenizer, m.variant, m.num_image_token * num_patches,
            cache_root_dir=str(m.cache_root),
        )
        loss_mask = conv['loss_masking'].clone()
        if any(r['loss_scope'] == 'subtask' for r in records):
            enc = m.tokenizer(
                conv['language_string'], padding=True, return_tensors='pt',
                add_special_tokens=False, return_offsets_mapping=True,
            )
            if not torch.equal(enc['input_ids'], conv['phrase_ids']):
                raise RuntimeError('re-tokenization with offsets disagrees with the chat-template tokenization')
            for i, r in enumerate(records):
                if r['loss_scope'] != 'subtask':
                    continue
                text = conv['language_string'][i]
                start = text.rfind(DRIVING_BEHAVIOR_MARKER)
                if start < 0:
                    loss_mask[i] = False
                    continue
                start += len(DRIVING_BEHAVIOR_MARKER)
                loss_mask[i] &= enc['offset_mapping'][i, :, 0] >= start

        def label(d, mask):
            return LanguageLabel(
                phrase_ids=d['phrase_ids'].to(m.device),
                phrase_valid=d['phrase_valid'].to(m.device),
                phrase_mask=d['phrase_mask'].to(m.device),
                placeholder_values=[{} for _ in records],
                language_string=d['language_string'],
                loss_masking=None if mask is None else mask.to(m.device),
            )

        b = len(records)
        # An HL-only checkpoint carries no driving adaptor (the LL owns waypoints), so read the
        # width when it is there and fall back to SimLingo's constant otherwise. These tensors are
        # dummies: DrivingLabel wants them, the language loss never reads them.
        driving_adaptor = getattr(getattr(m.model, 'adaptors', None), 'driving', None)
        speed_waypoint_count = int(getattr(driving_adaptor, 'future_speed_waypoints', 10) or 10)
        example = DrivingExample(
            driving_input=m.driving_input(pixels, label(conv, loss_mask), label(question, None)),
            driving_label=DrivingLabel(
                waypoints=torch.zeros(b, speed_waypoint_count, 2, device=m.device),
                path=torch.zeros(b, 20, 2, device=m.device),
                answer=LanguageLabel(None, None, None, None, [self._answer(r) for r in records], None),
                image_ff_org=torch.zeros(b, 1, device=m.device),
                eval_infos=None,
            ),
            run_id=None,
        )
        # Next-token targets: token t+1 is supervised where the mask is set.
        return example, int(loss_mask[:, 1:].sum())

    def train(self, records: list[dict], num_steps: int) -> dict:
        """``num_steps`` gradient steps on one batch; returns the metrics the actor logs."""
        loaded = []
        for r in records:
            image = _load_image(r['image_path'])
            if image is not None:
                loaded.append(dict(r, image=image))
        if not loaded:
            return dict(skipped='every record in the HL batch failed to decode')
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats(self.hl.device)
        model = self.hl.model
        model.train()
        micro = [loaded[i:i + self.micro_batch_size] for i in range(0, len(loaded), self.micro_batch_size)]
        losses, grad_norm, total_tokens = [], 0.0, 0
        kl_info: dict[str, float] = {}
        try:
            for _ in range(max(1, int(num_steps))):
                built = [self._build_example(chunk) for chunk in micro]
                total_tokens = sum(n for _, n in built)
                if total_tokens == 0:
                    return dict(skipped='HL batch has no supervised tokens (empty subtasks after masking)')
                step_loss = 0.0
                n_rows = sum(int(ex.driving_input.camera_images.shape[0]) for ex, _ in built)
                kl_sums = {'kl_to_ref': 0.0, 'ce_theta': 0.0, 'ce_ref': 0.0, 'log_ratio': 0.0}
                for example, _ in built:
                    loss_dict, _ = model.forward_loss(example, per_sample=True)
                    token_loss, token_mask = loss_dict['language_loss']
                    token_loss, token_mask = token_loss.float(), token_mask.float()
                    micro_sum = (token_loss * token_mask).sum()
                    objective = micro_sum / total_tokens
                    if self.ref_model is not None:
                        # k3 KL(pi_theta || pi_ref) on the per-example mean CE of the supervised
                        # tokens. The reference is forward-only, so gradient flows through ce_theta.
                        n_tok = token_mask.sum(-1).clamp(min=1.0)
                        ce_theta = (token_loss * token_mask).sum(-1) / n_tok
                        with torch.no_grad():
                            ref_dict, _ = self.ref_model.forward_loss(example, per_sample=True)
                            ce_ref = (ref_dict['language_loss'][0].float() * token_mask).sum(-1) / n_tok
                        log_ratio = ce_ref - ce_theta
                        kl = torch.exp(-log_ratio) + log_ratio - 1.0
                        weight = float(ce_theta.shape[0]) / n_rows
                        objective = objective + self.kl_coef * kl.mean() * weight
                        for key, value in (('kl_to_ref', kl), ('ce_theta', ce_theta),
                                           ('ce_ref', ce_ref), ('log_ratio', log_ratio)):
                            kl_sums[key] += float(value.detach().mean()) * weight
                    objective.backward()
                    step_loss += float(micro_sum.detach()) / total_tokens
                if self.ref_model is not None:
                    kl_info = dict(kl_sums, kl_penalty=self.kl_coef * kl_sums['kl_to_ref'])
                for p, master in zip(self.params, self.master):
                    master.grad = None if p.grad is None else p.grad.detach().float()
                    p.grad = None
                grad_norm = float(torch.nn.utils.clip_grad_norm_(self.master, self.grad_clip))
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    for p, master in zip(self.params, self.master):
                        p.copy_(master.to(p.dtype))
                losses.append(step_loss)
                self.grad_steps += 1
        finally:
            model.eval()
        out = dict(
            loss=float(np.mean(losses)),
            grad_norm=grad_norm,
            supervised_tokens=float(total_tokens),
            lr=float(self.lr),
            n_decoded=float(len(loaded)),
            policy_updates=float(self.grad_steps),
            update_seconds=float(time.time() - t0),
            hl_mem_peak_gb=float(torch.cuda.max_memory_allocated(self.hl.device)) / 1e9,
        )
        out.update(kl_info)
        return out

    def save(self, out_root: str, step: int) -> str:
        """``<out_root>/<step>/pytorch_model.bin`` + ``.hydra/config.yaml`` -- the layout the mixed
        eval config and simlingo's ``agent_steervla.py`` both load."""
        step_dir = Path(out_root) / str(int(step))
        (step_dir / '.hydra').mkdir(parents=True, exist_ok=True)
        state = {k: v.detach().cpu() for k, v in self.hl.model.state_dict().items()}
        torch.save(state, step_dir / 'pytorch_model.bin')
        shutil.copyfile(self.hl.config_path, step_dir / '.hydra' / 'config.yaml')
        return str(step_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fd', type=int, required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--trainable', action='store_true', help='build the HL optimizer in this process')
    parser.add_argument('--hl-lr', type=float, default=1e-5)
    parser.add_argument('--hl-weight-decay', type=float, default=0.1)
    parser.add_argument('--hl-grad-clip', type=float, default=1.0)
    parser.add_argument('--hl-kl-coef', type=float, default=0.0)
    parser.add_argument('--hl-micro-batch-size', type=int, default=8)
    args = parser.parse_args()
    conn = Connection(args.fd)
    try:
        root = Path(args.source_root)
        _import_simlingo(root)
        hl = _SimLingoModel(args.checkpoint, source_root=root, device=torch.device('cuda:0'), expect_type='hl')
        trainer = None
        if args.trainable:
            trainer = _HLTrainer(hl, lr=args.hl_lr, weight_decay=args.hl_weight_decay,
                                 grad_clip=args.hl_grad_clip, kl_coef=args.hl_kl_coef,
                                 micro_batch_size=args.hl_micro_batch_size)
        else:
            hl.model.requires_grad_(False)
        if hl.dataset_cfg.get('use_history_image', False):
            raise ValueError('History images are not supported by this HL bridge.')
        original_sample = hl.model.language_model.greedy_sample
        # Per-request HL sampling temperature (0 = greedy, the default). A per-request seed keeps
        # those draws reproducible.
        sampling = dict(temperature=0.0)

        def sample(*args, **kwargs):
            kwargs['temperature'] = sampling['temperature']
            return original_sample(*args, **kwargs)

        hl.model.language_model.greedy_sample = sample
        conn.send(dict(ready=True, weights=str(hl.weights),
                       use_ego_history=bool(hl.dataset_cfg.get('use_ego_state_history', False)),
                       history_count=int(hl.dataset_cfg.get('ego_state_history_count', 3)),
                       trainable=bool(trainer is not None)))
        while True:
            request = conn.recv()
            if request is None:
                break
            op = str(request.get('op') or 'generate')
            if op == 'train':
                if trainer is None:
                    raise RuntimeError('HL worker was not started with --trainable')
                conn.send(trainer.train(request['records'], int(request['num_steps'])))
                continue
            if op == 'save':
                if trainer is None:
                    raise RuntimeError('HL worker was not started with --trainable')
                conn.send(dict(path=trainer.save(request['out_root'], int(request['step']))))
                continue
            if op != 'generate':
                raise ValueError(f'unknown HL worker op: {op!r}')
            sampling['temperature'] = float(request.get('temperature', 0.0))
            if request.get('seed') is not None:
                torch.manual_seed(int(request['seed']))
            _, _, language = hl.generate(request['image'], request['prompt'])
            output = str(language[0] if language else '')
            reasoning, subtask = split_hl_output(output)
            if not subtask.strip():
                raise ValueError(f'InternVL2 returned an empty subtask: {output!r}')
            conn.send(dict(reasoning=reasoning, subtask=subtask, output=output))
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
