"""Isolated torch HL inference. Communication uses an inherited private socket."""
from __future__ import annotations

import argparse
from multiprocessing.connection import Connection
from pathlib import Path
import traceback

import torch

from simlingo_model import _SimLingoModel, _import_simlingo, split_hl_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fd', type=int, required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--source-root', required=True)
    args = parser.parse_args()
    conn = Connection(args.fd)
    try:
        root = Path(args.source_root)
        _import_simlingo(root)
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
        conn.send(dict(ready=True, weights=str(hl.weights),
                       use_ego_history=bool(hl.dataset_cfg.get('use_ego_state_history', False)),
                       history_count=int(hl.dataset_cfg.get('ego_state_history_count', 3))))
        while True:
            request = conn.recv()
            if request is None:
                break
            sampling['temperature'] = float(request.get('temperature', 0.0))
            if request.get('seed') is not None:
                torch.manual_seed(int(request['seed']))
            # num_samples > 1 draws that many candidates for one scene in a single batched pass.
            # Absent (or 1) this is the original one-request-one-sample path, unchanged.
            num_samples = int(request.get('num_samples', 1) or 1)
            if num_samples > 1:
                _, _, language = hl.generate_batch(request['image'], request['prompt'], num_samples)
            else:
                _, _, language = hl.generate(request['image'], request['prompt'])
            outputs = [str(text) for text in (language or [])]
            if len(outputs) != num_samples:
                raise ValueError(f'InternVL2 returned {len(outputs)} samples, expected {num_samples}')
            samples = []
            for output in outputs:
                reasoning, subtask = split_hl_output(output)
                if not subtask.strip():
                    raise ValueError(f'InternVL2 returned an empty subtask: {output!r}')
                samples.append(dict(reasoning=reasoning, subtask=subtask, output=output))
            if num_samples > 1:
                conn.send(dict(samples=samples))
            else:
                conn.send(samples[0])
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
