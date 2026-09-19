"""CPU checks for LL padding isolation and candidate order, without checkpoints."""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def test_ll_groups_equal_lengths_and_restores_candidate_order():
    source = ast.parse(Path(__file__).with_name('simlingo_steervla.py').read_text())
    cls = next(x for x in source.body if isinstance(x, ast.ClassDef) and x.name == '_SimLingoModel')
    method = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == 'generate_batch')
    ns = {'torch': torch, 'np': np}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), '<batch-test>', 'exec'), ns)
    groups = []
    def labels(prompts, patches):
        lengths = [len(p) for p in prompts]
        return SimpleNamespace(prompts=prompts, phrase_valid=torch.tensor([
            [True] * n + [False] * (max(lengths)-n) for n in lengths]))
    class Model:
        model_type = 'll'
        def __call__(self, inputs):
            prompts = inputs.prompts
            assert len({len(p) for p in prompts}) == 1
            groups.append(prompts)
            values = torch.tensor([ord(p[0]) for p in prompts])[:, None, None].expand(-1, 10, 2)
            return values, values + 1, []
    actor = SimpleNamespace(model=Model(), pixel_tiles=lambda image: torch.zeros(2, 3, 4, 4),
                            question_label=labels, driving_input=lambda tiles, label: label)
    (speed, route, _), overflow = ns['generate_batch'](actor, np.zeros((4, 4, 3)), ['aaa', 'b', 'ccc', 'dd'])
    assert groups == [['aaa', 'ccc'], ['b'], ['dd']]
    assert speed[:, 0, 0].tolist() == [97, 98, 99, 100]
    assert torch.equal(route, speed + 1)
    assert overflow is None
