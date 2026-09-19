"""Local-only PDM observer records. These never enter critic inputs or controls."""
import json
from pathlib import Path
import numpy as np


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [plain(v) for v in value]
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def plan_record(*, data, control, sim_time, plan_time, error, ego_matrix, route_index):
    valid = isinstance(data, dict) and len(data.get('route', [])) > 0 and error is None and plan_time == sim_time
    return plain(dict(
        schema='pdm_plan_observer_v1', valid=valid, error=error,
        sim_time=sim_time, plan_time=plan_time, ego_matrix=ego_matrix,
        planner_route_index=route_index, control=control, driving_data=data,
        route_coordinates='ego-local meters, x forward/y right',
        semantics='PDM-lite route, target speed and instantaneous planned control from the current state; not a counterfactual simulator rollout and not a 40D expert ground-truth action',
    ))


def save_query_comparison(save_dir, step, plan, frame, candidates, subtasks, routing_command, response, state, cadence):
    from PIL import Image
    if plan is None:
        raise RuntimeError('Expert plan missing from observation transport')
    action = np.asarray(candidates)
    if action.ndim != 2 or action.shape[1] != 40:
        raise ValueError('Expected all complete 40D candidate actions')
    root = Path(save_dir) / 'pdm-query-comparisons'; root.mkdir(parents=True, exist_ok=True)
    image_path = root / f'step-{step:06d}.jpg'
    Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(image_path, quality=95)
    record = plain(dict(step=step, pdm_plan=plan, image=str(image_path),
        state=state, candidates=action, subtasks=subtasks, routing_command=routing_command,
        action_representation='native_delta_xy_t_delta_xy_space', cadence=cadence,
        response=response))
    with (root / f'step-{step:06d}.json').open('x') as f:
        json.dump(record, f, allow_nan=False)
