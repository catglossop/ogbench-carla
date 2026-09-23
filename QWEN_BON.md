# Qwen BoN launch

Run the critic service from `qwen-critic` (commit `076f366` or later).
The new six-term critic adds correctness; scoring weights and thresholds belong
to the service. The existing ogbench HTTP client forwards its selected candidate
and retains all returned scores, including correctness.

Set `ADAPTER` to the trained checkpoint's `adapter` directory. These weights are
an illustrative starting point, **not a validated default**; use the settings
from the checkpoint comparison when available.

```bash
cd /home/celine/qwen-critic
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/serve_bon.py \
  --model Qwen/Qwen3.8-27B --adapter "$ADAPTER" --port 18834 \
  --prompt-profile scene_categories_v1 \
  --action-representation native_delta_xy_t_delta_xy_space \
  --no-include-current-speed --context-prefix '' \
  --risk-threshold 1.01 --crash-threshold 1.01 \
  --offroad-threshold 1.01 --traffic-threshold 1.01 \
  --risk-weight 1 --crash-weight 1 --offroad-weight 1 --traffic-weight 1 \
  --goal-weight 1 --progress-weight 1 --correctness-weight 1
```

The service enables `torch.compile` by default; `--no-torch-compile` selects
eager inference. Compilation has a warm-up cost. Projection, numerical action
formatting, and fresh scene descriptions are handled by the critic service.

In a second shell, set `ROUTE` to an evaluation route ID and launch:

```bash
cd /home/celine/ogbench-carla
CUDA_VISIBLE_DEVICES=4 UV_NO_SYNC=1 CARLA_0915_ROOT=/raid/users/celine/f2d_carla \
./run_carla.sh \
  --route "$ROUTE" --carla-config impls/configs/carla_config.yaml \
  --online-steps 6000 --seed 0 --run-group Debug --save-buffer false \
  --wandb-mode online --train-gpu 0 --render-adapter 4 \
  --carla-port 15580 --carla-streaming-port 15581 --tm-port 15680 \
  --x-display-num 831 --critic-mode none --train-mode rl \
  --enable-updates false --base-only true --max-retries 0 \
  --steervla-checkpoint /raid/users/cglossop/steervla_pi_ckpts/ll_heavy_unnormed_matchcrop/6000 \
  --actor-config pi05_steervla_cot_simplified_reasoning_no_ego_history \
  --bon-num-candidates 8 --bon-max-sample-attempts 1 \
  --bon-batch-policy-candidates true --bon-cot-temperature 1.0 \
  --bon-candidates-log-every 20 --bon-candidates-wandb false \
  --bon-qwen-select true --qwen-bon-url http://127.0.0.1:18834 \
  --bon-qwen-cadence 3 --qwen-online-train false --max-episodes 1 \
  --terminate-on-collision false --save-video-local true --eval-only true -- \
  --bon_qwen_label_source=subtask --bon_include_brake_candidate=false \
  --agent.steervla.actions_per_cot=5 \
  --agent.steervla.actions_per_model_query=3 \
  --agent.steervla.proprio_norm=false
```

The actor is the `ll_heavy_unnormed_matchcrop/.../6000` checkpoint. The CARLA
config caps episodes at 4000 steps. `--train-gpu 0` is logical device 0 inside
`CUDA_VISIBLE_DEVICES=4`; the renderer uses physical adapter 4.

Every candidate contains the full 10×4 action chunk. Only the cadence-sized
prefix is executed. Batched candidate text is reused according to executed
environment steps and refreshed at the next query once its age reaches
`actions_per_cot`; action chunks are freshly sampled at every query.

Use `--bon_qwen_label_source=subtask` for the current refined-label critic.
For a critic trained on raw actor commentary, select `commentary` instead.
Actor segment markers are removed without substituting one text head for the
other. Videos are logged to W&B and saved locally; candidate images are not
uploaded. No online critic updates or artificial brake candidate are enabled.
