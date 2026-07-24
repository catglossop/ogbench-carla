#!/usr/bin/env bash
# run_simlingo_fail2drive.sh — launcher for SimLingo residual SAC on Fail2Drive routes.
#
# This is a drop-in mirror of run_simlingo.sh, but targets Fail2Drive routes
# instead of Bench2Drive routes.  All flags are identical; only the default
# --route changes and the FAIL2DRIVE_CARLA_ROOT / --fail2drive-carla-root
# option is added.
#
# Fail2Drive route naming:
#   Routes can be specified in any of three forms:
#     f2d:<id>                    e.g. --route f2d:85
#     kebab scenario name         e.g. --route base-pedestrians-on-road-0085
#     original filename           e.g. --route Base_PedestriansOnRoad_0085
#
# Install the fail2drive Python package first (provides route XMLs + scenario classes):
#   pip install git+https://github.com/catglossop/fail2drive.git
#
# Fail2Drive asset packs (brickwall, walkingkid, ampel, etc.) must be installed
# into your CARLA 0.9.16 install:
#   bash install_f2d_content.sh /home/carla/carla-0-9-16 /tmp/f2d_content.zip
#
# Quick examples:
#
#   # Base-policy eval only — check SimLingo on a Fail2Drive route
#   bash run_simlingo_fail2drive.sh --eval-only --route base-pedestrians-on-road-0085
#
#   # SAC training on a Fail2Drive route
#   bash run_simlingo_fail2drive.sh --route f2d:42 --steps 50000
#
#   # Second instance on different ports
#   bash run_simlingo_fail2drive.sh --instance 1 --route generalization-animals-1075
#
#   # Offline W&B
#   WANDB_MODE=offline bash run_simlingo_fail2drive.sh --route f2d:85
#
# Assumes the Fail2Drive repo is cloned at ~/fail2drive (default). Override with
# FAIL2DRIVE_ROOT or the individual env vars below if your layout differs.
#
#   FAIL2DRIVE_ROOT=~/fail2drive bash run_simlingo_fail2drive.sh --route f2d:85 --eval-only

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# ── Defaults ──────────────────────────────────────────────────────────────────
SIMLINGO_CKPT="/home/celinet/simlingo_checkpoints/simlingo/checkpoints/epoch=013.ckpt"
POLICY_MODE="single"
HIGH_LEVEL_CKPT="/home/celinet/ogbench-carla/simlingo_checkpoints/2026_05_24_06_52_33_simlingo_seed1_bellman/checkpoints/epoch=019.ckpt"
LOW_LEVEL_CKPT="/home/celinet/ogbench-carla/simlingo_checkpoints/2026_05_23_21_39_41_simlingo_ll_vla_meta_conditioned/checkpoints/epoch=029.ckpt"
HIGH_LEVEL_HYDRA_CONFIG=""
LOW_LEVEL_HYDRA_CONFIG=""
HIERARCHICAL_SOURCE_ROOT=""
HIGH_LEVEL_SOURCE_ROOT="/scratch/current/celinet/simlingo-steervla"
LOW_LEVEL_SOURCE_ROOT="/scratch/current/celinet/simlingo-tian"
ROUTE="generalization-wall-1095"   # default Fail2Drive route (f2d:85)
STEPS="30000"
WARMUP="500"
LEARNING_STARTS="200"  # normalizer collects stats for 200 steps before freezing and before SAC updates begin
CHUNK_SIZE="1"
RES_SCALE_ACCEL="2.0"
RES_SCALE_STEER="2.0"
BATCH_SIZE="256"
BUFFER_CAP="10000"
UPDATES_PER_STEP="4"
ACTOR_LR="1e-4"
CRITIC_LR="1e-4"
RESIDUAL_CLIP_SCHEDULE_STEPS="500"
COLLISION_EVENT_PENALTY=""
COLLISION_CONTACT_PENALTY=""
OUTSIDE_ROUTE_EVENT_PENALTY=""
TRAFFIC_VIOLATION_PENALTY=""
CRASH_STUCK_PENALTY=""
PROGRESS_REWARD_WEIGHT=""
STEER_PENALTY_WEIGHT=""
BRAKE_PENALTY_WEIGHT=""
SPEED_LIMIT_PENALTY_WEIGHT=""
SUCCESS_BONUS=""
FAILURE_BONUS=""
SEED="0"
RUN_GROUP="Debug"
WANDB_PROJECT="OGBench-CARLA-SimLingo-F2D"
WANDB_RUN_NAME=""
WANDB_MODE="${WANDB_MODE:-online}"
LOG_INTERVAL="10"
VIDEO_LOG_INTERVAL="1"
SAVE_INTERVAL="2000"
EVAL_EPISODES="1"
EVAL_STEP_LIMIT="2000"
DEVICE="cuda"
CARLA_CFG="impls/configs/carla_config.yaml"
SIMLINGO_PYTHON="/home/celinet/miniconda3/envs/simlingo/bin/python"
TRAINING_MODE="sac_residual"
OBS_MODE="encoder"
ACTOR_L2_REG="0"
TERMINATE_ON_INFRACTION="false"
EVAL_ONLY="false"
INCLUDE_EGO_STATE="false"
DEBUG_NEG_SPEED="false"
DEBUG_TARGET_SPEED=""
EXPERT_DEBUG="false"
EXPERT_RECOVER_DEBUG="false"
EXPERT_CHECKPOINT=""
SAVE_VIDEO="true"
DRY_RUN="false"
DEBUG_OBS_HIST="false"
DEBUG_OBS_HIST_STEPS="2000"
USE_GEMINI_COACH="false"
LOG_Q_EXPERT_DIFF="false"
CRITIC_MODE="none"
GEMINI_MODEL="gemini-3.5-flash"
GEMINI_API_KEY=""
COACH_ACTION_CHUNK_STEPS="10"
COACH_QUERY_FREQ="0"
COACH_LABEL_MODE="bow"
COACH_EMBED_PLOT="false"
TRAIN_GPU=""          # empty = preserve inherited CUDA_VISIBLE_DEVICES
CARLA_HOST=""         # empty = use yaml value
CARLA_PORT=""         # empty = use yaml value
CARLA_STREAMING_PORT="" # empty = use yaml value or 0 when generating a temp config
CARLA_TM_PORT=""      # empty = derive from --port (+6000) or use yaml value
X_DISPLAY_NUM=""      # empty = derive from --port when generating a temp config
GPU_RANK=""           # CARLA -graphicsadapter; empty = use yaml value
RENDER_ADAPTER=""     # alias for GPU_RANK
INSTANCE=""           # compact parallel-run index
# ── Fail2Drive repo layout ────────────────────────────────────────────────────
# All three paths are derived from FAIL2DRIVE_ROOT (default: ~/fail2drive).
# Override any individually if your layout differs.
_F2D_ROOT="${FAIL2DRIVE_ROOT:-${HOME}/fail2drive}"
# Path to the 200 route XML files
FAIL2DRIVE_ROUTES_DIR="${FAIL2DRIVE_ROUTES_DIR:-${_F2D_ROOT}/fail2drive_split}"
# Path to fail2drive's srunner scenario classes (image_on_object.py, hard_break.py, etc.)
FAIL2DRIVE_SCENARIOS_DIR="${FAIL2DRIVE_SCENARIOS_DIR:-${_F2D_ROOT}/scenario_runner/srunner/scenarios}"
# Dedicated CARLA build for Fail2Drive (has the custom assets already baked in)
FAIL2DRIVE_CARLA_ROOT="${FAIL2DRIVE_CARLA_ROOT:-${_F2D_ROOT}/f2d_carla}"
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: bash run_simlingo_fail2drive.sh [options] [-- extra args passed to main_carla_simlingo.py]

This script is identical to run_simlingo.sh except it defaults to Fail2Drive
routes and sets WANDB_PROJECT to OGBench-CARLA-SimLingo-F2D.

Fail2Drive route names (use any form):
  f2d:<id>                    e.g. --route f2d:85
  kebab scenario name         e.g. --route base-pedestrians-on-road-0085
  original filename           e.g. --route Base_PedestriansOnRoad_0085

Mode:
  --eval-only               Run base policy only, no SAC training
  --terminate-on-infraction Terminate episode on collision, traffic violation, or off-route
  --training-mode MODE      sac_residual|dagger_residual. Default: sac_residual
  --policy-mode MODE        single|hierarchical. Default: single
  --ego-state               Enable ego state vector input to actor/critic (off by default)
  --no-ego-state            Disable ego state vector input to actor/critic
  --debug-neg-speed         Replace reward with -speed (m/s) — SAC should brake
  --debug-target-speed F    Replace reward with -|speed - F| (m/s)
  --expert-debug            Drive with CARLA expert action instead of base+residual (dagger_residual only)
  --expert-recover-debug    Run SimLingo for a random [70,200] ticks per episode, then switch to expert

Routing / environment:
  --route NAME              Fail2Drive (or Bench2Drive) route name/id
                            Default: base-pedestrians-on-road-0085
  --carla-config PATH       Path to carla_config.yaml
                            Default: impls/configs/carla_config.yaml
  --fail2drive-root PATH    Root of the cloned Fail2Drive repo. Sets routes/scenarios/CARLA
                            sub-paths automatically. Default: ~/fail2drive
  --fail2drive-routes-dir PATH
                            Path to Fail2Drive route XMLs (200 .xml files).
                            Default: \$FAIL2DRIVE_ROOT/fail2drive_split
  --fail2drive-scenarios-dir PATH
                            Path to Fail2Drive srunner scenarios dir.
                            Default: \$FAIL2DRIVE_ROOT/scenario_runner/srunner/scenarios
  --fail2drive-carla-root PATH
                            Dedicated CARLA build with Fail2Drive assets.
                            Default: \$FAIL2DRIVE_ROOT/f2d_carla

Multi-instance (ports):
  --instance N              Convenience preset for one concurrent run:
                              train GPU=N, CARLA adapter=N, port=2201+10*N,
                              traffic-manager port=8201+10*N
  --gpu N                   Alias for --train-gpu N --sim-gpu N
  --train-gpu N             CUDA_VISIBLE_DEVICES for SimLingo/PyTorch.
  --sim-gpu N               CARLA rendering adapter (-graphicsadapter).
  --render-adapter N        Alias for --sim-gpu N.
  --carla-host HOST         CARLA host override. Default: yaml value.
  --port N                  CARLA server port. Generates a temp config overriding the yaml.
  --carla-port N            Alias for --port.
  --tm-port N               Traffic manager port.
  --carla-streaming-port N  CARLA streaming port.
  --x-display-num N         Xvfb display number.

Model:
  --checkpoint PATH         SimLingo checkpoint directory
  --high-checkpoint PATH    High-level checkpoint for --policy-mode hierarchical
  --low-checkpoint PATH     Low-level checkpoint for --policy-mode hierarchical
  --high-hydra-config PATH  High-level Hydra config for hierarchical mode
  --low-hydra-config PATH   Low-level Hydra config for hierarchical mode
  --hierarchical-source-root PATH
  --high-source-root PATH
  --low-source-root PATH
  --device DEVICE           Torch device. Default: cuda
  --chunk-size N            Waypoints per VLM call (keep at 1). Default: 1

SAC hyperparameters:
  --steps N                 Total env steps. Default: 3000
  --warmup N                Warmup steps. Default: 500
  --learning-starts N       Buffer threshold before updates. Default: 200
  --res-scale F             Set both residual scales (accel+steer) to F
  --res-scale-accel F       Residual scale for acceleration. Default: 2.0
  --res-scale-steer F       Residual scale for steering. Default: 0.6
  --batch-size N            SAC mini-batch size. Default: 256
  --buffer-cap N            Replay buffer capacity. Default: 10000
  --updates-per-step N      SAC updates per env step. Default: 4
  --actor-lr F              Actor learning rate. Default: 1e-4
  --critic-lr F             Critic learning rate. Default: 1e-4
  --obs-mode MODE           encoder|vlm_hidden. Default: encoder
  --actor-l2-reg F          L2 regularization. Default: 0
  --gamma F                 Discount factor.
  --tau F                   Target network tau.
  --residual-clip-schedule-steps N

Reward coefficients:
  --collision-event-penalty F
  --collision-contact-penalty F
  --outside-route-event-penalty F
  --traffic-violation-penalty F
  --crash-stuck-penalty F
  --progress-reward-weight F
  --steer-penalty-weight F
  --brake-penalty-weight F
  --speed-limit-penalty-weight F
  --success-bonus F
  --failure-bonus F

Logging:
  --run-group NAME          W&B run group. Default: Debug
  --wandb-project NAME      W&B project. Default: OGBench-CARLA-SimLingo-F2D
  --wandb-mode MODE         online|offline|disabled. Default: \$WANDB_MODE or online
  --log-interval N          Log training metrics every N steps. Default: 10
  --video-log-interval N    Upload video every N episodes. Default: 1
  --save-interval N         Save SAC checkpoint every N steps. Default: 2000
  --no-video                Disable local mp4 saving
  --debug-obs-hist          Collect obs samples then plot per-component histograms and exit
  --debug-obs-hist-steps N  Steps to collect before plotting (default: 2000)
  --dry-run                 Print resolved config/command without launching

Critic mode (expert feedback as critic input):
  --critic-mode MODE        none|expert|language_bow|noise. Default: none
                              none         -> standard SAC critic (no extra input)
                              expert       -> feed expert planner (accel, steer, valid) as
                                             additional critic input each step
                              language_bow -> feed scene-grounded language BoW (SCENE_DELTA_VOCAB,
                                             26 words + validity) from expert-vs-agent action delta
                              noise        -> feed i.i.d. Gaussian noise (same 27-dim as language_bow)
                                             as ablation baseline to isolate capacity vs language signal

Gemini VLM coach:
  --use-gemini-coach
  --gemini-model MODEL
  --gemini-api-key KEY
  --coach-action-chunk-steps N
  --coach-query-freq N        Query Gemini every N episode steps (0 = episode end only). Default: 0
  --coach-label-mode MODE     bow|vlm_embed|vlm_embed_raw. Default: bow
  --coach-embed-plot          Log PCA scatter of label embeddings to wandb each episode

  -h, --help                Show this help

Examples:
  # Eval only — check base policy on a Fail2Drive route
  bash run_simlingo_fail2drive.sh --eval-only --route base-pedestrians-on-road-0085

  # SAC training
  bash run_simlingo_fail2drive.sh --route f2d:42 --steps 50000 --run-group F2D-SAC-v1

  # Two parallel instances
  bash run_simlingo_fail2drive.sh --instance 0 --route f2d:85 &
  bash run_simlingo_fail2drive.sh --instance 1 --route f2d:42 &

  # With dedicated Fail2Drive CARLA install
  FAIL2DRIVE_CARLA_ROOT=/home/carla/f2d_carla bash run_simlingo_fail2drive.sh \
      --route f2d:85 --eval-only
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --eval-only)           EVAL_ONLY="true"; shift ;;
    --training-mode|--training_mode) TRAINING_MODE="$2"; shift 2 ;;
    --obs-mode|--obs_mode)        OBS_MODE="$2"; shift 2 ;;
    --actor-l2-reg|--actor_l2_reg) ACTOR_L2_REG="$2"; shift 2 ;;
    --terminate-on-infraction|--terminate_on_infraction) TERMINATE_ON_INFRACTION="true"; shift ;;
    --policy-mode)         POLICY_MODE="$2"; shift 2 ;;
    --ego-state)           INCLUDE_EGO_STATE="true"; shift ;;
    --no-ego-state)        INCLUDE_EGO_STATE="false"; shift ;;
    --debug-neg-speed)     DEBUG_NEG_SPEED="true"; shift ;;
    --debug-target-speed)  DEBUG_TARGET_SPEED="$2"; shift 2 ;;
    --expert-debug)        EXPERT_DEBUG="true"; shift ;;
    --expert-recover-debug) EXPERT_RECOVER_DEBUG="true"; shift ;;
    --expert-checkpoint)   EXPERT_CHECKPOINT="$2"; shift 2 ;;
    --route)               ROUTE="$2"; shift 2 ;;
    --carla-config)        CARLA_CFG="$2"; shift 2 ;;
    --fail2drive-root)       _F2D_ROOT="$2"; FAIL2DRIVE_ROUTES_DIR="${FAIL2DRIVE_ROUTES_DIR:-${_F2D_ROOT}/fail2drive_split}"; FAIL2DRIVE_SCENARIOS_DIR="${FAIL2DRIVE_SCENARIOS_DIR:-${_F2D_ROOT}/scenario_runner/srunner/scenarios}"; FAIL2DRIVE_CARLA_ROOT="${FAIL2DRIVE_CARLA_ROOT:-${_F2D_ROOT}/f2d_carla}"; shift 2 ;;
    --fail2drive-routes-dir) FAIL2DRIVE_ROUTES_DIR="$2"; shift 2 ;;
    --fail2drive-scenarios-dir) FAIL2DRIVE_SCENARIOS_DIR="$2"; shift 2 ;;
    --fail2drive-carla-root) FAIL2DRIVE_CARLA_ROOT="$2"; shift 2 ;;
    --checkpoint)          SIMLINGO_CKPT="$2"; shift 2 ;;
    --high-checkpoint)     HIGH_LEVEL_CKPT="$2"; shift 2 ;;
    --low-checkpoint)      LOW_LEVEL_CKPT="$2"; shift 2 ;;
    --high-hydra-config)   HIGH_LEVEL_HYDRA_CONFIG="$2"; shift 2 ;;
    --low-hydra-config)    LOW_LEVEL_HYDRA_CONFIG="$2"; shift 2 ;;
    --hierarchical-source-root) HIERARCHICAL_SOURCE_ROOT="$2"; shift 2 ;;
    --high-source-root)    HIGH_LEVEL_SOURCE_ROOT="$2"; shift 2 ;;
    --low-source-root)     LOW_LEVEL_SOURCE_ROOT="$2"; shift 2 ;;
    --device)              DEVICE="$2"; shift 2 ;;
    --chunk-size)          CHUNK_SIZE="$2"; shift 2 ;;
    --steps)               STEPS="$2"; shift 2 ;;
    --warmup)              WARMUP="$2"; shift 2 ;;
    --learning-starts)     LEARNING_STARTS="$2"; shift 2 ;;
    --res-scale)           RES_SCALE_ACCEL="$2"; RES_SCALE_STEER="$2"; shift 2 ;;
    --res-scale-accel)     RES_SCALE_ACCEL="$2"; shift 2 ;;
    --res-scale-steer)     RES_SCALE_STEER="$2"; shift 2 ;;
    --batch-size)          BATCH_SIZE="$2"; shift 2 ;;
    --buffer-cap)          BUFFER_CAP="$2"; shift 2 ;;
    --updates-per-step)    UPDATES_PER_STEP="$2"; shift 2 ;;
    --actor-lr)            ACTOR_LR="$2"; shift 2 ;;
    --critic-lr)           CRITIC_LR="$2"; shift 2 ;;
    --gamma)               GAMMA="${2}"; shift 2 ;;
    --tau)                 TAU="${2}"; shift 2 ;;
    --residual-clip-schedule-steps|--residual_clip_schedule_steps) RESIDUAL_CLIP_SCHEDULE_STEPS="$2"; shift 2 ;;
    --collision-event-penalty|--collision_event_penalty) COLLISION_EVENT_PENALTY="$2"; shift 2 ;;
    --collision-contact-penalty|--collision_contact_penalty) COLLISION_CONTACT_PENALTY="$2"; shift 2 ;;
    --outside-route-event-penalty|--outside_route_event_penalty) OUTSIDE_ROUTE_EVENT_PENALTY="$2"; shift 2 ;;
    --traffic-violation-penalty|--traffic_violation_penalty) TRAFFIC_VIOLATION_PENALTY="$2"; shift 2 ;;
    --crash-stuck-penalty|--crash_stuck_penalty) CRASH_STUCK_PENALTY="$2"; shift 2 ;;
    --progress-reward-weight|--progress_reward_weight) PROGRESS_REWARD_WEIGHT="$2"; shift 2 ;;
    --steer-penalty-weight|--steer_penalty_weight) STEER_PENALTY_WEIGHT="$2"; shift 2 ;;
    --brake-penalty-weight|--brake_penalty_weight) BRAKE_PENALTY_WEIGHT="$2"; shift 2 ;;
    --speed-limit-penalty-weight|--speed_limit_penalty_weight) SPEED_LIMIT_PENALTY_WEIGHT="$2"; shift 2 ;;
    --success-bonus|--success_bonus) SUCCESS_BONUS="$2"; shift 2 ;;
    --failure-bonus|--failure_bonus) FAILURE_BONUS="$2"; shift 2 ;;
    --run-group)           RUN_GROUP="$2"; shift 2 ;;
    --wandb-project)       WANDB_PROJECT="$2"; shift 2 ;;
    --wandb-run-name)      WANDB_RUN_NAME="$2"; shift 2 ;;
    --wandb-mode)          WANDB_MODE="$2"; shift 2 ;;
    --log-interval)        LOG_INTERVAL="$2"; shift 2 ;;
    --video-log-interval)  VIDEO_LOG_INTERVAL="$2"; shift 2 ;;
    --save-interval)       SAVE_INTERVAL="$2"; shift 2 ;;
    --eval-episodes)       EVAL_EPISODES="$2"; shift 2 ;;
    --eval-step-limit)     EVAL_STEP_LIMIT="$2"; shift 2 ;;
    --seed)                SEED="$2"; shift 2 ;;
    --no-video)            SAVE_VIDEO="false"; shift ;;
    --critic-mode)         CRITIC_MODE="$2"; shift 2 ;;
    --log-q-expert-diff)   LOG_Q_EXPERT_DIFF="true"; shift ;;
    --use-gemini-coach)    USE_GEMINI_COACH="true"; shift ;;
    --gemini-model)        GEMINI_MODEL="$2"; shift 2 ;;
    --gemini-api-key)      GEMINI_API_KEY="$2"; shift 2 ;;
    --coach-action-chunk-steps) COACH_ACTION_CHUNK_STEPS="$2"; shift 2 ;;
    --coach-query-freq)         COACH_QUERY_FREQ="$2"; shift 2 ;;
    --coach-label-mode)         COACH_LABEL_MODE="$2"; shift 2 ;;
    --coach-embed-plot)         COACH_EMBED_PLOT="true"; shift ;;
    --debug-obs-hist)      DEBUG_OBS_HIST="true"; shift ;;
    --debug-obs-hist-steps) DEBUG_OBS_HIST_STEPS="$2"; shift 2 ;;
    --dry-run)             DRY_RUN="true"; shift ;;
    --instance)            INSTANCE="$2"; shift 2 ;;
    --gpu)                 TRAIN_GPU="$2"; GPU_RANK="$2"; shift 2 ;;
    --train-gpu|--cuda-gpu) TRAIN_GPU="$2"; shift 2 ;;
    --sim-gpu|--render-adapter) GPU_RANK="$2"; shift 2 ;;
    --carla-host)          CARLA_HOST="$2"; shift 2 ;;
    --port)                CARLA_PORT="$2"; shift 2 ;;
    --carla-port)          CARLA_PORT="$2"; shift 2 ;;
    --tm-port)             CARLA_TM_PORT="$2"; shift 2 ;;
    --carla-streaming-port) CARLA_STREAMING_PORT="$2"; shift 2 ;;
    --x-display-num)       X_DISPLAY_NUM="$2"; shift 2 ;;
    --gpu-rank)            GPU_RANK="$2"; shift 2 ;;
    -h|--help)             usage; exit 0 ;;
    --)                    shift; EXTRA_ARGS+=("$@"); break ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

# ── Critic mode validation ────────────────────────────────────────────────────
case "$CRITIC_MODE" in
  none|expert|language_bow|noise) ;;
  *)
    echo "Invalid --critic-mode: $CRITIC_MODE" >&2
    echo "Expected one of: none, expert, language_bow, noise" >&2
    exit 2
    ;;
esac
USE_EXPERT_IN_CRITIC="false"
USE_LANGUAGE_BOW_CRITIC="false"
USE_NOISE_CRITIC="false"
[[ "$CRITIC_MODE" == "expert" ]] && USE_EXPERT_IN_CRITIC="true"
[[ "$CRITIC_MODE" == "language_bow" ]] && USE_LANGUAGE_BOW_CRITIC="true"
[[ "$CRITIC_MODE" == "noise" ]] && USE_NOISE_CRITIC="true"

# ── Auto-suffix W&B run name with train mode and critic mode ──────────────────
_effective_mode="$( [[ "$EVAL_ONLY" == "true" ]] && echo "eval" || echo "$TRAINING_MODE" )"
if [[ -z "$WANDB_RUN_NAME" ]]; then
  WANDB_RUN_NAME="${ROUTE}_${_effective_mode}_${CRITIC_MODE}"
else
  WANDB_RUN_NAME="${WANDB_RUN_NAME}_${_effective_mode}_${CRITIC_MODE}"
fi

# ── Environment check ─────────────────────────────────────────────────────────
REQUIRED_PKGS=(ml_collections)
for pkg in "${REQUIRED_PKGS[@]}"; do
  import_name="${pkg//-/_}"
  if ! "$SIMLINGO_PYTHON" -c "import ${import_name}" 2>/dev/null; then
    echo "[run_simlingo_fail2drive.sh] Installing missing package into simlingo env: $pkg"
    "$SIMLINGO_PYTHON" -m pip install "$pkg" -q
  fi
done

# ── Fail2Drive env setup ──────────────────────────────────────────────────────
# Export the routes + scenarios dirs so route_registry.py and fail2drive_compat.py
# can find them without needing the fail2drive pip package.
if [[ -d "$FAIL2DRIVE_ROUTES_DIR" ]]; then
  export FAIL2DRIVE_ROUTES_DIR
  echo "[run_simlingo_fail2drive.sh] fail2drive_routes_dir=$FAIL2DRIVE_ROUTES_DIR"
else
  echo "[run_simlingo_fail2drive.sh] WARNING: FAIL2DRIVE_ROUTES_DIR=${FAIL2DRIVE_ROUTES_DIR} not found — fail2drive routes won't be available" >&2
fi

if [[ -d "$FAIL2DRIVE_SCENARIOS_DIR" ]]; then
  export FAIL2DRIVE_SCENARIOS_DIR
  echo "[run_simlingo_fail2drive.sh] fail2drive_scenarios_dir=$FAIL2DRIVE_SCENARIOS_DIR"
else
  echo "[run_simlingo_fail2drive.sh] WARNING: FAIL2DRIVE_SCENARIOS_DIR=${FAIL2DRIVE_SCENARIOS_DIR} not found — scenario classes won't be patched" >&2
fi

# Switch CARLA_ROOT to the Fail2Drive build (has the custom assets built in).
if [[ -n "$FAIL2DRIVE_CARLA_ROOT" ]]; then
  if [[ ! -d "$FAIL2DRIVE_CARLA_ROOT" ]]; then
    echo "[run_simlingo_fail2drive.sh] WARNING: FAIL2DRIVE_CARLA_ROOT=${FAIL2DRIVE_CARLA_ROOT} doesn't exist; not switching CARLA_ROOT." >&2
  else
    export CARLA_ROOT="$FAIL2DRIVE_CARLA_ROOT"
    export CARLA_PYTHON_API_ROOT="$FAIL2DRIVE_CARLA_ROOT/PythonAPI/carla"
    echo "[run_simlingo_fail2drive.sh] Fail2Drive CARLA_ROOT=${CARLA_ROOT}"
    echo "[run_simlingo_fail2drive.sh]   (make sure the CARLA server was launched from ${CARLA_ROOT}/CarlaUE4.sh)"
  fi
fi

# ── Runtime overrides: generate a scoped temp config ───────────────────────────
if [[ -n "$INSTANCE" ]]; then
  if [[ -z "$TRAIN_GPU" ]]; then
    TRAIN_GPU="$INSTANCE"
  fi
  if [[ -z "$GPU_RANK" ]]; then
    GPU_RANK="$INSTANCE"
  fi
  if [[ -z "$CARLA_PORT" ]]; then
    CARLA_PORT=$(( 2201 + 10 * INSTANCE ))
  fi
  if [[ -z "$CARLA_TM_PORT" ]]; then
    CARLA_TM_PORT=$(( 8201 + 10 * INSTANCE ))
  fi
fi

# Auto-derive CARLA port from GPU rank if no explicit port was given.
if [[ -z "$CARLA_PORT" ]]; then
  _gpu_for_port="${GPU_RANK:-${TRAIN_GPU:-}}"
  if [[ -n "$_gpu_for_port" ]]; then
    CARLA_PORT=$(( 2000 + 10 * _gpu_for_port ))
  fi
fi

if [[ -n "$RENDER_ADAPTER" ]]; then
  GPU_RANK="$RENDER_ADAPTER"
fi

TMP_CFG=""
NEEDS_TMP_CFG="false"
for _v in "$CARLA_HOST" "$CARLA_PORT" "$CARLA_STREAMING_PORT" "$CARLA_TM_PORT" "$X_DISPLAY_NUM" "$GPU_RANK" \
          "$COLLISION_EVENT_PENALTY" "$COLLISION_CONTACT_PENALTY" "$OUTSIDE_ROUTE_EVENT_PENALTY" \
          "$TRAFFIC_VIOLATION_PENALTY" "$CRASH_STUCK_PENALTY" "$PROGRESS_REWARD_WEIGHT" "$STEER_PENALTY_WEIGHT" \
          "$BRAKE_PENALTY_WEIGHT" "$SPEED_LIMIT_PENALTY_WEIGHT" "$SUCCESS_BONUS" "$FAILURE_BONUS"; do
  if [[ -n "$_v" ]]; then
    NEEDS_TMP_CFG="true"
  fi
done

if [[ "$NEEDS_TMP_CFG" == "true" ]]; then
  if [[ -z "$CARLA_TM_PORT" ]]; then
    if [[ -n "$CARLA_PORT" ]]; then
      CARLA_TM_PORT=$(( CARLA_PORT + 6000 ))
    fi
  fi
  if [[ -z "$CARLA_STREAMING_PORT" && -n "$CARLA_PORT" ]]; then
    CARLA_STREAMING_PORT="0"
  fi
  if [[ -z "$X_DISPLAY_NUM" && -n "$CARLA_PORT" ]]; then
    X_DISPLAY_NUM=$(( 10 + (CARLA_PORT % 90) ))
  fi

  TMP_ROOT="$ROOT_DIR/.run_simlingo"
  mkdir -p "$TMP_ROOT"
  TMP_DIR="$(mktemp -d "$TMP_ROOT/run.XXXXXX")"
  TMP_CFG="$TMP_DIR/carla_config.yaml"
  trap 'rm -rf "$TMP_DIR"' EXIT

  "$SIMLINGO_PYTHON" - "$CARLA_CFG" "$TMP_CFG" \
    "$CARLA_HOST" "$CARLA_PORT" "$CARLA_STREAMING_PORT" "$CARLA_TM_PORT" "$X_DISPLAY_NUM" "$GPU_RANK" \
    "$COLLISION_EVENT_PENALTY" "$COLLISION_CONTACT_PENALTY" "$OUTSIDE_ROUTE_EVENT_PENALTY" \
    "$TRAFFIC_VIOLATION_PENALTY" "$CRASH_STUCK_PENALTY" "$PROGRESS_REWARD_WEIGHT" "$STEER_PENALTY_WEIGHT" \
    "$BRAKE_PENALTY_WEIGHT" "$SPEED_LIMIT_PENALTY_WEIGHT" "$SUCCESS_BONUS" "$FAILURE_BONUS" <<'PY'
import sys
from pathlib import Path

import yaml

(
    base_path, out_path, host, port, streaming_port, tm_port, x_display_num, gpu_rank,
    collision_event_penalty, collision_contact_penalty, outside_route_event_penalty,
    traffic_violation_penalty, crash_stuck_penalty, progress_reward_weight, steer_penalty_weight,
    brake_penalty_weight, speed_limit_penalty_weight, success_bonus, failure_bonus,
) = sys.argv[1:20]
cfg = yaml.safe_load(Path(base_path).read_text())

if host:
    cfg["host"] = host
if port:
    cfg["port"] = int(port)
    cfg["checkpoint"] = f"./simulation_results_port{port}.json"
    cfg["debug_checkpoint"] = f"./live_results_port{port}.txt"
if streaming_port:
    cfg["streaming_port"] = int(streaming_port)
if tm_port:
    cfg["traffic_manager_port"] = int(tm_port)
if x_display_num:
    cfg["x_display_num"] = int(x_display_num)
if gpu_rank:
    cfg["gpu_rank"] = int(gpu_rank)

reward_overrides = {
    "collision_event_penalty": collision_event_penalty,
    "collision_contact_penalty": collision_contact_penalty,
    "outside_route_event_penalty": outside_route_event_penalty,
    "traffic_violation_penalty": traffic_violation_penalty,
    "crash_stuck_penalty": crash_stuck_penalty,
    "progress_reward_weight": progress_reward_weight,
    "steer_penalty_weight": steer_penalty_weight,
    "brake_penalty_weight": brake_penalty_weight,
    "speed_limit_penalty_weight": speed_limit_penalty_weight,
    "success_bonus": success_bonus,
    "failure_bonus": failure_bonus,
}
for key, value in reward_overrides.items():
    if value:
        cfg[key] = float(value)

Path(out_path).write_text(yaml.safe_dump(cfg, sort_keys=False))
PY

  # Expert debug: enable the SimLingo autopilot so _compute_expert_action() uses
  # the live PDM-Lite planner instead of the route-based approximation (which can
  # return zeros and fall back to random residual → collision).
  if [[ "$EXPERT_DEBUG" == "true" || "$EXPERT_RECOVER_DEBUG" == "true" ]]; then
    "$SIMLINGO_PYTHON" - "$TMP_CFG" <<'PY'
import sys, yaml
from pathlib import Path
p = Path(sys.argv[1])
cfg = yaml.safe_load(p.read_text())
cfg["expert_controller"] = "simlingo_autopilot"
p.write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
  fi

  CARLA_CFG="$TMP_CFG"
fi

# If no explicit CARLA adapter was given, preserve the yaml value.
if [[ -z "$GPU_RANK" ]]; then
  GPU_RANK="$("$SIMLINGO_PYTHON" - "$CARLA_CFG" <<'PY'
import sys
from pathlib import Path

import yaml

cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
print(int(cfg.get("gpu_rank", 0)))
PY
)"
fi

# ── Build the argument list ───────────────────────────────────────────────────
ARGS=(
  --simlingo_checkpoint="$SIMLINGO_CKPT"
  --policy_mode="$POLICY_MODE"
  --route="$ROUTE"
  --carla_config="$CARLA_CFG"
  --carla_root="$CARLA_ROOT"
  --device="$DEVICE"
  --chunk_size="$CHUNK_SIZE"
  --res_scale_accel="$RES_SCALE_ACCEL"
  --res_scale_steer="$RES_SCALE_STEER"
  --obs_mode="$OBS_MODE"
  --actor_l2_reg="$ACTOR_L2_REG"
  --include_ego_state="$INCLUDE_EGO_STATE"
  --use_expert_in_critic="$USE_EXPERT_IN_CRITIC"
  --use_language_bow_critic="$USE_LANGUAGE_BOW_CRITIC"
  --use_noise_critic="$USE_NOISE_CRITIC"
  --log_q_expert_diff="$LOG_Q_EXPERT_DIFF"
  --terminate_on_infraction="$TERMINATE_ON_INFRACTION"
  --seed="$SEED"
  --run_group="$RUN_GROUP"
  --wandb_project="$WANDB_PROJECT"
  --wandb_mode="$WANDB_MODE"
  --log_interval="$LOG_INTERVAL"
  --video_log_interval="$VIDEO_LOG_INTERVAL"
  --save_interval="$SAVE_INTERVAL"
  --save_video="$SAVE_VIDEO"
)

if [[ "$POLICY_MODE" == "hierarchical" ]]; then
  ARGS+=(
    --high_level_checkpoint="$HIGH_LEVEL_CKPT"
    --low_level_checkpoint="$LOW_LEVEL_CKPT"
    --high_level_hydra_config="$HIGH_LEVEL_HYDRA_CONFIG"
    --low_level_hydra_config="$LOW_LEVEL_HYDRA_CONFIG"
    --hierarchical_source_root="$HIERARCHICAL_SOURCE_ROOT"
    --high_level_source_root="$HIGH_LEVEL_SOURCE_ROOT"
    --low_level_source_root="$LOW_LEVEL_SOURCE_ROOT"
  )
fi

ARGS+=(--gpu_rank="$GPU_RANK")

# Scope save_dir per port so parallel runs don't share checkpoints/logs.
if [[ -n "$CARLA_PORT" ]]; then
  ARGS+=(--save_dir="./logs/simlingo_residual_f2d_port${CARLA_PORT}")
fi

[[ -n "$WANDB_RUN_NAME" ]] && ARGS+=(--wandb_run_name="$WANDB_RUN_NAME")

if [[ "$EVAL_ONLY" == "true" ]]; then
  ARGS+=(--eval_only --eval_episodes="$EVAL_EPISODES" --eval_step_limit="$EVAL_STEP_LIMIT")
else
  ARGS+=(--training_mode="$TRAINING_MODE")
  ARGS+=(
    --total_steps="$STEPS"
    --warmup_steps="$WARMUP"
    --learning_starts="$LEARNING_STARTS"
    --batch_size="$BATCH_SIZE"
    --buffer_capacity="$BUFFER_CAP"
    --updates_per_step="$UPDATES_PER_STEP"
    --actor_lr="$ACTOR_LR"
    --critic_lr="$CRITIC_LR"
  )
  [[ -n "${GAMMA:-}" ]] && ARGS+=(--gamma="$GAMMA")
  [[ -n "${TAU:-}" ]]   && ARGS+=(--tau="$TAU")
  [[ -n "$RESIDUAL_CLIP_SCHEDULE_STEPS" ]] && ARGS+=(--residual_clip_schedule_steps="$RESIDUAL_CLIP_SCHEDULE_STEPS")
fi

if [[ "$DEBUG_NEG_SPEED" == "true" ]]; then
  ARGS+=(--debug_neg_speed_reward)
fi

if [[ -n "$DEBUG_TARGET_SPEED" ]]; then
  ARGS+=(--debug_target_speed_reward="$DEBUG_TARGET_SPEED")
fi

if [[ "$EXPERT_DEBUG" == "true" ]]; then
  ARGS+=(--expert_debug)
fi

if [[ "$EXPERT_RECOVER_DEBUG" == "true" ]]; then
  ARGS+=(--expert_recover_debug)
fi

if [[ -n "$EXPERT_CHECKPOINT" ]]; then
  ARGS+=(--expert_checkpoint="$EXPERT_CHECKPOINT")
fi

if [[ "$USE_GEMINI_COACH" == "true" ]]; then
  ARGS+=(--use_gemini_coach)
  ARGS+=(--gemini_model="$GEMINI_MODEL")
  ARGS+=(--coach_action_chunk_steps="$COACH_ACTION_CHUNK_STEPS")
  ARGS+=(--coach_query_freq="$COACH_QUERY_FREQ")
  ARGS+=(--coach_label_mode="$COACH_LABEL_MODE")
  [[ "$COACH_EMBED_PLOT" == "true" ]] && ARGS+=(--coach_embed_plot)
  _gemini_key="${GEMINI_API_KEY:-${GOOGLE_API_KEY:-}}"
  if [[ -n "$_gemini_key" ]]; then
    ARGS+=(--gemini_api_key="$_gemini_key")
  fi
fi

ARGS+=("${EXTRA_ARGS[@]}")

echo "[run_simlingo_fail2drive.sh] route=$ROUTE  eval_only=$EVAL_ONLY  training_mode=$TRAINING_MODE  policy_mode=$POLICY_MODE  debug_neg_speed=$DEBUG_NEG_SPEED  debug_target_speed=${DEBUG_TARGET_SPEED:-off}  expert_debug=$EXPERT_DEBUG  expert_recover_debug=$EXPERT_RECOVER_DEBUG  expert_checkpoint=${EXPERT_CHECKPOINT:-none}"
echo "[run_simlingo_fail2drive.sh] steps=$STEPS  warmup=$WARMUP  chunk_size=$CHUNK_SIZE  res_scale_accel=$RES_SCALE_ACCEL  res_scale_steer=$RES_SCALE_STEER  obs_mode=$OBS_MODE  actor_l2_reg=$ACTOR_L2_REG  critic_mode=$CRITIC_MODE"
echo "[run_simlingo_fail2drive.sh] wandb_mode=$WANDB_MODE  run_group=$RUN_GROUP  wandb_project=$WANDB_PROJECT"
echo "[run_simlingo_fail2drive.sh] checkpoint=$SIMLINGO_CKPT"
if [[ "$POLICY_MODE" == "hierarchical" ]]; then
  echo "[run_simlingo_fail2drive.sh] high_checkpoint=$HIGH_LEVEL_CKPT"
  echo "[run_simlingo_fail2drive.sh] low_checkpoint=$LOW_LEVEL_CKPT"
fi
echo "[run_simlingo_fail2drive.sh] carla_config=$CARLA_CFG"
if [[ -n "$FAIL2DRIVE_CARLA_ROOT" ]]; then
  echo "[run_simlingo_fail2drive.sh] fail2drive_carla_root=$FAIL2DRIVE_CARLA_ROOT"
fi
if [[ "$USE_GEMINI_COACH" == "true" ]]; then
  echo "[run_simlingo_fail2drive.sh] gemini_coach=enabled  model=$GEMINI_MODEL  chunk_steps=$COACH_ACTION_CHUNK_STEPS  query_freq=$COACH_QUERY_FREQ  label_mode=$COACH_LABEL_MODE  embed_plot=$COACH_EMBED_PLOT"
fi
if [[ -n "$TRAIN_GPU" ]]; then
  echo "[run_simlingo_fail2drive.sh] train_gpu=$TRAIN_GPU  carla_gpu=$GPU_RANK"
else
  echo "[run_simlingo_fail2drive.sh] train_gpu=${CUDA_VISIBLE_DEVICES:-<inherited/all>}  carla_gpu=$GPU_RANK"
fi
if [[ -n "$CARLA_PORT" ]]; then
  echo "[run_simlingo_fail2drive.sh] carla_host=${CARLA_HOST:-<yaml>}  carla_port=$CARLA_PORT  tm_port=$CARLA_TM_PORT  streaming_port=$CARLA_STREAMING_PORT  x_display=:${X_DISPLAY_NUM}"
fi
if [[ -n "$COLLISION_EVENT_PENALTY$COLLISION_CONTACT_PENALTY$OUTSIDE_ROUTE_EVENT_PENALTY$TRAFFIC_VIOLATION_PENALTY$CRASH_STUCK_PENALTY$PROGRESS_REWARD_WEIGHT$STEER_PENALTY_WEIGHT$BRAKE_PENALTY_WEIGHT$SPEED_LIMIT_PENALTY_WEIGHT$SUCCESS_BONUS$FAILURE_BONUS" ]]; then
  echo "[run_simlingo_fail2drive.sh] reward_overrides collision_event=$COLLISION_EVENT_PENALTY collision_contact=$COLLISION_CONTACT_PENALTY outside_route=$OUTSIDE_ROUTE_EVENT_PENALTY traffic_violation=$TRAFFIC_VIOLATION_PENALTY crash_stuck=$CRASH_STUCK_PENALTY progress=$PROGRESS_REWARD_WEIGHT success=$SUCCESS_BONUS failure=$FAILURE_BONUS"
fi
echo ""

if [[ "$DEBUG_OBS_HIST" == "true" ]]; then
  ARGS+=(--debug_obs_hist --debug_obs_hist_steps="$DEBUG_OBS_HIST_STEPS")
fi

if [[ "$DRY_RUN" == "true" ]]; then
  printf '[run_simlingo_fail2drive.sh] command: '
  if [[ -n "$TRAIN_GPU" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$TRAIN_GPU"
  fi
  printf 'WANDB_MODE=%q %q impls/main_carla_simlingo.py' "$WANDB_MODE" "$SIMLINGO_PYTHON"
  printf ' %q' "${ARGS[@]}"
  printf '\n'
  exit 0
fi

# ── Stale-process cleanup: free the ports this run needs ──────────────────────
# Only kills the process holding the *exact* port this instance will use,
# so parallel runs on different ports are unaffected.
_kill_port_holder() {
  local port=$1 label=$2
  local pid
  pid=$(ss -Htlnp "sport = :${port}" 2>/dev/null | grep -oP '(?<=pid=)\d+' | head -1 || true)
  if [[ -n "$pid" ]]; then
    echo "[run_simlingo_fail2drive.sh] WARNING: ${label} port ${port} already held by pid ${pid} — killing stale process"
    kill "$pid" 2>/dev/null || true
    sleep 1
  fi
}

if [[ -n "$CARLA_TM_PORT" ]]; then
  _kill_port_holder "$CARLA_TM_PORT" "TM"
fi
if [[ -n "$CARLA_PORT" ]]; then
  _kill_port_holder "$CARLA_PORT" "CARLA RPC"
fi

if [[ -n "$TRAIN_GPU" ]]; then
  CUDA_VISIBLE_DEVICES="$TRAIN_GPU" WANDB_MODE="$WANDB_MODE" "$SIMLINGO_PYTHON" impls/main_carla_simlingo.py "${ARGS[@]}"
else
  WANDB_MODE="$WANDB_MODE" "$SIMLINGO_PYTHON" impls/main_carla_simlingo.py "${ARGS[@]}"
fi
