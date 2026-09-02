#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Machine-specific: override by exporting CARLA_ROOT (see ogbench/carla/README.md,
# "Machine-specific settings"). We warn rather than silently substituting a path from
# somebody else's box, which only produces a confusing import error later.
export CARLA_ROOT="${CARLA_ROOT:-/raid/users/${USER}/carla}"
CARLA_ROOT="${CARLA_ROOT%/}"  # strip trailing slash
export CARLA_ROOT
if [[ ! -d "$CARLA_ROOT" ]]; then
  echo "[run_carla.sh] WARNING: CARLA_ROOT=${CARLA_ROOT} does not exist. Export CARLA_ROOT." >&2
fi

ROUTE="parking-cut-in-001"
ONLINE_STEPS="50000"
SEED="0"
RUN_GROUP="Debug"
SAVE_BUFFER="true"
EXPERT_DEBUG="false"
EXPERT_RECOVER_DEBUG="false"
WANDB_MODE="${WANDB_MODE:-online}"

# Keep model/checkpoint caches and run artifacts off the home filesystem. Each can still
# be overridden explicitly by the caller.
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/raid/users/${USER}/openpi}"
export HF_HOME="${HF_HOME:-/raid/users/${USER}/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/raid/users/${USER}/cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/raid/users/${USER}/cache/xdg}"
export OGBENCH_SAVE_DIR="${OGBENCH_SAVE_DIR:-/raid/users/${USER}/carla_exps}"
mkdir -p "$OPENPI_DATA_HOME" "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$OGBENCH_SAVE_DIR"

TRAIN_GPU_RANK="2"
SIM_GPU_RANK=""  # defaults to TRAIN_GPU_RANK below
RENDER_ADAPTER=""
CARLA_HOST="localhost"
CARLA_PORT=""
CARLA_STREAMING_PORT=""
TM_PORT=""
X_DISPLAY_NUM=""

CRITIC_MODE="none"
TRAIN_MODE="dagger"
INCLUDE_PROPRIO="false"
DISCOUNT="0.97"
PRETRAINED_CRITIC=""
QGF_CRITIC_CKPT=""
QGF_GUIDANCE_WEIGHT="0.0"
EVAL_ONLY="false"
BON_CRITIC_CKPT=""
BON_NUM_CANDIDATES="8"
BON_ONLINE_CRITIC="false"
BON_CANDIDATES_LOG_EVERY="20"
BON_CANDIDATES_WANDB="false"
BON_MAX_SAMPLE_ATTEMPTS="6"
# 0.0 (the base config's default) makes subtask decoding greedy/deterministic, so every
# candidate would decode to the identical subtask text and the diversity search in
# _sample_diverse_candidates could never find a different one. run_carla_teleop.sh's
# manual candidate picker overrides this to 1.0 for the same reason; match it here
# whenever best-of-N is active (see the get_config() injection below).
BON_COT_TEMPERATURE="1.0"
BON_SHADOW_ONLY="false"
BON_CRITIC_ROLLOUT_CHUNK="false"
PID_BRAKE_SPEED=""
PID_STUCK_THRESHOLD=""
PID_CREEP_DURATION=""
PID_CREEP_THROTTLE=""
BON_GEMINI_SELECT="false"
GEMINI_MODEL=""
BON_GEMINI_ROLLOUT_CHUNK="false"
BON_QWEN_SELECT="false"
QWEN_BON_URL="http://127.0.0.1:18765"
BON_QWEN_CADENCE="5"
QWEN_ONLINE_TRAIN="false"
QWEN_ONLINE_WARMUP_EPISODES="2"
MAX_EPISODES=""
TERMINATE_ON_COLLISION="false"
EXPERT_CONTROLLER=""
SAVE_VIDEO_LOCAL="true"

# Default stays the DSRL config. The residual stacks are opt-in via --agent-config:
#   impls/configs/pi0_residual_sac_config.py    (routing-commands DSRL residual sub-agent)
#   impls/configs/steervla_residual_config.py   (surya/rl-token standalone sac_residual)
BASE_AGENT_CFG="impls/configs/steervla_dsrl_config.py"
BASE_CARLA_CFG="impls/configs/carla_config.yaml"

# Fail2Drive's custom static props (brickwall, walkingkid, ampel, autobahn,
# dirt, graffiti, screenshot1/2, smiley, snow, stickers, walkingkidlarge) have
# been copied into vanilla CARLA 0.9.16 at /home/carla/carla-0-9-16/CarlaUE4/Content/
# (six asset packs: WallAssets, ImageAssets, StopOcclusions, AnimalVarietyPack,
# FarmAnimalsPack, AfricanAnimalsPack). The vanilla 0.9.16 server registers them
# via the standard *.Package.json discovery, so we don't need to switch CARLA_ROOT.
#
# Set FAIL2DRIVE_CARLA_ROOT in the env (or pass --fail2drive-carla-root) only if
# you want to point Fail2Drive routes at a separate CARLA install (e.g. the
# standalone 0.9.15 f2d_carla drop). Empty by default = stay on whatever
# CARLA_ROOT is already set for every route.
FAIL2DRIVE_CARLA_ROOT="${FAIL2DRIVE_CARLA_ROOT:-}"

# --- from dev (master + surya/rl-token + cast_relabel) ---
# Empty -> leave the base config's steervla.hl_training_gpu_rank untouched.
HL_TRAIN_GPU_RANK=""
HL_CKPT_DIR=""
HL_CKPT_EVERY=""
HL_CKPT_KEEP_LAST=""
ENABLE_UPDATES="true"
BASE_ONLY=""
STATE_ENCODER=""
RLT_CHECKPOINT=""
# Override the frozen base policy: e.g. deploy a relabelled Stage-1 ckpt. Empty -> config default.
STEERVLA_CKPT=""
ACTOR_CONFIG=""
# GRPO-only: greedy-base warmup steps before scoring/updates begin. Empty -> config default (0).
GRPO_WARMUP=""
# GRPO-only: executed-candidate selection (argmax|random|first). Empty -> config default (argmax).
GRPO_SELECT=""
# GRPO-only: candidates sampled+scored per state. Empty -> config default. 1 = single-sample (no selection).
GRPO_GROUP_SIZE=""
# GRPO-only: candidate CoT sampling temperature. Empty -> config default. Low (even 0) = near-greedy BoN.
GRPO_SCORE_TEMP=""
# Override config.steervla.cot_temperature (e.g. 1.0 to sample the base CoT). Empty -> config default.
COT_TEMPERATURE=""
# Crash supervisor: relaunch main_carla (resuming from checkpoint) after a CARLA native
# crash (SIGSEGV/SIGABRT, exit code >=128). 0 disables the retry loop.
MAX_RETRIES="${MAX_RETRIES:-50}"
# Optional stable, human-readable run name. Used by sweep launchers so W&B
# names identify the precise hyperparameter setting instead of only route/seed.
EXP_NAME_OVERRIDE=""

EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash run_carla.sh [options] [-- extra args passed to impls/main_carla.py]

Options:
  --route NAME              Bench2Drive route name/id. Default: parking-cut-in-001
  --online-steps N          Number of env steps. Default: 5000
  --seed N                  Random seed. Default: 0
  --run-group NAME          W&B / experiment group. Default: Debug
  --save-buffer BOOL        true|false. Default: true
  --expert-debug BOOL       true|false. Default: false
  --expert-recover-debug BOOL  true|false. Default: false
  --wandb-mode MODE         online|offline|disabled. Default: WANDB_MODE or offline

  --train-gpu N             JAX / learner GPU rank. Default: 2
  --sim-gpu N               Alias for --render-adapter. Default: same as --train-gpu
  --render-adapter N        CARLA -graphicsadapter value. Default: 3

  --carla-host HOST         CARLA host. Default: localhost
  --carla-port PORT         CARLA RPC port. Default: 2000 + sim-gpu*20
  --carla-streaming-port P  CARLA streaming port. Default: 0 (auto)
  --tm-port PORT            Traffic manager port. Default: 8000 + sim-gpu*20
  --x-display-num N         Xvfb display number. Default: derived from carla port

  --critic-mode MODE        one of:
                              none         -> no extra critic info
                              expert       -> expert action (+validity flag) as critic
                                              input; state-only, Bellman-consistent.
                                              PID-decoded [accel, steer] in
                                              accel_steer residual mode.
                              delta        -> numeric action delta (expert - agent;
                                              depends on the logged action)
                              delta-lang   -> language on expert-agent delta
                              expert-lang  -> language on expert action
                            Default: delta

  --train-mode MODE         rl|dagger|sac_residual|dagger_residual|grpo_hl. Default: dagger

  --include-proprio BOOL    true|false. Include ego-state vector as explicit input to residual
                            actor and critic (residual_append_state). Default: false

  --discount FLOAT          RL discount factor (gamma). Default: 0.99

  --hl-gpu N                JAX GPU rank for the isolated HL (VLM-backbone) update.
  --hl-ckpt-dir PATH        Where to write params-only SteerVLA policy checkpoints for later
                            inference. Default: <save_dir>/steervla_hl_ckpt.
  --hl-ckpt-every N         Checkpoint every N env steps (plus once at exit). 0 disables.
  --hl-ckpt-keep-last N     Retain only the newest N checkpoints (~10 GB each). 0 = keep all.
  --enable-updates BOOL     true|false. false = rollout/buffer only. Default: true
  --base-only BOOL          true|false. Roll out the frozen base policy only (no RL).
  --state-encoder NAME      pi_prefix|pi_prefix_groups|siglip_pool|rl_token (residual stack).
  --rlt-checkpoint PATH     RLT autoencoder checkpoint (only for --state-encoder rl_token).
  --max-retries N           Auto-restart+resume after a CARLA crash. Default: 50 (0 disables)
  --exp-name NAME           Fixed W&B/artifact run name. Default: route + seed + timestamp.

  --agent-config PATH       Base agent config. Default: impls/configs/pi0_residual_sac_config.py
  --carla-config PATH       Base CARLA yaml. Default: impls/configs/carla_config.yaml
  --steervla-checkpoint PATH  Override config.steervla.checkpoint (deploy a relabelled ckpt).
  --actor-config NAME       Override config.steervla.actor_config (match the ckpt's model).
  --grpo-warmup N           GRPO only: drive the greedy base for N steps before scoring/updates.
  --grpo-select MODE        GRPO only: executed candidate = argmax|random|first. Default: argmax.
  --grpo-group-size N       GRPO only: candidates sampled/scored per state (1 = single-sample).
  --grpo-score-temp T       GRPO only: candidate CoT sampling temperature (low/0 = near-greedy BoN).
  --cot-temperature T       Override config.steervla.cot_temperature (e.g. 1.0 to sample base CoTs).

  --pretrained-critic PATH  Path to a pretrained critic .pkl from pretrain_critic.py.
                            Injects obs_encoder + critic params before online training begins.
                            Must match the online config (same encoder, hidden dims, action_mode,
                            critic_feedback_mode=none).

  --qgf-critic-ckpt PATH    Pretrained critic .pkl for QGF Q-gradient guidance at each pi0
                            denoising step (test-time only, no training).
  --qgf-guidance-weight F   Scale for QGF Q-gradient guidance. Default: 0.0 (disabled).

  --bon-critic-ckpt PATH    Pretrained critic .pkl (pretrain_critic.py, action_mode=waypoints)
                            used to score N candidate action chunks sampled straight from the
                            frozen pi0 base policy (no residual actor); the highest-Q candidate
                            is executed each step. Requires --train-mode=rl.
  --bon-num-candidates N    Number of candidates to sample/score per step. Default: 8.
  --bon-online-critic BOOL  true|false. Best-of-N scored with the live online critic instead
                            of a frozen --bon-critic-ckpt; keeps training via update_with_vla()
                            on collected transitions. Warm-start with --pretrained-critic.
                            Requires --eval-only=false. Default: false.
  --bon-candidates-log-every N
                            Save a local overlay frame every N env steps showing every best-of-N
                            candidate's subtask + score, selected one marked. 0 disables.
                            Default: 20.
  --bon-candidates-wandb BOOL
                            Also upload candidate panels to W&B. Default: false. Episode videos
                            and scalar metrics are unaffected.
  --bon-max-sample-attempts N
                            Best-of-N diverse-subtask search: max resample attempts per
                            candidate slot (beyond the first). Each attempt is a full VLA
                            forward pass; lower this to trade diversity for speed. Default: 6.
  --bon-cot-temperature F   CoT/subtask sampling temperature when best-of-N is active. The
                            base config defaults to 0.0 (greedy -- every candidate would
                            decode to the same subtask), so this defaults to 1.0 here,
                            matching run_carla_teleop.sh's manual candidate picker.
  --bon-critic-rollout-chunk BOOL
                            true|false. When --bon-critic-ckpt or --bon-online-critic is
                            active, execute the FULL selected action chunk (all
                            vla_action_horizon steps, shifted one step at a time) across
                            consecutive env steps instead of re-sampling N candidates and
                            re-scoring with the critic every single step. Only re-scores
                            once the chunk is exhausted (or on episode reset). Mirrors
                            --bon-gemini-rollout-chunk. Default: false (re-score every step).

  --save-video-local BOOL   true|false. Write each episode's rollout video to
                            <save_dir>/videos/epNNNN.mp4 locally (in addition to W&B). Default: true.

  --fail2drive-carla-root PATH
                            Override CARLA install used for Fail2Drive routes.
                            Default: \$FAIL2DRIVE_CARLA_ROOT (unset -> leave CARLA_ROOT alone)
  -h, --help                Show this help

Examples:
  # Pi0 residual SAC (default) on GPU 0/1:
  bash run_carla.sh --route parking-cut-in-001 --train-gpu 0 --sim-gpu 1
  # DAgger residual (supervised):
  bash run_carla.sh --train-mode dagger_residual --critic-mode delta
  # Standard DSRL RL:
  bash run_carla.sh --train-mode rl --agent-config impls/configs/steervla_dsrl_config.py
  # Expert debug (no RL):
  bash run_carla.sh --expert-debug true --save-buffer false
  # Second instance on different ports:
  bash run_carla.sh --carla-port 2002 --carla-streaming-port 2003 --tm-port 8002 --x-display-num 12
  # Best-of-N action selection from the frozen pi0 policy (no residual), scored by a
  # pretrained critic, eval-only:
  bash run_carla.sh --route 3936 --eval-only true --train-mode rl \
      --bon-critic-ckpt /path/to/critic.pkl --bon-num-candidates 8
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --route) ROUTE="$2"; shift 2 ;;
    --online-steps) ONLINE_STEPS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --run-group) RUN_GROUP="$2"; shift 2 ;;
    --save-buffer) SAVE_BUFFER="$2"; shift 2 ;;
    --expert-debug) EXPERT_DEBUG="$2"; shift 2 ;;
    --expert-recover-debug) EXPERT_RECOVER_DEBUG="$2"; shift 2 ;;
    --wandb-mode) WANDB_MODE="$2"; shift 2 ;;
    --train-gpu) TRAIN_GPU_RANK="$2"; shift 2 ;;
    --sim-gpu) SIM_GPU_RANK="$2"; shift 2 ;;
    --render-adapter) RENDER_ADAPTER="$2"; shift 2 ;;
    --carla-host) CARLA_HOST="$2"; shift 2 ;;
    --carla-port) CARLA_PORT="$2"; shift 2 ;;
    --carla-streaming-port) CARLA_STREAMING_PORT="$2"; shift 2 ;;
    --tm-port) TM_PORT="$2"; shift 2 ;;
    --x-display-num) X_DISPLAY_NUM="$2"; shift 2 ;;
    --critic-mode) CRITIC_MODE="$2"; shift 2 ;;
    --train-mode) TRAIN_MODE="$2"; shift 2 ;;
    --include-proprio) INCLUDE_PROPRIO="$2"; shift 2 ;;
    --discount) DISCOUNT="$2"; shift 2 ;;
    --hl-gpu) HL_TRAIN_GPU_RANK="$2"; shift 2 ;;
    --hl-ckpt-dir|--hl_ckpt_dir) HL_CKPT_DIR="$2"; shift 2 ;;
    --hl-ckpt-every|--hl_ckpt_every) HL_CKPT_EVERY="$2"; shift 2 ;;
    --hl-ckpt-keep-last|--hl_ckpt_keep_last) HL_CKPT_KEEP_LAST="$2"; shift 2 ;;
    --enable-updates) ENABLE_UPDATES="$2"; shift 2 ;;
    --base-only) BASE_ONLY="$2"; shift 2 ;;
    --state-encoder) STATE_ENCODER="$2"; shift 2 ;;
    --rlt-checkpoint) RLT_CHECKPOINT="$2"; shift 2 ;;
    --max-retries) MAX_RETRIES="$2"; shift 2 ;;
    --exp-name|--exp_name) EXP_NAME_OVERRIDE="$2"; shift 2 ;;
    --agent-config) BASE_AGENT_CFG="$2"; shift 2 ;;
    --carla-config) BASE_CARLA_CFG="$2"; shift 2 ;;
    --steervla-checkpoint|--steervla_checkpoint) STEERVLA_CKPT="$2"; shift 2 ;;
    --actor-config|--actor_config) ACTOR_CONFIG="$2"; shift 2 ;;
    --grpo-warmup|--grpo_warmup) GRPO_WARMUP="$2"; shift 2 ;;
    --grpo-select|--grpo_select) GRPO_SELECT="$2"; shift 2 ;;
    --grpo-group-size|--grpo_group_size) GRPO_GROUP_SIZE="$2"; shift 2 ;;
    --grpo-score-temp|--grpo_score_temp) GRPO_SCORE_TEMP="$2"; shift 2 ;;
    --cot-temperature|--cot_temperature) COT_TEMPERATURE="$2"; shift 2 ;;
    --pretrained-critic|--pretrained_critic) PRETRAINED_CRITIC="$2"; shift 2 ;;
    --qgf-critic-ckpt|--qgf_critic_ckpt) QGF_CRITIC_CKPT="$2"; shift 2 ;;
    --qgf-guidance-weight|--qgf_guidance_weight) QGF_GUIDANCE_WEIGHT="$2"; shift 2 ;;
    --bon-critic-ckpt|--bon_critic_ckpt) BON_CRITIC_CKPT="$2"; shift 2 ;;
    --bon-num-candidates|--bon_num_candidates) BON_NUM_CANDIDATES="$2"; shift 2 ;;
    --bon-online-critic|--bon_online_critic) BON_ONLINE_CRITIC="$2"; shift 2 ;;
    --bon-candidates-log-every|--bon_candidates_log_every) BON_CANDIDATES_LOG_EVERY="$2"; shift 2 ;;
    --bon-candidates-wandb|--bon_candidates_wandb) BON_CANDIDATES_WANDB="$2"; shift 2 ;;
    --bon-max-sample-attempts|--bon_max_sample_attempts) BON_MAX_SAMPLE_ATTEMPTS="$2"; shift 2 ;;
    --bon-cot-temperature|--bon_cot_temperature) BON_COT_TEMPERATURE="$2"; shift 2 ;;
    --bon-shadow-only|--bon_shadow_only) BON_SHADOW_ONLY="$2"; shift 2 ;;
    --bon-critic-rollout-chunk|--bon_critic_rollout_chunk) BON_CRITIC_ROLLOUT_CHUNK="$2"; shift 2 ;;
    --pid-brake-speed|--pid_brake_speed) PID_BRAKE_SPEED="$2"; shift 2 ;;
    --pid-stuck-threshold|--pid_stuck_threshold) PID_STUCK_THRESHOLD="$2"; shift 2 ;;
    --pid-creep-duration|--pid_creep_duration) PID_CREEP_DURATION="$2"; shift 2 ;;
    --pid-creep-throttle|--pid_creep_throttle) PID_CREEP_THROTTLE="$2"; shift 2 ;;
    --bon-gemini-select|--bon_gemini_select) BON_GEMINI_SELECT="$2"; shift 2 ;;
    --gemini-model|--gemini_model) GEMINI_MODEL="$2"; shift 2 ;;
    --bon-gemini-rollout-chunk|--bon_gemini_rollout_chunk) BON_GEMINI_ROLLOUT_CHUNK="$2"; shift 2 ;;
    --bon-qwen-select|--bon_qwen_select) BON_QWEN_SELECT="$2"; shift 2 ;;
    --qwen-bon-url|--qwen_bon_url) QWEN_BON_URL="$2"; shift 2 ;;
    --bon-qwen-cadence|--bon_qwen_cadence) BON_QWEN_CADENCE="$2"; shift 2 ;;
    --qwen-online-train|--qwen_online_train) QWEN_ONLINE_TRAIN="$2"; shift 2 ;;
    --qwen-online-warmup-episodes|--qwen_online_warmup_episodes) QWEN_ONLINE_WARMUP_EPISODES="$2"; shift 2 ;;
    --max-episodes|--max_episodes) MAX_EPISODES="$2"; shift 2 ;;
    --terminate-on-collision|--terminate_on_collision) TERMINATE_ON_COLLISION="$2"; shift 2 ;;
    --expert-controller|--expert_controller) EXPERT_CONTROLLER="$2"; shift 2 ;;
    --save-video-local|--save_video_local) SAVE_VIDEO_LOCAL="$2"; shift 2 ;;
    --eval-only|--eval_only) EVAL_ONLY="$2"; shift 2 ;;
    --fail2drive-carla-root) FAIL2DRIVE_CARLA_ROOT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

# SIM_GPU_RANK defaults to TRAIN_GPU_RANK so training and rendering share a GPU.
if [[ -z "$SIM_GPU_RANK" ]]; then
  SIM_GPU_RANK="$TRAIN_GPU_RANK"
fi

case "$CRITIC_MODE" in
  none) CRITIC_FEEDBACK_MODE="none" ;;
  expert) CRITIC_FEEDBACK_MODE="expert_action" ;;
  delta) CRITIC_FEEDBACK_MODE="action_delta" ;;
  delta-lang) CRITIC_FEEDBACK_MODE="delta_commentary_bow" ;;
  expert-lang) CRITIC_FEEDBACK_MODE="commentary_bow" ;;
  *)
    echo "Invalid --critic-mode: $CRITIC_MODE" >&2
    echo "Expected one of: none, expert, delta, delta-lang, expert-lang" >&2
    exit 2
    ;;
esac

case "$TRAIN_MODE" in
  rl|dagger|sac_residual|dagger_residual|grpo_hl) ;;
  *)
    echo "Invalid --train-mode: $TRAIN_MODE" >&2
    echo "Expected one of: rl, dagger, sac_residual, dagger_residual, grpo_hl" >&2
    exit 2
    ;;
esac

# OPT-IN: run the CARLA env against a separate 0.9.15 server + .venv-carla-0915
# subprocess, which avoids CARLA 0.9.16 segfaults on Town12/Town13 and the associated
# wire-protocol mismatch. The main JAX training process stays on Python 3.11 (.venv);
# only the CARLA env subprocess uses Python 3.10 (.venv-carla-0915) with carla 0.9.15.
#
# Machine-specific, so this is off unless you point CARLA_0915_ROOT at the install
# (routing-commands hardcoded /home/celinet/VLA_driving/software and hard-exited when it
# was missing, which broke every other box). Unset -> stay on CARLA_ROOT.
CARLA_0915_ROOT="${CARLA_0915_ROOT:-}"
_CARLA_0915_PYTHON="${CARLA_0915_PYTHON:-${ROOT_DIR}/.venv-carla-0915/bin/python}"

if [[ -n "$CARLA_0915_ROOT" ]]; then
  if [[ ! -d "$CARLA_0915_ROOT" ]]; then
    echo "[run_carla.sh] ERROR: CARLA_0915_ROOT=${CARLA_0915_ROOT} does not exist." >&2
    exit 1
  fi
  if [[ ! -x "$_CARLA_0915_PYTHON" ]]; then
    echo "[run_carla.sh] ERROR: 0.9.15 env python not found at $_CARLA_0915_PYTHON;" >&2
    echo "                      run setup first or set CARLA_0915_PYTHON." >&2
    exit 1
  fi
  export CARLA_ROOT="$CARLA_0915_ROOT"
  export CARLA_ENV_SUBPROCESS_PYTHON="$_CARLA_0915_PYTHON"
  echo "[run_carla.sh] CARLA 0.9.15 mode: CARLA_ROOT=${CARLA_ROOT}" >&2
fi

if [[ "$INCLUDE_PROPRIO" == "true" ]]; then
  _INCLUDE_PROPRIO_PY="True"
else
  _INCLUDE_PROPRIO_PY="False"
fi

case "$ENABLE_UPDATES" in
  true|false) ;;
  *) echo "Invalid --enable-updates: $ENABLE_UPDATES (expected true|false)" >&2; exit 2 ;;
esac
case "$BASE_ONLY" in
  ""|true|false) ;;
  *) echo "Invalid --base-only: $BASE_ONLY (expected true|false)" >&2; exit 2 ;;
esac

TMP_ROOT="$ROOT_DIR/.run_carla"
mkdir -p "$TMP_ROOT"
TMP_DIR="$(mktemp -d "$TMP_ROOT/run.XXXXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT

AGENT_CFG_TMP="$TMP_DIR/agent_config.py"
CARLA_CFG_TMP="$TMP_DIR/carla_config.yaml"

BASE_AGENT_CFG_ABS="$(cd "$(dirname "$BASE_AGENT_CFG")" && pwd)/$(basename "$BASE_AGENT_CFG")"
BASE_CARLA_CFG_ABS="$(cd "$(dirname "$BASE_CARLA_CFG")" && pwd)/$(basename "$BASE_CARLA_CFG")"

if [[ -n "$RENDER_ADAPTER" ]]; then
  SIM_GPU_RANK="$RENDER_ADAPTER"
fi

# Derive ports from SIM_GPU_RANK when not explicitly set (offset = gpu * 20)
_GPU_OFFSET=$((SIM_GPU_RANK * 20))
CARLA_PORT="${CARLA_PORT:-$((2000 + _GPU_OFFSET))}"
TM_PORT="${TM_PORT:-$((CARLA_PORT + 6000))}"
CARLA_STREAMING_PORT="${CARLA_STREAMING_PORT:-0}"

# Derive X display from CARLA_PORT (must be after port derivation)
if [[ -z "$X_DISPLAY_NUM" ]]; then
  X_DISPLAY_NUM="$((10 + (CARLA_PORT % 90)))"
fi

STATE_ENCODER_LINE=""
if [[ -n "$STATE_ENCODER" ]]; then
  STATE_ENCODER_LINE="config.state_encoder = \"${STATE_ENCODER}\""
fi
RLT_CHECKPOINT_LINE=""
if [[ -n "$RLT_CHECKPOINT" ]]; then
  RLT_CHECKPOINT_LINE="config.rl_token.checkpoint_path = r\"${RLT_CHECKPOINT}\""
fi
BASE_ONLY_LINE=""
if [[ -n "$BASE_ONLY" ]]; then
  BASE_ONLY_LINE="config.base_only = ${BASE_ONLY^}"
fi

cat > "$AGENT_CFG_TMP" <<EOF
from pathlib import Path
import runpy

_BASE_PATH = Path(r"${BASE_AGENT_CFG_ABS}")
_BASE_GET_CONFIG = runpy.run_path(str(_BASE_PATH))["get_config"]


def get_config():
    config = _BASE_GET_CONFIG()
    config.training_gpu_rank = ${TRAIN_GPU_RANK}
    config.siglip_device = "cuda:${TRAIN_GPU_RANK}"
    _HL_GPU = "${HL_TRAIN_GPU_RANK}"
    if _HL_GPU != "" and "steervla" in config:
        # Isolate the HL (VLM-backbone) update on its own GPU. See steervla_cast_relabel_config.py.
        config.steervla.hl_training_gpu_rank = int(_HL_GPU)
    # Policy checkpointing for later inference (params-only). Empty -> keep the config's value.
    _HL_CKPT_DIR = r"${HL_CKPT_DIR}"
    _HL_CKPT_EVERY = "${HL_CKPT_EVERY}"
    _HL_CKPT_KEEP = "${HL_CKPT_KEEP_LAST}"
    # Base-policy overrides (e.g. deploy a relabelled Stage-1 ckpt). Empty -> keep config value.
    _STEERVLA_CKPT = r"${STEERVLA_CKPT}"
    _ACTOR_CONFIG = r"${ACTOR_CONFIG}"
    _COT_TEMP = "${COT_TEMPERATURE}"
    if "steervla" in config:
        if _STEERVLA_CKPT != "":
            config.steervla.checkpoint = _STEERVLA_CKPT
        if _ACTOR_CONFIG != "":
            config.steervla.actor_config = _ACTOR_CONFIG
        if _COT_TEMP != "":
            config.steervla.cot_temperature = float(_COT_TEMP)
        if _HL_CKPT_DIR != "":
            config.steervla.hl_checkpoint_dir = _HL_CKPT_DIR
        if _HL_CKPT_EVERY != "":
            config.steervla.hl_checkpoint_every_steps = int(_HL_CKPT_EVERY)
        if _HL_CKPT_KEEP != "":
            config.steervla.hl_checkpoint_keep_last = int(_HL_CKPT_KEEP)
    _GRPO_WARMUP = "${GRPO_WARMUP}"
    if _GRPO_WARMUP != "" and "grpo" in config:
        config.grpo.warmup_steps = int(_GRPO_WARMUP)
    _GRPO_SELECT = "${GRPO_SELECT}"
    if _GRPO_SELECT != "" and "grpo" in config:
        config.grpo.select_mode = _GRPO_SELECT
    _GRPO_GROUP_SIZE = "${GRPO_GROUP_SIZE}"
    if _GRPO_GROUP_SIZE != "" and "grpo" in config:
        config.grpo.group_size = int(_GRPO_GROUP_SIZE)
    _GRPO_SCORE_TEMP = "${GRPO_SCORE_TEMP}"
    if _GRPO_SCORE_TEMP != "" and "grpo" in config:
        config.grpo.score_temperature = float(_GRPO_SCORE_TEMP)
    config.enable_updates = ${ENABLE_UPDATES^}
    config.critic_feedback_mode = "${CRITIC_FEEDBACK_MODE}"
    config.online_training_mode = "${TRAIN_MODE}"
    if config.critic_feedback_mode == "none":
        config.language_label_dim = 0
    elif config.critic_feedback_mode == "expert_action":
        config.language_label_dim = 2  # accel_steer 2D; no validity flag (matches pretrained critic)
    config.residual_append_state = ${_INCLUDE_PROPRIO_PY}
    config.discount = ${DISCOUNT}
    # Residual / RL-token knobs (surya/rl-token); empty unless the matching flag was passed.
    ${STATE_ENCODER_LINE}
    ${RLT_CHECKPOINT_LINE}
    ${BASE_ONLY_LINE}
$( [[ -n "$BON_CRITIC_CKPT" || "$BON_ONLINE_CRITIC" == "true" || "$BON_GEMINI_SELECT" == "true" || "$BON_QWEN_SELECT" == "true" ]] && \
   echo "    config.steervla.cot_temperature = ${BON_COT_TEMPERATURE}" )
    return config
EOF

uv run python - <<EOF
from pathlib import Path
import yaml

base_path = Path(r"${BASE_CARLA_CFG_ABS}")
cfg = yaml.safe_load(base_path.read_text())
cfg["host"] = "${CARLA_HOST}"
cfg["port"] = int("${CARLA_PORT}")
cfg["streaming_port"] = int("${CARLA_STREAMING_PORT}")
cfg["traffic_manager_port"] = int("${TM_PORT}")
cfg["gpu_rank"] = int("${SIM_GPU_RANK}")
cfg["x_display_num"] = int("${X_DISPLAY_NUM}")
cfg["use_cuda_visible_devices"] = False
if r"${EXPERT_CONTROLLER}":
    cfg["expert_controller"] = r"${EXPERT_CONTROLLER}"
Path(r"${CARLA_CFG_TMP}").write_text(yaml.safe_dump(cfg, sort_keys=False))
EOF

# Resolve route source (bench2drive | fail2drive) via the registry, then only
# override CARLA_ROOT if FAIL2DRIVE_CARLA_ROOT is explicitly set (e.g. you want
# to point Fail2Drive routes at a separate 0.9.15 install). By default the
# vanilla 0.9.16 install ships the Fail2Drive asset packs we copied in, so we
# leave CARLA_ROOT alone and both benchmarks talk to the same server.
ROUTE_SOURCE="$(uv run python -c "from ogbench.carla.route_registry import find_route; print(find_route('${ROUTE}').source)" 2>/dev/null || true)"
if [[ "$ROUTE_SOURCE" == "fail2drive" && -n "$FAIL2DRIVE_CARLA_ROOT" ]]; then
  if [[ ! -d "$FAIL2DRIVE_CARLA_ROOT" ]]; then
    echo "[run_carla.sh] WARNING: FAIL2DRIVE_CARLA_ROOT=${FAIL2DRIVE_CARLA_ROOT} doesn't exist; not switching CARLA_ROOT." >&2
  else
    export CARLA_ROOT="$FAIL2DRIVE_CARLA_ROOT"
    export CARLA_PYTHON_API_ROOT="$FAIL2DRIVE_CARLA_ROOT/PythonAPI/carla"
    echo "[run_carla.sh] Fail2Drive route detected: CARLA_ROOT=${CARLA_ROOT}"
    echo "[run_carla.sh]   (relaunch the CARLA server from ${CARLA_ROOT}/CarlaUE4.sh to match)"
  fi
fi

echo "[run_carla.sh] route=${ROUTE} (source=${ROUTE_SOURCE:-?})"
echo "[run_carla.sh] train_mode=${TRAIN_MODE}"
echo "[run_carla.sh] critic_mode=${CRITIC_FEEDBACK_MODE}"
echo "[run_carla.sh] train_gpu_rank=${TRAIN_GPU_RANK} render_adapter=${SIM_GPU_RANK}"
echo "[run_carla.sh] carla_host=${CARLA_HOST} carla_port=${CARLA_PORT} streaming_port=${CARLA_STREAMING_PORT} tm_port=${TM_PORT} x_display=:${X_DISPLAY_NUM}"
echo "[run_carla.sh] expert_debug=${EXPERT_DEBUG} expert_recover_debug=${EXPERT_RECOVER_DEBUG} save_buffer=${SAVE_BUFFER} online_steps=${ONLINE_STEPS} include_proprio=${INCLUDE_PROPRIO}"
echo "[run_carla.sh] temp agent config: ${AGENT_CFG_TMP}"
echo "[run_carla.sh] temp carla config: ${CARLA_CFG_TMP}"
if [[ -n "$BON_CRITIC_CKPT" ]]; then
  echo "[run_carla.sh] best-of-N: ckpt=${BON_CRITIC_CKPT} num_candidates=${BON_NUM_CANDIDATES}"
fi
echo "[run_carla.sh] save_video_local=${SAVE_VIDEO_LOCAL}"

# Export rather than prefixing the invocation: the run now happens inside the crash-supervisor
# loop below, so a one-shot `VAR=... cmd` prefix here would only have applied to the next echo.
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla:${ROOT_DIR}/simlingo-rebuttal${PYTHONPATH:+:$PYTHONPATH}"

echo "[run_carla.sh] agent_config=${BASE_AGENT_CFG}${STEERVLA_CKPT:+ steervla_checkpoint=${STEERVLA_CKPT}}${ACTOR_CONFIG:+ actor_config=${ACTOR_CONFIG}}"
echo "[run_carla.sh] enable_updates=${ENABLE_UPDATES} base_only=${BASE_ONLY:-<config default>} state_encoder=${STATE_ENCODER:-<config default>}${RLT_CHECKPOINT:+ rlt_checkpoint=${RLT_CHECKPOINT}}"
echo "[run_carla.sh] hl_gpu_rank=${HL_TRAIN_GPU_RANK:-<config>} max_retries=${MAX_RETRIES}${GRPO_WARMUP:+ grpo_warmup=${GRPO_WARMUP}}${GRPO_SELECT:+ grpo_select=${GRPO_SELECT}}${GRPO_GROUP_SIZE:+ grpo_group_size=${GRPO_GROUP_SIZE}}${GRPO_SCORE_TEMP:+ grpo_score_temp=${GRPO_SCORE_TEMP}}${COT_TEMPERATURE:+ cot_temperature=${COT_TEMPERATURE}}"
echo "[run_carla.sh] hl_ckpt_dir=${HL_CKPT_DIR:-<config>} hl_ckpt_every=${HL_CKPT_EVERY:-<config>} hl_ckpt_keep_last=${HL_CKPT_KEEP_LAST:-<config>}"

# Stable run name so save_dir + the W&B run id survive restarts (the supervisor resumes,
# it does not fork a new run).
ROUTE_TAG="$(printf '%s' "$ROUTE" | tr -c 'A-Za-z0-9._-' '-' | sed 's/-\{2,\}/-/g; s/^-//; s/-$//')"
EXP_NAME="${EXP_NAME_OVERRIDE:-${ROUTE_TAG}-sd$(printf '%03d' "$SEED")_$(date +%Y%m%d_%H%M%S)}"
echo "[run_carla.sh] exp_name=${EXP_NAME}"

# Supervisor loop (surya/rl-token): CARLA's leaderboard runs in-process and periodically
# segfaults on long runs, killing python outright so no in-process try/except can catch it.
# On a crash-signal exit (>=128) we kill any orphaned CARLA/Xvfb on THIS run's ports and
# relaunch with --resume=true. A clean exit (0), SIGINT (130), or a non-crash error (<128,
# e.g. a config bug) stops the loop.
attempt=0
RESUME_FLAG="false"
while :; do
  set +e
  WANDB_MODE="${WANDB_MODE}" uv run python impls/main_carla.py \
      --agent="${AGENT_CFG_TMP}" \
    --carla_config="${CARLA_CFG_TMP}" \
    --route="${ROUTE}" \
    --online_steps="${ONLINE_STEPS}" \
    --save_buffer="${SAVE_BUFFER}" \
    --seed="${SEED}" \
    --run_group="${RUN_GROUP}" \
    --expert_debug="${EXPERT_DEBUG}" \
    --expert_recover_debug="${EXPERT_RECOVER_DEBUG}" \
    ${PRETRAINED_CRITIC:+--pretrained_critic="${PRETRAINED_CRITIC}"} \
    ${QGF_CRITIC_CKPT:+--qgf_critic_ckpt="${QGF_CRITIC_CKPT}"} \
    ${QGF_GUIDANCE_WEIGHT:+--qgf_guidance_weight="${QGF_GUIDANCE_WEIGHT}"} \
    ${BON_CRITIC_CKPT:+--bon_critic_ckpt="${BON_CRITIC_CKPT}"} \
    --bon_num_candidates="${BON_NUM_CANDIDATES}" \
    --bon_online_critic="${BON_ONLINE_CRITIC}" \
    --bon_candidates_log_every="${BON_CANDIDATES_LOG_EVERY}" \
    --bon_candidates_wandb="${BON_CANDIDATES_WANDB}" \
    --bon_max_sample_attempts="${BON_MAX_SAMPLE_ATTEMPTS}" \
    --bon_shadow_only="${BON_SHADOW_ONLY}" \
    --bon_critic_rollout_chunk="${BON_CRITIC_ROLLOUT_CHUNK}" \
    ${PID_BRAKE_SPEED:+--pid_brake_speed="${PID_BRAKE_SPEED}"} \
    ${PID_STUCK_THRESHOLD:+--pid_stuck_threshold="${PID_STUCK_THRESHOLD}"} \
    ${PID_CREEP_DURATION:+--pid_creep_duration="${PID_CREEP_DURATION}"} \
    ${PID_CREEP_THROTTLE:+--pid_creep_throttle="${PID_CREEP_THROTTLE}"} \
    --bon_gemini_select="${BON_GEMINI_SELECT}" \
    ${GEMINI_MODEL:+--gemini_model="${GEMINI_MODEL}"} \
    --bon_gemini_rollout_chunk="${BON_GEMINI_ROLLOUT_CHUNK}" \
    --bon_qwen_select="${BON_QWEN_SELECT}" \
    --qwen_bon_url="${QWEN_BON_URL}" \
    --bon_qwen_cadence="${BON_QWEN_CADENCE}" \
    --qwen_online_train="${QWEN_ONLINE_TRAIN}" \
    --qwen_online_warmup_episodes="${QWEN_ONLINE_WARMUP_EPISODES}" \
    ${MAX_EPISODES:+--max_episodes="${MAX_EPISODES}"} \
    --terminate_on_collision="${TERMINATE_ON_COLLISION}" \
    --save_video_local="${SAVE_VIDEO_LOCAL}" \
    --eval_only="${EVAL_ONLY}" \
    "${EXTRA_ARGS[@]}" \
    --exp_name="${EXP_NAME}" \
    --resume="${RESUME_FLAG}"
  CODE=$?
  set -e

  if [[ $CODE -eq 0 ]]; then
    echo "[run_carla.sh] run completed (exit 0)."
    break
  fi
  if [[ $CODE -eq 130 ]]; then
    echo "[run_carla.sh] interrupted (SIGINT); not restarting."
    exit 130
  fi
  if [[ $CODE -lt 128 ]]; then
    echo "[run_carla.sh] exited with code ${CODE} (not a crash signal); not restarting."
    exit "$CODE"
  fi
  attempt=$((attempt + 1))
  if [[ "$MAX_RETRIES" -le 0 || $attempt -gt "$MAX_RETRIES" ]]; then
    echo "[run_carla.sh] crash (exit ${CODE}); retry budget exhausted (${attempt}/${MAX_RETRIES}). Giving up."
    exit "$CODE"
  fi
  echo "[run_carla.sh] main_carla crashed (exit ${CODE}, likely CARLA native SIGSEGV/SIGABRT); cleaning up + resuming (attempt ${attempt}/${MAX_RETRIES})."
  pkill -9 -f "CarlaUE4.*-carla-rpc-port=${CARLA_PORT}" 2>/dev/null || true
  pkill -9 -f "Xvfb :${X_DISPLAY_NUM} " 2>/dev/null || true
  sleep 8
  RESUME_FLAG="true"
done
