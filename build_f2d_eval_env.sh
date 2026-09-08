#!/usr/bin/env bash
# build_f2d_eval_env.sh — build the Python 3.10 env that can evaluate against Fail2Drive's
# own CARLA build (f2d_carla, 0.9.15).
#
# Why a second env at all
# -----------------------
# 10 of the 200 Fail2Drive routes (the Generalization_Animals ones) need
# ``walker.animal.*`` blueprints, which ONLY f2d_carla registers -- the vanilla 0.9.16
# install carries every other Fail2Drive asset but not those (they live in the cooked
# Walkers/WalkerFactory.uasset, and that file is NOT portable between 0.9.15 and 0.9.16:
# dropping it into a 0.9.16 tree makes every server boot die with
# "LowLevelFatalError: Unknown code token 30 ... WalkerFactory_C:GenerateDefinitions").
#
# So those routes must run against f2d_carla, which needs a **matching 0.9.15 client** --
# a 0.9.16 client segfaults (rc=139) against a 0.9.15 server. carla 0.9.15 ships no cp311
# wheel (PyPI stops at cp310), and the repo's main env is 3.11, hence Python 3.10 here.
#
# openpi pins ``requires-python = ">=3.11"``, but that is *almost* conservative: its whole
# tree byte-compiles under 3.10 and every one of its dependencies allows >=3.10. The only
# genuine 3.11 API in it is a single ``datetime.UTC`` (3.11+; 3.10 spells it
# ``datetime.timezone.utc``), which this script patches in a staged source copy. The
# upstream clone is never modified.
#
# Usage:  bash build_f2d_eval_env.sh
# Then:   ./run_leaderboard_f2d.sh --carla-root ~/f2d_carla \
#             --python /home/cglossop/ogbench-carla/.venv-f2d-eval/bin/python ...
set -euo pipefail

VENV="${F2D_EVAL_VENV:-/home/cglossop/ogbench-carla/.venv-f2d-eval}"
UPSTREAM="${OPENPI_SRC:-/home/cglossop/steervla-pi}"
SRC="$(mktemp -d)/openpi-src"
PY=3.10

[[ -d "$UPSTREAM" ]] || { echo "no openpi clone at $UPSTREAM (set OPENPI_SRC)" >&2; exit 1; }

echo "=== [1/7] staging a relaxed openpi source copy at $SRC ==="
mkdir -p "$SRC"
cp -a "$UPSTREAM/pyproject.toml" "$UPSTREAM/README.md" "$UPSTREAM/LICENSE" "$SRC/"
cp -a "$UPSTREAM/LICENSE_GEMMA.txt" "$SRC/" 2>/dev/null || true
cp -a "$UPSTREAM/src" "$UPSTREAM/packages" "$SRC/"
cp -a "$UPSTREAM/third_party" "$SRC/" 2>/dev/null || true
sed -i 's/^requires-python = ">=3.11"/requires-python = ">=3.10"/' "$SRC/pyproject.toml"
# The one real 3.11-only API in openpi. Runtime attribute, so a byte-compile check misses it.
sed -i 's/tzinfo=datetime\.UTC/tzinfo=datetime.timezone.utc/' "$SRC/src/openpi/shared/download.py"

echo "=== [2/7] creating $VENV (python $PY) ==="
uv venv --python "$PY" "$VENV"
VPY="$VENV/bin/python"
pipi() { uv pip install --python "$VPY" "$@"; }

echo "=== [3/7] openpi's pinned dependency set ==="
pipi \
  "ml-dtypes==0.4.1" "tensorstore==0.1.74" \
  "jax[cuda12]==0.5.3" "flax==0.10.2" "orbax-checkpoint==0.11.13" \
  "numpy>=1.22.4,<2.0.0" "transformers==4.53.2" \
  "augmax>=0.3.4" "beartype==0.19.0" "dm-tree>=0.1.8" "einops>=0.8.0" \
  "equinox>=0.11.8" "filelock>=3.16.1" "flatbuffers>=24.3.25" \
  "fsspec[gcs]>=2024.6.0" "imageio>=2.36.1" "jaxtyping==0.2.36" \
  "ml_collections==1.0.0" "numpydantic>=1.6.6" "opencv-python>=4.10.0.84" \
  "pillow>=11.0.0" "sentencepiece>=0.2.0" "tqdm-loggable>=0.2" \
  "typing-extensions>=4.12.2" "tyro>=0.9.5" "wandb>=0.19.1" \
  "treescope>=0.1.7" "rich>=14.0.0" "polars>=1.30.0" "gym-aloha>=0.1.1" \
  "pytest"   # openpi/models_pytorch/gemma_pytorch.py imports pytest at module level

echo "=== [4/7] lerobot at openpi's pinned rev, then openpi-client + openpi ==="
pipi "lerobot @ git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"
pipi --no-deps "$SRC/packages/openpi-client"
pipi --no-deps "$SRC"

echo "=== [5/7] CARLA 0.9.15 side + remaining ogbench runtime deps ==="
pipi "carla==0.9.15"
pipi "bench2drive @ git+https://github.com/catglossop/Bench2Drive.git"
pipi "fail2drive @ git+https://github.com/catglossop/fail2drive.git"
pipi "distrax>=0.1.5" "gymnasium" "matplotlib" "moviepy" "scipy" "pyyaml" \
     "absl-py" "tqdm" "google-genai" "shapely" "tabulate" "psutil"

echo "=== [6/7] pinning torch/torchvision to match the 3.11 env ==="
# lerobot drags torch to 2.14.0+cu130, which reports cuda_available=False on a 570 driver
# and installs a parallel CUDA-13 stack that shadows jax's cu12 libs. Force the same pair
# the 3.11 env uses, then repair the cu12 shared libs: the cu13 wheels install into the
# same nvidia/<lib>/lib directories, so uninstalling them deletes files the cu12 wheels own.
pipi "torch==2.7.1" "torchvision==0.22.1"
uv pip uninstall --python "$VPY" \
  nvidia-cublas nvidia-cuda-cupti nvidia-cuda-nvrtc nvidia-cuda-runtime nvidia-cudnn-cu13 \
  nvidia-cufft nvidia-cufile nvidia-curand nvidia-cusolver nvidia-cusparse \
  nvidia-cusparselt-cu13 nvidia-nccl-cu13 nvidia-nvjitlink nvidia-nvshmem-cu13 nvidia-nvtx \
  2>/dev/null || true
pipi --reinstall \
  "nvidia-cublas-cu12==12.6.4.1" "nvidia-cuda-cupti-cu12==12.6.80" "nvidia-cuda-nvcc-cu12==12.9.86" \
  "nvidia-cuda-nvrtc-cu12==12.6.77" "nvidia-cuda-runtime-cu12==12.6.77" "nvidia-cudnn-cu12==9.5.1.17" \
  "nvidia-cufft-cu12==11.3.0.4" "nvidia-cufile-cu12==1.11.1.6" "nvidia-curand-cu12==10.3.7.77" \
  "nvidia-cusolver-cu12==11.7.1.2" "nvidia-cusparse-cu12==12.5.4.2" "nvidia-cusparselt-cu12==0.6.3" \
  "nvidia-nccl-cu12==2.26.2" "nvidia-nvjitlink-cu12==12.6.85" "nvidia-nvtx-cu12==12.6.77"

echo "=== [7/7] smoke test ==="
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHONPATH="$REPO:$REPO/impls" CARLA_ROOT="${CARLA_ROOT:-/home/cglossop/f2d_carla}" "$VPY" - <<'PY'
import torch, torchvision, jax, jax.numpy as jnp, carla
print("torch", torch.__version__, "torchvision", torchvision.__version__,
      "cuda", torch.cuda.is_available())
print("jax devices", len(jax.devices()), "matmul", float((jnp.ones((256, 256)) @ jnp.ones((256, 256))).sum()))
print("carla client", carla.Client("localhost", 1).get_client_version())
from transformers import AutoProcessor  # needs torchvision matched to torch
from vlas.steervla import create_steervla_pi0_cot_sample_fn
import ogbench.carla.carla_utils  # noqa: F401
print("full main_carla import chain OK")
PY
rm -rf "$(dirname "$SRC")"
echo "=== DONE: $VENV ==="
