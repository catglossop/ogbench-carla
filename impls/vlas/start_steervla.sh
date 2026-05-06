#!/bin/bash
set -euo pipefail

USER=carla

cd /home/$USER/ogbench-carla
export PATH="$HOME/.local/bin:$PATH"

# # Install crcmod for fast GCS transfers if not already built with C extension.
# if ! python3 -c "import crcmod; assert crcmod._usingExtension" &>/dev/null; then
#     echo "Installing native crcmod..."
#     sudo apt-get install -y gcc python3-dev python3-setuptools
#     sudo pip3 uninstall -y crcmod || true
#     sudo pip3 install --no-cache-dir -U crcmod
# else
#     echo "Native crcmod already installed, skipping."
# fi

# Install uv only if not already present.
if ! command -v uv &>/dev/null; then
    echo "Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
else
    echo "UV already installed, skipping."
fi

# Only run uv sync if the venv doesn't exist or pyproject.toml is newer than the venv.
if [ ! -d .venv ] || [ pyproject.toml -nt .venv ]; then
    echo "Syncing virtual environment..."
    uv sync --extra all-tpu
else
    echo "Virtual environment up to date, skipping sync."
fi

WANDB_API_KEY="${1:?WANDB_API_KEY is required as first argument}"
HF_TOKEN="${2:?HF_TOKEN is required as second argument}"
CONFIG_NAME="${3:?CONFIG_NAME is required as third argument}"
CHECKPOINT_PATH="${4:?CHECKPOINT_PATH is required as fourth argument}"

source .venv/bin/activate

# libtpu/JAX writes driver logs under /tmp/tpu_logs (hardcoded). After a root job or system
# package, that dir can be root-owned and triggers "Permission denied" spam; fix ownership.
fix_tpu_logs_dir() {
    local d=/tmp/tpu_logs
    if [[ -d "$d" && ! -w "$d" ]]; then
        echo "Making $d writable for $(whoami) (sudo will prompt if needed)..."
        if ! sudo chown -R "$(id -un):$(id -gn)" "$d"; then
            echo >&2 "ERROR: could not chown $d. Run once: sudo chown -R \$(id -un):\$(id -gn) $d"
            return 1
        fi
    fi
    mkdir -p "$d"
    if [[ ! -w "$d" ]]; then
        echo >&2 "ERROR: cannot write to $d. Try: sudo rm -rf $d && mkdir -p $d && chown \$(id -un):\$(id -gn) $d"
        return 1
    fi
}
fix_tpu_logs_dir

# Log in only if not already authenticated.
echo "Logging into Weights & Biases..."
uv run wandb login "$WANDB_API_KEY"

echo "Logging into Hugging Face..."
hf auth login --token "$HF_TOKEN" --add-to-git-credential

echo "Starting inference server..."
XLA_PYTHON_CLIENT_PREALLOCATE="true" XLA_PYTHON_CLIENT_MEM_FRACTION="0.95" \ python impls/vlas/steervla_server.py --actor-config "$CONFIG_NAME" --checkpoint "$CHECKPOINT_PATH" --cot-jit-decode false --cot-jit-transformer-forward true
