#!/bin/bash

#!/bin/bash
set -euo pipefail

VM_NAME="${1:?Usage: $0 <vm-name> <config-name>}"
CONFIG_NAME="${2:?Usage: $0 <vm-name> <config-name>}"
CHECKPOINT_PATH="${3:?Usage: $0 <vm-name> <config-name> <checkpoint-path>}"
ZONE=us-central2-b
USER=carla

echo "Make sure you have the following environment variables set!!!
WANDB_API_KEY
HF_TOKEN"

echo "Launching TPU job on $VM_NAME with config=$CONFIG_NAME ..."

# Ensure SSH keys are provisioned on the VM (needed after VM recreation).
echo "Ensuring SSH keys are provisioned..."
gcloud compute tpus tpu-vm ssh "$VM_NAME" --zone="$ZONE" --command="echo ok" || {
    echo "ERROR: Cannot SSH into $VM_NAME. Is the TPU VM running?"
    exit 1
}

# Now get the SSH args for rsync via --dry-run.
SSH_CMD=$(gcloud compute tpus tpu-vm ssh "$VM_NAME" --zone="$ZONE" --dry-run 2>&1)
SSH_ARGS=$(echo "$SSH_CMD" | grep -oP "(?<=/usr/bin/ssh ).*(?= ${USER}@)") || {
    echo "ERROR: Could not parse SSH args from dry-run output:"
    echo "$SSH_CMD"
    exit 1
}
TARGET_HOST=$(echo "$SSH_CMD" | grep -oP "${USER}@[\d.]+") || {
    echo "ERROR: Could not parse target host from dry-run output:"
    echo "$SSH_CMD"
    exit 1
}

echo "Syncing to $TARGET_HOST ..."
rsync -avz \
    --exclude='.venv' \
    --exclude='ogbench-carla/exp' \
    --exclude='ogbench-carla/data' \
    --exclude='ogbench-carla/assets' \
    -e "/usr/bin/ssh ${SSH_ARGS//-t /}" \
    /home/$USER/ogbench-carla "$TARGET_HOST":/home/$USER

gcloud compute tpus tpu-vm ssh "$VM_NAME" --zone="$ZONE" \
    --command="bash ogbench-carla/impls/vlas/start_steervla.sh $WANDB_API_KEY $HF_TOKEN $CONFIG_NAME $CHECKPOINT_PATH"