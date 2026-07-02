#! /bin/bash

# Reset the Carla environment

# Stop the Carla server
pkill -f CarlaUE4.sh -9
pkill -f CarlaUE4-Linux-Shipping -9

# Kill the script
pkill -f main_carla -9

# Kill all virtual X servers started by CARLA/leaderboard
pkill -9 -f 'Xvfb :'
# Remove lock files for dead PIDs
for lock in /tmp/.X*-lock; do
  pid=$(tr -d ' \n' < "$lock" 2>/dev/null)
  kill -0 "$pid" 2>/dev/null || rm -f "$lock"
done

# Kill wandb helper processes that outlive a crashed training parent.
pkill -f 'wandb-core' -9 2>/dev/null || true
pkill -f 'wandb-xpu' -9 2>/dev/null || true

echo "[reset_carla] If ps shows <defunct> processes or nvidia-smi lists dead PIDs with VRAM:"
echo "  Zombies cannot be SIGKILL'd; GPU memory may stay until reboot."
echo "  Try: sudo reboot"
echo "  Primary/display GPUs cannot be reset live — reboot instead:"
echo "    sudo reboot"
echo "  Secondary GPUs (no display attached) may respond to:"
echo "    sudo nvidia-smi --gpu-reset -i <index>"
