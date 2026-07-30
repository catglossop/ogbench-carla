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

echo "[reset_carla] If nvidia-smi shows ERR! on a GPU:"
echo "  Primary/display GPUs cannot be reset live — reboot instead:"
echo "    sudo reboot"
echo "  Secondary GPUs (no display attached) may respond to:"
echo "    sudo nvidia-smi --gpu-reset -i <index>"
