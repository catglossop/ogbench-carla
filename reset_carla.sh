#! /bin/bash

# Reset the Carla environment

# Stop the Carla server
pkill -f CarlaUE4.sh -9

# Kill the script
pkill -f main_carla -9

# Kill the displays

# Kill all virtual X servers started by CARLA/leaderboard
pkill -9 -f 'Xvfb :'
# Remove lock files for dead PIDs
for lock in /tmp/.X*-lock; do
  pid=$(tr -d ' \n' < "$lock" 2>/dev/null)
  kill -0 "$pid" 2>/dev/null || rm -f "$lock"
done