#!/bin/bash

# --- Configuration ---
GAME_DIR="/home/ubuntu/.steam/steam/steamapps/common/Slay the Spire 2"

# --- 4. Launch Game ---
echo "Launching Slay the Spire 2..."
cd "$GAME_DIR"
# Use the direct binary. launch_vulkan.sh was causing segfaults.
# Redirecting to /tmp/sts2_stdout.log for E2E monitoring.
./SlayTheSpire2 --verbose "$@" > /tmp/sts2_stdout.log 2>&1
