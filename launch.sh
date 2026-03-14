#!/bin/bash

# --- Configuration ---
GAME_DIR="/home/ubuntu/.steam/steam/steamapps/common/Slay the Spire 2"

# --- 4. Launch Game ---
echo "Launching Slay the Spire 2..."
cd "$GAME_DIR"
# Use the direct binary. launch_vulkan.sh was causing segfaults.
# Redirecting to /tmp/sts2_stdout.log for E2E monitoring.
# Find libpython for LD_PRELOAD fix (required for NumPy C-extensions in embedded environment)
PYTHON_LIB=$(ls /usr/lib/x86_64-linux-gnu/libpython3.12.so.1.0 2>/dev/null || ls /usr/lib/libpython3.12.so.1.0 2>/dev/null)
echo "Using LD_PRELOAD=$PYTHON_LIB"
LD_PRELOAD="$PYTHON_LIB" ./SlayTheSpire2 --verbose "$@" > /tmp/sts2_stdout.log 2>&1
