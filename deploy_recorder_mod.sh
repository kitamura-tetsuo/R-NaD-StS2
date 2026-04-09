#!/bin/bash

set -e

# --- 0. Cleanup existing processes ---
echo "=== Cleaning up existing processes ==="
pkill -9 SlayTheSpire2 || true
pkill -9 python3 || true
sleep 1

# --- Configuration ---
GAME_DIR="/home/ubuntu/.steam/steam/steamapps/common/Slay the Spire 2"
MODS_DIR="$GAME_DIR/mods"
BIN_DIR="$GAME_DIR/bin"
SRC_ROOT="/home/ubuntu/src/R-NaD-StS2"

# --- 1. Create necessary directories and files ---
echo "=== Preparing Directories ==="
mkdir -p "$MODS_DIR"
mkdir -p "$BIN_DIR"

# --- 4. Build and Copy recorder-mod files ---
echo "Building recorder_mod..."
cd "$SRC_ROOT/recorder-mod"
dotnet publish -c Release

echo "Deploying recorder_mod..."
# For Godot.NET.Sdk, it puts it in similar publish folder
DLL_RECORDER_PATH="$SRC_ROOT/recorder-mod/.godot/mono/temp/bin/Release/publish/recorder_mod.dll"
# If not there, check standard bin
if [ ! -f "$DLL_RECORDER_PATH" ]; then
    DLL_RECORDER_PATH="$SRC_ROOT/recorder-mod/bin/Release/net9.0/publish/recorder_mod.dll"
fi
cp "$DLL_RECORDER_PATH" "$MODS_DIR/"
cp "$SRC_ROOT/recorder-mod/recorder_mod.json" "$MODS_DIR/"

echo "Deployment complete."
