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

# Create steam_appid.txt to allow launching directly (Steam AppID for StS2: 2868840)
echo "2868840" > "$GAME_DIR/steam_appid.txt"

echo "=== Building and Deploying R-NaD StS2 Mod ==="

# --- 2. Build and Copy communication-mod files ---
echo "Building communication_mod..."
cd "$SRC_ROOT/communication-mod"
dotnet publish -c ExportRelease

echo "Deploying communication_mod..."
# DLL (Godot mono export path)
DLL_PATH="$SRC_ROOT/communication-mod/.godot/mono/temp/bin/ExportRelease/publish/communication_mod.dll"
cp "$DLL_PATH" "$MODS_DIR/"

# JSON Manifest
cp "$SRC_ROOT/communication-mod/communication_mod.json" "$MODS_DIR/"

# PCK (Expected to be exported by Megadot target or manually)
PCK_PATH="$SRC_ROOT/communication-mod/communication_mod.pck"
if [ -f "$PCK_PATH" ]; then
    cp "$PCK_PATH" "$MODS_DIR/"
fi

# Copy BaseLib.json too
BASELIB_JSON=$(find "$SRC_ROOT/communication-mod/packages/alchyr.sts2.baselib" -name "BaseLib.json" | head -n 1)
if [ -n "$BASELIB_JSON" ]; then
    cp "$BASELIB_JSON" "$MODS_DIR/"
fi

# --- 3. Build and Copy GDExtension files ---
echo "Building GDExtension..."
cd "$SRC_ROOT/GDExtension"
cargo build || echo "GDExtension build failed, skipping..."

echo "Deploying GDExtension..."
# .so binary (Linux)
cp "$SRC_ROOT/GDExtension/target/debug/libai_bridge.so" "$BIN_DIR/"
# .gdextension config (Root)
cp "$SRC_ROOT/GDExtension/ai_bridge.gdextension" "$GAME_DIR/"

# # --- 4. Build and Copy recorder-mod files ---
# echo "Building recorder_mod..."
# cd "$SRC_ROOT/recorder-mod"
# dotnet publish -c Release

# echo "Deploying recorder_mod..."
# # For Godot.NET.Sdk, it puts it in similar publish folder
# DLL_RECORDER_PATH="$SRC_ROOT/recorder-mod/.godot/mono/temp/bin/Release/publish/recorder_mod.dll"
# # If not there, check standard bin
# if [ ! -f "$DLL_RECORDER_PATH" ]; then
#     DLL_RECORDER_PATH="$SRC_ROOT/recorder-mod/bin/Release/net9.0/publish/recorder_mod.dll"
# fi
# cp "$DLL_RECORDER_PATH" "$MODS_DIR/"
# cp "$SRC_ROOT/recorder-mod/recorder_mod.json" "$MODS_DIR/"

echo "Deployment complete."
