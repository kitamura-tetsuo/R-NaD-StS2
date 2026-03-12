#!/bin/bash

set -e

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

# PCK (Expected to be exported by Megadot target or manually)
# Note: The .csproj has a target that exports the pack to the mods folder.
# But we ensure it exists and copy if it's in the src root.
PCK_PATH="$SRC_ROOT/communication-mod/communication_mod.pck"
if [ -f "$PCK_PATH" ]; then
    cp "$PCK_PATH" "$MODS_DIR/"
fi

# --- 3. Build and Copy GDExtension files ---
echo "Building GDExtension..."
cd "$SRC_ROOT/GDExtension"
cargo build

echo "Deploying GDExtension..."
# .so binary (Linux)
cp "$SRC_ROOT/GDExtension/target/debug/libai_bridge.so" "$BIN_DIR/"
# .gdextension config (Root)
cp "$SRC_ROOT/GDExtension/ai_bridge.gdextension" "$GAME_DIR/"

echo "Deployment complete."

# --- 4. Launch Game ---
echo "Launching Slay the Spire 2..."
cd "$GAME_DIR"
# Use the direct binary. launch_vulkan.sh was causing segfaults.
./SlayTheSpire2 --verbose "$@"
