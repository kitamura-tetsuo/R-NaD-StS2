#!/bin/bash

# --- Configuration ---
GAME_DIR="/home/ubuntu/.steam/debian-installation/steamapps/common/Slay the Spire 2"
MODS_DIR="$GAME_DIR/mods"
BIN_DIR="$GAME_DIR/bin"
SRC_ROOT="/home/ubuntu/src/R-NaD-StS2"

# --- 1. Create necessary directories and files ---
mkdir -p "$MODS_DIR"
mkdir -p "$BIN_DIR"

# Create steam_appid.txt to allow launching directly (Steam AppID for StS2: 2868840)
echo "2868840" > "$GAME_DIR/steam_appid.txt"

echo "=== Deploying R-NaD StS2 Mod ==="

# --- 2. Copy communication-mod files ---
echo "Deploying communication-mod..."
# DLL (Godot mono export path)
DLL_PATH="$SRC_ROOT/communication-mod/.godot/mono/temp/bin/ExportRelease/publish/communication-mod.dll"
cp "$DLL_PATH" "$MODS_DIR/"

# PCK (Regenerated in project root)
PCK_PATH="$SRC_ROOT/communication-mod/communication-mod.pck"
if [ -f "$PCK_PATH" ]; then
    cp "$PCK_PATH" "$MODS_DIR/"
else
    echo "Warning: communication-mod.pck not found!"
fi

# --- 3. Copy GDExtension files ---
echo "Deploying GDExtension..."
# .so binary (Linux)
cp "$SRC_ROOT/GDExtension/target/debug/libai_bridge.so" "$BIN_DIR/"
# .gdextension config (Root)
cp "$SRC_ROOT/GDExtension/ai_bridge.gdextension" "$GAME_DIR/"

echo "Deployment complete."

# --- 4. Launch Game ---
echo "Launching Slay the Spire 2..."
cd "$GAME_DIR"
./SlayTheSpire2 --verbose "$@"
