#!/bin/bash

# Configuration
DEST_A="/home/ubuntu/.local/share/SlayTheSpire2/steam/76561198725031675/modded/profile1/saves"
DEST_B="/home/ubuntu/.local/share/Steam/userdata/764765947/2868840/remote/modded/profile1/saves"

if [ -z "$1" ]; then
    echo "Usage: $0 <backup_directory_path>"
    echo "Example: $0 ~/sts2_backups/backup_20240331_120000"
    exit 1
fi

BACKUP_DIR="$1"

if [ ! -d "$BACKUP_DIR" ]; then
    echo "Error: Backup directory not found: $BACKUP_DIR"
    exit 1
fi

echo "Warning: This will overwrite existing save data at Location A and Location B."
read -p "Are you sure you want to proceed? (y/n): " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Operation cancelled."
    exit 0
fi

# Restore Location A
if [ -d "$BACKUP_DIR/AppData" ]; then
    echo "Restoring Location A: $DEST_A"
    mkdir -p "$DEST_A"
    # Use -av to preserve attributes and provide verbose output
    cp -av "$BACKUP_DIR/AppData/." "$DEST_A/"
else
    echo "Skip: AppData not found in backup."
fi

# Restore Location B
if [ -d "$BACKUP_DIR/UserData" ]; then
    echo "Restoring Location B: $DEST_B"
    mkdir -p "$DEST_B"
    cp -av "$BACKUP_DIR/UserData/." "$DEST_B/"
else
    echo "Skip: UserData not found in backup."
fi

echo "Verifying files..."

# Validation
MISSING_A=0
MISSING_B=0

if [ ! -f "$DEST_A/current_run.save" ]; then
    echo "Warning: current_run.save is missing in Location A after restore!"
    MISSING_A=1
fi

if [ ! -f "$DEST_B/current_run.save" ]; then
    echo "Warning: current_run.save is missing in Location B after restore!"
    MISSING_B=1
fi

if [ $MISSING_A -eq 0 ] && [ $MISSING_B -eq 0 ]; then
    echo "Success: current_run.save found in both locations."
    echo "Restore completed successfully."
else
    echo "Error: Restore verification failed. Some save files are missing."
    exit 1
fi

