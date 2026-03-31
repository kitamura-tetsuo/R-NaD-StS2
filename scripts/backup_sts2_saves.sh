#!/bin/bash

# Configuration
SOURCE_A="/home/ubuntu/.local/share/SlayTheSpire2/steam/76561198725031675/modded/profile1/saves"
SOURCE_B="/home/ubuntu/.local/share/Steam/userdata/764765947/2868840/remote/modded/profile1/saves"
BACKUP_ROOT="$HOME/sts2_backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_DIR="$BACKUP_ROOT/backup_$TIMESTAMP"

echo "Starting StS2 modded saves backup..."

# Create backup directory
mkdir -p "$BACKUP_DIR/AppData"
mkdir -p "$BACKUP_DIR/UserData"

# Backup Location A
if [ -d "$SOURCE_A" ]; then
    echo "Backing up Location A: $SOURCE_A"
    cp -rp "$SOURCE_A/." "$BACKUP_DIR/AppData/"
else
    echo "Warning: Location A not found: $SOURCE_A"
fi

# Backup Location B
if [ -d "$SOURCE_B" ]; then
    echo "Backing up Location B: $SOURCE_B"
    cp -rp "$SOURCE_B/." "$BACKUP_DIR/UserData/"
else
    echo "Warning: Location B not found: $SOURCE_B"
fi

echo "Backup completed: $BACKUP_DIR"
