#!/bin/bash

# Configuration
SRC_ROOT="/home/ubuntu/src/R-NaD-StS2"
SIM_DIR="$SRC_ROOT/battle_simulator"
BRIDGE_DIR="$SRC_ROOT/R-NaD"
BRIDGE_URL="http://127.0.0.1:8081"

set -e

echo "=== 1. Building Simulator ==="
cd "$SIM_DIR"
cargo build --release

echo "=== 2. Versioning and Copying ==="
TIMESTAMP=$(date +%s)
NEW_SO="battle_simulator_v$TIMESTAMP.so"
cp "$SIM_DIR/target/release/libbattle_simulator.so" "$BRIDGE_DIR/$NEW_SO"
# Update the symlink for general use (though the bridge will load the versioned one)
ln -sf "$NEW_SO" "$BRIDGE_DIR/battle_simulator.so"

echo "=== 3. Signaling Bridge to Reload ==="
RESPONSE=$(curl -s "$BRIDGE_URL/reload_simulator?v=battle_simulator_v$TIMESTAMP")

if [[ "$RESPONSE" == *"success"* ]]; then
    echo "SUCCESS: Simulator reloaded to version $TIMESTAMP"
else
    echo "FAILURE: Bridge responded with: $RESPONSE"
    echo "Hint: If this is the first time setting up, you may need to restart the training loop once."
    exit 1
fi

