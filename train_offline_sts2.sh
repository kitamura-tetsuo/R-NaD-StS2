#!/bin/bash

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
R_NAD_DIR="$SCRIPT_DIR/R-NaD"
PYTHON_BIN="$R_NAD_DIR/venv/bin/python"

# Set PYTHONPATH to include R-NaD directory for absolute imports
export PYTHONPATH=$R_NAD_DIR:$PYTHONPATH
export SKIP_RNAD_INIT=1

# Check if python exists
if [ ! -f "$PYTHON_BIN" ]; then
    echo "Error: Virtual environment python not found at $PYTHON_BIN"
    exit 1
fi

echo "Running Offline Training for StS2..."
"$PYTHON_BIN" "$R_NAD_DIR/train_offline_sts2.py" "$@"
