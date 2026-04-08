#!/bin/bash

set -e

# Deploy the latest binaries (rebuild if needed)
./deploy.sh

set +e

# Ensure the simulator mirror is up to date
./reload_sim.sh

# Run the recording script with standard flags
./R-NaD/venv/bin/python R-NaD/record_sts2.py --no-speedup --mask-card-skip
