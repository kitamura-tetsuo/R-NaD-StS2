#!/bin/bash

set -e

./deploy.sh

set +e
./reload_sim.sh
./R-NaD/venv/bin/python R-NaD/train_sts2.py --no-speedup --mask-card-skip --ui

