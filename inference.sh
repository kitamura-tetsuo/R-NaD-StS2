#!/bin/bash

set -e
./deploy.sh

set +e
./reload_sim.sh
./R-NaD/venv/bin/python R-NaD/inference_sts2.py --seed 1 --route