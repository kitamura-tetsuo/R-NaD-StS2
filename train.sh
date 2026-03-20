#!/bin/bash

set -e

./deploy.sh
python3 R-NaD/train_sts2.py --seed 1 --no-speedup --route
