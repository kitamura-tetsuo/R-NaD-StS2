#!/bin/bash

set -e

./deploy.sh
python3 R-NaD/inference_sts2.py --seed 1