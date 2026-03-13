#!/bin/bash

set -e

./deploy.sh
./launch.sh "$@"
