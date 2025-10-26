#!/bin/bash
set -e
# Change to repository root, then call start-container
cd $(dirname "$0")/../..
./ML-Assignment2-Q5/minizero/scripts/start-container.sh -v $(pwd):/workspace -v $(pwd)/ML-Assignment2-Q5:/strength-detection --image kds285/strength-detection $@
