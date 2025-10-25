#!/bin/bash
set -e
./minizero/scripts/start-container.sh -v $(pwd):/strength-detection --image docker.io/kds285/strength-detection $@
