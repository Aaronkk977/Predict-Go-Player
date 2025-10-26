#!/bin/bash
set -e

# GPU-enabled version of start-container.sh
# This script starts the container with NVIDIA GPU support

cd $(dirname "$0")/../..

# Check if nvidia-smi is available on host
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  WARNING: nvidia-smi not found on host system"
    echo "GPU support may not be available"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Detect container runtime
if command -v docker &> /dev/null; then
    CONTAINER_CMD="docker"
    GPU_FLAG="--gpus all"
elif command -v podman &> /dev/null; then
    CONTAINER_CMD="podman"
    GPU_FLAG="--device nvidia.com/gpu=all --security-opt=label=disable"
else
    echo "❌ Error: Neither docker nor podman is installed"
    exit 1
fi

echo "🚀 Starting container with GPU support..."
echo "Container runtime: ${CONTAINER_CMD}"
echo ""

# Start container with GPU support
${CONTAINER_CMD} run \
    ${GPU_FLAG} \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --network=host \
    --ipc=host \
    --rm \
    -it \
    -v $(pwd):/workspace \
    -v $(pwd)/ML-Assignment2-Q5:/strength-detection \
    kds285/strength-detection:latest

