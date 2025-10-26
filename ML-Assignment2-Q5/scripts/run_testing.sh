#!/bin/bash

# Testing script for Go style detection
# This script performs inference and generates submission.csv

echo "======================================"
echo "Go Player Style Detection - Testing"
echo "======================================"

# Check if model path is provided
if [ -z "$1" ]; then
    echo "Usage: ./run_testing.sh <model_path>"
    echo "Example: ./run_testing.sh ./models/style_model_20231026_final.pt"
    exit 1
fi

MODEL_PATH=$1
GAME_TYPE="go"
CONF_FILE="../conf.cfg"

echo "Model Path: $MODEL_PATH"
echo "Game Type: $GAME_TYPE"
echo "Config File: $CONF_FILE"
echo ""

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model file not found: $MODEL_PATH"
    exit 1
fi

# Run testing
python3 testing.py $GAME_TYPE \
    -c $CONF_FILE \
    -m $MODEL_PATH

echo ""
echo "======================================"
echo "Testing completed!"
echo "Check submission.csv for results"
echo "======================================"
