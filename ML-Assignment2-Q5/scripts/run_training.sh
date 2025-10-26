#!/bin/bash

# Training script for Go style detection
# This script trains the model using triplet loss

echo "======================================"
echo "Go Player Style Detection - Training"
echo "======================================"

# Change to the code directory
cd "$(dirname "$0")/../style_detection/code" || exit 1
echo "Working directory: $(pwd)"
echo ""

# Configuration
RUN_ID="style_model_$(date +%Y%m%d_%H%M%S)"
GAME_TYPE="go"
CONF_FILE="../../conf.cfg"
MODELS_DIR="./models"
SAVE_EVERY=100
BACKUP_EVERY=500
VALIDATE_EVERY=500

echo "Run ID: $RUN_ID"
echo "Game Type: $GAME_TYPE"
echo "Config File: $CONF_FILE"
echo ""

# Verify config file exists
if [ ! -f "$CONF_FILE" ]; then
    echo "ERROR: Config file not found: $CONF_FILE"
    echo "Looking at: $(pwd)/$CONF_FILE"
    exit 1
fi

# Show key config values for verification
echo "Verifying config..."
grep "players_per_batch" $CONF_FILE
grep "games_per_player" $CONF_FILE
grep "n_frames" $CONF_FILE
grep "env_board_size" $CONF_FILE
grep "learner_training_step" $CONF_FILE
echo ""
echo "Models Directory: $MODELS_DIR"
echo ""

# Create models directory if it doesn't exist
mkdir -p $MODELS_DIR

# Run training
echo "Starting training..."
python3 train.py $GAME_TYPE $RUN_ID \
    -c $CONF_FILE \
    -m $MODELS_DIR \
    -s $SAVE_EVERY \
    -b $BACKUP_EVERY \
    -ve $VALIDATE_EVERY

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "======================================"
    echo "✅ Training completed successfully!"
    echo "======================================"
    echo "Model saved in: $MODELS_DIR"
    echo "Run ID: $RUN_ID"
    echo ""
    echo "To test the model, run:"
    echo "  cd /workspace/ML-Assignment2-Q5/style_detection/code"
    echo "  python3 testing.py go -c ../../conf.cfg -m models/${RUN_ID}_final.pt"
else
    echo "======================================"
    echo "❌ Training failed with exit code: $EXIT_CODE"
    echo "======================================"
fi

exit $EXIT_CODE
