#!/bin/bash

# Smoke Test Script for Go Style Detection
# 快速驗證整個訓練 pipeline 是否正常運作

echo "======================================"
echo "Smoke Test - Go Style Detection"
echo "======================================"
echo ""
echo "這是快速測試，只跑 10 步訓練以驗證："
echo "  ✓ C++ module 可正常載入"
echo "  ✓ SGF 資料可正確讀取"
echo "  ✓ Model forward pass 運作正常"
echo "  ✓ Triplet loss 計算正確"
echo "  ✓ Checkpoint 可以儲存"
echo ""

# Configuration
RUN_ID="smoke_test_$(date +%Y%m%d_%H%M%S)"
GAME_TYPE="go"
CONF_FILE="/workspace/ML-Assignment2-Q5/conf_smoke_test.cfg"
MODELS_DIR="./models_smoke_test"
SAVE_EVERY=5

echo "Run ID: $RUN_ID"
echo "Config File: $CONF_FILE"
echo ""

# Verify config file exists
if [ ! -f "$CONF_FILE" ]; then
    echo "ERROR: Config file not found: $CONF_FILE"
    exit 1
fi

# Show key config values for verification
echo "Verifying config..."
grep "players_per_batch" $CONF_FILE
grep "games_per_player" $CONF_FILE
grep "learner_training_step" $CONF_FILE
echo ""
echo "Models Directory: $MODELS_DIR"
echo ""

# Create models directory
mkdir -p $MODELS_DIR

# Check GPU availability
echo "Checking environment..."
python3 << 'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
else:
    print("Warning: Will run on CPU (slower)")
EOF
echo ""

# Run smoke test
echo "Starting smoke test training..."
echo "======================================"

python3 train.py $GAME_TYPE $RUN_ID \
    -c $CONF_FILE \
    -m $MODELS_DIR \
    -s $SAVE_EVERY \
    -b 100 \
    -ve 100

EXIT_CODE=$?

echo ""
echo "======================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Smoke test PASSED!"
    echo ""
    echo "Model saved in: $MODELS_DIR"
    echo "Check the output above for any warnings."
    echo ""
    echo "如果一切正常，可以用 ./run_training.sh 開始正式訓練"
else
    echo "✗ Smoke test FAILED!"
    echo "Exit code: $EXIT_CODE"
    echo ""
    echo "請檢查上面的錯誤訊息並修正問題"
fi
echo "======================================"
