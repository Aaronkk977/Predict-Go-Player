#!/bin/bash
# 快速測試腳本

echo "================================"
echo "🧪 快速測試圍棋棋風檢測系統"
echo "================================"

echo ""
echo "📝 測試 1: 檢查資料結構"
echo "訓練集檔案數量:"
ls train_set/*.sgf | wc -l

echo ""
echo "Query set 檔案數量:"
ls test_set/query_set/*.sgf | wc -l

echo ""
echo "Candidate set 檔案數量:"
ls test_set/cand_set/*.sgf | wc -l

echo ""
echo "================================"
echo "🚀 測試 2: 快速訓練 (3 epochs)"
echo "================================"
python3 simple_style_detection.py \
    --mode train \
    --epochs 3 \
    --batch_size 16 \
    --model_path model_test.pth

echo ""
echo "================================"
echo "完成！如果沒有錯誤，可以開始完整訓練："
echo "  python3 simple_style_detection.py --mode full --epochs 50"
echo "================================"
