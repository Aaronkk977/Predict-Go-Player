# 圍棋棋風檢測系統 - 使用指南

## 📋 系統功能

`simple_style_detection.py` 包含以下完整功能：

### ✅ 已實作的功能

1. **資料載入**
   - 訓練集: 自動載入 `train_set/` 的 200 個 SGF 檔案，按玩家名稱分組
   - 測試集: 載入 query_set 和 cand_set 的多遊戲 SGF 檔案

2. **模型訓練**
   - 使用 Triplet Loss 進行 metric learning
   - 讓相同玩家的遊戲 embedding 靠近，不同玩家的遊戲 embedding 遠離
   - 自動儲存最佳模型

3. **推理與預測**
   - 提取 query 和 candidate 的 embeddings
   - 計算 cosine similarity
   - 自動配對並生成 submission.csv

4. **特徵工程**
   - 棋盤狀態編碼 (黑子/白子/當前玩家)
   - 使用 n_frames 個歷史狀態
   - 資料增強 (隨機開始位置)

---

## 🚀 快速開始

### 1. 完整訓練 + 推理

```bash
# 一次完成訓練和推理 (推薦)
python3 simple_style_detection.py --mode full --epochs 50
```

這會：
1. 載入 train_set 訓練模型
2. 儲存最佳模型到 `model_best.pth`
3. 對 test_set 進行推理
4. 生成 `submission.csv`

### 2. 分步執行

#### 步驟 A: 只訓練
```bash
python3 simple_style_detection.py \
    --mode train \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.001 \
    --model_path my_model.pth
```

#### 步驟 B: 只推理 (使用已訓練的模型)
```bash
python3 simple_style_detection.py \
    --mode inference \
    --model_path my_model.pth \
    --output my_submission.csv
```

---

## ⚙️ 參數說明

### 基本參數
- `--mode`: 運行模式
  - `train`: 只訓練
  - `inference`: 只推理 (需要已訓練的模型)
  - `full`: 訓練 + 推理 (預設)

### 資料路徑
- `--train_dir`: 訓練資料目錄 (預設: `train_set`)
- `--query_dir`: Query 測試資料 (預設: `test_set/query_set`)
- `--cand_dir`: Candidate 測試資料 (預設: `test_set/cand_set`)
- `--model_path`: 模型儲存/載入路徑 (預設: `model_best.pth`)
- `--output`: 提交檔案路徑 (預設: `submission.csv`)

### 訓練參數
- `--epochs`: 訓練輪數 (預設: 50)
- `--batch_size`: Batch size (預設: 32)
- `--lr`: 學習率 (預設: 0.001)

### 模型參數
- `--n_frames`: 使用的歷史 frames 數量 (預設: 10)
- `--embedding_dim`: Embedding 向量維度 (預設: 128)

---

## 📊 完整訓練範例

### 基礎訓練 (CPU, 適合測試)
```bash
python3 simple_style_detection.py \
    --mode full \
    --epochs 30 \
    --batch_size 16
```

### 進階訓練 (調整參數)
```bash
python3 simple_style_detection.py \
    --mode full \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.0005 \
    --n_frames 15 \
    --embedding_dim 256 \
    --model_path model_v2.pth \
    --output submission_v2.csv
```

### 快速測試 (3 epochs)
```bash
python3 simple_style_detection.py \
    --mode train \
    --epochs 3 \
    --batch_size 8
```

---

## 📁 檔案結構

```
Predict-Go-Player/
├── simple_style_detection.py    # 主程式
├── train_set/                    # 訓練資料 (200 個 SGF)
│   ├── 1.sgf
│   ├── 2.sgf
│   └── ...
├── test_set/
│   ├── query_set/                # Query 檔案 (600 個)
│   │   ├── player001.sgf
│   │   └── ...
│   └── cand_set/                 # Candidate 檔案 (600 個)
│       ├── player001.sgf
│       └── ...
├── model_best.pth                # 訓練好的模型 (自動生成)
├── submission.csv                # 提交檔案 (自動生成)
└── submission_sample.csv         # 範例格式
```

---

## 🔍 系統工作流程

### 訓練階段 (mode=train)
1. 載入 train_set/*.sgf
2. 按玩家名稱 (PB/PW) 分組
3. 使用 Triplet Loss 訓練:
   - Anchor: 玩家 A 的遊戲 1
   - Positive: 玩家 A 的遊戲 2
   - Negative: 玩家 B 的遊戲
4. 儲存最佳模型

### 推理階段 (mode=inference)
1. 載入訓練好的模型
2. 對 query_set 的每個玩家:
   - 讀取多場遊戲
   - 提取特徵並計算平均 embedding
3. 對 cand_set 的 600 個候選玩家:
   - 同樣提取平均 embedding
4. 計算 cosine similarity:
   - 每個 query 找最相似的 candidate
5. 生成 submission.csv

---

## 📈 預期結果

### 訓練過程
```
Epoch 1/50: 100%|████| loss=0.9158
Epoch 2/50: 100%|████| loss=0.8542
...
✅ 訓練完成！最佳 Loss: 0.4235
```

### 推理結果
```
🔍 開始推理...
Query embedding: 100%|████████| 600/600
Candidate embedding: 100%|███| 600/600
配對: 100%|██████████████████| 600/600

💾 生成提交檔案: submission.csv
✓ 已儲存 600 筆預測結果

前 10 筆預測:
   id  label
0   1     54
1   2    128
2   3    456
...
```

---

## 🐛 疑難排解

### 問題 1: Out of Memory
```bash
# 減少 batch_size
python3 simple_style_detection.py --batch_size 8
```

### 問題 2: 訓練太慢
```bash
# 減少 epochs 或使用更少的 frames
python3 simple_style_detection.py --epochs 20 --n_frames 5
```

### 問題 3: Query/Candidate SGF 載入失敗
- 確認 `test_set/query_set/` 和 `test_set/cand_set/` 存在
- 檢查 SGF 檔案格式是否正確

---

## 💡 優化建議

### 提升準確率
1. **增加訓練輪數**: `--epochs 100`
2. **調整 embedding 維度**: `--embedding_dim 256`
3. **使用更多歷史 frames**: `--n_frames 20`
4. **調整 learning rate**: `--lr 0.0005`

### 加速訓練
1. 使用 GPU (自動偵測)
2. 增加 batch_size: `--batch_size 64`
3. 減少 n_frames: `--n_frames 5`

---

## ✅ 檢查清單

提交前確認：
- [ ] 訓練完成且 loss 收斂
- [ ] submission.csv 已生成
- [ ] CSV 格式正確 (id, label 兩欄)
- [ ] 共有 600 筆預測 (對應 600 個 query)
- [ ] label 範圍在 1-600 之間

---

## 🎯 完整執行範例

```bash
# 1. 快速測試 (確認無錯誤)
python3 simple_style_detection.py --mode train --epochs 3

# 2. 完整訓練並生成提交檔案
python3 simple_style_detection.py --mode full --epochs 50

# 3. 檢查結果
head submission.csv
wc -l submission.csv  # 應該是 601 行 (含標題)

# 4. 完成！
```
