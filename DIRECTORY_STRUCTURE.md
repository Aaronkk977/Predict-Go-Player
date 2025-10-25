# ML-Assignment2-Q5 目錄架構與實作流程

## 📂 目錄結構概覽

```
ML-Assignment2-Q5/
├── minizero/          # 🎮 遊戲引擎核心 (助教提供，完整)
├── style_detection/   # 🎯 作業核心 (助教半完成，你要填空)
├── build/             # 🔧 編譯輸出目錄 (自動生成)
├── scripts/           # 🚀 自動化腳本 (助教提供)
└── conf.cfg           # ⚙️ 設定檔
```

---

## 🎮 1. minizero/ - 遊戲引擎核心

### **職責**
完整的 AlphaZero/MuZero 強化學習框架，支援多種棋類遊戲。

### **結構**
```
minizero/
├── minizero/
│   ├── actor/          # MCTS 搜尋引擎
│   ├── console/        # 命令列介面
│   ├── environment/    # 遊戲環境 ⭐⭐⭐
│   │   ├── go/         # 圍棋規則實作
│   │   │   ├── go.cpp  # 完整圍棋邏輯
│   │   │   └── go.h    # (提子、劫、氣等)
│   │   └── ...         # 其他遊戲
│   ├── network/        # 神經網路介面
│   ├── learner/        # 訓練邏輯
│   │   └── data_loader.cpp  # 基礎資料載入器
│   └── utils/          # 工具函式
└── CMakeLists.txt
```

### **在流程中的位置**
```
階段 0: 基礎設施 (你不需要改)
┌─────────────────────────────────────────────┐
│ minizero/ 提供:                              │
│ ✅ 圍棋規則引擎 (19×19, 提子, 劫, 氣)        │
│ ✅ SGF 檔案解析                              │
│ ✅ 棋盤狀態管理                              │
│ ✅ AlphaZero 網路架構基礎                    │
│ ✅ 多執行緒資料載入器                        │
└─────────────────────────────────────────────┘
         ↓ (被 style_detection 使用)
```

### **關鍵檔案**
- `minizero/environment/go/go.cpp` - 完整圍棋規則
- `minizero/learner/data_loader.cpp` - 資料載入基類
- `minizero/utils/sgf_loader.h` - SGF 解析器

---

## 🎯 2. style_detection/ - 作業核心 (你的工作區)

### **職責**
實作棋風檢測的特定功能，基於 minizero 框架。

### **結構**
```
style_detection/
├── code/                      # Python 訓練程式碼
│   ├── encoder/
│   │   ├── model.py          # ⚠️ 空白 - 你要實作神經網路
│   │   └── train.py          # ✅ 助教完成 - 訓練框架
│   ├── testing.py            # ⚠️ 範本 - 你要補完推論邏輯
│   └── train.py              # ✅ 助教完成 - 訓練入口
│
├── sd_data_loader.cpp/h       # ✅ 助教完成 - 資料載入器
├── sd_mode_handler.cpp/h      # ⚠️ 有 TODO - SGF 載入
├── sd_configuration.cpp/h     # ✅ 助教完成 - 設定管理
├── pybind.cpp                 # ✅ 助教完成 - Python 綁定
├── main.cpp                   # ✅ 助教完成 - C++ 主程式
└── CMakeLists.txt             # ✅ 助教完成 - 編譯設定
```

### **在流程中的位置**
```
階段 1: 資料準備
┌─────────────────────────────────────────────┐
│ sd_data_loader.cpp (C++)                    │
│ ↓ 使用 minizero 引擎                        │
│ ✅ 載入 SGF 檔案                            │
│ ✅ 提取棋盤特徵 (n_frames × channels)       │
│ ✅ 轉換為 Tensor                            │
└─────────────────────────────────────────────┘
         ↓ pybind.cpp (Python ↔ C++ 橋接)
         
階段 2: 模型訓練 (Python)
┌─────────────────────────────────────────────┐
│ code/train.py                               │
│ ↓                                           │
│ code/encoder/train.py                       │
│ ├─ MinizeroDataset (✅ 完成)               │
│ │   └─ 呼叫 style_py.DataLoader            │
│ └─ model.py (⚠️ 你要實作)                  │
│     └─ Encoder 類別                         │
│         ├─ __init__() - 定義網路結構        │
│         ├─ forward() - 前向傳播             │
│         └─ loss() - 損失函數                │
└─────────────────────────────────────────────┘
         ↓
         
階段 3: 測試推論 (Python)
┌─────────────────────────────────────────────┐
│ code/testing.py (⚠️ 你要補完)              │
│ ├─ load query_set (600 玩家)               │
│ ├─ 提取 embeddings                          │
│ ├─ load candidate_set (600 玩家)           │
│ ├─ 計算相似度                               │
│ └─ 生成 submission.csv                      │
└─────────────────────────────────────────────┘
```

### **你需要完成的部分**

#### ⚠️ **必須實作**
1. **`code/encoder/model.py`** - 神經網路架構
   ```python
   class Encoder(nn.Module):
       def __init__(self, loss_device, conf_file, game_type):
           # 讀取設定
           # 定義 CNN layers
           # 定義 embedding layer
           
       def forward(self, inputs):
           # (batch, n_frames, channels, 19, 19)
           # → CNN 處理
           # → 輸出 embedding (batch, embedding_dim)
           
       def loss(self, embeddings, labels):
           # Triplet Loss 或其他度量學習損失
   ```

2. **`code/testing.py`** - 補完推論邏輯
   ```python
   def testing(self):
       # 1. 載入 query_set
       # 2. 提取所有 query embeddings
       # 3. 載入 candidate_set
       # 4. 提取所有 candidate embeddings
       # 5. 計算 cosine similarity
       # 6. 輸出 submission.csv
   ```

#### ⚠️ **可選實作**
3. **`sd_mode_handler.cpp:42`** - SGF 載入 (如果要用 C++ 主程式)
   ```cpp
   std::vector<EnvironmentLoader> SDModeHandler::loadEnvironmentLoaders()
   {
       // TODO: read sgf
       // 讀取 SGF 檔案並轉換為 EnvironmentLoader
   }
   ```

---

## 🔧 3. build/ - 編譯輸出目錄

### **職責**
存放編譯產生的二進位檔案和 Python 模組。

### **結構**
```
build/
└── go/                        # 編譯遊戲類型為 go
    ├── Makefile               # 自動生成
    ├── CMakeCache.txt         # CMake 快取
    ├── style_go               # C++ 主程式 (可執行檔)
    ├── style_py.*.so          # ⭐⭐⭐ Python 模組 (最重要!)
    ├── minizero/              # minizero 編譯產物
    │   └── libminizero.a
    └── style_detection/       # style_detection 編譯產物
        └── *.o
```

### **在流程中的位置**
```
編譯階段 (scripts/build.sh)
┌─────────────────────────────────────────────┐
│ 輸入: *.cpp, *.h, CMakeLists.txt            │
│   ↓ CMake 配置                               │
│   ↓ Make 編譯                                │
│ 輸出: build/go/                              │
│   ├── style_py.*.so  ⭐ Python 可 import    │
│   └── style_go       (C++ 獨立程式)          │
└─────────────────────────────────────────────┘
         ↓
Python 程式碼可以這樣用:
┌─────────────────────────────────────────────┐
│ import build.go.style_py as style_py        │
│ style_py.load_config_file("conf.cfg")       │
│ data_loader = style_py.DataLoader(...)      │
│ features = data_loader.get_feature(...)     │
└─────────────────────────────────────────────┘
```

### **關鍵檔案**
- **`build/go/style_py.cpython-*.so`** - Python 可導入的 C++ 模組
  - 這是連接 Python 和 C++ 的橋樑
  - 通過 pybind11 生成
  - 提供 `DataLoader`, `Environment` 等類別給 Python

---

## 🚀 4. scripts/ - 自動化腳本

### **職責**
提供便利的自動化工具，簡化編譯、容器管理、訓練等操作。

### **檔案**
```
scripts/
├── build.sh              # 🔨 編譯腳本
├── start-container.sh    # 🐳 啟動 container
└── train-sl.sh           # 🎓 訓練腳本 (範例)
```

### **詳細說明**

#### **build.sh** - 編譯腳本
```bash
# 用途: 編譯整個專案
# 語法: ./scripts/build.sh <game_type> [build_type]

# 流程:
./scripts/build.sh go release
  ↓
1. 建立 build/go/ 目錄
2. 執行 CMake 配置
   - 設定 GAME_TYPE=GO
   - 找 PyTorch, pybind11
3. 執行 Make 編譯
   - 編譯 minizero
   - 編譯 style_detection
   - 生成 style_py.so
4. 輸出到 build/go/
```

#### **start-container.sh** - Container 管理
```bash
# 用途: 啟動 Docker/Podman container
# 內容:
./minizero/scripts/start-container.sh \
  -v $(pwd):/strength-detection \
  --image docker.io/kds285/strength-detection

# 功能:
- 掛載當前目錄到 container
- 啟動包含所有依賴的環境
- 在 container 內執行編譯和訓練
```

#### **train-sl.sh** - 訓練範例
```bash
# 用途: 展示如何啟動訓練
# 內容:
PYTHONPATH=. python -u strength_detection/code/train_sl.py \
  go ${DIR} test.cfg

# 說明: 這是助教提供的範例，實際訓練要用 train.py
```

### **在流程中的位置**
```
完整工作流程:

0. 準備環境
   $ scripts/start-container.sh
   (進入 container，已安裝所有依賴)

1. 編譯
   $ scripts/build.sh go release
   → 生成 build/go/style_py.so

2. 訓練
   $ cd style_detection/code
   $ python train.py go my_model -c ../../conf.cfg -d ../../train_set
   → 使用 build.go.style_py 載入資料
   → 訓練模型
   → 儲存到 models/my_model/

3. 測試
   $ python testing.py go -c ../../conf.cfg
   → 載入模型
   → 推論 test_set
   → 生成 submission.csv
```

---

## 🔄 完整實作流程圖

```
┌─────────────────────────────────────────────────────────────────┐
│                    階段 0: 環境準備                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
    scripts/start-container.sh (啟動 container)
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    階段 1: 編譯                                   │
│                                                                   │
│  scripts/build.sh go release                                     │
│    ↓                                                             │
│  CMake 讀取:                                                      │
│    - ML-Assignment2-Q5/CMakeLists.txt (主設定)                   │
│    - minizero/CMakeLists.txt (引擎)                              │
│    - style_detection/CMakeLists.txt (作業)                       │
│    ↓                                                             │
│  編譯順序:                                                        │
│    1. minizero/ → libminizero.a                                  │
│    2. style_detection/ → style_py.so ⭐                          │
│    3. 連結所有模組                                                │
│    ↓                                                             │
│  輸出: build/go/style_py.cpython-*.so                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    階段 2: 實作模型                               │
│                                                                   │
│  你要寫的檔案:                                                    │
│    1. style_detection/code/encoder/model.py                      │
│       class Encoder(nn.Module):                                  │
│         - __init__: 定義 CNN 架構                                │
│         - forward: 前向傳播                                       │
│         - loss: Triplet Loss                                     │
│                                                                   │
│    2. style_detection/code/testing.py                            │
│       - 補完 inference() 方法                                     │
│       - 補完 testing() 方法                                       │
│       - 生成 submission.csv                                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    階段 3: 訓練                                   │
│                                                                   │
│  cd style_detection/code                                         │
│  python train.py go my_run \                                     │
│    -d ../../train_set \                                          │
│    -c ../../conf.cfg                                             │
│    ↓                                                             │
│  train.py (助教完成) 呼叫:                                        │
│    ├─ import build.go.style_py                                   │
│    ├─ style_py.DataLoader() ← C++ 提供                           │
│    ├─ MinizeroDataset (助教完成)                                 │
│    ├─ encoder.model.Encoder (你實作)                             │
│    └─ 訓練循環                                                    │
│         ├─ 取得批次資料                                           │
│         ├─ model.forward()                                       │
│         ├─ model.loss()                                          │
│         └─ optimizer.step()                                      │
│    ↓                                                             │
│  輸出: models/my_run/model_*.pth                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    階段 4: 測試                                   │
│                                                                   │
│  python testing.py go -c ../../conf.cfg                          │
│    ↓                                                             │
│  testing.py (你補完) 執行:                                        │
│    ├─ 載入訓練好的模型                                            │
│    ├─ 讀取 query_set/player*.sgf (600個)                         │
│    │   └─ 用 style_py.DataLoader                                │
│    ├─ 提取 query embeddings                                      │
│    ├─ 讀取 cand_set/player*.sgf (600個)                          │
│    ├─ 提取 candidate embeddings                                  │
│    ├─ 計算 cosine similarity                                     │
│    │   for each query:                                           │
│    │     找最相似的 candidate                                     │
│    └─ 生成 submission.csv                                        │
│         query1,cand234                                           │
│         query2,cand567                                           │
│         ...                                                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📋 各目錄的依賴關係

```
依賴層次 (由下到上):

Layer 4: Scripts (自動化)
  └── scripts/
       ├── build.sh → 呼叫 CMake
       └── start-container.sh → 啟動環境

Layer 3: Build Output (產物)
  └── build/go/
       └── style_py.so ← 被 Python import

Layer 2: Application (你的工作)
  └── style_detection/
       ├── C++ 資料載入器 → 使用 Layer 1
       └── Python 訓練/測試 → 使用 Layer 3

Layer 1: Engine (基礎設施)
  └── minizero/
       └── 圍棋引擎、MCTS、神經網路介面

Layer 0: Config (設定)
  └── conf.cfg
```

---

## 🎯 總結: 你需要做什麼

### ✅ **不需要改的 (助教完成)**
- ❌ minizero/ - 完整的遊戲引擎
- ❌ scripts/ - 編譯和啟動腳本
- ❌ style_detection/sd_data_loader.* - 資料載入器
- ❌ style_detection/pybind.cpp - Python 綁定
- ❌ style_detection/code/encoder/train.py - 訓練框架

### ⚠️ **你要實作的**
1. **style_detection/code/encoder/model.py** (核心)
   - 設計 CNN 架構提取棋風特徵
   - 實作 Triplet Loss 或其他度量學習損失
   
2. **style_detection/code/testing.py** (推論)
   - 補完推論邏輯
   - 生成 submission.csv

### 🔧 **依賴編譯的部分**
- **build/go/style_py.so** - 必須成功編譯才能用
  - 需要: podman + container image
  - 或: 直接在有完整環境的機器上編譯

---

現在下載還在進行嗎？要我幫你檢查進度或開始設計 model.py 嗎？
