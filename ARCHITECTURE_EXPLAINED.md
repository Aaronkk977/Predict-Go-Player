# 圍棋棋風檢測系統 - 完整架構說明

## 📋 目錄
1. [完整實作流程](#完整實作流程)
2. [助教已完成的部分](#助教已完成的部分)
3. [需要 C++ 編譯的部分](#需要-c-編譯的部分)
4. [你的選擇方案](#你的選擇方案)

---

## 🔄 完整實作流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                         整體工作流程                                  │
└─────────────────────────────────────────────────────────────────────┘

第一階段: 資料讀取與前處理
┌────────────────────────────────────────────────────────────────┐
│  輸入: train_set/*.sgf (200 個訓練檔案)                          │
│         ↓                                                       │
│  [SGF Parser] → 解析圍棋棋譜                                     │
│         ↓                                                       │
│  [特徵提取] → 轉換為神經網路輸入                                  │
│         ↓                                                       │
│  特徵張量: (C, 19, 19)                                          │
│  - C = n_frames × 3 channels (黑子/白子/當前玩家)                │
│  - 或 C = n_frames × 3 + 3 (加上氣+合法著手)                     │
└────────────────────────────────────────────────────────────────┘

第二階段: 模型訓練
┌────────────────────────────────────────────────────────────────┐
│  [特徵張量] (batch, C, 19, 19)                                  │
│         ↓                                                       │
│  [CNN Encoder] → 提取棋風特徵                                    │
│         ↓                                                       │
│  [Embedding] → 128 維向量 (玩家棋風表示)                         │
│         ↓                                                       │
│  [Triplet Loss] → 學習相似度                                     │
│   - Anchor:   玩家 A 的某局棋                                    │
│   - Positive: 玩家 A 的另一局棋 (應該相似)                       │
│   - Negative: 玩家 B 的棋局 (應該不同)                           │
│         ↓                                                       │
│  [訓練 N 個 Epochs] → 儲存模型                                   │
└────────────────────────────────────────────────────────────────┘

第三階段: 測試與預測
┌────────────────────────────────────────────────────────────────┐
│  輸入 1: test_set/query_set/player*.sgf (600 個查詢玩家)         │
│         ↓                                                       │
│  [特徵提取] → (600, C, 19, 19)                                  │
│         ↓                                                       │
│  [CNN Encoder] → 600 個 Query Embeddings                        │
│                                                                 │
│  輸入 2: test_set/cand_set/player*.sgf (600 個候選玩家)          │
│         ↓                                                       │
│  [特徵提取] → (600, C, 19, 19)                                  │
│         ↓                                                       │
│  [CNN Encoder] → 600 個 Candidate Embeddings                    │
│                                                                 │
│  [計算相似度]                                                    │
│         ↓                                                       │
│  對每個 Query i:                                                │
│    計算 cosine_similarity(query_i, cand_j) for all j           │
│    選擇最高分的 cand_j                                           │
│         ↓                                                       │
│  輸出: submission.csv                                           │
│    query1, cand234                                             │
│    query2, cand789                                             │
│    ...                                                         │
└────────────────────────────────────────────────────────────────┘
```

---

## ✅ 助教已完成的部分

### 1️⃣ **MiniZero 框架** (C++ 核心)
```
ML-Assignment2-Q5/minizero/
├── minizero/
│   ├── actor/       # MCTS 搜尋引擎
│   ├── network/     # 神經網路介面
│   ├── environment/ # 圍棋規則引擎 ⭐⭐⭐
│   └── utils/       # 工具函式
└── CMakeLists.txt
```

**功能:**
- ✅ 完整的圍棋規則實作 (提子、打劫、氣的計算)
- ✅ AlphaZero 網路架構
- ✅ MCTS (Monte Carlo Tree Search)
- ✅ 支援 SGF 格式解析

**狀態:** 完整、無需修改

---

### 2️⃣ **Style Detection C++ Backend**
```
ML-Assignment2-Q5/style_detection/
├── sd_data_loader.cpp/h     # 資料載入器 ⭐
├── sd_configuration.cpp/h   # 設定管理
├── pybind.cpp              # Python 綁定 ⭐⭐
├── sd_mode_handler.cpp/h    # ⚠️ 有 TODO
└── CMakeLists.txt
```

**已完成功能:**
```cpp
// pybind.cpp 已實作的 Python 介面
class DataLoader:
    - loadDataFromFile(sgf_path)         # 載入 SGF
    - get_feature_and_label(player, game, start, is_train)  # 提取特徵
    - get_random_feature_and_label(...)   # 隨機位置特徵
    - Clear_Sgf()                         # 清空資料
    - Check_Sgf()                         # 檢查資料
    - get_num_of_player()                 # 玩家數量
```

**未完成部分:**
```cpp
// sd_mode_handler.cpp:42
std::vector<EnvironmentLoader> SDModeHandler::loadEnvironmentLoaders()
{
    std::vector<EnvironmentLoader> env_loaders;
    // TODO: read sgf  ⚠️⚠️⚠️ 這裡要自己實作
    return env_loaders;
}
```

**狀態:** 大部分完成，但有關鍵 TODO

---

### 3️⃣ **Python 訓練/測試框架**
```
ML-Assignment2-Q5/style_detection/code/
├── testing.py         # 測試流程範本 ⭐⭐⭐
├── train.py           # 訓練流程框架
└── encoder/
    ├── model.py       # ⚠️ 空白模板 (要自己實作)
    └── train.py       # 訓練邏輯框架
```

**testing.py 提供的範本:**
```python
class style_detection:
    def __init__(self, conf_file, game_type):
        # 初始化 C++ DataLoader
        self.data_loader = style_py.DataLoader(conf_file)
        
    def read_sgf(self, sgf_dir):
        # 載入所有 600 個玩家的 SGF
        for i in range(600):
            self.data_loader.load_data_from_file(
                sgf_dir + "player" + str(i+1) + ".sgf"
            )
    
    def load_model(self, model_path):
        # 載入訓練好的模型 (要自己實作)
        self.model = Encoder(...)
        
    def inference(self, data_loader):
        # 對所有玩家進行推論 (要自己實作)
        for player in range(600):
            for game in range(100):
                features = data_loader.get_feature_and_label(...)
                output = self.model(features)  # ⚠️ 這裡要完成
    
    def testing(self):
        # 完整測試流程
        self.read_sgf("./test_set/query_set")
        self.inference(self.data_loader)
        self.data_loader.Clear_Sgf()
        
        self.read_sgf("./test_set/cand_set")
        self.inference(self.data_loader)
        # ⚠️ 要自己加上相似度計算和輸出 submission.csv
```

**狀態:** 框架完整，但核心邏輯要自己實作

---

### 4️⃣ **設定檔**
```ini
# conf.cfg (已配置好)
env_board_size=19
env_go_komi=7.5

# Strength Detection 參數
players_per_batch=20      # 每批次訓練多少玩家
games_per_player=9        # 每個玩家取幾局棋
n_frames=10               # 歷史幀數
move_step_to_choose=4     # 從哪步開始取樣
```

**狀態:** 完整可用

---

## ⚙️ 需要 C++ 編譯的部分

### 🔴 **必須編譯才能使用的元件**

```
┌─────────────────────────────────────────────────────────────┐
│  C++ 編譯流程                                                │
└─────────────────────────────────────────────────────────────┘

1. 編譯 MiniZero 核心
   ┌────────────────────────────────────────────────┐
   │ cd ML-Assignment2-Q5/minizero                  │
   │ mkdir build && cd build                        │
   │ cmake ..                                       │
   │ make -j8                                       │
   └────────────────────────────────────────────────┘
   
   依賴項:
   - ❌ CMake >= 3.12
   - ❌ pybind11
   - ❌ PyTorch C++ API (libtorch)
   - ❌ CUDA (可選)
   - ❌ Boost (部分功能)
   
   輸出: libminizero.a

2. 編譯 Style Detection
   ┌────────────────────────────────────────────────┐
   │ cd ML-Assignment2-Q5                           │
   │ mkdir -p build/go                              │
   │ cd build/go                                    │
   │ cmake ../..                                    │
   │ make -j8                                       │
   └────────────────────────────────────────────────┘
   
   依賴項:
   - ❌ 上面的 libminizero.a
   - ❌ pybind11 (生成 Python 綁定)
   - ❌ Python 開發標頭檔
   
   輸出: style_py.cpython-*.so  ⭐⭐⭐ 關鍵產物

3. Python 使用編譯好的模組
   ┌────────────────────────────────────────────────┐
   │ import build.go.style_py as style_py           │
   │ style_py.load_config_file("conf.cfg")          │
   │ data_loader = style_py.DataLoader("conf.cfg")  │
   │ data_loader.load_data_from_file("1.sgf")       │
   └────────────────────────────────────────────────┘
```

### 🟡 **編譯問題的具體位置**

#### **問題 1: pybind11 路徑錯誤**
```bash
位置: ML-Assignment2-Q5/style_detection/CMakeLists.txt

錯誤訊息:
CMake Error: Could not find pybind11

原因:
- 系統的 pybind11 安裝位置與 CMake 搜尋路徑不符
- Python 3.13 的 pybind11 路徑可能在非標準位置

解決方案:
1. 手動指定路徑: cmake -Dpybind11_DIR=...
2. 或使用 pip 安裝的版本
```

#### **問題 2: MiniZero 依賴**
```bash
位置: ML-Assignment2-Q5/minizero/

錯誤:
- ALE (Atari Learning Environment) 找不到
- 部分 Boost 函式庫缺失

原因:
- MiniZero 支援多種遊戲 (圍棋/Atari 等)
- 但這次只需要圍棋功能

解決方案:
- 修改 CMakeLists.txt 關閉 Atari 支援
- 或安裝所有依賴 (較麻煩)
```

#### **問題 3: 工作站限制**
```bash
環境: 台大工作站 ws3.csie.ntu.edu.tw

限制:
❌ 無 Docker/Podman (無法用 container)
❌ 可能沒有 sudo 權限 (無法安裝系統套件)
⚠️ Python 3.13.5 (較新版本,某些套件可能不相容)

現況:
你之前嘗試編譯時遇到這些問題,所以轉向純 Python 方案
```

---

## 🎯 你的選擇方案

### 📊 **方案比較表**

| 項目 | 方案 A: 純 Python | 方案 B: C++ + Python | 方案 C: 向 TA 要 .so |
|------|------------------|---------------------|---------------------|
| **需要編譯** | ❌ 不需要 | ✅ 需要成功編譯 | ❌ 不需要 |
| **圍棋規則** | 自己實作 | ✅ MiniZero 提供 | ✅ MiniZero 提供 |
| **特徵提取** | 自己寫 (sgfmill) | ✅ C++ DataLoader | ✅ C++ DataLoader |
| **訓練時間** | ~10-15 分鐘 | ~5-10 分鐘 | ~5-10 分鐘 |
| **準確率** | ⚠️ 可能稍低 | ✅ 標準實作 | ✅ 標準實作 |
| **難度** | 🟢 簡單 | 🔴 困難 | 🟡 中等 |
| **目前狀態** | ✅ 已完成 | ❌ 編譯失敗 | ❓ 未嘗試 |

---

### 🔍 **詳細分析**

#### **方案 A: 純 Python** (你目前的方案)

**你已經完成:**
```
simple_style_detection.py      # 簡單版本
improved_go_features.py        # 改進版本 (完整圍棋規則)
balanced_style_detection.py    # 平衡版本
```

**優點:**
- ✅ 完全不需要處理 C++ 編譯問題
- ✅ 可以立即開始訓練和測試
- ✅ improved_go_features.py 已實作完整圍棋規則
- ✅ 程式碼清晰易懂,容易 debug

**缺點:**
- ⚠️ 如果助教**要求必須使用** MiniZero,會無法繳交
- ⚠️ 特徵提取可能不如 C++ 版本快 (但差異不大)
- ⚠️ 需要確認圍棋規則實作正確性

**適用情況:**
- 助教說「只要結果正確就好」
- 時間緊迫,不想處理編譯問題
- 想要完全掌控整個流程

---

#### **方案 B: 完整 C++ 編譯** (原始設計)

**需要完成:**
1. ✅ 編譯 MiniZero 核心
2. ✅ 編譯 style_py.so
3. ⚠️ 完成 sd_mode_handler.cpp 的 TODO
4. ⚠️ 實作 encoder/model.py
5. ✅ 使用 testing.py 框架

**優點:**
- ✅ 使用助教提供的完整框架
- ✅ 圍棋規則由 MiniZero 保證正確
- ✅ 特徵提取經過優化,可能更快
- ✅ 符合原始作業設計

**缺點:**
- ❌ 編譯問題複雜 (pybind11, 依賴套件)
- ❌ 可能需要助教或系統管理員協助
- ❌ Debug 較困難 (C++/Python 混合)
- ⏰ 花費時間不確定

**適用情況:**
- 助教明確要求使用 MiniZero
- 有時間處理編譯問題
- 可以取得 sudo 權限或管理員協助

---

#### **方案 C: 混合方案** (推薦嘗試)

**策略:**
1. 向助教詢問:「可以提供編譯好的 style_py.so 嗎?」
2. 或問其他同學是否成功編譯
3. 拿到 .so 後,使用 testing.py 框架

**優點:**
- ✅ 避免編譯問題
- ✅ 使用標準框架
- ✅ 快速完成

**缺點:**
- ⚠️ 依賴他人提供檔案
- ⚠️ 版本相容性問題 (Python 3.13)

---

## 📝 **建議行動計畫**

### **第一步: 確認需求** (最重要!)
```bash
# 寄信或當面問助教:
「請問這次作業是否一定要使用 MiniZero 框架?
 還是只要 submission.csv 的結果正確即可?」
```

### **第二步: 根據回答選擇**

#### **如果 TA 說:「只要結果正確就好」**
```bash
# 使用你的 improved 版本
cd /tmp2/b12902115/Predict-Go-Player
python3 balanced_style_detection.py --mode improved --epochs 50

# 訓練完成後測試
# (要自己加上推論和 submission.csv 生成)
```

#### **如果 TA 說:「必須用 MiniZero」**
```bash
選項 1: 請 TA 提供編譯好的 style_py.so
選項 2: 找成功編譯的同學分享
選項 3: 花時間解決編譯問題 (風險高)
```

---

## 🎓 **總結**

### **助教完成度評估**

| 元件 | 完成度 | 說明 |
|------|--------|------|
| MiniZero 框架 | ✅ 100% | 完整可用 |
| C++ DataLoader | ✅ 95% | 只差 TODO 的 SGF 載入 |
| Python 介面 | ✅ 90% | pybind.cpp 完整 |
| 訓練框架 | ⚠️ 50% | model.py 是空白範本 |
| 測試框架 | ⚠️ 70% | testing.py 有範本但要補完 |
| **編譯系統** | ❌ 0% | **需要學生自己處理** |

### **C++ 編譯相關度**

```
完全依賴 C++ 編譯:
- testing.py 的 style_py.DataLoader  ⭐⭐⭐
- MiniZero 的圍棋規則引擎
- 特徵提取優化

可以繞過 C++ 編譯:
- 自己用 sgfmill 解析 SGF          ✅ 已完成
- 自己實作圍棋規則                  ✅ improved_go_features.py
- 自己寫 PyTorch 模型               ✅ simple_style_detection.py
- 自己實作訓練/測試流程             ✅ 已完成
```

---

## 🚀 **立即行動**

```bash
# 1. 先確認你的 improved 版本能否執行完整流程
cd /tmp2/b12902115/Predict-Go-Player
python3 -c "from improved_go_features import sgf_to_features_with_rules; print('✓ OK')"

# 2. 詢問助教需求
# (寄信或課後問)

# 3. 如果可以用純 Python:
#    → 完成 balanced_style_detection.py 的推論部分
#    → 加上 submission.csv 生成

# 4. 如果必須用 MiniZero:
#    → 先嘗試取得編譯好的 style_py.so
#    → 再考慮自己編譯
```

**你現在最想做什麼?** 🤔
1. 先完成純 Python 版本的推論功能
2. 嘗試解決 C++ 編譯問題
3. 寫信問助教需求
