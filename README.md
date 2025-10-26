ML model to predicting Go players' style.

## 🚀 Quick Start

### Step 1: 主機端 - 啟動 Docker 容器

```bash
cd /tmp2/b12902115/Predict-Go-Player/ML-Assignment2-Q5

# 拉取 image
docker pull docker.io/kds285/strength-detection:latest

# 啟動容器（選一個）
./scripts/start-container-gpu.sh  # GPU 版本（推薦，快 10-50 倍）
# 或
./scripts/start-container.sh      # CPU 版本
```

### Step 2: 容器內 - 編譯 C++ 後端

```bash
cd /workspace/ML-Assignment2-Q5

./scripts/build.sh go # Compile
```

### Step 3: 容器內 - 開始訓練

```bash
./scripts/run_training.sh

# quick test (debug)
./scripts/run_smoke_test.sh
```
