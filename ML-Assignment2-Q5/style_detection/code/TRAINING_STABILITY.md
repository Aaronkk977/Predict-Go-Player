# 訓練穩定性優化指南

## 已應用的改進

### 1. DataLoader 穩定性
- ✅ 改為 `num_workers=0`（避免多進程問題）
- ✅ 關閉 `persistent_workers` 和 `pin_memory`
- 📝 穩定後可逐步增加到 2-4

### 2. 環境變數管理
- ✅ 創建 `setup_env.py` 設定 cache 路徑到 /tmp2
- ✅ 避免 /home 或小空間被佔滿

### 3. tmux 保護機制
- ✅ 創建 `start_training_stable.sh`
- ✅ 即使 VS Code 斷線，訓練也會繼續
- ✅ 自動記錄 log 到 /tmp2

## 使用方式

### Smoke Test（推薦先執行）

```bash
cd /workspace/ML-Assignment2-Q5/style_detection/code

# 方法 1: 直接執行
chmod +x run_smoke_test.sh
./run_smoke_test.sh

# 方法 2: 使用 tmux（更安全）
chmod +x start_training_stable.sh
./start_training_stable.sh conf_smoke_test.cfg run_smoke_test.sh
```

### 正式訓練

```bash
# 使用 tmux 啟動正式訓練
./start_training_stable.sh ../conf.cfg run_training.sh

# 查看訓練進度
tmux attach -t go_style_training

# 離開但保持訓練運行
按 Ctrl+B 然後 D

# 重新連接
tmux attach -t go_style_training
```

## 監控命令

### 檢查 GPU 使用
```bash
watch -n1 nvidia-smi
```

### 檢查記憶體
```bash
# 找到 Python 進程 PID
ps aux | grep python

# 監控該進程
watch -n1 "ps -o pid,ppid,%mem,%cpu,rss,cmd -p <PID>"
```

### 檢查訓練 log
```bash
# 即時查看最新 log
tail -f /tmp2/b12902115/training_*.log

# 查看所有 log 檔案
ls -lth /tmp2/b12902115/training_*.log
```

### 檢查開檔數（如果懷疑檔案洩漏）
```bash
PID=<your_python_pid>
watch -n1 "ls /proc/$PID/fd | wc -l"
```

## 疑難排解

### 如果 OOM (Out of Memory)
1. 減少 `players_per_batch` 和 `games_per_player`
2. 減少 `n_frames`
3. 檢查 `dmesg | tail -50` 確認是否為 OOM

### 如果訓練很慢
1. 檢查是否使用 GPU：`nvidia-smi`
2. 考慮增加 `num_workers` 到 2（在 `encoder/train.py`）
3. 啟用 `pin_memory=True`（如果 GPU 可用）

### 如果中途中斷
1. 檢查 tmux session：`tmux ls`
2. 檢查 log：`tail /tmp2/b12902115/training_*.log`
3. 檢查系統 log：`dmesg | tail`

## 不需要的調整

以下是 GPT 建議但你的環境**不需要**的：

❌ **rsync 資料到 /tmp2**  
→ 你的資料已經在 /tmp2，Docker mount 就夠

❌ **手動管理 SGF 檔案開關**  
→ C++ DataLoader 已處理

❌ **調整 /dev/shm**  
→ 你的 pipeline 不大量使用 shared memory

❌ **降低系統 NOFILE limit**  
→ 除非出現 "Too many open files" 錯誤

## 下一步

1. **執行 smoke test**（10 步，2-3 分鐘）
2. **如果成功**：調整 conf.cfg 為正式設定
3. **啟動正式訓練**：用 tmux 保護
4. **監控**：定期檢查 GPU、記憶體、log

## 建議的正式訓練設定

```ini
# conf.cfg 建議值（根據你的硬體調整）

learner_training_step=5000      # 完整訓練
players_per_batch=10            # 根據 GPU 記憶體調整
games_per_player=9              # 標準值
n_frames=10                     # 標準值
```

開始前先用 smoke test 驗證這些值不會 OOM！
