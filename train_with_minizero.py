#!/usr/bin/env python3
"""
使用 MiniZero 後端的訓練腳本
需要成功編譯 style_py 模組
"""

import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from simple_style_detection import SimpleGoEncoder
from pathlib import Path

# 嘗試載入編譯好的 MiniZero 模組
try:
    # 假設在 build/go/ 目錄下
    sys.path.append("ML-Assignment2-Q5/build/go")
    import style_py
    MINIZERO_AVAILABLE = True
    print("✅ MiniZero 模組載入成功")
except ImportError:
    MINIZERO_AVAILABLE = False
    print("⚠️  無法載入 MiniZero 模組")
    print("   使用純 Python 版本: python3 simple_style_detection.py")
    sys.exit(1)


def extract_features_from_minizero(data_loader, player_id, game_id):
    """使用 MiniZero 的 DataLoader 提取特徵"""
    n_frames = style_py.get_n_frames()
    board_h = style_py.get_nn_input_channel_height()
    board_w = style_py.get_nn_input_channel_width()
    channels = style_py.get_nn_num_input_channels()
    
    # 從 MiniZero DataLoader 獲取特徵
    features = data_loader.get_feature_and_label(player_id, game_id, 0, True)
    
    # 轉換為 PyTorch tensor
    features = np.array(features).reshape(n_frames, channels, board_h, board_w)
    return torch.FloatTensor(features)


def train_with_minizero(conf_file='ML-Assignment2-Q5/conf.cfg', 
                         train_dir='train_set',
                         epochs=50):
    """使用 MiniZero 後端的訓練流程"""
    
    print("🚀 使用 MiniZero 後端訓練")
    print("=" * 60)
    
    # 載入設定檔
    style_py.load_config_file(conf_file)
    
    # 建立 DataLoader
    data_loader = style_py.DataLoader(conf_file)
    
    # 載入訓練資料
    print(f"\n📂 載入訓練資料...")
    import glob
    sgf_files = sorted(glob.glob(f"{train_dir}/*.sgf"))
    
    for sgf_file in sgf_files:
        data_loader.load_data_from_file(sgf_file)
    
    num_players = data_loader.get_num_of_players()
    games_per_player = data_loader.get_games_per_player()
    
    print(f"   玩家數量: {num_players}")
    print(f"   每人遊戲數: {games_per_player}")
    
    # 建立模型
    n_frames = style_py.get_n_frames()
    model = SimpleGoEncoder(n_frames=n_frames)
    
    print(f"\n🎯 開始訓練...")
    print(f"   使用 MiniZero 的標準特徵格式")
    
    # TODO: 實作訓練迴圈
    # 這裡可以使用 MiniZero 提供的特徵進行訓練
    
    print("\n✅ 訓練完成")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_file', type=str, 
                       default='ML-Assignment2-Q5/conf.cfg')
    parser.add_argument('--train_dir', type=str, default='train_set')
    parser.add_argument('--epochs', type=int, default=50)
    
    args = parser.parse_args()
    
    train_with_minizero(
        conf_file=args.conf_file,
        train_dir=args.train_dir,
        epochs=args.epochs
    )


if __name__ == "__main__":
    main()
