#!/usr/bin/env python3
"""
平衡版本 - 在簡單和完整之間取得平衡
策略: 使用改進的特徵提取，但保持 Python 的簡潔性
"""

import os
import sys
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import argparse

# 導入改進的圍棋規則
from improved_go_features import sgf_to_features_with_rules, GoBoard

# 導入原有模型架構
from simple_style_detection import (
    SimpleGoEncoder, load_sgf_file, load_multi_game_sgf,
    GoStyleDataset, collate_triplet_fn
)


class ImprovedGoEncoder(nn.Module):
    """改進版編碼器 - 適配更多通道"""
    
    def __init__(self, board_size=19, n_frames=10, embedding_dim=128):
        super().__init__()
        self.board_size = board_size
        self.n_frames = n_frames
        self.embedding_dim = embedding_dim
        
        # 輸入通道數: n_frames*3 + 3 (氣+合法著手)
        input_channels = n_frames * 3 + 3
        
        # 更深的 CNN
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Embedding head
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, embedding_dim)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x


class ImprovedGoStyleDataset(Dataset):
    """使用改進特徵的資料集"""
    
    def __init__(self, player_games, n_frames=10, use_rules=True):
        self.player_games = player_games
        self.player_ids = list(player_games.keys())
        self.n_frames = n_frames
        self.use_rules = use_rules
        
        self.samples = []
        for player_id in self.player_ids:
            games = player_games[player_id]
            for game in games:
                if len(game['moves']) >= n_frames:
                    self.samples.append((player_id, game))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        player_id, game = self.samples[idx]
        
        if self.use_rules:
            features = sgf_to_features_with_rules(
                game['moves'], 
                self.n_frames, 
                random_start=True
            )
        else:
            from simple_style_detection import sgf_to_features
            features = sgf_to_features(
                game['moves'], 
                self.n_frames, 
                random_start=True
            )
        
        if features is None:
            # Fallback
            channels = self.n_frames * 3 + 3 if self.use_rules else self.n_frames * 3
            features = np.zeros((channels, 19, 19), dtype=np.float32)
        
        return torch.FloatTensor(features), player_id


def train_improved_model(
    train_dir='train_set',
    epochs=50,
    batch_size=32,
    lr=0.001,
    use_rules=True,
    model_path='model_improved.pth',
    device='cuda'
):
    """訓練改進版模型"""
    
    print("=" * 70)
    print("🚀 訓練改進版圍棋棋風檢測模型")
    print("=" * 70)
    print(f"使用完整圍棋規則: {'✅ 是' if use_rules else '❌ 否'}")
    print(f"設備: {device}")
    
    # 載入資料
    print(f"\n📂 載入訓練資料...")
    sgf_files = sorted(glob.glob(f"{train_dir}/*.sgf"))
    player_games = defaultdict(list)
    
    for filepath in tqdm(sgf_files, desc="讀取 SGF"):
        game_data = load_sgf_file(filepath)
        if game_data and len(game_data['moves']) >= 10:
            player_id = game_data['black']
            player_games[player_id].append(game_data)
    
    print(f"   玩家數: {len(player_games)}")
    
    # 建立資料集
    dataset = ImprovedGoStyleDataset(
        player_games, 
        n_frames=10, 
        use_rules=use_rules
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_triplet_fn,
        num_workers=0
    )
    
    # 建立模型
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = ImprovedGoEncoder(n_frames=10, embedding_dim=128).to(device)
    
    print(f"\n🎯 模型參數: {sum(p.numel() for p in model.parameters()):,}")
    
    # 訓練
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    triplet_loss_fn = nn.TripletMarginLoss(margin=1.0)
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        valid_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            if batch is None:
                continue
            
            anchors, positives, negatives = batch
            anchors = anchors.to(device)
            positives = positives.to(device)
            negatives = negatives.to(device)
            
            anchor_emb = model(anchors)
            positive_emb = model(positives)
            negative_emb = model(negatives)
            
            loss = triplet_loss_fn(anchor_emb, positive_emb, negative_emb)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            valid_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        if valid_batches > 0:
            avg_loss = epoch_loss / valid_batches
            print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), model_path)
                print(f"✓ 儲存最佳模型")
    
    print(f"\n✅ 訓練完成！最佳 Loss: {best_loss:.4f}")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['simple', 'improved', 'compare'], 
                       default='improved')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    
    args = parser.parse_args()
    
    if args.mode == 'simple':
        print("🔹 使用簡單版本 (快速但可能不準)")
        from simple_style_detection import StyleDetectionSystem
        system = StyleDetectionSystem()
        player_games = system.load_train_data()
        system.train(player_games, epochs=args.epochs)
        
    elif args.mode == 'improved':
        print("⭐ 使用改進版本 (準確但稍慢)")
        train_improved_model(
            epochs=args.epochs,
            batch_size=args.batch_size,
            use_rules=True
        )
        
    elif args.mode == 'compare':
        print("📊 比較兩種版本...")
        print("\n1️⃣ 簡單版本:")
        train_improved_model(
            epochs=5,
            use_rules=False,
            model_path='model_simple_test.pth'
        )
        
        print("\n2️⃣ 改進版本:")
        train_improved_model(
            epochs=5,
            use_rules=True,
            model_path='model_improved_test.pth'
        )


if __name__ == "__main__":
    main()
