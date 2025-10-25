#!/usr/bin/env python3
"""
簡化版的圍棋棋風檢測 - 不依賴 C++ 編譯
直接使用 Python 和 PyTorch 進行開發

完整功能:
1. 訓練資料載入與玩家分組
2. Triplet Loss 訓練
3. Query/Candidate 推理
4. 生成 submission.csv
"""

import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sgfmill import sgf
from pathlib import Path
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import argparse

class SimpleGoEncoder(nn.Module):
    """簡單的圍棋棋風編碼器"""
    
    def __init__(self, board_size=19, n_frames=10, embedding_dim=128):
        super().__init__()
        self.board_size = board_size
        self.n_frames = n_frames
        self.embedding_dim = embedding_dim
        
        # CNN 特徵提取
        self.conv_layers = nn.Sequential(
            nn.Conv2d(n_frames * 3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Embedding layers
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # L2 normalize for better similarity computation
        x = F.normalize(x, p=2, dim=1)
        return x


def load_sgf_file(filepath):
    """載入單個 SGF 檔案並解析"""
    try:
        with open(filepath, 'rb') as f:
            game = sgf.Sgf_game.from_bytes(f.read())
        
        root = game.get_root()
        player_black = root.get('PB') if root.has_property('PB') else 'Unknown'
        player_white = root.get('PW') if root.has_property('PW') else 'Unknown'
        
        # 獲取棋譜
        moves = []
        node = root
        while node:
            children = node
            node = children[0] if children else None
            if node:
                move = node.get_move()
                if move[1]:  # 如果有落子
                    color, (row, col) = move
                    moves.append((color, row, col))
        
        return {
            'black': player_black,
            'white': player_white,
            'moves': moves,
            'filepath': filepath
        }
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def load_multi_game_sgf(filepath):
    """載入包含多場遊戲的 SGF 檔案 (用於 query/candidate set)"""
    try:
        with open(filepath, 'rb') as f:
            content = f.read()
        
        # sgfmill 不直接支援多遊戲檔案，需要手動分割
        # 簡單方法：按 "(;" 分割並逐個解析
        games = []
        game_strings = content.split(b'\n\n')  # 假設遊戲間有空行
        
        for game_str in game_strings:
            if not game_str.strip() or not game_str.startswith(b'('):
                continue
            try:
                game = sgf.Sgf_game.from_bytes(game_str)
                root = game.get_root()
                
                moves = []
                node = root
                while node:
                    children = node
                    node = children[0] if children else None
                    if node:
                        move = node.get_move()
                        if move[1]:
                            color, (row, col) = move
                            moves.append((color, row, col))
                
                if moves:  # 只保留有棋步的遊戲
                    games.append({'moves': moves})
            except:
                continue
        
        return games
    except Exception as e:
        print(f"Error loading multi-game SGF {filepath}: {e}")
        return []


def sgf_to_features(moves, n_frames=10, board_size=19, random_start=True):
    """將 SGF 棋步轉換為神經網路輸入特徵
    
    輸入格式: (n_frames*3, 19, 19)
    3 個通道: [黑子位置, 白子位置, 當前玩家]
    """
    if len(moves) < n_frames:
        return None
    
    # 選擇開始位置
    if random_start:
        start_idx = np.random.randint(0, max(1, len(moves) - n_frames + 1))
    else:
        start_idx = 0
    
    features = np.zeros((n_frames, 3, board_size, board_size), dtype=np.float32)
    board_black = np.zeros((board_size, board_size))
    board_white = np.zeros((board_size, board_size))
    
    for i, idx in enumerate(range(start_idx, min(start_idx + n_frames, len(moves)))):
        color, row, col = moves[idx]
        
        if color == 'b':
            board_black[row, col] = 1
            features[i, 0, :, :] = board_black.copy()
            features[i, 1, :, :] = board_white.copy()
            features[i, 2, :, :] = 1  # 黑方回合
        else:
            board_white[row, col] = 1
            features[i, 0, :, :] = board_black.copy()
            features[i, 1, :, :] = board_white.copy()
            features[i, 2, :, :] = 0  # 白方回合
    
    return features.reshape(-1, board_size, board_size)  # (n_frames*3, 19, 19)


class GoStyleDataset(Dataset):
    """圍棋風格資料集 - 用於訓練"""
    
    def __init__(self, player_games, n_frames=10):
        """
        Args:
            player_games: dict {player_id: [list of game dicts with 'moves']}
        """
        self.player_games = player_games
        self.player_ids = list(player_games.keys())
        self.n_frames = n_frames
        
        # 為每個玩家建立遊戲索引
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
        features = sgf_to_features(game['moves'], self.n_frames, random_start=True)
        
        if features is None:
            # Fallback: 回傳零向量
            features = np.zeros((self.n_frames * 3, 19, 19), dtype=np.float32)
        
        return torch.FloatTensor(features), player_id


def collate_triplet_fn(batch):
    """自定義 collate function 用於生成 triplet"""
    # 按玩家分組
    player_samples = defaultdict(list)
    for features, player_id in batch:
        player_samples[player_id].append(features)
    
    # 生成 triplets
    anchors, positives, negatives = [], [], []
    player_ids = list(player_samples.keys())
    
    if len(player_ids) < 2:
        # 不足以形成 triplet，回傳空
        return None
    
    for player_id in player_ids:
        if len(player_samples[player_id]) < 2:
            continue
        
        # Anchor 和 Positive 來自同一玩家
        anchor = player_samples[player_id][0]
        positive = player_samples[player_id][1] if len(player_samples[player_id]) > 1 else player_samples[player_id][0]
        
        # Negative 來自不同玩家
        other_players = [p for p in player_ids if p != player_id]
        if other_players:
            neg_player = np.random.choice(other_players)
            negative = player_samples[neg_player][0]
            
            anchors.append(anchor)
            positives.append(positive)
            negatives.append(negative)
    
    if not anchors:
        return None
    
    return (
        torch.stack(anchors),
        torch.stack(positives),
        torch.stack(negatives)
    )


class StyleDetectionSystem:
    """完整的棋風檢測系統"""
    
    def __init__(self, n_frames=10, embedding_dim=128, device='cuda'):
        self.n_frames = n_frames
        self.embedding_dim = embedding_dim
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        self.model = SimpleGoEncoder(
            n_frames=n_frames, 
            embedding_dim=embedding_dim
        ).to(self.device)
        
        print(f"🎯 模型參數數量: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"🖥️  使用設備: {self.device}")
    
    def load_train_data(self, train_dir='train_set'):
        """載入訓練資料並按玩家分組"""
        print(f"\n📂 載入訓練資料: {train_dir}")
        
        sgf_files = sorted(glob.glob(f"{train_dir}/*.sgf"))
        print(f"   找到 {len(sgf_files)} 個 SGF 檔案")
        
        # 按玩家名稱分組
        player_games = defaultdict(list)
        
        for filepath in tqdm(sgf_files, desc="讀取 SGF"):
            game_data = load_sgf_file(filepath)
            if game_data and len(game_data['moves']) >= self.n_frames:
                # 使用黑方玩家作為主要 ID
                player_id = game_data['black']
                player_games[player_id].append(game_data)
        
        print(f"   共有 {len(player_games)} 個不同玩家")
        print(f"   平均每人 {np.mean([len(games) for games in player_games.values()]):.1f} 場遊戲")
        
        return player_games
    
    def train(self, player_games, epochs=50, batch_size=32, lr=0.001, save_path='model_best.pth'):
        """訓練模型使用 Triplet Loss"""
        print(f"\n🚀 開始訓練...")
        
        dataset = GoStyleDataset(player_games, n_frames=self.n_frames)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            collate_fn=collate_triplet_fn,
            num_workers=0
        )
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        triplet_loss_fn = nn.TripletMarginLoss(margin=1.0)
        
        best_loss = float('inf')
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            valid_batches = 0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch in pbar:
                if batch is None:
                    continue
                
                anchors, positives, negatives = batch
                anchors = anchors.to(self.device)
                positives = positives.to(self.device)
                negatives = negatives.to(self.device)
                
                # Forward pass
                anchor_emb = self.model(anchors)
                positive_emb = self.model(positives)
                negative_emb = self.model(negatives)
                
                # Compute loss
                loss = triplet_loss_fn(anchor_emb, positive_emb, negative_emb)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                valid_batches += 1
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            if valid_batches > 0:
                avg_loss = epoch_loss / valid_batches
                print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")
                
                # 儲存最佳模型
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    torch.save(self.model.state_dict(), save_path)
                    print(f"✓ 儲存最佳模型: {save_path}")
        
        print(f"\n✅ 訓練完成！最佳 Loss: {best_loss:.4f}")
    
    def load_model(self, model_path):
        """載入訓練好的模型"""
        print(f"\n📥 載入模型: {model_path}")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print("✓ 模型載入完成")
    
    def extract_embedding(self, moves, num_samples=3):
        """從一場或多場遊戲提取 embedding"""
        self.model.eval()
        embeddings = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                features = sgf_to_features(moves, self.n_frames, random_start=True)
                if features is not None:
                    features = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                    embedding = self.model(features)
                    embeddings.append(embedding.cpu())
        
        if embeddings:
            # 平均多次採樣
            avg_embedding = torch.mean(torch.cat(embeddings, dim=0), dim=0)
            return avg_embedding
        return None
    
    def extract_player_embedding(self, games, max_games=9):
        """從多場遊戲提取玩家的平均 embedding"""
        embeddings = []
        
        for game in games[:max_games]:
            if 'moves' in game:
                emb = self.extract_embedding(game['moves'], num_samples=1)
                if emb is not None:
                    embeddings.append(emb)
        
        if embeddings:
            return torch.mean(torch.stack(embeddings), dim=0)
        return None
    
    def inference_on_test_set(self, query_dir='test_set/query_set', 
                               cand_dir='test_set/cand_set',
                               output_file='submission.csv'):
        """對測試集進行推理並生成提交檔案"""
        print(f"\n🔍 開始推理...")
        
        # 載入 Query Set
        print(f"\n📂 載入 Query Set: {query_dir}")
        query_files = sorted(glob.glob(f"{query_dir}/player*.sgf"))
        print(f"   找到 {len(query_files)} 個 query 檔案")
        
        query_embeddings = {}
        for filepath in tqdm(query_files, desc="Query embedding"):
            player_num = int(Path(filepath).stem.replace('player', ''))
            games = load_multi_game_sgf(filepath)
            
            if games:
                embedding = self.extract_player_embedding(games, max_games=9)
                if embedding is not None:
                    query_embeddings[player_num] = embedding
        
        print(f"   成功提取 {len(query_embeddings)} 個 query embeddings")
        
        # 載入 Candidate Set
        print(f"\n� 載入 Candidate Set: {cand_dir}")
        cand_files = sorted(glob.glob(f"{cand_dir}/player*.sgf"))
        print(f"   找到 {len(cand_files)} 個 candidate 檔案")
        
        cand_embeddings = {}
        for filepath in tqdm(cand_files, desc="Candidate embedding"):
            player_num = int(Path(filepath).stem.replace('player', ''))
            games = load_multi_game_sgf(filepath)
            
            if games:
                embedding = self.extract_player_embedding(games, max_games=9)
                if embedding is not None:
                    cand_embeddings[player_num] = embedding
        
        print(f"   成功提取 {len(cand_embeddings)} 個 candidate embeddings")
        
        # 計算相似度並配對
        print(f"\n🎯 計算相似度...")
        predictions = {}
        
        for query_id, query_emb in tqdm(query_embeddings.items(), desc="配對"):
            best_cand_id = None
            best_similarity = -1
            
            for cand_id, cand_emb in cand_embeddings.items():
                # Cosine similarity (因為 embedding 已經 normalized)
                similarity = torch.dot(query_emb, cand_emb).item()
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_cand_id = cand_id
            
            predictions[query_id] = best_cand_id
        
        # 生成 submission.csv
        print(f"\n💾 生成提交檔案: {output_file}")
        df = pd.DataFrame({
            'id': sorted(predictions.keys()),
            'label': [predictions[k] for k in sorted(predictions.keys())]
        })
        df.to_csv(output_file, index=False)
        print(f"✓ 已儲存 {len(df)} 筆預測結果")
        print(f"\n前 10 筆預測:")
        print(df.head(10))


def main():
    """主程式"""
    parser = argparse.ArgumentParser(description='圍棋棋風檢測系統')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference', 'full'],
                       help='運行模式: train/inference/full')
    parser.add_argument('--train_dir', type=str, default='train_set', help='訓練資料目錄')
    parser.add_argument('--query_dir', type=str, default='test_set/query_set', help='Query 資料目錄')
    parser.add_argument('--cand_dir', type=str, default='test_set/cand_set', help='Candidate 資料目錄')
    parser.add_argument('--model_path', type=str, default='model_best.pth', help='模型儲存路徑')
    parser.add_argument('--output', type=str, default='submission.csv', help='提交檔案路徑')
    parser.add_argument('--epochs', type=int, default=50, help='訓練輪數')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='學習率')
    parser.add_argument('--n_frames', type=int, default=10, help='使用的歷史 frames 數量')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding 維度')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔍 圍棋棋風檢測系統")
    print("=" * 60)
    
    # 初始化系統
    system = StyleDetectionSystem(
        n_frames=args.n_frames,
        embedding_dim=args.embedding_dim
    )
    
    if args.mode in ['train', 'full']:
        # 訓練模式
        player_games = system.load_train_data(args.train_dir)
        system.train(
            player_games,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            save_path=args.model_path
        )
    
    if args.mode in ['inference', 'full']:
        # 推理模式
        if args.mode == 'inference':
            system.load_model(args.model_path)
        
        system.inference_on_test_set(
            query_dir=args.query_dir,
            cand_dir=args.cand_dir,
            output_file=args.output
        )
    
    print("\n" + "=" * 60)
    print("✅ 完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
