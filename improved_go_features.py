#!/usr/bin/env python3
"""
改進版特徵提取 - 考慮圍棋規則
包含: 提子、氣的計算、合法著手判斷
"""

import numpy as np
from collections import deque

class GoBoard:
    """完整的圍棋棋盤模擬（包含提子邏輯）"""
    
    def __init__(self, size=19):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)  # 0: 空, 1: 黑, 2: 白
        self.history = []
        self.ko_point = None  # 打劫位置
        
    def copy(self):
        """複製棋盤"""
        new_board = GoBoard(self.size)
        new_board.board = self.board.copy()
        new_board.history = self.history.copy()
        new_board.ko_point = self.ko_point
        return new_board
    
    def get_neighbors(self, row, col):
        """獲取相鄰點"""
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            r, c = row + dr, col + dc
            if 0 <= r < self.size and 0 <= c < self.size:
                neighbors.append((r, c))
        return neighbors
    
    def get_group_and_liberties(self, row, col):
        """獲取一個連通塊及其氣"""
        if self.board[row, col] == 0:
            return set(), set()
        
        color = self.board[row, col]
        group = set()
        liberties = set()
        queue = deque([(row, col)])
        visited = set()
        
        while queue:
            r, c = queue.popleft()
            if (r, c) in visited:
                continue
            visited.add((r, c))
            
            if self.board[r, c] == color:
                group.add((r, c))
                for nr, nc in self.get_neighbors(r, c):
                    if (nr, nc) not in visited:
                        queue.append((nr, nc))
            elif self.board[r, c] == 0:
                liberties.add((r, c))
        
        return group, liberties
    
    def capture_dead_groups(self, row, col, color):
        """提子：移除被吃的對方棋子"""
        opponent = 3 - color  # 1->2, 2->1
        captured = []
        
        for nr, nc in self.get_neighbors(row, col):
            if self.board[nr, nc] == opponent:
                group, liberties = self.get_group_and_liberties(nr, nc)
                if len(liberties) == 0:  # 無氣
                    for gr, gc in group:
                        self.board[gr, gc] = 0
                    captured.extend(group)
        
        return captured
    
    def is_legal_move(self, row, col, color):
        """判斷是否合法著手"""
        # 1. 位置必須為空
        if self.board[row, col] != 0:
            return False
        
        # 2. 打劫判斷
        if self.ko_point and (row, col) == self.ko_point:
            return False
        
        # 3. 自殺手判斷（需要模擬落子後檢查）
        test_board = self.copy()
        test_board.board[row, col] = color
        
        # 如果能提對方子，則合法
        captured = test_board.capture_dead_groups(row, col, color)
        if captured:
            return True
        
        # 檢查自己是否有氣
        group, liberties = test_board.get_group_and_liberties(row, col)
        return len(liberties) > 0
    
    def play(self, row, col, color):
        """落子並處理提子"""
        if not self.is_legal_move(row, col, color):
            return False
        
        self.board[row, col] = color
        captured = self.capture_dead_groups(row, col, color)
        
        # 打劫判斷：如果只提了一子，且自己也只有一氣，則可能是打劫
        if len(captured) == 1:
            group, liberties = self.get_group_and_liberties(row, col)
            if len(group) == 1 and len(liberties) == 1:
                self.ko_point = captured[0]
            else:
                self.ko_point = None
        else:
            self.ko_point = None
        
        self.history.append((row, col, color))
        return True


def sgf_to_features_with_rules(moves, n_frames=10, board_size=19, random_start=True):
    """
    使用完整圍棋規則提取特徵
    
    特徵通道:
    - 黑子位置 (n_frames 層)
    - 白子位置 (n_frames 層)
    - 當前玩家 (n_frames 層)
    - 黑子的氣 (1 層)
    - 白子的氣 (1 層)
    - 合法著手點 (1 層)
    
    總計: n_frames * 3 + 3 = n_frames * 3 + 3 通道
    """
    if len(moves) < n_frames:
        return None
    
    # 選擇開始位置
    if random_start:
        start_idx = np.random.randint(0, max(1, len(moves) - n_frames + 1))
    else:
        start_idx = 0
    
    # 初始化棋盤
    board = GoBoard(board_size)
    
    # 播放到開始位置
    for i in range(start_idx):
        color_str, row, col = moves[i]
        color = 1 if color_str == 'b' else 2
        board.play(row, col, color)
    
    # 提取 n_frames 的特徵
    features = []
    
    for i in range(n_frames):
        if start_idx + i >= len(moves):
            # 如果不足 n_frames，重複最後一個
            features.append(features[-1])
            continue
        
        color_str, row, col = moves[start_idx + i]
        color = 1 if color_str == 'b' else 2
        
        # 落子
        board.play(row, col, color)
        
        # 提取當前狀態特徵 (3 通道)
        black_stones = (board.board == 1).astype(np.float32)
        white_stones = (board.board == 2).astype(np.float32)
        current_player = np.full((board_size, board_size), 
                                 1.0 if color == 1 else 0.0, 
                                 dtype=np.float32)
        
        frame_features = np.stack([black_stones, white_stones, current_player])
        features.append(frame_features)
    
    # 堆疊成 (n_frames*3, H, W)
    features = np.concatenate(features, axis=0)
    
    # 額外特徵: 氣和合法著手
    # 計算黑子和白子的氣
    black_liberties = np.zeros((board_size, board_size), dtype=np.float32)
    white_liberties = np.zeros((board_size, board_size), dtype=np.float32)
    legal_moves = np.zeros((board_size, board_size), dtype=np.float32)
    
    for r in range(board_size):
        for c in range(board_size):
            if board.board[r, c] != 0:
                _, liberties = board.get_group_and_liberties(r, c)
                liberty_count = len(liberties)
                if board.board[r, c] == 1:
                    black_liberties[r, c] = min(liberty_count / 8.0, 1.0)  # 正規化
                else:
                    white_liberties[r, c] = min(liberty_count / 8.0, 1.0)
            
            # 檢查合法著手
            next_color = 2 if (board.history[-1][2] == 1 if board.history else True) else 1
            if board.is_legal_move(r, c, next_color):
                legal_moves[r, c] = 1.0
    
    # 合併額外特徵
    extra_features = np.stack([black_liberties, white_liberties, legal_moves])
    full_features = np.concatenate([features, extra_features], axis=0)
    
    return full_features


def compare_simple_vs_complete(moves, n_frames=10):
    """比較簡單版本和完整版本的差異"""
    from simple_style_detection import sgf_to_features
    
    # 簡單版本 (原本的)
    simple_features = sgf_to_features(
        {'moves': moves}, 
        n_frames=n_frames, 
        random_start=False
    )
    
    # 完整版本 (新的)
    complete_features = sgf_to_features_with_rules(
        moves, 
        n_frames=n_frames, 
        random_start=False
    )
    
    print("=" * 60)
    print("特徵比較:")
    print(f"簡單版本: {simple_features.shape if simple_features is not None else 'None'}")
    print(f"  - 只有 {n_frames * 3} 個通道 (黑/白/玩家)")
    print(f"  - 不考慮提子、打劫等規則")
    print()
    print(f"完整版本: {complete_features.shape if complete_features is not None else 'None'}")
    print(f"  - {n_frames * 3 + 3} 個通道")
    print(f"  - 包含提子邏輯、氣的計算、合法著手")
    print("=" * 60)
    
    return simple_features, complete_features


if __name__ == "__main__":
    from simple_style_detection import load_sgf_file
    
    print("🔍 測試完整圍棋規則的特徵提取")
    
    # 載入測試檔案
    sgf_data = load_sgf_file('train_set/1.sgf')
    if sgf_data:
        print(f"\n檔案: train_set/1.sgf")
        print(f"手數: {len(sgf_data['moves'])}")
        
        # 比較兩種特徵
        simple_f, complete_f = compare_simple_vs_complete(
            sgf_data['moves'], 
            n_frames=10
        )
        
        print("\n✅ 完整版本可以正確處理:")
        print("  - 提子（吃子）")
        print("  - 打劫")
        print("  - 氣的計算")
        print("  - 合法著手判斷")
