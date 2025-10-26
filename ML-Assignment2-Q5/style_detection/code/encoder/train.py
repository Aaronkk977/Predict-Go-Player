from encoder.model import Encoder
from profiler import Profiler
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset
from torch.utils.data import get_worker_info
import numpy as np
import sys
import torch
import glob
import time
import random


class MinizeroDataset(IterableDataset):
    def __init__(self, files, conf_file, game_type, style_py_module):

        # Fully implemented dataloader that connects to the C++ side via pybind
        # and retrieves complete features from the backend.
        # 
        # IMPORTANT: We do NOT load all files at once to avoid OOM.
        # Instead, we load files in batches and clear after use.
        #
        # NOTE: We receive style_py as parameter to avoid re-loading config

        # Use the passed style_py module (already configured)
        self.style_py = style_py_module
        self.data_loader = self.style_py.DataLoader(conf_file)
        
        self.files = files  # Store file paths, don't load yet
        self.conf_file = conf_file
        self.game_type = game_type
        
        self.player_choosed = 0
        self.games_per_player = self.style_py.get_games_per_player()
        self.players_per_batch = self.style_py.get_players_per_batch()
        self.n_frames = self.style_py.get_n_frames()
        self.training_feature_mode = 2
        self.input_channel_feature = self.style_py.get_nn_num_input_channels()
        self.board_size_h = self.style_py.get_nn_input_channel_height()
        self.board_size_w = self.style_py.get_nn_input_channel_width()
        
        # Batch loading parameters
        self.files_per_batch = 10  # Load 10 SGF files at a time
        self.current_file_idx = 0
        self.files_loaded = False
        
        print(f"Dataset initialized with {len(self.files)} SGF files")
        print(f"Config: players_per_batch={self.players_per_batch}, games_per_player={self.games_per_player}, n_frames={self.n_frames}")
        print(f"Board: {self.board_size_h}x{self.board_size_w}, channels={self.input_channel_feature}")
        print(f"Will load in batches of {self.files_per_batch} files to save memory")
        
    def _load_next_batch(self):
        """Load next batch of SGF files to avoid OOM"""
        # CRITICAL: Always clear before loading new batch!
        # C++ data_loader accumulates data, must clear first
        if self.files_loaded:
            print("Clearing previous batch from memory...")
        self.data_loader.Clear_Sgf()  # Clear regardless of files_loaded state
        
        # Get next batch of files
        if self.current_file_idx + self.files_per_batch <= len(self.files):
            end_idx = self.current_file_idx + self.files_per_batch
            batch_files = self.files[self.current_file_idx:end_idx]
        else:
            # Wrap around if we reach the end
            end_idx = min(self.files_per_batch, len(self.files))
            batch_files = self.files[0:end_idx]
        
        print(f"Loading files {self.current_file_idx+1}-{end_idx}/{len(self.files)}...")
        for file_name in batch_files:
            try:
                self.data_loader.load_data_from_file(file_name)
            except Exception as e:
                print(f"Warning: Failed to load {file_name}: {e}")
                continue
        
        num_players = self.data_loader.get_num_of_player()
        if num_players == 0:
            print("ERROR: No players loaded! Check SGF files.")
            return
            
        self.random_player = [i for i in range(num_players)]
        random.shuffle(self.random_player)
        self.player_choosed = 0
        self.files_loaded = True
        
        print(f"Loaded {num_players} players from {len(batch_files)} files")
        
        # Move to next batch
        self.current_file_idx = end_idx
        if self.current_file_idx >= len(self.files):
            self.current_file_idx = 0  # Loop back

    def __iter__(self):

        # Each time calling "inputs = next(data_loader_iterator)"
        # will directly get one complete batch of data.
        # 
        # With batched file loading, we need to check if we need more data
        # and reload when necessary.

        while True:
            # Load first batch if not loaded yet
            if not self.files_loaded:
                self._load_next_batch()
            
            # Safety check
            if len(self.random_player) == 0:
                print("ERROR: No players available!")
                break
            
            # Check if we need to load next batch
            if self.player_choosed >= len(self.random_player):
                print("Finished current batch, loading next...")
                self._load_next_batch()
                continue  # Skip this iteration, start fresh with new batch
            
            # Reset if we've gone through enough players for this batch
            if self.player_choosed >= self.players_per_batch:
                self.player_choosed = 0
                random.shuffle(self.random_player)
            
            # Safety check before accessing data
            player_idx = self.random_player[self.player_choosed]
            if player_idx >= self.data_loader.get_num_of_player():
                print(f"Warning: player_idx {player_idx} out of range, resetting...")
                self.player_choosed = 0
                continue
                
            try:
                if self.training_feature_mode == 1:
                    features = self.data_loader.get_feature_and_label(player_idx, 1, 0, 1)
                elif self.training_feature_mode == 2:
                    features = self.data_loader.get_random_feature_and_label(player_idx, 1, 0, 1)
                    
                yield torch.FloatTensor(features).view(
                    1 * self.games_per_player, 
                    self.n_frames, 
                    self.input_channel_feature, 
                    self.board_size_h, 
                    self.board_size_w
                )
                self.player_choosed += 1
                
            except Exception as e:
                print(f"Error getting features for player {player_idx}: {e}")
                self.player_choosed += 1
                continue


def sync(device: torch.device):
    # For correct profiling (cuda operations are async)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train(run_id: str, game_type: str, data_dir: str, validate_data_dir: str, models_dir: Path,
          save_every: int, backup_every: int, validate_every: int, force_restart: bool, conf_file: str
          ):
    # This function shows a simplified example of the overall training flow.
    # It demonstrates how to create the dataset, data loader, model, and
    # perform a forward pass to obtain outputs from the neural network.
    # Students are expected to design their own model architecture, loss
    # computation, and optimization steps.

    # Import the compiled pybind library for the given game type
    import os
    build_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..', 'build', game_type))
    if build_path not in sys.path:
        sys.path.insert(0, build_path)
    
    import style_py
    
    # Load config ONCE and show what we loaded
    print(f"Loading configuration from: {conf_file}")
    style_py.load_config_file(conf_file)
    
    # Verify config was loaded correctly
    print(f"Config verification:")
    print(f"  - Training steps: {style_py.get_training_step()}")
    print(f"  - Players per batch: {style_py.get_players_per_batch()}")
    print(f"  - Games per player: {style_py.get_games_per_player()}")
    print(f"  - N frames: {style_py.get_n_frames()}")
    print(f"  - Board size: {style_py.get_nn_input_channel_height()}x{style_py.get_nn_input_channel_width()}")
    print("")

    # Load all SGF files from the training set directory
    # Try multiple possible locations
    possible_paths = [
        "/workspace/data_set/train_set/*.sgf",
        "../../data_set/train_set/*.sgf",
        "/tmp2/b12902115/Predict-Go-Player/data_set/train_set/*.sgf"
    ]
    
    all_file_list = []
    for sgf_location in possible_paths:
        all_file_list = glob.glob(sgf_location)
        if len(all_file_list) > 0:
            print(f"Found SGF files at: {sgf_location}")
            break
    
    if len(all_file_list) == 0:
        print(f"ERROR: No SGF files found in any of these locations:")
        for path in possible_paths:
            print(f"  - {path}")
        print("\nPlease check the path and make sure training data exists.")
        return
    
    # Limit files for testing/debugging (comment out for full training)
    # For smoke test, use only a few files
    if "smoke" in run_id.lower() or style_py.get_training_step() <= 50:
        all_file_list = all_file_list[:3]  # Only first 3 files for smoke test
        print(f"SMOKE TEST MODE: Limited to {len(all_file_list)} files")
    
    print(f"Total SGF files for training: {len(all_file_list)}")

    # Create dataset and dataloader
    # Pass style_py module to avoid re-loading config
    dataset = MinizeroDataset(all_file_list, conf_file, game_type, style_py)
    
    # Dataset yields one player at a time, DataLoader will batch them
    # Use default collate_fn which will stack tensors along batch dimension
    
    # Setup device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    
    # Enable GPU optimizations if CUDA is available
    use_gpu = torch.cuda.is_available()
    num_workers = 2 if use_gpu else 0  # Use workers for GPU
    
    data_loader = DataLoader(
        dataset, 
        batch_size=style_py.get_players_per_batch(),
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=use_gpu,  # Enable pin_memory for GPU
        drop_last=False  # Keep incomplete batches at the end
    )
    data_loader_iterator = iter(data_loader)

    # Create model and optimizer
    model = Encoder(device, conf_file, game_type)

    multi_gpu = False
    if torch.cuda.device_count() > 1:
        multi_gpu = True
        model = torch.nn.DataParallel(model)
        print(f"🔥 Using {torch.cuda.device_count()} GPUs with DataParallel")
    elif torch.cuda.is_available():
        print(f"🔥 Using single GPU: {torch.cuda.get_device_name(0)}")
    
    model.to(device)

    model.train()

    # Get board dimensions from config
    board_size_h = style_py.get_nn_input_channel_height()
    board_size_w = style_py.get_nn_input_channel_width()

    # Main training loop
    # Triplet loss training with anchor, positive, negative samples
    step = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    print(f"\n{'='*60}")
    print(f"🎯 Starting training for {style_py.get_training_step()} steps")
    print(f"{'='*60}")
    print(f"Board size: {board_size_h}x{board_size_w}")
    print(f"Players per batch: {style_py.get_players_per_batch()}")
    print(f"Games per player: {style_py.get_games_per_player()}")
    print(f"Input channels: {style_py.get_nn_num_input_channels()}")
    print(f"N frames: {style_py.get_n_frames()}")
    print(f"DataLoader workers: {num_workers}")
    print(f"Pin memory: {use_gpu}")
    print(f"{'='*60}\n")
    
    while step < style_py.get_training_step():
        step = step + 1

        try:
            # Retrieve one batch of input features
            # Shape: (players_per_batch, games_per_player, n_frames, channels, H, W)
            inputs = next(data_loader_iterator)

            # Debug: print actual shape for first few steps
            if step <= 3:
                print(f"DEBUG Step {step} - Input shape: {inputs.shape}, numel: {inputs.numel()}")
                print(f"  Expected: [{style_py.get_players_per_batch()}, {style_py.get_games_per_player()}, {style_py.get_n_frames()}, {style_py.get_nn_num_input_channels()}, {board_size_h}, {board_size_w}]")

            # Get actual batch size (might be less than players_per_batch)
            actual_batch_size = inputs.size(0)
            
            # Skip if batch is too small for triplet loss
            if actual_batch_size < 2:
                print(f"Warning: Batch too small ({actual_batch_size}), skipping...")
                continue

            # Reshape input to the expected tensor shape:
            # [actual_batch_size * games_per_player,
            #  n_frames, num_input_channels, board_H, board_W]
            inputs = inputs.reshape(
                actual_batch_size * style_py.get_games_per_player(),
                style_py.get_n_frames(),
                style_py.get_nn_num_input_channels(),
                board_size_h,
                board_size_w
            ).to(device)

            # Synchronize device before forward pass
            sync(device)

            # Forward pass through the model to get embeddings
            # Shape: (actual_batch_size * games_per_player, 128)
            embeddings = model(inputs)
            
            # Reshape embeddings back to (actual_batch_size, games_per_player, 128)
            embeddings = embeddings.view(
                actual_batch_size,
                style_py.get_games_per_player(),
                -1
            )
            
            # Construct triplets for triplet loss
            # For each player:
            #   - Anchor: first game
            #   - Positive: other games from same player
            #   - Negative: games from different players
            
            anchor_list = []
            positive_list = []
            negative_list = []
            
            for player_idx in range(actual_batch_size):
                # Anchor: use first game of current player
                anchor = embeddings[player_idx, 0, :]
                
                # Positive: use other games from same player
                for game_idx in range(1, style_py.get_games_per_player()):
                    positive = embeddings[player_idx, game_idx, :]
                    
                    # Negative: randomly select from different player
                    neg_player = (player_idx + np.random.randint(1, actual_batch_size)) % actual_batch_size
                    neg_game = np.random.randint(0, style_py.get_games_per_player())
                    negative = embeddings[neg_player, neg_game, :]
                    
                    anchor_list.append(anchor)
                    positive_list.append(positive)
                    negative_list.append(negative)
            
            # Stack all triplets
            anchors = torch.stack(anchor_list)
            positives = torch.stack(positive_list)
            negatives = torch.stack(negative_list)
            
            # Calculate triplet loss
            if multi_gpu:
                loss = model.module.loss(anchors, positives, negatives, margin=1.0)
            else:
                loss = model.loss(anchors, positives, negatives, margin=1.0)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Print progress
            if step % 10 == 0:
                print(f"Step {step}/{style_py.get_training_step()}: Loss = {loss.item():.4f}, Batch size = {actual_batch_size}")
            
            # Save checkpoint
            if save_every > 0 and step % save_every == 0:
                checkpoint_path = models_dir / f"{run_id}_step_{step}.pt"
                if multi_gpu:
                    torch.save({
                        'step': step,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss.item(),
                    }, checkpoint_path)
                else:
                    torch.save({
                        'step': step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss.item(),
                    }, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
                
        except Exception as e:
            print(f"Error at step {step}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save final model
    final_path = models_dir / f"{run_id}_final.pt"
    if multi_gpu:
        torch.save(model.module.state_dict(), final_path)
    else:
        torch.save(model.state_dict(), final_path)
    print(f"Training completed! Final model saved to {final_path}")



