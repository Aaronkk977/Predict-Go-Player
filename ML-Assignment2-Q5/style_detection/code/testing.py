import argparse 
import glob 
from numpy import dot 
import numpy as np 
import torch 
from encoder.model import Encoder 
import sys
import time
import csv
import os
import io
import contextlib
import torch.nn.functional as F
sys.path.append("../style_detection/")

def parse_args() : 
    # parse command-line arguments for game type and config file
    parser = argparse.ArgumentParser()
    parser.add_argument("game_type", type=str, help="Name for game type.")
    parser.add_argument("-c", "--conf_file", type=str, default="./conf.cfg", help="Configuration file path")
    parser.add_argument("-m", "--model_path", type=str, required=True, help="Path to trained model weights")
    args = parser.parse_args()
    return args


class style_detection:
    """
    Style detection inference for Go player identification.
    Extracts embeddings for query and candidate players,
    computes similarity, and generates submission.csv.
    """

    def __init__(self, conf_file, game_type):
        # Initialize configuration and DataLoader from C++ backend
        self.game_type = game_type
        self.conf_file = conf_file
        
        # Import C++ module
        build_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'build', game_type))
        if build_path not in sys.path:
            sys.path.insert(0, build_path)
        
        import style_py
        global style_py
        style_py.load_config_file(conf_file)
        
        self.data_loader = style_py.DataLoader(conf_file)
        self.n_frames = style_py.get_n_frames()
        self.board_size_h = style_py.get_nn_input_channel_height()
        self.board_size_w = style_py.get_nn_input_channel_width()
        self.input_channel_feature = style_py.get_nn_num_input_channels()
        
        # Use random sampling mode
        self.testing_feature_mode = 2
        self.start = 0  # starting move position
        
        # Storage for embeddings
        self.query_embeddings = []
        self.cand_embeddings = []

    def read_sgf(self, sgf_dir):
        """Load all SGF files from directory (600 players)"""
        print('Start reading SGF files from:', sgf_dir)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}]")
        print(f"DEBUG: Players before loading: {self.data_loader.get_num_of_player()}")

        for i in range(600):
            # Test set uses player001.sgf format (3-digit padding)
            sgf_path = sgf_dir + "/player" + str(i + 1).zfill(3) + ".sgf"
            self.data_loader.load_data_from_file(sgf_path)
            if (i + 1) % 100 == 0:
                print(f"  Loaded {i + 1}/600 players")
                print(f"  DEBUG: Current player count: {self.data_loader.get_num_of_player()}")
        
        # Check immediately after loading
        self.data_loader.Check_Sgf()  # Print debug info from C++
        
        final_count = self.data_loader.get_num_of_player()
        print(f'Finished reading SGF files')
        print(f"DEBUG: Final player count: {final_count}")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}]")

    def load_model(self, model_path):
        """Load trained model"""
        print(f"Loading model from: {model_path}")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.model = Encoder(self.device, self.conf_file, self.game_type)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")
        else:
            self.model.load_state_dict(checkpoint)
        
        # Handle multi-GPU if available
        multi_gpu = False
        if torch.cuda.device_count() > 1:
            multi_gpu = True
            self.model = torch.nn.DataParallel(self.model)
            print(f"  Using {torch.cuda.device_count()} GPUs")

        self.model.to(self.device)
        self.model.eval()  # set model to eval mode
        print("Model loaded successfully")

    def inference(self, data_loader, is_query=True):
        """
        Run inference on all players and collect embeddings
        
        Args:
            data_loader: C++ DataLoader instance
            is_query: True for query set, False for candidate set
        """
        num_players = data_loader.get_num_of_player()
        print(f"\nRunning inference on {num_players} players...")
        
        embeddings_list = []
        
        for player_id in range(num_players):
            player_embeddings = []
            failed_count = 0
            
            # Process each game for this player
            for game_id in range(100):  # 100 games per player
                try:
                    # Get features from C++ backend
                    if self.testing_feature_mode == 1:
                        features = data_loader.get_feature_and_label(player_id, game_id, self.start, 0)
                    elif self.testing_feature_mode == 2:
                        features = data_loader.get_random_feature_and_label(player_id, game_id, self.start, 0)

                    # Convert features to tensor
                    features = torch.FloatTensor(features).view(
                        1, self.n_frames, self.input_channel_feature, 
                        self.board_size_h, self.board_size_w
                    )
                    features = features.to(self.device)

                    # Forward pass to get embedding
                    with torch.no_grad():
                        embedding = self.model(features)
                    
                    player_embeddings.append(embedding.cpu())
                
                except Exception as e:
                    # Track failures for debugging
                    failed_count += 1
                    if failed_count == 1 and player_id < 3:  # Print first few failures
                        print(f"  [Warn] Player {player_id} game {game_id} failed: {type(e).__name__}")
                    continue
            
            if len(player_embeddings) > 0:
                # Average embeddings across all games for this player
                player_embedding = torch.stack(player_embeddings).mean(dim=0)
                embeddings_list.append(player_embedding)
                if failed_count > 50:
                    print(f"  [Warn] Player {player_id}: only {len(player_embeddings)}/100 games succeeded")
            else:
                # If no valid games, use zero embedding (will be caught by diagnostics)
                embeddings_list.append(torch.zeros(1, 128))
                print(f"  [ERROR] Player {player_id}: ALL games failed! Using zero embedding.")
            
            if (player_id + 1) % 50 == 0:
                print(f"  Processed {player_id + 1}/{num_players} players")
        
        # Stack all embeddings
        all_embeddings = torch.cat(embeddings_list, dim=0)  # (num_players, 128)
        
        # Simple diagnostics: count zero-norm embeddings (can cause degenerate similarities)
        with torch.no_grad():
            norms = torch.norm(all_embeddings, p=2, dim=1)
            zero_count = int((norms == 0).sum().item())
            total = all_embeddings.shape[0]
            if is_query:
                self.query_embeddings = all_embeddings
                print(f"Query embeddings shape: {self.query_embeddings.shape}")
                print(f"[Diag] Query zero-norm embeddings: {zero_count}/{total}")
            else:
                self.cand_embeddings = all_embeddings
                print(f"Candidate embeddings shape: {self.cand_embeddings.shape}")
                print(f"[Diag] Candidate zero-norm embeddings: {zero_count}/{total}")

    def compute_similarity_and_save(self, output_path="submission.csv"):
        """
        Compute cosine similarity between query and candidate embeddings
        and generate submission.csv
        """
        print("\nComputing similarities...")
        
        # Validate embeddings
        if not isinstance(self.query_embeddings, torch.Tensor) or not isinstance(self.cand_embeddings, torch.Tensor):
            raise RuntimeError("Embeddings not computed. Run inference for query and candidate first.")

        num_queries = self.query_embeddings.shape[0]
        num_cands = self.cand_embeddings.shape[0]
        print(f"Embeddings: queries={num_queries}, candidates={num_cands}")
        if num_queries == 0 or num_cands == 0:
            raise RuntimeError("No embeddings available to compute similarity.")

        # Normalize embeddings for cosine similarity (add eps to avoid NaNs on zero vectors)
        query_norm = F.normalize(self.query_embeddings, p=2, dim=1, eps=1e-12)
        cand_norm = F.normalize(self.cand_embeddings, p=2, dim=1, eps=1e-12)
        
        # Compute similarity matrix (num_queries x num_cands)
        similarity_matrix = torch.mm(query_norm, cand_norm.t())
        
        # Replace NaN/Inf values (can appear if any degenerate vectors slipped through)
        similarity_matrix = torch.nan_to_num(similarity_matrix, nan=-1.0, posinf=1.0, neginf=-1.0)
        
        # For each query, find the most similar candidate
        predictions = []
        for query_id in range(num_queries):
            similarities = similarity_matrix[query_id]
            most_similar_cand = torch.argmax(similarities).item()
            predictions.append((query_id + 1, most_similar_cand + 1))
        
        # Save to CSV (competition format: id,label)
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'label'])  # header
            for query_id, cand_id in predictions:
                writer.writerow([query_id, cand_id])
        
        print(f"Saved predictions to {output_path}")
        print(f"Total predictions: {len(predictions)}")

    def testing(self, model_path):
        """Complete testing procedure"""
        with torch.no_grad():
            # Load model first
            self.load_model(model_path=model_path)
            
            # Load and process query set
            print("\n" + "="*60)
            print("PHASE 1: Processing Query Set")
            print("="*60)
            self.read_sgf(sgf_dir="../../../data_set/test_set/query_set")
            print(f"DEBUG: Number of players loaded: {self.data_loader.get_num_of_player()}")
            self.inference(self.data_loader, is_query=True)

            # Clear SGF data
            self.data_loader.Clear_Sgf()
            print("\nCleared query set from memory")

            # Load and process candidate set
            print("\n" + "="*60)
            print("PHASE 2: Processing Candidate Set")
            print("="*60)
            self.read_sgf(sgf_dir="../../../data_set/test_set/cand_set")
            self.inference(self.data_loader, is_query=False)

            # Clear again
            self.data_loader.Clear_Sgf()
            print("\nCleared candidate set from memory")
            
            # Compute similarity and generate submission
            print("\n" + "="*60)
            print("PHASE 3: Computing Similarity and Generating Submission")
            print("="*60)
            self.compute_similarity_and_save("../../../submission.csv")
            
            print("\n" + "="*60)
            print("TESTING COMPLETED!")
            print("="*60)


if __name__ == "__main__":
    # Parse arguments and run testing
    args = parse_args()
    
    # Initialize style_py (will be done in style_detection __init__)
    # Just create tester and run
    test = style_detection(args.conf_file, args.game_type)
    test.testing(args.model_path)

