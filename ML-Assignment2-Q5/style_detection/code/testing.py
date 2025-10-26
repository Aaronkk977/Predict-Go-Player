import argparse 
import glob 
from numpy import dot 
import numpy as np 
import torch 
from encoder.model import Encoder 
import sys
import time
import csv
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
        import os
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

        for i in range(600):
            sgf_path = sgf_dir + "/player" + str(i + 1) + ".sgf"
            self.data_loader.load_data_from_file(sgf_path)
            if (i + 1) % 100 == 0:
                print(f"  Loaded {i + 1}/600 players")

        print('Finished reading SGF files')
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
                    # Skip if game cannot be processed
                    continue
            
            if len(player_embeddings) > 0:
                # Average embeddings across all games for this player
                player_embedding = torch.stack(player_embeddings).mean(dim=0)
                embeddings_list.append(player_embedding)
            else:
                # If no valid games, use zero embedding
                embeddings_list.append(torch.zeros(1, 128))
            
            if (player_id + 1) % 50 == 0:
                print(f"  Processed {player_id + 1}/{num_players} players")
        
        # Stack all embeddings
        all_embeddings = torch.cat(embeddings_list, dim=0)  # (600, 128)
        
        if is_query:
            self.query_embeddings = all_embeddings
            print(f"Query embeddings shape: {self.query_embeddings.shape}")
        else:
            self.cand_embeddings = all_embeddings
            print(f"Candidate embeddings shape: {self.cand_embeddings.shape}")

    def compute_similarity_and_save(self, output_path="submission.csv"):
        """
        Compute cosine similarity between query and candidate embeddings
        and generate submission.csv
        """
        print("\nComputing similarities...")
        
        # Normalize embeddings for cosine similarity
        query_norm = F.normalize(self.query_embeddings, p=2, dim=1)
        cand_norm = F.normalize(self.cand_embeddings, p=2, dim=1)
        
        # Compute similarity matrix (600 x 600)
        similarity_matrix = torch.mm(query_norm, cand_norm.t())
        
        # For each query, find the most similar candidate
        predictions = []
        for query_id in range(600):
            similarities = similarity_matrix[query_id]
            most_similar_cand = torch.argmax(similarities).item()
            predictions.append((query_id + 1, most_similar_cand + 1))
        
        # Save to CSV
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['query', 'candidate'])  # header
            for query_id, cand_id in predictions:
                writer.writerow([f"player{query_id}", f"player{cand_id}"])
        
        print(f"Saved predictions to {output_path}")
        print(f"Total predictions: {len(predictions)}")

    def testing(self, model_path):
        """Complete testing procedure"""
        with torch.no_grad():
            # Load and process query set
            print("\n" + "="*60)
            print("PHASE 1: Processing Query Set")
            print("="*60)
            self.read_sgf(sgf_dir="../../data_set/test_set/query_set")
            self.load_model(model_path=model_path)
            self.inference(self.data_loader, is_query=True)

            # Clear SGF data
            self.data_loader.Clear_Sgf()
            print("\nCleared query set from memory")

            # Load and process candidate set
            print("\n" + "="*60)
            print("PHASE 2: Processing Candidate Set")
            print("="*60)
            self.read_sgf(sgf_dir="../../data_set/test_set/cand_set")
            self.inference(self.data_loader, is_query=False)

            # Clear again
            self.data_loader.Clear_Sgf()
            print("\nCleared candidate set from memory")
            
            # Compute similarity and generate submission
            print("\n" + "="*60)
            print("PHASE 3: Computing Similarity and Generating Submission")
            print("="*60)
            self.compute_similarity_and_save("submission.csv")
            
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

