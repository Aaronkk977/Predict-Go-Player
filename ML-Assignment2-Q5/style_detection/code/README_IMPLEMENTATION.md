# Go Player Style Detection - C++ + Python Implementation

This implementation uses MiniZero C++ backend for SGF parsing and feature extraction, combined with PyTorch for neural network training.

## Architecture

### Components
1. **C++ Backend** (`style_detection/`)
   - `sd_data_loader.cpp/h`: SGF parsing and feature extraction
   - `sd_mode_handler.cpp/h`: Mode handling and environment loading
   - `pybind.cpp`: Python bindings

2. **Python Model** (`code/encoder/`)
   - `model.py`: CNN encoder with triplet loss
   - `train.py`: Training loop with triplet learning

3. **Testing** (`code/`)
   - `testing.py`: Inference and submission generation

## Model Architecture

### Encoder (model.py)
- **Input**: (batch, n_frames, channels, 19, 19)
- **CNN Layers**:
  - Conv1: in_channels → 64, 3x3
  - Conv2: 64 → 128, 3x3, MaxPool (19x19 → 9x9)
  - Conv3: 128 → 256, 3x3
  - Conv4: 256 → 512, 3x3, MaxPool (9x9 → 4x4)
- **Global Average Pooling**: 512 → 512
- **FC Layers**: 512 → 256 → 128
- **Output**: (batch, 128) L2-normalized embeddings

### Training Strategy
- **Loss Function**: Triplet Loss
  - Anchor: Player A's game 1
  - Positive: Player A's game 2-9
  - Negative: Different player's random game
  - Margin: 1.0
- **Optimizer**: Adam (lr=0.001, weight_decay=1e-4)
- **Batch Size**: 20 players × 9 games per player

## Setup

### 1. Build C++ Backend

```bash
cd ML-Assignment2-Q5

# Start Docker container (with GPU support)
./scripts/start-container.sh

# Inside container:
cd /workspace/ML-Assignment2-Q5
./scripts/build.sh
```

This will:
- Compile MiniZero core
- Compile style_detection with pybind11
- Generate `build/go/style_py.cpython-*.so`

### 2. Verify Build

```bash
cd style_detection/code
python3 -c "import sys; sys.path.insert(0, '..'); from build.go import style_py; print('✓ Import successful')"
```

## Training

### Quick Start

```bash
cd ML-Assignment2-Q5/style_detection/code

# Run training
./run_training.sh
```

### Manual Training

```bash
python3 train.py go my_model_run \
    -c ../conf.cfg \
    -m ./models \
    -s 100 \
    -b 500 \
    -ve 500
```

Parameters:
- `go`: Game type
- `my_model_run`: Run ID for this training session
- `-c`: Configuration file path
- `-m`: Models directory for checkpoints
- `-s`: Save checkpoint every N steps
- `-b`: Backup checkpoint every N steps
- `-ve`: Validate every N steps

### Training Configuration (conf.cfg)

Key parameters:
```ini
learner_training_step=500      # Total training steps
players_per_batch=20           # Players per batch
games_per_player=9             # Games per player in batch
n_frames=10                    # History frames
move_step_to_choose=4          # Starting move position
```

### Expected Output

```
Starting training for 500 steps
Board size: 19x19
Players per batch: 20
Games per player: 9
Step 10/500: Loss = 0.8523
Step 20/500: Loss = 0.7234
...
Step 500/500: Loss = 0.2145
Saved checkpoint to ./models/my_model_run_step_500.pt
Training completed! Final model saved to ./models/my_model_run_final.pt
```

## Testing

### Quick Start

```bash
cd ML-Assignment2-Q5/style_detection/code

# Run testing with trained model
./run_testing.sh ./models/my_model_run_final.pt
```

### Manual Testing

```bash
python3 testing.py go \
    -c ../conf.cfg \
    -m ./models/my_model_run_final.pt
```

### Testing Process

1. **Load Query Set** (600 players)
   - Read 600 × 100 SGF files
   - Extract features for each game
   - Compute embeddings
   - Average embeddings per player

2. **Load Candidate Set** (600 players)
   - Same process as query set

3. **Compute Similarity**
   - Calculate cosine similarity matrix (600 × 600)
   - For each query, find most similar candidate
   - Generate `submission.csv`

### Expected Output

```
============================================================
PHASE 1: Processing Query Set
============================================================
Start reading SGF files from: ../../data_set/test_set/query_set
  Loaded 100/600 players
  Loaded 200/600 players
  ...
Finished reading SGF files
Loading model from: ./models/my_model_run_final.pt
Using device: cuda
Model loaded successfully
Running inference on 600 players...
  Processed 50/600 players
  ...
Query embeddings shape: torch.Size([600, 128])

============================================================
PHASE 2: Processing Candidate Set
============================================================
[Similar process for candidate set]

============================================================
PHASE 3: Computing Similarity and Generating Submission
============================================================
Computing similarities...
Saved predictions to submission.csv
Total predictions: 600

============================================================
TESTING COMPLETED!
============================================================
```

## File Structure

```
ML-Assignment2-Q5/
├── conf.cfg                    # Configuration file
├── style_detection/
│   ├── sd_data_loader.cpp/h    # C++ data loader
│   ├── sd_mode_handler.cpp/h   # C++ mode handler
│   ├── pybind.cpp              # Python bindings
│   └── code/
│       ├── encoder/
│       │   ├── model.py        # CNN encoder model
│       │   └── train.py        # Training loop
│       ├── testing.py          # Testing script
│       ├── train.py            # Training entry point
│       ├── run_training.sh     # Training launcher
│       └── run_testing.sh      # Testing launcher
└── build/
    └── go/
        └── style_py.*.so       # Compiled Python module

data_set/
├── train_set/
│   ├── 1.sgf                   # 200 training files
│   ├── 2.sgf
│   └── ...
└── test_set/
    ├── query_set/
    │   ├── player1.sgf         # 600 query players
    │   └── ...
    └── cand_set/
        ├── player1.sgf         # 600 candidate players
        └── ...
```

## Troubleshooting

### Import Error: Cannot find style_py

```bash
# Make sure you're in the code/ directory
cd ML-Assignment2-Q5/style_detection/code

# Check if .so exists
ls ../build/go/*.so

# Try importing manually
python3 -c "import sys; sys.path.insert(0, '..'); from build.go import style_py"
```

### SGF Files Not Found

```bash
# Check paths in training script
cd ML-Assignment2-Q5/style_detection/code
python3 -c "import glob; print(glob.glob('../../data_set/train_set/*.sgf')[:5])"

# Should show: ['../../data_set/train_set/1.sgf', ...]
```

### CUDA Out of Memory

Reduce batch size in `conf.cfg`:
```ini
players_per_batch=10  # Reduce from 20
```

Or reduce worker count in `encoder/train.py`:
```python
data_loader = DataLoader(dataset, batch_size=..., num_workers=4)  # Reduce from 8
```

## Hyperparameter Tuning

### Model Architecture
- Adjust CNN channels in `model.py`
- Change embedding dimension (currently 128)
- Modify dropout rate (currently 0.5)

### Training Parameters
- Learning rate: Change in `encoder/train.py` (currently 0.001)
- Triplet margin: Modify in `model.loss()` call (currently 1.0)
- Weight decay: Currently 1e-4

### Data Parameters (conf.cfg)
- `n_frames`: More frames = more context (currently 10)
- `games_per_player`: More games = better player representation (currently 9)
- `move_step_to_choose`: Starting position in game (currently 4)

## Performance Tips

### Training Speed
- Use multi-GPU if available (automatic with DataParallel)
- Increase `num_workers` in DataLoader (default: 8)
- Use CUDA if available (automatic detection)

### Accuracy Improvements
- Train for more steps (increase `learner_training_step`)
- Use more games per player (increase `games_per_player`)
- Add data augmentation (rotation already enabled in C++)
- Tune triplet margin parameter

## Next Steps

1. **Train the model**:
   ```bash
   ./run_training.sh
   ```

2. **Test and generate submission**:
   ```bash
   ./run_testing.sh ./models/your_model_final.pt
   ```

3. **Submit results**:
   - Check `submission.csv`
   - Format: `query,candidate` with 600 rows
   - Example: `player1,player234`

4. **Iterate and improve**:
   - Adjust hyperparameters
   - Try different architectures
   - Analyze results and refine

## References

- MiniZero Framework: AlphaZero-based Go AI
- Triplet Loss: Learning embeddings for similarity
- PyTorch: Deep learning framework
- SGF Format: Standard Go game notation
