from torch import nn
import numpy as np
import torch
import torch.nn.functional as F
import sys


class Encoder(nn.Module):
    """
    CNN-based encoder for Go game style detection.
    
    Architecture:
    - Input: (batch, n_frames, channels, 19, 19)
    - Multiple convolutional layers to extract spatial features
    - Global average pooling
    - Fully connected layers to produce embedding
    - Output: (batch, 128) style embedding vector
    """

    def __init__(self, loss_device, conf_file, game_type):
        super().__init__()
        
        # Import and load C++ configuration
        # Add the build directory to Python path
        import os
        build_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..', 'build', game_type))
        if build_path not in sys.path:
            sys.path.insert(0, build_path)
        
        import style_py
        style_py.load_config_file(conf_file)

        self.loss_device = loss_device
        
        # Get dimensions from config
        self.n_frames = style_py.get_n_frames()
        self.input_channels = style_py.get_nn_num_input_channels()
        
        # Total input channels: n_frames * input_channels
        in_channels = self.n_frames * self.input_channels
        
        # CNN architecture
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 19x19 -> 9x9
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 9x9 -> 4x4
        )
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128)
        )
        
        # L2 normalization for embeddings (for cosine similarity)
        self.l2_norm = True

    def forward(self, inputs):
        """
        Forward pass
        
        Args:
            inputs: tensor of shape (batch, n_frames, channels, 19, 19)
        
        Returns:
            embeddings: tensor of shape (batch, 128)
        """
        # Reshape: (batch, n_frames, channels, H, W) -> (batch, n_frames*channels, H, W)
        batch_size = inputs.size(0)
        x = inputs.view(batch_size, -1, inputs.size(-2), inputs.size(-1))
        
        # CNN layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(batch_size, -1)
        
        # FC layers
        embeddings = self.fc(x)
        
        # L2 normalize embeddings
        if self.l2_norm:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def loss(self, anchor, positive, negative, margin=1.0):
        """
        Triplet loss for training style embeddings
        
        Args:
            anchor: embeddings of anchor samples (batch, 128)
            positive: embeddings of positive samples (batch, 128)
            negative: embeddings of negative samples (batch, 128)
            margin: margin for triplet loss
        
        Returns:
            loss: triplet loss value
        """
        # Calculate distances
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        
        # Triplet loss: max(d(a,p) - d(a,n) + margin, 0)
        losses = F.relu(pos_dist - neg_dist + margin)
        
        return losses.mean()

