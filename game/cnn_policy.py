"""
Custom CNN-based feature extractor for Battleship.

Provides spatial inductive bias for learning patterns like parity and adjacency.
Compatible with Stable-Baselines3's MultiInputPolicy.
"""

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class BattleshipCNN(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for Battleship environment.

    Processes the attack_board with convolutional layers to capture spatial patterns,
    then combines with auxiliary inputs (remaining_ships, move_count).

    Architecture:
    - Conv2d layers for spatial pattern recognition
    - Adaptive pooling to handle variable board sizes
    - Fully connected layer to combine features
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128):
        """
        Initialize CNN feature extractor.

        Args:
            observation_space: Dictionary observation space with:
                - attack_board: Box(H, W) - board state
                - remaining_ships: Box(5,) - ship counts
                - move_count: Box(1,) - current move number
            features_dim: Dimensionality of output features
        """
        super().__init__(observation_space, features_dim)

        # Extract observation space shapes
        attack_board_shape = observation_space['attack_board'].shape  # (H, W)
        remaining_ships_dim = observation_space['remaining_ships'].shape[0]  # 5
        move_count_dim = observation_space['move_count'].shape[0]  # 1

        # Convolutional layers for attack_board
        # Input: 1 channel (attack_board values: 0=unknown, 1=miss, 2=hit)
        # Layer 1: Detect local patterns (3x3 kernel sees adjacent cells)
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            padding=1  # Keep spatial dimensions
        )

        # Layer 2: Detect larger patterns (checkerboard, lines)
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        # Global average pooling: reduces spatial dims to 1x1
        # Works for any board size and is MPS-compatible
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Calculate flattened CNN output size
        cnn_output_size = 64  # After global pooling: 64 x 1 x 1 → 64

        # Combine CNN features with auxiliary inputs
        combined_size = cnn_output_size + remaining_ships_dim + move_count_dim

        # Final FC layer to produce feature vector
        self.fc = nn.Linear(combined_size, features_dim)

        # Activation
        self.relu = nn.ReLU()

    def forward(self, observations: dict) -> torch.Tensor:
        """
        Forward pass through CNN feature extractor.

        Args:
            observations: Dictionary with keys:
                - attack_board: (batch, H, W)
                - remaining_ships: (batch, 5)
                - move_count: (batch, 1)

        Returns:
            Feature tensor of shape (batch, features_dim)
        """
        # Extract components
        attack_board = observations['attack_board']  # (batch, H, W)
        remaining_ships = observations['remaining_ships']  # (batch, 5)
        move_count = observations['move_count']  # (batch, 1)

        # Add channel dimension to attack_board: (batch, H, W) → (batch, 1, H, W)
        x = attack_board.unsqueeze(1)

        # Convolutional layers with ReLU activation
        x = self.relu(self.conv1(x))  # (batch, 32, H, W)
        x = self.relu(self.conv2(x))  # (batch, 64, H, W)

        # Global average pooling: (batch, 64, H, W) → (batch, 64, 1, 1)
        x = self.global_pool(x)

        # Flatten CNN output: (batch, 64, 1, 1) → (batch, 64)
        x = torch.flatten(x, start_dim=1)

        # Concatenate with auxiliary inputs
        combined = torch.cat([x, remaining_ships, move_count], dim=1)

        # Final FC layer
        features = self.relu(self.fc(combined))

        return features
