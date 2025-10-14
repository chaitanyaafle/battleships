"""Random agent that selects unattacked cells uniformly."""

import numpy as np
from typing import Dict
from game.agents.base import BattleshipAgent


class RandomAgent(BattleshipAgent):
    """Random agent that selects unattacked cells uniformly."""

    def __init__(self, seed: int = None):
        """
        Initialize random agent.

        Args:
            seed: Random seed for reproducibility
        """
        super().__init__("Random")
        self.rng = np.random.default_rng(seed)

    def select_action(self, observation: Dict[str, np.ndarray]) -> int:
        """
        Select random unattacked cell.

        Args:
            observation: Current observation from environment

        Returns:
            Random valid action (flattened board index)

        Raises:
            ValueError: If no valid moves remaining
        """
        attack_board = observation['attack_board']
        rows, cols = attack_board.shape

        # Find all unattacked cells (value = 0)
        unattacked = []
        for i in range(rows):
            for j in range(cols):
                if attack_board[i, j] == 0:
                    unattacked.append(i * cols + j)

        if not unattacked:
            raise ValueError("No valid moves remaining")

        return int(self.rng.choice(unattacked))

    def reset(self):
        """No state to reset for random agent."""
        pass
