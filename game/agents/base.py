"""Abstract base class for Battleship agents."""

from abc import ABC, abstractmethod
from typing import Dict
import numpy as np


class BattleshipAgent(ABC):
    """Abstract base class for Battleship agents."""

    def __init__(self, name: str):
        """
        Initialize agent.

        Args:
            name: Human-readable name for the agent
        """
        self.name = name

    @abstractmethod
    def select_action(self, observation: Dict[str, np.ndarray], env=None) -> int:
        """
        Select action given current observation.

        Args:
            observation: Dict with keys:
                - 'attack_board': (rows, cols) array
                    0 = unknown/not attacked
                    1 = miss
                    2 = hit
                - 'remaining_ships': (5,) array of unsunk ship sizes
                - 'move_count': (1,) array with current move count
            env: Optional environment for forced target mode masking (used by RL agents)

        Returns:
            action: Flattened board index (0 to rows*cols-1)

        Raises:
            ValueError: If no valid moves available
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset agent state for new episode."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
