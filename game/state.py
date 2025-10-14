"""Game state data structures for single-arena Battleship environment."""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set
import numpy as np


@dataclass
class Ship:
    """Represents a single ship in the game."""

    name: str
    size: int
    coords: List[Tuple[int, int]]
    hits: Set[Tuple[int, int]] = field(default_factory=set)

    @property
    def is_sunk(self) -> bool:
        """Check if all ship cells have been hit."""
        return len(self.hits) == self.size

    def register_hit(self, coord: Tuple[int, int]) -> bool:
        """
        Register a hit on this ship.

        Args:
            coord: (row, col) coordinate of the hit

        Returns:
            True if ship just sunk (this hit completed it), False otherwise
        """
        if coord in self.coords:
            was_sunk = self.is_sunk
            self.hits.add(coord)
            return not was_sunk and self.is_sunk
        return False

    def __repr__(self) -> str:
        return f"Ship(name={self.name}, size={self.size}, hits={len(self.hits)}/{self.size})"


@dataclass
class GameState:
    """Single-arena game state for agent play."""

    board_size: Tuple[int, int]      # (rows, cols)
    ship_board: np.ndarray           # Hidden board: 0=water, ship_id=ship
    attack_board: np.ndarray         # Visible: 0=unknown, 1=miss, 2=hit
    ships: Dict[str, Ship]           # ship_name -> Ship object
    move_count: int = 0
    done: bool = False
    total_ship_cells: int = 0        # For win detection

    def get_observation(self) -> Dict[str, np.ndarray]:
        """
        Return observation dict for Gymnasium.

        Returns:
            Dict with keys:
                - 'attack_board': (rows, cols) array showing attack history
                - 'remaining_ships': (5,) array of unsunk ship sizes (padded with 0s)
                - 'move_count': (1,) array with current move count
        """
        remaining = [s.size for s in self.ships.values() if not s.is_sunk]
        # Sort for consistency and pad to 5 ships (max fleet size)
        remaining = sorted(remaining, reverse=True)
        remaining += [0] * (5 - len(remaining))

        return {
            'attack_board': self.attack_board.copy(),
            'remaining_ships': np.array(remaining[:5], dtype=np.int32),
            'move_count': np.array([self.move_count], dtype=np.int32)
        }

    def __repr__(self) -> str:
        ships_remaining = len([s for s in self.ships.values() if not s.is_sunk])
        return (f"GameState(board_size={self.board_size}, "
                f"ships={ships_remaining}/{len(self.ships)}, "
                f"moves={self.move_count}, done={self.done})")
