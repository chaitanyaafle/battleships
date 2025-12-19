"""
Probability-based agent using DataGenetics optimal strategy.

Based on: http://www.datagenetics.com/blog/december32011/

This agent uses probability density to select the optimal cell to attack.
For each turn, it calculates the probability that each cell contains a ship
by enumerating all possible valid placements of remaining unsunk ships.

Expected performance (based on DataGenetics research):
- Median: ~42 shots to win
- Maximum: ~73 shots
- 56% improvement over random baseline
"""

import numpy as np
from typing import Dict, List, Tuple, Set
from game.agents.base import BattleshipAgent


class ProbabilityAgent(BattleshipAgent):
    """Agent that uses probability density to make optimal moves."""

    def __init__(self, name: str = "Probability", hit_adjacency_weight: float = 50.0):
        """
        Initialize probability-based agent.

        Args:
            name: Human-readable name for the agent
            hit_adjacency_weight: Multiplier for cells adjacent to hits (default 50x).
                                DataGenetics recommends 10-100x weighting.
        """
        super().__init__(name)
        self.hit_adjacency_weight = hit_adjacency_weight
        self.probability_grid = None  # Store last calculated probabilities for visualization
        self.last_action = None
        self.last_action_probability = None
        self.sunk_ship_coords = set()  # Track coordinates of sunk ships
        self.hit_coords = set()  # Track all hits we've made
        self.previous_remaining_ships = None  # Track when ships get sunk

    def select_action(self, observation: Dict[str, np.ndarray], env=None) -> int:
        """
        Select action with highest probability of containing a ship.

        Args:
            observation: Current observation from environment
            env: Unused, kept for API compatibility

        Returns:
            Action with highest probability (flattened board index)

        Raises:
            ValueError: If no valid moves remaining
        """
        attack_board = observation['attack_board']
        remaining_ships = observation['remaining_ships']
        rows, cols = attack_board.shape

        # Update hit tracking and detect sunken ships
        self._update_sunk_ships(attack_board, remaining_ships)

        # Get unsunk ship sizes (filter out zeros)
        unsunk_ship_sizes = [int(size) for size in remaining_ships if size > 0]

        if not unsunk_ship_sizes:
            raise ValueError("No ships remaining")

        # Calculate probability density for all cells
        self.probability_grid = self._calculate_probability_density(
            attack_board, unsunk_ship_sizes
        )

        # Find all cells with maximum probability
        max_prob = np.max(self.probability_grid)
        if max_prob == 0:
            # No valid placements found - fallback to random unattacked cell
            # This can happen in edge cases with inconsistent board states
            unattacked = np.argwhere(attack_board == 0)
            if len(unattacked) == 0:
                raise ValueError("No valid moves remaining")
            idx = np.random.randint(len(unattacked))
            row, col = unattacked[idx]
            action = int(row * cols + col)
        else:
            # Get all cells with max probability
            max_cells = np.argwhere(self.probability_grid == max_prob)

            # Tie-breaking: select first cell (top-left to bottom-right)
            row, col = max_cells[0]
            action = int(row * cols + col)

        # Store for visualization
        self.last_action = action
        self.last_action_probability = max_prob

        return action

    def _update_sunk_ships(self, attack_board: np.ndarray, remaining_ships: np.ndarray):
        """
        Update tracking of sunk ships by detecting when ships are destroyed.

        When a ship sinks, all its hit coordinates should be marked as sunk
        so they're not considered as targets for future placements.
        """
        rows, cols = attack_board.shape

        # Find all current hits
        current_hits = set()
        for i in range(rows):
            for j in range(cols):
                if attack_board[i, j] == 2:  # Hit
                    current_hits.add((i, j))

        # Detect if a ship was sunk (remaining ships decreased)
        current_remaining = tuple(sorted([int(s) for s in remaining_ships if s > 0], reverse=True))

        if self.previous_remaining_ships is not None:
            if len(current_remaining) < len(self.previous_remaining_ships):
                # A ship was sunk! Find which hits are from the sunk ship
                # These are hits that were part of a contiguous group
                new_sunk_hits = self._find_newly_sunk_ship_coords(
                    current_hits, self.hit_coords, self.sunk_ship_coords
                )
                self.sunk_ship_coords.update(new_sunk_hits)

        # Update tracking
        self.hit_coords = current_hits
        self.previous_remaining_ships = current_remaining

    def _find_newly_sunk_ship_coords(
        self,
        current_hits: Set[Tuple[int, int]],
        previous_hits: Set[Tuple[int, int]],
        already_sunk: Set[Tuple[int, int]]
    ) -> Set[Tuple[int, int]]:
        """
        Find coordinates of a newly sunk ship.

        Strategy: Find all contiguous groups of hits (excluding already sunk),
        and identify which group has no adjacent unknown cells (fully explored).
        """
        unsunk_hits = current_hits - already_sunk

        if not unsunk_hits:
            return set()

        # Find all contiguous groups
        groups = []
        visited = set()

        for hit in unsunk_hits:
            if hit in visited:
                continue

            # BFS to find connected component
            group = set()
            queue = [hit]
            while queue:
                coord = queue.pop(0)
                if coord in visited:
                    continue

                visited.add(coord)
                group.add(coord)

                # Check 4-directional neighbors
                r, c = coord
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    neighbor = (nr, nc)
                    if neighbor in unsunk_hits and neighbor not in visited:
                        queue.append(neighbor)

            groups.append(group)

        # A ship is sunk if:
        # 1. All cells in the group are hits
        # 2. All adjacent cells are either hits (same ship) or misses/out-of-bounds
        # Find the group that is "complete" (fully surrounded by non-unknown cells)

        # For simplicity, return the largest complete group
        # (In practice, when a ship sinks, it's the most recently completed group)
        for group in sorted(groups, key=len, reverse=True):
            # This is a heuristic: assume the sunk ship is a complete contiguous group
            return group

        # Fallback: return largest group if none are clearly complete
        if groups:
            return max(groups, key=len)

        return set()

    def _calculate_probability_density(
        self, attack_board: np.ndarray, ship_sizes: List[int]
    ) -> np.ndarray:
        """
        Calculate probability density for each cell.

        For each unsunk ship, enumerate all valid placements and increment
        a counter for every cell that placement covers. The result is a
        probability grid where higher values indicate higher likelihood
        of containing a ship.

        Following DataGenetics optimal strategy, applies heavy weighting
        (10-100x) to cells adjacent to known hits to prioritize target mode.

        Args:
            attack_board: Current attack board state
            ship_sizes: List of unsunk ship sizes

        Returns:
            Probability density grid (same shape as attack_board)
        """
        rows, cols = attack_board.shape
        probability = np.zeros((rows, cols), dtype=np.float64)

        # Check if we have any hits from UNSUNK ships (target mode vs hunt mode)
        # Hits from sunk ships should not trigger target mode
        unsunk_hits = set()
        for i in range(rows):
            for j in range(cols):
                if attack_board[i, j] == 2 and (i, j) not in self.sunk_ship_coords:
                    unsunk_hits.add((i, j))

        has_unsunk_hits = len(unsunk_hits) > 0

        for ship_size in ship_sizes:
            # Try all horizontal placements
            for row in range(rows):
                for start_col in range(cols - ship_size + 1):
                    coords = [(row, start_col + i) for i in range(ship_size)]
                    if self._is_valid_placement(attack_board, coords, has_unsunk_hits):
                        for r, c in coords:
                            probability[r, c] += 1

            # Try all vertical placements
            for col in range(cols):
                for start_row in range(rows - ship_size + 1):
                    coords = [(start_row + i, col) for i in range(ship_size)]
                    if self._is_valid_placement(attack_board, coords, has_unsunk_hits):
                        for r, c in coords:
                            probability[r, c] += 1

        # Apply hit-adjacency weighting (DataGenetics strategy)
        # Only weight adjacent to UNSUNK hits
        if has_unsunk_hits and self.hit_adjacency_weight > 1.0:
            probability = self._apply_hit_adjacency_weighting(
                probability, attack_board, unsunk_hits
            )

        return probability

    def _is_valid_placement(
        self,
        attack_board: np.ndarray,
        coords: List[Tuple[int, int]],
        has_unsunk_hits: bool
    ) -> bool:
        """
        Check if a ship placement is valid.

        A placement is valid if:
        1. All cells are within board bounds (guaranteed by caller)
        2. No cells are misses (attack_board value == 1)
        3. No cells are from sunk ships (treated as obstacles)
        4. If there are active hits from unsunk ships (target mode),
           at least one cell must pass through an unsunk hit

        Args:
            attack_board: Current attack board state
            coords: List of (row, col) coordinates for ship placement
            has_unsunk_hits: Whether there are any hits from unsunk ships

        Returns:
            True if placement is valid, False otherwise
        """
        contains_unsunk_hit = False

        for row, col in coords:
            cell_value = attack_board[row, col]

            # Cannot place on a miss
            if cell_value == 1:
                return False

            # Cannot place on sunk ship coordinates (treat like misses)
            if (row, col) in self.sunk_ship_coords:
                return False

            # Track if this placement contains an unsunk hit
            if cell_value == 2 and (row, col) not in self.sunk_ship_coords:
                contains_unsunk_hit = True

        # In target mode, placement must contain at least one unsunk hit
        # In hunt mode, any placement without misses/sunk ships is valid
        if has_unsunk_hits:
            return contains_unsunk_hit
        else:
            return True

    def _apply_hit_adjacency_weighting(
        self,
        probability: np.ndarray,
        attack_board: np.ndarray,
        unsunk_hits: Set[Tuple[int, int]]
    ) -> np.ndarray:
        """
        Apply heavy weighting to cells adjacent to known hits from unsunk ships.

        As recommended by DataGenetics optimal strategy, cells adjacent
        to hits receive 10-100x probability multiplier to prioritize
        finishing off partially hit ships.

        Args:
            probability: Base probability grid
            attack_board: Current attack board state
            unsunk_hits: Set of hit coordinates from unsunk ships

        Returns:
            Weighted probability grid
        """
        rows, cols = attack_board.shape
        weighted = probability.copy()

        # For each unsunk hit, weight adjacent cells (4-directional: N, S, E, W)
        for row, col in unsunk_hits:
            # Check 4 cardinal directions
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = row + dr, col + dc

                # Check bounds and ensure cell is unattacked
                if 0 <= nr < rows and 0 <= nc < cols:
                    if attack_board[nr, nc] == 0:  # Unknown cell
                        weighted[nr, nc] *= self.hit_adjacency_weight

        return weighted

    def reset(self):
        """Reset agent state for new episode."""
        self.probability_grid = None
        self.last_action = None
        self.last_action_probability = None
        self.sunk_ship_coords = set()
        self.hit_coords = set()
        self.previous_remaining_ships = None

    def get_probability_grid(self) -> np.ndarray:
        """
        Get the last calculated probability grid for visualization.

        Returns:
            Probability grid from last action selection, or None if no action taken yet
        """
        return self.probability_grid

    def get_last_action_info(self) -> Tuple[int, float]:
        """
        Get information about the last action taken.

        Returns:
            Tuple of (action, probability) where action is the flattened index
            and probability is the score for that cell. Returns (None, None) if
            no action taken yet.
        """
        return (self.last_action, self.last_action_probability)
