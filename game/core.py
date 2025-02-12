import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

_HIT_IDX = 2
_MISS_IDX = -1
_SHIP_IDX = 1
_EMPTY_IDX = 0
_HIT_REWARD = 1
_MISS_REWARD = -1
_WIN_REWARD = 10
_LOSE_REWARD = -10

@dataclass
class BoardState:
    sizes: List[Tuple[int, int]] = field(default_factory=lambda: [(10, 10), (10, 10)])  # Board sizes for both players
    boards: List[np.ndarray] = field(init=False)
    hit_boards: List[np.ndarray] = field(init=False)
    ship_coords: List[Dict[str, List[Tuple[int, int]]]] = field(default_factory=lambda: [{}, {}])
    remaining_hits: List[int] = field(default_factory=lambda: [0, 0])
    done: bool = False
    winner: int = -1  # -1 means no winner yet

    def __post_init__(self):
        self.boards = [
            np.zeros(self.sizes[0], dtype=int),
            np.zeros(self.sizes[1], dtype=int)
        ]
        self.hit_boards = [
            np.zeros(self.sizes[0], dtype=int),
            np.zeros(self.sizes[1], dtype=int)
        ]

class BattleshipEnv:
    def __init__(self, board_sizes=None, ship_specs=None):
        """Initialize the Battleship environment.

        Args:
            board_sizes (List[Tuple[int, int]]): Board sizes for both players [(rows1, cols1), (rows2, cols2)].
            ship_specs (dict): Dictionary specifying ship types and their sizes.
        """
        self.board_sizes = board_sizes if board_sizes else [(10, 10), (10, 10)]
        self.ship_specs = ship_specs if ship_specs else {
            "destroyer": 2
        }

    def reset(self) -> BoardState:
        """Resets the game environment and returns initial state."""
        state = BoardState(sizes=self.board_sizes)
        state = self._place_ships(state, 0)
        state = self._place_ships(state, 1)
        return state

    def _place_ships(self, state: BoardState, player: int) -> BoardState:
        """Randomly places ships on the board for a player."""
        for ship, size in self.ship_specs.items():
            placed = False
            while not placed:
                horizontal = np.random.choice([True, False])
                if horizontal:
                    row = np.random.randint(0, self.board_sizes[player][0])
                    col = np.random.randint(0, self.board_sizes[player][1] - size + 1)
                    coords = [(row, col + i) for i in range(size)]
                else:
                    row = np.random.randint(0, self.board_sizes[player][0] - size + 1)
                    col = np.random.randint(0, self.board_sizes[player][1])
                    coords = [(row + i, col) for i in range(size)]

                if all(state.boards[player][r, c] == 0 for r, c in coords):
                    for r, c in coords:
                        state.boards[player][r, c] = _SHIP_IDX
                    state.ship_coords[player][ship] = coords
                    state.remaining_hits[player] += size
                    placed = True

        return state

    def player_move(self, state: BoardState, player: int, action: int) -> Tuple[BoardState, float, bool]:
        """Takes an action for a specific player and updates the game state.

        Args:
            state (BoardState): Current game state
            player (int): Player making the move (0 or 1)
            action (int): A flattened index representing the cell to attack

        Returns:
            Tuple[BoardState, float, bool]: (new state, reward, done)
        """
        if state.done:
            raise ValueError("Game is already over. Please reset the environment.")

        opponent = 1 - player
        row, col = divmod(action, self.board_sizes[player][1])

        # Check if cell was already attacked
        if state.hit_boards[opponent][row, col] in [_MISS_IDX, _HIT_IDX]:
            return state, _MISS_REWARD, state.done

        # Hit a ship
        if state.boards[opponent][row, col] == _SHIP_IDX:
            state.boards[opponent][row, col] = _HIT_IDX
            state.hit_boards[opponent][row, col] = _HIT_IDX
            state.remaining_hits[opponent] -= 1

            # Check if any ship is sunk
            for ship, coords in list(state.ship_coords[opponent].items()):
                if all(state.boards[opponent][r, c] == _HIT_IDX for r, c in coords):
                    del state.ship_coords[opponent][ship]
                    break

            # Check if game is over
            if state.remaining_hits[opponent] == 0:
                state.done = True
                state.winner = player
                return state, _WIN_REWARD, state.done

            return state, _HIT_REWARD, state.done

        # Miss
        else:
            state.hit_boards[opponent][row, col] = _MISS_IDX
            return state, _MISS_REWARD, state.done

    def get_valid_moves(self, state: BoardState, player: int) -> List[int]:
        """Returns a list of valid moves for the given player.

        Args:
            state (BoardState): Current game state
            player (int): Player to check moves for (0 or 1)

        Returns:
            List[int]: List of valid move indices
        """
        opponent = 1 - player
        valid_moves = []
        
        for i in range(self.board_sizes[player][0]):
            for j in range(self.board_sizes[player][1]):
                if state.hit_boards[opponent][i, j] not in [_MISS_IDX, _HIT_IDX]:
                    valid_moves.append(i * self.board_sizes[player][1] + j)
                    
        return valid_moves

    def get_observation(self, state: BoardState, player: int) -> Dict:
        """Returns the observable state for a specific player.

        Args:
            state (BoardState): Current game state
            player (int): Player perspective (0 or 1)

        Returns:
            Dict: Observable game state
        """
        opponent = 1 - player
        return {
            "done": state.done,
            "winner": state.winner,
            "player_board": state.boards[player].copy(),
            "player_hits": state.hit_boards[player].copy(),
            "opponent_hits": state.hit_boards[opponent].copy(),
            "remaining_ships": {
                "player": len(state.ship_coords[player]),
                "opponent": len(state.ship_coords[opponent])
            }
        }