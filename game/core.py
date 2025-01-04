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
    boards: List[np.ndarray] = field(default_factory=lambda: [
        np.zeros((10, 10), dtype=int),
        np.zeros((10, 10), dtype=int)
    ])
    hit_boards: List[np.ndarray] = field(default_factory=lambda: [
        np.zeros((10, 10), dtype=int),
        np.zeros((10, 10), dtype=int)
    ])
    ship_coords: List[Dict[str, List[Tuple[int, int]]]] = field(default_factory=lambda: [{}, {}])
    remaining_hits: List[int] = field(default_factory=lambda: [0, 0])
    current_player: int = 0
    done: bool = False

class BattleshipEnv:
    def __init__(self, board_size=10, ship_specs=None):
        """Initialize the Battleship environment.

        Args:
            board_size (int): Size of the board (board_size x board_size).
            ship_specs (dict): Dictionary specifying ship types and their sizes.
        """
        self.board_size = board_size
        self.ship_specs = ship_specs if ship_specs else {
            "carrier": 5,
            "battleship": 4,
            "cruiser": 3,
            "submarine": 3,
            "destroyer": 2
        }

    def reset(self) -> BoardState:
        """Resets the game environment and returns initial state."""
        state = BoardState()
        state = self._place_ships(state, 0)
        state = self._place_ships(state, 1)
        return state

    def _place_ships(self, state: BoardState, player: int) -> BoardState:
        """Randomly places ships on the board for a player.

        Args:
            state (BoardState): Current game state
            player (int): Player index (0 or 1)

        Returns:
            BoardState: Updated game state
        """
        for ship, size in self.ship_specs.items():
            placed = False
            while not placed:
                horizontal = np.random.choice([True, False])
                if horizontal:
                    row = np.random.randint(0, self.board_size)
                    col = np.random.randint(0, self.board_size - size + 1)
                    coords = [(row, col + i) for i in range(size)]
                else:
                    row = np.random.randint(0, self.board_size - size + 1)
                    col = np.random.randint(0, self.board_size)
                    coords = [(row + i, col) for i in range(size)]

                if all(state.boards[player][r, c] == 0 for r, c in coords):
                    for r, c in coords:
                        state.boards[player][r, c] = _SHIP_IDX
                    state.ship_coords[player][ship] = coords
                    state.remaining_hits[player] += size
                    placed = True

        return state

    def step(self, state: BoardState, action: int) -> Tuple[BoardState, float, bool]:
        """Takes an action and updates the game state.

        Args:
            state (BoardState): Current game state
            action (int): A flattened index representing the cell to attack

        Returns:
            Tuple[BoardState, float, bool]: (new state, reward, done)
        """
        if state.done:
            raise ValueError("Game is already over. Please reset the environment.")

        opponent = 1 - state.current_player
        row, col = divmod(action, self.board_size)

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
                return state, _WIN_REWARD if state.current_player == 0 else _LOSE_REWARD, state.done

            state.current_player = opponent
            return state, _HIT_REWARD, state.done

        # Miss
        else:
            state.hit_boards[opponent][row, col] = _MISS_IDX
            state.current_player = opponent
            return state, _MISS_REWARD, state.done

    def get_observation(self, state: BoardState) -> Dict:
        """Returns the current state of the boards.

        Args:
            state (BoardState): Current game state

        Returns:
            Dict: Observable game state
        """
        return {
            "current_player": state.current_player,
            "done": state.done,
            "player_board": [
                state.boards[state.current_player].copy(),
                state.hit_boards[state.current_player].copy()
            ],
            "opponent_board": [
                state.boards[1 - state.current_player].copy(),
                state.hit_boards[1 - state.current_player].copy()
            ]
        }