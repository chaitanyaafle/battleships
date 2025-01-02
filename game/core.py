import numpy as np

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
        self.reset()

    def reset(self):
        """Resets the game environment."""
        self.boards = [np.zeros((self.board_size, self.board_size), dtype=int) for _ in range(2)]  # Each player has a board
        self.ship_coords = [{}, {}]  # Ship coordinates for each player
        self.remaining_hits = [0, 0]  # Remaining hits for each player
        self.current_player = 0  # Player 1 starts
        self.done = False
        for i in range(2):
            self._place_ships(i)
        return self._get_observation()

    def _place_ships(self, player):
        """Randomly places ships on the board for a player."""
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

                if all(self.boards[player][r, c] == 0 for r, c in coords):
                    for r, c in coords:
                        self.boards[player][r, c] = 1
                    self.ship_coords[player][ship] = coords
                    self.remaining_hits[player] += size
                    placed = True

    def step(self, action):
        """Takes an action and updates the game state.

        Args:
            action (int): A flattened index representing the cell to attack.

        Returns:
            observation: The current state of the boards after the action.
            reward: Reward for the action (-1 for miss, 1 for hit, 10/-10 for winning/losing).
            done: Whether the game is over.
        """
        if self.done:
            raise ValueError("Game is already over. Please reset the environment.")

        opponent = 1 - self.current_player
        row, col = divmod(action, self.board_size)

        if self.boards[opponent][row, col] in [-1, 2]:  # Already attacked
            return self._get_observation(), -1, self.done

        if self.boards[opponent][row, col] == 1:  # Hit a ship
            self.boards[opponent][row, col] = 2  # Mark as hit
            self.remaining_hits[opponent] -= 1

            # Check if any ship is sunk
            for ship, coords in self.ship_coords[opponent].items():
                if all(self.boards[opponent][r, c] == 2 for r, c in coords):
                    del self.ship_coords[opponent][ship]
                    break

            if self.remaining_hits[opponent] == 0:
                self.done = True
                return self._get_observation(), 10 if self.current_player == 0 else -10, self.done  # Game over
            return self._get_observation(), 1, self.done  # Hit
        else:
            self.boards[opponent][row, col] = -1  # Mark as miss
            self.current_player = opponent  # Switch player
            return self._get_observation(), -1, self.done  # Miss

    def _get_observation(self):
        """Returns the current state of the boards."""
        return {
            "player_board": self.boards[self.current_player].copy(),
            "opponent_board": self.boards[1 - self.current_player].copy()  # Hide opponent ships
        }


