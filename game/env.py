"""Gymnasium-compatible Battleship environment."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any
from game.state import GameState, Ship
from game.config import get_ship_config, validate_board_size
from game.placement import place_ships


class BattleshipEnv(gym.Env):
    """
    Gymnasium-compatible Battleship environment.

    Single-arena design: Agent attacks hidden board to find all ships.
    """

    metadata = {"render_modes": ["human", "html", "ansi"]}

    def __init__(
        self,
        board_size: Tuple[int, int] = (10, 10),
        render_mode: Optional[str] = None,
        ship_placement: Optional[Dict[str, list]] = None,
        allow_adjacent_ships: bool = True,
        max_episode_length: Optional[int] = None
    ):
        """
        Initialize environment.

        Args:
            board_size: (rows, cols) tuple
            render_mode: One of ["human", "html", "ansi"]
            ship_placement: Optional manual ship placement (dict of ship_name -> coords)
            allow_adjacent_ships: If True, ships can touch (easier placement).
                                 If False, enforces no-touch constraint.
            max_episode_length: Maximum steps per episode. If None, defaults to 2 × total board cells.
        """
        super().__init__()

        # Validate board size
        validate_board_size(board_size)

        self.board_size = board_size
        self.render_mode = render_mode
        self.manual_placement = ship_placement
        self.allow_adjacent_ships = allow_adjacent_ships

        # Calculate max episode length
        if max_episode_length is None:
            self.max_episode_length = 2 * board_size[0] * board_size[1]
        else:
            self.max_episode_length = max_episode_length

        rows, cols = board_size

        # Action space: flattened board indices
        self.action_space = spaces.Discrete(rows * cols)

        # Observation space
        self.observation_space = spaces.Dict({
            'attack_board': spaces.Box(
                low=0, high=2,
                shape=(rows, cols),
                dtype=np.int32
            ),
            'remaining_ships': spaces.Box(
                low=0, high=5,
                shape=(5,),  # Max 5 ships
                dtype=np.int32
            ),
            'move_count': spaces.Box(
                low=0, high=np.iinfo(np.int32).max,
                shape=(1,),
                dtype=np.int32
            )
        })

        self.state: Optional[GameState] = None

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[Dict, Dict]:
        """
        Reset environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)

        Returns:
            observation: Initial observation dict
            info: Additional info dict
        """
        super().reset(seed=seed)

        # Create new game state
        self.state = self._create_initial_state()

        obs = self.state.get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            action: Flattened board index (0 to rows*cols-1)

        Returns:
            observation: Updated observation
            reward: Reward for this action
            terminated: Whether game ended (win)
            truncated: Whether episode truncated (not used)
            info: Additional information
        """
        if self.state is None:
            raise RuntimeError("Must call reset() before step()")

        if self.state.done:
            raise RuntimeError("Episode is done, call reset()")

        # Validate action
        if not self.action_space.contains(action):
            return (
                self.state.get_observation(),
                -50.0,  # Invalid move penalty
                False,
                False,
                {"error": "Action out of bounds", "result": "invalid"}
            )

        row, col = divmod(action, self.board_size[1])

        # Check if already attacked
        if self.state.attack_board[row, col] != 0:
            return (
                self.state.get_observation(),
                -50.0,  # Invalid move penalty
                False,
                False,
                {"error": f"Cell ({row}, {col}) already attacked", "result": "invalid"}
            )

        # Process attack
        reward, info = self._process_attack(row, col)

        # Update move count
        self.state.move_count += 1

        # Check win condition
        terminated = self.state.done

        # Check truncation (episode length limit)
        truncated = self.state.move_count >= self.max_episode_length

        return (
            self.state.get_observation(),
            reward,
            terminated,
            truncated,
            info
        )

    def _process_attack(self, row: int, col: int) -> Tuple[float, Dict]:
        """
        Process an attack at given coordinates.

        Args:
            row: Row index
            col: Column index

        Returns:
            reward: Reward for this attack
            info: Information dict with result and ship_sunk
        """
        cell_value = self.state.ship_board[row, col]

        if cell_value == 0:  # Miss
            self.state.attack_board[row, col] = 1
            return -1.0, {"result": "miss", "ship_sunk": None}

        else:  # Hit
            self.state.attack_board[row, col] = 2

            # Find which ship was hit
            ship_sunk = None
            for ship in self.state.ships.values():
                if (row, col) in ship.coords:
                    just_sunk = ship.register_hit((row, col))
                    if just_sunk:
                        ship_sunk = ship.name
                    break

            # Check if all ships sunk (win)
            all_sunk = all(s.is_sunk for s in self.state.ships.values())
            if all_sunk:
                self.state.done = True
                return 100.0, {"result": "win", "ship_sunk": ship_sunk}

            # Ship sunk but game continues
            if ship_sunk:
                return 10.0, {"result": "hit", "ship_sunk": ship_sunk}

            # Regular hit
            return 5.0, {"result": "hit", "ship_sunk": None}

    def render(self):
        """Render the environment based on render_mode."""
        if self.state is None:
            return None

        if self.render_mode == "human":
            from game.renderers.html import render_html
            return render_html(self.state)
        elif self.render_mode == "ansi":
            from game.renderers.console import render_console
            return render_console(self.state)
        elif self.render_mode == "html":
            from game.renderers.html import render_html
            return render_html(self.state)

    def _create_initial_state(self) -> GameState:
        """Create initial game state with ships placed."""
        ship_config = get_ship_config(self.board_size)

        if self.manual_placement:
            # Use manual placement (with validation)
            from game.placement import validate_manual_placement
            validate_manual_placement(self.manual_placement, self.board_size)

            # Convert manual placement to Ship objects
            ships = {}
            ship_board = np.zeros(self.board_size, dtype=np.int32)

            for ship_id, (ship_name, coords) in enumerate(self.manual_placement.items(), start=1):
                ships[ship_name] = Ship(
                    name=ship_name,
                    size=len(coords),
                    coords=coords
                )
                for r, c in coords:
                    ship_board[r, c] = ship_id
        else:
            # Random placement
            ships, ship_board = place_ships(
                self.board_size,
                ship_config,
                self.np_random,  # Use seeded RNG
                allow_adjacent=self.allow_adjacent_ships
            )

        rows, cols = self.board_size
        attack_board = np.zeros((rows, cols), dtype=np.int32)

        total_cells = sum(s.size for s in ships.values())

        return GameState(
            board_size=self.board_size,
            ship_board=ship_board,
            attack_board=attack_board,
            ships=ships,
            total_ship_cells=total_cells
        )

    def _get_info(self) -> Dict[str, Any]:
        """Get additional info for observation."""
        return {
            "ships_remaining": len([s for s in self.state.ships.values() if not s.is_sunk]),
            "total_ships": len(self.state.ships),
            "move_count": self.state.move_count
        }

    def close(self):
        """Clean up resources."""
        pass
