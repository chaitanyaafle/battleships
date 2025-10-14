# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Battleship game simulation repository being refactored to support AI agent comparison.

**CURRENT STATE**: Two-player turn-based game with pygame visualization
**TARGET STATE**: Single-arena Gymnasium environment for training/comparing AI agents

### Planned Features (Post-Refactor)
- Gymnasium-compatible environment for RL training
- Single-arena design (agent attacks hidden board)
- Abstract agent interface supporting multiple AI types (human, heuristic, RL, LLM, hybrid)
- HTML and console rendering
- Configurable board sizes (5x5 to 12x12) with adaptive ship configurations
- Comprehensive validation and error handling

## Running the Game

### Run the main pygame visualization
```bash
cd game
python main.py
```

### Run in Jupyter notebook
```bash
jupyter notebook battleships.ipynb
```

## Architecture

### Core Game Logic (`game/core.py`)

The `BattleshipEnv` class is the main game engine:
- **Initialization**: `BattleshipEnv(board_sizes, ship_specs)`
  - `board_sizes`: List of tuples `[(rows1, cols1), (rows2, cols2)]` - supports different board sizes per player
  - `ship_specs`: Dictionary of ship names to sizes, e.g., `{"destroyer": 2, "cruiser": 3}`
- **Game state**: Managed via `BoardState` dataclass containing:
  - `boards`: Actual ship placements for both players
  - `hit_boards`: Attack history (hits/misses) for both players
  - `ship_coords`: Coordinates of each ship for both players
  - `remaining_hits`: Count of ship cells not yet hit
- **Key methods**:
  - `reset()`: Initialize new game with random ship placement
  - `player_move(state, player, action)`: Execute move for player (0 or 1), returns `(new_state, reward, done)`
  - `get_valid_moves(state, player)`: Returns list of valid flattened board indices
  - `get_observation(state, player)`: Returns player's view of the game (their board, both hit boards)

**Important**: Actions are flattened board indices: `action = row * board_width + col`

### Board State Encoding

Cell values in boards/hit_boards:
- `_EMPTY_IDX = 0`: Empty water
- `_SHIP_IDX = 1`: Ship present (only visible in own board)
- `_HIT_IDX = 2`: Successful hit
- `_MISS_IDX = -1`: Missed attack

Rewards:
- `_HIT_REWARD = 1`: Hit a ship
- `_MISS_REWARD = -1`: Missed
- `_WIN_REWARD = 10`: Sunk all opponent ships
- `_LOSE_REWARD = -10`: All own ships sunk

### Visualization

Three visualization methods available:

1. **ASCII** (`game/visualization.py`): Text-based using emojis
   - `visualize_state_ascii(state, player, board_size)`

2. **Matplotlib** (`game/visualization.py`): Static matplotlib plots
   - `visualize_state_matplotlib(state, player, board_size)`

3. **Pygame** (`game/visualization_pygame.py`): Real-time interactive display
   - `BattleshipVisualizer` class handles rendering both boards side-by-side
   - Supports different board sizes per player
   - Used in `game/main.py` for live gameplay

### MCTS Implementation (`mcts/mcts.py`)

**WARNING**: The MCTS implementation has known issues and may not work correctly:
- References `state.current_player` which doesn't exist in `BoardState`
- Hardcoded for 10x10 boards only
- Uses `env.step()` method that doesn't exist (should be `player_move()`)
- Not compatible with asymmetric board sizes

If working on MCTS:
- Need to track current player in game state or pass explicitly
- Update `_get_valid_actions()` to use `env.get_valid_moves()`
- Fix `_simulate()` to use correct API

### Entry Points

- `game/main.py`: Pygame visualization with random players
- `battleships.ipynb`: Interactive notebook for testing
- `mcts/mcts.py`: Contains example usage function `play_game()` (broken)

---

## REFACTORING DESIGN PLAN

### 1. Executive Summary

**Goal**: Transform the two-player Battleship game into a Gymnasium-compatible single-arena environment for AI agent training and comparison.

**Paradigm Shift**:
- FROM: Two players attacking each other's boards simultaneously
- TO: Single agent attacking one hidden board (classic Battleship puzzle-solving)

**Key Drivers**:
- Enable RL agent training (PPO, DQN via Stable-Baselines3)
- Support multiple agent types (human, heuristic, RL, LLM, hybrid)
- Standardize interface for fair agent comparison
- Provide rich visualization for analysis

### 2. Gap Analysis

#### Current Implementation Limitations
1. **Two-player design**: Requires both boards, not suitable for single-agent RL
2. **Non-standard API**: Custom `player_move(state, player, action)` instead of Gymnasium's `step(action)`
3. **Reward structure**: Simple hit/miss rewards don't incentivize ship sinking
4. **No ship-sinking detection**: Can't reward agents for completing ships
5. **No ship-touching constraint**: Ships can be placed adjacent (unrealistic)
6. **Limited state formats**: Only numpy arrays, no text/HTML for LLMs/humans
7. **No agent abstraction**: Each AI type would need custom integration
8. **Visualization**: Designed for two-player view, not agent perspective
9. **No validation framework**: No `env.check()` for SB3 compatibility

#### New Requirements
1. Single arena with hidden opponent board
2. Gymnasium interface: `reset()`, `step()`, `render()`, `close()`
3. Enhanced rewards: +5 hit, +10 ship sunk, +100 win, -50 invalid move
4. Track individual ship sinking
5. Ships cannot touch (including diagonally)
6. Board sizes 5x5 to 12x12 with adaptive ship configurations
7. Abstract `BattleshipAgent` interface
8. Multi-format state: numpy (RL), text (LLM), HTML (human)
9. HTML renderer with clean white background
10. Full validation and error handling

### 3. Architecture Design

#### 3.1 Module Organization

```
game/
├── __init__.py
├── env.py                 # NEW: Gymnasium environment
├── state.py               # NEW: Single-arena game state
├── config.py              # NEW: Ship configurations by board size
├── placement.py           # REFACTOR: Ship placement with no-touch constraint
├── agents/
│   ├── __init__.py
│   ├── base.py           # NEW: Abstract BattleshipAgent
│   └── random_agent.py   # NEW: Random baseline agent
├── renderers/
│   ├── __init__.py
│   ├── html.py           # NEW: HTML renderer
│   └── console.py        # NEW: ASCII console renderer (simplified)
├── core.py               # KEEP: Legacy two-player (for reference/migration)
├── main.py               # REFACTOR: Demo using new env + agents
└── visualization_pygame.py  # ARCHIVE: May refactor later for agent view

mcts/
├── mcts.py               # REFACTOR LATER: Adapt to new env

tests/
├── __init__.py
├── test_env.py           # NEW: Environment tests
├── test_placement.py     # NEW: Ship placement tests
└── test_agents.py        # NEW: Agent interface tests
```

#### 3.2 Key Data Structures

```python
# game/state.py
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
        """Register a hit on this ship. Returns True if ship just sunk."""
        if coord in self.coords:
            was_sunk = self.is_sunk
            self.hits.add(coord)
            return not was_sunk and self.is_sunk
        return False

@dataclass
class GameState:
    """Single-arena game state for agent play."""
    board_size: Tuple[int, int]  # (rows, cols)
    ship_board: np.ndarray       # Hidden board: 0=water, ship_id=ship
    attack_board: np.ndarray     # Visible: 0=unknown, 1=miss, 2=hit
    ships: Dict[str, Ship]       # ship_name -> Ship object
    move_count: int = 0
    done: bool = False
    total_ship_cells: int = 0    # For win detection

    def get_observation(self) -> Dict:
        """Return observation dict for Gymnasium."""
        remaining = [s.size for s in self.ships.values() if not s.is_sunk]
        # Pad to 5 ships (max fleet size)
        remaining += [0] * (5 - len(remaining))

        return {
            'attack_board': self.attack_board.copy(),
            'remaining_ships': np.array(remaining[:5], dtype=np.int32),
            'move_count': np.array([self.move_count], dtype=np.int32)
        }
```

```python
# game/config.py
from typing import Dict

def get_ship_config(board_size: Tuple[int, int]) -> Dict[str, int]:
    """
    Get ship configuration based on board size.

    Board Size    | Ships
    --------------|----------------------------------
    5x5 to 7x7    | Destroyer(2), Cruiser(3)
    8x8 to 9x9    | + Submarine(3)
    10x10+        | + Battleship(4), Carrier(5)
    """
    rows, cols = board_size
    min_dim = min(rows, cols)

    if min_dim < 5:
        raise ValueError("Board size must be at least 5x5")

    if min_dim <= 7:
        return {
            "destroyer": 2,
            "cruiser": 3
        }
    elif min_dim <= 9:
        return {
            "destroyer": 2,
            "cruiser": 3,
            "submarine": 3
        }
    else:  # 10+
        return {
            "carrier": 5,
            "battleship": 4,
            "cruiser": 3,
            "submarine": 3,
            "destroyer": 2
        }
```

#### 3.3 Gymnasium Environment API

```python
# game/env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any

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
        ship_placement: Optional[Dict[str, List[Tuple[int, int]]]] = None
    ):
        """
        Initialize environment.

        Args:
            board_size: (rows, cols) tuple
            render_mode: One of ["human", "html", "ansi"]
            ship_placement: Optional manual ship placement
        """
        super().__init__()

        self.board_size = board_size
        self.render_mode = render_mode
        self.manual_placement = ship_placement

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
                low=0, high=np.inf,
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
            terminated: Whether game ended (win/lose)
            truncated: Whether episode truncated (not used here)
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
                {"error": "Action out of bounds"}
            )

        row, col = divmod(action, self.board_size[1])

        # Check if already attacked
        if self.state.attack_board[row, col] != 0:
            return (
                self.state.get_observation(),
                -50.0,  # Invalid move penalty
                False,
                False,
                {"error": f"Cell ({row}, {col}) already attacked"}
            )

        # Process attack
        reward, info = self._process_attack(row, col)

        # Update move count
        self.state.move_count += 1

        # Check win condition
        terminated = self.state.done

        return (
            self.state.get_observation(),
            reward,
            terminated,
            False,  # truncated
            info
        )

    def _process_attack(self, row: int, col: int) -> Tuple[float, Dict]:
        """
        Process an attack at given coordinates.

        Returns:
            reward: Reward for this attack
            info: Information dict
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
        from game.placement import place_ships

        ship_config = get_ship_config(self.board_size)

        if self.manual_placement:
            # Use manual placement (with validation)
            ships, ship_board = self._validate_manual_placement()
        else:
            # Random placement
            ships, ship_board = place_ships(
                self.board_size,
                ship_config,
                self.np_random  # Use seeded RNG
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
```

#### 3.4 Agent Interface

```python
# game/agents/base.py
from abc import ABC, abstractmethod
from typing import Dict
import numpy as np

class BattleshipAgent(ABC):
    """Abstract base class for Battleship agents."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def select_action(self, observation: Dict[str, np.ndarray]) -> int:
        """
        Select action given current observation.

        Args:
            observation: Dict with keys:
                - 'attack_board': (rows, cols) array
                - 'remaining_ships': (5,) array of ship sizes
                - 'move_count': (1,) array

        Returns:
            action: Flattened board index (0 to rows*cols-1)
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset agent state for new episode."""
        pass

# game/agents/random_agent.py
import numpy as np
from .base import BattleshipAgent

class RandomAgent(BattleshipAgent):
    """Random agent that selects unattacked cells uniformly."""

    def __init__(self):
        super().__init__("Random")

    def select_action(self, observation: Dict[str, np.ndarray]) -> int:
        """Select random unattacked cell."""
        attack_board = observation['attack_board']
        rows, cols = attack_board.shape

        # Find all unattacked cells
        unattacked = []
        for i in range(rows):
            for j in range(cols):
                if attack_board[i, j] == 0:
                    unattacked.append(i * cols + j)

        if not unattacked:
            raise ValueError("No valid moves remaining")

        return np.random.choice(unattacked)

    def reset(self):
        """No state to reset for random agent."""
        pass
```

#### 3.5 Ship Placement with No-Touch Constraint

```python
# game/placement.py
import numpy as np
from typing import Dict, List, Tuple, Set
from game.state import Ship

def place_ships(
    board_size: Tuple[int, int],
    ship_config: Dict[str, int],
    rng: np.random.Generator
) -> Tuple[Dict[str, Ship], np.ndarray]:
    """
    Randomly place ships on board with no-touch constraint.

    Ships cannot touch each other, including diagonally.

    Args:
        board_size: (rows, cols)
        ship_config: {ship_name: size}
        rng: Random number generator (for seeding)

    Returns:
        ships: Dict of Ship objects
        ship_board: Board array with ship positions
    """
    rows, cols = board_size
    ship_board = np.zeros((rows, cols), dtype=np.int32)
    occupied_cells: Set[Tuple[int, int]] = set()
    ships = {}

    # Sort ships by size (largest first for easier placement)
    sorted_ships = sorted(ship_config.items(), key=lambda x: x[1], reverse=True)

    for ship_id, (ship_name, size) in enumerate(sorted_ships, start=1):
        max_attempts = 1000
        placed = False

        for _ in range(max_attempts):
            # Random orientation and position
            horizontal = rng.choice([True, False])

            if horizontal:
                row = rng.integers(0, rows)
                col = rng.integers(0, cols - size + 1)
                coords = [(row, col + i) for i in range(size)]
            else:
                row = rng.integers(0, rows - size + 1)
                col = rng.integers(0, cols)
                coords = [(row + i, col) for i in range(size)]

            # Check if placement is valid (no overlap or touching)
            if _is_valid_placement(coords, occupied_cells, board_size):
                # Place ship
                for r, c in coords:
                    ship_board[r, c] = ship_id

                # Mark occupied cells (including buffer zone)
                occupied_cells.update(_get_buffer_zone(coords, board_size))

                ships[ship_name] = Ship(
                    name=ship_name,
                    size=size,
                    coords=coords
                )

                placed = True
                break

        if not placed:
            raise RuntimeError(
                f"Failed to place {ship_name} after {max_attempts} attempts. "
                f"Board may be too small for ship configuration."
            )

    return ships, ship_board

def _is_valid_placement(
    coords: List[Tuple[int, int]],
    occupied: Set[Tuple[int, int]],
    board_size: Tuple[int, int]
) -> bool:
    """Check if ship placement is valid (no touching)."""
    rows, cols = board_size

    # Check all coordinates and their neighbors
    for r, c in coords:
        # Check if in bounds
        if not (0 <= r < rows and 0 <= c < cols):
            return False

        # Check 8 neighbors + self for any occupation
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if (nr, nc) in occupied:
                        return False

    return True

def _get_buffer_zone(
    coords: List[Tuple[int, int]],
    board_size: Tuple[int, int]
) -> Set[Tuple[int, int]]:
    """Get all cells occupied by ship + buffer (8-neighbors)."""
    rows, cols = board_size
    buffer = set(coords)

    for r, c in coords:
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    buffer.add((nr, nc))

    return buffer
```

#### 3.6 HTML Renderer

```python
# game/renderers/html.py
from game.state import GameState
import numpy as np

def render_html(state: GameState) -> str:
    """
    Render game state as HTML.

    Color scheme:
    - Blue (#3498db): Unknown cells
    - Gray (#95a5a6): Miss
    - Red (#e74c3c): Hit
    - Black (#2c3e50): Sunk ship
    """
    rows, cols = state.board_size

    # Determine cell color
    def get_cell_color(r, c):
        attack_val = state.attack_board[r, c]
        if attack_val == 0:  # Unknown
            return '#3498db'
        elif attack_val == 1:  # Miss
            return '#95a5a6'
        elif attack_val == 2:  # Hit
            # Check if ship is sunk
            for ship in state.ships.values():
                if (r, c) in ship.coords and ship.is_sunk:
                    return '#2c3e50'  # Sunk
            return '#e74c3c'  # Regular hit

    # Build HTML
    html = """
    <div style="background-color: white; padding: 20px; font-family: Arial, sans-serif;">
        <h2 style="color: #2c3e50;">Battleship Game</h2>
        <div style="margin-bottom: 15px;">
            <strong>Move Count:</strong> {move_count}<br>
            <strong>Ships Remaining:</strong> {ships_remaining} / {total_ships}<br>
            <strong>Status:</strong> {status}
        </div>
        <table style="border-collapse: collapse; border: 2px solid #2c3e50;">
            <tr>
                <th style="border: 1px solid #2c3e50; padding: 8px;"></th>
    """.format(
        move_count=state.move_count,
        ships_remaining=len([s for s in state.ships.values() if not s.is_sunk]),
        total_ships=len(state.ships),
        status="Game Over - Victory!" if state.done else "In Progress"
    )

    # Column headers
    for j in range(cols):
        html += f'<th style="border: 1px solid #2c3e50; padding: 8px;">{j}</th>'
    html += '</tr>\n'

    # Grid rows
    for i in range(rows):
        html += f'<tr><th style="border: 1px solid #2c3e50; padding: 8px;">{chr(65+i)}</th>'
        for j in range(cols):
            color = get_cell_color(i, j)
            html += f'<td style="border: 1px solid #2c3e50; background-color: {color}; width: 30px; height: 30px;"></td>'
        html += '</tr>\n'

    html += """
        </table>
        <div style="margin-top: 15px;">
            <strong>Legend:</strong><br>
            <span style="display: inline-block; width: 15px; height: 15px; background-color: #3498db; border: 1px solid black;"></span> Unknown &nbsp;
            <span style="display: inline-block; width: 15px; height: 15px; background-color: #95a5a6; border: 1px solid black;"></span> Miss &nbsp;
            <span style="display: inline-block; width: 15px; height: 15px; background-color: #e74c3c; border: 1px solid black;"></span> Hit &nbsp;
            <span style="display: inline-block; width: 15px; height: 15px; background-color: #2c3e50; border: 1px solid black;"></span> Sunk
        </div>
    </div>
    """

    return html
```

### 4. Implementation Strategy

#### Phase 1: Core Infrastructure (Days 1-2)
1. Create new module structure
2. Implement `GameState` and `Ship` classes
3. Implement ship configuration system
4. Build ship placement with no-touch constraint
5. Write placement tests

#### Phase 2: Gymnasium Environment (Days 3-4)
1. Implement `BattleshipEnv` class
2. Implement `reset()` and `step()` methods
3. Implement reward logic and ship sinking detection
4. Add action/observation validation
5. Write environment tests

#### Phase 3: Agents & Renderers (Day 5)
1. Implement `BattleshipAgent` abstract class
2. Implement `RandomAgent` for testing
3. Implement HTML renderer
4. Implement console renderer
5. Write agent and renderer tests

#### Phase 4: Integration & Validation (Day 6)
1. Refactor `main.py` to use new environment
2. Create example scripts for each agent type
3. Run Stable-Baselines3 `env.check()`
4. Comprehensive integration testing
5. Update documentation

#### Phase 5: Migration & Cleanup (Day 7)
1. Archive old code (`core.py`, `visualization.py`, etc.)
2. Update notebook with new examples
3. Fix MCTS to work with new env (optional)
4. Performance profiling and optimization
5. Final documentation pass

### 5. Backward Compatibility Strategy

**Archive Approach**: Keep existing code in `legacy/` folder
- Move `core.py` → `legacy/core.py`
- Move `visualization.py` → `legacy/visualization.py`
- Move `visualization_pygame.py` → `legacy/visualization_pygame.py`
- Update imports in `mcts/mcts.py` to point to legacy if not refactored

**Migration Path**:
- Old two-player code remains functional
- New single-arena code lives alongside
- Users can choose which to use
- Eventually deprecate legacy code

### 6. Testing Strategy

#### Unit Tests
```python
# tests/test_placement.py
def test_ships_dont_touch():
    """Verify no ships touch each other."""

def test_all_ships_placed():
    """Verify all ships from config are placed."""

def test_ships_within_bounds():
    """Verify all ships are within board bounds."""

# tests/test_env.py
def test_reset():
    """Test environment reset."""

def test_valid_move_rewards():
    """Test reward values for hit/miss/sunk/win."""

def test_invalid_move_penalty():
    """Test invalid move returns -50 reward."""

def test_win_condition():
    """Test game ends when all ships sunk."""

def test_observation_space():
    """Test observation matches space definition."""

# tests/test_agents.py
def test_random_agent_valid_moves():
    """Test random agent only selects valid moves."""
```

#### Integration Tests
- SB3 `env.check()` compliance
- Multi-episode gameplay
- Different board sizes
- Different ship configurations

### 7. Potential Challenges & Solutions

#### Challenge 1: Ship Placement Failure
**Problem**: Small boards with many ships may fail to place
**Solution**:
- Validate board size vs. ship configuration in `__init__`
- Raise informative error if impossible
- Provide ship config suggestions for small boards

#### Challenge 2: Performance
**Problem**: 100,000+ episodes for RL training
**Solution**:
- Profile placement algorithm
- Consider pre-generating ship placements
- Optimize numpy operations
- Use numba JIT if needed

#### Challenge 3: LLM Agent Integration
**Problem**: LLMs need text descriptions, not just numpy arrays
**Solution**:
- Create `observation_to_text()` function
- Include in renderer module
- Provide examples in docs

#### Challenge 4: SB3 Compatibility
**Problem**: Some SB3 algorithms have specific requirements
**Solution**:
- Follow Gymnasium standards strictly
- Test with multiple SB3 algorithms (PPO, DQN, A2C)
- Provide wrapper if needed

### 8. Success Criteria

- [ ] Environment passes `gymnasium.utils.env_checker.check_env()`
- [ ] Environment passes SB3 `check_env()`
- [ ] All unit tests pass (>90% coverage)
- [ ] Random agent can complete 100 consecutive games
- [ ] HTML rendering works in Jupyter
- [ ] Console rendering works in terminal
- [ ] RL agent can be trained with SB3 (proof of concept)
- [ ] Documentation is complete and accurate

### 9. Future Extensions (Post-Refactor)

1. **Heuristic Agent**: Implement DataGenetics probability-based algorithm
2. **LLM Agent**: Integration with Claude/GPT APIs
3. **Hybrid Agent**: Combine RL policy with LLM reasoning
4. **Tournament Framework**: Automated agent comparison
5. **Advanced Visualization**: Interactive pygame for human play
6. **Rule Variations**: Salvo mode, hidden mines, etc.

---

## Testing

No formal test suite exists. Manual testing via:
```bash
cd game
python main.py
```
or running cells in the Jupyter notebook.

## Key Design Notes

- The environment supports **asymmetric boards** - each player can have different board dimensions
- Ship placement is random on each reset
- Players alternate turns - no "hunt until miss" rule
- Game state is immutable-style (dataclasses used but not frozen)
- All coordinates use (row, col) indexing with 0-based indices
