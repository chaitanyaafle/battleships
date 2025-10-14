# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Battleship game simulation repository with support for:
- Two-player turn-based gameplay with customizable board sizes (asymmetric boards supported)
- Multiple visualization methods (ASCII, matplotlib, pygame)
- MCTS (Monte Carlo Tree Search) AI implementation (currently incomplete/buggy)
- Interactive Jupyter notebook for experimentation

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
