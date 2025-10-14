# Battleship Game Simulation

A Python implementation of the classic Battleship game with support for AI agents, multiple visualization methods, and customizable board configurations.

## Features

- **Flexible Board Sizes**: Support for asymmetric boards (different sizes for each player)
- **Multiple Visualization Methods**:
  - ASCII text-based visualization with emoji
  - Matplotlib static plots
  - Real-time Pygame visualization
- **AI Support**: Monte Carlo Tree Search (MCTS) implementation (experimental)
- **Interactive Development**: Jupyter notebook included for experimentation

## Installation

### Requirements

- Python 3.7+
- NumPy
- Pygame
- Matplotlib
- Jupyter (optional, for notebook)

### Setup

```bash
pip install numpy pygame matplotlib jupyter
```

## Quick Start

### Run the game with Pygame visualization

```bash
cd game
python main.py
```

This will start a game with random players and real-time pygame visualization.

### Run in Jupyter Notebook

```bash
jupyter notebook battleships.ipynb
```

## Usage

### Basic Game Setup

```python
from game.core import BattleshipEnv

# Create environment with custom board sizes
board_sizes = [(10, 10), (8, 8)]  # Player 1: 10x10, Player 2: 8x8
ship_specs = {
    "destroyer": 2,
    "cruiser": 3,
    "battleship": 4
}

env = BattleshipEnv(board_sizes=board_sizes, ship_specs=ship_specs)
state = env.reset()
```

### Making Moves

```python
# Player 0 attacks position (3, 5)
action = 3 * board_sizes[0][1] + 5  # Flattened index
state, reward, done = env.player_move(state, player=0, action=action)

# Get valid moves for a player
valid_moves = env.get_valid_moves(state, player=0)
```

### Visualization

```python
# ASCII visualization
from game.visualization import visualize_state_ascii
visualize_state_ascii(state, player=0, board_size=10)

# Pygame visualization
from game.visualization_pygame import BattleshipVisualizer
visualizer = BattleshipVisualizer()
visualizer.initialize_display(board_sizes)
visualizer.visualize_state(state)
```

## Game Rules

- Players take turns attacking positions on their opponent's board
- Each attack returns a reward:
  - `+1` for hitting a ship
  - `-1` for missing
  - `+10` for sinking all opponent ships (winning)
- Ships are randomly placed at the start of each game
- A ship is sunk when all its cells are hit
- First player to sink all opponent ships wins

## Project Structure

```
battleships/
├── game/
│   ├── core.py                    # Core game logic and environment
│   ├── visualization.py           # ASCII and matplotlib visualization
│   ├── visualization_pygame.py    # Pygame real-time visualization
│   └── main.py                    # Main entry point with pygame
├── mcts/
│   └── mcts.py                    # MCTS AI implementation (experimental)
├── battleships.ipynb              # Interactive Jupyter notebook
├── CLAUDE.md                      # Developer documentation
└── README.md                      # This file
```

## Board Encoding

Cells in the game boards are encoded as:
- `0`: Empty water
- `1`: Ship (only visible in own board)
- `2`: Hit ship
- `-1`: Miss

## Development

See `CLAUDE.md` for detailed architecture documentation and development guidance.

## License

See LICENSE file for details.

## Notes

- The MCTS implementation is experimental and may have bugs
- Board sizes can be different for each player (asymmetric gameplay)
- Actions are represented as flattened indices: `action = row * board_width + col`
