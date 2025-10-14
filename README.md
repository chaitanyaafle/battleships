# Battleship Game Simulation

A Python implementation of the classic Battleship game designed for AI agent training and comparison.

## Overview

This repository provides a **Gymnasium-compatible** Battleship environment suitable for:
- Reinforcement Learning (RL) agent training
- AI agent comparison and benchmarking
- Interactive experimentation with different strategies
- Research on game-playing AI

## Features

### Core Environment
- **Gymnasium API**: Standard RL interface (`reset()`, `step()`, `render()`)
- **Single-Arena Design**: Agent attacks hidden board (classic puzzle-solving)
- **Adaptive Ship Configuration**: Board size determines fleet composition
- **No-Touch Constraint**: Ships cannot be adjacent (including diagonally)
- **Rich Reward Structure**: Rewards for hits (+5), ship sinking (+10), and victory (+100)
- **Ship Sinking Detection**: Track individual ship destruction

### Visualization
- **HTML Rendering**: Clean, colorful HTML for Jupyter notebooks
- **Console Rendering**: ASCII art for terminal/debugging
- **Pygame Support**: Legacy two-player visualization (in `game/core.py`)

### Agent Interface
- **Abstract Base Class**: Easy to implement new agent types
- **Random Agent**: Baseline for comparison
- **Extensible**: Support for heuristic, RL, LLM, and hybrid agents

## Installation

### Using Conda (Recommended)

```bash
# Create environment
conda create -n battleship_env python=3.12
conda activate battleship_env

# Install dependencies
pip install -r requirements.txt
```

### Using pip

```bash
pip install gymnasium numpy pytest
```

### Requirements

- Python 3.8+
- Gymnasium >= 0.29.0
- NumPy >= 1.24.0
- pytest >= 7.4.0 (for testing)

## Quick Start

### Run Demo

```bash
# Activate environment (if using conda)
conda activate battleship_env

# Run demo with random agent
python demo.py
```

### Use in Code

```python
from game.env import BattleshipEnv
from game.agents.random_agent import RandomAgent

# Create environment
env = BattleshipEnv(board_size=(10, 10), render_mode="ansi")

# Create agent
agent = RandomAgent()

# Play episode
obs, info = env.reset(seed=42)
done = False

while not done:
    action = agent.select_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

print(env.render())
```

### Run in Jupyter Notebook

```bash
jupyter notebook battleships.ipynb
```

## Usage

### Environment Configuration

```python
from game.env import BattleshipEnv

# Default 10x10 board
env = BattleshipEnv()

# Custom board size (5x5 to 12x12 supported)
env = BattleshipEnv(board_size=(8, 8))

# With rendering
env = BattleshipEnv(board_size=(10, 10), render_mode="html")
```

### Board Sizes and Ship Configurations

The environment automatically adapts ship fleets to board size:

| Board Size | Ships |
|------------|-------|
| 5x5 - 7x7  | Destroyer (2), Cruiser (3) |
| 8x8 - 9x9  | + Submarine (3) |
| 10x10+     | + Battleship (4), Carrier (5) |

### Creating Custom Agents

```python
from game.agents.base import BattleshipAgent
import numpy as np

class MyAgent(BattleshipAgent):
    def __init__(self):
        super().__init__("MyAgent")

    def select_action(self, observation):
        # observation contains:
        # - 'attack_board': (rows, cols) array
        # - 'remaining_ships': (5,) array of ship sizes
        # - 'move_count': (1,) array

        # Your strategy here
        attack_board = observation['attack_board']
        # Return flattened index
        return 0

    def reset(self):
        # Reset any internal state
        pass
```

### Observation and Action Spaces

**Action Space**: `Discrete(rows * cols)`
- Flattened board indices: `action = row * cols + col`

**Observation Space**: `Dict` with:
- `attack_board`: `(rows, cols)` array
  - `0` = unknown (not attacked)
  - `1` = miss
  - `2` = hit
- `remaining_ships`: `(5,)` array of unsunk ship sizes
- `move_count`: `(1,)` array

### Rewards

| Event | Reward |
|-------|--------|
| Miss | -1 |
| Hit (ship not sunk) | +5 |
| Ship sunk | +10 |
| Victory (all ships sunk) | +100 |
| Invalid move | -50 |

## Project Structure

```
battleships/
├── game/
│   ├── env.py                     # Gymnasium environment (NEW)
│   ├── state.py                   # Game state and Ship classes (NEW)
│   ├── config.py                  # Ship configurations (NEW)
│   ├── placement.py               # Ship placement with no-touch (NEW)
│   ├── agents/
│   │   ├── base.py               # Abstract agent interface (NEW)
│   │   └── random_agent.py       # Random baseline agent (NEW)
│   ├── renderers/
│   │   ├── html.py               # HTML renderer (NEW)
│   │   └── console.py            # Console renderer (NEW)
│   ├── core.py                   # Legacy two-player environment
│   ├── visualization.py          # Legacy visualizations
│   ├── visualization_pygame.py   # Legacy pygame visualization
│   └── main.py                   # Legacy pygame demo
├── tests/
│   └── test_placement.py         # Placement tests (NEW)
├── mcts/
│   └── mcts.py                   # MCTS (needs update for new env)
├── prompts/
│   ├── CLAUDE.md                 # Developer documentation
│   └── prompt.md                 # Design requirements
├── demo.py                       # Demo script (NEW)
├── requirements.txt              # Dependencies (NEW)
├── battleships.ipynb             # Interactive notebook
└── README.md                     # This file
```

## Testing

```bash
# Run placement tests
pytest tests/test_placement.py -v

# Run all tests
pytest tests/ -v

# Check environment compatibility
python -c "from game.env import BattleshipEnv; env = BattleshipEnv(); print('Environment OK')"
```

## Game Rules

- Agent attacks a hidden board to find and sink all ships
- Ships cannot touch each other (including diagonally)
- Ships are randomly placed at game start
- A ship is sunk when all its cells are hit
- Game ends when all ships are sunk

## Future Agent Types (Planned)

1. **Heuristic Agent**: Probability-based targeting (DataGenetics algorithm)
2. **RL Agent**: PPO/DQN trained via Stable-Baselines3
3. **LLM Agent**: Claude/GPT with chain-of-thought reasoning
4. **Hybrid Agent**: RL + LLM combination

## Development

See `prompts/CLAUDE.md` for:
- Complete architecture documentation
- Design decisions and rationale
- Implementation phases
- Testing strategy
- Future extensions

## Legacy Code

The original two-player environment is preserved in:
- `game/core.py` - Two-player game logic
- `game/visualization.py` - ASCII/matplotlib rendering
- `game/visualization_pygame.py` - Pygame visualization
- `game/main.py` - Two-player demo

These may be moved to `legacy/` in a future update.

## License

See LICENSE file for details.

## Contributing

Contributions welcome! Please see the design plan in `prompts/CLAUDE.md` for architecture guidelines.
