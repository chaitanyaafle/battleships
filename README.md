---
type: project
status: active
tags:
  - ml
  - research
  - reinforcement-learning
  - battleships
---
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
- **Probability Heatmap**: Side-by-side attack board and probability density visualization
- **Animated Replay**: Interactive HTML with step controls and auto-play
- **Pygame Support**: Legacy two-player visualization (archived in `legacy/`)

### Agent Interface
- **Abstract Base Class**: Easy to implement new agent types
- **Random Agent**: Baseline for comparison
- **Probability Agent**: Near-optimal DataGenetics strategy (median ~49 moves)
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

### Play Interactively (Human vs Computer)

```bash
# Activate environment (if using conda)
conda activate battleship_env

# Play against the computer
python play_human.py
```

### Watch Agent Demos

```bash
# Activate environment (if using conda)
conda activate battleship_env

# Watch random agent play
python demo.py

# Watch probability agent (DataGenetics optimal strategy)
python demo_probability.py --mode single --seed 42

# Run multiple games and see statistics
python demo_probability.py --mode multiple --num-games 20

# Create interactive animated HTML demo
python create_animated_demo.py --seed 42 --output game.html
```

### Train RL Agents (PPO with Action Masking)

```bash
# Install RL dependencies (if not already installed)
pip install stable-baselines3 sb3-contrib

# Train PPO agent with action masking (50k timesteps)
python training/train_ppo_masked.py --timesteps 50000

# Continue training from checkpoint
python training/train_ppo_masked.py \
  --resume models/ppo_masked_XXXXXX/final_model.zip \
  --timesteps 50000

# Evaluate trained model
python training/evaluate.py \
  --models models/ppo_masked_XXXXXX/final_model.zip \
  --n-episodes 100 \
  --verbose

# Compare multiple models + baselines
python training/evaluate.py \
  --models \
    models/ppo_masked_20251212_232027/final_model.zip \
    models/ppo_masked_20251031_161654/final_model.zip \
  --baselines \
  --n-episodes 100 \
  --output results.csv

# Evaluate all masked PPO models
python training/evaluate.py \
  --models models/ppo_masked_*/final_model.zip \
  --baselines \
  --n-episodes 100
```

**Performance Benchmarks** (10×10 board):
- **Random Agent**: ~96 moves median
- **Probability Agent (DataGenetics)**: ~49 moves median
- **PPO with Action Masking (50k steps)**: ~92 moves, 100% win rate
- **PPO with Action Masking (100k+ steps)**: ~60-70 moves (expected)

**Why Action Masking?**
- Standard PPO wastes training time on invalid moves
- Action masking prevents selecting already-attacked cells
- 4-10× faster convergence (50k vs 200k+ timesteps)
- Guaranteed 0 invalid moves during play

See [ACTION_MASKING_SOLUTION.md](ACTION_MASKING_SOLUTION.md) for details.

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
│   ├── env.py                     # Gymnasium environment
│   ├── state.py                   # Game state and Ship classes
│   ├── config.py                  # Ship configurations
│   ├── placement.py               # Ship placement with no-touch
│   ├── wrappers.py                # Gym wrappers (ActionMask, etc.)
│   ├── agents/
│   │   ├── base.py               # Abstract agent interface
│   │   ├── random_agent.py       # Random baseline agent
│   │   ├── probability_agent.py  # DataGenetics optimal strategy
│   │   └── rl_agent.py           # RL agent (PPO/DQN)
│   └── renderers/
│       ├── html.py               # HTML renderer
│       ├── console.py            # Console renderer
│       ├── probability_html.py   # Probability heatmap visualization
│       └── animated_html.py      # Interactive game replay
├── training/
│   ├── train_ppo.py              # Train PPO (standard)
│   ├── train_ppo_masked.py       # Train PPO with action masking
│   ├── train_dqn.py              # Train DQN
│   └── evaluate.py               # Evaluate and compare agents
├── legacy/
│   ├── README.md                 # Legacy code documentation
│   ├── core.py                   # Old two-player environment
│   ├── visualization.py          # Old visualizations
│   ├── visualization_pygame.py   # Old pygame visualization
│   └── main.py                   # Old pygame demo
├── tests/
│   └── test_placement.py         # Placement tests (19 passing)
├── mcts/
│   └── mcts.py                   # MCTS (broken, needs rewrite)
├── prompts/
│   ├── CLAUDE.md                 # Developer documentation
│   ├── prompt.md                 # Design requirements
│   ├── research_plan.md          # Research notes
│   └── datagenetics_battleship_strategy.md  # Strategy reference
├── play_human.py                 # Interactive human player
├── demo.py                       # Random agent demo
├── demo_probability.py           # Probability agent demo
├── create_animated_demo.py       # Generate interactive HTML
├── ACTION_MASKING_SOLUTION.md    # Action masking guide
├── PROBABILITY_AGENT_README.md   # Probability agent guide
├── requirements.txt              # Dependencies
├── battleships.ipynb             # Interactive notebook (needs update)
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

## Implemented Agents

### 1. Random Agent (`game/agents/random_agent.py`)
- **Strategy**: Selects random unattacked cells
- **Performance**: ~96 moves median (baseline)
- **Use case**: Baseline comparison

### 2. Probability Agent (`game/agents/probability_agent.py`)
- **Strategy**: DataGenetics optimal probability density algorithm
- **Performance**: ~49 moves median (56% improvement over random!)
- **Features**:
  - Enumerates all valid ship placements
  - Calculates probability density for each cell
  - 50× hit-adjacency weighting for target mode
  - Automatic hunt/target mode switching
  - Tracks sunk ships to avoid wasted shots
- **Visualization**: Side-by-side probability heatmap
- **Source**: http://www.datagenetics.com/blog/december32011/

See `PROBABILITY_AGENT_README.md` for detailed usage and examples.

### 3. RL Agent (`game/agents/rl_agent.py`)
- **Strategy**: Reinforcement learning (PPO/DQN) via Stable-Baselines3
- **Performance**:
  - With action masking (50k steps): ~92 moves, 100% win rate
  - With action masking (100k+ steps): ~60-70 moves (expected)
- **Features**:
  - Supports both PPO and DQN algorithms
  - MaskablePPO support for action masking
  - Automatic algorithm detection from filename
  - Deterministic evaluation mode
- **Training**: See `training/train_ppo_masked.py`
- **Evaluation**: See `training/evaluate.py`

See `ACTION_MASKING_SOLUTION.md` for training guide and performance details.

### Future Agent Types (Planned)

1. **LLM Agent**: Claude/GPT with chain-of-thought reasoning
2. **Hybrid Agent**: RL + LLM combination

## Development

See `prompts/CLAUDE.md` for:
- Complete architecture documentation
- Design decisions and rationale
- Implementation phases
- Testing strategy
- Future extensions

## Legacy Code

The original two-player environment has been archived in `legacy/`:
- `legacy/core.py` - Old two-player game logic
- `legacy/visualization.py` - Old ASCII/matplotlib rendering
- `legacy/visualization_pygame.py` - Old pygame visualization
- `legacy/main.py` - Old two-player demo
- `legacy/README.md` - Migration guide and documentation

See `legacy/README.md` for usage instructions and migration guide to the new Gymnasium environment.

## License

See LICENSE file for details.

## Contributing

Contributions welcome! Please see the design plan in `prompts/CLAUDE.md` for architecture guidelines.
