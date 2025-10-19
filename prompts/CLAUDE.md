# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a **Gymnasium-compatible Battleship environment** for AI agent training and comparison.

**STATUS**: ✅ Refactoring COMPLETE - Production ready!

### Current Features
- ✅ Gymnasium-compatible environment for RL training
- ✅ Single-arena design (agent attacks hidden board)
- ✅ Abstract agent interface for multiple AI types
- ✅ HTML and console rendering
- ✅ Probability heatmap visualization (side-by-side view)
- ✅ Animated HTML game replay with interactive controls
- ✅ Configurable board sizes (5x5 to 12x12) with adaptive ship configurations
- ✅ No-touch ship placement constraint (diagonal)
- ✅ Ship sinking detection and rewards
- ✅ Comprehensive test suite
- ✅ Interactive human player mode
- ✅ DataGenetics optimal probability agent (median ~49 moves)

### Architecture

**New Gymnasium System (Active):**
- `game/env.py` - Gymnasium environment
- `game/state.py` - GameState and Ship classes
- `game/config.py` - Adaptive ship configurations
- `game/placement.py` - No-touch ship placement
- `game/agents/` - Agent interfaces (Random, Probability)
- `game/renderers/` - HTML, console, probability heatmap, animated replay
- `tests/` - Test suite
- `demo_probability.py` - Probability agent demo script
- `create_animated_demo.py` - Generate interactive HTML replays

**Legacy Two-Player System (Archived):**
- `legacy/core.py` - Old two-player environment
- `legacy/visualization.py` - Old visualizations
- `legacy/main.py` - Old pygame demo
- See `legacy/README.md` for details

## Quick Start

### Setup Environment
```bash
conda activate battleship_env
# or: pip install -r requirements.txt
```

### Play Interactively (Human vs Computer)
```bash
python play_human.py
```

### Watch Random Agent Demo
```bash
python demo.py
```

### Run Tests
```bash
pytest tests/ -v
```

### Use in Code
```python
from game.env import BattleshipEnv
from game.agents.random_agent import RandomAgent

env = BattleshipEnv(board_size=(10, 10), render_mode="ansi")
agent = RandomAgent()

obs, info = env.reset(seed=42)
done = False

while not done:
    action = agent.select_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

print(env.render())
```

## Current Architecture

### Core Game Logic (`game/env.py`)

The `BattleshipEnv` class implements the Gymnasium interface:

**Initialization**: `BattleshipEnv(board_size, render_mode, ship_placement)`
- `board_size`: Tuple `(rows, cols)` - default (10, 10)
- `render_mode`: One of ["human", "html", "ansi"]
- `ship_placement`: Optional manual placement dict

**Game State** (`game/state.py`):
- `GameState` dataclass with:
  - `ship_board`: Hidden ship positions (0=water, ship_id>0)
  - `attack_board`: Visible attack history (0=unknown, 1=miss, 2=hit)
  - `ships`: Dict of Ship objects with sinking tracking
  - `move_count`, `done`, `total_ship_cells`

**Key Methods**:
- `reset(seed, options)`: Returns `(observation, info)`
- `step(action)`: Returns `(observation, reward, terminated, truncated, info)`
- `render()`: Returns HTML/console string based on render_mode
- `close()`: Cleanup

**Actions**: Flattened board indices: `action = row * cols + col`

### Observation Space

Dictionary with:
- `attack_board`: `(rows, cols)` array
  - 0 = unknown (not attacked)
  - 1 = miss
  - 2 = hit
- `remaining_ships`: `(5,)` array of unsunk ship sizes
- `move_count`: `(1,)` array

### Reward Structure

| Event | Reward |
|-------|--------|
| Miss | -1 |
| Hit (not sunk) | +5 |
| Ship sunk | +10 |
| Victory (all ships sunk) | +100 |
| Invalid move | -50 |

### Ship Configuration (Adaptive)

Automatically determined by board size:

| Board Size | Ships |
|------------|-------|
| 5x5 to 7x7 | Destroyer(2), Cruiser(3) |
| 8x8 to 9x9 | + Submarine(3) |
| 10x10+ | + Battleship(4), Carrier(5) |

### Visualization

**New Renderers** (`game/renderers/`):
1. **HTML** (`html.py`): Clean HTML for Jupyter notebooks
   - Color-coded cells (blue=unknown, gray=miss, red=hit, black=sunk)
   - Ship status tracking
   - White background

2. **Console** (`console.py`): ASCII for terminal
   - Symbols: · (unknown), ○ (miss), ✕ (hit), ■ (sunk)
   - Ship status list

3. **Probability HTML** (`probability_html.py`): Probability heatmap visualization
   - Side-by-side attack board and probability density
   - Color gradient (blue→green→yellow→red) for probability
   - Highlights next action
   - Real-time probability calculations

4. **Animated HTML** (`animated_html.py`): Interactive game replay
   - JavaScript-based step controls
   - Auto-play with adjustable speed
   - Complete game history navigation
   - Probability evolution visualization

**Legacy Visualizations** (`legacy/`):
- Old two-player pygame visualization
- See `legacy/README.md` for usage

### Agent Interface

**Base Class** (`game/agents/base.py`):
```python
class BattleshipAgent(ABC):
    @abstractmethod
    def select_action(self, observation: Dict) -> int:
        """Return action given observation."""

    @abstractmethod
    def reset(self):
        """Reset agent state."""
```

**Implemented Agents**:
- `RandomAgent` - Baseline random strategy (~96 moves median)
- `ProbabilityAgent` - DataGenetics optimal strategy (~49 moves median, 56% improvement!)

**Future Agents**:
- RL (PPO/DQN via Stable-Baselines3)
- LLM (Claude/GPT with chain-of-thought)
- Hybrid (RL + LLM)

### MCTS Implementation (`mcts/mcts.py`)

**STATUS**: ⚠️ BROKEN - Needs complete rewrite

The MCTS code uses the old two-player API and is incompatible with the new environment.

Issues:
- Imports from `legacy/core.py`
- Uses old `BoardState` structure
- Needs rewrite for Gymnasium API

### Entry Points

**Active**:
- `play_human.py` - Interactive human gameplay
- `demo.py` - Random agent demonstration
- `demo_probability.py` - Probability agent demo with visualization
- `create_animated_demo.py` - Generate interactive HTML replays
- `tests/` - Test suite

**Legacy** (archived):
- `legacy/main.py` - Old pygame two-player demo
- `battleships.ipynb` - Notebook (needs update for new env)

---

## Testing

### Current Test Suite

**Placement Tests** (`tests/test_placement.py`): ✅ 19 tests passing
```bash
pytest tests/test_placement.py -v
```

Tests cover:
- Ship placement without overlap
- No-touch constraint (diagonal)
- Boundary validation
- Manual placement validation
- Helper functions (buffer zones, straight lines)

**Manual Testing**:
```bash
# Interactive gameplay
python play_human.py

# Random agent demo
python demo.py

# Environment check
python -c "from game.env import BattleshipEnv; env = BattleshipEnv(); print('OK')"
```

### TODO: Additional Tests

**Environment Tests** (`tests/test_env.py`):
- Reset functionality
- Step mechanics
- Reward values
- Invalid move handling
- Win conditions
- Observation space compliance

**Agent Tests** (`tests/test_agents.py`):
- Agent interface compliance
- Valid move selection
- State management

**Integration Tests**:
- Gymnasium compatibility check
- Stable-Baselines3 compatibility
- Multi-episode gameplay
- Different board sizes

## Key Design Notes

**Current System**:
- Single-arena design (agent vs hidden board)
- Board sizes 5x5 to 12x12 supported
- Ships cannot touch (including diagonally)
- Ship placement is random on each reset (seeded RNG)
- Ship sinking tracked individually
- All coordinates use (row, col) indexing with 0-based indices
- Actions are flattened: `action = row * cols + col`

**Legacy System** (archived in `legacy/`):
- Two-player simultaneous play
- Asymmetric boards supported
- See `legacy/README.md` for details
