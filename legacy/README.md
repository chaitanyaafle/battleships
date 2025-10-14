# Legacy Two-Player Battleship Code

This folder contains the original two-player Battleship implementation that has been replaced by the new Gymnasium-compatible single-arena environment.

## Files

- **core.py**: Original two-player `BattleshipEnv` class
  - Two boards (asymmetric sizes supported)
  - `player_move(state, player, action)` API
  - Simple rewards: +1 hit, -1 miss, +10 win

- **visualization.py**: ASCII and matplotlib visualization
  - Functions: `visualize_state_ascii()`, `visualize_state_matplotlib()`
  - Shows both players' boards

- **visualization_pygame.py**: Pygame real-time visualization
  - `BattleshipVisualizer` class
  - Side-by-side board display

- **main.py**: Two-player demo with pygame
  - Random vs random gameplay
  - Visual display

## Why Replaced?

The new Gymnasium environment (`game/env.py`) provides:
- Standard RL interface (Gymnasium API)
- Single-arena design (agent attacks hidden board)
- Better reward structure (+5 hit, +10 ship sunk, +100 win)
- Ship sinking detection
- No-touch ship placement constraint
- Modular renderers (HTML, console)
- Agent abstraction layer

## Running Legacy Code

If you want to run the old pygame visualization:

```bash
conda activate battleship_env

# From legacy/ directory:
cd legacy
python main.py

# Or update imports and run from root
```

**Note**: You may need to update imports to use `legacy.core` instead of `game.core`.

## Migration

To migrate code from legacy to new system:

**Old (Two-Player):**
```python
from game.core import BattleshipEnv

env = BattleshipEnv(board_sizes=[(10, 10), (10, 10)])
state = env.reset()
state, reward, done = env.player_move(state, player=0, action=55)
```

**New (Gymnasium):**
```python
from game.env import BattleshipEnv

env = BattleshipEnv(board_size=(10, 10))
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(55)
```

## Future

This legacy code may be deleted in a future version once all dependencies are migrated.
