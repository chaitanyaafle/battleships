# DataGenetics Probability Agent

This implementation brings the optimal Battleship strategy from [DataGenetics](http://www.datagenetics.com/blog/december32011/) to life with beautiful visualizations.

## Quick Start

### 1. Run a Single Game with Visualization

```bash
conda activate battleship_env
python demo_probability.py --mode single --seed 42 --save
```

This runs one game and saves HTML snapshots to `output/game_42/`.

### 2. Run Multiple Games for Statistics

```bash
python demo_probability.py --mode multiple --num-games 20
```

Expected output:
```
Summary Statistics
==================
Games played: 20
Victories: 20/20 (100.0%)

Moves to completion:
  Min:    40
  Max:    66
  Mean:   ~50
  Median: 49

DataGenetics Expected: Median ~42 shots, Max ~73 shots
```

### 3. Create Animated HTML Demo

```bash
python create_animated_demo.py --seed 42 --output game_demo.html
```

Opens an interactive HTML page where you can:
- Step through each move
- See probability heatmap evolution
- Auto-play the game
- Control animation speed

## Features

### Probability Heatmap Visualization

The visualization shows two boards side-by-side:

**Attack Board (Left)**
- Blue: Unknown cells
- Gray: Misses
- Red: Hits
- Black: Sunk ships
- Orange border: Next action

**Probability Density (Right)**
- Color gradient shows likelihood of ships
- Blue (low) → Green → Yellow → Red (high)
- Numbers show maximum probability cells
- Updated after each move

### Strategy Implementation

The agent implements the full DataGenetics optimal strategy:

1. **Hunt Mode** (no active hits)
   - Calculates probability density for all unsunk ships
   - Higher probability in center (more placement options)
   - Lower probability at edges/corners

2. **Target Mode** (active hits from unsunk ships)
   - Requires placements to pass through unsunk hits
   - Applies 50× weighting to cells adjacent to hits
   - Focuses search on completing partially hit ships

3. **Sunk Ship Tracking**
   - Treats sunk ship coordinates as obstacles
   - Prevents wasting shots on completed ships
   - Automatically switches from target to hunt mode

### Performance

Based on testing:
- **Median:** ~49 moves (DataGenetics: 42)
- **Range:** 40-66 moves (DataGenetics: max 73)
- **Success rate:** 100% victory
- **Accuracy:** ~33% hit rate

Performance is close to DataGenetics benchmarks. Slight difference may be due to:
- No-touch ship placement constraint (diagonal spacing)
- Random board variations
- Implementation details

## Code Structure

```
game/agents/probability_agent.py    # Core agent implementation
game/renderers/
  ├── probability_html.py            # Heatmap visualization
  └── animated_html.py               # Interactive game replay
demo_probability.py                  # Demo script
create_animated_demo.py              # Generate interactive HTML
```

## Usage in Code

```python
from game.env import BattleshipEnv
from game.agents.probability_agent import ProbabilityAgent
from game.renderers.probability_html import render_probability_html

# Create environment and agent
env = BattleshipEnv(board_size=(10, 10))
agent = ProbabilityAgent(hit_adjacency_weight=50.0)

# Run game
obs, info = env.reset(seed=42)
agent.reset()

while not done:
    action = agent.select_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)

    # Generate visualization
    html = render_probability_html(env.state, agent, action)
    # Save or display html
    done = terminated or truncated

print(f"Game completed in {env.state.move_count} moves!")
```

## Algorithm Details

### Probability Calculation

For each unsunk ship:
1. Enumerate all valid horizontal placements
2. Enumerate all valid vertical placements
3. Increment counter for each cell covered
4. Apply hit-adjacency weighting (50×)

### Placement Validation

A placement is valid if:
- All cells within board boundaries
- No cells are misses
- No cells are sunk ship coordinates
- In target mode: passes through ≥1 unsunk hit

### Sunk Ship Detection

Agent tracks:
- All hit coordinates
- When ships are destroyed (remaining_ships decreases)
- Identifies sunk ship coords via connected component analysis
- Marks sunk coords as obstacles

## Customization

### Adjust Hit-Adjacency Weight

```python
# Lower weight (more exploration)
agent = ProbabilityAgent(hit_adjacency_weight=10.0)

# Higher weight (more focused targeting)
agent = ProbabilityAgent(hit_adjacency_weight=100.0)

# Default (recommended)
agent = ProbabilityAgent(hit_adjacency_weight=50.0)
```

### Save Game History

```python
python demo_probability.py --mode single --seed 123 --save --output-dir my_games
```

## Troubleshooting

**Agent takes too long:**
- This is normal for 10×10 boards (~100ms per move)
- Probability calculation is O(W × H × N × L)

**Performance worse than expected:**
- Run more games (20+) for stable statistics
- Check board configuration (no-touch constraint affects difficulty)
- Try different random seeds

**Visualization not showing:**
- Ensure you're using `--save` flag
- Check `output/` directory for HTML files
- Open `.html` files in a web browser

## References

- Original strategy: http://www.datagenetics.com/blog/december32011/
- Strategy document: `prompts/datagenetics_battleship_strategy.md`
- Implementation guide: `prompts/CLAUDE.md`
