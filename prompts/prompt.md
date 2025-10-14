## Project Overview
I'm developing a Battleship AI comparison project where multiple agent types will compete. The game engine needs to be agent-agnostic, allowing any AI approach to interact with it seamlessly.

## Agent Types (for future implementation)
1. Human player (manual move selection)
2. Optimal heuristic strategy (DataGenetics probability-based algorithm: http://www.datagenetics.com/blog/december32011/)
3. Reinforcement Learning agent (PPO/DQN via Stable-Baselines3)
4. LLM-based agent (Claude/GPT API with chain-of-thought prompting)
5. Hybrid RL+LLM agent

## Immediate Goal: Core Game Engine
Build a single-arena Battleship environment that meets these requirements:

### Functional Requirements
1. **Single Arena Design**: Only one board is needed since agents guess opponent ship positions (not two-player simultaneous play)
2. **Configurable Board**: Default 10×10 grid, but support variable sizes (5×5 to 12×12) for future generalization testing
3. **Ship Configuration**: Standard Battleship fleet:
   - Carrier (5 cells) (for 10 X 10 or above grid size)
   - Battleship (4 cells) (for 10 X 10 or above grid size)
   - Cruiser (3 cells)
   - Submarine (3 cells)
   - Destroyer (2 cells)
4. **Ship Placement**: Support both manual placement and random valid placement (no overlaps, ships don't touch, within bounds)
5. **Game State Tracking**: Track hits, misses, sunk ships, remaining ships, move count, game outcome

### Technical Requirements
1. **Gymnasium Interface**: Implement as a Gymnasium environment for RL compatibility
   - `action_space`: Discrete(100) for 10×10 grid (0-99 representing coordinates)
   - `observation_space`: Dict containing:
     - `attack_board`: 10×10 array (0=unknown, 1=miss, 2=hit)
     - `remaining_ships`: array of unsunk ship sizes
     - `move_count`: integer
   - `reset()`: Initialize new game, return initial observation
   - `step(action)`: Process move, return (observation, reward, terminated, truncated, info)
   
2. **Reward Function**: For RL training compatibility
   - Miss: -1
   - Hit: +5
   - Ship sunk: +10
   - Game won: +100
   - Invalid move (already shot): -50

3. **State Representation**: Support multiple formats
   - Numerical arrays for RL agents (numpy)
   - Text descriptions for LLM agents (formatted string with grid visualization)
   - HTML rendering for human visualization

4. **Agent Interface**: Define abstract base class:
   ```python
   class BattleshipAgent(ABC):
       @abstractmethod
       def select_action(self, observation: Dict) -> int:
           """Return action (0-99) given current game state"""
           pass
       
       @abstractmethod
       def reset(self):
           """Reset agent state for new game"""
           pass
   ```

### Visualization Requirements
1. **HTML Rendering**: 
   - Clean white background
   - Clear grid with cell borders
   - Color coding: blue=unknown, gray=miss, red=hit, black=sunk ship
   - Display remaining ships, move count, game status
   - Update in real-time or step-by-step

2. **Optional Console Rendering**: Simple ASCII art for debugging

### Validation & Error Handling
- Validate all moves (in bounds, not already shot)
- Validate ship placements (no overlaps, within bounds, straight lines)
- Return informative error messages
- Support `env.check()` for Stable-Baselines3 compatibility

## Existing Code
Boilerplate exists in `@game/core.py` and `@main/main.py`. Refactor as needed to meet the above specifications.

## Design Considerations
- **Modularity**: Separate game logic, visualization, and agent interfaces
- **Testability**: Pure functions where possible, deterministic for same random seed
- **Extensibility**: Easy to add new agent types, board sizes, rule variations
- **Performance**: Fast enough for 100,000+ training games (minimize overhead)

## IMPORTANT: Process Instructions
**FIRST**: Before making any code changes:
1. Review the existing code in `@game/core.py` and `@main/main.py`
2. Create a detailed design plan covering:
   - Architecture overview (class hierarchy, module organization)
   - Key data structures and their relationships
   - Implementation approach for each requirement
   - File structure and what goes where
   - What needs to be refactored vs. built from scratch
   - Potential challenges and how to address them
3. Update `@claude.md` with your complete design plan
4. Wait for my approval before implementing

**THEN**: After I approve the plan, implement the design.

## Deliverables for This Phase
1. Gymnasium-compatible Battleship environment
2. HTML visualization (basic but functional)
3. Abstract agent interface definition
4. Random agent implementation (for testing)
5. Validation tests confirming environment works correctly
6. Updated `@claude.md` with architecture documentation