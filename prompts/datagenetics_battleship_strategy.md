# DataGenetics Optimal Battleship Strategy

**Source:** http://www.datagenetics.com/blog/december32011/

This document describes the near-optimal Battleship playing strategy developed by DataGenetics, validated through 100 million game simulations.

---

## Game Configuration

**Standard Battleship Setup:**
- Board: 10×10 grid (100 squares)
- Ships (17 total targets):
  - Carrier: 5 cells
  - Battleship: 4 cells
  - Cruiser: 3 cells
  - Submarine: 3 cells
  - Destroyer: 2 cells
- Ships are placed horizontally or vertically (not diagonally)
- Ships cannot overlap but can touch each other
- Feedback: HIT, MISS, and SUNK (with ship type)

---

## Strategy Evolution

### 1. Random Firing (Baseline)

**Performance:**
- Median: 96 shots to complete
- Average: ~96 shots
- Best in 100M simulations: 44 shots
- Perfect game probability: 1 in 6,650,134,872,937,201,800 games

**Analysis:**
Pure random firing is extremely inefficient. The majority of squares must be hit to ensure all ships are sunk. This provides a baseline for comparison.

---

### 2. Hunt/Target Mode (Basic Improvement)

**Algorithm:**

**Hunt Mode (Initial Search):**
- Fire at random unexplored locations
- Continue until a ship is hit

**Target Mode (Ship Pursuit):**
- When a hit is detected, switch to Target mode
- Create a stack of "potential targets" (4 cardinal directions: N, S, E, W)
- Add adjacent unexplored cells to the stack
- Pop targets from the stack and fire
- On hit: add new adjacent cells to stack
- On miss: continue popping from stack
- Return to Hunt mode when stack is empty or all ships sunk

**Limitations:**
- No concept of ship shape or size
- Must blindly walk around all edges of hit clusters
- Cannot distinguish when a ship is fully sunk
- Wastes shots checking all adjacent cells "just to be sure"

**Performance:**
- Median: ~64 shots to complete
- Significant improvement over random (64 vs. 96)
- Still inefficient due to edge-walking

---

### 3. Parity Optimization (Checkerboard Pattern)

**Key Insight:**
Since the minimum ship length is 2 units, every ship must cover at least one square of each parity (like a checkerboard).

**Implementation:**
- Imagine the board as a checkerboard (alternating blue/white squares)
- Label squares 1-100: even numbers (blue) and odd numbers (white)
- **In Hunt mode:** Only fire at even-parity (blue) squares
- **In Target mode:** Lift parity restriction, explore all adjacent cells
- Once the destroyer (length 2) is sunk, increase parity spacing

**Benefits:**
- Reduces Hunt mode search space by 50%
- Guarantees hitting every ship at least once
- No ship can be placed to avoid all blue squares

**Performance:**
- Median: ~60 shots to complete
- Incremental improvement over basic Hunt/Target
- Still inefficient in Target mode (edge-walking persists)

---

### 4. Probability Density Strategy (Near-Optimal)

This is the **optimal strategy** that achieves the best performance.

**Core Principle:**
> "Once you eliminate the impossible, whatever remains, no matter how improbable, must be the truth." — Sherlock Holmes

**Algorithm Overview:**

At the start of every turn:
1. Consider all unsunk ships and their lengths
2. Calculate a probability density map by:
   - For each unsunk ship, enumerate all possible valid placements
   - A placement is valid if it:
     - Fits within the board
     - Doesn't overlap with known misses
     - Doesn't overlap with sunk ships
     - Passes through known hits (in Target mode)
   - For each valid placement, increment a counter for every cell it covers
3. Create a superposition of all possibilities
4. Select the cell with the highest probability count
5. Fire at that cell
6. Update the board state and repeat

**Hunt Mode (No Active Hits):**
- States considered: unvisited, misses, sunk ships
- Misses and sunk ships are treated as obstructions
- Ships cannot pass through these cells
- Probability naturally higher in center (more placement options)
- Probability naturally lower at edges/corners (fewer placement options)

**Target Mode (Active Hits Present):**
- Known hits are treated as "must pass through" locations
- Ships must pass through at least one hit cell
- Heavy weighting applied to cells adjacent to hits
- Heavily weights continuations along discovered ship orientations

**Adaptive Search Space:**
- As ships are sunk, remove them from probability calculations
- Larger ships create different patterns than smaller ships
- Fewer remaining ships = more constrained search space
- When only the carrier remains, probabilities become highly concentrated

**Key Advantages:**
- No unnecessary edge-walking
- Adapts to any board configuration
- Uses all available information optimally
- Adjusts strategy as ships are sunk
- Focuses search on high-probability areas

---

## Performance Results (100 Million Simulations)

### Probability Density Strategy (Optimal)
- **Median: 42 shots**
- **Maximum: 73 shots** (worst case)
- **Perfect games (17 shots):** ~1 in 1,000,000 games
- **99th percentile:** ~58 shots

### Comparison Table

| Strategy | Median Shots | Max Shots | Improvement |
|----------|--------------|-----------|-------------|
| Random | 96 | 100 | Baseline |
| Hunt/Target | 64 | ~90 | 33% better |
| Parity Hunt/Target | 60 | ~85 | 38% better |
| **Probability Density** | **42** | **73** | **56% better** |

---

## Implementation Details

### Probability Calculation

For each cell (i, j) on the board:

```
probability[i][j] = 0

for each unsunk_ship in remaining_ships:
    ship_length = unsunk_ship.length
    
    # Try horizontal placements
    for start_col in range(0, board_width - ship_length + 1):
        if placement_is_valid(i, start_col, ship_length, HORIZONTAL):
            for k in range(ship_length):
                probability[i][start_col + k] += 1
    
    # Try vertical placements
    for start_row in range(0, board_height - ship_length + 1):
        if placement_is_valid(start_row, j, ship_length, VERTICAL):
            for k in range(ship_length):
                probability[start_row + k][j] += 1

# Apply heavy weighting to cells adjacent to known hits
for each known_hit in hits:
    for each adjacent_cell in get_adjacent(known_hit):
        probability[adjacent_cell] *= WEIGHT_FACTOR  # e.g., 10-100x

next_target = cell with max(probability)
```

### Placement Validation

A ship placement is valid if:
1. All cells are within board boundaries
2. No cells overlap with known misses
3. No cells overlap with already sunk ships
4. In Target mode: at least one cell overlaps with a known hit

### Weighting Strategy

- **Base probability:** Count of valid ship placements through each cell
- **Hit adjacency:** Multiply probability by 10-100× for cells adjacent to hits
- **Tie-breaking:** If multiple cells have equal probability, select based on consistent rule (e.g., first numerically from top-left)

---

## Key Insights

1. **Sunk ship information is crucial:** Knowing when a ship is sunk (not just hit) dramatically improves efficiency by:
   - Removing that ship from probability calculations
   - Stopping wasteful edge-walking around sunk ships
   - Adjusting minimum ship size for remaining search

2. **Edge effects matter:** Center squares have higher base probability because more ship orientations can cover them. Corner squares have lowest probability.

3. **Information compounds:** Each shot provides information that eliminates possibilities, making subsequent shots more informed.

4. **Adaptive spacing:** As larger ships are sunk, the algorithm naturally adjusts search patterns (implicit parity optimization based on remaining ship sizes).

5. **No perfect strategy exists:** Battleship is not fully "solved." The best known algorithms achieve median 42 shots, but mathematical lower bounds are unknown.

---

## Computational Complexity

- **Per-turn calculation:** O(W × H × N × L) where:
  - W = board width (10)
  - H = board height (10)
  - N = number of unsunk ships (≤5)
  - L = average ship length (~3.4)

- **Typical performance:** < 1ms per move on modern hardware
- **Memory:** O(W × H) for probability grid (100 floats)

---

## Practical Notes

- Algorithm is deterministic for a given board configuration
- No machine learning or training required
- Works optimally with standard "sunk" feedback
- Can be adapted to work without "sunk" feedback (less efficient)
- Generalizes to different board sizes and ship configurations
- No randomness in optimal mode (always picks highest probability)

---

## Example Game Walkthrough

**Game solved in 34 moves:**

1. **Initial shot:** Center of board (highest base probability) - MISS
2. **Recalculate:** Next highest probability - HIT
3. **Target mode:** Check adjacent cells - HIT (now 2 in a row)
4. **Linear search:** Continue along line - MISS (try other direction)
5. **Reverse direction:** - MISS, MISS
6. **Perpendicular check:** - HIT (find vertical component)
7. **Continue:** SUNK (first ship destroyed)
8. **Back to hunt:** Search high-probability areas - HIT
9. **Target second ship:** Quick elimination - SUNK
10. **Repeat:** Continue until all ships found

The algorithm continuously recalculates probability density after each shot, adapting its strategy based on all available information.

---

## Conclusion

The DataGenetics probability density strategy achieves near-optimal Battleship play through:
- Systematic enumeration of all possible ship configurations
- Superposition of probabilities across all possibilities
- Adaptive search based on remaining ships
- Optimal use of "sunk" information
- No wasted shots on unnecessary edge-walking

With a median of **42 shots** and maximum of **73 shots**, this strategy represents the gold standard for deterministic Battleship algorithms and serves as an excellent baseline for evaluating AI agents.