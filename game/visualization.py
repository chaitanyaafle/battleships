import matplotlib.pyplot as plt
import numpy as np

def visualize_state_ascii(state: BoardState) -> None:
    """
    Visualizes the game state using ASCII characters.
    ◻️ : Empty
    🚢 : Ship
    💥 : Hit
    ❌ : Miss
    """
    symbols = {
        0: "◻️ ",  # Empty
        1: "🚢 ",  # Ship
        2: "💥 ",  # Hit
        -1: "❌ "  # Miss
    }
    
    print("\nPlayer 1's Board:")
    print("  0 1 2 3 4 5 6 7 8 9")
    for i in range(10):
        row = chr(65 + i) + " "
        for j in range(10):
            row += symbols[state.boards[0][i, j]]
        print(row)
    
    print("\nPlayer 2's Board:")
    print("  0 1 2 3 4 5 6 7 8 9")
    for i in range(10):
        row = chr(65 + i) + " "
        for j in range(10):
            # Only show hits and misses for opponent's board
            if state.hit_boards[1][i, j] in [-1, 2]:
                row += symbols[state.hit_boards[1][i, j]]
            else:
                row += symbols[0]  # Show empty for unknown cells
        print(row)

def visualize_state_matplotlib(state: BoardState) -> None:
    """
    Visualizes the game state using matplotlib with a cleaner look.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Create custom colormap
    cmap = plt.cm.colors.ListedColormap(['lightgray', 'navy', 'red', 'black'])
    bounds = [-1.5, -0.5, 0.5, 1.5, 2.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
    
    # Plot player's board
    ax1.imshow(state.boards[0], cmap=cmap, norm=norm)
    ax1.set_title("Player 1's Board")
    
    # Plot opponent's board (only showing hits and misses)
    opponent_view = np.zeros_like(state.boards[1])
    opponent_view[state.hit_boards[1] == _HIT_IDX] = _HIT_IDX
    opponent_view[state.hit_boards[1] == _MISS_IDX] = _MISS_IDX
    ax2.imshow(opponent_view, cmap=cmap, norm=norm)
    ax2.set_title("Player 2's Board (Opponent)")
    
    # Add grid
    for ax in [ax1, ax2]:
        ax.grid(True, which='major', color='white', linewidth=2)
        ax.set_xticks(np.arange(-.5, 10, 1))
        ax.set_yticks(np.arange(-.5, 10, 1))
        ax.set_xticklabels(np.arange(0, 10))
        ax.set_yticklabels([chr(65+i) for i in range(10)])
    
    plt.tight_layout()
    plt.show()

# Example usage function
def play_and_visualize():
    env = BattleshipEnv()
    state = env.reset()
    
    # Visualize initial state
    print("Initial State:")
    visualize_state_ascii(state)
    visualize_state_matplotlib(state)
    
    # Make some moves
    actions = [0, 15, 22, 45]  # example moves
    for action in actions:
        row, col = divmod(action, 10)
        print(f"\nMaking move at position: {chr(65+row)}{col}")
        state, reward, done = env.step(state, action)
        print(f"Reward: {reward}, Game Over: {done}")
        visualize_state_ascii(state)
        visualize_state_matplotlib(state)
        
        if done:
            print("\nGame Over!")
            break
