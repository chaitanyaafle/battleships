import matplotlib.pyplot as plt
import numpy as np
from core import BattleshipEnv, BoardState

def visualize_state_ascii(state: BoardState, board_size: int = 10) -> None:
    """
    Visualizes the game state using ASCII characters with customizable board size.
    ◻️ : Empty
    🚢 : Ship
    💥 : Hit
    ❌ : Miss
    
    Args:
        state: BoardState object containing game state
        board_size: Size of the game board (board_size x board_size)
    """
    symbols = {
        0: "◻️ ",  # Empty
        1: "🚢 ",  # Ship
        2: "💥 ",  # Hit
        -1: "❌ "  # Miss
    }
    
    # Header row with column numbers
    header = "  " + " ".join(str(i) for i in range(board_size))
    
    print("\nPlayer 1's Board:")
    print(header)
    for i in range(board_size):
        row = chr(65 + i) + " "  # Use letters A, B, C, etc. for row labels
        for j in range(board_size):
            row += symbols[state.boards[0][i, j]]
        print(row)
    
    print("\nPlayer 2's Board:")
    print(header)
    for i in range(board_size):
        row = chr(65 + i) + " "
        for j in range(board_size):
            # Only show hits and misses for opponent's board
            if state.hit_boards[1][i, j] in [-1, 2]:
                row += symbols[state.hit_boards[1][i, j]]
            else:
                row += symbols[0]  # Show empty for unknown cells
        print(row)

def visualize_state_matplotlib(state: BoardState, board_size: int = 10) -> None:
    """
    Visualizes the game state using matplotlib with customizable board size.
    
    Args:
        state: BoardState object containing game state
        board_size: Size of the game board (board_size x board_size)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Create custom colormap
    cmap = plt.cm.colors.ListedColormap(['lightgray', 'navy', 'red', 'black'])
    bounds = [-1.5, -0.5, 0.5, 1.5, 2.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
    
    # Plot player's board
    ax1.imshow(state.boards[0][:board_size, :board_size], cmap=cmap, norm=norm)
    ax1.set_title("Player 1's Board")
    
    # Plot opponent's board (only showing hits and misses)
    opponent_view = np.zeros((board_size, board_size))
    opponent_view[:] = state.hit_boards[1][:board_size, :board_size]
    mask = (opponent_view != -1) & (opponent_view != 2)
    opponent_view[mask] = 0
    ax2.imshow(opponent_view, cmap=cmap, norm=norm)
    ax2.set_title("Player 2's Board (Opponent)")
    
    # Add grid and labels
    for ax in [ax1, ax2]:
        ax.grid(True, which='major', color='white', linewidth=2)
        ax.set_xticks(np.arange(-.5, board_size - 0.5, 1))
        ax.set_yticks(np.arange(-.5, board_size - 0.5, 1))
        ax.set_xticklabels(np.arange(0, board_size))
        ax.set_yticklabels([chr(65+i) for i in range(board_size)])
        
        # Make sure the axes show only the board size we want
        ax.set_xlim(-0.5, board_size - 0.5)
        ax.set_ylim(board_size - 0.5, -0.5)  # Reverse y-axis to match game coordinates
    
    plt.tight_layout()
    plt.show()

# Example usage function
def play_and_visualize(board_size: int = 10):
    """
    Play and visualize the game with a custom board size.
    
    Args:
        board_size: Size of the game board (board_size x board_size)
    """
    env = BattleshipEnv(board_size=board_size)
    state = env.reset()
    
    # Visualize initial state
    print("Initial State:")
    visualize_state_ascii(state, board_size)
    visualize_state_matplotlib(state, board_size)
    
    # Make some moves (adjusted for board size)
    actions = [0, board_size+1, 2*board_size+2]  # example moves
    for action in actions:
        row, col = divmod(action, board_size)
        print(f"\nMaking move at position: {chr(65+row)}{col}")
        state, reward, done = env.step(state, action)
        print(f"Reward: {reward}, Game Over: {done}")
        visualize_state_ascii(state, board_size)
        visualize_state_matplotlib(state, board_size)
        
        if done:
            print("\nGame Over!")
            break