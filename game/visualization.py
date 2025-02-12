import matplotlib.pyplot as plt
import numpy as np
from game.core import BattleshipEnv, BoardState
def visualize_state_ascii(state: BoardState, player: int, board_size: int = 10) -> None:
    """
    Visualizes a single player's game state using ASCII characters.
    ◻️ : Empty
    🚢 : Ship
    💥 : Hit
    💀 : Destroyed Ship
    ❌ : Miss
    
    Args:
        state: BoardState object containing game state
        player: Which player's board to show (0 for player 1, 1 for player 2)
        board_size: Size of the game board (board_size x board_size)
    """
    symbols = {
        0: "◻️ ",   # Empty
        1: "🚢 ",   # Ship
        2: "💥 ",   # Hit
        3: "💀 ",   # Destroyed ship
        -1: "❌ "   # Miss
    }
    
    # Header row with column numbers
    header = "  " + " ".join(str(i) for i in range(board_size))
    
    print(f"\nPlayer {player + 1}'s Board:")
    print(header)
    for i in range(board_size):
        row = chr(65 + i) + " "  # Use letters A, B, C, etc. for row labels
        for j in range(board_size):
            cell_value = state.boards[0][i, j] if player == 0 else state.hit_boards[1][i, j]
            
            # Check if this cell is part of a destroyed ship
            if cell_value == 2 and is_ship_destroyed(state, i, j, player):
                row += symbols[3]  # Use destroyed ship symbol
            elif player == 1 and cell_value not in [-1, 2]:
                row += symbols[0]  # Show empty for unknown cells on opponent's board
            else:
                row += symbols[cell_value]
        print(row)

def visualize_state_matplotlib(state: BoardState, player: int, board_size: int = 10) -> None:
    """
    Visualizes a single player's game state using matplotlib.
    
    Args:
        state: BoardState object containing game state
        player: Which player's board to show (0 for player 1, 1 for player 2)
        board_size: Size of the game board (board_size x board_size)
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Create custom colormap with destroyed ships
    cmap = plt.cm.colors.ListedColormap(['lightgray', 'navy', 'red', 'purple', 'black'])
    bounds = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
    
    # Create the board data
    board_data = np.zeros((board_size, board_size))
    if player == 0:
        board_data = state.boards[0][:board_size, :board_size].copy()
        title = "Player 1's Board"
    else:
        board_data[:] = state.hit_boards[1][:board_size, :board_size]
        mask = (board_data != -1) & (board_data != 2)
        board_data[mask] = 0
        title = "Player 2's Board (Opponent)"
    
    # Mark destroyed ships
    for i in range(board_size):
        for j in range(board_size):
            if board_data[i, j] == 2 and is_ship_destroyed(state, i, j, player):
                board_data[i, j] = 3  # Use new value for destroyed ships
    
    ax.imshow(board_data, cmap=cmap, norm=norm)
    ax.set_title(title, size=12)
    
    # Add grid and labels
    ax.grid(True, which='major', color='white', linewidth=2)
    ax.set_xticks(np.arange(-.5, board_size - 0.5, 1))
    ax.set_yticks(np.arange(-.5, board_size - 0.5, 1))
    ax.set_xticklabels([str(i) for i in (np.arange(0, board_size))])
    ax.set_yticklabels([chr(65+i) for i in range(board_size)], va='center')
    
    # Make sure the axes show only the board size we want
    ax.set_xlim(-0.5, board_size - 0.5)
    ax.set_ylim(board_size - 0.5, -0.5)  # Reverse y-axis to match game coordinates
    
    plt.tight_layout()
    plt.show()

def is_ship_destroyed(state: BoardState, row: int, col: int, player: int) -> bool:
    """
    Check if a ship at the given position is completely destroyed.
    
    Args:
        state: BoardState object containing game state
        row: Row index of the cell
        col: Column index of the cell
        player: Which player's board to check (0 for player 1, 1 for player 2)
    
    Returns:
        bool: True if the ship is completely destroyed, False otherwise
    """
    board = state.boards[0] if player == 0 else state.boards[1]
    hit_board = state.hit_boards[0] if player == 0 else state.hit_boards[1]
    
    # If this isn't a hit, return False
    if hit_board[row, col] != 2:
        return False
    
    # Check horizontally
    def check_direction(start_row, start_col, d_row, d_col):
        r, c = start_row, start_col
        while 0 <= r < len(board) and 0 <= c < len(board):
            if board[r, c] == 1:  # If there's a ship piece
                if hit_board[r, c] != 2:  # If it's not hit
                    return False
            else:
                break
            r, c = r + d_row, c + d_col
        return True
    
    # Check all four directions
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    for d_row, d_col in directions:
        if not check_direction(row, col, d_row, d_col):
            return False
    
    return True

# Example usage function remains the same as before
def play_and_visualize(player: int, board_size: int = 10):
    """
    Play and visualize the game with a custom board size.
    
    Args:
        player: Which player's board to show (0 for player 1, 1 for player 2)
        board_size: Size of the game board (board_size x board_size)
    """
    env = BattleshipEnv(board_size=board_size)
    state = env.reset()
    
    # Visualize initial state
    print("Initial State:")
    visualize_state_ascii(state, player, board_size)
    visualize_state_matplotlib(state, player, board_size)
    
    # Make some moves (adjusted for board size)
    actions = [0, board_size+1, 2*board_size+2]  # example moves
    for action in actions:
        row, col = divmod(action, board_size)
        print(f"\nMaking move at position: {chr(65+row)}{col}")
        state, reward, done = env.step(state, action)
        print(f"Reward: {reward}, Game Over: {done}")
        visualize_state_ascii(state, player, board_size)
        visualize_state_matplotlib(state, player, board_size)
        
        if done:
            print("\nGame Over!")
            break