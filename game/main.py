from core import BattleshipEnv
from visualization import visualize_state_ascii, visualize_state_matplotlib

def main():
    # Create the environment
    env = BattleshipEnv()
    
    # Reset to get initial state
    state = env.reset()
    
    # Visualize initial state
    print("\nInitial State:")
    visualize_state_ascii(state)
    visualize_state_matplotlib(state)
    
    # Let's make some specific moves
    # Moves are represented as numbers 0-99 for a 10x10 board
    # To convert row A-J and column 0-9 to a move number:
    # move = row * 10 + column
    # For example: 
    # A0 = 0 * 10 + 0 = 0
    # B5 = 1 * 10 + 5 = 15
    # C2 = 2 * 10 + 2 = 22
    
    moves = [
        15,  # B5
        22,  # C2
        45,  # E5
        77,  # H7
        99   # J9
    ]
    
    for move in moves:
        row, col = divmod(move, 10)
        print(f"\nMaking move at position: {chr(65+row)}{col}")
        
        # Make the move
        state, reward, done = env.step(state, move)
        
        # Print result
        print(f"Reward: {reward}")
        if reward == 1:
            print("HIT!")
        elif reward == -1:
            print("MISS!")
        
        # Visualize the new state
        visualize_state_ascii(state)
        visualize_state_matplotlib(state)
        
        if done:
            print("\nGame Over!")
            break

if __name__ == "__main__":
    main()