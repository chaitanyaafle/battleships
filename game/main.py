from core import BattleshipEnv
from visualization import visualize_state_ascii, visualize_state_matplotlib
import random


def main():
    # Create the environment
    board_size = 5
    env = BattleshipEnv(board_size=board_size)
    
    # Reset to get initial state
    state = env.reset()
    
    # Visualize initial state
    print("\nInitial State:")
    visualize_state_ascii(state, board_size)
    visualize_state_matplotlib(state, board_size)

    # For a 5x5 board, valid moves are 0-24
    moves = [
        0,   # A0
        6,   # B1
        12,  # C2
        18,  # D3
        24   # E4
    ]

    moves = random.sample(range(25), 10)
    
    print("moves = {}".format(moves))
    for move in moves:
        print("------------------------------------------------------")
        row, col = divmod(move, board_size)  # Note: using board_size instead of 10
        print(f"\nMaking move at position: {chr(65+row)}{col}")
        
        state, reward, done = env.step(state, move)
        print(f"Reward: {reward}")
        
        visualize_state_ascii(state, board_size)
        visualize_state_matplotlib(state, board_size)
        
        if done:
            print("\nGame Over!")
            break
            
if __name__ == "__main__":
    main()