from core import BattleshipEnv
from visualization_pygame import BattleshipVisualizer
import random
import time
import pygame

def main():
    # Create the environment with different board sizes for each player
    board_sizes = [(5, 5), (6, 6)]  # Player 1: 5x5, Player 2: 6x6
    env = BattleshipEnv(board_sizes=board_sizes)
    
    # Create visualizer
    visualizer = BattleshipVisualizer()
    visualizer.initialize_display(board_sizes)
    
    # Reset to get initial state
    state = env.reset()
    
    # Visualize initial state
    visualizer.visualize_state(state)

    # Make some random moves
    max_moves = 15
    move_count = 0
    
    try:
        while move_count < max_moves and not state.done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
            
            # Alternate between players
            current_player = move_count % 2
            board_size = board_sizes[current_player]
            max_pos = board_size[0] * board_size[1]
            
            # Make a random move
            move = random.randint(0, max_pos - 1)
            row, col = divmod(move, board_size[1])
            print(f"\nPlayer {current_player + 1} moving at position: {chr(65+row)}{col}")
            
            state, reward, done = env.player_move(state, current_player, move)
            print(f"Reward: {reward}")
            
            visualizer.visualize_state(state)
            time.sleep(1)  # Add delay to make it easier to follow
            
            move_count += 1
            
            if done:
                print(f"\nGame Over! Player {state.winner + 1} wins!")
                time.sleep(3)  # Show final state for a few seconds
                break
                
    except KeyboardInterrupt:
        print("\nGame terminated by user")
    finally:
        visualizer.close()

if __name__ == "__main__":
    main()