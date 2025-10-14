"""Interactive human player script."""

from game.env import BattleshipEnv


def get_human_action(obs):
    """Get action from human input."""
    attack_board = obs['attack_board']
    rows, cols = attack_board.shape

    while True:
        try:
            # Get input like "A5" or "B3"
            move = input("\nEnter your move (e.g., A5, B3): ").strip().upper()

            if not move:
                continue

            # Parse input
            row_letter = move[0]
            col_str = move[1:]

            # Convert to indices
            row = ord(row_letter) - ord('A')
            col = int(col_str)

            # Validate
            if row < 0 or row >= rows or col < 0 or col >= cols:
                print(f"Invalid move! Row must be A-{chr(65+rows-1)}, column must be 0-{cols-1}")
                continue

            # Check if already attacked
            if attack_board[row, col] != 0:
                print("Already attacked that cell! Try again.")
                continue

            # Convert to flattened index
            action = row * cols + col
            return action

        except (ValueError, IndexError):
            print("Invalid format! Use letter+number (e.g., A5, B3)")
            continue


def main():
    """Run interactive game."""
    print("=" * 60)
    print("Battleship - Human Player")
    print("=" * 60)
    print()
    print("How to play:")
    print("  - Enter coordinates like 'A5' or 'B3'")
    print("  - Letters = rows (A, B, C...), Numbers = columns (0, 1, 2...)")
    print("  - Find and sink all 5 ships to win!")
    print()

    # Create environment
    env = BattleshipEnv(board_size=(10, 10), render_mode="ansi")

    # Reset
    obs, info = env.reset(seed=None)  # Random seed each time
    print(f"Game started! Ships to find: {info['total_ships']}")
    print()
    print(env.render())

    # Game loop
    done = False
    move_count = 0

    while not done:
        # Get human input
        action = get_human_action(obs)

        # Execute move
        obs, reward, terminated, truncated, info = env.step(action)
        move_count += 1

        # Show result
        print()
        if reward == -1:
            print("❌ MISS!")
        elif reward == 5:
            print("💥 HIT!")
        elif reward == 10:
            print(f"🎯 HIT! Ship sunk: {info['ship_sunk'].upper()}!")
        elif reward == 100:
            print(f"🎯 HIT! Final ship sunk: {info['ship_sunk'].upper()}!")
            print()
            print("🎉" * 20)
            print(f"🎉 VICTORY! You won in {move_count} moves! 🎉")
            print("🎉" * 20)
        elif reward == -50:
            print(f"⚠️  Invalid move: {info.get('error', 'Unknown error')}")
            move_count -= 1  # Don't count invalid moves
            continue

        # Show updated board
        print()
        print(env.render())
        print()
        print(f"Ships remaining: {info.get('ships_remaining', 0)} | Moves: {move_count}")

        done = terminated or truncated

    env.close()


if __name__ == "__main__":
    main()
