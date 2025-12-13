"""Demo script to watch a trained PPO agent play Battleship."""

import argparse
from pathlib import Path
import time

from game.env import BattleshipEnv
from game.agents.rl_agent import RLAgent


def main():
    parser = argparse.ArgumentParser(description="Watch a trained PPO agent play Battleship")
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model (e.g., models/ppo_20231029_123456/final_model.zip)'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=1,
        help='Number of episodes to watch'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=0.5,
        help='Delay in seconds between moves (0 for no delay)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--board-size',
        type=int,
        nargs=2,
        default=[10, 10],
        metavar=('ROWS', 'COLS'),
        help='Board size (default: 10 10)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show adjacency bonus and escalating penalty messages'
    )
    args = parser.parse_args()

    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("\nTip: Train a model first using:")
        print("  python training/train_ppo.py")
        return

    # Load agent
    print("=" * 60)
    print("PPO Agent Demo")
    print("=" * 60)
    print(f"Loading model from: {model_path}")

    try:
        agent = RLAgent(
            name="PPO Agent",
            model_path=model_path,
            deterministic=True  # Use deterministic policy for demo
        )
        print(f"✓ Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Create environment with rendering
    env = BattleshipEnv(board_size=tuple(args.board_size), render_mode="ansi", verbose=args.verbose)

    print(f"\nRunning {args.episodes} episode(s)...\n")

    for episode in range(args.episodes):
        print("=" * 60)
        print(f"Episode {episode + 1}/{args.episodes}")
        print("=" * 60)

        # Reset
        obs, info = env.reset(seed=args.seed + episode if args.seed else None)
        agent.reset()

        done = False
        step_count = 0
        total_reward = 0

        # Show initial board (all unknown)
        print("\nInitial board (hidden ships):")
        print(env.render())
        print()

        # Track repeated moves to detect stuck agents
        move_history = set()
        repeated_moves = 0

        # Play episode
        while not done:
            # Agent selects action
            action = agent.select_action(obs)
            row, col = divmod(action, args.board_size[1])

            # Track if this is a repeated move
            if action in move_history:
                repeated_moves += 1
            move_history.add(action)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            step_count += 1
            total_reward += reward

            # Print move result
            result = info.get('result', 'unknown')
            if result == 'invalid':
                print(f"Move {step_count}: Attack ({row}, {col}) -> ⚠️  INVALID (already attacked!) (reward: {reward:+.1f})")
            else:
                print(f"Move {step_count}: Attack ({row}, {col}) -> {result.upper()} (reward: {reward:+.1f})")

            if info.get('ship_sunk'):
                print(f"  🎯 SUNK: {info['ship_sunk']}")

            if terminated:
                print(f"  🎉 VICTORY! All ships destroyed!")

            # Warn if too many repeated moves
            if repeated_moves > 10 and step_count < 50:
                print(f"  ⚠️  Warning: Agent has made {repeated_moves} repeated moves - model may not be trained properly!")

            # Show current board state
            if args.delay > 0 or done:
                print(env.render())
                print()

            # Delay between moves
            if args.delay > 0 and not done:
                time.sleep(args.delay)

        # Episode summary
        print("-" * 60)
        print(f"Episode {episode + 1} Summary:")
        print(f"  Total moves: {step_count}")
        print(f"  Unique moves: {len(move_history)}")
        print(f"  Repeated moves: {repeated_moves}")
        print(f"  Total reward: {total_reward:.1f}")
        print(f"  Ships remaining: {info.get('ships_remaining', 0)}")
        if step_count > 0:
            print(f"  Average reward per move: {total_reward / step_count:.2f}")
        print("-" * 60)

        # Diagnosis
        if repeated_moves > len(move_history) * 0.5:
            print("⚠️  DIAGNOSIS: Model is making many repeated moves!")
            print("   This suggests the model hasn't learned properly.")
            print("   Possible causes:")
            print("   - Training was too short (try more timesteps)")
            print("   - Model didn't converge (check TensorBoard logs)")
            print("   - Observation space issue (model can't see what it attacked)")
        elif step_count > 90:
            print("⚠️  DIAGNOSIS: Model is taking too many moves (>90)")
            print("   Random baseline: ~96 moves, Optimal: ~49 moves")
            print("   The model needs more training to improve.")
        elif step_count < 60:
            print("✓ Model is performing reasonably well!")
            print("  (Better than random, but can still improve)")

        print("=" * 60)
        print()

        # Pause between episodes
        if episode < args.episodes - 1:
            input("Press Enter to continue to next episode...")
            print()

    print("\n📊 Baseline comparisons:")
    print("  Random agent: ~96 moves median")
    print("  Probability agent: ~49 moves median (near-optimal)")
    print()


if __name__ == "__main__":
    main()
