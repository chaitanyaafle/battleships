"""Debug script to understand what the PPO model sees and predicts."""

import argparse
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO

from game.env import BattleshipEnv


def main():
    parser = argparse.ArgumentParser(description="Debug PPO model observations")
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model'
    )
    args = parser.parse_args()

    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    print("Loading model...")
    model = PPO.load(model_path)
    print("✓ Model loaded")

    # Create environment
    env = BattleshipEnv(board_size=(10, 10), render_mode="ansi")
    obs, info = env.reset(seed=42)

    print("\n" + "="*60)
    print("INITIAL OBSERVATION")
    print("="*60)
    print(f"attack_board shape: {obs['attack_board'].shape}")
    print(f"remaining_ships: {obs['remaining_ships']}")
    print(f"move_count: {obs['move_count']}")
    print("\nAttack board (0=unknown, 1=miss, 2=hit):")
    print(obs['attack_board'])

    # Test first 10 moves
    print("\n" + "="*60)
    print("TESTING FIRST 10 ACTIONS")
    print("="*60)

    for step in range(10):
        # Get model's action
        action, _states = model.predict(obs, deterministic=True)
        action = int(action)
        row, col = divmod(action, 10)

        # Check if it's a valid move
        already_attacked = obs['attack_board'][row, col] != 0

        print(f"\nStep {step + 1}:")
        print(f"  Model chose action {action} -> ({row}, {col})")
        print(f"  Already attacked: {already_attacked}")

        if already_attacked:
            print(f"  ⚠️ INVALID! Cell already has value: {obs['attack_board'][row, col]}")

        # Take action
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"  Reward: {reward:+.1f}")
        print(f"  Result: {info.get('result', 'unknown')}")

        if step < 3 or already_attacked:
            print(f"  Attack board after move:")
            print(obs['attack_board'])

        if terminated:
            print("\n  Game ended!")
            break

    # Count how many cells have been attacked
    attacked_count = np.sum(obs['attack_board'] != 0)
    print("\n" + "="*60)
    print(f"Total cells attacked: {attacked_count}/100")
    print(f"Unique actions: {attacked_count}")
    print(f"Total actions: {step + 1}")
    print(f"Invalid actions: {(step + 1) - attacked_count}")

    # Check if model always predicts the same action
    print("\n" + "="*60)
    print("TESTING IF MODEL IS STUCK")
    print("="*60)

    # Reset and collect actions
    obs, _ = env.reset(seed=123)
    actions = []
    for _ in range(20):
        action, _ = model.predict(obs, deterministic=True)
        actions.append(int(action))
        obs, _, _, _, _ = env.step(action)

    unique_actions = len(set(actions))
    print(f"Actions taken: {actions[:10]}...")
    print(f"Unique actions out of 20: {unique_actions}")

    if unique_actions < 5:
        print("\n⚠️ PROBLEM: Model is stuck predicting very few actions!")
        print("This suggests:")
        print("  1. Model hasn't learned to use attack_board observation")
        print("  2. Network might not have enough capacity")
        print("  3. Training might have gotten stuck in local minimum")


if __name__ == "__main__":
    main()
