"""Debug script to verify reward structure and adjacency bonus."""

from game.env import BattleshipEnv
import numpy as np

# Create environment
env = BattleshipEnv(board_size=(10, 10))

# Reset environment
obs, info = env.reset(seed=42)

print("=" * 60)
print("REWARD STRUCTURE TEST")
print("=" * 60)

# Simulate some moves and check rewards
moves = []
total_adjacency_bonuses = 0
total_moves = 0

done = False
while not done:
    # Random action on unattacked cell
    attack_board = obs['attack_board']
    unattacked = np.where(attack_board.flatten() == 0)[0]

    if len(unattacked) == 0:
        break

    action = np.random.choice(unattacked)
    row, col = divmod(action, 10)

    # Check if this action is adjacent to a hit BEFORE taking action
    adjacent_to_hit = False
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        r, c = row + dr, col + dc
        if 0 <= r < 10 and 0 <= c < 10:
            if attack_board[r, c] == 2:  # Adjacent to hit
                adjacent_to_hit = True
                break

    obs, reward, terminated, truncated, info_step = env.step(action)
    done = terminated or truncated
    total_moves += 1

    result = info_step.get('result', 'unknown')

    # Track adjacency bonus usage
    if result == 'hit' and adjacent_to_hit:
        # Should get base (5.0) + adjacency (5.0) + time_penalty (-0.3) = 9.7
        expected_reward = 5.0 + 5.0 - 0.3
        if abs(reward - expected_reward) < 0.01:
            total_adjacency_bonuses += 1
            print(f"Move {total_moves}: Adjacent hit! Reward: {reward:.1f} (expected: {expected_reward:.1f}) ✓")

    if result == 'hit' and not adjacent_to_hit:
        # Should get base (5.0) + 0 + time_penalty (-0.3) = 4.7
        expected_reward = 5.0 - 0.3
        print(f"Move {total_moves}: Regular hit. Reward: {reward:.1f} (expected: {expected_reward:.1f})")

    if done:
        print(f"\n{'='*60}")
        print(f"GAME OVER in {total_moves} moves")
        print(f"Adjacency bonuses received: {total_adjacency_bonuses}")
        print(f"Adjacency bonus usage: {total_adjacency_bonuses/total_moves*100:.1f}%")
        print(f"Final reward: {reward:.1f}")
        print(f"Result: {result}")
        print(f"{'='*60}")

print(f"\nTotal moves: {total_moves}")
print(f"Win: {info_step.get('result') == 'win'}")
