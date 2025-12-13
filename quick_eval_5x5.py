"""Quick eval script to compare PPO vs Random on 5x5 board."""

from game.env import BattleshipEnv
from game.agents.random_agent import RandomAgent
from game.agents.rl_agent import RLAgent
import numpy as np

# Create 5x5 environment
env = BattleshipEnv(board_size=(5, 5))

# Test Random agent
print("Testing Random Agent...")
random_agent = RandomAgent("Random")
random_lengths = []

for i in range(100):
    obs, _ = env.reset(seed=42 + i)
    random_agent.reset()
    done = False
    length = 0

    while not done:
        action = random_agent.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        length += 1

    random_lengths.append(length)

print(f"Random Agent: {np.mean(random_lengths):.1f} ± {np.std(random_lengths):.1f} moves")
print(f"  Median: {np.median(random_lengths):.1f}, Min: {np.min(random_lengths)}, Max: {np.max(random_lengths)}")

# Test PPO agent
print("\nTesting PPO Agent...")
ppo_agent = RLAgent(
    name="PPO",
    model_path="models/ppo_masked_20251213_103912/final_model.zip",
    deterministic=True
)
ppo_lengths = []

for i in range(100):
    obs, _ = env.reset(seed=42 + i)
    ppo_agent.reset()
    done = False
    length = 0

    while not done:
        action = ppo_agent.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        length += 1

    ppo_lengths.append(length)

print(f"PPO Agent: {np.mean(ppo_lengths):.1f} ± {np.std(ppo_lengths):.1f} moves")
print(f"  Median: {np.median(ppo_lengths):.1f}, Min: {np.min(ppo_lengths)}, Max: {np.max(ppo_lengths)}")

# Comparison
print("\n" + "="*50)
print("COMPARISON")
print("="*50)
improvement = (np.mean(random_lengths) - np.mean(ppo_lengths)) / np.mean(random_lengths) * 100
print(f"PPO is {improvement:.1f}% better than Random")
if improvement < 1:
    print("⚠️  Model is NOT learning anything meaningful!")
elif improvement < 10:
    print("⚠️  Model is learning something, but not much")
else:
    print("✓ Model IS learning!")
