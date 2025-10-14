"""Demo script to test the new Gymnasium environment."""

from game.env import BattleshipEnv
from game.agents.random_agent import RandomAgent


def main():
    """Run a demo game with a random agent."""
    print("=" * 60)
    print("Battleship Gymnasium Environment Demo")
    print("=" * 60)
    print()

    # Create environment
    env = BattleshipEnv(board_size=(10, 10), render_mode="ansi")

    # Create agent
    agent = RandomAgent(seed=42)

    # Reset environment
    obs, info = env.reset(seed=42)
    print(f"Environment reset. Ships: {info['total_ships']}")
    print()

    # Play game
    done = False
    step_count = 0

    while not done and step_count < 100:  # Max 100 steps
        # Agent selects action
        action = agent.select_action(obs)

        # Environment steps
        obs, reward, terminated, truncated, info = env.step(action)

        step_count += 1

        # Print result every 10 steps or on special events
        if reward > 1 or step_count % 10 == 0 or terminated:
            print(f"Step {step_count}: action={action}, reward={reward}, result={info.get('result', 'N/A')}")
            if info.get('ship_sunk'):
                print(f"  -> Ship sunk: {info['ship_sunk']}")

        done = terminated or truncated

    # Final render
    print()
    print(env.render())
    print()

    if terminated:
        print(f"🎉 Victory! Game completed in {step_count} moves.")
    else:
        print(f"Game stopped after {step_count} moves.")

    print()
    print(f"Final stats:")
    print(f"  Ships remaining: {info.get('ships_remaining', 0)} / {info.get('total_ships', 'N/A')}")
    print(f"  Total moves: {info.get('move_count', step_count)}")

    env.close()


if __name__ == "__main__":
    main()
