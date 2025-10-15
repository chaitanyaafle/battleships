"""
Demo script for DataGenetics Probability Agent.

This script demonstrates the optimal Battleship playing strategy
developed by DataGenetics, with visualization of the probability
density calculations at each step.

Expected performance (from DataGenetics research):
- Median: ~42 shots to win
- Maximum: ~73 shots
- 56% improvement over random baseline
"""

import os
from pathlib import Path
from game.env import BattleshipEnv
from game.agents.probability_agent import ProbabilityAgent
from game.renderers.probability_html import render_probability_html


def run_single_game(
    seed: int = 42,
    verbose: bool = True,
    save_snapshots: bool = False,
    output_dir: str = "output"
) -> dict:
    """
    Run a single game with the probability agent.

    Args:
        seed: Random seed for reproducibility
        verbose: Whether to print progress
        save_snapshots: Whether to save HTML snapshots
        output_dir: Directory to save snapshots

    Returns:
        Dictionary with game statistics
    """
    # Create environment
    env = BattleshipEnv(board_size=(10, 10), render_mode="html")

    # Create agent
    agent = ProbabilityAgent(hit_adjacency_weight=50.0)

    # Setup output directory if saving snapshots
    if save_snapshots:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        game_dir = output_path / f"game_{seed}"
        game_dir.mkdir(exist_ok=True)

    # Reset environment
    obs, info = env.reset(seed=seed)
    agent.reset()

    if verbose:
        print("=" * 70)
        print("DataGenetics Probability Agent Demo")
        print("=" * 70)
        print(f"Seed: {seed}")
        print(f"Ships: {info['total_ships']}")
        print()

    # Play game
    done = False
    step_count = 0
    total_reward = 0
    ships_sunk = []
    hit_count = 0
    miss_count = 0

    while not done:
        # Agent selects action
        action = agent.select_action(obs)
        action_row, action_col = divmod(action, 10)

        # Environment steps
        obs, reward, terminated, truncated, info = env.step(action)

        step_count += 1
        total_reward += reward

        result = info.get('result', 'unknown')

        # Track statistics
        if result == 'hit':
            hit_count += 1
        elif result == 'miss':
            miss_count += 1

        # Print and save on important events
        save_this_step = False
        if info.get('ship_sunk'):
            ships_sunk.append(info['ship_sunk'])
            if verbose:
                print(f"Move {step_count}: [{chr(65+action_row)}{action_col}] HIT - {info['ship_sunk'].upper()} SUNK! ⚓")
            save_this_step = True
        elif result == 'hit':
            if verbose and step_count % 5 == 0:
                print(f"Move {step_count}: [{chr(65+action_row)}{action_col}] Hit")
        elif verbose and step_count % 10 == 0:
            print(f"Move {step_count}: [{chr(65+action_row)}{action_col}] Miss")

        # Save snapshots at key moments
        if save_snapshots and (save_this_step or step_count == 1 or terminated):
            snapshot_html = render_probability_html(env.state, agent, action)
            snapshot_path = game_dir / f"move_{step_count:03d}.html"
            with open(snapshot_path, 'w') as f:
                f.write(snapshot_html)

        done = terminated or truncated

    # Final results
    if verbose:
        print()
        print("=" * 70)
        if terminated:
            print(f"🎉 VICTORY! Game completed in {step_count} moves")
        else:
            print(f"Game stopped after {step_count} moves")
        print("=" * 70)
        print(f"Total moves: {step_count}")
        print(f"Hits: {hit_count}")
        print(f"Misses: {miss_count}")
        print(f"Accuracy: {hit_count / step_count * 100:.1f}%")
        print(f"Total reward: {total_reward}")
        print(f"Ships sunk: {len(ships_sunk)}/{info.get('total_ships', 0)}")
        if ships_sunk:
            print(f"Order: {' → '.join([s.title() for s in ships_sunk])}")
        print("=" * 70)

    # Save final board
    if save_snapshots:
        final_html = render_probability_html(env.state, agent)
        final_path = game_dir / "final.html"
        with open(final_path, 'w') as f:
            f.write(final_html)

        if verbose:
            print(f"\n📁 Snapshots saved to: {game_dir.absolute()}")
            print()

    env.close()

    return {
        'seed': seed,
        'moves': step_count,
        'hits': hit_count,
        'misses': miss_count,
        'accuracy': hit_count / step_count,
        'total_reward': total_reward,
        'ships_sunk': len(ships_sunk),
        'victory': terminated,
        'ships_order': ships_sunk
    }


def run_multiple_games(num_games: int = 10, start_seed: int = 100) -> None:
    """
    Run multiple games and show statistics.

    Args:
        num_games: Number of games to run
        start_seed: Starting seed value
    """
    print("=" * 70)
    print(f"Running {num_games} games with Probability Agent")
    print("=" * 70)
    print()

    results = []

    for i in range(num_games):
        seed = start_seed + i
        result = run_single_game(seed=seed, verbose=False, save_snapshots=False)
        results.append(result)

        print(f"Game {i+1:2d}: {result['moves']:2d} moves - "
              f"{result['accuracy']*100:5.1f}% accuracy - "
              f"{'✓ Victory' if result['victory'] else '✗ Failed'}")

    # Compute statistics
    print()
    print("=" * 70)
    print("Summary Statistics")
    print("=" * 70)

    moves = [r['moves'] for r in results]
    accuracies = [r['accuracy'] * 100 for r in results]
    victories = sum(1 for r in results if r['victory'])

    print(f"Games played: {num_games}")
    print(f"Victories: {victories}/{num_games} ({victories/num_games*100:.1f}%)")
    print()
    print(f"Moves to completion:")
    print(f"  Min:    {min(moves):2d}")
    print(f"  Max:    {max(moves):2d}")
    print(f"  Mean:   {sum(moves)/len(moves):5.1f}")
    print(f"  Median: {sorted(moves)[len(moves)//2]:2d}")
    print()
    print(f"Accuracy:")
    print(f"  Min:  {min(accuracies):5.1f}%")
    print(f"  Max:  {max(accuracies):5.1f}%")
    print(f"  Mean: {sum(accuracies)/len(accuracies):5.1f}%")
    print()
    print("DataGenetics Expected Performance:")
    print("  Median: ~42 shots")
    print("  Maximum: ~73 shots")
    print("=" * 70)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Demo of DataGenetics Probability Agent"
    )
    parser.add_argument(
        '--mode',
        choices=['single', 'multiple'],
        default='single',
        help='Run mode: single game or multiple games'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for single game mode'
    )
    parser.add_argument(
        '--num-games',
        type=int,
        default=10,
        help='Number of games for multiple mode'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save HTML snapshots (single game mode only)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Output directory for snapshots'
    )

    args = parser.parse_args()

    if args.mode == 'single':
        run_single_game(
            seed=args.seed,
            verbose=True,
            save_snapshots=args.save,
            output_dir=args.output_dir
        )
    else:
        run_multiple_games(
            num_games=args.num_games,
            start_seed=100
        )


if __name__ == "__main__":
    main()
