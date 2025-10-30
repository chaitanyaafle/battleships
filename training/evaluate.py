"""Evaluate and compare Battleship agents."""

import argparse
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import pandas as pd

from game.env import BattleshipEnv
from game.agents.base import BattleshipAgent
from game.agents.random_agent import RandomAgent
from game.agents.probability_agent import ProbabilityAgent
from game.agents.rl_agent import RLAgent


def evaluate_agent(
    agent: BattleshipAgent,
    env: BattleshipEnv,
    n_episodes: int = 100,
    seed: Optional[int] = None,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Evaluate a single agent.

    Args:
        agent: Agent to evaluate
        env: Environment
        n_episodes: Number of episodes to run
        seed: Random seed for reproducibility
        verbose: Print episode-by-episode results

    Returns:
        Dictionary with evaluation metrics
    """
    episode_lengths = []
    episode_rewards = []
    wins = 0

    if seed is not None:
        np.random.seed(seed)

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep if seed is not None else None)
        agent.reset()

        done = False
        episode_reward = 0
        episode_length = 0

        while not done:
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1

        episode_lengths.append(episode_length)
        episode_rewards.append(episode_reward)

        # Check if won (all ships sunk)
        if info.get('all_sunk', False):
            wins += 1

        if verbose and (ep + 1) % 10 == 0:
            print(f"  Episode {ep + 1}/{n_episodes}: {episode_length} moves, reward={episode_reward:.1f}")

    return {
        'agent_name': agent.name,
        'n_episodes': n_episodes,
        'mean_length': np.mean(episode_lengths),
        'median_length': np.median(episode_lengths),
        'std_length': np.std(episode_lengths),
        'min_length': np.min(episode_lengths),
        'max_length': np.max(episode_lengths),
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'win_rate': wins / n_episodes,
    }


def compare_agents(
    agents: List[BattleshipAgent],
    env: BattleshipEnv,
    n_episodes: int = 100,
    seed: Optional[int] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compare multiple agents.

    Args:
        agents: List of agents to evaluate
        env: Environment
        n_episodes: Number of episodes per agent
        seed: Random seed
        verbose: Print progress

    Returns:
        DataFrame with comparison results
    """
    results = []

    for agent in agents:
        if verbose:
            print(f"\nEvaluating {agent.name}...")

        result = evaluate_agent(agent, env, n_episodes, seed, verbose)
        results.append(result)

        if verbose:
            print(f"  Results: {result['mean_length']:.1f} ± {result['std_length']:.1f} moves (median: {result['median_length']:.1f})")

    df = pd.DataFrame(results)
    return df


def print_comparison_table(df: pd.DataFrame):
    """
    Print a formatted comparison table.

    Args:
        df: DataFrame with agent comparison results
    """
    print("\n" + "="*80)
    print("AGENT COMPARISON")
    print("="*80)
    print(f"{'Agent':<25} {'Mean':<10} {'Median':<10} {'Std':<10} {'Min':<8} {'Max':<8} {'Win %':<8}")
    print("-"*80)

    for _, row in df.iterrows():
        print(f"{row['agent_name']:<25} "
              f"{row['mean_length']:<10.1f} "
              f"{row['median_length']:<10.1f} "
              f"{row['std_length']:<10.1f} "
              f"{row['min_length']:<8.0f} "
              f"{row['max_length']:<8.0f} "
              f"{row['win_rate']*100:<8.1f}")

    print("="*80)

    # Find best agent
    best_idx = df['median_length'].idxmin()
    best_agent = df.loc[best_idx]
    print(f"\n🏆 Best agent (by median): {best_agent['agent_name']} ({best_agent['median_length']:.1f} moves)")

    # Comparison to baselines
    print("\n📊 Baseline comparisons:")
    random_row = df[df['agent_name'].str.contains('Random', case=False)]
    prob_row = df[df['agent_name'].str.contains('Probability', case=False)]

    if not random_row.empty:
        print(f"   Random baseline: {random_row.iloc[0]['median_length']:.1f} moves")
    else:
        print("   Random baseline: ~96 moves (from research)")

    if not prob_row.empty:
        print(f"   Probability baseline: {prob_row.iloc[0]['median_length']:.1f} moves")
    else:
        print("   Probability baseline: ~49 moves (near-optimal)")

    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Battleship agents")
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        help='Paths to trained models to evaluate'
    )
    parser.add_argument(
        '--baselines',
        action='store_true',
        help='Include baseline agents (Random, Probability)'
    )
    parser.add_argument(
        '--n-episodes',
        type=int,
        default=100,
        help='Number of evaluation episodes per agent'
    )
    parser.add_argument(
        '--board-size',
        type=int,
        nargs=2,
        default=[10, 10],
        help='Board size (rows cols)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Save results to CSV file'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress'
    )
    args = parser.parse_args()

    # Create environment
    env = BattleshipEnv(board_size=tuple(args.board_size))

    # Collect agents to evaluate
    agents = []

    # Add baseline agents
    if args.baselines:
        agents.append(RandomAgent("Random Agent"))
        agents.append(ProbabilityAgent("Probability Agent (DataGenetics)"))

    # Add trained models
    if args.models:
        for model_path in args.models:
            model_path = Path(model_path)
            if not model_path.exists():
                print(f"Warning: Model not found: {model_path}")
                continue

            # Try to infer agent name from path
            if 'ppo' in str(model_path).lower():
                agent_name = f"PPO ({model_path.parent.name})"
            elif 'dqn' in str(model_path).lower():
                agent_name = f"DQN ({model_path.parent.name})"
            else:
                agent_name = f"RL ({model_path.parent.name})"

            try:
                agent = RLAgent(
                    name=agent_name,
                    model_path=model_path,
                    deterministic=True
                )
                agents.append(agent)
                print(f"✓ Loaded {agent_name}")
            except Exception as e:
                print(f"Error loading {model_path}: {e}")

    if not agents:
        print("Error: No agents to evaluate!")
        print("Use --baselines to evaluate baseline agents or --models to specify trained models")
        return

    # Run comparison
    print(f"\nEvaluating {len(agents)} agents over {args.n_episodes} episodes each...")
    print(f"Board size: {args.board_size[0]}x{args.board_size[1]}")
    print(f"Random seed: {args.seed}")

    results_df = compare_agents(
        agents,
        env,
        n_episodes=args.n_episodes,
        seed=args.seed,
        verbose=args.verbose
    )

    # Print results
    print_comparison_table(results_df)

    # Save to CSV if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        print(f"\n✓ Results saved to {output_path}")


if __name__ == "__main__":
    main()
