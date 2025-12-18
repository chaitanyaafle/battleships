"""Evaluate and compare Battleship agents."""

import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
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

    print(f"[DEBUG] Starting episode loop, n_episodes={n_episodes}", flush=True)
    for ep in range(n_episodes):
        print(f"[DEBUG] Episode {ep + 1} starting...", flush=True)
        if verbose:
            print(f"  Starting episode {ep + 1}/{n_episodes}...", end='', flush=True)

        print(f"[DEBUG] About to reset env...", flush=True)
        obs, _ = env.reset(seed=seed + ep if seed is not None else None)
        print(f"[DEBUG] Env reset complete", flush=True)
        agent.reset()

        done = False
        episode_reward = 0
        episode_length = 0

        print(f"[DEBUG] Starting episode loop...", flush=True)
        while not done:
            if episode_length == 0:
                print(f"[DEBUG] About to select first action...", flush=True)
            # No forced masking - evaluate agent's true learned behavior
            action = agent.select_action(obs)
            if episode_length == 0:
                print(f"[DEBUG] First action selected: {action}", flush=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1

            # Debug: print progress for very long episodes
            if verbose and episode_length % 50 == 0:
                print(f" {episode_length} moves...", end='', flush=True)

        episode_lengths.append(episode_length)
        episode_rewards.append(episode_reward)

        # Check if won (all ships sunk)
        if info.get('result') == 'win':
            wins += 1

        if verbose:
            win_status = "WIN" if info.get('result') == 'win' else "TRUNCATED" if truncated else "?"
            print(f" Done! {episode_length} moves, reward={episode_reward:.1f} [{win_status}]")

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
        print(f"\n[DEBUG] About to evaluate {agent.name}...", flush=True)
        if verbose:
            print(f"\nEvaluating {agent.name}...")

        print(f"[DEBUG] Calling evaluate_agent...", flush=True)
        result = evaluate_agent(agent, env, n_episodes, seed, verbose)
        print(f"[DEBUG] evaluate_agent returned", flush=True)
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

    # Check if board_size column exists
    has_board_size = 'board_size' in df.columns

    if has_board_size:
        print(f"{'Agent':<25} {'Board':<8} {'Mean':<10} {'Median':<10} {'Std':<10} {'Min':<8} {'Max':<8} {'Win %':<8}")
    else:
        print(f"{'Agent':<25} {'Mean':<10} {'Median':<10} {'Std':<10} {'Min':<8} {'Max':<8} {'Win %':<8}")

    print("-"*80)

    for _, row in df.iterrows():
        if has_board_size:
            print(f"{row['agent_name']:<25} "
                  f"{row['board_size']:<8} "
                  f"{row['mean_length']:<10.1f} "
                  f"{row['median_length']:<10.1f} "
                  f"{row['std_length']:<10.1f} "
                  f"{row['min_length']:<8.0f} "
                  f"{row['max_length']:<8.0f} "
                  f"{row['win_rate']*100:<8.1f}")
        else:
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


def detect_board_size_from_config(model_path: Path) -> Optional[Tuple[int, int]]:
    """
    Detect board size from model's config.yaml file.

    Args:
        model_path: Path to model file (e.g., final_model.zip)

    Returns:
        (rows, cols) tuple or None if config not found
    """
    # Look for config.yaml in the same directory as the model
    config_path = model_path.parent / "config.yaml"

    if not config_path.exists():
        return None

    try:
        import yaml
        with open(config_path, 'r') as f:
            # Use FullLoader to handle PyTorch-specific tags in saved configs
            config = yaml.load(f, Loader=yaml.FullLoader)

        board_size = config.get('environment', {}).get('board_size')
        if board_size and len(board_size) == 2:
            return tuple(board_size)
    except Exception as e:
        print(f"Warning: Could not read config from {config_path}: {e}")

    return None


def group_models_by_board_size(model_paths: List[Path]) -> Dict[Tuple[int, int], List[Path]]:
    """
    Group model paths by their board size.

    Args:
        model_paths: List of paths to model files

    Returns:
        Dictionary mapping board_size -> list of model paths
    """
    groups = {}

    for model_path in model_paths:
        board_size = detect_board_size_from_config(model_path)

        if board_size is None:
            print(f"Warning: Could not detect board size for {model_path}, skipping")
            continue

        if board_size not in groups:
            groups[board_size] = []
        groups[board_size].append(model_path)

    return groups


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
    parser.add_argument(
        '--env-verbose',
        action='store_true',
        help='Show adjacency bonus and escalating penalty messages'
    )
    parser.add_argument(
        '--auto-detect-board-size',
        action='store_true',
        help='Auto-detect board size from each model\'s config.yaml (allows mixed board sizes)'
    )
    args = parser.parse_args()

    # Check if auto-detect mode is enabled
    if args.auto_detect_board_size and args.models:
        # Auto-detection mode: group models by board size
        print("Auto-detecting board sizes from model configs...")

        model_paths = [Path(p) for p in args.models]
        model_groups = group_models_by_board_size(model_paths)

        if not model_groups:
            print("Error: No valid models found after board size detection!")
            return

        all_results = []

        # Evaluate each board size group
        for board_size, group_model_paths in sorted(model_groups.items()):
            print(f"\n{'='*80}")
            print(f"Evaluating models trained on {board_size[0]}x{board_size[1]} board")
            print(f"{'='*80}")

            # Create environment for this board size
            env = BattleshipEnv(board_size=board_size, verbose=args.env_verbose)

            # Collect agents for this board size
            agents = []

            # Add baseline agents
            if args.baselines:
                agents.append(RandomAgent(f"Random ({board_size[0]}x{board_size[1]})"))
                agents.append(ProbabilityAgent(f"Probability ({board_size[0]}x{board_size[1]})"))

            # Add trained models
            for model_path in group_model_paths:
                # Try to infer agent name from path
                if 'ppo' in str(model_path).lower():
                    agent_name = f"PPO ({model_path.parent.name})"
                elif 'dqn' in str(model_path).lower():
                    agent_name = f"DQN ({model_path.parent.name})"
                else:
                    agent_name = f"RL ({model_path.parent.name})"

                print(f"Loading model from {model_path}...", flush=True)
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
                print(f"Warning: No agents to evaluate for {board_size[0]}x{board_size[1]} board")
                continue

            # Evaluate this group
            print(f"\nEvaluating {len(agents)} agents over {args.n_episodes} episodes each...")
            print(f"Random seed: {args.seed}")

            results_df = compare_agents(
                agents,
                env,
                n_episodes=args.n_episodes,
                seed=args.seed,
                verbose=args.verbose
            )

            # Add board size column
            results_df['board_size'] = f"{board_size[0]}x{board_size[1]}"
            all_results.append(results_df)

        # Combine all results
        if not all_results:
            print("Error: No results to display!")
            return

        combined_results = pd.concat(all_results, ignore_index=True)

        # Print combined results
        print_comparison_table(combined_results)

        # Save to CSV if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            combined_results.to_csv(output_path, index=False)
            print(f"\n✓ Results saved to {output_path}")

    else:
        # Standard mode: single board size for all models
        env = BattleshipEnv(board_size=tuple(args.board_size), verbose=args.env_verbose)

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

                print(f"Loading model from {model_path}...", flush=True)
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
