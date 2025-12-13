"""Train a DQN agent to play Battleship."""

import argparse
from datetime import datetime
from pathlib import Path
import yaml
import numpy as np

import wandb
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList
)
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from game.env import BattleshipEnv


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dict
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def evaluate_against_baselines(
    model,
    env,
    n_episodes: int = 100,
    verbose: bool = True
) -> dict:
    """
    Evaluate trained model against baseline agents.

    Args:
        model: Trained DQN model
        env: Battleship environment
        n_episodes: Number of evaluation episodes
        verbose: Print results

    Returns:
        Dictionary with evaluation metrics
    """
    episode_lengths = []
    episode_rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1

        episode_lengths.append(episode_length)
        episode_rewards.append(episode_reward)

    results = {
        'mean_episode_length': np.mean(episode_lengths),
        'median_episode_length': np.median(episode_lengths),
        'std_episode_length': np.std(episode_lengths),
        'min_episode_length': np.min(episode_lengths),
        'max_episode_length': np.max(episode_lengths),
        'mean_episode_reward': np.mean(episode_rewards),
        'std_episode_reward': np.std(episode_rewards),
    }

    if verbose:
        print("\n" + "="*60)
        print("EVALUATION RESULTS (vs Optimal Ship Placement)")
        print("="*60)
        print(f"Episodes: {n_episodes}")
        print(f"Mean episode length: {results['mean_episode_length']:.1f} moves")
        print(f"Median episode length: {results['median_episode_length']:.1f} moves")
        print(f"Std: {results['std_episode_length']:.1f}")
        print(f"Min: {results['min_episode_length']:.0f}, Max: {results['max_episode_length']:.0f}")
        print(f"Mean reward: {results['mean_episode_reward']:.1f}")
        print("\nBaseline comparisons:")
        print("  Random agent: ~96 moves median")
        print("  Probability agent: ~49 moves median (near-optimal)")
        print("\nNote: DQN may struggle with sparse rewards (see research plan)")
        print("="*60)

    return results


def main():
    parser = argparse.ArgumentParser(description="Train DQN agent for Battleship")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable Weights & Biases logging'
    )
    parser.add_argument(
        '--timesteps',
        type=int,
        default=None,
        help='Override total timesteps from config'
    )
    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='Experiment name for W&B'
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override config with command-line args
    if args.timesteps:
        config['training']['total_timesteps'] = args.timesteps

    # Generate experiment name
    if args.name:
        exp_name = args.name
    elif config['logging']['experiment_name']:
        exp_name = config['logging']['experiment_name']
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"dqn_{timestamp}"

    # Create directories
    log_dir = Path(config['logging']['log_dir']) / exp_name
    save_dir = Path(config['logging']['save_dir']) / exp_name
    log_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Initialize W&B
    use_wandb = config['logging']['use_wandb'] and not args.no_wandb
    if use_wandb:
        wandb.init(
            project=config['logging']['project_name'],
            name=exp_name,
            config={
                'algorithm': 'DQN',
                **config['environment'],
                **config['training'],
                **config['dqn']
            },
            sync_tensorboard=True,  # Sync TensorBoard logs to W&B
        )

    # Create environment
    env = BattleshipEnv(
        board_size=tuple(config['environment']['board_size']),
        render_mode=config['environment']['render_mode']
    )

    # Validate environment
    print("Validating environment...")
    check_env(env, warn=True)
    print("✓ Environment validated!")

    # Wrap environment with Monitor for logging
    env = Monitor(env, str(log_dir / "train"))

    # Create eval environment
    eval_env = BattleshipEnv(board_size=tuple(config['environment']['board_size']))
    eval_env = Monitor(eval_env, str(log_dir / "eval"))

    # Extract DQN hyperparameters
    dqn_config = config['dqn'].copy()

    # Handle policy_kwargs separately
    policy_kwargs = dqn_config.pop('policy_kwargs', {})
    # Convert activation_fn string to actual function if needed
    if 'activation_fn' in policy_kwargs:
        if policy_kwargs['activation_fn'] == 'relu':
            import torch.nn as nn
            policy_kwargs['activation_fn'] = nn.ReLU
        elif policy_kwargs['activation_fn'] == 'tanh':
            import torch.nn as nn
            policy_kwargs['activation_fn'] = nn.Tanh

    # Create DQN agent
    print(f"\nInitializing DQN agent with config:")
    print(f"  Learning rate: {dqn_config['learning_rate']}")
    print(f"  Buffer size: {dqn_config['buffer_size']:,}")
    print(f"  Exploration fraction: {dqn_config['exploration_fraction']}")
    print(f"  Network architecture: {policy_kwargs.get('net_arch', 'default')}")
    print("\nNote: DQN may struggle with Battleship's sparse rewards (see research plan)")
    print("Consider reward shaping or PPO if performance is poor.\n")

    model = DQN(
        dqn_config['policy'],
        env,
        verbose=config['training']['verbose'],
        tensorboard_log=str(log_dir) if config['logging']['use_tensorboard'] else None,
        policy_kwargs=policy_kwargs,
        **{k: v for k, v in dqn_config.items() if k != 'policy'}
    )

    # Setup callbacks
    callbacks = []

    # # Evaluation callback
    # eval_callback = EvalCallback(
    #     eval_env,
    #     best_model_save_path=str(save_dir / "best_model"),
    #     log_path=str(log_dir / "eval"),
    #     eval_freq=config['training']['eval_freq'],
    #     n_eval_episodes=config['training']['n_eval_episodes'],
    #     deterministic=True,
    #     render=False,
    #     verbose=1
    # )
    # callbacks.append(eval_callback)

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=config['training']['save_freq'],
        save_path=str(save_dir / "checkpoints"),
        name_prefix="dqn_checkpoint",
        save_replay_buffer=True,  # DQN benefits from saving replay buffer
        save_vecnormalize=False,
    )
    callbacks.append(checkpoint_callback)

    callback_list = CallbackList(callbacks)

    # Train the agent
    print(f"\nStarting training for {config['training']['total_timesteps']:,} timesteps...")
    print(f"Experiment: {exp_name}")
    print(f"Logs: {log_dir}")
    print(f"Models: {save_dir}")

    try:
        model.learn(
            total_timesteps=config['training']['total_timesteps'],
            callback=callback_list,
            log_interval=10,
            progress_bar=True
        )

        # Save final model
        final_model_path = save_dir / "final_model"
        model.save(final_model_path)
        print(f"\n✓ Training completed! Final model saved to {final_model_path}")

        # # Final evaluation
        # print("\nRunning final evaluation...")
        # results = evaluate_against_baselines(
        #     model,
        #     eval_env,
        #     n_episodes=config['training']['n_eval_episodes'],
        #     verbose=True
        # )

        # Log to W&B
        if use_wandb:
            wandb.log({
                'final_eval/mean_episode_length': results['mean_episode_length'],
                'final_eval/median_episode_length': results['median_episode_length'],
                'final_eval/mean_episode_reward': results['mean_episode_reward'],
            })

        # Save config alongside model
        config_save_path = save_dir / "config.yaml"
        with open(config_save_path, 'w') as f:
            yaml.dump(config, f)

        print(f"\n✓ All done! Model and config saved to {save_dir}")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        interrupted_model_path = save_dir / "interrupted_model"
        model.save(interrupted_model_path)
        print(f"Saving interrupted model to {interrupted_model_path}")

    finally:
        if use_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
