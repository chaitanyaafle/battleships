"""Train a PPO agent with action masking to play Battleship.

This version uses sb3-contrib's MaskablePPO which prevents the agent
from selecting invalid actions (already-attacked cells).
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
from datetime import datetime
import yaml
import numpy as np
import torch

try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
except ImportError:
    print("Error: sb3-contrib not installed!")
    print("Install with: pip install sb3-contrib")
    exit(1)

import wandb
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from game.env import BattleshipEnv


def mask_fn(env) -> np.ndarray:
    """
    Generate action mask for the current environment state.

    FORCED TARGET MODE: When adjacent cells to unsunk hits exist,
    ONLY those cells are valid actions. This removes the need to
    learn target mode behavior - it's enforced by the mask.

    Args:
        env: Battleship environment

    Returns:
        Boolean array where True = valid action
    """
    if env.state is None:
        return np.ones(env.action_space.n, dtype=bool)

    # Check for adjacent cells to unsunk hits (target mode)
    adjacent_cells = env._get_valid_adjacent_cells()

    if adjacent_cells:
        # FORCE target mode: only allow adjacent cells
        mask = np.zeros(env.action_space.n, dtype=bool)
        for action in adjacent_cells:
            mask[action] = True
        return mask

    # Hunt mode: allow all unattacked cells
    attack_board = env.state.attack_board
    return (attack_board == 0).flatten()


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class EpisodeStatsCallback(BaseCallback):
    """
    Custom callback to log additional episode statistics to wandb.
    Logs min, max, and std of episode lengths, rewards, and adjacency exploitation.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_lengths = []
        self.episode_rewards = []
        self.adjacency_rates = []  # Track how often agent exploits adjacency opportunities

    def _on_step(self) -> bool:
        # Check if episode ended
        if self.locals.get("dones") is not None:
            done = self.locals["dones"][0] if isinstance(self.locals["dones"], np.ndarray) else self.locals["dones"]

            if done:
                # Get episode info from Monitor wrapper
                info = self.locals.get("infos", [{}])[0]

                if "episode" in info:
                    ep_len = info["episode"]["l"]
                    ep_rew = info["episode"]["r"]

                    self.episode_lengths.append(ep_len)
                    self.episode_rewards.append(ep_rew)

                    # Track adjacency exploitation rate
                    if "adjacency_rate" in info:
                        self.adjacency_rates.append(info["adjacency_rate"])

                    # Log to wandb every episode
                    if len(self.episode_lengths) >= 1:
                        log_dict = {
                            "rollout/ep_len_min": min(self.episode_lengths[-100:]),  # Min over last 100 episodes
                            "rollout/ep_len_max": max(self.episode_lengths[-100:]),
                            "rollout/ep_len_std": np.std(self.episode_lengths[-100:]),
                            "rollout/ep_rew_min": min(self.episode_rewards[-100:]),
                            "rollout/ep_rew_max": max(self.episode_rewards[-100:]),
                            "rollout/ep_rew_std": np.std(self.episode_rewards[-100:]),
                        }

                        # Add adjacency metrics if available
                        if len(self.adjacency_rates) > 0:
                            log_dict["rollout/adjacency_rate_mean"] = np.mean(self.adjacency_rates[-100:])
                            log_dict["rollout/adjacency_rate_min"] = min(self.adjacency_rates[-100:])
                            log_dict["rollout/adjacency_rate_max"] = max(self.adjacency_rates[-100:])

                        wandb.log(log_dict, step=self.num_timesteps)

        return True


def main():
    print("Training PPO with Action Masking...")
    parser = argparse.ArgumentParser(description="Train Masked PPO agent for Battleship")
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--timesteps', type=int, default=None)
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None, help='Path to model to resume training from')
    parser.add_argument('--env-verbose', action='store_true', help='Show adjacency bonus and escalating penalty messages')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    if args.timesteps:
        config['training']['total_timesteps'] = args.timesteps

    # Generate experiment name
    if args.name:
        exp_name = args.name
    elif config['logging']['experiment_name']:
        exp_name = config['logging']['experiment_name']
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"ppo_masked_{timestamp}"

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
                'algorithm': 'MaskablePPO',
                **config['environment'],
                **config['training'],
                **config['ppo']
            },
            sync_tensorboard=True,
        )

    # Create environment
    env = BattleshipEnv(
        board_size=tuple(config['environment']['board_size']),
        render_mode=config['environment']['render_mode'],
        verbose=args.env_verbose
    )

    # Wrap with action masker
    env = ActionMasker(env, mask_fn)

    # Validate environment
    print("Validating environment...")
    check_env(env.env, warn=True)  # Check unwrapped env
    print("✓ Environment validated!")

    # Wrap with Monitor
    env = Monitor(env, str(log_dir / "train"))

    # Create eval environment
    eval_env = BattleshipEnv(
        board_size=tuple(config['environment']['board_size']),
        verbose=args.env_verbose
    )
    eval_env = ActionMasker(eval_env, mask_fn)
    eval_env = Monitor(eval_env, str(log_dir / "eval"))

    # Extract PPO hyperparameters
    ppo_config = config['ppo'].copy()
    policy_kwargs = ppo_config.pop('policy_kwargs', {})

    if 'activation_fn' in policy_kwargs:
        if policy_kwargs['activation_fn'] == 'relu':
            import torch.nn as nn
            policy_kwargs['activation_fn'] = nn.ReLU
        elif policy_kwargs['activation_fn'] == 'tanh':
            import torch.nn as nn
            policy_kwargs['activation_fn'] = nn.Tanh

    # Determine device (use MPS for M1/M2/M3 Macs)
    if torch.backends.mps.is_available():
        device = "mps"
        print("🚀 Using M3 GPU (Metal Performance Shaders)")
    elif torch.cuda.is_available():
        device = "cuda"
        print("🚀 Using CUDA GPU")
    else:
        device = "cpu"
        print("⚠️  Using CPU (no GPU acceleration)")

    # Create or load Maskable PPO agent
    if args.resume:
        print(f"\n📂 Loading model from {args.resume}")
        model = MaskablePPO.load(
            args.resume,
            env=env,
            device=device,
            tensorboard_log=str(log_dir) if config['logging']['use_tensorboard'] else None,
        )
        print("✓ Model loaded! Continuing training...")
    else:
        print(f"\nInitializing MaskablePPO agent with config:")
        print(f"  Device: {device}")
        print(f"  Learning rate: {ppo_config['learning_rate']}")
        print(f"  Batch size: {ppo_config['batch_size']}")
        print(f"  Network architecture: {policy_kwargs.get('net_arch', 'default')}")
        print(f"  ✨ Action masking: ENABLED")

        model = MaskablePPO(
            ppo_config['policy'],
            env,
            verbose=config['training']['verbose'],
            tensorboard_log=str(log_dir) if config['logging']['use_tensorboard'] else None,
            policy_kwargs=policy_kwargs,
            device=device,
            **{k: v for k, v in ppo_config.items() if k != 'policy'}
        )

    # Setup callbacks
    callbacks = []

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=config['training']['save_freq'],
        save_path=str(save_dir / "checkpoints"),
        name_prefix="ppo_masked_checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    callbacks.append(checkpoint_callback)

    # Episode stats callback (logs min/max/std to wandb)
    if use_wandb:
        episode_stats_callback = EpisodeStatsCallback(verbose=0)
        callbacks.append(episode_stats_callback)

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
            progress_bar=True
        )

        # Save final model
        final_model_path = save_dir / "final_model"
        model.save(final_model_path)
        print(f"\n✓ Training completed! Final model saved to {final_model_path}")

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
