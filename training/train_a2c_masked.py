"""Train an A2C agent with action masking to play Battleship.

Quick sanity check to see if A2C's simpler, single-pass updates help discover
the adjacency bonus where PPO failed.
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
    from sb3_contrib.common.wrappers import ActionMasker
except ImportError:
    print("Error: sb3-contrib not installed!")
    print("Install with: pip install sb3-contrib")
    exit(1)

import wandb
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from game.env import BattleshipEnv


def mask_fn(env) -> np.ndarray:
    """
    Generate action mask for the current environment state.

    Args:
        env: Battleship environment

    Returns:
        Boolean array where True = valid action
    """
    if env.state is None:
        return np.ones(env.action_space.n, dtype=bool)

    # Valid actions are cells that haven't been attacked
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
    Logs min, max, and std of episode lengths and rewards.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_lengths = []
        self.episode_rewards = []

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

                    # Log to wandb every episode
                    if len(self.episode_lengths) >= 1:
                        wandb.log({
                            "rollout/ep_len_min": min(self.episode_lengths[-100:]),  # Min over last 100 episodes
                            "rollout/ep_len_max": max(self.episode_lengths[-100:]),
                            "rollout/ep_len_std": np.std(self.episode_lengths[-100:]),
                            "rollout/ep_rew_min": min(self.episode_rewards[-100:]),
                            "rollout/ep_rew_max": max(self.episode_rewards[-100:]),
                            "rollout/ep_rew_std": np.std(self.episode_rewards[-100:]),
                        }, step=self.num_timesteps)

        return True


def main():
    print("Training A2C with Action Masking (Sanity Check)...")
    parser = argparse.ArgumentParser(description="Train A2C agent for Battleship")
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--timesteps', type=int, default=50000)  # Default 50k for quick test
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None, help='Path to model to resume training from')
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
        exp_name = f"a2c_masked_{timestamp}"

    # Create directories
    log_dir = Path(config['logging']['log_dir']) / exp_name
    save_dir = Path(config['logging']['save_dir']) / exp_name
    log_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    # A2C-specific hyperparameters (hardcoded for quick test)
    a2c_config = {
        'learning_rate': 0.0007,  # Slightly higher than PPO
        'n_steps': 5,  # A2C typically uses small n_steps
        'gamma': 0.95,
        'gae_lambda': 0.95,
        'ent_coef': 0.8,  # Same high exploration as PPO
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'policy': 'MultiInputPolicy'
    }

    # Initialize W&B
    use_wandb = config['logging']['use_wandb'] and not args.no_wandb
    if use_wandb:
        wandb.init(
            project=config['logging']['project_name'],
            name=exp_name,
            config={
                'algorithm': 'A2C',
                **config['environment'],
                **a2c_config
            },
            sync_tensorboard=True,
        )

    # Create environment
    env = BattleshipEnv(
        board_size=tuple(config['environment']['board_size']),
        render_mode=config['environment']['render_mode']
    )

    # Wrap with action masker
    env = ActionMasker(env, mask_fn)

    # Validate environment
    print("Validating environment...")
    check_env(env.env, warn=True)  # Check unwrapped env
    print("✓ Environment validated!")

    # Wrap with Monitor
    env = Monitor(env, str(log_dir / "train"))

    # Network architecture (same as PPO)
    policy_kwargs = {
        'net_arch': [512, 512, 256, 128],
        'activation_fn': torch.nn.ReLU
    }

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

    # Create or load A2C agent
    if args.resume:
        print(f"\n📂 Loading model from {args.resume}")
        model = A2C.load(
            args.resume,
            env=env,
            device=device,
            tensorboard_log=str(log_dir) if config['logging']['use_tensorboard'] else None,
        )
        print("✓ Model loaded! Continuing training...")
    else:
        print(f"\nInitializing A2C agent with config:")
        print(f"  Device: {device}")
        print(f"  Learning rate: {a2c_config['learning_rate']}")
        print(f"  n_steps: {a2c_config['n_steps']} (single-pass updates)")
        print(f"  ent_coef: {a2c_config['ent_coef']} (high exploration)")
        print(f"  Network architecture: {policy_kwargs.get('net_arch', 'default')}")
        print(f"  ✨ Action masking: ENABLED")

        model = A2C(
            a2c_config['policy'],
            env,
            learning_rate=a2c_config['learning_rate'],
            n_steps=a2c_config['n_steps'],
            gamma=a2c_config['gamma'],
            gae_lambda=a2c_config['gae_lambda'],
            ent_coef=a2c_config['ent_coef'],
            vf_coef=a2c_config['vf_coef'],
            max_grad_norm=a2c_config['max_grad_norm'],
            verbose=config['training']['verbose'],
            tensorboard_log=str(log_dir) if config['logging']['use_tensorboard'] else None,
            policy_kwargs=policy_kwargs,
            device=device,
        )

    # Setup callbacks
    callbacks = []

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=config['training']['save_freq'],
        save_path=str(save_dir / "checkpoints"),
        name_prefix="a2c_masked_checkpoint",
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
