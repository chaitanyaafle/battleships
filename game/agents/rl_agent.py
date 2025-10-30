"""Reinforcement learning agent wrapper for Stable-Baselines3 models."""

from pathlib import Path
from typing import Dict, Optional, Union
import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.base_class import BaseAlgorithm

from game.agents.base import BattleshipAgent


class RLAgent(BattleshipAgent):
    """
    Wrapper for Stable-Baselines3 RL agents.

    Supports PPO, DQN, and other SB3 algorithms.
    """

    def __init__(
        self,
        name: str,
        model_path: Optional[Union[str, Path]] = None,
        model: Optional[BaseAlgorithm] = None,
        deterministic: bool = True
    ):
        """
        Initialize RL agent.

        Args:
            name: Human-readable name for the agent
            model_path: Path to saved model (for loading trained models)
            model: Pre-initialized SB3 model (for training)
            deterministic: Use deterministic policy (True for evaluation)

        Raises:
            ValueError: If neither model_path nor model is provided
        """
        super().__init__(name)

        self.deterministic = deterministic
        self.model_path = model_path

        if model is not None:
            self.model = model
        elif model_path is not None:
            self.model = self._load_model(model_path)
        else:
            raise ValueError("Must provide either model_path or model")

    def _load_model(self, model_path: Union[str, Path]) -> BaseAlgorithm:
        """
        Load model from file.

        Auto-detects algorithm type from filename or tries common types.

        Args:
            model_path: Path to saved model

        Returns:
            Loaded SB3 model

        Raises:
            ValueError: If model cannot be loaded
        """
        model_path = Path(model_path)

        if not model_path.exists():
            raise ValueError(f"Model file not found: {model_path}")

        # Try to infer algorithm from filename
        filename = model_path.stem.lower()

        try:
            if 'ppo' in filename:
                return PPO.load(model_path)
            elif 'dqn' in filename:
                return DQN.load(model_path)
            else:
                # Try PPO first (most common)
                try:
                    return PPO.load(model_path)
                except:
                    return DQN.load(model_path)
        except Exception as e:
            raise ValueError(f"Failed to load model from {model_path}: {e}")

    def select_action(self, observation: Dict[str, np.ndarray]) -> int:
        """
        Select action using the RL model.

        Args:
            observation: Current game observation

        Returns:
            action: Flattened board index
        """
        # SB3 models expect flat dict observations
        action, _states = self.model.predict(
            observation,
            deterministic=self.deterministic
        )

        # Convert numpy array to int if needed
        if isinstance(action, np.ndarray):
            action = int(action.item())
        else:
            action = int(action)

        return action

    def reset(self):
        """Reset agent state (no-op for stateless RL models)."""
        # Most SB3 models are stateless, but we keep this for consistency
        # and potential future use with recurrent policies
        pass

    def save(self, save_path: Union[str, Path]):
        """
        Save the model to disk.

        Args:
            save_path: Path to save the model
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(save_path)
        print(f"Model saved to {save_path}")

    def __repr__(self) -> str:
        model_type = type(self.model).__name__
        return f"RLAgent(name={self.name}, model={model_type}, deterministic={self.deterministic})"


def load_ppo_agent(
    model_path: Union[str, Path],
    name: str = "PPO Agent",
    deterministic: bool = True
) -> RLAgent:
    """
    Convenience function to load a PPO agent.

    Args:
        model_path: Path to saved PPO model
        name: Agent name
        deterministic: Use deterministic policy

    Returns:
        RLAgent with loaded PPO model
    """
    model = PPO.load(model_path)
    return RLAgent(name=name, model=model, deterministic=deterministic)


def load_dqn_agent(
    model_path: Union[str, Path],
    name: str = "DQN Agent",
    deterministic: bool = True
) -> RLAgent:
    """
    Convenience function to load a DQN agent.

    Args:
        model_path: Path to saved DQN model
        name: Agent name
        deterministic: Use deterministic policy

    Returns:
        RLAgent with loaded DQN model
    """
    model = DQN.load(model_path)
    return RLAgent(name=name, model=model, deterministic=deterministic)
