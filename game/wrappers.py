"""Gymnasium wrappers for Battleship environment."""

import gymnasium as gym
import numpy as np
from typing import Dict, Tuple, Optional


class ActionMaskWrapper(gym.Wrapper):
    """
    Wrapper that masks invalid actions (already attacked cells).

    This modifies the agent's action selection to prevent selecting
    cells that have already been attacked, making training much more efficient.
    """

    def __init__(self, env):
        """
        Initialize action mask wrapper.

        Args:
            env: Battleship environment to wrap
        """
        super().__init__(env)

    def get_action_mask(self) -> np.ndarray:
        """
        Get boolean mask of valid actions.

        Returns:
            Boolean array where True = valid action, False = invalid
        """
        if self.env.state is None:
            # No game in progress, all actions valid
            return np.ones(self.action_space.n, dtype=bool)

        # Valid actions are cells that haven't been attacked yet
        attack_board = self.env.state.attack_board
        valid_mask = (attack_board == 0).flatten()

        return valid_mask

    def step(self, action: int):
        """Step with the given action."""
        return self.env.step(action)

    def reset(self, **kwargs):
        """Reset the environment."""
        return self.env.reset(**kwargs)


class InvalidActionPenaltyWrapper(gym.Wrapper):
    """
    Wrapper that gives increasingly harsh penalties for repeated invalid actions.

    This helps the agent learn faster that invalid actions are very bad.
    """

    def __init__(self, env, base_penalty: float = -50.0, escalation: float = 2.0):
        """
        Initialize wrapper.

        Args:
            env: Environment to wrap
            base_penalty: Base penalty for invalid action
            escalation: Multiplier for consecutive invalid actions
        """
        super().__init__(env)
        self.base_penalty = base_penalty
        self.escalation = escalation
        self.consecutive_invalid = 0

    def step(self, action: int):
        """Step with escalating penalties for invalid actions."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Check if action was invalid
        if info.get('result') == 'invalid':
            self.consecutive_invalid += 1
            # Escalating penalty
            penalty = self.base_penalty * (self.escalation ** (self.consecutive_invalid - 1))
            reward = max(penalty, -200.0)  # Cap at -200
            info['escalated_penalty'] = penalty
        else:
            self.consecutive_invalid = 0

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset the environment and counter."""
        self.consecutive_invalid = 0
        return self.env.reset(**kwargs)


class FlattenDictObservationWrapper(gym.ObservationWrapper):
    """
    Flatten Dict observation space to single Box for simpler algorithms.

    WARNING: This loses spatial structure! Only use for simple baselines.
    """

    def __init__(self, env):
        """Initialize wrapper."""
        super().__init__(env)

        # Calculate flattened observation size
        total_size = 0
        for key, space in self.observation_space.spaces.items():
            total_size += np.prod(space.shape)

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_size,),
            dtype=np.float32
        )

    def observation(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """Flatten dict observation to array."""
        parts = []
        for key in sorted(observation.keys()):
            parts.append(observation[key].flatten())
        return np.concatenate(parts).astype(np.float32)
