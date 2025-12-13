"""Custom Gym wrappers for Battleship environment."""

import gymnasium as gym
import numpy as np
from typing import Any, Dict, Tuple


class InvalidActionFilter(gym.Wrapper):
    """
    Wrapper that prevents invalid actions by filtering them out.

    Unlike ActionMasker (which requires algorithm support), this wrapper
    works with ANY algorithm by post-processing actions.

    If the agent selects an invalid action, this wrapper randomly picks
    a valid one instead.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.last_obs = None

    def reset(self, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset environment and store observation."""
        obs, info = self.env.reset(**kwargs)
        self.last_obs = obs
        return obs, info

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Execute action, filtering invalid actions.

        If action is invalid (already attacked cell), randomly select
        a valid action instead.
        """
        # Get valid actions from current observation
        attack_board = self.last_obs['attack_board']
        valid_mask = (attack_board == 0).flatten()
        valid_actions = np.where(valid_mask)[0]

        # If action is invalid, pick a random valid one
        if not valid_mask[action]:
            if len(valid_actions) > 0:
                action = np.random.choice(valid_actions)
                # Note: This means agent gets random action when it tries invalid move
                # Not ideal for learning, but prevents catastrophic -50 penalties

        # Execute the (possibly corrected) action
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_obs = obs

        return obs, reward, terminated, truncated, info
