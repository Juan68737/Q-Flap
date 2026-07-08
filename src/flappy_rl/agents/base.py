"""Abstract agent interface.

Every algorithm (DQN today, PPO tomorrow) implements this so the training and
evaluation scripts stay algorithm-agnostic. Adding a new algorithm should mean
writing a new `Agent` subclass and touching nothing else.
"""

from abc import ABC, abstractmethod

import numpy as np


class Agent(ABC):
    @abstractmethod
    def act(self, obs: np.ndarray, eps: float = 0.0) -> int:
        """Return an action for a single observation (eps=0 => greedy)."""

    @abstractmethod
    def update(self, batch) -> float:
        """Run one learning step on a sampled batch; return the loss."""

    @abstractmethod
    def save(self, path) -> None:
        """Persist the policy weights."""

    @abstractmethod
    def load(self, path) -> None:
        """Load policy weights saved by `save`."""
