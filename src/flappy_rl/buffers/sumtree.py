"""Segment trees for prioritized experience replay.

A `SumTree` supports O(log n) point-update of a priority and O(log n) sampling
proportional to priority (draw u ~ U(0, total), descend picking the child whose
subtree sum covers u). A `MinTree` gives O(log n) min for the max importance-
sampling weight. This is the standard structure behind proportional PER
(Schaul et al. 2015) — the same iterative segment tree from competitive
programming, with sums (or mins) at internal nodes.
"""

import numpy as np


class SumTree:
    """Fixed-capacity binary tree over `cap` leaves; internal nodes hold subtree sums."""

    def __init__(self, cap: int):
        self.cap = cap
        self.tree = np.zeros(2 * cap, dtype=np.float64)  # 1-indexed; leaves at [cap, 2cap)

    def update(self, i: int, p: float) -> None:
        """Set leaf i (0-indexed) to priority p and fix ancestors."""
        idx = i + self.cap
        self.tree[idx] = p
        idx >>= 1
        while idx:
            self.tree[idx] = self.tree[2 * idx] + self.tree[2 * idx + 1]
            idx >>= 1

    def sample(self, u: float) -> int:
        """Return the leaf index (0-indexed) whose cumulative range contains u in [0, total)."""
        idx = 1
        while idx < self.cap:
            left = 2 * idx
            if u < self.tree[left]:
                idx = left
            else:
                u -= self.tree[left]
                idx = left + 1
        return idx - self.cap

    @property
    def total(self) -> float:
        return float(self.tree[1])


class MinTree:
    """Fixed-capacity binary tree over `cap` leaves; internal nodes hold subtree mins."""

    def __init__(self, cap: int):
        self.cap = cap
        self.tree = np.full(2 * cap, np.inf, dtype=np.float64)

    def update(self, i: int, p: float) -> None:
        idx = i + self.cap
        self.tree[idx] = p
        idx >>= 1
        while idx:
            self.tree[idx] = min(self.tree[2 * idx], self.tree[2 * idx + 1])
            idx >>= 1

    @property
    def min(self) -> float:
        return float(self.tree[1])
