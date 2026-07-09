"""Proportional prioritized experience replay (Schaul et al. 2015).

Storage is the same pre-allocated ring as the uniform buffer; a SumTree over
priority^alpha drives sampling and a MinTree gives the max importance-sampling
weight for normalization. New transitions enter at the current max priority so
they're seen at least once. Priorities are refreshed from |TD error| after each
update. Importance-sampling weights (annealed by beta) correct the bias.
"""

import numpy as np
import torch

from .sumtree import SumTree, MinTree


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, device, alpha: float, eps: float):
        self.cap = capacity
        self.device = device
        self.alpha = alpha
        self.eps = eps
        self.s = np.zeros((capacity, state_dim), dtype=np.float32)
        self.a = np.zeros((capacity,), dtype=np.int64)
        self.r = np.zeros((capacity,), dtype=np.float32)
        self.sn = np.zeros((capacity, state_dim), dtype=np.float32)
        self.d = np.zeros((capacity,), dtype=np.bool_)
        self.idx = 0
        self.full = False
        self.sum = SumTree(capacity)
        self.min = MinTree(capacity)
        self.max_priority = 1.0

    def __len__(self):
        return self.cap if self.full else self.idx

    def push(self, s, a, r, sn, d):
        i = self.idx
        self.s[i] = s
        self.a[i] = a
        self.r[i] = r
        self.sn[i] = sn
        self.d[i] = d
        p = self.max_priority ** self.alpha       # new samples enter at max priority
        self.sum.update(i, p)
        self.min.update(i, p)
        self.idx = (self.idx + 1) % self.cap
        self.full = self.full or self.idx == 0

    def sample(self, batch_size: int, beta: float):
        """Return (batch tensors, indices, IS weights). Sample proportional to priority."""
        n = len(self)
        total = self.sum.total
        # stratified sampling: one draw per equal-width segment of the total mass
        segment = total / batch_size
        idxs = np.empty(batch_size, dtype=np.int64)
        for k in range(batch_size):
            u = (k + np.random.random()) * segment
            idxs[k] = self.sum.sample(u)

        # importance-sampling weights: w_i = (N * P(i))^-beta, normalized by the max
        probs = self.sum.tree[idxs + self.cap] / total
        weights = (n * probs) ** (-beta)
        min_prob = self.min.min / total
        max_weight = (n * min_prob) ** (-beta)
        weights = (weights / max_weight).astype(np.float32)

        def t(arr, dt):
            return torch.as_tensor(arr[idxs], dtype=dt, device=self.device)

        batch = (
            t(self.s, torch.float32),
            t(self.a, torch.long),
            t(self.r, torch.float32),
            t(self.sn, torch.float32),
            t(self.d, torch.bool),
        )
        w = torch.as_tensor(weights, dtype=torch.float32, device=self.device)
        return batch, idxs, w

    def update_priorities(self, idxs, td_errors):
        """Refresh priorities from |TD error| (+eps) after a learning step."""
        prios = np.abs(td_errors) + self.eps
        self.max_priority = max(self.max_priority, float(prios.max()))
        pa = prios ** self.alpha
        for i, p in zip(idxs, pa):
            self.sum.update(int(i), float(p))
            self.min.update(int(i), float(p))
