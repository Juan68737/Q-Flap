"""Uniform experience-replay buffer (pre-allocated NumPy arrays).

Stores (s, a, r, s', done) transitions in ring buffers and samples uniform
minibatches as torch tensors on the target device.
"""

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, device):
        self.cap = capacity
        self.device = device
        self.s = np.zeros((capacity, state_dim), dtype=np.float32)
        self.a = np.zeros((capacity,), dtype=np.int64)
        self.r = np.zeros((capacity,), dtype=np.float32)
        self.sn = np.zeros((capacity, state_dim), dtype=np.float32)
        self.d = np.zeros((capacity,), dtype=np.bool_)
        self.idx = 0
        self.full = False

    def __len__(self):
        return self.cap if self.full else self.idx

    def push(self, s, a, r, sn, d):
        i = self.idx
        self.s[i] = s
        self.a[i] = a
        self.r[i] = r
        self.sn[i] = sn
        self.d[i] = d
        self.idx = (self.idx + 1) % self.cap
        self.full = self.full or self.idx == 0

    def sample(self, batch_size: int):
        m = len(self)
        idxs = np.random.randint(0, m, size=batch_size)
        t = lambda arr, dt: torch.as_tensor(arr[idxs], dtype=dt, device=self.device)
        return (
            t(self.s, torch.float32),
            t(self.a, torch.long),
            t(self.r, torch.float32),
            t(self.sn, torch.float32),
            t(self.d, torch.bool),
        )
