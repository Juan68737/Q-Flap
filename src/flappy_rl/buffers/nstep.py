"""N-step return accumulation (per environment).

Folds the last n transitions into a single (s0, a0, R, s_n, done) tuple where
R = sum_{k=0}^{n-1} gamma^k r_k, so the pass reward propagates n steps faster.
Emits a transition once n are buffered; on episode end, flushes the tail as
shorter-horizon returns. The stored discount for the bootstrap is gamma**n
(handled in the agent), except for tail/terminal transitions whose `done` flag
zeroes the bootstrap anyway.
"""

from collections import deque


class NStep:
    def __init__(self, n: int, gamma: float):
        self.n = n
        self.gamma = gamma
        self.buf = deque()

    def _make(self):
        """Fold the current buffer into one n-step transition from its oldest entry."""
        s0, a0, _, _, _ = self.buf[0]
        R = 0.0
        for k, (_, _, r, _, _) in enumerate(self.buf):
            R += (self.gamma ** k) * r
        _, _, _, sn_last, d_last = self.buf[-1]
        return (s0, a0, R, sn_last, d_last)

    def push(self, s, a, r, sn, d):
        """Add a raw transition; return a list of ready n-step transitions to store."""
        self.buf.append((s, a, r, sn, d))
        out = []
        if d:
            # episode ended: flush every remaining start point as a shorter return
            while self.buf:
                out.append(self._make())
                self.buf.popleft()
        elif len(self.buf) >= self.n:
            out.append(self._make())
            self.buf.popleft()
        return out
