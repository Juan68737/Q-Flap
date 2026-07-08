"""Double-DQN agent.

Holds all the algorithm logic (action selection + the update rule) so the
training loop stays dumb. The loss is a *pure* function of
(batch, q_net, target_net, gamma) so it can be unit-tested on a hand-made batch.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .base import Agent
from ..models.networks import QNet


def dqn_loss(batch, q_net, target_net, gamma: float):
    """Double-DQN smooth-L1 loss. Pure: no reliance on external/global state."""
    s, a, r, sn, d = batch
    with torch.no_grad():
        next_actions = q_net(sn).argmax(1, keepdim=True)        # action from online net
        next_q = target_net(sn).gather(1, next_actions).squeeze(1)  # value from target net
        target = r + gamma * next_q * (~d)
    q_sa = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
    return F.smooth_l1_loss(q_sa, target)


class DQNAgent(Agent):
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.q = QNet(cfg.state_dim, cfg.hidden, cfg.n_actions).to(device)
        self.tgt = QNet(cfg.state_dim, cfg.hidden, cfg.n_actions).to(device)
        self.tgt.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.train_steps = 0

    # ---- action selection ----
    @torch.no_grad()
    def act(self, obs: np.ndarray, eps: float = 0.0) -> int:
        if np.random.rand() < eps:
            return int(np.random.randint(0, self.cfg.n_actions))
        s = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return int(self.q(s).argmax(1).item())

    @torch.no_grad()
    def act_batch(self, states, eps: float) -> np.ndarray:
        """Per-env epsilon-greedy over a batch of states (used by the vec env)."""
        s = torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device)
        greedy = self.q(s).argmax(1).detach().cpu().numpy().astype(np.int64)
        rand_actions = np.random.randint(0, self.cfg.n_actions, size=len(states), dtype=np.int64)
        explore = np.random.rand(len(states)) < eps
        return np.where(explore, rand_actions, greedy)

    # ---- learning ----
    def update(self, batch) -> float:
        loss = dqn_loss(batch, self.q, self.tgt, self.cfg.gamma)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.grad_clip_norm)
        self.opt.step()
        self._soft_update()
        self.train_steps += 1
        return float(loss.item())

    @torch.no_grad()
    def _soft_update(self):
        tau = self.cfg.target_tau
        for tp, p in zip(self.tgt.parameters(), self.q.parameters()):
            tp.data.mul_(1.0 - tau).add_(tau * p.data)

    # ---- persistence (policy weights only; resume state lives in utils/checkpoint) ----
    def save(self, path) -> None:
        torch.save(self.q.state_dict(), path)

    def load(self, path) -> None:
        sd = torch.load(path, map_location=self.device)
        self.q.load_state_dict(sd)
        self.tgt.load_state_dict(self.q.state_dict())
