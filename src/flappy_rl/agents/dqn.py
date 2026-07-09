"""Double-DQN agent (with optional V2 upgrades).

Holds all the algorithm logic (action selection + the update rule) so the
training loop stays dumb. The loss is a *pure* function of the batch and the two
networks, so it can be unit-tested on a hand-made batch. V2 knobs (dueling /
LayerNorm net, n-step targets, PER importance weights, EMA deploy snapshot) are
config-gated; with them off this is the exact V1 agent.
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .base import Agent
from ..models.networks import build_qnet


def dqn_loss(batch, q_net, target_net, gamma: float, is_weights=None):
    """Double-DQN smooth-L1 loss. Returns (loss, per-sample TD error).

    Pure: no reliance on external/global state. `gamma` is the effective
    discount for the bootstrap (gamma**n_step for n-step targets). `is_weights`
    are optional importance-sampling weights (for PER).
    """
    s, a, r, sn, d = batch
    with torch.no_grad():
        next_actions = q_net(sn).argmax(1, keepdim=True)            # action from online net
        next_q = target_net(sn).gather(1, next_actions).squeeze(1)  # value from target net
        target = r + gamma * next_q * (~d)
    q_sa = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
    td = q_sa - target                                             # per-sample TD error
    per_sample = F.smooth_l1_loss(q_sa, target, reduction="none")
    loss = (is_weights * per_sample).mean() if is_weights is not None else per_sample.mean()
    return loss, td.detach()


class DQNAgent(Agent):
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.q = build_qnet(cfg).to(device)
        self.tgt = build_qnet(cfg).to(device)
        self.tgt.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.train_steps = 0
        self.gamma_n = cfg.gamma ** getattr(cfg, "n_step", 1)
        self.last_td_errors = None

        # EMA (Polyak) snapshot of the weights, used for deployment/eval only.
        self.ema = None
        if getattr(cfg, "ema_decay", 0.0) and cfg.ema_decay > 0.0:
            self.ema = copy.deepcopy(self.q)
            for p in self.ema.parameters():
                p.requires_grad_(False)
            self.ema.eval()

    # ---- action selection (behavior policy uses the online net) ----
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
    def update(self, batch, is_weights=None) -> float:
        loss, td = dqn_loss(batch, self.q, self.tgt, self.gamma_n, is_weights)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.grad_clip_norm)
        self.opt.step()
        self._soft_update()
        if self.ema is not None:
            self._ema_update()
        self.train_steps += 1
        self.last_td_errors = td.abs().cpu().numpy()
        return float(loss.item())

    @torch.no_grad()
    def _soft_update(self):
        tau = self.cfg.target_tau
        for tp, p in zip(self.tgt.parameters(), self.q.parameters()):
            tp.data.mul_(1.0 - tau).add_(tau * p.data)

    @torch.no_grad()
    def _ema_update(self):
        d = self.cfg.ema_decay
        for ep, p in zip(self.ema.parameters(), self.q.parameters()):
            ep.data.mul_(d).add_((1.0 - d) * p.data)
        for eb, b in zip(self.ema.buffers(), self.q.buffers()):
            eb.data.copy_(b.data)

    def set_lr(self, lr: float):
        for g in self.opt.param_groups:
            g["lr"] = lr

    def deploy_net(self):
        """Network to save/evaluate: the EMA snapshot if enabled, else the online net."""
        return self.ema if self.ema is not None else self.q

    # ---- persistence (policy weights only; resume state lives in utils/checkpoint) ----
    def save(self, path) -> None:
        torch.save(self.deploy_net().state_dict(), path)

    def load(self, path) -> None:
        sd = torch.load(path, map_location=self.device)
        self.q.load_state_dict(sd)
        self.tgt.load_state_dict(self.q.state_dict())
        if self.ema is not None:
            self.ema.load_state_dict(self.q.state_dict())
