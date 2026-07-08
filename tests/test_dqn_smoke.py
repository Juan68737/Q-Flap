"""Smoke test: the DQN update runs N steps on random data without NaNs, and
the loss trends down on a trivial fixed-target task.

Runs headless and needs no game window. Skipped automatically if torch isn't
installed.
"""

import os

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

torch = pytest.importorskip("torch")

from flappy_rl.config import Config
from flappy_rl.agents.dqn import DQNAgent, dqn_loss


def _tiny_cfg():
    cfg = Config.from_yaml("configs/dqn.yaml")
    # shrink for a fast test while keeping the network shape
    return Config(**{**cfg.to_dict(), "buffer_size": 1000, "batch_size": 32})


def _random_batch(cfg, device, n):
    return (
        torch.randn(n, cfg.state_dim, device=device),
        torch.randint(0, cfg.n_actions, (n,), device=device),
        torch.randn(n, device=device),
        torch.randn(n, cfg.state_dim, device=device),
        torch.zeros(n, dtype=torch.bool, device=device),
    )


def test_update_no_nan():
    cfg = _tiny_cfg()
    device = torch.device("cpu")
    agent = DQNAgent(cfg, device)
    losses = []
    for _ in range(50):
        loss = agent.update(_random_batch(cfg, device, cfg.batch_size))
        assert np.isfinite(loss), "loss became non-finite"
        losses.append(loss)
    assert agent.train_steps == 50


def test_loss_decreases_on_fixed_target():
    cfg = _tiny_cfg()
    device = torch.device("cpu")
    agent = DQNAgent(cfg, device)
    batch = _random_batch(cfg, device, cfg.batch_size)  # one fixed batch, overfit it
    first = dqn_loss(batch, agent.q, agent.tgt, cfg.gamma).item()
    for _ in range(200):
        agent.update(batch)
    last = dqn_loss(batch, agent.q, agent.tgt, cfg.gamma).item()
    assert last < first, f"loss did not decrease ({first:.4f} -> {last:.4f})"
