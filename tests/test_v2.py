"""Unit tests for the DQN V2 upgrades: sum-tree, prioritized replay, n-step
returns, dueling/LayerNorm networks, and the agent on a V2 config."""

import os

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

torch = pytest.importorskip("torch")

from flappy_rl.buffers.sumtree import SumTree, MinTree
from flappy_rl.buffers.prioritized import PrioritizedReplayBuffer
from flappy_rl.buffers.nstep import NStep
from flappy_rl.models.networks import QNet, DuelingQNet, build_qnet


# ---------------- sum-tree / min-tree ----------------

def test_sumtree_total_and_update():
    t = SumTree(8)
    for i, p in enumerate([1, 2, 3, 4, 0, 0, 0, 0]):
        t.update(i, p)
    assert t.total == 10
    t.update(0, 5)          # 1 -> 5
    assert t.total == 14


def test_sumtree_sample_boundaries():
    t = SumTree(4)
    for i, p in enumerate([1.0, 2.0, 3.0, 4.0]):
        t.update(i, p)
    # cumulative ranges: leaf0 [0,1) leaf1 [1,3) leaf2 [3,6) leaf3 [6,10)
    assert t.sample(0.5) == 0
    assert t.sample(2.0) == 1
    assert t.sample(5.9) == 2
    assert t.sample(9.9) == 3


def test_sumtree_sampling_is_proportional():
    t = SumTree(4)
    weights = [1.0, 1.0, 2.0, 6.0]   # leaf 3 should get ~60%
    for i, p in enumerate(weights):
        t.update(i, p)
    rng = np.random.default_rng(0)
    counts = np.zeros(4)
    for _ in range(20000):
        counts[t.sample(rng.random() * t.total)] += 1
    freq = counts / counts.sum()
    assert abs(freq[3] - 0.6) < 0.03
    assert abs(freq[2] - 0.2) < 0.03


def test_mintree():
    t = MinTree(4)
    for i, p in enumerate([5.0, 3.0, 9.0, 1.0]):
        t.update(i, p)
    assert t.min == 1.0
    t.update(3, 7.0)
    assert t.min == 3.0


# ---------------- prioritized replay ----------------

def test_per_sample_shapes_and_weights():
    buf = PrioritizedReplayBuffer(1000, state_dim=7, device=torch.device("cpu"),
                                  alpha=0.6, eps=1e-5)
    for _ in range(300):
        buf.push(np.random.randn(7), 1, 0.5, np.random.randn(7), False)
    (s, a, r, sn, d), idxs, w = buf.sample(32, beta=0.4)
    assert s.shape == (32, 7) and a.shape == (32,)
    assert idxs.shape == (32,) and w.shape == (32,)
    assert torch.all(w > 0) and float(w.max()) <= 1.0 + 1e-5   # normalized by max weight


def test_per_prioritizes_high_td_error():
    buf = PrioritizedReplayBuffer(1000, state_dim=2, device=torch.device("cpu"),
                                  alpha=1.0, eps=1e-6)
    for _ in range(100):
        buf.push(np.zeros(2), 0, 0.0, np.zeros(2), False)
    # give index 0 a huge TD error; it should dominate sampling
    buf.update_priorities(np.array([0]), np.array([1000.0]))
    hits = 0
    for _ in range(2000):
        _, idxs, _ = buf.sample(1, beta=1.0)
        hits += int(idxs[0] == 0)
    assert hits > 1500   # index 0 sampled the large majority of the time


# ---------------- n-step returns ----------------

def test_nstep_full_window_return():
    ns = NStep(n=3, gamma=0.5)
    out = ns.push("s0", 0, 1.0, "s1", False); assert out == []
    out = ns.push("s1", 0, 1.0, "s2", False); assert out == []
    out = ns.push("s2", 0, 1.0, "s3", False)          # window of 3 ready
    assert len(out) == 1
    s0, a0, R, sn, d = out[0]
    assert s0 == "s0" and sn == "s3" and d is False
    assert R == pytest.approx(1.0 + 0.5 * 1.0 + 0.25 * 1.0)   # 1.75


def test_nstep_flushes_on_done():
    ns = NStep(n=3, gamma=1.0)
    ns.push("s0", 0, 1.0, "s1", False)
    out = ns.push("s1", 0, 1.0, "s2", True)   # terminal before window fills -> flush both
    assert len(out) == 2
    assert out[0][0] == "s0" and out[0][2] == pytest.approx(2.0) and out[0][4] is True
    assert out[1][0] == "s1" and out[1][2] == pytest.approx(1.0) and out[1][4] is True


# ---------------- networks ----------------

def test_dueling_and_layernorm_shapes():
    x = torch.randn(5, 7)
    for net in (QNet(7, 32, 2, layernorm=True),
                DuelingQNet(7, 32, 2, layernorm=False),
                DuelingQNet(7, 32, 2, layernorm=True)):
        assert net(x).shape == (5, 2)


def test_plain_qnet_is_backward_compatible():
    # layernorm=False must keep the exact module layout so old checkpoints load
    old = QNet(7, 16, 2)
    new = QNet(7, 16, 2, layernorm=False)
    assert set(old.state_dict().keys()) == set(new.state_dict().keys())
    assert {"net.0.weight", "net.2.weight", "net.4.weight"} <= set(old.state_dict().keys())


# ---------------- agent on a V2 config ----------------

def test_agent_update_on_v2_config():
    from flappy_rl.config import Config
    from flappy_rl.agents.dqn import DQNAgent
    cfg = Config.from_yaml("configs/dqn_v2.yaml")
    cfg = Config(**{**cfg.to_dict(), "buffer_size": 500, "batch_size": 16})
    agent = DQNAgent(cfg, torch.device("cpu"))
    assert agent.ema is not None                      # ema_decay > 0
    n = cfg.batch_size
    batch = (torch.randn(n, 7), torch.randint(0, 2, (n,)), torch.randn(n),
             torch.randn(n, 7), torch.zeros(n, dtype=torch.bool))
    w = torch.rand(n)
    loss = agent.update(batch, is_weights=w)
    assert np.isfinite(loss)
    assert agent.last_td_errors.shape == (n,)         # TD errors returned for PER
