"""Network definitions only — no training logic lives here.

`QNet` (plain MLP) is byte-compatible with the shipped models/dqn_final.pth when
`layernorm=False`. `DuelingQNet` and LayerNorm are V2 upgrades. Use
`build_qnet(cfg)` so the agent and evaluator always construct the network the
config asks for.
"""

import torch.nn as nn


def _mlp_block(in_dim: int, out_dim: int, layernorm: bool):
    layers = [nn.Linear(in_dim, out_dim)]
    if layernorm:
        layers.append(nn.LayerNorm(out_dim))
    layers.append(nn.ReLU())
    return layers


class QNet(nn.Module):
    """Feed-forward Q-network: state_dim -> hidden -> hidden -> n_actions.

    With layernorm=False the module layout is identical to the original network,
    so existing checkpoints load unchanged.
    """

    def __init__(self, state_dim: int, hidden: int, n_actions: int, layernorm: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            *_mlp_block(state_dim, hidden, layernorm),
            *_mlp_block(hidden, hidden, layernorm),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class DuelingQNet(nn.Module):
    """Dueling architecture: Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a)).

    Good fit for low-advantage control like Flappy, where value is dominated by
    "am I alive and centered" and both actions usually survive.
    """

    def __init__(self, state_dim: int, hidden: int, n_actions: int, layernorm: bool = False):
        super().__init__()
        self.feature = nn.Sequential(
            *_mlp_block(state_dim, hidden, layernorm),
            *_mlp_block(hidden, hidden, layernorm),
        )
        self.value = nn.Sequential(*_mlp_block(hidden, hidden, layernorm), nn.Linear(hidden, 1))
        self.advantage = nn.Sequential(*_mlp_block(hidden, hidden, layernorm), nn.Linear(hidden, n_actions))

    def forward(self, x):
        f = self.feature(x)
        v = self.value(f)
        a = self.advantage(f)
        return v + (a - a.mean(dim=1, keepdim=True))


def build_qnet(cfg) -> nn.Module:
    """Construct the Q-network the config asks for (plain vs dueling, +/- LayerNorm)."""
    ln = getattr(cfg, "layernorm", False)
    if getattr(cfg, "dueling", False):
        return DuelingQNet(cfg.state_dim, cfg.hidden, cfg.n_actions, layernorm=ln)
    return QNet(cfg.state_dim, cfg.hidden, cfg.n_actions, layernorm=ln)
