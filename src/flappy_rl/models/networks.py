"""Network definitions only — no training logic lives here.

`QNet` is the MLP the DQN agent uses. Its layout (state_dim -> hidden -> hidden
-> n_actions) must match the saved weights in models/dqn_final.pth.
"""

import torch.nn as nn


class QNet(nn.Module):
    """Feed-forward Q-network: state_dim -> hidden -> hidden -> n_actions."""

    def __init__(self, state_dim: int, hidden: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)
