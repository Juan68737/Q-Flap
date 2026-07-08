"""Observation + reward-shaping wrappers.

This is the one place reward tinkering lives — kept out of both the raw env
(envs/flappy.py) and the agent. `get_state` defines the observation the network
sees; `shaped_reward` defines what the agent is optimizing.
"""

import numpy as np

from .flappy import WIN_WIDTH, WIN_HEIGHT

# The observation is 7 features (must match Config.state_dim and the network).
STATE_DIM = 7


def next_pipe_idx(bird, pipes) -> int:
    """Index of the pipe the bird is currently flying toward."""
    idx = 0
    if len(pipes) > 1 and bird.x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
        idx = 1
    return idx


def get_state(bird, pipes) -> np.ndarray:
    """7 normalized features of the bird relative to the next pipe."""
    if not pipes:
        return np.zeros(STATE_DIM, dtype=np.float32)
    p = pipes[next_pipe_idx(bird, pipes)]
    top, bottom = p.height, p.bottom
    gap_center = 0.5 * (top + bottom)
    gap_size = bottom - top
    return np.array([
        bird.y / WIN_HEIGHT,                          # vertical position
        bird.vel / 16.0,                              # velocity
        (p.x - bird.x) / WIN_WIDTH,                   # horizontal distance to pipe
        (bird.y - gap_center) / max(1.0, gap_size),   # offset from gap center
        gap_size / WIN_HEIGHT,                         # gap size
        (top - bird.y) / WIN_HEIGHT,                   # top-of-gap relative
        (bottom - bird.y) / WIN_HEIGHT,               # bottom-of-gap relative
    ], dtype=np.float32)


def shaped_reward(cfg, alive: bool, passed_pipe: bool, bird, pipes, action: int) -> float:
    """Reward = alive bonus + pass bonus + centering bonus - flap penalty - death.

    `cfg` supplies the shaping weights (see configs/dqn.yaml). Clipped to [-1, 1].
    """
    r = cfg.reward_alive
    if passed_pipe:
        r += cfg.reward_pass_pipe
    if action == 1:
        r += cfg.reward_flap_pen
    if pipes:
        p = pipes[next_pipe_idx(bird, pipes)]
        dx = (p.x - bird.x) / WIN_WIDTH
        if 0.0 <= dx <= cfg.bonus_dx_window:
            gap_center = 0.5 * (p.height + p.bottom)
            gap_half = max(1.0, (p.bottom - p.height) / 2.0)
            dist = abs(bird.y - gap_center) / gap_half
            bonus = cfg.center_bonus_w * (1.0 - min(1.0, dist))
            proximity = 1.0 - (dx / cfg.bonus_dx_window)
            r += bonus * proximity
    if not alive:
        r += cfg.reward_death
    return float(np.clip(r, -1.0, 1.0))
