"""Config loading: YAML -> frozen dataclass.

Every hyperparameter lives in a YAML file (see configs/dqn.yaml). Code reads a
`Config` object; it never inlines a magic number. The object is frozen so a run
can't accidentally mutate its own hyperparameters halfway through.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, fields, MISSING
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass(frozen=True)
class Config:
    run_name: str
    seed: int

    # rollout / environment
    num_envs: int
    total_env_steps: int
    max_steps_per_ep: int
    action_repeat: int

    # replay buffer
    buffer_size: int
    batch_size: int
    warmup_steps: int

    # optimization / learning
    lr: float
    gamma: float
    target_tau: float
    grad_clip_norm: float
    update_ratio: float

    # exploration
    eps_start: float
    eps_end: float
    eps_decay_steps: int

    # reward shaping
    reward_death: float
    reward_pass_pipe: float
    reward_alive: float
    reward_flap_pen: float
    center_bonus_w: float
    bonus_dx_window: float
    pass_edge_floor: float      # fraction of the pass reward earned at the gap edge (middle earns full)

    # network
    state_dim: int
    hidden: int
    n_actions: int

    # logging / checkpointing
    log_every_updates: int
    save_every_steps: int

    # --- V2 upgrades (all default to V1 behavior so old configs load unchanged) ---
    # network
    dueling: bool = False            # dueling head: Q = V(s) + (A(s,a) - mean_a A)
    layernorm: bool = False          # LayerNorm on hidden layers (tames the deadly triad)
    # multi-step returns
    n_step: int = 1                  # n-step TD targets (1 = vanilla)
    # prioritized experience replay (sum-tree)
    prioritized: bool = False
    per_alpha: float = 0.6           # how strongly to prioritize (0 = uniform)
    per_beta_start: float = 0.4      # importance-sampling correction start
    per_beta_end: float = 1.0        # ...annealed to 1.0 by end of training
    per_eps: float = 1e-5            # priority floor so nothing gets prob 0
    # stability / deployment
    ema_decay: float = 0.0           # Polyak snapshot of weights for deploy (0 = off; try 0.999)
    lr_final: Optional[float] = None # cosine-decay LR to this value (None = flat lr)
    # greedy-eval-based checkpoint selection (replaces noisy train-score selection)
    eval_every_steps: int = 0        # 0 = pick best by rolling train score; >0 = periodic greedy eval
    eval_episodes: int = 20
    eval_max_steps: int = 3000

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r") as f:
            raw: Dict[str, Any] = yaml.safe_load(f)
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
        required = {f.name for f in fields(cls)
                    if f.default is MISSING and f.default_factory is MISSING}
        missing = required - set(raw)
        if missing:
            raise ValueError(f"Missing config keys in {path}: {sorted(missing)}")
        return cls(**raw)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
