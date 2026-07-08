"""Config loading: YAML -> frozen dataclass.

Every hyperparameter lives in a YAML file (see configs/dqn.yaml). Code reads a
`Config` object; it never inlines a magic number. The object is frozen so a run
can't accidentally mutate its own hyperparameters halfway through.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Any, Dict

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

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r") as f:
            raw: Dict[str, Any] = yaml.safe_load(f)
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
        missing = known - set(raw)
        if missing:
            raise ValueError(f"Missing config keys in {path}: {sorted(missing)}")
        return cls(**raw)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
