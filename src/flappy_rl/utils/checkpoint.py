"""Run directories + self-describing checkpoints.

A checkpoint carries everything needed to resume *and* to trace it back to what
produced it: model + optimizer + step count + RNG state + a snapshot of the
resolved config + the git SHA. Build this in from day one, don't bolt it on.
"""

import json
import random
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def make_run_dir(base: str, run_name: str) -> Path:
    """experiments/<run_name>_<YYYY-MM-DD_HH-MM-SS>/ — a fresh folder per run."""
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(base) / f"{run_name}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def snapshot_config(run_dir: Path, cfg) -> None:
    payload = {"config": cfg.to_dict(), "git_sha": git_sha()}
    (Path(run_dir) / "config.json").write_text(json.dumps(payload, indent=2))


def save_checkpoint(path, agent, cfg, global_steps: int) -> None:
    torch.save({
        "model": agent.q.state_dict(),
        "target": agent.tgt.state_dict(),
        "optimizer": agent.opt.state_dict(),
        "train_steps": agent.train_steps,
        "global_steps": global_steps,
        "config": cfg.to_dict(),
        "git_sha": git_sha(),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        },
    }, path)


def load_checkpoint(path, agent, restore_rng: bool = True) -> int:
    """Restore agent + optimizer (+ optional RNG). Returns the global step count."""
    ckpt = torch.load(path, map_location=agent.device)
    agent.q.load_state_dict(ckpt["model"])
    agent.tgt.load_state_dict(ckpt["target"])
    agent.opt.load_state_dict(ckpt["optimizer"])
    agent.train_steps = ckpt.get("train_steps", 0)
    if restore_rng and "rng" in ckpt:
        random.setstate(ckpt["rng"]["python"])
        np.random.set_state(ckpt["rng"]["numpy"])
        torch.set_rng_state(ckpt["rng"]["torch"])
    return ckpt.get("global_steps", 0)
