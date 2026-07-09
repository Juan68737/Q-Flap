"""Flappy Bird RL command line (Typer).

    flappy-rl train                 # train, promote best -> models/dqn_final.pth
    flappy-rl train --name dqn2     # train, save best  -> models/dqn2.pth
    flappy-rl eval 5                # 5 games with models/dqn_final.pth
    flappy-rl eval 6 dqn2.pth       # 6 games with models/dqn2.pth

Usually driven through the `justfile` (`just train`, `just eval 5`, ...).
"""

import os
from dataclasses import replace
from pathlib import Path

import typer

from .config import Config

app = typer.Typer(add_completion=False, help="Train and evaluate Flappy Bird RL agents.")

MODELS = Path("models")
DEFAULT_CONFIG = "configs/dqn.yaml"


def _resolve_model(name: str) -> str:
    """Accept a path, or a bare name/filename resolved under models/."""
    p = Path(name)
    if p.exists():
        return str(p)
    candidate = MODELS / (name if name.endswith(".pth") else f"{name}.pth")
    if candidate.exists():
        return str(candidate)
    return name  # let the loader raise a clear error


@app.command()
def train(
    name: str = typer.Argument("dqn_final", help="save the best model to models/<name>.pth (default replaces dqn_final)"),
    config: str = typer.Option(DEFAULT_CONFIG, "--config", "-c"),
    steps: int = typer.Option(None, "--steps", "-s", help="override total env steps (quick runs)"),
):
    """Train a DQN agent and promote its best checkpoint to models/<name>.pth."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # headless; before importing the env
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    from .training import train as run_train  # late import so SDL is set first

    cfg = Config.from_yaml(config)
    if steps is not None:
        cfg = replace(cfg, total_env_steps=steps)
    model_name = name if name.endswith(".pth") else f"{name}.pth"
    run_train(cfg, out_model=MODELS / model_name)


@app.command()
def eval(
    episodes: int = typer.Argument(10, help="number of games to play"),
    model: str = typer.Argument("dqn_final", help="model name (under models/) or a path"),
    config: str = typer.Option(DEFAULT_CONFIG, "--config", "-c",
                               help="config matching the model's architecture (V2 models need their config)"),
    fps: int = typer.Option(45, "--fps"),
    max_steps: int = typer.Option(2000, "--max-steps", help="cap per game; 0 = unlimited"),
    action_repeat: int = typer.Option(None, "--action-repeat"),
):
    """Watch a trained model play `episodes` games."""
    from .evaluation import evaluate as run_eval  # late import (opens a window)

    cfg = Config.from_yaml(config)
    run_eval(_resolve_model(model), cfg, episodes=episodes, fps=fps,
             max_steps=max_steps, action_repeat=action_repeat)


def main():
    import multiprocessing as mp
    mp.freeze_support()
    app()


if __name__ == "__main__":
    main()
