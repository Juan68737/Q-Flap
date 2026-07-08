"""Thin DQN training entrypoint.

    pip install -e .
    python scripts/train.py --config configs/dqn.yaml

The loop reads like pseudocode: act -> step envs -> store -> maybe update ->
log -> checkpoint. All the real logic (action selection, the update rule) lives
in the agent; all hyperparameters live in the config.
"""

import os
# Train headless: no game window. Must be set before importing the env.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import argparse
from collections import deque

import numpy as np
import torch

from flappy_rl.config import Config
from flappy_rl.agents.dqn import DQNAgent
from flappy_rl.buffers.replay import ReplayBuffer
from flappy_rl.envs.vec_env import VecEnv
from flappy_rl.utils.seeding import seed_everything
from flappy_rl.utils import checkpoint as ckpt
from flappy_rl.utils.logging import Logger


def epsilon_by_step(t: int, cfg: Config) -> float:
    if t >= cfg.eps_decay_steps:
        return cfg.eps_end
    return cfg.eps_start + (cfg.eps_end - cfg.eps_start) * (t / float(cfg.eps_decay_steps))


def train(cfg: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(cfg.seed)

    run_dir = ckpt.make_run_dir("experiments", cfg.run_name)
    ckpt.snapshot_config(run_dir, cfg)
    logger = Logger(run_dir)
    print(f"Device: {device} | run dir: {run_dir}")

    agent = DQNAgent(cfg, device)
    buffer = ReplayBuffer(cfg.buffer_size, cfg.state_dim, device)
    vec = VecEnv(cfg)

    global_steps = 0
    updates_backlog = 0.0
    recent_scores = deque(maxlen=1000)          # episodic scores across all workers
    env_scores = [0 for _ in range(cfg.num_envs)]

    try:
        while global_steps < cfg.total_env_steps:
            warming_up = global_steps < cfg.warmup_steps
            eps = cfg.eps_start if warming_up else epsilon_by_step(global_steps, cfg)

            if warming_up:
                actions = np.random.randint(0, cfg.n_actions, size=cfg.num_envs, dtype=np.int64)
            else:
                actions = agent.act_batch(vec.states, eps)

            results = vec.step(actions)
            next_states = []
            for i, res in enumerate(results):
                buffer.push(vec.states[i], int(actions[i]), float(res.r), res.ns, bool(res.d))
                next_states.append(res.ns)
                env_scores[i] += res.score_inc
                if res.d:
                    recent_scores.append(env_scores[i])
                    env_scores[i] = 0
            vec.states = next_states
            global_steps += cfg.num_envs

            # updates proportional to data collected
            if not warming_up:
                updates_backlog += cfg.num_envs * cfg.update_ratio
                num_updates = int(updates_backlog)
                if num_updates > 0:
                    last_loss = None
                    for _ in range(num_updates):
                        if len(buffer) >= cfg.batch_size:
                            last_loss = agent.update(buffer.sample(cfg.batch_size))
                    updates_backlog -= num_updates

                    if (last_loss is not None and agent.train_steps
                            and agent.train_steps % cfg.log_every_updates == 0):
                        probe = torch.as_tensor(
                            np.array(vec.states[:min(32, cfg.num_envs)]),
                            dtype=torch.float32, device=device)
                        with torch.no_grad():
                            mean_q = float(agent.q(probe).mean().item())
                        logger.log_scalars(
                            global_steps,
                            loss=last_loss, mean_q=mean_q, eps=eps,
                            buffer=len(buffer),
                            avg_episode_score=float(np.mean(recent_scores) if recent_scores else 0.0),
                        )

            if global_steps % cfg.save_every_steps == 0:
                ckpt.save_checkpoint(run_dir / f"steps_{global_steps}.pth", agent, cfg, global_steps)
    except KeyboardInterrupt:
        ckpt.save_checkpoint(run_dir / "interrupt.pth", agent, cfg, global_steps)
        print(f"\nInterrupted. Saved -> {run_dir / 'interrupt.pth'}")
    finally:
        vec.close()
        logger.close()

    ckpt.save_checkpoint(run_dir / "final.pth", agent, cfg, global_steps)
    agent.save(run_dir / "policy_final.pth")  # weights-only, ready for scripts/evaluate.py
    print(f"Done. Checkpoints in {run_dir}")
    print(f"To watch it play: python scripts/evaluate.py --model {run_dir / 'policy_final.pth'}")


def main():
    ap = argparse.ArgumentParser(description="Train a DQN agent on Flappy Bird")
    ap.add_argument("--config", default="configs/dqn.yaml")
    args = ap.parse_args()
    train(Config.from_yaml(args.config))


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
