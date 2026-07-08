"""DQN training loop.

Reads like pseudocode: act -> step envs -> store -> maybe update -> log ->
checkpoint. All real logic lives in the agent; all hyperparameters in the
config. Tracks the best-performing snapshot (by rolling average score) so a late
collapse can't cost you the good model, and optionally promotes it to
models/<name>.pth.
"""

import os
import shutil
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch


def _fmt_secs(s: float) -> str:
    s = int(s)
    h, m = s // 3600, (s % 3600) // 60
    return f"{h}h{m:02d}m" if h else f"{m}m{s % 60:02d}s"

from .config import Config
from .agents.dqn import DQNAgent
from .buffers.replay import ReplayBuffer
from .envs.vec_env import VecEnv
from .utils.seeding import seed_everything
from .utils import checkpoint as ckpt
from .utils.logging import Logger


def epsilon_by_step(t: int, cfg: Config) -> float:
    if t >= cfg.eps_decay_steps:
        return cfg.eps_end
    return cfg.eps_start + (cfg.eps_end - cfg.eps_start) * (t / float(cfg.eps_decay_steps))


def train(cfg: Config, out_model: Path | None = None) -> Path:
    """Run training. Returns the run directory. If out_model is given, the best
    snapshot is copied there (weights-only, ready for evaluation)."""
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
    best_avg = -1.0
    best_path = run_dir / "best.pth"

    start_t = time.time()
    last_hb = start_t
    print(f"Training for {cfg.total_env_steps:,} steps "
          f"(first {cfg.warmup_steps:,} are silent warmup). Ctrl-C saves and stops.", flush=True)

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

            # live progress heartbeat (~every 5s) so it's never a silent screen
            now = time.time()
            if now - last_hb >= 5.0:
                last_hb = now
                elapsed = now - start_t
                sps = global_steps / max(elapsed, 1e-9)
                eta = (cfg.total_env_steps - global_steps) / max(sps, 1e-9)
                pct = 100.0 * global_steps / cfg.total_env_steps
                avg = float(np.mean(recent_scores)) if recent_scores else 0.0
                phase = "warmup" if warming_up else "train "
                print(f"  [{phase}] {global_steps:>10,}/{cfg.total_env_steps:,} ({pct:4.1f}%) | "
                      f"{sps:5.0f} steps/s | elapsed {_fmt_secs(elapsed)} | eta {_fmt_secs(eta)} | "
                      f"avgScore {avg:.2f}", flush=True)

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
                        avg = float(np.mean(recent_scores)) if len(recent_scores) >= 50 else 0.0
                        logger.log_scalars(
                            global_steps,
                            loss=last_loss, mean_q=mean_q, eps=eps,
                            buffer=len(buffer), avg_episode_score=avg,
                        )
                        # bank the best model seen so far (a late collapse can't erase it)
                        if avg > best_avg:
                            best_avg = avg
                            agent.save(best_path)

            if global_steps % cfg.save_every_steps == 0:
                ckpt.save_checkpoint(run_dir / f"steps_{global_steps}.pth", agent, cfg, global_steps)
    except KeyboardInterrupt:
        ckpt.save_checkpoint(run_dir / "interrupt.pth", agent, cfg, global_steps)
        print(f"\nInterrupted. Saved -> {run_dir / 'interrupt.pth'}")
    finally:
        vec.close()
        logger.close()

    ckpt.save_checkpoint(run_dir / "final.pth", agent, cfg, global_steps)
    agent.save(run_dir / "policy_final.pth")

    # the model worth keeping: the best snapshot if we captured one, else the last
    winner = best_path if best_path.exists() else (run_dir / "policy_final.pth")
    print(f"Done. Checkpoints in {run_dir} (best avg score ~{best_avg:.1f})")

    if out_model is not None:
        out_model = Path(out_model)
        out_model.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(winner, out_model)
        print(f"Promoted best model -> {out_model}")
        print(f"Watch it:  flappy-rl eval 10 {out_model.name}")
    else:
        print(f"Best model: {winner}")
    return run_dir
