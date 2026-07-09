"""DQN training loop (V1 + config-gated V2 upgrades).

Reads like pseudocode: act -> step envs -> store -> maybe update -> log ->
checkpoint. All real logic lives in the agent; all hyperparameters in the
config. V2 knobs (n-step, prioritized replay, cosine LR, EMA deploy net,
greedy-eval checkpoint selection) turn on from the config; with them off this is
the exact V1 loop. Tracks the best model so a late collapse can't cost you it,
and optionally promotes it to models/<name>.pth.
"""

import math
import shutil
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .agents.dqn import DQNAgent
from .buffers.replay import ReplayBuffer
from .buffers.prioritized import PrioritizedReplayBuffer
from .buffers.nstep import NStep
from .envs.vec_env import VecEnv
from .evaluation import score_policy
from .utils.seeding import seed_everything
from .utils import checkpoint as ckpt
from .utils.logging import Logger


def _fmt_secs(s: float) -> str:
    s = int(s)
    h, m = s // 3600, (s % 3600) // 60
    return f"{h}h{m:02d}m" if h else f"{m}m{s % 60:02d}s"


def epsilon_by_step(t: int, cfg: Config) -> float:
    if t >= cfg.eps_decay_steps:
        return cfg.eps_end
    return cfg.eps_start + (cfg.eps_end - cfg.eps_start) * (t / float(cfg.eps_decay_steps))


def _lr_at(step: int, cfg: Config) -> float:
    """Flat LR unless cfg.lr_final is set, then cosine-decay lr -> lr_final."""
    if cfg.lr_final is None:
        return cfg.lr
    frac = min(1.0, step / float(cfg.total_env_steps))
    return cfg.lr_final + 0.5 * (cfg.lr - cfg.lr_final) * (1.0 + math.cos(math.pi * frac))


def _beta_at(step: int, cfg: Config) -> float:
    frac = min(1.0, step / float(cfg.total_env_steps))
    return cfg.per_beta_start + frac * (cfg.per_beta_end - cfg.per_beta_start)


def train(cfg: Config, out_model: Path | None = None) -> Path:
    """Run training. Returns the run dir. If out_model is given, the best snapshot
    is copied there (weights-only, ready for evaluation)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(cfg.seed)

    run_dir = ckpt.make_run_dir("experiments", cfg.run_name)
    ckpt.snapshot_config(run_dir, cfg)
    logger = Logger(run_dir)

    prioritized = cfg.prioritized
    use_nstep = cfg.n_step > 1
    eval_select = cfg.eval_every_steps > 0
    feats = [f"dueling={cfg.dueling}", f"layernorm={cfg.layernorm}", f"n_step={cfg.n_step}",
             f"per={prioritized}", f"ema={cfg.ema_decay}", f"eval_select={eval_select}"]
    print(f"Device: {device} | run dir: {run_dir}")
    print("V2 features: " + " ".join(feats))

    agent = DQNAgent(cfg, device)
    if prioritized:
        buffer = PrioritizedReplayBuffer(cfg.buffer_size, cfg.state_dim, device, cfg.per_alpha, cfg.per_eps)
    else:
        buffer = ReplayBuffer(cfg.buffer_size, cfg.state_dim, device)
    nsteppers = [NStep(cfg.n_step, cfg.gamma) for _ in range(cfg.num_envs)] if use_nstep else None
    vec = VecEnv(cfg)

    global_steps = 0
    updates_backlog = 0.0
    recent_scores = deque(maxlen=1000)          # episodic scores across all workers
    env_scores = [0 for _ in range(cfg.num_envs)]
    best_metric = -1.0                          # best score seen (eval score, or rolling train score)
    best_path = run_dir / "best.pth"
    evals_done = 0

    start_t = time.time()
    last_hb = start_t
    print(f"Training for {cfg.total_env_steps:,} steps "
          f"(first {cfg.warmup_steps:,} are silent warmup). Ctrl-C saves and stops.", flush=True)

    def push_transition(i, res):
        raw = (vec.states[i], int(actions[i]), float(res.r), res.ns, bool(res.d))
        if use_nstep:
            for tr in nsteppers[i].push(*raw):
                buffer.push(*tr)
        else:
            buffer.push(*raw)

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
                push_transition(i, res)
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

            if warming_up:
                continue

            # cosine LR schedule
            if cfg.lr_final is not None:
                agent.set_lr(_lr_at(global_steps, cfg))

            # updates proportional to data collected
            updates_backlog += cfg.num_envs * cfg.update_ratio
            num_updates = int(updates_backlog)
            if num_updates > 0:
                beta = _beta_at(global_steps, cfg)
                last_loss = None
                for _ in range(num_updates):
                    if len(buffer) < cfg.batch_size:
                        break
                    if prioritized:
                        batch, idxs, weights = buffer.sample(cfg.batch_size, beta)
                        last_loss = agent.update(batch, is_weights=weights)
                        buffer.update_priorities(idxs, agent.last_td_errors)
                    else:
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
                    logger.log_scalars(global_steps, loss=last_loss, mean_q=mean_q, eps=eps,
                                       buffer=len(buffer), avg_episode_score=avg)
                    # with eval-based selection we bank on greedy eval instead (below)
                    if not eval_select and avg > best_metric:
                        best_metric = avg
                        agent.save(best_path)

            # greedy-eval checkpoint selection: score the deploy net on fixed seeds
            if eval_select and global_steps // cfg.eval_every_steps > evals_done:
                evals_done = global_steps // cfg.eval_every_steps
                eval_score = score_policy(agent.deploy_net(), cfg, device,
                                          cfg.eval_episodes, cfg.eval_max_steps)
                logger.log_scalars(global_steps, eval_score=eval_score)
                marker = ""
                if eval_score > best_metric:
                    best_metric = eval_score
                    agent.save(best_path)
                    marker = "  <- new best"
                print(f"  [eval @ {global_steps:,}] greedy score {eval_score:.1f}"
                      f" (best {best_metric:.1f}){marker}", flush=True)

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

    winner = best_path if best_path.exists() else (run_dir / "policy_final.pth")
    metric_name = "greedy eval score" if eval_select else "avg train score"
    print(f"Done. Checkpoints in {run_dir} (best {metric_name} ~{best_metric:.1f})")

    if out_model is not None:
        out_model = Path(out_model)
        out_model.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(winner, out_model)
        print(f"Promoted best model -> {out_model}")
        print(f"Watch it:  flappy-rl eval 10 {out_model.name}")
    else:
        print(f"Best model: {winner}")
    return run_dir
