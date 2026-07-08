# Tabular Q-learning trainer (multiprocess) for Flappy Bird.
#
# This is the "classic" bins-and-a-Q-table approach, kept as a second algorithm
# alongside the DQN (scripts/train.py). It is intentionally self-contained
# rather than plugged into the agents/ interface, because it has no replay
# buffer or neural network. Outputs land in experiments/qlearn/.
#
#     pip install -e .
#     python scripts/train_qlearn.py --workers 8 --episodes-per-worker 5000
"""
Tabular Q-learning trainer with slow epsilon decay for thorough exploration.

Notes:
- Slow epsilon decay (0.9995) and a low floor (0.01) => ~4000 episodes of
  meaningful exploration across 5000 episodes per worker.

Epsilon schedule:
- Episode 0: 50%
- Episode 500: 39%
- Episode 1000: 30%
- Episode 2000: 18%
- Episode 3000: 11%
- Episode 4000: 7%
- Episode 5000: 4%
"""

# Edit src/flappy_rl/envs/flappy.py to change difficulty settings!

import math
import os
import random
import time
import json
import importlib
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional

import numpy as np
import multiprocessing as mp

# ==================== HYPERPARAMETERS ====================
LEARNING_RATE = 0.15
DISCOUNT_FACTOR = 0.99
EXPLORATION_START = 0.5
EXPLORATION_DECAY = 0.9995  # IMPROVED: Much slower for better exploration!
MIN_EXPLORATION = 0.01       # IMPROVED: Lower minimum
TOTAL_EPISODES_PER_WORKER = 5000  # IMPROVED: More episodes
SYNC_INTERVAL = 250                # IMPROVED: Sync less often
MAX_STEPS = 3000
MAX_WORKERS = 15

# State discretization
Y_BINS = 6
DY_GAP_BINS = 8
TTB_BINS = 6
VEL_BINS = 5
NEXT_GAP_BINS = 6
ACTIONS = 2

# Rewards
DEATH_PENALTY = -10.0
PIPE_PASS_REWARD = 10.0
ALIVE_REWARD = 0.1
DISTANCE_REWARD_SCALE = 0.5

# Multi-horizon
HORIZONS = [1, 5, 10]
OPTIMISTIC_INIT_VALUE = 1.0

# System
DEFAULT_NICE_VALUE = 10
USE_IONICE = False
DEFAULT_MAXTASKSPERCHILD = 200

# Export
EXPORT_DIR_NAME = "experiments/qlearn"
SAVE_BEST_Q = True
SAVE_AVG_Q = True
SAVE_REPLAY = True

EXPORT_DIR = Path(EXPORT_DIR_NAME)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# ==================== Copy all the helper functions ====================
def set_process_priority(nice_value: int = 10, ionice: bool = False, affinity: Optional[List[int]] = None):
    try:
        import psutil
    except Exception:
        psutil = None
    try:
        if os.name == "posix":
            if nice_value is not None:
                try:
                    os.nice(int(nice_value))
                except OSError:
                    pass
        elif os.name == "nt" and psutil:
            p = psutil.Process(os.getpid())
            if nice_value is None or nice_value < 5:
                p.nice(psutil.NORMAL_PRIORITY_CLASS)
            elif nice_value >= 15:
                p.nice(psutil.IDLE_PRIORITY_CLASS)
            else:
                p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass

def _pool_initializer(nice_value: int, ionice_flag: bool, affinity: Optional[List[int]], headless: bool):
    if headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        os.environ.setdefault("FB_NO_DISPLAY", "1")
    set_process_priority(nice_value, ionice_flag, affinity)

def load_fb_module():
    fb = importlib.import_module("flappy_rl.envs.flappy")
    return fb

def make_env_objects(fb):
    Bird, Pipe, Base = fb.Bird, fb.Pipe, fb.Base
    WIN_WIDTH, WIN_HEIGHT, FLOOR = fb.WIN_WIDTH, fb.WIN_HEIGHT, fb.FLOOR
    return Bird, Pipe, Base, WIN_WIDTH, WIN_HEIGHT, FLOOR

def pipe_index_for_bird(bird_x, pipes):
    if not pipes:
        return 0
    for i, p in enumerate(pipes):
        if bird_x <= p.x + p.PIPE_TOP.get_width():
            return i
    return len(pipes) - 1

def gap_center(p): 
    return (p.height + p.bottom) / 2.0

class Discretizer:
    def __init__(self, FLOOR: int):
        self.FLOOR = FLOOR
        self.y_edges = np.linspace(0, FLOOR, Y_BINS + 1)[1:-1]
        self.dy_max = 300.0
        self.dy_edges = np.linspace(0, self.dy_max, DY_GAP_BINS + 1)[1:-1]
        self.ttb_max = 100.0
        self.ttb_edges = np.linspace(0, self.ttb_max, TTB_BINS + 1)[1:-1]
        self.vel_min, self.vel_max = -12.0, 12.0
        self.vel_edges = np.linspace(self.vel_min, self.vel_max, VEL_BINS + 1)[1:-1]
        self.next_gap_edges = np.linspace(0, FLOOR, NEXT_GAP_BINS + 1)[1:-1]

    @staticmethod
    def _bin(val, edges):
        return int(np.clip(np.digitize(val, edges), 0, len(edges)))

    def discretize(self, bird, pipes):
        if not pipes:
            return (0, 0, 0, VEL_BINS // 2, 0)
        
        idx = pipe_index_for_bird(bird.x, pipes)
        cur = pipes[idx]
        nxt = pipes[min(idx + 1, len(pipes) - 1)]
        
        y = float(np.clip(bird.y, 0, self.FLOOR))
        gc = float(np.clip(gap_center(cur), 0, self.FLOOR))
        dy = min(self.dy_max, abs(y - gc))
        dist = max(0.0, cur.x - bird.x)
        ttb = min(self.ttb_max, dist / 8.0)
        vel = float(np.clip(bird.vel, self.vel_min, self.vel_max))
        next_gc = float(np.clip(gap_center(nxt), 0, self.FLOOR))
        
        return (
            self._bin(y, self.y_edges),
            self._bin(dy, self.dy_edges),
            self._bin(ttb, self.ttb_edges),
            self._bin(vel, self.vel_edges),
            self._bin(next_gc, self.next_gap_edges),
        )

def improved_reward(alive, passed_pipe, bird, pipes, action):
    if not alive:
        return DEATH_PENALTY
    
    r = ALIVE_REWARD
    
    if passed_pipe:
        r += PIPE_PASS_REWARD
    
    if pipes:
        idx = pipe_index_for_bird(bird.x, pipes)
        p = pipes[idx]
        gc = gap_center(p)
        gap_height = p.bottom - p.height
        
        dist = abs(bird.y - gc)
        normalized_dist = dist / (gap_height / 2.0)
        
        centering_reward = DISTANCE_REWARD_SCALE * (1.0 - normalized_dist)
        r += max(0, centering_reward)
    
    return r

def q_shape() -> Tuple[int, ...]:
    return (Y_BINS, DY_GAP_BINS, TTB_BINS, VEL_BINS, NEXT_GAP_BINS, ACTIONS)

def optimistic_q_init(q: np.ndarray):
    q.fill(OPTIMISTIC_INIT_VALUE)

class QAgent:
    def __init__(self, d: Discretizer, init_q: Optional[np.ndarray] = None, init_eps: Optional[float] = None):
        self.d = d
        self.Q = np.zeros(q_shape(), dtype=np.float32)
        optimistic_q_init(self.Q)
        if init_q is not None:
            np.copyto(self.Q, init_q.astype(np.float32, copy=False))
        
        self.eps = init_eps if init_eps is not None else EXPLORATION_START
        self.episode_count = 0

    def act(self, s):
        if random.random() < self.eps:
            return random.randint(0, ACTIONS - 1)
        return int(np.argmax(self.Q[s]))

    def learn_multihorizon(self, s, a, r, sp, terminal):
        if terminal:
            target = r
        else:
            targets = []
            for h in HORIZONS:
                gamma_h = DISCOUNT_FACTOR ** h
                targets.append(r + gamma_h * np.max(self.Q[sp]))
            target = np.mean(targets)
        
        current_q = self.Q[s + (a,)]
        td_error = target - current_q
        self.Q[s + (a,)] += LEARNING_RATE * td_error
        
        return td_error

    def decay_eps(self):
        self.episode_count += 1
        self.eps = max(MIN_EXPLORATION, self.eps * EXPLORATION_DECAY)

class ReplayRecorder:
    def __init__(self):
        self.frames = []
    
    def log(self, step, bird, pipes, score, WIN_WIDTH, FLOOR):
        frame = {
            "step": step,
            "bird_y": float(bird.y),
            "bird_vel": float(bird.vel),
            "score": int(score),
            "pipes": [{"x": float(p.x), "height": float(p.height), "bottom": float(p.bottom)} for p in pipes]
        }
        self.frames.append(frame)
    
    def save(self, path: Path, meta: dict):
        data = {"meta": meta, "frames": self.frames}
        np.savez_compressed(path, data=json.dumps(data))

def run_episodes(agent: QAgent, num_episodes: int, max_steps: int, Bird, Pipe, Base, WIN_WIDTH, FLOOR):
    best_score = -1
    total_score = 0
    
    for ep in range(num_episodes):
        bird = Bird(230, random.randint(250, 450))
        base = Base(FLOOR)
        pipes = [Pipe(700)]
        score = 0
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            steps += 1
            s = agent.d.discretize(bird, pipes)
            a = agent.act(s)
            
            if a == 1:
                bird.jump()
            
            bird.move()
            base.move()
            
            rem, passed = [], False
            for p in list(pipes):
                p.move()
                if p.collide(bird, None):
                    done = True
                    break
                if p.x + p.PIPE_TOP.get_width() < 0:
                    rem.append(p)
                if not p.passed and p.x < bird.x:
                    p.passed = True
                    passed = True
            
            for r in rem:
                pipes.remove(r)
            
            if passed:
                score += 1
                pipes.append(Pipe(WIN_WIDTH))
            
            if bird.y + bird.img.get_height() >= FLOOR or bird.y < -50:
                done = True
            
            sp = agent.d.discretize(bird, pipes)
            r = improved_reward(not done, passed, bird, pipes, a)
            agent.learn_multihorizon(s, a, r, sp, done)
        
        agent.decay_eps()
        best_score = max(best_score, score)
        total_score += score
    
    return {
        "best_score": best_score,
        "avg": total_score / max(1, num_episodes),
        "eps": agent.eps,
        "Q": agent.Q,
    }

def worker_round(worker_id: int, seed: int, episodes_this_round: int, max_steps: int, 
                 global_q_snapshot: np.ndarray, global_eps: float):
    fb = load_fb_module()
    Bird, Pipe, Base, WIN_WIDTH, WIN_HEIGHT, FLOOR = make_env_objects(fb)
    
    random.seed(seed)
    np.random.seed(seed)
    d = Discretizer(FLOOR)
    agent = QAgent(d, init_q=global_q_snapshot, init_eps=global_eps)
    
    worker_variation = 0.8 + 0.4 * (worker_id % 3) / 2.0
    agent.eps = min(1.0, agent.eps * worker_variation)
    
    stats = run_episodes(agent, episodes_this_round, max_steps, Bird, Pipe, Base, WIN_WIDTH, FLOOR)
    return {
        "worker": worker_id,
        "best_score": stats["best_score"],
        "avg": stats["avg"],
        "eps": agent.eps,
        "Q": stats["Q"].astype(np.float32, copy=False),
    }

def average_q(list_of_q: List[np.ndarray]) -> np.ndarray:
    out = np.zeros_like(list_of_q[0], dtype=np.float32)
    for q in list_of_q:
        out += q.astype(np.float32, copy=False)
    out /= float(len(list_of_q))
    return out

def _clamp_workers(n: int) -> int:
    return max(1, min(int(n), MAX_WORKERS))

def train_multiprocess(num_workers: Optional[int] = None,
                       total_episodes_per_worker: int = TOTAL_EPISODES_PER_WORKER,
                       sync_interval: int = SYNC_INTERVAL,
                       max_steps: int = MAX_STEPS,
                       nice_value: int = DEFAULT_NICE_VALUE,
                       ionice_flag: bool = USE_IONICE,
                       affinity: Optional[List[int]] = None,
                       headless: bool = True,
                       maxtasksperchild: Optional[int] = DEFAULT_MAXTASKSPERCHILD):
    if num_workers is None:
        num_workers = mp.cpu_count()
    num_workers = _clamp_workers(num_workers)
    
    rounds = int(math.ceil(total_episodes_per_worker / float(sync_interval)))
    
    q_size = np.prod(q_shape())
    q_mb = (q_size * 4) / (1024 * 1024)
    
    # Print difficulty settings
    fb = load_fb_module()
    if hasattr(fb, 'print_difficulty_settings'):
        fb.print_difficulty_settings()
    
    print("=" * 60)
    print("Q-LEARNING WITH EASY MODE (IMPROVED EXPLORATION)")
    print("=" * 60)
    print(f"CPU Workers: {num_workers}")
    print(f"Episodes/Worker: {total_episodes_per_worker} | Sync: {sync_interval} | Rounds: {rounds}")
    print(f"State space: {q_shape()} = {q_size:,} states")
    print(f"Exploration: {EXPLORATION_START:.2%} → {MIN_EXPLORATION:.2%}")
    print("=" * 60)
    
    set_process_priority(nice_value, ionice_flag, affinity)
    
    gQ = np.zeros(q_shape(), dtype=np.float32)
    optimistic_q_init(gQ)
    global_best = {"score": -1, "Q": gQ.copy()}
    global_eps = EXPLORATION_START
    
    start = time.time()
    with mp.Pool(processes=num_workers,
                 initializer=_pool_initializer,
                 initargs=(nice_value, ionice_flag, affinity, headless),
                 maxtasksperchild=(maxtasksperchild if (maxtasksperchild is None or maxtasksperchild > 0) else None)) as pool:
        
        for rd in range(rounds):
            ep_this = sync_interval if (rd < rounds - 1) else (total_episodes_per_worker - sync_interval * (rounds - 1))
            seeds = [(rd + 1) * 10000 + 123 + w for w in range(num_workers)]
            tasks = [(w, seeds[w], ep_this, max_steps, gQ, global_eps) for w in range(num_workers)]
            
            results = pool.starmap(worker_round, tasks)
            
            local_Qs = [res["Q"] for res in results]
            gQ = average_q(local_Qs)
            global_eps = float(np.mean([r["eps"] for r in results]))
            
            best_in_round = max(results, key=lambda r: r["best_score"])
            avg_avgs = float(np.mean([r["avg"] for r in results]))
            
            print(f"[Round {rd+1}/{rounds}] "
                  f"Worker#{best_in_round['worker']} score={best_in_round['best_score']} | "
                  f"Avg={avg_avgs:.2f} | eps={global_eps:.4f}")
            
            if best_in_round["best_score"] > global_best["score"]:
                global_best["score"] = best_in_round["best_score"]
                global_best["Q"] = best_in_round["Q"].copy()
                print(f"  🎯 NEW BEST: {best_in_round['best_score']} pipes!")
    
    dur = time.time() - start
    total_episodes = num_workers * total_episodes_per_worker
    print("\n" + "=" * 60)
    print(f"✅ TRAINING COMPLETE in {dur:.1f}s | {total_episodes/dur:.1f} eps/sec")
    print(f"🏆 Best Score: {global_best['score']} pipes")
    print(f"📉 Final Exploration: {global_eps:.4f}")
    print("=" * 60)
    
    if SAVE_BEST_Q:
        np.save(EXPORT_DIR / "best_q_table_easy.npy", global_best["Q"])
        print(f"💾 Saved best Q -> {EXPORT_DIR/'best_q_table_easy.npy'}")
    
    if SAVE_AVG_Q:
        np.save(EXPORT_DIR / "avg_q_table_easy.npy", gQ)
        print(f"💾 Saved averaged Q -> {EXPORT_DIR/'avg_q_table_easy.npy'}")
    
    if SAVE_REPLAY:
        print("\n🎬 Recording best replay (trying multiple games)...")
        best_score = -1
        best_replay_data = None
        best_seed = None
        
        # Try multiple games to get the best one
        num_attempts = 20
        for attempt in range(num_attempts):
            seed = 123 + attempt
            replay_data, score = record_greedy_replay(
                global_best["Q"], seed=seed, max_steps=5000
            )
            print(f"  Attempt {attempt+1}/{num_attempts}: Score = {score} (seed={seed})")
            
            if score > best_score:
                best_score = score
                best_replay_data = replay_data
                best_seed = seed
        
        # Save the best replay
        meta = {
            "seed": best_seed,
            "episode_score": int(best_score),
            "note": f"Best of {num_attempts} attempts",
            "attempts_tried": num_attempts
        }
        save_path = EXPORT_DIR / "replay_best.npz"
        best_replay_data.save(save_path, meta)
        print(f"\n🎥 BEST replay saved (score {best_score}) -> {save_path}")
        print(f"   (replay is a JSON blob inside the .npz under key 'data')")
    
    return global_best, gQ

def record_greedy_replay(Q, seed: int, max_steps: int = 5000):
    """Record a greedy replay and return the recorder and score"""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("FB_NO_DISPLAY", "1")
    fb = load_fb_module()
    Bird, Pipe, Base, WIN_WIDTH, WIN_HEIGHT, FLOOR = make_env_objects(fb)
    
    random.seed(seed)
    np.random.seed(seed)
    d = Discretizer(FLOOR)
    agent = QAgent(d, init_q=Q, init_eps=0.0)
    
    bird = Bird(230, random.randint(250, 450))
    base = Base(FLOOR)
    pipes = [Pipe(700)]
    score, done, steps = 0, False, 0
    
    rec = ReplayRecorder()
    
    while not done and steps < max_steps:
        steps += 1
        s = d.discretize(bird, pipes)
        a = agent.act(s)
        
        if a == 1:
            bird.jump()
        
        bird.move()
        base.move()
        
        rem, passed = [], False
        for p in list(pipes):
            p.move()
            if p.collide(bird, None):
                done = True
                break
            if p.x + p.PIPE_TOP.get_width() < 0:
                rem.append(p)
            if not p.passed and p.x < bird.x:
                p.passed = True
                passed = True
        
        for r in rem:
            pipes.remove(r)
        
        if passed:
            score += 1
            pipes.append(Pipe(WIN_WIDTH))
        
        if bird.y + bird.img.get_height() >= FLOOR or bird.y < -50:
            done = True
        
        rec.log(steps, bird, pipes, score, WIN_WIDTH, FLOOR)
    
    return rec, score

def parse_affinity(s: Optional[str]) -> Optional[List[int]]:
    if not s:
        return None
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out

def main():
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    
    ap = argparse.ArgumentParser(description="Q-learning with Easy Mode")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--episodes-per-worker", type=int, default=TOTAL_EPISODES_PER_WORKER)
    ap.add_argument("--sync-interval", type=int, default=SYNC_INTERVAL)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--nice", type=int, default=DEFAULT_NICE_VALUE)
    ap.add_argument("--ionice", action="store_true")
    ap.add_argument("--affinity", type=str, default=None)
    ap.add_argument("--maxtasksperchild", type=int, default=DEFAULT_MAXTASKSPERCHILD)
    args = ap.parse_args()
    
    requested = args.workers if args.workers is not None else mp.cpu_count()
    num_workers = _clamp_workers(requested)
    affinity = parse_affinity(args.affinity)
    
    train_multiprocess(num_workers=num_workers,
                       total_episodes_per_worker=args.episodes_per_worker,
                       sync_interval=args.sync_interval,
                       max_steps=args.max_steps,
                       nice_value=args.nice,
                       ionice_flag=args.ionice,
                       affinity=affinity,
                       headless=True,
                       maxtasksperchild=args.maxtasksperchild)

if __name__ == "__main__":
    mp.freeze_support()
    main()