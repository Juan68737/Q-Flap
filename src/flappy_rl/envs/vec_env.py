"""Vectorized (multiprocess) Flappy Bird environment.

Each worker runs its own game in a separate process and auto-resets on death;
the main process drives them in lockstep via pipes. This is the data source
that feeds the replay buffer during DQN training.
"""

import random
from dataclasses import dataclass
from typing import List

import multiprocessing as mp
import numpy as np
import pygame

from .flappy import Bird, Pipe, Base, WIN_WIDTH, FLOOR
from .wrappers import get_state, shaped_reward


@dataclass
class StepResult:
    ns: np.ndarray      # next state
    r: float            # reward
    d: bool             # done
    passed: bool        # passed a pipe this step
    score_inc: int      # pipes passed this step (0/1)


def worker_proc(conn, seed: int, cfg):
    random.seed(seed)
    np.random.seed(seed)
    pygame.init()
    try:
        bird = Bird(230, random.randint(250, 450))
        base = Base(FLOOR)
        pipes = [Pipe(700)]
        ep_steps = 0

        def reset_env():
            nonlocal bird, base, pipes, ep_steps
            bird = Bird(230, random.randint(250, 450))
            base = Base(FLOOR)
            pipes = [Pipe(700)]
            ep_steps = 0
            return get_state(bird, pipes)

        s = get_state(bird, pipes)

        while True:
            cmd, payload = conn.recv()
            if cmd == "reset":
                s = reset_env()
                conn.send(s)
            elif cmd == "step":
                action = int(payload)
                total_passed = False
                done = False
                for _ in range(cfg.action_repeat):
                    if action == 1:
                        bird.jump()
                    bird.move()
                    base.move()
                    rem, add_pipe = [], False
                    for p in pipes:
                        p.move()
                        if p.collide(bird, None):
                            done = True
                            break
                        if p.x + p.PIPE_TOP.get_width() < 0:
                            rem.append(p)
                        if not p.passed and p.x < bird.x:
                            p.passed = True
                            add_pipe = True
                    if add_pipe:
                        pipes.append(Pipe(WIN_WIDTH))
                        total_passed = True
                    for rp in rem:
                        pipes.remove(rp)
                    if bird.y + bird.img.get_height() >= FLOOR or bird.y < -50:
                        done = True
                    if done:
                        break
                r = shaped_reward(cfg, not done, total_passed, bird, pipes, action)
                ns = get_state(bird, pipes) if not done else np.zeros_like(s, dtype=np.float32)
                result = StepResult(ns=ns, r=r, d=done, passed=total_passed,
                                    score_inc=(1 if total_passed else 0))
                conn.send(result)
                s = ns
                ep_steps += 1
                if done or ep_steps >= cfg.max_steps_per_ep:
                    s = reset_env()
            elif cmd == "close":
                break
    finally:
        try:
            conn.close()
        except Exception:
            pass
        pygame.quit()


class VecEnv:
    """Spawns `cfg.num_envs` worker processes and steps them in lockstep."""

    def __init__(self, cfg):
        self.n = cfg.num_envs
        self.parent_conns, self.child_conns = zip(*[mp.Pipe() for _ in range(self.n)])
        self.ps = []
        for i in range(self.n):
            p = mp.Process(target=worker_proc,
                           args=(self.child_conns[i], cfg.seed + i + 1, cfg),
                           daemon=True)
            p.start()
            self.ps.append(p)
        for c in self.child_conns:
            c.close()
        for pc in self.parent_conns:
            pc.send(("reset", None))
        self.states = [pc.recv() for pc in self.parent_conns]

    def step(self, actions: List[int]) -> List[StepResult]:
        for pc, a in zip(self.parent_conns, actions):
            pc.send(("step", int(a)))
        return [pc.recv() for pc in self.parent_conns]

    def reset(self):
        for pc in self.parent_conns:
            pc.send(("reset", None))
        self.states = [pc.recv() for pc in self.parent_conns]
        return self.states

    def close(self):
        for pc in self.parent_conns:
            try:
                pc.send(("close", None))
            except Exception:
                pass
        for p in self.ps:
            p.join(timeout=1.0)
            if p.is_alive():
                p.terminate()
