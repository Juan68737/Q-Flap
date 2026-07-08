"""Evaluation loop: load a trained policy and play (optionally in a window).

Algorithm-agnostic — it only needs a network mapping observations to action
values. Accepts either a weights-only file (from agent.save) or a full training
checkpoint (from utils.checkpoint.save_checkpoint).
"""

import numpy as np
import pygame
import torch

from .config import Config
from .models.networks import QNet
from .envs.flappy import Bird, Pipe, Base, WIN_WIDTH, WIN_HEIGHT, FLOOR, ASSETS_DIR
from .envs.wrappers import get_state


def load_policy(model_path, cfg, device):
    net = QNet(cfg.state_dim, cfg.hidden, cfg.n_actions).to(device)
    obj = torch.load(model_path, map_location=device)
    state_dict = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    net.load_state_dict(state_dict)
    net.eval()
    return net


def _run_episode(net, win, clock, fps, bg_img, device, action_repeat, max_steps):
    bird, base, pipes = Bird(230, 350), Base(FLOOR), [Pipe(700)]
    score, done = 0, False
    frame, action = 0, 0
    # A strong agent can survive indefinitely, so cap the episode (0 = unlimited).
    while not done and (max_steps <= 0 or frame < max_steps):
        clock.tick(fps)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        # Re-decide every `action_repeat` frames — the rhythm the agent trained with.
        if frame % action_repeat == 0:
            obs = torch.as_tensor(get_state(bird, pipes), dtype=torch.float32,
                                  device=device).unsqueeze(0)
            with torch.no_grad():
                action = int(net(obs).argmax(1).item())
        frame += 1
        if action == 1:
            bird.jump()

        bird.move()
        base.move()
        rem, add_pipe = [], False
        for pipe in pipes:
            pipe.move()
            if pipe.collide(bird, win):
                done = True
                break
            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)
            if not pipe.passed and pipe.x < bird.x:
                pipe.passed = True
                add_pipe = True
        if add_pipe:
            score += 1
            pipes.append(Pipe(WIN_WIDTH))
        for r in rem:
            pipes.remove(r)
        if bird.y + bird.img.get_height() >= FLOOR or bird.y < -50:
            done = True

        win.blit(bg_img, (0, 0))
        for pipe in pipes:
            pipe.draw(win)
        base.draw(win)
        bird.draw(win)
        font = pygame.font.SysFont("comicsans", 40)
        win.blit(font.render(f"Score: {score}", True, (255, 255, 255)), (10, 10))
        pygame.display.update()
    return score


def evaluate(model_path, cfg: Config, episodes=10, fps=45, max_steps=2000, action_repeat=None):
    """Play `episodes` games with the model and print scores. Returns the list."""
    if action_repeat is None:
        action_repeat = cfg.action_repeat
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pygame.init()
    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    pygame.display.set_caption("Flappy Bird - DQN Agent")
    clock = pygame.time.Clock()
    bg_img = pygame.transform.scale(
        pygame.image.load(str(ASSETS_DIR / "bg.png")), (WIN_WIDTH, WIN_HEIGHT))

    net = load_policy(model_path, cfg, device)
    print(f"Loaded {model_path} on {device} (action_repeat={action_repeat})")

    scores = []
    for i in range(episodes):
        score = _run_episode(net, win, clock, fps, bg_img, device, action_repeat, max_steps)
        scores.append(score)
        print(f"Episode {i + 1}/{episodes}: score = {score}")

    print("=" * 40)
    print(f"Episodes: {len(scores)} | avg: {np.mean(scores):.2f} | best: {int(np.max(scores))}")
    pygame.quit()
    return scores
