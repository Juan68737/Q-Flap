"""Load a trained policy and watch it play (with a visible window).

    python scripts/evaluate.py                                  # models/dqn_final.pth
    python scripts/evaluate.py --model experiments/<run>/policy_final.pth --episodes 20

Algorithm-agnostic: it only needs a network that maps observations to action
values. Accepts either a weights-only file (from agent.save) or a full training
checkpoint (from utils.checkpoint.save_checkpoint).
"""

import argparse

import numpy as np
import pygame
import torch

from flappy_rl.config import Config
from flappy_rl.models.networks import QNet
from flappy_rl.envs.flappy import (
    Bird, Pipe, Base, WIN_WIDTH, WIN_HEIGHT, FLOOR, ASSETS_DIR)
from flappy_rl.envs.wrappers import get_state


def load_policy(model_path, cfg, device):
    net = QNet(cfg.state_dim, cfg.hidden, cfg.n_actions).to(device)
    obj = torch.load(model_path, map_location=device)
    state_dict = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    net.load_state_dict(state_dict)
    net.eval()
    return net


def run_episode(net, win, clock, fps, bg_img, device):
    bird, base, pipes = Bird(230, 350), Base(FLOOR), [Pipe(700)]
    score, done = 0, False
    while not done:
        clock.tick(fps)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        obs = torch.as_tensor(get_state(bird, pipes), dtype=torch.float32,
                              device=device).unsqueeze(0)
        with torch.no_grad():
            action = int(net(obs).argmax(1).item())
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


def main():
    ap = argparse.ArgumentParser(description="Watch a trained DQN play Flappy Bird")
    ap.add_argument("--model", default="models/dqn_final.pth")
    ap.add_argument("--config", default="configs/dqn.yaml")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--fps", type=int, default=45)
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pygame.init()
    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    pygame.display.set_caption("Flappy Bird - DQN Agent")
    clock = pygame.time.Clock()
    bg_img = pygame.transform.scale(
        pygame.image.load(str(ASSETS_DIR / "bg.png")), (WIN_WIDTH, WIN_HEIGHT))

    net = load_policy(args.model, cfg, device)
    print(f"Loaded {args.model} on {device}")

    scores = []
    for i in range(args.episodes):
        score = run_episode(net, win, clock, args.fps, bg_img, device)
        scores.append(score)
        print(f"Episode {i + 1}/{args.episodes}: score = {score}")

    print("=" * 40)
    print(f"Episodes: {len(scores)} | avg: {np.mean(scores):.2f} | best: {int(np.max(scores))}")
    pygame.quit()


if __name__ == "__main__":
    main()
