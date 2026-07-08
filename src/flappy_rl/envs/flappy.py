"""
Flappy Bird game environment: raw physics + rendering.

`Bird` / `Pipe` / `Base` are the game objects; scripts drive them directly.
Difficulty knobs are the constants below (they define the *environment*, so
they live here rather than in the RL config). See docs/difficulty.md.

Runs headless automatically when SDL_VIDEODRIVER=dummy or FB_NO_DISPLAY=1
(training sets this). Sprites are resolved relative to the repo root, so the
env loads correctly no matter what the current working directory is.
"""

import os
import random
from pathlib import Path

import pygame

pygame.font.init()

WIN_WIDTH = 600
WIN_HEIGHT = 800
FLOOR = 730

# ==================== DIFFICULTY SETTINGS ====================
# PIPE DIFFICULTY
PIPE_GAP = 250          # Default: 200 | Easier: 250-300 | Harder: 150-180
PIPE_VELOCITY = 4       # Default: 5   | Easier: 3-4     | Harder: 6-8

# BIRD PHYSICS
GRAVITY_ACCEL = 1.2     # Default: 1.5   | Easier: 0.8-1.2 | Harder: 2.0+
JUMP_STRENGTH = -9.0    # Default: -10.5 | Easier: -8..-9  | Harder: -12+
TERMINAL_VELOCITY = 14  # Default: 16    | Easier: 12-14   | Harder: 18+

# COLLISION TOLERANCE (lower = more forgiving; bird hitbox shrinks)
COLLISION_SHRINK = 0.85  # Default: 1.0 | Easier: 0.7-0.9 | Range: 0.5-1.0

# PIPE SPACING
MIN_PIPE_HEIGHT = 80    # Default: 50  | Easier: 100-150 | Range: 50-200
MAX_PIPE_HEIGHT = 400   # Default: 450 | Easier: 350-400 | Range: 300-450
# =============================================================

# imgs/ lives at the repo root: .../src/flappy_rl/envs/flappy.py -> parents[3]
ASSETS_DIR = Path(os.environ.get(
    "FLAPPY_ASSETS", Path(__file__).resolve().parents[3] / "imgs"))

_HEADLESS = (os.environ.get("SDL_VIDEODRIVER") == "dummy"
             or os.environ.get("FB_NO_DISPLAY") == "1")


def _load(name):
    return pygame.image.load(str(ASSETS_DIR / name))


if not _HEADLESS:
    WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    pygame.display.set_caption("Flappy Bird")
    pipe_img = pygame.transform.scale2x(_load("pipe.png").convert_alpha())
    bg_img = pygame.transform.scale(_load("bg.png").convert_alpha(), (600, 900))
    bird_images = [pygame.transform.scale2x(_load(f"bird{x}.png")) for x in range(1, 4)]
    base_img = pygame.transform.scale2x(_load("base.png").convert_alpha())
else:
    # Headless: dummy surfaces so training needs no display.
    WIN = None
    pipe_img = pygame.Surface((52, 320)); pipe_img.fill((0, 255, 0))
    bg_img = pygame.Surface((600, 900)); bg_img.fill((135, 206, 250))
    bird_images = [pygame.Surface((34, 24)) for _ in range(3)]
    for img in bird_images:
        img.fill((255, 255, 0))
    base_img = pygame.Surface((336, 112)); base_img.fill((222, 216, 149))


class Bird:
    """Bird with configurable physics."""
    MAX_ROTATION = 25
    IMGS = bird_images
    ROT_VEL = 20
    ANIMATION_TIME = 5

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0
        self.vel = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.IMGS[0]

    def jump(self):
        self.vel = JUMP_STRENGTH
        self.tick_count = 0
        self.height = self.y

    def move(self):
        self.tick_count += 1
        displacement = self.vel * 1 + GRAVITY_ACCEL * (self.tick_count)
        if displacement >= TERMINAL_VELOCITY:
            displacement = TERMINAL_VELOCITY
        if displacement < 0:
            displacement -= 2
        self.y = self.y + displacement

        if displacement < 0 or self.y < self.height + 50:
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        else:
            if self.tilt > -90:
                self.tilt -= self.ROT_VEL

    def draw(self, win):
        if win is None:
            return
        self.img_count += 1
        if self.img_count <= self.ANIMATION_TIME:
            self.img = self.IMGS[0]
        elif self.img_count <= self.ANIMATION_TIME * 2:
            self.img = self.IMGS[1]
        elif self.img_count <= self.ANIMATION_TIME * 3:
            self.img = self.IMGS[2]
        elif self.img_count <= self.ANIMATION_TIME * 4:
            self.img = self.IMGS[1]
        elif self.img_count == self.ANIMATION_TIME * 4 + 1:
            self.img = self.IMGS[0]
            self.img_count = 0
        if self.tilt <= -80:
            self.img = self.IMGS[1]
            self.img_count = self.ANIMATION_TIME * 2
        blitRotateCenter(win, self.img, (self.x, self.y), self.tilt)

    def get_mask(self):
        """Collision mask, optionally shrunk (COLLISION_SHRINK) for forgiveness."""
        if COLLISION_SHRINK >= 1.0:
            return pygame.mask.from_surface(self.img)
        original_size = self.img.get_size()
        new_width = int(original_size[0] * COLLISION_SHRINK)
        new_height = int(original_size[1] * COLLISION_SHRINK)
        shrunk_img = pygame.transform.scale(self.img, (new_width, new_height))
        padded_surface = pygame.Surface(original_size, pygame.SRCALPHA)
        padded_surface.fill((0, 0, 0, 0))
        offset_x = (original_size[0] - new_width) // 2
        offset_y = (original_size[1] - new_height) // 2
        padded_surface.blit(shrunk_img, (offset_x, offset_y))
        return pygame.mask.from_surface(padded_surface)


class Pipe:
    """Pipe with configurable difficulty."""
    GAP = PIPE_GAP
    VEL = PIPE_VELOCITY

    def __init__(self, x):
        self.x = x
        self.height = 0
        self.top = 0
        self.bottom = 0
        self.PIPE_TOP = pygame.transform.flip(pipe_img, False, True)
        self.PIPE_BOTTOM = pipe_img
        self.passed = False
        self.set_height()

    def set_height(self):
        self.height = random.randrange(MIN_PIPE_HEIGHT, MAX_PIPE_HEIGHT)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP

    def move(self):
        self.x -= self.VEL

    def draw(self, win):
        if win is None:
            return
        win.blit(self.PIPE_TOP, (self.x, self.top))
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    def collide(self, bird, win):
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)
        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))
        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        t_point = bird_mask.overlap(top_mask, top_offset)
        return bool(b_point or t_point)


class Base:
    """Moving floor (velocity matches pipes)."""
    VEL = PIPE_VELOCITY
    WIDTH = base_img.get_width()
    IMG = base_img

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):
        self.x1 -= self.VEL
        self.x2 -= self.VEL
        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH
        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self, win):
        if win is None:
            return
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))


def blitRotateCenter(surf, image, topleft, angle):
    if surf is None:
        return
    rotated_image = pygame.transform.rotate(image, angle)
    new_rect = rotated_image.get_rect(center=image.get_rect(topleft=topleft).center)
    surf.blit(rotated_image, new_rect.topleft)


def print_difficulty_settings():
    print("\n" + "=" * 60)
    print("FLAPPY BIRD - DIFFICULTY SETTINGS")
    print("=" * 60)
    print(f"Pipe Gap:          {PIPE_GAP} pixels (default: 200)")
    print(f"Pipe Velocity:     {PIPE_VELOCITY} px/frame (default: 5)")
    print(f"Gravity:           {GRAVITY_ACCEL} px/frame^2 (default: 1.5)")
    print(f"Jump Strength:     {JUMP_STRENGTH} px/frame (default: -10.5)")
    print(f"Terminal Velocity: {TERMINAL_VELOCITY} px/frame (default: 16)")
    print(f"Collision Shrink:  {COLLISION_SHRINK}x (default: 1.0)")
    print(f"Pipe Height Range: {MIN_PIPE_HEIGHT}-{MAX_PIPE_HEIGHT} (default: 50-450)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    print_difficulty_settings()
