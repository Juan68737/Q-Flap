# Flappy Bird — Reinforcement Learning

Teaching an agent to play Flappy Bird, organized as a small production-style RL
codebase. Two algorithms live here:

- **Deep Q-Network (DQN)** — the one that plays well. Trained weights ship in
  `models/dqn_final.pth`, so you can watch it play immediately.
- **Tabular Q-learning** — the classic bins-and-a-Q-table approach, kept as a
  second experiment (`scripts/train_qlearn.py`).

Game sprites/physics originally from
[tech-with-tim / NEAT-Flappy-Bird](https://github.com/techwithtim/NEAT-Flappy-Bird).
[Live demo video.](https://docs.google.com/presentation/d/1QmJBoLjd7lCqaK8SakvsAAFlDFRR_EzajVqxlQBpZKQ/edit?usp=sharing)

---

## TL;DR

Everything runs through [`just`](https://github.com/casey/just) (`brew install just`):

```bash
just bootstrap     # create .venv and install everything (one time)
just eval 5        # watch the trained agent play 5 games
just train         # train your own and replace models/dqn_final.pth
```

`just` (no args) lists every command.

---

## How this works — where does the "data" come from?

**There is no dataset to download.** This is the key idea of reinforcement
learning: the agent *generates its own training data by playing the game*.

When you train, this loop runs millions of times:

```
   ┌─────────────────────────────────────────────────────────────┐
   │                                                             │
   ▼                                                             │
 ENV plays a frame ──► produces an experience ──► REPLAY BUFFER  │
 (Flappy Bird)         (state, action, reward,    (last ~1,000,000│
   ▲                    next_state, done)          experiences,   │
   │                                               kept in RAM)   │
   │                                                    │         │
   │                                                    ▼         │
   └──── AGENT chooses the next action ◄──── learns from random ──┘
         (the neural network)                batches of the buffer
```

- **State** = 7 numbers describing the bird vs. the next pipe (height, velocity,
  distance to pipe, gap position…). Computed by `envs/wrappers.py:get_state`.
- **Action** = flap or don't (2 choices).
- **Reward** = tiered by *where* the bird flies: dying is negative, passing near
  an edge is a small positive ("ok"), passing through the **middle** of the gap
  is the full reward ("perfect"). Defined in `envs/wrappers.py:shaped_reward`.

The replay buffer is **temporary, in-memory data** — discarded when training
ends. What gets *saved* to disk is the trained neural network (a `.pth` file).
**That `.pth` file is the only thing the game needs to play** — it's the agent's
"brain."

```
playing ─► experiences ─► replay buffer (RAM) ─► trained network ─► saved as .pth ─► `just eval` plays it
```

---

## Setup

Install the [`just`](https://github.com/casey/just) task runner, then bootstrap:

```bash
brew install just     # macOS  (Linux: see just's README; Windows: scoop install just)
just bootstrap        # creates .venv and installs the project + extras
source .venv/bin/activate   # optional — the just recipes use .venv automatically
```

- If `torch` fails to install on a brand-new Python, use a Python 3.12/3.13
  interpreter, or `pip install torch` on its own first.
- Managing environments:
  - `just venv` — (re)create the default `.venv`
  - `just venv experimentA` — create/install a named venv at `.venvs/experimentA`
  - `just venvs` (or `just view`) — list your venvs
  - venv layout differs by OS (`.venv/bin` on mac/Linux, `.venv/Scripts` on
    Windows); the recipes handle the python path automatically.

Under the hood `just` calls a Typer CLI you can also use directly:
`flappy-rl train`, `flappy-rl eval 5`.

---

## Walkthrough: your first few minutes

**1. Watch the pre-trained agent** (uses the shipped `models/dqn_final.pth`):

```bash
just eval 5           # 5 games, windowed
# just play           # same idea, quick 5-game watch
```

**2. Do a short training run** to see the whole pipeline produce a model. Full
training is ~10M steps (hours on CPU); for a first look, cap it with `--steps`
and save it under a throwaway name so it doesn't touch your good model:

```bash
just train test1 --steps 200000
```

You'll see lines like:

```
Device: cpu | run dir: experiments/dqn_baseline_2026-07-08_00-00-00
[step    120000] loss=0.0123 mean_q=2.41 eps=0.83 buffer=120000 avg_episode_score=0.4
[step    240000] loss=0.0098 mean_q=3.02 eps=0.71 buffer=240000 avg_episode_score=1.1
```

- `eps` = how random it still is (1.0 = all random, decays as it learns).
- `avg_episode_score` = pipes cleared per game — the number you want going up.

**3. It automatically keeps the best model.** Training tracks the best-scoring
snapshot (so a late collapse can't cost you a good model) and promotes it:

```
experiments/dqn_baseline_<timestamp>/
├── config.json          # exact hyperparameters + git commit (reproducible)
├── tb/                  # TensorBoard logs
├── steps_500000.pth     # periodic full checkpoints
├── best.pth             # best snapshot by score  ← promoted for you
├── final.pth            # last full checkpoint (for resuming)
└── policy_final.pth     # last weights only
```

`just train test1` copies `best.pth` to `models/test1.pth`.

**4. Watch your model, then bless it if it's good:**

```bash
just eval 5 test1.pth        # watch models/test1.pth
just train                   # a full run that replaces models/dqn_final.pth
```

`just train` (no name) promotes the best model straight to `models/dqn_final.pth`
— the default everything plays. `experiments/` is git-ignored (regenerable
output); `models/` is version-tracked (the models you chose to keep).

---

## Command reference

| Command | What it does |
|---|---|
| `just` | list all recipes |
| `just bootstrap` | create `.venv` + install everything |
| `just eval 5` | play 5 games with `models/dqn_final.pth` |
| `just eval 6 dqn2.pth` | play 6 games with `models/dqn2.pth` |
| `just play` | quick 5-game windowed watch |
| `just train` | train, replace `models/dqn_final.pth` with the best snapshot |
| `just train dqn2` | train, save best to `models/dqn2.pth` (leaves dqn_final alone) |
| `just train dqn2 --steps 200000` | shorter run |
| `just train-qlearn --workers 8` | run the tabular Q-learning experiment |
| `just test` | run the test suite |
| `just venv NAME` / `just venvs` | manage / list virtualenvs |
| `just clean` | delete generated runs under `experiments/` |

Watch training live: `tensorboard --logdir experiments`.

---

## Changing how it trains — one file, no magic numbers

**Every hyperparameter lives in `configs/dqn.yaml`.** Common edits:

| Want to… | Change |
|---|---|
| Train shorter (quick test) | lower `total_env_steps` (or pass `--steps`) |
| Learn faster (riskier) | raise `lr` |
| Explore more / longer | raise `eps_decay_steps` |
| Reward the middle harder | raise `center_bonus_w`, lower `pass_edge_floor` |
| More parallel games | `num_envs` |

To make the **game itself** easier/harder, edit the constants at the top of
`src/flappy_rl/envs/flappy.py` (`PIPE_GAP`, `GRAVITY_ACCEL`, …). Full guide:
[`docs/difficulty.md`](docs/difficulty.md).

---

## Project layout

```
justfile                    # task runner: bootstrap / train / eval / venv / test
configs/dqn.yaml            # ALL hyperparameters (no magic numbers in code)
src/flappy_rl/
  cli.py                    # Typer command line (flappy-rl train / eval)
  config.py                 # YAML -> frozen Config dataclass
  training.py               # DQN training loop (+ best-checkpoint tracking & promote)
  evaluation.py             # play/eval loop
  envs/
    flappy.py               # raw game: Bird / Pipe / Base physics + rendering
    wrappers.py             # observation (get_state) + reward shaping  ← the "data" definition
    vec_env.py              # multiprocess vectorized env (runs 15 games in parallel)
  models/networks.py        # QNet (nn.Module only, no training logic)
  buffers/replay.py         # experience replay buffer (the in-RAM training data)
  agents/
    base.py                 # Agent interface: act / update / save / load
    dqn.py                  # Double-DQN agent + a pure, testable loss function
  utils/
    seeding.py              # seed python / numpy / torch in one place
    checkpoint.py           # run dirs + self-describing checkpoints (weights+opt+rng+config+git sha)
    logging.py              # scalar logging (TensorBoard if installed, else console)
scripts/train_qlearn.py     # tabular Q-learning trainer (the non-DQN approach)
tests/test_dqn_smoke.py     # update runs without NaN + loss decreases on a trivial task
models/dqn_final.pth        # curated, version-tracked trained weights ("the brain")
imgs/                       # sprites
docs/difficulty.md          # what each env difficulty knob does
experiments/                # git-ignored: all generated runs/checkpoints/logs
```

Rule of thumb for where a file goes: **if deleting it and re-running code would
regenerate it, it's an artifact** → `experiments/` (git-ignored). Everything in
`src/`, `configs/`, `scripts/`, `tests/` is small, versioned, and reviewed.

---

## The observation & network (keep these in sync)

The DQN sees 7 normalized features of the bird vs. the next pipe (`bird_y`,
`velocity`, `dx to pipe`, `offset from gap center`, `gap size`, `top-relative`,
`bottom-relative`) and chooses one of two actions (flap / don't). The saved
weights are an MLP `7 → 512 → 512 → 2`. Three places must agree — change one,
change all three:

- `configs/dqn.yaml` → `state_dim`, `hidden`, `n_actions`
- `src/flappy_rl/envs/wrappers.py` → `get_state` (produces the 7 features)
- `src/flappy_rl/models/networks.py` → `QNet` (consumes them)
