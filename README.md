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

```bash
pip install -e .            # one-time setup
python scripts/evaluate.py  # watch the trained agent play right now
python scripts/train.py --config configs/dqn.yaml   # train your own
```

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
- **Reward** = +10 for passing a pipe, a small bonus for staying centered,
  −1 for dying. Defined in `envs/wrappers.py:shaped_reward`.

The replay buffer is **temporary, in-memory data** — it's discarded when
training ends. What gets *saved* to disk is the trained neural network (a
`.pth` file). **That `.pth` file is the only thing the game needs to play** —
it's the agent's "brain."

```
playing ─► experiences ─► replay buffer (RAM) ─► trained network ─► saved as .pth ─► evaluate.py plays with it
```

---

## Install (one time)

```bash
cd Flap_Flap
python3 -m venv venv && source venv/bin/activate   # optional but recommended
pip install -e .            # installs the flappy_rl package + torch, pygame, numpy, pyyaml
```

`pip install -e .` is what makes `import flappy_rl...` work everywhere (scripts,
tests, notebooks, debugger) with no `sys.path` hacks. Do it once.

- CPU / macOS: if the `torch` install errors, install it plainly first
  (`pip install torch`), then re-run `pip install -e .`.
- Optional extras: `pip install -e ".[logging,dev]"` (TensorBoard + pytest).

---

## Walkthrough: your first few minutes

**1. Watch the pre-trained agent play** (nothing to train — uses the shipped model):

```bash
python scripts/evaluate.py
```

A window opens and the bird flies through pipes for 50 games, printing each
score and a final average/best.

**2. Do a *short* training run** to see the whole pipeline produce a model.
Full training is ~10M steps (hours on CPU); for a first look, lower it. Either
edit `total_env_steps` in `configs/dqn.yaml` to `200000`, or just run it and
stop early with `Ctrl-C` (it saves on interrupt):

```bash
python scripts/train.py --config configs/dqn.yaml
```

You'll see console lines like:

```
Device: cpu | run dir: experiments/dqn_baseline_2026-07-07_23-40-12
[step    120000] loss=0.0123 mean_q=2.41 eps=0.83 buffer=120000 avg_episode_score=0.4
[step    240000] loss=0.0098 mean_q=3.02 eps=0.71 buffer=240000 avg_episode_score=1.1
```

- `eps` = how random it still is (starts at 1.0 = all random, decays as it learns).
- `avg_episode_score` = pipes cleared per game — this is the number you want going up.

**3. Find what it produced.** Every run gets its own timestamped folder:

```
experiments/dqn_baseline_2026-07-07_23-40-12/
├── config.json          # exact hyperparameters + git commit used (reproducible)
├── tb/                  # TensorBoard logs
├── steps_500000.pth     # checkpoint saved every save_every_steps
├── final.pth            # full checkpoint (weights + optimizer + RNG — for resuming)
└── policy_final.pth     # weights only — THIS is what evaluate.py loads
```

**4. Watch *your* newly trained agent:**

```bash
python scripts/evaluate.py --model experiments/dqn_baseline_2026-07-07_23-40-12/policy_final.pth
```

**5. Like it? Make it the default** the game uses:

```bash
cp experiments/dqn_baseline_<timestamp>/policy_final.pth models/dqn_final.pth
```

> `experiments/` is git-ignored on purpose — it's regenerable output, not source.

---

## Watch training live (optional)

```bash
pip install -e ".[logging]"       # adds tensorboard
tensorboard --logdir experiments  # open the URL it prints
```

Charts for loss, mean Q-value, epsilon, and average episode score. Without it,
the same numbers print to the console.

---

## Changing how it trains — one file, no magic numbers

**Every hyperparameter lives in `configs/dqn.yaml`.** Common edits:

| Want to… | Change |
|---|---|
| Train shorter (quick test) | lower `total_env_steps` |
| Learn faster (riskier) | raise `lr` |
| Explore more / longer | raise `eps_decay_steps` |
| Change the reward | `reward_pass_pipe`, `center_bonus_w`, … |
| More parallel games | `num_envs` |

To make the **game itself** easier/harder, edit the constants at the top of
`src/flappy_rl/envs/flappy.py` (`PIPE_GAP`, `GRAVITY_ACCEL`, …). Full guide:
[`docs/difficulty.md`](docs/difficulty.md).

---

## The tabular Q-learning experiment

```bash
python scripts/train_qlearn.py --workers 8 --episodes-per-worker 5000
# -> writes Q-tables (.npy) to experiments/qlearn/
```

Note: earlier runs of this one scored 0 pipes — it's kept for you to improve.
The **DQN** is the approach that actually plays well.

---

## Project layout

```
configs/dqn.yaml            # ALL hyperparameters (no magic numbers in code)
src/flappy_rl/
  config.py                 # YAML -> frozen Config dataclass
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
scripts/
  train.py                  # thin DQN training loop: act, step, store, update, log, save
  evaluate.py               # load a checkpoint, play greedy episodes with a window
  train_qlearn.py           # tabular Q-learning trainer
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

---

## Tests

```bash
pip install -e ".[dev]"
pytest      # checks the DQN update runs without NaN and the loss goes down
```
