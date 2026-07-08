# Making Flappy Bird Easier for AI

## Quick Answer

Edit the settings at the top of `src/flappy_rl/envs/flappy.py`:

```python
# EASIEST SETTINGS (Beginner AI)
PIPE_GAP = 300              # Huge gap
PIPE_VELOCITY = 3           # Slow pipes
GRAVITY_ACCEL = 1.0         # Gentle gravity
JUMP_STRENGTH = -8.0        # Soft jump
COLLISION_SHRINK = 0.75     # Forgiving hitbox
```

## All Difficulty Parameters

### 1. **PIPE_GAP** - Gap Size Between Pipes

The most important parameter!

```python
PIPE_GAP = 250  # Current easy setting
```

| Value   | Difficulty | Description                |
| ------- | ---------- | -------------------------- |
| 350+    | Tutorial   | Almost impossible to fail  |
| 280-320 | Very Easy  | Great for initial learning |
| 250-280 | Easy       | Good learning balance      |
| 220-250 | Medium     | Requires skill             |
| 200-220 | Normal     | Standard game              |
| 180-200 | Hard       | Expert level               |
| <180    | Very Hard  | Nearly impossible          |

**Recommendation**: Start with 280-300 for Q-learning

---

### 2. **PIPE_VELOCITY** - How Fast Pipes Move

```python
PIPE_VELOCITY = 4  # Current easy setting
```

| Value | Difficulty | Effect                      |
| ----- | ---------- | --------------------------- |
| 2-3   | Very Easy  | Slow, lots of reaction time |
| 4     | Easy       | Comfortable pace            |
| 5     | Normal     | Standard game               |
| 6-7   | Hard       | Quick reactions needed      |
| 8+    | Very Hard  | Extremely fast              |

**Recommendation**: Use 3-4 for easier learning

---

### 3. **GRAVITY_ACCEL** - Fall Speed

```python
GRAVITY_ACCEL = 1.2  # Current easy setting
```

| Value   | Difficulty | Effect                   |
| ------- | ---------- | ------------------------ |
| 0.8-1.0 | Very Easy  | Floaty, gentle falling   |
| 1.2     | Easy       | Slightly reduced gravity |
| 1.5     | Normal     | Standard physics         |
| 1.8-2.0 | Hard       | Heavy, fast falling      |
| 2.5+    | Very Hard  | Drops like a rock        |

**Recommendation**: Use 1.0-1.2 for easier control

---

### 4. **JUMP_STRENGTH** - How High the Bird Jumps

```python
JUMP_STRENGTH = -9.0  # Current easy setting
```

| Value     | Difficulty | Effect                    |
| --------- | ---------- | ------------------------- |
| -7 to -8  | Easy       | Gentle, controlled jumps  |
| -9 to -10 | Medium     | Moderate jumps            |
| -10.5     | Normal     | Standard jump             |
| -12+      | Hard       | Powerful, hard to control |

**Recommendation**: Use -8 to -9 for smoother control

---

### 5. **COLLISION_SHRINK** - Hitbox Forgiveness

```python
COLLISION_SHRINK = 0.85  # Current easy setting
```

| Value     | Difficulty | Effect                      |
| --------- | ---------- | --------------------------- |
| 0.5-0.7   | Very Easy  | Tiny hitbox, very forgiving |
| 0.75-0.85 | Easy       | Reduced hitbox              |
| 0.9-0.95  | Medium     | Slightly smaller            |
| 1.0       | Normal     | Exact collision             |

**Recommendation**: Use 0.75-0.85 for easier training

---

### 6. **MIN/MAX_PIPE_HEIGHT** - Pipe Spawn Range

```python
MIN_PIPE_HEIGHT = 80   # Current easy setting
MAX_PIPE_HEIGHT = 400  # Current easy setting
```

**Normal**: 50-450 (pipes can spawn very high or low)  
**Easy**: 100-400 (more centered pipes)  
**Very Easy**: 150-350 (only middle area)

This prevents pipes from spawning too close to floor/ceiling.

---

## Difficulty Presets

### BEGINNER MODE (For Initial Training)

```python
PIPE_GAP = 300
PIPE_VELOCITY = 3
GRAVITY_ACCEL = 1.0
JUMP_STRENGTH = -8.0
TERMINAL_VELOCITY = 12
COLLISION_SHRINK = 0.75
MIN_PIPE_HEIGHT = 100
MAX_PIPE_HEIGHT = 400
```

**Expected Performance**: 20-50 pipes

---

### EASY MODE (Current Settings)

```python
PIPE_GAP = 250
PIPE_VELOCITY = 4
GRAVITY_ACCEL = 1.2
JUMP_STRENGTH = -9.0
TERMINAL_VELOCITY = 14
COLLISION_SHRINK = 0.85
MIN_PIPE_HEIGHT = 80
MAX_PIPE_HEIGHT = 400
```

**Expected Performance**: 15-40 pipes

---

### MEDIUM MODE

```python
PIPE_GAP = 220
PIPE_VELOCITY = 4.5
GRAVITY_ACCEL = 1.35
JUMP_STRENGTH = -10.0
TERMINAL_VELOCITY = 15
COLLISION_SHRINK = 0.92
MIN_PIPE_HEIGHT = 70
MAX_PIPE_HEIGHT = 420
```

**Expected Performance**: 10-25 pipes

---

### NORMAL MODE (Original Game)

```python
PIPE_GAP = 200
PIPE_VELOCITY = 5
GRAVITY_ACCEL = 1.5
JUMP_STRENGTH = -10.5
TERMINAL_VELOCITY = 16
COLLISION_SHRINK = 1.0
MIN_PIPE_HEIGHT = 50
MAX_PIPE_HEIGHT = 450
```

**Expected Performance**: 5-15 pipes

---

## How to Use Easy Mode

### 1. Edit Difficulty Settings

Open `src/flappy_rl/envs/flappy.py` and change the values at the top:

```python
# ==================== DIFFICULTY SETTINGS ====================
PIPE_GAP = 300  # Make it easier!
PIPE_VELOCITY = 3
# ... etc
```

### 2. Train with Easy Mode

```bash
python scripts/train_qlearn.py --workers 15 --episodes-per-worker 3000
```

This will:

- Use settings from `src/flappy_rl/envs/flappy.py`
- Save results to `exports/` folder
- Show difficulty settings at startup

### 3. Play/Test Your Trained Agent

```bash
# Play with the trained easy-mode Q-table
python scripts/evaluate.py   # (DQN agent; the tabular runs have no dedicated viewer)
```

---

## Progressive Training Strategy

Train in stages with increasing difficulty:

### Stage 1: Learn Basic Behavior (BEGINNER)

```python
PIPE_GAP = 300, PIPE_VELOCITY = 3
```

Train: 2000 episodes  
Goal: Learn to flap and avoid pipes

### Stage 2: Improve Precision (EASY)

```python
PIPE_GAP = 250, PIPE_VELOCITY = 4
```

Train: 3000 episodes (start from Stage 1 Q-table)  
Goal: Navigate smaller gaps

### Stage 3: Reach Normal Difficulty (MEDIUM)

```python
PIPE_GAP = 220, PIPE_VELOCITY = 4.5
```

Train: 5000 episodes (start from Stage 2 Q-table)  
Goal: Handle standard game

### Stage 4: Master the Game (NORMAL)

```python
PIPE_GAP = 200, PIPE_VELOCITY = 5
```

Train: 10000+ episodes (start from Stage 3 Q-table)  
Goal: Expert performance

---

## Comparison: Easy vs Normal

| Metric         | Normal Mode | Easy Mode           |
| -------------- | ----------- | ------------------- |
| Gap Size       | 200px       | 250px (+25%)        |
| Pipe Speed     | 5 px/frame  | 4 px/frame (-20%)   |
| Gravity        | 1.5         | 1.2 (-20%)          |
| Jump           | -10.5       | -9.0 (-14%)         |
| Collision      | 1.0x        | 0.85x (15% smaller) |
| **Difficulty** | ⭐⭐⭐⭐⭐  | ⭐⭐                |

With easy mode, the AI has:

- 25% more space to fly through
- 20% more time to react
- Gentler physics (easier to control)
- 15% forgiveness on collisions

---

## Troubleshooting

### AI Still Can't Learn

- **Increase gap to 300+**
- **Reduce velocity to 3**
- **Make collision even more forgiving (0.7)**

### AI Learns Too Slowly

- **Train longer** (5000+ episodes)
- **Check epsilon decay** (should reach 5%)
- **Verify state discretization** (not too many bins)

### AI Does Well in Easy Mode but Fails in Normal

- **Use progressive training** (increase difficulty gradually)
- **Transfer learning** (start with easy-mode Q-table)
- **Train longer at each stage**

### Want Maximum Learning Speed

Use **BEGINNER preset** with:

```python
PIPE_GAP = 350
PIPE_VELOCITY = 2.5
COLLISION_SHRINK = 0.6
```

---

## Example: Customizing Difficulty

Let's say you want medium-easy difficulty:

```python
# Edit src/flappy_rl/envs/flappy.py
PIPE_GAP = 260           # Between easy and medium
PIPE_VELOCITY = 4        # Easy speed
GRAVITY_ACCEL = 1.3      # Medium gravity
JUMP_STRENGTH = -9.5     # Medium jump
COLLISION_SHRINK = 0.88  # Slightly forgiving
MIN_PIPE_HEIGHT = 90
MAX_PIPE_HEIGHT = 410
```

Then train:

```bash
python scripts/train_qlearn.py --workers 12 --episodes-per-worker 4000
```

---

## Tips for Best Results

1. **Start VERY easy** - Gap=300, Vel=3
2. **Train until convergent** - Watch average score plateau
3. **Gradually increase difficulty** - Don't jump from easy to hard
4. **Save Q-tables at each stage** - Use them as starting points
5. **Monitor epsilon decay** - Should reach minimum (5%)
6. **Test frequently** - Use `scripts/evaluate.py` to watch the DQN agent

---

## Quick Cheat Sheet

| Goal                  | Gap | Velocity | Gravity |
| --------------------- | --- | -------- | ------- |
| Fast initial learning | 300 | 3        | 1.0     |
| Balanced easy mode    | 250 | 4        | 1.2     |
| Challenge the AI      | 220 | 4.5      | 1.35    |
| Match original game   | 200 | 5        | 1.5     |
| Extreme difficulty    | 180 | 6        | 1.8     |
