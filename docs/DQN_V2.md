# DQN V2 — upgrade notes

What changed from the original DQN ("V1"), why, and how to use it. Everything is
**config-gated**: `configs/dqn.yaml` is unchanged and still reproduces
`models/dqn_final.pth`; the new behavior lives in `configs/dqn_v2.yaml`.

```bash
just train dqn_v2 --config configs/dqn_v2.yaml      # train V2
just eval 5 dqn_v2.pth --config configs/dqn_v2.yaml  # watch it (config carries the arch)
```

---

## The framing (why these changes and not others)

The task is already solved *mid-run*: the 1.5M–4.5M-step checkpoints were
**immortal**, while the final checkpoint collapsed to ~1 pipe. So this is **not**
a learning or exploration problem — it's a **stability + robustness + deployment-
selection** problem. That rules out the hyped exploration pieces (Noisy Nets,
C51) and points at the stability-oriented ones.

**Likely collapse mechanism — catastrophic forgetting of death:** once the agent
is immortal, it stops crashing, so crash transitions age out of the 1M replay
ring. "What a wall looks like" decays from the training distribution, Q drifts,
it eventually clips a pipe, and the distribution shifts out from under it. This
framing is why PER below is doing *robustness* work (resurfacing rare deaths),
not just going faster.

---

## Changes at a glance

| # | Change | Type | Tier | Fixes / helps |
|---|---|---|---|---|
| 1 | EMA (Polyak) deploy weights | stability | 0 | checkpoint lottery / collapse |
| 2 | Greedy-eval checkpoint selection | selection | 0 | picking the wrong snapshot |
| 3 | Cosine LR decay | stability | 0 | late-training divergence |
| 4 | LayerNorm on hidden layers | stability | 1 | the deadly triad |
| 5 | Dueling head | representation | 1 | low-advantage value estimation |
| 6 | n-step returns (n=3) | credit assignment | 1 | slow reward propagation |
| 7 | Prioritized replay (sum-tree) | sampling | 2 | forgetting rare deaths |

Deliberately **skipped:** Noisy Nets (exploration isn't the bottleneck), C51
(prefer QR/IQN if we ever add distributional), full 7-piece Rainbow.

---

## Details

### Tier 0 — stability & selection (the actual collapse fix)

**1. EMA deploy weights** (`agents/dqn.py`, `ema_decay: 0.999`).
Keep a Polyak-averaged copy of the online weights and **deploy that**, not the
raw net. This is SWA-for-RL: it smooths over the checkpoint lottery so a single
bad late update doesn't define the shipped model. `agent.save()` writes the EMA
net when enabled.

**2. Greedy-eval checkpoint selection** (`training.py` + `evaluation.py:score_policy`,
`eval_every_steps: 250000`).
V1 chose `best.pth` by **ε-greedy training return** — a noisy behavior-policy
signal. V2 periodically runs a **true greedy eval** (ε=0, fixed seeds, unshaped
score) on the deploy net and banks the best by that. You now select on the thing
you actually care about.

**3. Cosine LR decay** (`training.py`, `lr_final: 0.0`).
Anneal `lr` → `lr_final` over the run. A flat `2.5e-4` late in training is a
common cause of slow divergence; decay both stabilizes and gives a natural
stopping point.

### Tier 1 — cheap algorithmic upgrades

**4. LayerNorm** (`models/networks.py`, `layernorm: true`).
Highest-ROI single change in modern value-based RL: it directly tames the deadly
triad (bootstrapping + off-policy + function approximation). ~3 lines.

**5. Dueling head** (`models/networks.py`, `dueling: true`).
`Q(s,a) = V(s) + (A(s,a) − mean_a A(s,a))`. Flappy is *low-advantage*: in most
states both actions survive and value is dominated by "am I alive and centered."
Decoupling V from A fits that and is nearly free.

**6. n-step returns** (`buffers/nstep.py`, `n_step: 3`).
`R = Σ_{k<n} γ^k r_k`, bootstrap with `γ^n · Q(s_n)`. With action-repeat and
dense shaping, 3-step targets propagate the pass reward faster. A per-env deque
folds transitions at insert time and flushes the tail on episode end.

### Tier 2 — the buffer upgrade (your "advanced data structure")

**7. Prioritized experience replay via a sum-tree**
(`buffers/sumtree.py`, `buffers/prioritized.py`, `prioritized: true`).
Storage stays the pre-allocated ring; a **SumTree** over `priority^α` drives
sampling and a **MinTree** gives the max importance-sampling weight.

- **Sampling** = draw `u ~ U(0, total)`, descend from the root picking the child
  whose subtree sum covers `u` → the leaf. **O(log n)**.
- **Update** = set a leaf, walk to the root refreshing sums. **O(log n)**.
- It's the iterative segment tree from competitive programming, with sums at
  internal nodes. This is the canonical proportional-PER implementation
  (Schaul et al. 2015); every serious library (Dopamine, SB3, torchrl, Reverb)
  ships it.
- Priority = `|TD error| + ε`; new samples enter at max priority. Stratified
  sampling + IS-weight correction with **β annealed 0.4 → 1.0**.

For us it's a **robustness** fix: it resurfaces exactly the rare death/near-miss
transitions that Tier-0 says are being forgotten.

---

## Before / after

| | V1 (`dqn.yaml`) | V2 (`dqn_v2.yaml`) |
|---|---|---|
| Network | MLP 7→512→512→2 | + LayerNorm, + Dueling head |
| Targets | 1-step | 3-step returns |
| Replay | uniform ring | prioritized (sum-tree) + IS weights |
| LR | flat 2.5e-4 | cosine → 0 |
| Deploy weights | raw final/best online net | EMA (Polyak, 0.999) snapshot |
| Best selected by | noisy ε-greedy train score | greedy eval (ε=0, fixed seeds) |
| Collapse risk | high (final.pth degraded to ~1 pipe) | mitigated on three axes |

**Expected benefits:** far less checkpoint lottery, a deploy model chosen on real
performance, fewer late-training divergences, faster/steadier learning, and rare
crash transitions kept "fresh" so the immortal policy is actually *held*, not
just briefly reached. (These are design expectations — validate with a full run;
the mechanics are unit-tested and the pipeline is smoke-tested end-to-end.)

---

## What a studio would also consider (not implemented)

- **Distributional RL (QR-DQN / IQN):** model the return distribution, then do
  **risk-averse (CVaR) action selection** — a principled version of what the
  tiered "fly through the middle" reward hand-engineers (edge passes have fatter
  left tails, so a risk-averse agent prefers the center). Biggest single
  remaining algorithmic step if we want it.
- **PPO / discrete SAC:** for a dense-reward, near-solved control task, on-policy
  PPO sidesteps replay staleness and the deadly triad, and this collapse largely
  doesn't manifest. Worth a serious look if the goal is a robust *shipped* agent
  rather than leveling up DQN craft specifically.
- **GPU data path:** GPU-resident replay + a tensorized env stepping thousands of
  games in parallel. Structural (data-path) win, not algorithmic.

---

## Tests

`tests/test_v2.py` covers: sum-tree totals/updates/proportional sampling, min-tree,
PER shapes + IS-weight normalization + high-TD-error prioritization, n-step return
math (full window + terminal flush), dueling/LayerNorm forward shapes, plain-QNet
backward-compat (identical keys → old checkpoints still load), and an agent update
on the V2 config (EMA present, finite loss, TD errors returned). Run `just test`.
