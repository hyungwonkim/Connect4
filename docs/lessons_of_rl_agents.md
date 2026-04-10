# Lessons from Training RL Agents on Connect 4

This document collects practical lessons from building reinforcement-learning
agents for Connect 4. Each chapter covers one agent family (AlphaZero, DQN,
PPO, …) — what the algorithm is, what we tried first, why it failed, and what
ultimately worked.

---

## Chapter 1: AlphaZero

### 1.1 Algorithm Overview

AlphaZero combines **Monte Carlo Tree Search (MCTS)** with a **deep neural
network** that jointly outputs a policy and a value:

- **Policy head** π(a|s): predicts the probability of each legal action.
- **Value head** v(s) ∈ [-1, 1]: predicts the expected game result from the
  current player's perspective (+1 = win, −1 = loss, 0 = draw).

The network guides MCTS: at each simulation, the tree expands a leaf, asks the
network for (policy prior, value estimate), and backpropagates the value up the
path. After many simulations, the visit-count distribution at the root becomes
a *stronger* policy than the raw network output — this is the "policy
improvement" step.

**Training loop:**

```
repeat forever:
    1. Self-play: use current network + MCTS to play games against itself.
       Store (state, mcts_visits, final_result) tuples.
    2. Training: sample mini-batches and minimize
           L = -Σ π_target · log π_network   (policy cross-entropy)
             + (v_network - z)^2             (value MSE)
             + c · ||θ||^2                   (L2 weight decay)
       where π_target is the MCTS visit distribution and z is the final result.
    3. Evaluation: optionally pit the new network against the old; keep the
       stronger one.
```

**Why it works.** The network and MCTS bootstrap each other: MCTS turns a weak
network into a stronger planner, and training distills the planner's outputs
back into the network. No human games, no handcrafted features, no explicit
reward shaping — just game results and the game rules.

**Architecture (standard).** A convolutional residual tower (tens of blocks
for Go/Chess) with two small heads — one for policy, one for value. For
Connect 4, a much smaller tower suffices because the state space is tiny
compared to Go.

### 1.2 First Attempt (v1 → v3)

We started with a textbook AlphaZero implementation and iterated on size,
search depth, and training tricks:

| Version | Network                  | Params | MCTS sims | Training time |
|---------|--------------------------|--------|-----------|---------------|
| v1      | 3-conv 64-channel CNN    | 1.4M   | 200–600   | ~2 days       |
| v2      | 256-channel 6-res-block  | 7.1M   | 200–800   | ~3 days       |
| v3      | 64-channel 4-res-block   | 305K   | 200       | ~5 days       |

For each version, we added what felt like reasonable ideas:
- Bigger network → smaller network (overfitting fear)
- Longer training (up to 200 iterations × 100 games/iter)
- Opponent pool (play past versions)
- Diverse openings (random/greedy first 2–5 moves, then self-play)
- Cosine-annealed learning rate

**Result: the same failure pattern every time.**

- Vs-Greedy win rate **started at 40–50%** (iteration 1, essentially random).
- Climbed briefly to ~60%.
- **Declined to 10–20%** and plateaued there by iteration ~50.
- Value loss **stuck flat at ~0.73** for hundreds of iterations.
- Policy loss did drop (1.9 → 1.2), but MCTS visit-count targets were
  low-quality, so the "improvement" was largely cosmetic.

At 1000 eval simulations, the best checkpoint still only managed ~50% vs
Greedy. Greedy is a one-ply lookahead heuristic; our AlphaZero could not
reliably beat it despite days of training and 600–800 simulations per move.

### 1.3 Root Causes

After stepping back, the problems fell into two categories.

**Category A — Silent bugs in the search.**

1. **Sign-flipped value backpropagation.** The MCTS backprop expected values
   from the *parent's* perspective at the leaf (so selection, which uses
   `+child.q_value` without negation, correctly maximizes). Terminal states
   were backpropagated correctly (from mover's POV = parent's POV). But the
   network value was passed in *leaf-player's* POV — opposite sign. This meant
   that whenever MCTS used the value head to evaluate a leaf, it pushed
   selection in the **wrong direction**. The value head could never learn
   anything coherent because its output was being used backwards.

2. **No explicit terminal detection at expansion.** MCTS only checked
   terminality when a node was visited as a leaf. A 3-in-a-row + immediate win
   was therefore invisible to the search until it was randomly reached and
   evaluated by the (weak) value network. Greedy exploited this by setting up
   one-move tactical threats that AlphaZero's search didn't see.

**Category B — Standard-AlphaZero features we forgot.**

3. **No Dirichlet noise at the root.** Self-play collapsed into a narrow set
   of trajectories. The agent never learned to defend against the aggressive
   column-stacking style Greedy uses.

4. **No data augmentation.** Connect 4 is horizontally symmetric, but the
   replay buffer stored only one orientation. We were discarding a free 2×
   effective dataset.

5. **Too few training simulations.** 200 MCTS sims produced noisy visit-count
   targets. The "improvement" the network was supposed to distill was itself
   too noisy to learn from.

### 1.4 What Worked (v4)

The v4 retraining fixed all five issues at once:

| Fix | Change |
|-----|--------|
| Sign bug | Negate NN value before backprop |
| Terminal detection | Check `check_winner()`/`is_draw()` at **expansion** time; store `is_terminal`/`terminal_value` on child nodes; stop selection at terminal nodes |
| Dirichlet noise | Add Dir(α=0.3) with ε=0.25 to root priors during self-play (not at inference) |
| Symmetry | Every (state, π) is stored twice: original + horizontal flip (flip state via `torch.flip(dims=[-1])`, reverse π) |
| Training sims | 200 → 400 |
| Model size | 305K → 842K params (AlphaZeroNetV2: 96 channels × 5 residual blocks) |
| Buffer | 50K → 150K |
| LR schedule | Flat 0.002 for 50 warmup iters, then cosine decay |
| Greedy teacher | 10% of training games are AZ vs Greedy (collect training data from AZ's moves) |

No single change was the silver bullet — but the **sign fix + terminal
detection** were the two that turned a broken system into a working one.

### 1.5 Outcome

After just **10 iterations** of v4 training (~2.5 hours on an M2 Mac):

| Opponent | Win rate (100 games) |
|----------|----------------------|
| Random   | 99%  |
| Greedy   | 95%  |
| DQN      | 100% |
| PPO      | 100% |

And by **iteration 70** (after ~17 hours of training):

| Opponent | Win rate (100 games) |
|----------|----------------------|
| Random   | 100% |
| Greedy   | 100% |
| DQN      | 100% |
| PPO      | 100% |

Training losses converged in parallel:
- Policy loss: 1.93 → 0.95 (flattening)
- Value loss: 0.75 → 0.47 (flattening)

The value head — which had been stuck at 0.73 across *every* prior run — was
finally learning because MCTS was finally using its output in the correct
direction.

### 1.6 Lessons

1. **Verify the sign convention in MCTS with a unit test.** Set up a position
   with an obvious winning move and confirm MCTS finds it with a *random*
   network. If the sign is wrong, the search will actively avoid winning
   moves; if it's right, terminal detection alone will find them.

2. **Terminal detection at expansion is huge in shallow games.** Connect 4's
   branching factor is only 7. Even a small search (50 sims) should find
   one-move wins instantly — but only if terminal states are hardcoded rather
   than estimated by the value head.

3. **A broken value head is worse than no value head.** For hundreds of
   iterations, our value loss was flat at 0.73 — not because the problem was
   hard, but because the value estimate was being *used backwards*. A useless
   value head defaults toward 0 and is harmless; an inverted one actively
   misguides MCTS.

4. **Ablate the standard tricks before inventing new ones.** Dirichlet noise
   and symmetry augmentation are in the AlphaZero paper for a reason. We
   spent days trying exotic ideas (opponent pools, diverse openings, greedy
   mixing) before adding two features that took 30 lines of code.

5. **A broken learner will not be saved by a bigger network or longer
   training.** Scaling up hides bugs by making training slower. Start small,
   verify the loop works, *then* scale.

---

## Chapter 2: DQN

### 2.1 Algorithm Overview

Deep Q-Network (DQN) learns an action-value function **Q(s, a)** that
estimates the expected cumulative reward for taking action *a* in state *s* and
playing optimally afterwards. At inference, the agent simply picks
`argmax_a Q(s, a)`.

**Training loop:**

```
repeat:
    1. Observe state s, pick action a (epsilon-greedy),
       receive reward r, observe next state s'.
    2. Store (s, a, r, s', done) in a replay buffer.
    3. Sample a mini-batch and minimize TD error:
           L = (r + gamma * max_a' Q_target(s', a') - Q(s, a))^2
       where Q_target is a periodically-synced copy of Q.
```

**Double DQN** decouples action selection from evaluation: the *policy* network
chooses the best next action, but the *target* network evaluates it. This
reduces the maximization bias that causes Q-value overestimation.

**Dueling DQN** splits the final layers into two streams — V(s) (state value)
and A(s, a) (advantage per action) — and recombines them as
Q(s, a) = V(s) + (A(s, a) - mean_a A(s, a)). This helps in games like
Connect 4 where many positions have one critical move and the rest are
interchangeable: the value stream learns the "how am I doing?" signal
independently from "which column matters?".

### 2.2 First Attempt (v1)

The initial DQN agent used a straightforward setup:

| Component         | v1 choice                         |
|-------------------|-----------------------------------|
| Architecture      | 3-conv 64-channel CNN + 2 FC      |
| Replay buffer     | Uniform deque, 100K               |
| Opponent          | Random only                       |
| Rewards           | +1 (win) / -1 (loss) / 0 (draw)  |
| Epsilon schedule  | 1.0 → 0.05 over 30K steps        |
| Target sync       | Every 1,000 updates               |
| Double DQN        | Yes                               |
| Loss              | Huber (smooth_l1)                 |
| Gamma             | 1.0                               |
| LR                | 1e-4                              |

Training ran for 60K episodes. The agent quickly learned to beat Random (~85%
win rate), and we assumed Greedy would follow. It did not.

**Result: vs Greedy win rate stayed around 15-20%.**

The v1 DQN could not reliably beat a simple one-ply heuristic.

### 2.3 Reading the TensorBoard — Diagnosing the Problems

The TensorBoard traces told a clear story once we knew what to look for:

**1. Q-value overestimation (the "delusional winner").**

The `mean_q` chart was the smoking gun. A well-calibrated DQN for a
win/loss/draw game should produce Q-values in roughly [-1, +1], with mean
drifting toward 0. Our `mean_q` started near 0 but steadily climbed past 0.5,
past 1.0, and eventually reached 1.5+. The network was predicting "I will
definitely win from here" in positions where it would actually lose.

Why did this happen? The agent only trained against Random. Beating Random is
easy, so the buffer was full of winning trajectories. The max operator in the
Bellman target (`max_a Q_target(s', a')`) amplifies positive bias when most
experiences end in victory — the network bootstraps from its own optimistic
estimates, each generation more inflated than the last. Double DQN slows
this, but does not stop it when the training distribution is this lopsided.

**2. No tactical exposure.**

Training exclusively against Random meant the agent never encountered forcing
lines — two-sided threats, mandatory blocks, or positions where one wrong
column costs the game. Greedy's entire strategy is "extend my longest run /
block yours," and our agent had never seen an opponent that does either.
It was like training a boxer only against punching bags, then entering the
ring.

**3. Epsilon schedule was well-calibrated for v1** — 30K decay over 60K
episodes meant the agent reached exploitation by mid-training. But this
wouldn't help when the exploration was against the wrong opponent. The agent
explored every corner of "how to beat Random" and learned nothing about how
to survive against a real threat.

### 2.4 First Round of Fixes

Armed with the TensorBoard diagnosis, we redesigned the training around four
ideas: **architecture**, **opponent variety**, **reward asymmetry**, and
**exploration**.

| Component         | v1                    | Revised                           |
|-------------------|-----------------------|-----------------------------------|
| Architecture      | Plain DQN             | **Dueling DQN** (V + A streams)   |
| Replay buffer     | Uniform               | **PER** (alpha=0.6, beta 0.4→1.0) |
| Opponent pool     | Random only           | 20% Random / 40% Greedy / 40% self-snapshots |
| Rewards           | +1 / -1 / 0          | **+1 / -1.5 / 0** (asymmetric)   |
| Epsilon schedule  | 30K steps             | **150K steps** (longer exploration)|
| Reward shaping    | None                  | +0.05 create 3-in-a-row, -0.05 allow opponent 3, +0.02 block, +0.01 center |
| Target sync       | 1,000                 | 1,000                             |
| LR                | 1e-4                  | 1e-4                              |
| Diagnostics       | Loss only             | **mean_q, max_q, min_q, td_error_abs_mean** |

The reasoning behind each change:

- **Dueling architecture** helps the network separate "this board position is
  good" from "this particular column matters." In Connect 4, most mid-game
  positions have one or two critical columns; the rest are noise. The value
  stream can learn the base evaluation while the advantage stream sharpens
  the action choice.

- **PER** replays surprising transitions more often. Losses against Greedy —
  especially the sudden, "I didn't see that column" kind — have high TD
  error and get replayed disproportionately. This is exactly the type of
  experience the agent needs to internalize.

- **Opponent pool** is the direct fix for the "never saw a real opponent"
  problem. Self-play snapshots add diversity (the agent plays its own past
  selves), while Greedy provides the tactical pressure the agent needs to
  develop blocking reflexes.

- **Asymmetric rewards** (-1.5 for a loss) directly combat Q-value
  overestimation. The idea: if losing hurts 50% more than winning pays, the
  Q-network has a penalty budget to offset the positive bias from the
  max operator. Terminal values get anchored lower, and the network can't
  drift into "I always win" territory as easily.

- **Reward shaping** provides intermediate signal in a sparse-reward game.
  Without it, the agent gets zero feedback until the game ends. With it,
  creating a three-in-a-row or blocking an opponent's three registers
  immediately. This helps the agent develop tactical patterns faster.

### 2.5 First Run Failed — Epsilon Miscalibration

We launched training with 60K episodes and eps_decay_steps=150K. It wasn't
until checking TensorBoard at episode 50K that we noticed: **epsilon was still
at 0.62.** The agent was making random moves 62% of the time with only 10K
episodes left to train.

The bug: eps_decay_steps counts *training steps* (one per episode after buffer
warmup), and we had set it to 150K for 60K episodes. The epsilon schedule
was designed for a much longer run and never had time to anneal.

This was easy to spot in TensorBoard — the `training/epsilon` curve was
nearly flat. We killed the run and restarted with eps_decay_steps=40K, so
the agent would reach eps=0.05 by episode 40K and have 20K episodes of
nearly-pure exploitation.

**Lesson: always plot epsilon alongside win rate.** If epsilon is still high
when win rate plateaus, you're measuring exploration noise, not policy quality.

### 2.6 Second Run — Peak Then Diverge

With corrected epsilon decay (40K steps over 60K episodes), training looked
promising at first:

| Episode window | mean_q | wr vs Greedy (training) | wr vs Greedy (clean eval) |
|----------------|--------|-------------------------|---------------------------|
| 0-10K          | 0.06   | 2%                      | —                         |
| 10K-20K        | 0.49   | 5%                      | —                         |
| 20K-30K        | 0.49   | 13%                     | 28%                       |
| 30K-40K        | 0.62   | 33%                     | —                         |
| **40K-50K**    | **0.95**| **60%**                 | **64% / 56%**             |
| 50K-60K        | **1.47**| 49%                     | 48% / 42%                |

The agent peaked around episode 45K (clean eval: 64% as P1, 56% as P2 vs
Greedy), then **regressed**. The TensorBoard made the cause unmistakable:
`mean_q` shot from 0.62 to 1.47 in 20K episodes. The Q-value overestimation
we thought we'd solved was back.

What went wrong? Once epsilon dropped to 0.05 (episode 40K), the agent entered
exploitation mode. It started winning more games, filling the buffer with
high-reward transitions. The target network, synced every 1K updates, started
bootstrapping from these inflated values. The asymmetric -1.5 loss penalty
slowed the drift but couldn't stop it. Meanwhile, the self-snapshot pool
contained mostly weak early-game copies — winning against them was easy,
further inflating Q-values. A feedback loop formed: optimistic Q-values →
over-confident play → more wins against weak snapshots → even higher Q-values.

The final checkpoint (episode 60K) was *worse* than the episode-45K checkpoint.
The "best.pt" we saved was post-divergence garbage.

### 2.7 What Worked — The Full Fix

Three more changes turned the corner:

| Fix | Rationale |
|-----|-----------|
| **60% Greedy in opponent pool** (up from 40%) | Forces the agent to face tactical threats every game, not just weak self-snapshots. Reduced Random to 10%, snapshots to 30% |
| **LR step-down** 1e-4 → 3e-5 at episode 40K | Once the agent enters exploitation, large gradient steps amplify Q-drift. Halving the LR stabilizes updates exactly when they matter most |
| **Target sync every 2,000 updates** (up from 1,000) | Slower target updates = smoother bootstrapping. The target network lags further behind, dampening the positive feedback loop |
| **Softer loss penalty** -1.2 (down from -1.5) | The -1.5 penalty over-corrected in the opposite direction, creating large TD errors that destabilized PER. -1.2 kept the asymmetry without the variance |
| **Clean deterministic eval for checkpoint selection** | Training-time win rates include epsilon noise and opponent mix. Every 5K episodes, we ran 40 games (both sides, argmax policy, no epsilon) against Greedy and only saved `best.pt` when the clean eval improved |
| **80K episodes** (up from 60K) | More exploitation time after the LR drop |

The clean-eval progression:

```
ep  5K:  50.0%  <- NEW BEST
ep 15K:  65.0%  <- NEW BEST
ep 25K:  70.0%  <- NEW BEST
ep 40K:  80.0%  <- NEW BEST  (LR dropped here)
ep 55K:  82.5%  <- NEW BEST
ep 80K:  82.5%
```

The LR drop at episode 40K was the inflection point: win rate jumped from 70%
to 80% in one eval window. After that, it continued climbing slowly and held
stable through the end of training, with no sign of the late-training collapse
we saw before.

`mean_q` still drifted upward (reaching ~1.15 by episode 80K), but the
slower target sync and reduced LR kept it from the runaway divergence of the
previous run. TD error held flat at 0.18-0.21 throughout.

### 2.8 Final Results

100-game deterministic eval (both sides) of the final `best.pt`:

| Opponent   | DQN 1st player | DQN 2nd player | Avg   |
|------------|----------------|----------------|-------|
| Random     | 96%            | 92%            | 94%   |
| Greedy     | **82%**        | **68%**        | **75%** |
| AlphaZero  | 0%             | 0%             | 0%    |

The 2nd-player deficit against Greedy (68% vs 82%) is structural — Connect 4
is a first-player-wins game with perfect play — not a training issue.

### 2.9 Lessons

1. **Never train only against Random.** The agent will overfit to the easiest
   possible opponent and learn nothing transferable. Include Greedy (or
   stronger) in the training pool from day one.

2. **Plot `mean_q` alongside win rate — always.** Q-value overestimation is
   the single most common failure mode of DQN variants. For a game with
   rewards in [-1.5, +1], `mean_q` should stay in [0, 0.5]. If it crosses
   1.0, something is wrong. When it reaches 1.5, the agent is delusional.

3. **The epsilon schedule must be calibrated to the training length.** We
   wasted 60K episodes because eps_decay_steps was set to 150K. Always
   verify that epsilon reaches its minimum value with at least 30% of
   training remaining.

4. **Late-training divergence is a real threat in DQN.** The exploitation
   phase (low epsilon) creates a positive feedback loop: more wins → higher
   Q → more confident play → more wins. Counter it with: slower target sync,
   LR reduction, and checkpoint selection by clean eval rather than
   training-time metrics.

5. **Clean eval is the only honest metric.** Training-time win rates are
   depressed by epsilon exploration and diluted by the opponent mix.
   A 42% training win rate against Greedy turned out to be 75% in clean
   eval. Conversely, a "best" training checkpoint can be post-divergence
   garbage. Always select checkpoints by clean deterministic evaluation.

6. **Asymmetric rewards help, but don't overdo them.** -1.5 for a loss
   created large TD errors that destabilized training. -1.2 preserved the
   "defense matters more than offense" signal without the variance spike.

7. **Dueling + PER are strong but not magical.** They didn't fix training
   when the opponent pool was wrong or epsilon was miscalibrated. Get the
   training distribution right first, then add architectural refinements.

---

## Chapter 3: PPO

### 3.1 Algorithm Overview

Proximal Policy Optimization (PPO) is an **actor-critic** method that directly
optimizes a policy (the actor) while learning a value function (the critic)
to reduce gradient variance.

**Key idea:** constrain policy updates so the new policy doesn't drift too far
from the old one. This is the "proximal" part — stability over speed.

```
repeat:
    1. Collect a batch of trajectories using the current policy.
    2. Compute advantages using GAE (Generalized Advantage Estimation):
           A_t = delta_t + (gamma * lambda) * delta_{t+1} + ...
       where delta_t = r_t + gamma * V(s_{t+1}) - V(s_t).
    3. For several epochs over the same batch, minimize:
           L = L_policy + c_value * L_value - c_entropy * H(pi)

       L_policy = -min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)
       L_value  = (V(s) - R_target)^2
       H(pi)    = entropy bonus (encourages exploration)

       ratio = pi_new(a|s) / pi_old(a|s)
```

**Why it works.** The clipped objective prevents catastrophically large policy
updates. The entropy bonus prevents premature convergence to a deterministic
policy. GAE balances bias vs variance in the advantage estimate through
lambda. PPO is simpler and more stable than earlier policy gradient methods
(TRPO, A3C), which makes it a popular first choice for game-playing agents.

**Architecture.** Typically a shared backbone (CNN for board games) with two
heads: one for action logits (actor) and one for a scalar state value
(critic).

### 3.2 First Attempt (v1)

| Component         | v1 choice                         |
|-------------------|-----------------------------------|
| Architecture      | 3-conv 64-channel CNN + actor/critic heads |
| Opponent          | Random + self-play snapshots      |
| LR                | 3e-4                              |
| Entropy coeff     | 0.01 (fixed)                      |
| Value coeff       | 0.5                               |
| GAE lambda        | 0.95                              |
| Clip epsilon      | 0.2                               |
| PPO epochs        | 4                                 |
| Games per iter    | 50                                |
| Gamma             | 1.0                               |
| Gradient clip     | 0.5 (max norm)                    |

We trained for 1,000 iterations (50K games total). The agent beat Random
handily but could not break even against Greedy.

**Result: vs Greedy win rate hovered around 20-30%.**

### 3.3 Reading the TensorBoard — A Different Disease

PPO's TensorBoard traces looked completely different from DQN's, and pointed
to a different failure mode:

**1. Entropy was erratic, not decaying.**

A healthy PPO agent's entropy curve should decrease smoothly as the policy
becomes more decisive. Ours was bouncing between 0.1 and 1.0 like a ping-pong
ball, sometimes within a single iteration. This meant the policy was
oscillating between near-deterministic play and near-random play — never
settling into a stable strategy.

The cause: the learning rate (3e-4) was too high. Each PPO update shoved the
policy hard enough to destabilize it, and the next iteration's rollouts came
from a very different policy than the one being updated. The clipped objective
limited the damage per mini-batch, but over 4 epochs of updates on stale
data, the cumulative drift was too large.

**2. Policy loss vibrated around zero with no trend.**

In a learning PPO agent, policy loss typically decreases (the advantage-weighted
likelihood increases). Ours oscillated around zero for hundreds of iterations.
The policy was changing — entropy confirmed that — but not *improving*. Each
update was as likely to make the policy worse as better.

This was a downstream effect of the high LR: gradients were noisy, the
policy surface was being traversed too aggressively, and the clipping
mechanism was firing constantly (a sign that the update is trying to change
too much).

**3. Value loss plateaued early.**

The critic's loss dropped initially, then flatlined. The critic was not
learning the value of board positions — it was learning the average outcome
against Random, which is "I almost always win." Against a different opponent,
these value estimates would be useless. The critic learned a constant
function (approximately +0.7) and stopped there.

**4. No Greedy in the training pool.**

Same mistake as DQN v1: training exclusively against Random and self-play.
Self-play snapshots of a weak agent are still weak. The agent had zero
exposure to Greedy's blocking/extending strategy.

### 3.4 What We Changed

The diagnosis pointed to four independent problems: LR too high, entropy
uncontrolled, critic undertrained, and opponent pool too easy. We fixed all
four:

| Component         | v1                    | Revised                           |
|-------------------|-----------------------|-----------------------------------|
| LR                | 3e-4                  | **1e-4** (3x reduction)           |
| Entropy coeff     | 0.01 (fixed)          | **0.01 → 0.001** (linear decay)  |
| Value coeff       | 0.5                   | **1.0** (doubled)                 |
| Games per iter    | 50                    | **100** (more stable gradients)   |
| Opponent pool     | Random + self         | **10% Random / 60% Greedy / 30% self** |
| Reward shaping    | None                  | Same as DQN: +0.05 create-3, -0.05 allow-3, +0.02 block, +0.01 center |
| Diagnostics       | Basic                 | **approx_kl per update** (tracks policy drift) |
| Checkpoint select | Last iteration        | **Clean deterministic eval vs Greedy every 50 iters** |

The reasoning:

- **Lower LR (1e-4)** directly addresses the entropy oscillation. Smaller
  updates mean the policy evolves gradually, and the 4 PPO epochs over each
  batch remain on-policy enough to be useful.

- **Decaying entropy coefficient** is more nuanced than a fixed value. Early
  in training, high entropy encourages exploration — the agent tries
  different columns and discovers which ones lead to wins. Late in training,
  low entropy lets the agent commit to the best strategy it's found. A fixed
  coefficient forces a single trade-off for the entire training run.

  We initially considered *raising* entropy to combat the erratic behavior,
  but the TensorBoard showed entropy was already too high at times — the
  problem was instability, not under-exploration. Decaying toward 0.001
  gradually tightened the policy as training progressed.

- **Doubling the value coefficient (0.5 → 1.0)** forces the critic to learn
  faster. PPO's critic has a harder job than DQN's Q-function because it must
  evaluate positions from scratch each iteration (no replay buffer). Giving
  its gradient 2x the weight relative to the policy gradient means the
  critic keeps up with the changing policy.

- **100 games per iteration** (up from 50) produces roughly 2x as many
  transitions per PPO update. This directly reduces gradient variance, which
  compounds with the lower LR — each update is smaller *and* more accurate.

- **60% Greedy in the opponent pool** ensures the agent faces tactical
  pressure in the majority of games. This was the single most impactful
  change, just as with DQN.

### 3.5 Training Progression

The clean-eval progression told a smoother story than DQN:

```
iter  50:  30.0%  <- NEW BEST
iter 100:  40.0%  <- NEW BEST
iter 150:  52.5%  <- NEW BEST
iter 200:  67.5%  <- NEW BEST
iter 450:  70.0%  <- NEW BEST
iter 900:  72.5%  <- NEW BEST
iter 1000: 70.0%
```

Three phases were visible:

1. **Rapid climb (iter 1-200):** 0% → 67.5%. The agent learned basic tactical
   patterns — blocking immediate threats, favoring center columns, building
   three-in-a-row. Entropy decreased steadily. `approx_kl` stayed below 0.02,
   confirming stable updates.

2. **Slow improvement (iter 200-900):** 67.5% → 72.5%. Gains came slowly as
   the agent refined its positional play. The entropy coefficient was in its
   decay phase, the policy was committing to specific strategies. Value loss
   continued trending downward (the critic was still learning), unlike v1
   where it had plateaued.

3. **Plateau (iter 900+):** Clean eval stabilized around 70-72%. The policy
   had found its ceiling given the architecture and opponent distribution.

Unlike DQN, there was **no late-training collapse**. PPO's clipped objective
inherently limits how far the policy can drift per update, which prevented
the kind of runaway divergence we saw with DQN's Q-value inflation. The
lower LR and entropy decay kept updates stable throughout.

### 3.6 Final Results

100-game leaderboard evaluation:

| Opponent   | PPO 1st player | PPO 2nd player | Avg   |
|------------|----------------|----------------|-------|
| Random     | 96%            | 84%            | 90%   |
| Greedy     | **94%**        | **26%**        | **60%** |
| DQN        | 0%             | 0%             | 0%    |
| AlphaZero  | 0%             | 0%             | 0%    |

The first-player vs second-player asymmetry against Greedy is striking: **94%
as P1 but only 26% as P2.** This is far more lopsided than DQN's 82/68 split.

The reason is architectural: PPO learns a *policy* (probability distribution
over moves), while DQN learns *values* (how good is each move). PPO's policy
is trained on whatever positions it encounters, and since it goes first in
half its training games, it develops strong first-player openings but weaker
defensive second-player patterns. DQN's value function is more transferable
between sides because it evaluates board states regardless of who moved.

PPO also lost 100-0 to DQN in the leaderboard — a deterministic-policy
artifact (same game replayed 50 times) that exaggerates the gap but does
reflect a genuine ranking difference.

### 3.7 PPO vs DQN — Why the Gap?

PPO achieved 72.5% vs Greedy; DQN achieved 82.5%. Both used the same
opponent pool, reward shaping, and similar total training compute. The gap
comes down to fundamental algorithm properties:

1. **Sample efficiency.** DQN stores every transition in a replay buffer and
   trains on each one multiple times (via PER, the most informative ones
   get replayed dozens of times). PPO uses each batch for only 4 epochs,
   then discards it. For a game with sparse terminal rewards, DQN extracts
   far more learning per game played.

2. **Off-policy vs on-policy.** DQN learns from old transitions (off-policy),
   which means the buffer contains a rich mix of early-game exploration and
   late-game exploitation. PPO can only learn from data generated by the
   current policy (on-policy). When the policy makes a mistake, PPO doesn't
   revisit that mistake unless it happens again naturally.

3. **Value function generalization.** DQN's Q-function directly answers "how
   good is column 3 in this position?" — a single forward pass gives
   actionable information. PPO's critic only answers "how good is this
   position overall?" — the actor must then figure out which column to pick.
   The extra indirection makes PPO harder to train to the same precision.

4. **Stability vs ceiling.** PPO's clipped objective prevents collapse (we
   never saw the kind of Q-value divergence DQN suffered), but it also limits
   how aggressively the policy can improve. DQN's unconstrained optimization
   allows higher peaks at the cost of potential instability.

### 3.8 Lessons

1. **A too-high LR in PPO manifests as entropy oscillation, not diverging
   loss.** The policy loss might look "fine" (hovering near zero) while the
   policy itself is chaotic. Entropy is the canary in the coal mine.

2. **Decay entropy, don't fix it.** A fixed entropy coefficient is a single
   compromise for an entire training run. Early training needs exploration;
   late training needs exploitation. Linear decay from 0.01 to 0.001
   handled both regimes.

3. **Double the value coefficient for on-policy methods.** Without a replay
   buffer, the critic trains on each datum exactly once (times the number of
   PPO epochs). It needs proportionally more gradient weight to keep up with
   the evolving policy.

4. **PPO is naturally stable but naturally weaker than DQN for this task.**
   We never needed the LR-schedule or target-sync tricks that saved DQN from
   collapse, but we also couldn't push PPO past 72% vs Greedy. For games
   with sparse terminal rewards and moderate state spaces, value-based
   methods (DQN) may simply have a higher ceiling than policy-gradient
   methods (PPO).

5. **First-player/second-player asymmetry reveals what the agent actually
   learned.** PPO's 94%/26% split against Greedy shows it learned a strong
   first-player opening but weak defensive play. DQN's 82%/68% split is
   more balanced, suggesting its value function generalizes across sides
   better than PPO's policy.

6. **Same lesson as DQN and AlphaZero: include Greedy in the training pool
   from the start.** This was the single most impactful change for every
   agent we trained. An agent that has never seen tactical pressure will not
   develop tactical skill, no matter how sophisticated the algorithm.

---

## Epilogue: Final Leaderboard

After all training and optimization, we ran a full round-robin tournament
(100 games per matchup: 50 as P1, 50 as P2):

|            | Random | Greedy | DQN   | PPO   | AlphaZero | **Avg** |
|------------|--------|--------|-------|-------|-----------|---------|
| **Random** | —      | 1%     | 7%    | 11%   | 0%        | 4.8%    |
| **Greedy** | 99%    | —      | 20%   | 41%   | 0%        | 40.0%   |
| **DQN**    | 96%    | 73%    | —     | 100%  | 0%        | 67.2%   |
| **PPO**    | 90%    | 60%    | 0%    | —     | 0%        | 37.5%   |
| **AlphaZero** | 100% | 100% | 100%  | 100%  | —         | 100.0%  |

**Ranking:**
1. AlphaZero (100% avg) — undefeated, 400-0 across all matchups
2. DQN (67.2% avg) — beats Greedy and PPO decisively
3. Greedy (40.0% avg) — simple heuristic, still beats PPO overall
4. PPO (37.5% avg) — beats Greedy as P1 but loses elsewhere
5. Random (4.8% avg) — baseline

The gulf between AlphaZero and everything else is enormous. MCTS + a trained
network is simply in a different class from pure RL for a game this tactical.
But among the pure RL agents, the journey from "loses to Greedy" to "reliably
beats Greedy" taught us more about practical RL than any textbook.
