## Implementation Summary: Self-Play RL for Connect 4

This summary outlines the architectural differences and implementation requirements for training a Connect 4 agent from scratch using AlphaZero, PPO, and DQN.

---

### 1. AlphaZero (MCTS-Guided Policy Iteration)

AlphaZero is a "Model-Based" algorithm because it uses the game rules (a forward model) to look ahead during the decision-making process.

* **Core Logic:** The agent doesn't just "react" to the board; it runs a **Monte Carlo Tree Search (MCTS)**. The Neural Network acts as a guide for this search by providing a **Policy** (which moves look promising) and a **Value** (how likely is a win from here).
* **Training Loop:**
1. **Self-Play:** The agent plays against itself using MCTS to select moves.
2. **Data Storage:** It stores the search statistics (visit counts) as the "target policy" and the final game result as the "target value."
3. **Optimization:** The network is trained to minimize the error between its raw predictions and the MCTS-improved results.



---

### 2. PPO Self-Play (Policy Gradient / Actor-Critic)

Proximal Policy Optimization is a "Model-Free" algorithm. It learns a direct mapping from the board state to the best action without an internal "search tree."

* **Core Logic:** Uses an **Actor** (to pick moves) and a **Critic** (to estimate the win probability). It uses a "clipped" objective function to ensure that network updates aren't too drastic, which maintains training stability.
* **Self-Play Strategy:** Unlike AlphaZero, PPO needs a **Policy Pool**.
* The agent plays against "snapshots" of its previous versions.
* This prevents "strategy cycling," where the agent learns to beat its current self but forgets how to beat older, different strategies.


* **Training Loop:**
1. **Rollout:** Collect transitions (State, Action, Log-Probability, Reward) by playing against the pool.
2. **Update:** Perform multiple epochs of stochastic gradient descent on the collected data.



---

### 3. DQN Self-Play (Value-Based / Q-Learning)

Deep Q-Learning is also "Model-Free" and focuses on learning the "Value" of every possible action in a given state.

* **Core Logic:** The network learns a **Q-Function** $Q(s, a)$. The "best" move is simply the one with the highest $Q$ value: $arg\max_a Q(s, a)$.
* **Self-Play Strategy:** Uses an **Epsilon-Greedy** approach during self-play to ensure exploration (occasionally picking random moves). Like PPO, it requires an opponent history to remain robust.
* **Training Loop:**
1. **Experience Replay:** Stores every move in a massive buffer (State, Action, Reward, Next State).
2. **Temporal Difference (TD) Learning:** Updates the network by comparing the current Q-value prediction to the "Target" (Reward + discounted Q-value of the next state).



---

### Comparative Implementation Table

| Feature | AlphaZero | PPO (Self-Play) | DQN (Self-Play) |
| --- | --- | --- | --- |
| **Network Type** | Policy + Value Heads | Actor + Critic Heads | Q-Value Head |
| **Inference** | MCTS (Slow, but strong) | Direct Forward Pass (Fast) | Direct Forward Pass (Fast) |
| **Rules Usage** | **Required** (for MCTS) | Not required for inference | Not required for inference |
| **Stability** | Very High | High (due to clipping) | Moderate (requires tuning) |
| **Sample Efficiency** | High | Low (needs many games) | Moderate |

---

### Recommended "Unified" Training Strategy

The **Environment** should be the constant. It must handle:

1. **Turn-taking logic:** Switching between Player 1 (1) and Player 2 (-1).
2. **Board Canonicalization:** Always presenting the board to the agent such that the "current player" is represented by 1.
3. **Invalid Move Masking:** Ensuring the agents don't try to drop a piece into a full column.


## MCTS algorithm consideration for AlphaZero. 
Implementing **Monte Carlo Tree Search (MCTS)** is the core challenge of AlphaZero. While PPO and DQN rely on the neural network to make immediate decisions, AlphaZero uses the network merely as a "heuristic" to guide a deep look-ahead search.

Here is a Pythonic outline for the MCTS logic.

### 1. The MCTS Node

Each node in the search tree represents a board state. It must track four key values for every possible move (action $a$):

* $N(s, a)$: Visit count (how many times we explored this branch).
* $W(s, a)$: Total action value (sum of all evaluations from this branch).
* $Q(s, a)$: Mean action value ($W / N$).
* $P(s, a)$: Prior probability (the raw "hunch" from the Neural Network).

### 2. The Search Logic (The `Search` Class)

The search follows four distinct phases: **Selection, Expansion, Evaluation, and Backpropagation.**

```python
import numpy as np
import math

class MCTS:
    def __init__(self, game, model, args):
        self.game = game
        self.model = model
        self.args = args # e.g., {'C_puct': 1.41, 'num_simulations': 800}
        self.Qsa = {}    # stores Q values for (s,a)
        self.Nsa = {}    # stores visit counts for (s,a)
        self.Ps = {}     # stores initial policy (next move probabilities)
        self.Es = {}     # stores game endings (win/loss/draw)

    def get_action_probs(self, canonical_board, temp=1):
        """
        Runs num_simulations of MCTS starting from the current board.
        Returns a probability vector for the next move.
        """
        for _ in range(self.args['num_simulations']):
            self.search(canonical_board)

        s = self.game.string_representation(canonical_board)
        counts = [self.Nsa.get((s, a), 0) for a in range(self.game.get_action_size())]

        # Temperature 'temp' controls exploration vs exploitation
        if temp == 0:
            best_as = np.array(np.argwhere(counts == np.max(counts))).flatten()
            probs = [0] * len(counts)
            probs[np.random.choice(best_as)] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        probs = [x / float(sum(counts)) for x in counts]
        return probs

    def search(self, board):
        """
        The recursive search function.
        """
        s = self.game.string_representation(board)

        # 1. CHECK FOR TERMINAL STATE
        if s not in self.Es:
            self.Es[s] = self.game.get_game_ended(board, 1)
        if self.Es[s] != 0:
            return -self.Es[s]

        # 2. EXPAND AND EVALUATE (Leaf Node)
        if s not in self.Ps:
            # Neural Network provides Policy (P) and Value (v)
            self.Ps[s], v = self.model.predict(board)
            valids = self.game.get_valid_moves(board)
            self.Ps[s] = self.Ps[s] * valids # Mask invalid moves
            self.Ps[s] /= np.sum(self.Ps[s]) # Renormalize
            return -v

        # 3. SELECTION (Using UCB Formula)
        best_u = -float('inf')
        best_a = -1
        
        for a in range(self.game.get_action_size()):
            if self.game.get_valid_moves(board)[a]:
                # Upper Confidence Bound applied to Trees (UCT)
                u = self.Qsa.get((s, a), 0) + self.args['C_puct'] * self.Ps[s][a] * \
                    math.sqrt(sum([self.Nsa.get((s, i), 0) for i in range(self.game.get_action_size())])) / \
                    (1 + self.Nsa.get((s, a), 0))

                if u > best_u:
                    best_u = u
                    best_a = a

        a = best_a
        next_s, next_player = self.game.get_next_state(board, 1, a)
        next_s = self.game.get_canonical_form(next_s, next_player)

        # 4. RECURSION & BACKPROPAGATION
        v = self.search(next_s)

        # Update Q and N values
        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        return -v

```

### 3. Key Mathematical Principle: The UCB Formula

The core of "Selective Search" is the Upper Confidence Bound formula. It ensures the agent explores new moves while still focusing on the ones the Neural Network thinks are good.

$$U(s, a) = Q(s, a) + C_{puct} \cdot P(s, a) \cdot \frac{\sqrt{\sum N}}{1 + N(s, a)}$$

* **Exploitation:** $Q(s, a)$ drives the search toward moves that have historically yielded wins.
* **Exploration:** The second half of the formula drives the search toward moves with high prior probability ($P$) but low visit counts ($N$).

