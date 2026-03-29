# Connect 4 App Development Plan

Here's a structured journey from local Python prototype to iOS app, with concrete Claude Code prompts at each stage.

---

## Overview of the Journey

**Phase 1** → Local Python CLI game  
**Phase 2** → AI players (Random, Greedy, ε-Greedy, AlphaZero, DQN, PPO structure)  
**Phase 3** → Local GUI (Pygame or Tkinter)  
**Phase 4** → GitHub + clean architecture  
**Phase 5** → Web app (FastAPI + React/Vue)  
**Phase 6** → iOS (React Native or Swift)

---

## Phase 1: Core Game Engine

**Goal:** A working Connect 4 game engine with a clean CLI interface.

**Prompt for Claude Code:**  
```  
Create a Python Connect 4 game engine with the following requirements:

1. A `Board` class in `connect4/board.py` that:  
   - Represents a 6x7 Connect 4 grid  
   - Has methods: `drop_piece(col, player)`, `is_valid_move(col)`, `get_valid_moves()`, `check_winner()`, `is_draw()`, `copy()`, and `__str__` for display  
   - Uses 0 for empty, 1 for player 1, 2 for player 2

2. A `Game` class in `connect4/game.py` that:  
   - Manages game state and turn logic  
   - Accepts two player objects and alternates between them  
   - Has a `run()` method that loops until win or draw

3. A `HumanPlayer` class in `connect4/players/human.py` that prompts CLI input

4. A `main.py` at the root that starts a Human vs Human game

5. A `pyproject.toml` using modern Python packaging

Structure everything so AI players can be dropped in later without changing Game or Board.  
```

---

## Phase 2: AI Player Interface + Implementations

**Goal:** Define a common AI interface, then implement all four AI types.

### Step 2a — Base Interface

**Prompt:**  
```  
Create an abstract base class `BasePlayer` in `connect4/players/base.py` with:

- Abstract method `choose_move(board: Board) -> int` that returns a column index  
- Optional `name: str` property  
- Optional `reset()` method called at the start of each game (default no-op)

Refactor `HumanPlayer` to extend `BasePlayer`. Make sure `Game` uses `BasePlayer` typed arguments.  
```

### Step 2b — Random, Greedy, and Epsilon-Greedy AI

**Prompt:**  
```  
Implement three AI players, all extending `BasePlayer`, in `connect4/players/`:

1. `RandomPlayer` (`random_player.py`): picks a random valid column.

2. `GreedyPlayer` (`greedy_player.py`):  
   - Wins immediately if possible (checks all valid moves for a winning move)  
   - Blocks opponent's immediate win if no winning move exists  
   - Otherwise picks randomly among remaining valid moves  
   - Constructor takes `player_id` (1 or 2) so it knows which pieces are its own

3. `EpsilonGreedyPlayer` (`epsilon_greedy_player.py`):  
   - Takes `epsilon: float` (default 0.1) in constructor  
   - With probability epsilon, plays randomly  
   - Otherwise plays greedily (reuse or inherit GreedyPlayer logic)

Add a `main.py` CLI flag `--player2` that accepts `human`, `random`, `greedy`, `epsilon_greedy` to select the opponent.  
```

### Step 2c — MuZero-Style AI Structure

**Prompt:**  
```  
Create the structural scaffold for a MuZero-style Connect 4 AI in `connect4/players/muzero/`. 

The structure should include:

1. `muzero_player.py`: A `MuZeroPlayer(BasePlayer)` that:  
   - Loads a trained model checkpoint from a file path passed to its constructor  
   - Implements `choose_move(board)` using MCTS guided by the neural network  
   - Falls back to random if no checkpoint is loaded (so it can run untrained)

2. `networks.py`: PyTorch nn.Module classes:  
   - `RepresentationNetwork(board_input) -> hidden_state`  
   - `DynamicsNetwork(hidden_state, action) -> (next_hidden_state, reward)`  
   - `PredictionNetwork(hidden_state) -> (policy_logits, value)`  
   - All sized appropriately for a 6x7 Connect 4 board

3. `mcts.py`: A `MCTS` class with `run(root_state, num_simulations)` that returns visit counts per action. Include `MCTSNode` with `visit_count`, `value_sum`, `prior`, `children`. Use UCB selection. Leave `expand()` and `backup()` clearly stubbed with docstrings explaining what training will fill in.

4. `trainer.py`: A `MuZeroTrainer` class scaffold with:  
   - `self_play()` method stub  
   - `train_step(batch)` method stub  
   - `save_checkpoint(path)` / `load_checkpoint(path)` methods  
   - Clear docstrings explaining the training loop

5. `replay_buffer.py`: A `ReplayBuffer` with `store(game_history)` and `sample(batch_size)` stubs.

6. `config.py`: A `MuZeroConfig` dataclass with all hyperparameters (num_simulations, lr, batch_size, hidden_size, etc.)

Do NOT implement the training logic — leave it stubbed with detailed docstrings. The goal is a structure I can train myself. All components should be importable and the player should work in the existing Game loop even without a checkpoint.  
```

---

## Phase 3: Local GUI

**Goal:** A visual desktop app using Pygame.

**Prompt:**  
```  
Add a Pygame-based GUI for the Connect 4 game in `connect4/gui/pygame_gui.py`:

- Render the 6x7 board with colored circles (yellow/red pieces, blue board)  
- Animate piece dropping  
- Show whose turn it is and display win/draw messages  
- Create a `PygameGame` class that wraps the existing `Game` engine — do not duplicate game logic  
- Add a simple start screen where the user can select their opponent from: Human, Random, Greedy, Epsilon-Greedy, MuZero  
- Add a `--gui` flag to `main.py` to launch Pygame instead of CLI

Use pygame 2.x. Add pygame to pyproject.toml dependencies.  
```

---

## Phase 4: GitHub-Ready Polish

**Prompt:**  
```  
Prepare the project for open source release:

1. Add a comprehensive `README.md` covering: project description, installation, how to play (CLI and GUI), AI descriptions, how to train MuZero, and project structure

2. Add `.gitignore` for Python, checkpoints, and virtual envs

3. Add `tests/` directory with pytest tests for:  
   - Board: valid moves, win detection (horizontal, vertical, diagonal), draw detection  
   - GreedyPlayer: verify it takes winning moves and blocks opponent wins  
   - Game: full game simulation with two random players runs to completion

4. Add a `Makefile` with targets: `install`, `test`, `run-cli`, `run-gui`

5. Ensure the project installs cleanly with `pip install -e .`  
```

---

## Phase 5: Web API + Frontend

**Prompt:**  
```  
Create a web version of the Connect 4 game:

1. `server/api.py` using FastAPI:  
   - `POST /game/new` — creates a new game session, returns game_id, accepts `opponent` parameter  
   - `POST /game/{game_id}/move` — accepts `column`, returns new board state, winner if any  
   - `GET /game/{game_id}/state` — returns current board as JSON  
   - Store game sessions in memory (dict) for now  
   - AI players run server-side

2. A single-file `frontend/index.html` using vanilla JS and CSS:  
   - Renders a Connect 4 board with clickable columns  
   - Calls the FastAPI backend  
   - Shows win/draw messages and a restart button  
   - Opponent selector dropdown

3. Add uvicorn to dependencies and a `make run-web` target

Keep it simple — no database, no auth. This is a prototype.  
```

---

## Phase 6: iOS App

At this stage the path forks depending on your preference. Here's the prompt to discuss the options:

**Prompt:**  
```  
I want to make the Connect 4 app available on iOS. My backend is a FastAPI Python server.   
Recommend the best approach for building the iOS frontend given:  
- I want to share code with the web version if possible  
- I am comfortable with Python but less so with Swift  
- I want to submit to the App Store eventually

Give me a comparison of: React Native, Flutter, and a Swift app using WKWebView (wrapping the web frontend). Then scaffold whichever you recommend as the best balance of effort and quality for a solo developer.  
```

---

## MuZero Training (Your Local GPU Work)

After Phase 2, when you're ready to train:

**Prompt:**  
```  
I have the MuZero scaffold built. Help me implement the training loop in `connect4/players/muzero/trainer.py`. 

I have a local machine with a GPU. Implement:  
1. The self-play loop that generates game trajectories using MCTS  
2. The training step with the MuZero loss: L = L_value + L_policy + L_reward  
3. Target computation: use MCTS visit counts as policy targets, bootstrapped value targets  
4. Training script `train_muzero.py` at the root with argument parsing for config overrides  
5. TensorBoard logging of losses and game statistics

Use PyTorch. Assume CUDA is available but add a CPU fallback.  
```

---

## Recommended Order of Execution

Start each Claude Code session fresh from the project root with the prompts above in order. Between phases, always run the tests before moving on. A good rule of thumb: if Claude Code's output for a phase exceeds ~300 lines, break it into sub-prompts (a, b, c) as shown in Phase 2.