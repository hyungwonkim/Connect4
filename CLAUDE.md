# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Connect 4 game with rule-based and RL-trained AI opponents. Phases: Python CLI → AI players → Pygame GUI → Web app (FastAPI + vanilla JS) → iOS app. See `Connect4Plan.md` for the full roadmap.

## Build & Run

```bash
pip install -e .                                    # Install project (pyproject.toml, requires Python ≥3.9)
python main.py                                      # CLI human vs human
python main.py --player2 dqn --gui                  # GUI with RL opponent
python -m connect4.leaderboard --num-games 50       # Round-robin tournament
python -m connect4.training.train_dqn               # Train DQN (see --help for args)
python -m connect4.training.train_ppo               # Train PPO
python -m connect4.training.train_alphazero          # Train AlphaZero
tensorboard --logdir runs/                           # Monitor training
```

## Testing

Tests use co-located `<module>_test.py` convention (configured in pyproject.toml). No Makefile — use pytest directly.

```bash
pytest                                                  # All tests
pytest connect4/board_test.py -k "test_horizontal_win"  # Single test
```

## Architecture

**Game engine** (`connect4/board.py`, `connect4/game.py`): `Board` is a 6x7 numpy int8 grid. Constants: `EMPTY=0`, `P1=1`, `P2=2`, `ROWS=6`, `COLS=7`.

**Player hierarchy** (`connect4/players/`): Abstract `BasePlayer.choose_move(board) -> int`. Implementations: Human, Random, Greedy, EpsilonGreedy, DQN, PPO, AlphaZero.

**RL subsystem** (`connect4/players/rl/`):
- `common.py`: `board_to_tensor(board, player)` canonicalizes to 3-channel (my pieces, opponent pieces, valid moves) tensor. `mask_invalid()` sets full-column logits to -inf. `get_device()` prefers MPS on Apple Silicon.
- `networks.py`: All networks share `Connect4CNN` backbone (3 conv layers, 64ch → 2688 features). `DQNNet` is Dueling (V+A streams). `PPONet` is actor-critic. `AlphaZeroNetV2` uses residual blocks with separate policy/value heads.
- Players load from `checkpoints/<agent>/best.pt`; gracefully fall back to random play if missing.

**Training** (`connect4/training/`):
- `opponents.py`: `OpponentPool` — weighted sampling over Random/Greedy/self-snapshots (frozen past network copies).
- `rewards.py`: Shaping bonuses (<0.05) for 3-in-a-row, blocking, center play. Enabled by default, disable with `--no-shape-rewards`.
- `per_buffer.py`: Prioritized Experience Replay for DQN (proportional, α=0.6, β annealed).
- Training scripts save `best.pt` based on deterministic eval vs Greedy (both sides, argmax policy).

**Leaderboard** (`connect4/leaderboard/`): `agents.py` has a `PlayerFactory` registry. `matchup.py` builds NxN win-rate matrices. Run via `python -m connect4.leaderboard`.

**Board state canonicalization** is a key abstraction: all RL networks see the board from the "current player's perspective" regardless of whether the agent plays as P1 or P2. This is handled by `board_to_tensor(board, current_player)`.

## Key Files

| Area | File | Purpose |
|------|------|---------|
| Entry | `main.py` | CLI with `--player2`, `--checkpoint`, `--gui` flags |
| Board | `connect4/board.py` | Core game logic, win detection, P1/P2 constants |
| RL common | `connect4/players/rl/common.py` | Tensor conversion, move masking, device selection |
| Networks | `connect4/players/rl/networks.py` | All neural architectures (CNN backbone, DQN, PPO, AlphaZero) |
| Training | `connect4/training/train_dqn.py` | DQN training (Dueling + PER + opponent pool) |
| Training | `connect4/training/train_ppo.py` | PPO training (GAE + entropy decay + opponent pool) |
| Training | `connect4/training/train_alphazero.py` | AlphaZero self-play + MCTS training |
| Eval | `connect4/leaderboard/matchup.py` | Round-robin tournament logic |

## Conventions

- Trained weights (`checkpoints/*/best.pt`) are committed to the repo; intermediate checkpoints are gitignored.
- Training logs go to `runs/` (gitignored), viewable with TensorBoard.
- `--player2` choices in `main.py`: `human`, `random`, `greedy`, `epsilon_greedy`, `dqn`, `ppo`, `alphazero`.
