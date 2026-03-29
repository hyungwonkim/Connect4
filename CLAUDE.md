# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Connect 4 game progressing through multiple phases: Python CLI → AI players → Pygame GUI → GitHub polish → Web app (FastAPI + vanilla JS) → iOS app. See `Connect4Plan.md` for the full roadmap and phase-specific prompts.

## Architecture

- **Game engine** (`connect4/board.py`, `connect4/game.py`): Core logic with `Board` (6x7 grid, 0=empty/1=P1/2=P2) and `Game` (turn management, player loop).
- **Player hierarchy** (`connect4/players/`): Abstract `BasePlayer` with `choose_move(board) -> int`. Implementations: `HumanPlayer`, `RandomPlayer`, `GreedyPlayer`, `EpsilonGreedyPlayer`, `MuZeroPlayer`.
- **MuZero subsystem** (`connect4/players/muzero/`): PyTorch-based with `RepresentationNetwork`, `DynamicsNetwork`, `PredictionNetwork`, MCTS, replay buffer, and trainer. Falls back to random play without a checkpoint.
- **GUI** (`connect4/gui/pygame_gui.py`): Pygame 2.x wrapper around existing `Game` engine — no duplicated game logic.
- **Web** (`server/api.py`): FastAPI backend with in-memory game sessions; `frontend/index.html` vanilla JS frontend.
- **Entry point**: `main.py` with `--player2` flag (human/random/greedy/epsilon_greedy) and `--gui` flag.

## Build & Run Commands

```bash
pip install -e .           # Install project
python main.py             # CLI human vs human
python main.py --player2 greedy --gui  # GUI with AI opponent
make install               # Install dependencies
make test                  # Run pytest suite
make run-cli               # CLI mode
make run-gui               # Pygame mode
make run-web               # FastAPI + frontend (uvicorn)
```

## Testing

Tests live alongside the source files they test, using the `<module>_test.py` naming convention (e.g., `connect4/board_test.py` tests `connect4/board.py`). Key test areas: Board (win detection for all orientations), GreedyPlayer (`_longest_sequence` correctness), EpsilonGreedyPlayer (matches greedy at ε=0, explores at high ε).

```bash
pytest                                                  # Run all tests
pytest connect4/board_test.py -k "test_horizontal_win"  # Single test
```
