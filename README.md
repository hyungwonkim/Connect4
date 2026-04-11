# Connect 4

A Connect 4 game engine with rule-based and RL-trained AI opponents.

## Play in your browser

No install required — play against all three RL agents (DQN, PPO, AlphaZero with in-browser MCTS) at
**[hyungwonkim.github.io/Connect4/play/](https://hyungwonkim.github.io/Connect4/play/)**.

The web build is a static Vite + TypeScript app under `web/`, with the Python networks exported to
ONNX and run via `onnxruntime-web`. See `web/README.md` for local dev instructions.

## Quick Start

```bash
pip install -e .
```

## Play

```bash
# Human vs Human
python main.py

# Human vs AI
python main.py --player2 greedy
python main.py --player2 dqn
python main.py --player2 alphazero

# Pygame GUI
python main.py --gui
```

**Opponents:** `human`, `random`, `greedy`, `epsilon_greedy`, `dqn`, `ppo`, `alphazero`

RL players (`dqn`, `ppo`, `alphazero`) require a trained checkpoint. Use `--checkpoint path/to/model.pt` or place it at the default location (`checkpoints/<agent>/best.pt`).

## Train

```bash
python -m connect4.training.train_dqn
python -m connect4.training.train_ppo
python -m connect4.training.train_alphazero
```

Checkpoints save to `checkpoints/`. Monitor training with TensorBoard:

```bash
tensorboard --logdir runs/
```

## Evaluate

```bash
python -m connect4.training.evaluate
```

Runs a round-robin tournament and prints a win-rate matrix.

## Docs

Full documentation: [hyungwonkim.github.io/Connect4](https://hyungwonkim.github.io/Connect4/)
