"""Dump Python-side DQN and PPO outputs + chosen moves as JSON fixtures.

Used by the Vitest suite (via onnxruntime-node) to verify that the TypeScript
inference pipeline reproduces the exact same chosen moves as the Python
players. This locks down:
  - ONNX export correctness
  - Canonicalization parity
  - Mask-invalid logic
  - argmax behavior (ties broken identically on the same raw values)

Run:
    python scripts/dump_nn_fixtures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from connect4.board import Board, P1, P2
from connect4.players.rl.common import board_to_tensor, mask_invalid
from connect4.players.rl.networks import DQNNet, PPONet

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS = REPO_ROOT / "checkpoints"
OUT_PATH = REPO_ROOT / "web" / "src" / "players" / "__fixtures__" / "nn_outputs.json"


def _play_random(seed: int, num_moves: int) -> tuple[Board, int]:
    rng = np.random.default_rng(seed)
    board = Board()
    current = P1
    for _ in range(num_moves):
        valid = board.get_valid_moves()
        if not valid or board.check_winner() is not None:
            break
        col = int(rng.choice(valid))
        board.drop_piece(col, current)
        current = P2 if current == P1 else P1
    return board, current


@torch.no_grad()
def _dqn_move(net: DQNNet, board: Board, current_player: int) -> tuple[list[float], int]:
    state = board_to_tensor(board, current_player).unsqueeze(0)
    q = net(state).squeeze(0)
    masked = mask_invalid(q.clone(), board)
    return q.tolist(), int(masked.argmax().item())


@torch.no_grad()
def _ppo_move(net: PPONet, board: Board, current_player: int) -> tuple[list[float], int]:
    state = board_to_tensor(board, current_player).unsqueeze(0)
    logits, _ = net(state)
    logits = logits.squeeze(0)
    masked = mask_invalid(logits.clone(), board)
    return logits.tolist(), int(masked.argmax().item())


def main() -> None:
    dqn = DQNNet()
    dqn.load_state_dict(torch.load(CHECKPOINTS / "dqn" / "best.pt", map_location="cpu", weights_only=True))
    dqn.eval()

    ppo = PPONet()
    ppo.load_state_dict(torch.load(CHECKPOINTS / "ppo" / "best.pt", map_location="cpu", weights_only=True))
    ppo.eval()

    cases: list[tuple[str, Board, int]] = []

    # Named positions.
    cases.append(("empty_p1", Board(), P1))
    cases.append(("empty_p2", Board(), P2))

    b = Board()
    b.drop_piece(3, P1)
    cases.append(("center_p2_to_move", b, P2))

    # Mid-game random positions.
    for seed, moves in [(1, 5), (2, 10), (3, 15), (4, 20), (5, 25)]:
        board, current = _play_random(seed, moves)
        cases.append((f"random_seed{seed}_moves{moves}", board, current))

    fixtures = []
    for name, board, current in cases:
        dqn_raw, dqn_move = _dqn_move(dqn, board, current)
        ppo_raw, ppo_move = _ppo_move(ppo, board, current)
        fixtures.append({
            "name": name,
            "current_player": current,
            "grid": board.grid.astype(int).flatten().tolist(),
            "dqn_q_values": dqn_raw,
            "dqn_chosen_move": dqn_move,
            "ppo_logits": ppo_raw,
            "ppo_chosen_move": ppo_move,
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(fixtures, indent=2))
    print(f"Wrote {len(fixtures)} fixtures to {OUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
