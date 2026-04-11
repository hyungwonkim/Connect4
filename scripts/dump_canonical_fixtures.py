"""Dump Python-side board_to_tensor outputs as JSON fixtures.

The TypeScript port of `canonicalize` must match the Python version byte-for-
byte, or ONNX inference will receive the wrong input planes and produce
garbage. This script generates a set of representative boards and writes the
canonical tensors to `web/src/players/__fixtures__/canonical.json` so the
Vitest suite can assert parity.

Run:
    python scripts/dump_canonical_fixtures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from connect4.board import Board, P1, P2
from connect4.players.rl.common import board_to_tensor

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "web" / "src" / "players" / "__fixtures__" / "canonical.json"


def _play_random(seed: int, num_moves: int) -> tuple[Board, int]:
    """Play `num_moves` random moves and return the resulting board + next player."""
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


def _fixture(name: str, board: Board, current_player: int) -> dict:
    tensor = board_to_tensor(board, current_player)  # (3, 6, 7) float32
    return {
        "name": name,
        "current_player": current_player,
        # row-major grid, values in {0,1,2}
        "grid": board.grid.astype(int).flatten().tolist(),
        # flat CHW float array, length 3*6*7 = 126
        "tensor": tensor.numpy().flatten().tolist(),
    }


def main() -> None:
    fixtures = []

    # Empty board, P1 to move.
    fixtures.append(_fixture("empty_p1", Board(), P1))
    # Empty board, P2 to move (plane swap).
    fixtures.append(_fixture("empty_p2", Board(), P2))

    # Single piece in the center.
    b = Board()
    b.drop_piece(3, P1)
    fixtures.append(_fixture("center_p1_next_p2", b, P2))

    # One full column (column 0 filled by P1 six times).
    b = Board()
    for _ in range(6):
        b.drop_piece(0, P1)
    # Note: this wouldn't happen in a real game, but exercises the
    # valid-move mask (column 0 should be unavailable).
    fixtures.append(_fixture("full_col0", b, P1))

    # Random partial games at a few depths + seeds.
    for seed, moves in [(1, 5), (2, 12), (3, 20), (4, 30)]:
        board, current = _play_random(seed, moves)
        fixtures.append(_fixture(f"random_seed{seed}_moves{moves}", board, current))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(fixtures, indent=2))
    print(f"Wrote {len(fixtures)} fixtures to {OUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
