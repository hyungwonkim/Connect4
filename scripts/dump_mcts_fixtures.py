"""Dump Python AlphaZero MCTS chosen moves on a set of fixed positions.

The TypeScript port is tested against these fixtures. Because MCTS uses
batched virtual-loss traversal, exact parity is impractical — we instead
require that the top move matches in the large majority of positions
(see web/src/players/alphazero/mcts.test.ts).

Run:
    python scripts/dump_mcts_fixtures.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from connect4.board import Board, P1, P2
from connect4.players.rl.alphazero.mcts import MCTS
from connect4.players.rl.networks import AlphaZeroNetV2

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = REPO_ROOT / "checkpoints" / "alphazero" / "best.pt"
OUT_PATH = REPO_ROOT / "web" / "src" / "players" / "__fixtures__" / "mcts_positions.json"

NUM_SIMULATIONS = 200


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


def main() -> None:
    # Deterministic runtime state.
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # Match the shipped AlphaZero checkpoint architecture.
    net = AlphaZeroNetV2(channels=96, num_blocks=5)
    net.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True))
    net.eval()

    mcts = MCTS(net, num_simulations=NUM_SIMULATIONS, c_puct=1.41, batch_size=8)

    positions: list[tuple[str, Board, int]] = []

    # Empty board.
    positions.append(("empty_p1", Board(), P1))

    # Must-win for P1.
    b = Board()
    b.drop_piece(0, P1); b.drop_piece(0, P2)
    b.drop_piece(1, P1); b.drop_piece(1, P2)
    b.drop_piece(2, P1); b.drop_piece(2, P2)
    positions.append(("must_win_col3", b, P1))

    # Must-block: P2 threatens bottom row 0,1,2; P1 must block col 3.
    b = Board()
    b.drop_piece(0, P2); b.drop_piece(0, P1)
    b.drop_piece(1, P2); b.drop_piece(1, P1)
    b.drop_piece(2, P2); b.drop_piece(6, P1)
    positions.append(("must_block_col3", b, P1))

    # Assorted mid-game random positions.
    for seed, moves in [
        (1, 4), (2, 6), (3, 8), (4, 10),
        (5, 12), (6, 14), (7, 16), (8, 18),
    ]:
        board, current = _play_random(seed, moves)
        if board.check_winner() is None and board.get_valid_moves():
            positions.append((f"rand_s{seed}_m{moves}", board, current))

    fixtures = []
    for name, board, current in positions:
        visits = mcts.search(board, current)
        chosen = int(max(range(7), key=lambda c: visits[c]))
        fixtures.append({
            "name": name,
            "current_player": current,
            "grid": board.grid.astype(int).flatten().tolist(),
            "num_simulations": NUM_SIMULATIONS,
            "visits": visits,
            "chosen_move": chosen,
        })
        print(f"  {name}: chose {chosen}  visits={['%.2f' % v for v in visits]}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(fixtures, indent=2))
    print(f"\nWrote {len(fixtures)} MCTS fixtures to {OUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
