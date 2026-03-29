"""Tests for AlphaZero MCTS."""

import torch

from connect4.board import Board, P1, P2
from connect4.players.rl.networks import AlphaZeroNet
from connect4.players.rl.alphazero.mcts import MCTS


def _make_mcts(num_simulations=50):
    """Create MCTS with a fresh (untrained) network on CPU."""
    net = AlphaZeroNet()
    net.eval()
    return MCTS(net, num_simulations=num_simulations, c_puct=1.41)


def test_search_returns_distribution():
    mcts = _make_mcts()
    board = Board()
    visits = mcts.search(board, P1)
    assert len(visits) == 7
    assert abs(sum(visits) - 1.0) < 1e-5


def test_search_avoids_full_columns():
    mcts = _make_mcts()
    board = Board()
    # Fill column 3
    for _ in range(3):
        board.drop_piece(3, P1)
        board.drop_piece(3, P2)
    visits = mcts.search(board, P1)
    assert visits[3] == 0.0  # can't play in full column


def test_finds_winning_move():
    """With enough simulations, MCTS should find an obvious winning move."""
    mcts = _make_mcts(num_simulations=100)
    board = Board()
    # P1 has 3 in a row at bottom: cols 0,1,2. Col 3 wins.
    board.drop_piece(0, P1)
    board.drop_piece(0, P2)
    board.drop_piece(1, P1)
    board.drop_piece(1, P2)
    board.drop_piece(2, P1)
    board.drop_piece(2, P2)

    visits = mcts.search(board, P1)
    # Column 3 should get the most visits (winning move)
    best = max(range(7), key=lambda c: visits[c])
    assert best == 3, f"Expected col 3 (winning), got col {best}"


def test_blocks_opponent_win():
    """MCTS should block an obvious opponent winning move."""
    mcts = _make_mcts(num_simulations=100)
    board = Board()
    # P2 has 3 in a row at bottom: cols 0,1,2. P1 must block col 3.
    board.drop_piece(0, P2)
    board.drop_piece(0, P1)  # P1 plays on top
    board.drop_piece(1, P2)
    board.drop_piece(1, P1)
    board.drop_piece(2, P2)
    board.drop_piece(6, P1)  # P1 plays elsewhere

    visits = mcts.search(board, P1)
    best = max(range(7), key=lambda c: visits[c])
    assert best == 3, f"Expected col 3 (blocking), got col {best}"
