"""Tests for RL common utilities."""

import torch

from connect4.board import Board, P1, P2, ROWS, COLS
from connect4.players.rl.common import board_to_tensor, mask_invalid


def test_board_to_tensor_shape():
    board = Board()
    tensor = board_to_tensor(board, P1)
    assert tensor.shape == (3, ROWS, COLS)


def test_board_to_tensor_empty_board():
    board = Board()
    tensor = board_to_tensor(board, P1)
    # No pieces on empty board
    assert tensor[0].sum() == 0  # current player plane
    assert tensor[1].sum() == 0  # opponent plane
    # All 7 columns are valid
    assert tensor[2, 0, :].sum() == 7


def test_board_to_tensor_canonicalization():
    """Current player's pieces always on plane 0 regardless of P1/P2."""
    board = Board()
    board.drop_piece(3, P1)
    board.drop_piece(4, P2)

    # From P1's perspective
    t1 = board_to_tensor(board, P1)
    assert t1[0, 5, 3] == 1.0  # P1's piece on plane 0
    assert t1[1, 5, 4] == 1.0  # P2's piece on plane 1

    # From P2's perspective — planes are swapped
    t2 = board_to_tensor(board, P2)
    assert t2[0, 5, 4] == 1.0  # P2's piece on plane 0
    assert t2[1, 5, 3] == 1.0  # P1's piece on plane 1


def test_board_to_tensor_valid_move_plane():
    board = Board()
    # Fill column 0 completely
    for _ in range(3):
        board.drop_piece(0, P1)
        board.drop_piece(0, P2)

    tensor = board_to_tensor(board, P1)
    assert tensor[2, 0, 0] == 0.0  # column 0 is full
    assert tensor[2, 0, 1] == 1.0  # column 1 is valid


def test_mask_invalid():
    board = Board()
    # Fill column 0
    for _ in range(3):
        board.drop_piece(0, P1)
        board.drop_piece(0, P2)

    logits = torch.zeros(7)
    masked = mask_invalid(logits, board)

    assert masked[0] == float("-inf")
    assert masked[1] == 0.0  # valid column unchanged


def test_mask_invalid_preserves_values():
    board = Board()
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    masked = mask_invalid(logits, board)
    # All columns valid on empty board — values preserved
    assert torch.equal(masked, logits)
