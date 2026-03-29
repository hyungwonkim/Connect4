"""Shared utilities for RL players."""

import torch

from connect4.board import Board, ROWS, COLS, P1, P2


def board_to_tensor(board: Board, current_player: int) -> torch.Tensor:
    """Convert a Board to a canonical (3, 6, 7) tensor.

    Plane 0: current player's pieces (1s)
    Plane 1: opponent's pieces (1s)
    Plane 2: valid-move indicator (1 in top row of each playable column)
    """
    opponent = P2 if current_player == P1 else P1
    tensor = torch.zeros(3, ROWS, COLS, dtype=torch.float32)

    for r in range(ROWS):
        for c in range(COLS):
            cell = board.grid[r][c]
            if cell == current_player:
                tensor[0, r, c] = 1.0
            elif cell == opponent:
                tensor[1, r, c] = 1.0

    for c in board.get_valid_moves():
        tensor[2, 0, c] = 1.0

    return tensor


def mask_invalid(logits: torch.Tensor, board: Board) -> torch.Tensor:
    """Set logits for full columns to -inf (before softmax)."""
    valid = board.get_valid_moves()
    mask = torch.full_like(logits, float("-inf"))
    for c in valid:
        mask[c] = 0.0
    return logits + mask


def get_device() -> torch.device:
    """Return best available device (MPS for Apple Silicon, else CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
