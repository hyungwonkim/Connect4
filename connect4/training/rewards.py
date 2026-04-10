"""Reward shaping helpers for RL training.

All shaping signals are small (<= 0.05) so the terminal ±1.x reward still
dominates the return. Reward-shaping is applied AFTER the agent's move and,
optionally, after the opponent's reply.
"""

from __future__ import annotations

import numpy as np

from connect4.board import Board, ROWS, COLS, EMPTY

# Direction vectors (horizontal, vertical, two diagonals)
_DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]

# Center columns (mathematically stronger in Connect 4)
_CENTER_COLS = {2, 3, 4}

# Shaping magnitudes
R_CREATE_THREE = 0.05
R_ALLOW_THREE = -0.05
R_BLOCK_THREE = 0.02
R_CENTER = 0.01


def _longest_run_through(grid: np.ndarray, row: int, col: int, player: int) -> int:
    """Return the longest contiguous run of *player* passing through (row, col)."""
    best = 1
    for dr, dc in _DIRECTIONS:
        count = 1
        for sign in (1, -1):
            r, c = row + dr * sign, col + dc * sign
            while 0 <= r < ROWS and 0 <= c < COLS and grid[r, c] == player:
                count += 1
                r += dr * sign
                c += dc * sign
        if count > best:
            best = count
    return best


def _had_immediate_threat(board: Board, player: int) -> bool:
    """True if *player* had an immediate winning move on this board."""
    for col in board.get_valid_moves():
        sim = board.copy()
        sim.drop_piece(col, player)
        if sim.check_winner() == player:
            return True
    return False


def shape_agent_move(
    board_before: Board,
    board_after: Board,
    action: int,
    row_placed: int,
    player: int,
) -> float:
    """Shaping reward for the *agent's* own move.

    Components:
      - +0.02 if the move blocked an opponent immediate-win threat.
      - +0.05 if the move created a new 3-in-a-row (run of exactly 3).
      - +0.01 if the chosen column is a center column.
    """
    opponent = 3 - player  # P1<->P2 (P1=1, P2=2)
    reward = 0.0

    # Block detection: opponent had an immediate win, and now doesn't.
    if _had_immediate_threat(board_before, opponent) and \
       not _had_immediate_threat(board_after, opponent):
        reward += R_BLOCK_THREE

    # Created a new 3-run through the placed piece?
    run = _longest_run_through(board_after.grid, row_placed, action, player)
    if run == 3:
        reward += R_CREATE_THREE

    # Center preference
    if action in _CENTER_COLS:
        reward += R_CENTER

    return reward


def shape_opponent_move(
    board_before: Board,
    board_after: Board,
    opponent_action: int,
    opponent_row: int,
    opponent: int,
) -> float:
    """Shaping penalty applied to the agent if the opponent just created a 3-run.

    This approximates "the agent failed to prevent a threat."
    Returns a NEGATIVE number (or 0).
    """
    run = _longest_run_through(board_after.grid, opponent_row, opponent_action, opponent)
    if run == 3:
        return R_ALLOW_THREE
    return 0.0
