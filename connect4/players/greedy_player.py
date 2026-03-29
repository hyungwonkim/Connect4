from __future__ import annotations

import random

from connect4.board import Board, P1, P2, ROWS, COLS
from connect4.players.base import BasePlayer

# Direction vectors: horizontal, vertical, diagonal down-right, diagonal down-left
_DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]


class GreedyPlayer(BasePlayer):
    def __init__(self, player_id: int):
        self.player_id = player_id
        self.opponent_id = P1 if player_id == P2 else P2

    @staticmethod
    def _longest_sequence(grid, row: int, col: int, player: int) -> int:
        """Return the longest contiguous run through (row, col) across all directions."""
        best = 1
        for dr, dc in _DIRECTIONS:
            count = 1
            for sign in (1, -1):
                r, c = row + dr * sign, col + dc * sign
                while 0 <= r < ROWS and 0 <= c < COLS and grid[r][c] == player:
                    count += 1
                    r += dr * sign
                    c += dc * sign
            if count > best:
                best = count
        return best

    def choose_move(self, board: Board) -> int:
        valid = board.get_valid_moves()

        # Check for immediate win
        for col in valid:
            sim = board.copy()
            sim.drop_piece(col, self.player_id)
            if sim.check_winner() == self.player_id:
                return col

        # Check for opponent immediate win and block it
        for col in valid:
            sim = board.copy()
            sim.drop_piece(col, self.opponent_id)
            if sim.check_winner() == self.opponent_id:
                return col

        # Pick the move that extends the longest existing sequence
        best_len = 0
        best_cols = []
        for col in valid:
            sim = board.copy()
            row = sim.drop_piece(col, self.player_id)
            seq = self._longest_sequence(sim.grid, row, col, self.player_id)
            if seq > best_len:
                best_len = seq
                best_cols = [col]
            elif seq == best_len:
                best_cols.append(col)

        return random.choice(best_cols)

    @property
    def name(self) -> str:
        return "Greedy"
