from __future__ import annotations

import copy
from typing import Optional

ROWS = 6
COLS = 7
EMPTY = 0
P1 = 1
P2 = 2


class Board:
    def __init__(self):
        self.grid = [[EMPTY] * COLS for _ in range(ROWS)]

    def is_valid_move(self, col: int) -> bool:
        return 0 <= col < COLS and self.grid[0][col] == EMPTY

    def get_valid_moves(self) -> list[int]:
        return [c for c in range(COLS) if self.is_valid_move(c)]

    def drop_piece(self, col: int, player: int) -> int:
        if not self.is_valid_move(col):
            raise ValueError(f"Invalid move: column {col}")
        for row in range(ROWS - 1, -1, -1):
            if self.grid[row][col] == EMPTY:
                self.grid[row][col] = player
                return row
        raise ValueError(f"Column {col} is full")

    def check_winner(self) -> Optional[int]:
        for r in range(ROWS):
            for c in range(COLS):
                p = self.grid[r][c]
                if p == EMPTY:
                    continue
                # horizontal
                if c + 3 < COLS and all(self.grid[r][c + i] == p for i in range(4)):
                    return p
                # vertical
                if r + 3 < ROWS and all(self.grid[r + i][c] == p for i in range(4)):
                    return p
                # diagonal down-right
                if r + 3 < ROWS and c + 3 < COLS and all(self.grid[r + i][c + i] == p for i in range(4)):
                    return p
                # diagonal down-left
                if r + 3 < ROWS and c - 3 >= 0 and all(self.grid[r + i][c - i] == p for i in range(4)):
                    return p
        return None

    def is_draw(self) -> bool:
        return self.check_winner() is None and len(self.get_valid_moves()) == 0

    def copy(self) -> "Board":
        return copy.deepcopy(self)

    def __str__(self) -> str:
        symbols = {EMPTY: ".", P1: "X", P2: "O"}
        rows = [" ".join(symbols[cell] for cell in row) for row in self.grid]
        header = " ".join(str(c) for c in range(COLS))
        return "\n".join(rows) + "\n" + header
