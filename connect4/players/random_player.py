from __future__ import annotations

import random

from connect4.board import Board
from connect4.players.base import BasePlayer


class RandomPlayer(BasePlayer):
    def choose_move(self, board: Board) -> int:
        return random.choice(board.get_valid_moves())

    @property
    def name(self) -> str:
        return "Random"
