from __future__ import annotations

import random

from connect4.board import Board
from connect4.players.greedy_player import GreedyPlayer


class EpsilonGreedyPlayer(GreedyPlayer):
    def __init__(self, player_id: int, epsilon: float = 0.1):
        super().__init__(player_id)
        self.epsilon = epsilon

    def choose_move(self, board: Board) -> int:
        if random.random() < self.epsilon:
            return random.choice(board.get_valid_moves())
        return super().choose_move(board)

    @property
    def name(self) -> str:
        return f"EpsilonGreedy(ε={self.epsilon})"
