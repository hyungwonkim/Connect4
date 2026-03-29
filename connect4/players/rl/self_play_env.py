"""Lightweight self-play environment for training (no console output)."""

import copy

from connect4.board import Board, P1, P2
from connect4.players.rl.common import board_to_tensor


class SelfPlayEnv:
    """Training wrapper around Board with canonical observations."""

    def __init__(self):
        self.board = Board()
        self.current_player = P1

    def reset(self):
        """Reset to empty board, P1 to move. Returns canonical state tensor."""
        self.board = Board()
        self.current_player = P1
        return board_to_tensor(self.board, self.current_player)

    def step(self, action: int):
        """Drop piece, check outcome, switch player.

        Returns:
            (next_obs, reward, done, info) where reward is from the
            mover's perspective: +1 win, -1 loss, 0 draw/ongoing.
        """
        mover = self.current_player
        self.board.drop_piece(action, mover)

        winner = self.board.check_winner()
        if winner is not None:
            # Game over — mover won
            obs = board_to_tensor(self.board, mover)
            return obs, 1.0, True, {"winner": mover}

        if self.board.is_draw():
            obs = board_to_tensor(self.board, mover)
            return obs, 0.0, True, {"winner": None}

        # Switch player
        self.current_player = P2 if mover == P1 else P1
        obs = board_to_tensor(self.board, self.current_player)
        return obs, 0.0, False, {}

    def get_valid_actions(self) -> list[int]:
        return self.board.get_valid_moves()

    def clone(self) -> "SelfPlayEnv":
        return copy.deepcopy(self)
