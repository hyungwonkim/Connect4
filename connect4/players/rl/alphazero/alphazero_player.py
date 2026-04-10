"""AlphaZero player — inference wrapper using MCTS + neural network."""

import torch

from connect4.board import Board
from connect4.players.base import BasePlayer
from connect4.players.rl.common import get_device
from connect4.players.rl.networks import AlphaZeroNetV2
from connect4.players.rl.alphazero.mcts import MCTS


class AlphaZeroPlayer(BasePlayer):
    """Plays Connect 4 using AlphaZero-style MCTS with a learned network."""

    def __init__(
        self,
        player_id: int,
        checkpoint_path: str = "checkpoints/alphazero/best.pt",
        num_simulations: int = 400,
        c_puct: float = 1.41,
    ):
        self.player_id = player_id
        self.checkpoint_path = checkpoint_path
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.device = get_device()

        self.network = AlphaZeroNetV2(channels=96, num_blocks=5).to(self.device)
        self._has_checkpoint = self._load_checkpoint()
        self.network.eval()
        self.mcts = MCTS(self.network, num_simulations, c_puct)

    def _load_checkpoint(self) -> bool:
        try:
            state_dict = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
            self.network.load_state_dict(state_dict)
            return True
        except (FileNotFoundError, RuntimeError):
            return False

    def choose_move(self, board: Board) -> int:
        visits = self.mcts.search(board, self.player_id)
        return max(range(7), key=lambda c: visits[c])

    @property
    def name(self) -> str:
        suffix = "" if self._has_checkpoint else " (untrained)"
        return f"AlphaZero{suffix}"
