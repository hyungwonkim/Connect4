"""PPO player — inference wrapper."""

import torch

from connect4.board import Board
from connect4.players.base import BasePlayer
from connect4.players.rl.common import board_to_tensor, mask_invalid, get_device
from connect4.players.rl.networks import PPONet


class PPOPlayer(BasePlayer):
    """Plays Connect 4 using a trained PPO policy network."""

    def __init__(
        self,
        player_id: int,
        checkpoint_path: str = "checkpoints/ppo/best.pt",
    ):
        self.player_id = player_id
        self.checkpoint_path = checkpoint_path
        self.device = get_device()

        self.network = PPONet().to(self.device)
        self._has_checkpoint = self._load_checkpoint()
        self.network.eval()

    def _load_checkpoint(self) -> bool:
        try:
            state_dict = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
            self.network.load_state_dict(state_dict)
            return True
        except (FileNotFoundError, RuntimeError):
            return False

    @torch.no_grad()
    def choose_move(self, board: Board) -> int:
        state = board_to_tensor(board, self.player_id).unsqueeze(0).to(self.device)
        logits, _ = self.network(state)
        logits = logits.squeeze(0).cpu()
        masked = mask_invalid(logits, board)
        return masked.argmax().item()

    @property
    def name(self) -> str:
        suffix = "" if self._has_checkpoint else " (untrained)"
        return f"PPO{suffix}"
