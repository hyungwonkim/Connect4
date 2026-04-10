"""Opponent pool for DQN v2 / PPO v2 training.

Mixes rule-based baselines (Random, Greedy) with past network snapshots so the
learner cannot converge to a lazy strategy that only beats itself.
"""

from __future__ import annotations

import copy
import random
from typing import Callable, Literal

import torch

from connect4.board import Board
from connect4.players.random_player import RandomPlayer
from connect4.players.greedy_player import GreedyPlayer
from connect4.players.rl.common import board_to_tensor, mask_invalid

# An opponent is a callable (board, player_id) -> int (column)
OpponentFn = Callable[[Board, int], int]

# Tag is included so callers can log per-opponent-type win rates
OpponentTag = Literal["random", "greedy", "snapshot"]


def random_opponent() -> OpponentFn:
    player = RandomPlayer()
    def fn(board: Board, player_id: int) -> int:
        return player.choose_move(board)
    return fn


def greedy_opponent(player_id: int) -> OpponentFn:
    player = GreedyPlayer(player_id)
    def fn(board: Board, pid: int) -> int:
        return player.choose_move(board)
    return fn


def snapshot_opponent(network: torch.nn.Module, device: torch.device, is_policy: bool) -> OpponentFn:
    """Create an opponent from a frozen copy of a network.

    is_policy=True  -> treat outputs as (logits, value) (PPO-style)
    is_policy=False -> treat outputs as Q-values (DQN-style)
    """
    net = copy.deepcopy(network)
    net.eval()

    @torch.no_grad()
    def fn(board: Board, player_id: int) -> int:
        state = board_to_tensor(board, player_id).unsqueeze(0).to(device)
        out = net(state)
        if is_policy:
            logits = out[0]
        else:
            logits = out
        logits = logits.squeeze(0).cpu()
        masked = mask_invalid(logits, board)
        return masked.argmax().item()

    return fn


class OpponentPool:
    """Weighted sampler over Random / Greedy / self-snapshots.

    Weights are supplied at construction time. Snapshots are added
    incrementally; a cap keeps only the most recent *max_snapshots*.
    """

    def __init__(
        self,
        *,
        device: torch.device,
        is_policy: bool,
        weights: tuple[float, float, float],  # (random, greedy, snapshot)
        opponent_player_id: int,
        max_snapshots: int = 8,
    ):
        assert len(weights) == 3
        self.device = device
        self.is_policy = is_policy
        self.weights = weights
        self.opponent_player_id = opponent_player_id
        self.max_snapshots = max_snapshots

        self._random = random_opponent()
        self._greedy = greedy_opponent(opponent_player_id)
        self._snapshots: list[OpponentFn] = []

    def add_snapshot(self, network: torch.nn.Module) -> None:
        self._snapshots.append(snapshot_opponent(network, self.device, self.is_policy))
        if len(self._snapshots) > self.max_snapshots:
            self._snapshots.pop(0)

    def sample(self) -> tuple[OpponentFn, OpponentTag]:
        w_random, w_greedy, w_snapshot = self.weights
        # If no snapshots yet, redistribute snapshot weight across the other two
        if not self._snapshots:
            total_rg = w_random + w_greedy
            if total_rg <= 0:
                return self._random, "random"
            p_random = w_random / total_rg
            if random.random() < p_random:
                return self._random, "random"
            return self._greedy, "greedy"

        total = w_random + w_greedy + w_snapshot
        r = random.random() * total
        if r < w_random:
            return self._random, "random"
        if r < w_random + w_greedy:
            return self._greedy, "greedy"
        return random.choice(self._snapshots), "snapshot"

    @property
    def num_snapshots(self) -> int:
        return len(self._snapshots)
