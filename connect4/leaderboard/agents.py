"""Agent factory registry: produces players with the correct player_id."""

from __future__ import annotations

from typing import Callable

from connect4.players.base import BasePlayer

# A factory takes player_id (int) and returns a BasePlayer.
PlayerFactory = Callable[[int], BasePlayer]


def get_default_agents(
    dqn_ckpt: str = "checkpoints/dqn/best.pt",
    ppo_ckpt: str = "checkpoints/ppo/best.pt",
    alphazero_ckpt: str = "checkpoints/alphazero/best.pt",
) -> dict[str, PlayerFactory]:
    """Return the standard agent registry.

    Each value is a callable that takes ``player_id`` and returns a player
    configured for that side of the board.
    """
    from connect4.players.random_player import RandomPlayer
    from connect4.players.greedy_player import GreedyPlayer
    from connect4.players.epsilon_greedy_player import EpsilonGreedyPlayer
    from connect4.players.rl.dqn.dqn_player import DQNPlayer
    from connect4.players.rl.ppo.ppo_player import PPOPlayer
    from connect4.players.rl.alphazero.alphazero_player import AlphaZeroPlayer

    return {
        "Random": lambda pid: RandomPlayer(),
        "Greedy": lambda pid: GreedyPlayer(pid),
        "EpsilonGreedy": lambda pid: EpsilonGreedyPlayer(pid, epsilon=0.1),
        "DQN": lambda pid: DQNPlayer(pid, dqn_ckpt),
        "PPO": lambda pid: PPOPlayer(pid, ppo_ckpt),
        "AlphaZero": lambda pid: AlphaZeroPlayer(pid, alphazero_ckpt),
    }
