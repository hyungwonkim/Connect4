"""Evaluation: pit agents against each other in round-robin tournaments."""

from __future__ import annotations

import argparse

from connect4.board import Board, P1, P2
from connect4.players.base import BasePlayer
from connect4.players.random_player import RandomPlayer
from connect4.players.greedy_player import GreedyPlayer
from connect4.players.rl.alphazero.alphazero_player import AlphaZeroPlayer
from connect4.players.rl.ppo.ppo_player import PPOPlayer
from connect4.players.rl.dqn.dqn_player import DQNPlayer


def play_game(player1: BasePlayer, player2: BasePlayer) -> int | None:
    """Play a single game silently. Returns winner (P1/P2) or None for draw."""
    board = Board()
    players = {P1: player1, P2: player2}
    current = P1

    while True:
        action = players[current].choose_move(board)
        board.drop_piece(action, current)

        winner = board.check_winner()
        if winner is not None:
            return winner
        if board.is_draw():
            return None

        current = P2 if current == P1 else P1


def pit(player1: BasePlayer, player2: BasePlayer, num_games: int = 100) -> dict:
    """Play num_games (half as P1, half as P2). Returns win rates."""
    half = num_games // 2
    p1_wins = 0
    p2_wins = 0
    draws = 0

    # player1 goes first
    for _ in range(half):
        result = play_game(player1, player2)
        if result == P1:
            p1_wins += 1
        elif result == P2:
            p2_wins += 1
        else:
            draws += 1

    # player2 goes first
    for _ in range(num_games - half):
        result = play_game(player2, player1)
        if result == P1:
            p2_wins += 1  # player2 was P1
        elif result == P2:
            p1_wins += 1  # player1 was P2
        else:
            draws += 1

    return {
        "p1_name": player1.name,
        "p2_name": player2.name,
        "p1_wins": p1_wins,
        "p2_wins": p2_wins,
        "draws": draws,
        "p1_win_rate": p1_wins / num_games,
        "p2_win_rate": p2_wins / num_games,
    }


def round_robin(players: list[BasePlayer], num_games: int = 100):
    """All-vs-all tournament. Prints results table."""
    names = [p.name for p in players]
    n = len(players)

    # Win rate matrix
    matrix = [[None] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            result = pit(players[i], players[j], num_games)
            matrix[i][j] = result["p1_win_rate"]
            matrix[j][i] = result["p2_win_rate"]
            print(f"{names[i]} vs {names[j]}: "
                  f"{result['p1_wins']}W / {result['draws']}D / {result['p2_wins']}L "
                  f"({result['p1_win_rate']:.1%})")

    # Summary
    print(f"\n{'Player':<25} {'Avg Win Rate':>12}")
    print("-" * 38)
    for i in range(n):
        rates = [matrix[i][j] for j in range(n) if matrix[i][j] is not None]
        avg = sum(rates) / len(rates) if rates else 0.0
        print(f"{names[i]:<25} {avg:>11.1%}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Connect 4 agents")
    parser.add_argument("--num-games", type=int, default=100, help="Games per matchup")
    parser.add_argument("--alphazero-ckpt", default="checkpoints/alphazero/best.pt")
    parser.add_argument("--ppo-ckpt", default="checkpoints/ppo/best.pt")
    parser.add_argument("--dqn-ckpt", default="checkpoints/dqn/best.pt")
    args = parser.parse_args()

    players = [
        RandomPlayer(),
        GreedyPlayer(P1),
        AlphaZeroPlayer(P1, args.alphazero_ckpt),
        PPOPlayer(P1, args.ppo_ckpt),
        DQNPlayer(P1, args.dqn_ckpt),
    ]

    print(f"Round-robin tournament ({args.num_games} games per matchup)\n")
    round_robin(players, args.num_games)


if __name__ == "__main__":
    main()
