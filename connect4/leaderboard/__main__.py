"""CLI entry point: python -m connect4.leaderboard"""

import argparse

from connect4.leaderboard.agents import get_default_agents
from connect4.leaderboard.matchup import build_matrix, print_matrix


def main():
    parser = argparse.ArgumentParser(
        description="Connect 4 Leaderboard: NxN first-player win-rate matrix",
    )
    parser.add_argument(
        "--num-games", type=int, default=50,
        help="Games per directional matchup (default: 50)",
    )
    parser.add_argument("--dqn-ckpt", default="checkpoints/dqn/best.pt")
    parser.add_argument("--ppo-ckpt", default="checkpoints/ppo/best.pt")
    parser.add_argument("--alphazero-ckpt", default="checkpoints/alphazero/best.pt")
    args = parser.parse_args()

    agents = get_default_agents(
        dqn_ckpt=args.dqn_ckpt,
        ppo_ckpt=args.ppo_ckpt,
        alphazero_ckpt=args.alphazero_ckpt,
    )

    n = len(agents)
    total_games = n * (n - 1) * args.num_games
    print(f"Leaderboard: {n} agents, {args.num_games} games/matchup, {total_games} total games\n")

    names, matrix = build_matrix(agents, num_games=args.num_games)

    print("\n=== LEADERBOARD (row = first player, value = win rate) ===\n")
    print_matrix(names, matrix)


if __name__ == "__main__":
    main()
