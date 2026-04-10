"""Core matchup logic: play N games with a fixed first player, build the NxN matrix."""

from __future__ import annotations

from connect4.board import P1, P2
from connect4.training.evaluate import play_game
from connect4.leaderboard.agents import PlayerFactory


def play_matchup(
    first_factory: PlayerFactory,
    second_factory: PlayerFactory,
    num_games: int = 50,
) -> tuple[int, int, int]:
    """Play *num_games* with *first_factory* as P1 and *second_factory* as P2.

    Returns ``(first_wins, second_wins, draws)``.
    """
    p1 = first_factory(P1)
    p2 = second_factory(P2)

    first_wins = 0
    second_wins = 0
    draws = 0

    for _ in range(num_games):
        result = play_game(p1, p2)
        if result == P1:
            first_wins += 1
        elif result == P2:
            second_wins += 1
        else:
            draws += 1

    return first_wins, second_wins, draws


def build_matrix(
    agents: dict[str, PlayerFactory],
    num_games: int = 50,
    progress: bool = True,
) -> tuple[list[str], list[list[float | None]]]:
    """Build the NxN win-rate matrix.

    ``matrix[i][j]`` = win rate of agent *i* going **first** against agent *j*.
    Diagonal entries are ``None``.

    Returns ``(agent_names, matrix)``.
    """
    names = list(agents.keys())
    factories = list(agents.values())
    n = len(names)
    matrix: list[list[float | None]] = [[None] * n for _ in range(n)]

    total = n * (n - 1)
    done = 0

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            first_wins, second_wins, draws = play_matchup(
                factories[i], factories[j], num_games,
            )
            matrix[i][j] = first_wins / num_games
            done += 1
            if progress:
                print(
                    f"[{done}/{total}] {names[i]} (1st) vs {names[j]}: "
                    f"{first_wins}W / {draws}D / {second_wins}L "
                    f"({first_wins / num_games:.0%})"
                )

    return names, matrix


def print_matrix(names: list[str], matrix: list[list[float | None]]) -> None:
    """Pretty-print the win-rate matrix to the console."""
    n = len(names)
    col_w = max(len(name) for name in names) + 2

    # Header
    header = " " * col_w + "".join(name.rjust(col_w) for name in names)
    print(header)
    print("-" * len(header))

    for i in range(n):
        row = names[i].ljust(col_w)
        for j in range(n):
            if matrix[i][j] is None:
                row += "--".rjust(col_w)
            else:
                row += f"{matrix[i][j]:.0%}".rjust(col_w)
        print(row)
