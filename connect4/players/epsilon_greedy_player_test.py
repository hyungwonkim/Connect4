from connect4.board import Board, P1, P2
from connect4.players.greedy_player import GreedyPlayer
from connect4.players.epsilon_greedy_player import EpsilonGreedyPlayer


class TestEpsilonGreedyMatchesGreedy:
    """With epsilon=0, EpsilonGreedyPlayer should behave identically to GreedyPlayer."""

    def _make_players(self):
        greedy = GreedyPlayer(P2)
        eps_greedy = EpsilonGreedyPlayer(P2, epsilon=0.0)
        return greedy, eps_greedy

    def test_takes_immediate_win(self):
        """Both players should take an immediate winning move."""
        b = Board()
        for c in [0, 1, 2]:
            b.drop_piece(c, P2)
        greedy, eps_greedy = self._make_players()
        assert greedy.choose_move(b) == 3
        assert eps_greedy.choose_move(b) == 3

    def test_blocks_opponent_win(self):
        """Both players should block the opponent's immediate win."""
        b = Board()
        for c in [0, 1, 2]:
            b.drop_piece(c, P1)
        greedy, eps_greedy = self._make_players()
        assert greedy.choose_move(b) == 3
        assert eps_greedy.choose_move(b) == 3

    def test_prioritizes_own_win_over_block(self):
        """If both own win and opponent win exist, take the win."""
        b = Board()
        # P2 three in a row on bottom
        for c in [0, 1, 2]:
            b.drop_piece(c, P2)
        # P1 vertical threat on col 6
        for _ in range(3):
            b.drop_piece(6, P1)
        # P2 can win at col 3, P1 can win at col 6 (vertical)
        # Both players (as P2) should take col 3 (own win)
        greedy, eps_greedy = self._make_players()
        assert greedy.choose_move(b) == 3
        assert eps_greedy.choose_move(b) == 3


class TestEpsilonGreedyExploration:
    """With high epsilon, EpsilonGreedyPlayer should sometimes deviate from greedy."""

    def test_high_epsilon_deviates(self):
        """With epsilon=0.99, at least one of 10 runs should differ from greedy."""
        b = Board()
        # Set up a clear greedy preference: P2 has two in a row, col 2 extends to 3
        b.drop_piece(0, P2)
        b.drop_piece(1, P2)
        # No immediate wins or blocks, so greedy always picks col 2
        # (extends horizontal to 3, best among all moves)
        greedy = GreedyPlayer(P2)
        greedy_move = greedy.choose_move(b)

        eps_greedy = EpsilonGreedyPlayer(P2, epsilon=0.99)
        moves = [eps_greedy.choose_move(b) for _ in range(10)]
        # With epsilon=0.99, ~99% chance each move is random (7 columns),
        # so almost certainly at least one will differ from the greedy choice
        assert any(m != greedy_move for m in moves), (
            f"Expected at least one deviation from greedy move {greedy_move}, "
            f"but all 10 moves were: {moves}"
        )
