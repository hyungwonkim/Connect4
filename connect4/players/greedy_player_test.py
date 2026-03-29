from connect4.board import Board, P1, P2, EMPTY
from connect4.players.greedy_player import GreedyPlayer


class TestLongestSequence:
    def test_single_piece(self):
        b = Board()
        b.drop_piece(3, P1)
        # Single piece at (5, 3) -> sequence of 1
        assert GreedyPlayer._longest_sequence(b.grid, 5, 3, P1) == 1

    def test_horizontal_two(self):
        b = Board()
        b.drop_piece(2, P1)
        b.drop_piece(3, P1)
        # (5,3) is adjacent to (5,2) -> horizontal run of 2
        assert GreedyPlayer._longest_sequence(b.grid, 5, 3, P1) == 2

    def test_horizontal_three(self):
        b = Board()
        for c in [1, 2, 3]:
            b.drop_piece(c, P1)
        # Middle piece at (5,2) connects both sides -> 3
        assert GreedyPlayer._longest_sequence(b.grid, 5, 2, P1) == 3

    def test_vertical_sequence(self):
        b = Board()
        for _ in range(3):
            b.drop_piece(0, P1)
        # Top of the stack is at row 3 (dropped 3 pieces: rows 5, 4, 3)
        assert GreedyPlayer._longest_sequence(b.grid, 3, 0, P1) == 3

    def test_diagonal_sequence(self):
        """Diagonal down-right: pieces at (5,0), (4,1), (3,2)."""
        b = Board()
        b.drop_piece(0, P1)       # (5, 0)
        b.drop_piece(1, P2)       # (5, 1) filler
        b.drop_piece(1, P1)       # (4, 1)
        b.drop_piece(2, P2)       # (5, 2) filler
        b.drop_piece(2, P2)       # (4, 2) filler
        b.drop_piece(2, P1)       # (3, 2)
        # Diagonal through (4,1) -> length 3
        assert GreedyPlayer._longest_sequence(b.grid, 4, 1, P1) == 3

    def test_broken_by_opponent(self):
        """Opponent piece in the middle breaks the sequence."""
        b = Board()
        b.drop_piece(0, P1)       # (5, 0)
        b.drop_piece(1, P2)       # (5, 1) opponent breaks horizontal
        b.drop_piece(2, P1)       # (5, 2)
        # From (5,0), horizontal is blocked by P2 at (5,1) -> only 1
        assert GreedyPlayer._longest_sequence(b.grid, 5, 0, P1) == 1

    def test_ignores_other_player(self):
        """Only counts the specified player's pieces."""
        b = Board()
        for c in range(4):
            b.drop_piece(c, P2)
        # Asking for P1's sequence at a P2 cell still checks for P1 neighbors (none)
        assert GreedyPlayer._longest_sequence(b.grid, 5, 0, P1) == 1

    def test_counts_from_edge(self):
        b = Board()
        b.drop_piece(0, P1)
        b.drop_piece(1, P1)
        # From the edge piece at (5,0), horizontal extends right -> 2
        assert GreedyPlayer._longest_sequence(b.grid, 5, 0, P1) == 2
