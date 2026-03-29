from connect4.board import Board, P1, P2


class TestCheckWinner:
    def test_horizontal_win_p1(self):
        b = Board()
        for c in range(4):
            b.drop_piece(c, P1)
        assert b.check_winner() == P1

    def test_horizontal_win_p2(self):
        b = Board()
        for c in range(3, 7):
            b.drop_piece(c, P2)
        assert b.check_winner() == P2

    def test_vertical_win(self):
        b = Board()
        for _ in range(4):
            b.drop_piece(0, P1)
        assert b.check_winner() == P1

    def test_diagonal_down_right_win(self):
        """Diagonal from top-left to bottom-right."""
        b = Board()
        # Build a staircase so P1 gets a diagonal:
        # col 0: P1
        # col 1: P2, P1
        # col 2: P2, P2, P1
        # col 3: P2, P2, P2, P1
        b.drop_piece(0, P1)
        b.drop_piece(1, P2)
        b.drop_piece(1, P1)
        b.drop_piece(2, P2)
        b.drop_piece(2, P2)
        b.drop_piece(2, P1)
        b.drop_piece(3, P2)
        b.drop_piece(3, P2)
        b.drop_piece(3, P2)
        b.drop_piece(3, P1)
        assert b.check_winner() == P1

    def test_diagonal_down_left_win(self):
        """Diagonal from top-right to bottom-left."""
        b = Board()
        # Mirror staircase: col 6, 5, 4, 3
        b.drop_piece(6, P1)
        b.drop_piece(5, P2)
        b.drop_piece(5, P1)
        b.drop_piece(4, P2)
        b.drop_piece(4, P2)
        b.drop_piece(4, P1)
        b.drop_piece(3, P2)
        b.drop_piece(3, P2)
        b.drop_piece(3, P2)
        b.drop_piece(3, P1)
        assert b.check_winner() == P1

    def test_no_winner_on_empty_board(self):
        b = Board()
        assert b.check_winner() is None

    def test_no_winner_with_three_in_a_row(self):
        b = Board()
        for c in range(3):
            b.drop_piece(c, P1)
        assert b.check_winner() is None

    def test_no_winner_mixed_row(self):
        """Four in a row broken by opponent piece."""
        b = Board()
        b.drop_piece(0, P1)
        b.drop_piece(1, P1)
        b.drop_piece(2, P2)  # breaks the run
        b.drop_piece(3, P1)
        b.drop_piece(4, P1)
        assert b.check_winner() is None

    def test_win_not_attributed_to_wrong_player(self):
        b = Board()
        for c in range(4):
            b.drop_piece(c, P1)
        assert b.check_winner() != P2
