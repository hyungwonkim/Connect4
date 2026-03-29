from connect4.board import Board
from connect4.players.base import BasePlayer


class HumanPlayer(BasePlayer):
    def __init__(self, name: str = "Human"):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def choose_move(self, board: Board) -> int:
        while True:
            try:
                col = int(input(f"{self.name}, choose a column (0-6): "))
            except ValueError:
                print("Please enter a number between 0 and 6.")
                continue
            if board.is_valid_move(col):
                return col
            print(f"Column {col} is not a valid move. Try again.")
