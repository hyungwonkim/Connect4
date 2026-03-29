from connect4.board import Board, P1, P2


class Game:
    def __init__(self, player1, player2):
        self.board = Board()
        self.players = {P1: player1, P2: player2}
        self.current = P1

    def run(self):
        symbols = {P1: "X", P2: "O"}
        while True:
            print(f"\n{self.board}\n")
            player = self.players[self.current]
            col = player.choose_move(self.board)
            self.board.drop_piece(col, self.current)

            winner = self.board.check_winner()
            if winner:
                print(f"\n{self.board}\n")
                print(f"{player.name} ({symbols[winner]}) wins!")
                return

            if self.board.is_draw():
                print(f"\n{self.board}\n")
                print("It's a draw!")
                return

            self.current = P2 if self.current == P1 else P1
