import argparse

from connect4.board import P2
from connect4.game import Game
from connect4.players.human import HumanPlayer
from connect4.players.random_player import RandomPlayer
from connect4.players.greedy_player import GreedyPlayer
from connect4.players.epsilon_greedy_player import EpsilonGreedyPlayer
from connect4.players.rl.alphazero.alphazero_player import AlphaZeroPlayer
from connect4.players.rl.ppo.ppo_player import PPOPlayer
from connect4.players.rl.dqn.dqn_player import DQNPlayer


def main():
    parser = argparse.ArgumentParser(description="Connect 4")
    parser.add_argument(
        "--player2",
        choices=["human", "random", "greedy", "epsilon_greedy",
                 "alphazero", "ppo", "dqn"],
        default="human",
        help="Type of player 2 (default: human)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (for RL players)",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch Pygame GUI instead of CLI",
    )
    args = parser.parse_args()

    if args.gui:
        from connect4.gui.pygame_gui import PygameGame
        PygameGame().run()
        return

    player1 = HumanPlayer("Player 1")

    player2_map = {
        "human": lambda: HumanPlayer("Player 2"),
        "random": lambda: RandomPlayer(),
        "greedy": lambda: GreedyPlayer(P2),
        "epsilon_greedy": lambda: EpsilonGreedyPlayer(P2),
        "alphazero": lambda: AlphaZeroPlayer(P2, args.checkpoint or "checkpoints/alphazero/best.pt"),
        "ppo": lambda: PPOPlayer(P2, args.checkpoint or "checkpoints/ppo/best.pt"),
        "dqn": lambda: DQNPlayer(P2, args.checkpoint or "checkpoints/dqn/best.pt"),
    }
    player2 = player2_map[args.player2]()

    game = Game(player1, player2)
    game.run()


if __name__ == "__main__":
    main()
