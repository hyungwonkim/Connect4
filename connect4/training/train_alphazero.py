"""AlphaZero training: self-play with MCTS → train policy+value → evaluate."""

import os
import random
from collections import deque

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from connect4.board import Board, P1, P2
from connect4.players.rl.common import board_to_tensor, get_device
from connect4.players.rl.networks import AlphaZeroNet
from connect4.players.rl.alphazero.mcts import MCTS


def self_play_game(network, device, num_simulations=200, temp_threshold=15):
    """Play one game of self-play, collecting training data.

    Returns list of (state_tensor, mcts_policy, result_from_current_player).
    """
    mcts = MCTS(network, num_simulations)
    board = Board()
    current_player = P1
    history = []  # (state_tensor, mcts_policy, current_player)
    move_count = 0

    while True:
        state = board_to_tensor(board, current_player)
        visits = mcts.search(board, current_player)

        # Temperature: explore early, exploit later
        if move_count < temp_threshold:
            # Sample proportional to visit counts
            total = sum(visits)
            if total > 0:
                probs = [v / total for v in visits]
                action = random.choices(range(7), weights=probs, k=1)[0]
            else:
                action = random.choice(board.get_valid_moves())
        else:
            action = max(range(7), key=lambda c: visits[c])

        history.append((state, visits, current_player))
        board.drop_piece(action, current_player)
        move_count += 1

        winner = board.check_winner()
        if winner is not None:
            # Assign results: +1 for winner, -1 for loser
            examples = []
            for s, pi, player in history:
                result = 1.0 if player == winner else -1.0
                examples.append((s, torch.tensor(pi, dtype=torch.float32), result))
            return examples

        if board.is_draw():
            examples = []
            for s, pi, player in history:
                examples.append((s, torch.tensor(pi, dtype=torch.float32), 0.0))
            return examples

        current_player = P2 if current_player == P1 else P1


def train(
    num_iterations: int = 100,
    games_per_iteration: int = 100,
    num_simulations: int = 200,
    batch_size: int = 64,
    lr: float = 0.001,
    buffer_size: int = 50000,
    checkpoint_dir: str = "checkpoints/alphazero",
):
    device = get_device()
    network = AlphaZeroNet().to(device)
    optimizer = optim.Adam(network.parameters(), lr=lr, weight_decay=1e-4)
    replay_buffer = deque(maxlen=buffer_size)

    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join("runs", "alphazero"))

    for iteration in range(1, num_iterations + 1):
        print(f"\n=== Iteration {iteration}/{num_iterations} ===")

        # Self-play
        network.eval()
        new_examples = []
        game_results = []  # track outcomes
        for g in range(games_per_iteration):
            examples = self_play_game(network, device, num_simulations)
            new_examples.extend(examples)
            # Determine outcome from final example's result value
            if examples:
                final_result = examples[-1][2]
                if final_result > 0:
                    game_results.append("p1_win")
                elif final_result < 0:
                    game_results.append("p2_win")
                else:
                    game_results.append("draw")
            if (g + 1) % 10 == 0:
                print(f"  Self-play: {g + 1}/{games_per_iteration} games")

        replay_buffer.extend(new_examples)
        print(f"  Buffer size: {len(replay_buffer)}")

        # Log game outcome distribution
        n_games = len(game_results)
        if n_games > 0:
            p1_wins = sum(1 for r in game_results if r == "p1_win")
            p2_wins = sum(1 for r in game_results if r == "p2_win")
            draws = sum(1 for r in game_results if r == "draw")
            writer.add_scalar("outcome/p1_win_rate", p1_wins / n_games, iteration)
            writer.add_scalar("outcome/p2_win_rate", p2_wins / n_games, iteration)
            writer.add_scalar("outcome/draw_rate", draws / n_games, iteration)
        writer.add_scalar("training/buffer_size", len(replay_buffer), iteration)
        writer.add_scalar("training/examples_per_iter", len(new_examples), iteration)

        # Training
        network.train()
        if len(replay_buffer) < batch_size:
            continue

        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = max(1, len(new_examples) // batch_size)

        for _ in range(num_batches):
            batch = random.sample(list(replay_buffer), min(batch_size, len(replay_buffer)))
            states = torch.stack([s for s, _, _ in batch]).to(device)
            target_pis = torch.stack([pi for _, pi, _ in batch]).to(device)
            target_vs = torch.tensor([v for _, _, v in batch], dtype=torch.float32).to(device)

            log_policy, value = network(states)
            value = value.squeeze(-1)

            policy_loss = -torch.mean(torch.sum(target_pis * log_policy, dim=1))
            value_loss = F.mse_loss(value, target_vs)
            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()

        avg_policy_loss = total_policy_loss / num_batches
        avg_value_loss = total_value_loss / num_batches
        writer.add_scalar("training/policy_loss", avg_policy_loss, iteration)
        writer.add_scalar("training/value_loss", avg_value_loss, iteration)
        writer.add_scalar("training/total_loss", avg_policy_loss + avg_value_loss, iteration)

        print(f"  Policy loss: {avg_policy_loss:.4f}")
        print(f"  Value loss: {avg_value_loss:.4f}")

        # Save checkpoint
        if iteration % 10 == 0 or iteration == num_iterations:
            path = os.path.join(checkpoint_dir, f"iter_{iteration}.pt")
            torch.save(network.state_dict(), path)
            print(f"  Saved checkpoint: {path}")

            best_path = os.path.join(checkpoint_dir, "best.pt")
            torch.save(network.state_dict(), best_path)
            print(f"  Saved best: {best_path}")

    writer.close()
    print("\nTraining complete.")


if __name__ == "__main__":
    train()
