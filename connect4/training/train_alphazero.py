"""AlphaZero training: self-play with MCTS → train policy+value → evaluate."""

import copy
import os
import random
from collections import deque

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from connect4.board import Board, P1, P2
from connect4.players.greedy_player import GreedyPlayer
from connect4.players.random_player import RandomPlayer
from connect4.players.rl.common import board_to_tensor, get_device
from connect4.players.rl.networks import AlphaZeroNetV2
from connect4.players.rl.alphazero.mcts import MCTS


def _augment(state: torch.Tensor, pi: list[float]):
    """Return [(state, pi), (flipped_state, flipped_pi)] for horizontal symmetry."""
    flipped_state = torch.flip(state, dims=[-1])
    flipped_pi = list(reversed(pi))
    return [(state, pi), (flipped_state, flipped_pi)]


def _finish_game(history, winner):
    """Convert history into training examples with results (with symmetry aug)."""
    examples = []
    for s, pi, player in history:
        if winner is None:
            result = 0.0
        else:
            result = 1.0 if player == winner else -1.0
        for aug_s, aug_pi in _augment(s, pi):
            examples.append(
                (aug_s, torch.tensor(aug_pi, dtype=torch.float32), result)
            )
    return examples


def self_play_game(network, device, num_simulations=400, temp_threshold=15):
    """Play one game of self-play, collecting training data.

    Returns (examples, winner).
    """
    mcts = MCTS(network, num_simulations)
    board = Board()
    current_player = P1
    history = []
    move_count = 0

    while True:
        state = board_to_tensor(board, current_player)
        visits = mcts.search(board, current_player, add_noise=True)

        if move_count < temp_threshold:
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
            return _finish_game(history, winner), winner
        if board.is_draw():
            return _finish_game(history, None), None
        current_player = P2 if current_player == P1 else P1


def play_vs_opponent(network, opponent_net, device, num_simulations=400, temp_threshold=15):
    """Play one game: current network vs a past version.

    Each side uses its own MCTS. Training data collected from current network's
    perspective only. Current network randomly assigned P1 or P2.

    Returns (examples, winner).
    """
    current_mcts = MCTS(network, num_simulations)
    opponent_mcts = MCTS(opponent_net, num_simulations)
    board = Board()
    current_player = P1

    current_side = random.choice([P1, P2])
    history = []
    move_count = 0

    while True:
        if current_player == current_side:
            state = board_to_tensor(board, current_player)
            visits = current_mcts.search(board, current_player, add_noise=True)

            if move_count < temp_threshold:
                total = sum(visits)
                if total > 0:
                    probs = [v / total for v in visits]
                    action = random.choices(range(7), weights=probs, k=1)[0]
                else:
                    action = random.choice(board.get_valid_moves())
            else:
                action = max(range(7), key=lambda c: visits[c])

            history.append((state, visits, current_player))
        else:
            visits = opponent_mcts.search(board, current_player, add_noise=False)
            action = max(range(7), key=lambda c: visits[c])

        board.drop_piece(action, current_player)
        move_count += 1

        winner = board.check_winner()
        if winner is not None:
            return _finish_game(history, winner), winner
        if board.is_draw():
            return _finish_game(history, None), None
        current_player = P2 if current_player == P1 else P1


def self_play_with_opening(network, device, num_simulations=400,
                           temp_threshold=15, opening_moves=5):
    """Self-play game starting from a diverse opening position.

    First few moves are played by random/greedy, then self-play with MCTS.
    Training data collected only from the MCTS self-play portion.

    Returns (examples, winner).
    """
    board = Board()
    current_player = P1

    # Play opening moves with random or greedy
    opener_type = random.choice(["random", "greedy"])
    if opener_type == "greedy":
        opener_p1 = GreedyPlayer(P1)
        opener_p2 = GreedyPlayer(P2)
    else:
        opener_p1 = RandomPlayer()
        opener_p2 = RandomPlayer()

    num_opening = random.randint(2, opening_moves)
    for _ in range(num_opening):
        opener = opener_p1 if current_player == P1 else opener_p2
        action = opener.choose_move(board)
        board.drop_piece(action, current_player)

        if board.check_winner() is not None or board.is_draw():
            return [], board.check_winner()

        current_player = P2 if current_player == P1 else P1

    # Now self-play with MCTS from this position
    mcts = MCTS(network, num_simulations)
    history = []
    move_count = num_opening

    while True:
        state = board_to_tensor(board, current_player)
        visits = mcts.search(board, current_player, add_noise=True)

        if move_count < temp_threshold:
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
            return _finish_game(history, winner), winner
        if board.is_draw():
            return _finish_game(history, None), None
        current_player = P2 if current_player == P1 else P1


def play_vs_greedy_training(network, device, num_simulations=400, temp_threshold=15):
    """AZ plays vs Greedy, collecting training data from AZ's moves only.

    Returns (examples, winner). Greedy teacher.
    """
    mcts = MCTS(network, num_simulations)
    board = Board()
    current_player = P1
    az_side = random.choice([P1, P2])
    greedy = GreedyPlayer(P2 if az_side == P1 else P1)
    history = []
    move_count = 0

    while True:
        if current_player == az_side:
            state = board_to_tensor(board, current_player)
            visits = mcts.search(board, current_player, add_noise=True)
            if move_count < temp_threshold:
                total = sum(visits)
                if total > 0:
                    probs = [v / total for v in visits]
                    action = random.choices(range(7), weights=probs, k=1)[0]
                else:
                    action = random.choice(board.get_valid_moves())
            else:
                action = max(range(7), key=lambda c: visits[c])
            history.append((state, visits, current_player))
        else:
            action = greedy.choose_move(board)

        board.drop_piece(action, current_player)
        move_count += 1

        winner = board.check_winner()
        if winner is not None:
            return _finish_game(history, winner), winner
        if board.is_draw():
            return _finish_game(history, None), None
        current_player = P2 if current_player == P1 else P1


def play_vs_greedy_eval(network, device, num_simulations=400):
    """Play one evaluation game vs GreedyPlayer (no training data, no noise).

    Returns (winner, az_player).
    """
    mcts = MCTS(network, num_simulations)
    board = Board()
    current_player = P1

    az_player = random.choice([P1, P2])
    greedy = GreedyPlayer(P2 if az_player == P1 else P1)

    while True:
        if current_player == az_player:
            visits = mcts.search(board, current_player, add_noise=False)
            action = max(range(7), key=lambda c: visits[c])
        else:
            action = greedy.choose_move(board)

        board.drop_piece(action, current_player)

        winner = board.check_winner()
        if winner is not None:
            return winner, az_player
        if board.is_draw():
            return None, az_player
        current_player = P2 if current_player == P1 else P1


def train(
    num_iterations: int = 500,
    games_per_iteration: int = 100,
    num_simulations: int = 400,
    batch_size: int = 64,
    lr: float = 0.002,
    buffer_size: int = 150_000,
    checkpoint_dir: str = "checkpoints/alphazero",
    tb_subdir: str = "alphazero",
    resume_from: str = None,
    greedy_eval_games: int = 20,
    opponent_pool_frac: float = 0.2,
    diverse_opening_frac: float = 0.1,
    greedy_teacher_frac: float = 0.1,
    pool_update_freq: int = 10,
    warmup_iters: int = 50,
    net_channels: int = 96,
    net_blocks: int = 5,
):
    device = get_device()
    network = AlphaZeroNetV2(channels=net_channels, num_blocks=net_blocks).to(device)
    opponent_net = AlphaZeroNetV2(channels=net_channels, num_blocks=net_blocks).to(device)
    num_params = sum(p.numel() for p in network.parameters())
    print(f"Network params: {num_params:,}")

    optimizer = optim.Adam(network.parameters(), lr=lr, weight_decay=1e-4)

    # LR: flat for warmup_iters, then cosine over the remainder.
    warmup = optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=warmup_iters)
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, num_iterations - warmup_iters)
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_iters]
    )

    replay_buffer = deque(maxlen=buffer_size)
    opponent_pool = []

    start_iteration = 1
    if resume_from and os.path.exists(resume_from):
        checkpoint = torch.load(resume_from, map_location=device)
        network.load_state_dict(checkpoint)
        basename = os.path.basename(resume_from)
        if basename.startswith("iter_") and basename.endswith(".pt"):
            start_iteration = int(basename[5:-3]) + 1
        print(f"Resumed from {resume_from} (starting at iteration {start_iteration})")

    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join("runs", tb_subdir))

    end_iteration = start_iteration + num_iterations - 1
    for iteration in range(start_iteration, end_iteration + 1):
        print(f"\n=== Iteration {iteration}/{end_iteration} ===")

        if iteration % pool_update_freq == 0 or iteration == start_iteration:
            opponent_pool.append(copy.deepcopy(network.state_dict()))
            if len(opponent_pool) > 10:
                opponent_pool.pop(0)
            print(f"  Opponent pool size: {len(opponent_pool)}")

        num_opponent = int(games_per_iteration * opponent_pool_frac) if opponent_pool else 0
        num_opening = int(games_per_iteration * diverse_opening_frac)
        num_greedy_teacher = int(games_per_iteration * greedy_teacher_frac)
        num_selfplay = games_per_iteration - num_opponent - num_opening - num_greedy_teacher

        network.eval()
        new_examples = []
        game_results = []

        # Regular self-play
        for g in range(num_selfplay):
            examples, winner = self_play_game(network, device, num_simulations)
            new_examples.extend(examples)
            game_results.append(
                "p1_win" if winner == P1 else "p2_win" if winner == P2 else "draw"
            )
            if (g + 1) % 10 == 0:
                print(f"  Self-play: {g + 1}/{num_selfplay} games")

        # Games vs past versions (opponent pool)
        if num_opponent > 0:
            for g in range(num_opponent):
                opp_state = random.choice(opponent_pool)
                opponent_net.load_state_dict(opp_state)
                opponent_net.eval()
                examples, winner = play_vs_opponent(
                    network, opponent_net, device, num_simulations
                )
                new_examples.extend(examples)
                game_results.append(
                    "p1_win" if winner == P1 else "p2_win" if winner == P2 else "draw"
                )
                if (g + 1) % 10 == 0:
                    print(f"  Vs past: {g + 1}/{num_opponent} games")

        # Self-play with diverse openings
        for g in range(num_opening):
            examples, winner = self_play_with_opening(
                network, device, num_simulations
            )
            new_examples.extend(examples)
            game_results.append(
                "p1_win" if winner == P1 else "p2_win" if winner == P2 else "draw"
            )
            if (g + 1) % 10 == 0:
                print(f"  Diverse opening: {g + 1}/{num_opening} games")

        # Greedy teacher games (training data from AZ's moves)
        for g in range(num_greedy_teacher):
            examples, winner = play_vs_greedy_training(
                network, device, num_simulations
            )
            new_examples.extend(examples)
            game_results.append(
                "p1_win" if winner == P1 else "p2_win" if winner == P2 else "draw"
            )
            if (g + 1) % 10 == 0:
                print(f"  Greedy teacher: {g + 1}/{num_greedy_teacher} games")

        replay_buffer.extend(new_examples)
        print(f"  Buffer size: {len(replay_buffer)}")

        # Evaluation vs Greedy (no training data, no noise)
        greedy_results = []
        for g in range(greedy_eval_games):
            winner, az_player = play_vs_greedy_eval(network, device, num_simulations)
            if winner == az_player:
                greedy_results.append("az_win")
            elif winner is None:
                greedy_results.append("draw")
            else:
                greedy_results.append("az_loss")

        az_wins = sum(1 for r in greedy_results if r == "az_win")
        az_draws = sum(1 for r in greedy_results if r == "draw")
        writer.add_scalar("outcome/vs_greedy_win_rate", az_wins / len(greedy_results), iteration)
        writer.add_scalar("outcome/vs_greedy_draw_rate", az_draws / len(greedy_results), iteration)
        print(f"  Vs Greedy: {az_wins}/{len(greedy_results)} wins, {az_draws} draws")

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
            scheduler.step()
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

        scheduler.step()
        writer.add_scalar("training/lr", scheduler.get_last_lr()[0], iteration)

        # Save checkpoint
        if iteration % 10 == 0 or iteration == end_iteration:
            path = os.path.join(checkpoint_dir, f"iter_{iteration}.pt")
            torch.save(network.state_dict(), path)
            print(f"  Saved checkpoint: {path}")

            best_path = os.path.join(checkpoint_dir, "best.pt")
            torch.save(network.state_dict(), best_path)
            print(f"  Saved best: {best_path}")

    writer.close()
    print("\nTraining complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AlphaZero training")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--num-iterations", type=int, default=500)
    parser.add_argument("--num-simulations", type=int, default=400)
    parser.add_argument("--games-per-iteration", type=int, default=100)
    args = parser.parse_args()

    train(
        num_iterations=args.num_iterations,
        num_simulations=args.num_simulations,
        games_per_iteration=args.games_per_iteration,
        resume_from=args.resume_from,
    )
