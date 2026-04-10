"""PPO training.

Architecture & techniques:
  - Actor-critic with shared CNN backbone
  - Opponent pool (Random / Greedy / self-snapshots)
  - Entropy coefficient linearly decayed 0.01 -> 0.001
  - Reward shaping ON by default
  - Per-opponent win-rate + approx_kl diagnostic
  - GAE lambda=0.95, clip=0.2, ppo_epochs=4, advantage norm,
    gamma=1.0, gradient clip 0.5.
"""

from __future__ import annotations

import argparse
import os
import random
from collections import deque

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from connect4.board import Board, P1, P2
from connect4.players.rl.common import board_to_tensor, mask_invalid, get_device
from connect4.players.rl.networks import PPONet
from connect4.training.opponents import OpponentPool
from connect4.training.rewards import (
    shape_agent_move,
    shape_opponent_move,
)
R_WIN = 1.0
R_LOSS = -1.0
R_DRAW = 0.0


def compute_gae(transitions, gamma=1.0, lam=0.95):
    """Compute Generalized Advantage Estimation."""
    advantages = []
    returns = []
    gae = 0.0

    for i in reversed(range(len(transitions))):
        t = transitions[i]
        if t["done"]:
            next_value = 0.0
        elif i + 1 < len(transitions) and not transitions[i]["done"]:
            next_value = transitions[i + 1]["value"]
        else:
            next_value = 0.0

        delta = t["reward"] + gamma * next_value - t["value"]
        gae = delta + gamma * lam * gae * (0.0 if t["done"] else 1.0)
        advantages.insert(0, gae)
        returns.insert(0, gae + t["value"])

    return advantages, returns


def clean_eval_vs_greedy(network, device, num_games: int) -> float:
    """Deterministic eval (argmax policy) vs Greedy, both sides. Draws = 0.5."""
    from connect4.players.greedy_player import GreedyPlayer
    network.eval()
    wins = 0.0
    half = num_games // 2
    # PPO as P1
    for _ in range(half):
        board = Board()
        cur = P1
        greedy = GreedyPlayer(P2)
        while True:
            if cur == P1:
                state = board_to_tensor(board, P1)
                with torch.no_grad():
                    logits, _ = network(state.unsqueeze(0).to(device))
                logits = logits.squeeze(0).cpu()
                a = mask_invalid(logits, board).argmax().item()
                board.drop_piece(a, P1)
                if board.check_winner() == P1:
                    wins += 1.0; break
                if board.is_draw():
                    wins += 0.5; break
            else:
                a = greedy.choose_move(board)
                board.drop_piece(a, P2)
                if board.check_winner() == P2:
                    break
                if board.is_draw():
                    wins += 0.5; break
            cur = P2 if cur == P1 else P1
    # PPO as P2
    for _ in range(num_games - half):
        board = Board()
        cur = P1
        greedy = GreedyPlayer(P1)
        while True:
            if cur == P1:
                a = greedy.choose_move(board)
                board.drop_piece(a, P1)
                if board.check_winner() == P1:
                    break
                if board.is_draw():
                    wins += 0.5; break
            else:
                state = board_to_tensor(board, P2)
                with torch.no_grad():
                    logits, _ = network(state.unsqueeze(0).to(device))
                logits = logits.squeeze(0).cpu()
                a = mask_invalid(logits, board).argmax().item()
                board.drop_piece(a, P2)
                if board.check_winner() == P2:
                    wins += 1.0; break
                if board.is_draw():
                    wins += 0.5; break
            cur = P2 if cur == P1 else P1
    return wins / num_games


def collect_rollout(
    network,
    opponent_pool: OpponentPool,
    device,
    num_games: int,
    shape_rewards: bool,
):
    """Collect transitions (agent plays as P1) against opponents sampled from pool.

    Returns (transitions, outcomes, opp_tags).
    """
    transitions = []
    outcomes = []
    opp_tags = []

    for _ in range(num_games):
        opponent_fn, tag = opponent_pool.sample()
        board = Board()
        current_player = P1
        game_transitions = []

        while True:
            if current_player == P1:
                state = board_to_tensor(board, P1)
                state_batch = state.unsqueeze(0).to(device)
                with torch.no_grad():
                    logits, value = network(state_batch)
                logits = logits.squeeze(0).cpu()
                value = value.item()

                masked = mask_invalid(logits, board)
                dist = Categorical(logits=masked)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                board_before = board.copy()
                row_placed = board.drop_piece(action.item(), P1)

                step_reward = 0.0
                if shape_rewards:
                    step_reward = shape_agent_move(
                        board_before, board, action.item(), row_placed, P1,
                    )

                game_transitions.append({
                    "state": state,
                    "action": action.item(),
                    "log_prob": log_prob.item(),
                    "value": value,
                    "reward": step_reward,
                    "done": False,
                })
            else:
                board_before = board.copy()
                action = opponent_fn(board, P2)
                row_placed = board.drop_piece(action, P2)
                if shape_rewards and game_transitions:
                    game_transitions[-1]["reward"] += shape_opponent_move(
                        board_before, board, action, row_placed, P2,
                    )

            # Terminal checks
            winner = board.check_winner()
            if winner is not None:
                if game_transitions:
                    if winner == P1:
                        game_transitions[-1]["reward"] += R_WIN
                        outcomes.append("win")
                    else:
                        game_transitions[-1]["reward"] += R_LOSS
                        outcomes.append("loss")
                    game_transitions[-1]["done"] = True
                    opp_tags.append(tag)
                break

            if board.is_draw():
                if game_transitions:
                    game_transitions[-1]["reward"] += R_DRAW
                    game_transitions[-1]["done"] = True
                    outcomes.append("draw")
                    opp_tags.append(tag)
                break

            current_player = P2 if current_player == P1 else P1

        transitions.extend(game_transitions)

    return transitions, outcomes, opp_tags


def train(
    num_iterations: int = 1000,
    games_per_iter: int = 100,
    clip_eps: float = 0.2,
    ppo_epochs: int = 4,
    batch_size: int = 64,
    lr: float = 1e-4,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
    entropy_start: float = 0.01,
    entropy_end: float = 0.001,
    value_coeff: float = 1.0,
    pool_update_freq: int = 50,
    max_snapshots: int = 9,
    shape_rewards: bool = True,
    eval_every_iters: int = 50,
    eval_games: int = 40,
    checkpoint_dir: str = "checkpoints/ppo",
    log_dir: str = "runs/ppo",
):
    device = get_device()
    network = PPONet().to(device)
    optimizer = optim.Adam(network.parameters(), lr=lr)

    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    opponent_pool = OpponentPool(
        device=device,
        is_policy=True,  # PPONet returns (logits, value)
        weights=(0.10, 0.60, 0.30),  # Random / Greedy / snapshots
        opponent_player_id=P2,
        max_snapshots=max_snapshots,
    )

    print(f"Training PPO on {device} | shape_rewards={shape_rewards}")
    print(f"  iterations={num_iterations}, games/iter={games_per_iter}, lr={lr}")

    best_eval_wr = -1.0

    for iteration in range(1, num_iterations + 1):
        # Entropy coefficient linear decay
        progress = (iteration - 1) / max(num_iterations - 1, 1)
        entropy_coeff = entropy_start + (entropy_end - entropy_start) * progress

        network.eval()
        transitions, outcomes, opp_tags = collect_rollout(
            network, opponent_pool, device, games_per_iter, shape_rewards,
        )

        if not transitions:
            continue

        # Per-opponent win rates
        per_opp = {"random": [], "greedy": [], "snapshot": []}
        for o, t in zip(outcomes, opp_tags):
            per_opp[t].append(o)
        for tag, lst in per_opp.items():
            if lst:
                wr = sum(1 for o in lst if o == "win") / len(lst)
                writer.add_scalar(f"eval/win_rate_vs_{tag}", wr, iteration)

        # Overall outcome
        n_games = len(outcomes)
        wins = sum(1 for o in outcomes if o == "win")
        losses = sum(1 for o in outcomes if o == "loss")
        draws = sum(1 for o in outcomes if o == "draw")
        writer.add_scalar("outcome/win_rate", wins / n_games, iteration)
        writer.add_scalar("outcome/loss_rate", losses / n_games, iteration)
        writer.add_scalar("outcome/draw_rate", draws / n_games, iteration)

        advantages, returns = compute_gae(transitions, gamma=gamma, lam=gae_lambda)

        states = torch.stack([t["state"] for t in transitions]).to(device)
        actions = torch.tensor([t["action"] for t in transitions], dtype=torch.long).to(device)
        old_log_probs = torch.tensor([t["log_prob"] for t in transitions], dtype=torch.float32).to(device)
        advantages_t = torch.tensor(advantages, dtype=torch.float32).to(device)
        returns_t = torch.tensor(returns, dtype=torch.float32).to(device)

        # Normalize advantages
        if len(advantages_t) > 1:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        network.train()
        dataset_size = len(transitions)
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        epoch_entropy = 0.0
        epoch_kl = 0.0
        num_updates = 0

        for _ in range(ppo_epochs):
            idx_order = list(range(dataset_size))
            random.shuffle(idx_order)
            for start in range(0, dataset_size, batch_size):
                end = min(start + batch_size, dataset_size)
                idx = idx_order[start:end]

                b_states = states[idx]
                b_actions = actions[idx]
                b_old_lp = old_log_probs[idx]
                b_adv = advantages_t[idx]
                b_ret = returns_t[idx]

                logits, values = network(b_states)
                values = values.squeeze(-1)

                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(b_actions)
                entropy = dist.entropy()

                ratio = torch.exp(new_log_probs - b_old_lp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values, b_ret)
                entropy_term = -entropy_coeff * entropy.mean()

                loss = policy_loss + value_coeff * value_loss + entropy_term

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 0.5)
                optimizer.step()

                # Approximate KL: mean(old_log_prob - new_log_prob)
                with torch.no_grad():
                    approx_kl = (b_old_lp - new_log_probs).mean().item()

                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                epoch_entropy += entropy.mean().item()
                epoch_kl += approx_kl
                num_updates += 1

        if num_updates > 0:
            writer.add_scalar("training/policy_loss", epoch_policy_loss / num_updates, iteration)
            writer.add_scalar("training/value_loss", epoch_value_loss / num_updates, iteration)
            writer.add_scalar("training/entropy", epoch_entropy / num_updates, iteration)
            writer.add_scalar("training/approx_kl", epoch_kl / num_updates, iteration)
        writer.add_scalar("training/entropy_coeff", entropy_coeff, iteration)
        writer.add_scalar("training/transitions", dataset_size, iteration)
        writer.add_scalar("training/num_snapshots", opponent_pool.num_snapshots, iteration)
        writer.add_scalar("training/value_predictions_mean", values.mean().item(), iteration)
        writer.add_scalar("training/returns_mean", returns_t.mean().item(), iteration)

        # Snapshot
        if iteration % pool_update_freq == 0:
            opponent_pool.add_snapshot(network)

        if iteration % 10 == 0:
            wr = wins / n_games
            print(
                f"Iter {iteration}/{num_iterations} | games={n_games} | "
                f"win={wr:.2%} | ent_coeff={entropy_coeff:.4f} | "
                f"snaps={opponent_pool.num_snapshots}"
            )

        if iteration % eval_every_iters == 0 or iteration == num_iterations:
            path = os.path.join(checkpoint_dir, f"iter_{iteration}.pt")
            torch.save(network.state_dict(), path)
            eval_wr = clean_eval_vs_greedy(network, device, eval_games)
            writer.add_scalar("clean_eval/win_rate_vs_greedy", eval_wr, iteration)
            msg = f"  [iter {iteration}] clean_eval vs Greedy: {eval_wr:.1%}"
            if eval_wr > best_eval_wr:
                best_eval_wr = eval_wr
                best_path = os.path.join(checkpoint_dir, "best.pt")
                torch.save(network.state_dict(), best_path)
                msg += "  <- NEW BEST"
            print(msg)

    writer.close()
    print(f"\nTraining complete. Best clean_eval vs Greedy: {best_eval_wr:.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO training")
    parser.add_argument("--num-iterations", type=int, default=1000)
    parser.add_argument("--games-per-iter", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--no-shape-rewards", action="store_true",
                        help="Disable reward shaping (shaping is on by default)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/ppo")
    parser.add_argument("--log-dir", type=str, default="runs/ppo")
    args = parser.parse_args()

    train(
        num_iterations=args.num_iterations,
        games_per_iter=args.games_per_iter,
        lr=args.lr,
        shape_rewards=not args.no_shape_rewards,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
    )
