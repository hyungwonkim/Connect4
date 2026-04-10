"""DQN training.

Architecture & techniques:
  - Dueling Q-network (V + A streams)
  - Double DQN, Huber (smooth_l1) loss, gamma=1.0
  - Prioritized Experience Replay
  - Opponent pool (Random / Greedy / self-snapshots)
  - Asymmetric terminal rewards: win=+1.0, loss=-1.2, draw=0.0
  - Reward shaping (3-in-a-row, blocking, center) — ON by default
  - LR schedule with step-down, per-opponent win-rate + Q-value diagnostics
"""

from __future__ import annotations

import argparse
import os
import random
from collections import deque

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from connect4.board import Board, P1, P2
from connect4.players.rl.common import board_to_tensor, mask_invalid, get_device
from connect4.players.rl.networks import DQNNet
from connect4.training.opponents import OpponentPool
from connect4.training.per_buffer import PrioritizedReplayBuffer
from connect4.training.rewards import (
    shape_agent_move,
    shape_opponent_move,
)

# Terminal rewards (asymmetric to combat Q-value overestimation)
R_WIN = 1.0
R_LOSS = -1.2
R_DRAW = 0.0


def clean_eval_vs_greedy(policy_net, device, num_games: int) -> float:
    """Deterministic eval (eps=0) vs Greedy, averaged over both player sides.

    Returns win rate in [0, 1]. Draws count as 0.5.
    """
    from connect4.players.greedy_player import GreedyPlayer
    policy_net.eval()
    wins = 0.0
    half = num_games // 2
    # DQN as P1
    for _ in range(half):
        board = Board()
        cur = P1
        greedy = GreedyPlayer(P2)
        while True:
            if cur == P1:
                state = board_to_tensor(board, P1)
                with torch.no_grad():
                    q = policy_net(state.unsqueeze(0).to(device)).squeeze(0).cpu()
                    a = mask_invalid(q, board).argmax().item()
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
    # DQN as P2
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
                    q = policy_net(state.unsqueeze(0).to(device)).squeeze(0).cpu()
                    a = mask_invalid(q, board).argmax().item()
                board.drop_piece(a, P2)
                if board.check_winner() == P2:
                    wins += 1.0; break
                if board.is_draw():
                    wins += 0.5; break
            cur = P2 if cur == P1 else P1
    policy_net.train()
    return wins / num_games


def play_episode(
    policy_net,
    device,
    buffer: PrioritizedReplayBuffer,
    opponent_pool: OpponentPool,
    epsilon: float,
    shape_rewards: bool,
) -> tuple[str, str]:
    """Play one game: agent (P1) vs sampled opponent (P2).

    Returns (outcome, opponent_tag) where outcome in {"win","loss","draw"}.
    """
    opponent_fn, opp_tag = opponent_pool.sample()
    board = Board()
    current_player = P1

    last_state = None
    last_action = None
    pending_reward = 0.0  # shaping from agent's most recent move + opponent's reply

    while True:
        if current_player == P1:
            # Commit previous transition now that we've seen the board after opponent moved.
            if last_state is not None:
                next_state = board_to_tensor(board, P1)
                buffer.push(last_state, last_action, pending_reward, next_state, False)
                pending_reward = 0.0

            state = board_to_tensor(board, P1)

            # epsilon-greedy
            if random.random() < epsilon:
                action = random.choice(board.get_valid_moves())
            else:
                with torch.no_grad():
                    q = policy_net(state.unsqueeze(0).to(device)).squeeze(0).cpu()
                    action = mask_invalid(q, board).argmax().item()

            board_before = board.copy()
            row_placed = board.drop_piece(action, P1)

            # Shaping from agent's own move
            if shape_rewards:
                pending_reward += shape_agent_move(
                    board_before, board, action, row_placed, P1,
                )

            last_state = state
            last_action = action

            winner = board.check_winner()
            if winner == P1:
                final = board_to_tensor(board, P1)
                buffer.push(last_state, last_action, R_WIN, final, True)
                return "win", opp_tag
            if board.is_draw():
                final = board_to_tensor(board, P1)
                buffer.push(last_state, last_action, R_DRAW, final, True)
                return "draw", opp_tag

        else:
            # Opponent's turn
            board_before = board.copy()
            action = opponent_fn(board, P2)
            row_placed = board.drop_piece(action, P2)

            # Shaping penalty if opponent created a 3-in-a-row
            if shape_rewards and last_state is not None:
                pending_reward += shape_opponent_move(
                    board_before, board, action, row_placed, P2,
                )

            winner = board.check_winner()
            if winner == P2:
                if last_state is not None:
                    final = board_to_tensor(board, P1)
                    buffer.push(last_state, last_action, R_LOSS, final, True)
                return "loss", opp_tag
            if board.is_draw():
                if last_state is not None:
                    final = board_to_tensor(board, P1)
                    buffer.push(last_state, last_action, R_DRAW, final, True)
                return "draw", opp_tag

        current_player = P2 if current_player == P1 else P1


def train(
    num_episodes: int = 80000,
    batch_size: int = 256,
    lr: float = 1e-4,
    lr_final: float = 3e-5,
    lr_drop_episode: int = 40000,
    gamma: float = 1.0,
    buffer_size: int = 100_000,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay_steps: int = 50_000,
    target_sync_freq: int = 2000,
    eval_every_episodes: int = 5000,
    eval_games: int = 40,
    per_alpha: float = 0.6,
    per_beta_start: float = 0.4,
    per_beta_end: float = 1.0,
    snapshot_every_episodes: int = 2000,
    max_snapshots: int = 8,
    shape_rewards: bool = True,
    checkpoint_dir: str = "checkpoints/dqn",
    log_dir: str = "runs/dqn",
):
    device = get_device()
    policy_net = DQNNet().to(device)
    target_net = DQNNet().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    # Huber loss (smooth_l1) — already correct from v1. NOTE: do NOT switch to MSE.
    buffer = PrioritizedReplayBuffer(capacity=buffer_size, alpha=per_alpha)

    opponent_pool = OpponentPool(
        device=device,
        is_policy=False,   # DQN outputs Q-values, not (logits, value)
        weights=(0.10, 0.60, 0.30),  # Random / Greedy / snapshots
        opponent_player_id=P2,
        max_snapshots=max_snapshots,
    )

    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    total_steps = 0
    recent_outcomes = deque(maxlen=1000)
    recent_by_opp: dict[str, deque] = {
        "random": deque(maxlen=500),
        "greedy": deque(maxlen=500),
        "snapshot": deque(maxlen=500),
    }

    print(f"Training DQN on {device} | shape_rewards={shape_rewards}")
    print(f"  episodes={num_episodes}, buffer={buffer_size}, eps_decay={eps_decay_steps}")
    print(f"  lr: {lr} -> {lr_final} at ep {lr_drop_episode}, target_sync={target_sync_freq}")

    lr_dropped = False
    best_eval_wr = -1.0

    for episode in range(1, num_episodes + 1):
        # LR schedule: step-down at lr_drop_episode
        if not lr_dropped and episode >= lr_drop_episode:
            for g in optimizer.param_groups:
                g["lr"] = lr_final
            lr_dropped = True
            print(f"  [ep {episode}] LR dropped to {lr_final}")

        # Linear epsilon decay
        epsilon = max(
            eps_end,
            eps_start - (eps_start - eps_end) * total_steps / eps_decay_steps,
        )
        # Linear PER beta anneal
        beta = min(
            per_beta_end,
            per_beta_start + (per_beta_end - per_beta_start) * episode / num_episodes,
        )

        outcome, opp_tag = play_episode(
            policy_net, device, buffer, opponent_pool, epsilon, shape_rewards,
        )
        recent_outcomes.append(outcome)
        recent_by_opp[opp_tag].append(outcome)

        # Training step
        loss_val = None
        mean_q = max_q = min_q = td_mean = None
        if len(buffer) >= batch_size:
            states, actions, rewards, next_states, dones, is_weights, indices = buffer.sample(
                batch_size, beta=beta,
            )
            states = states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            next_states = next_states.to(device)
            dones = dones.to(device)
            is_weights = is_weights.to(device)

            q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

            # Double DQN target
            with torch.no_grad():
                next_q_policy = policy_net(next_states)
                next_actions = next_q_policy.argmax(dim=1)
                next_q_target = target_net(next_states).gather(
                    1, next_actions.unsqueeze(1),
                ).squeeze(1)
                target = rewards + gamma * next_q_target * (1 - dones)

            td_errors = target - q_values
            # Weighted Huber loss using importance-sampling weights
            elementwise = F.smooth_l1_loss(q_values, target, reduction="none")
            loss = (is_weights * elementwise).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
            optimizer.step()

            # Update PER priorities with new TD errors
            buffer.update_priorities(indices, td_errors)

            total_steps += 1
            loss_val = loss.item()
            mean_q = q_values.mean().item()
            max_q = q_values.max().item()
            min_q = q_values.min().item()
            td_mean = td_errors.abs().mean().item()

            # Sync target net
            if total_steps % target_sync_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())

        # Snapshot for opponent pool
        if episode % snapshot_every_episodes == 0:
            opponent_pool.add_snapshot(policy_net)

        # Logging
        if episode % 100 == 0:
            n = len(recent_outcomes)
            wins = sum(1 for o in recent_outcomes if o == "win")
            losses = sum(1 for o in recent_outcomes if o == "loss")
            draws = sum(1 for o in recent_outcomes if o == "draw")
            writer.add_scalar("outcome/win_rate", wins / n, episode)
            writer.add_scalar("outcome/loss_rate", losses / n, episode)
            writer.add_scalar("outcome/draw_rate", draws / n, episode)
            writer.add_scalar("training/epsilon", epsilon, episode)
            writer.add_scalar("training/per_beta", beta, episode)
            writer.add_scalar("training/buffer_size", len(buffer), episode)
            writer.add_scalar("training/num_snapshots", opponent_pool.num_snapshots, episode)

            if loss_val is not None:
                writer.add_scalar("training/loss", loss_val, episode)
                writer.add_scalar("diagnostics/mean_q_batch", mean_q, episode)
                writer.add_scalar("diagnostics/max_q_batch", max_q, episode)
                writer.add_scalar("diagnostics/min_q_batch", min_q, episode)
                writer.add_scalar("diagnostics/td_error_abs_mean", td_mean, episode)

            for tag, dq in recent_by_opp.items():
                if dq:
                    wr = sum(1 for o in dq if o == "win") / len(dq)
                    writer.add_scalar(f"eval/win_rate_vs_{tag}", wr, episode)

        if episode % 1000 == 0:
            wr = sum(1 for o in recent_outcomes if o == "win") / len(recent_outcomes)
            mq_str = f"{mean_q:.3f}" if mean_q is not None else "n/a"
            print(
                f"Ep {episode}/{num_episodes} | eps={epsilon:.3f} | "
                f"buf={len(buffer)} | snaps={opponent_pool.num_snapshots} | "
                f"win_rate={wr:.2%} | mean_q={mq_str}"
            )

        if episode % eval_every_episodes == 0 or episode == num_episodes:
            path = os.path.join(checkpoint_dir, f"episode_{episode}.pt")
            torch.save(policy_net.state_dict(), path)
            # Clean deterministic eval vs Greedy (both sides, eps=0)
            eval_wr = clean_eval_vs_greedy(policy_net, device, eval_games)
            writer.add_scalar("clean_eval/win_rate_vs_greedy", eval_wr, episode)
            msg = f"  [ep {episode}] clean_eval vs Greedy: {eval_wr:.1%}"
            if eval_wr > best_eval_wr:
                best_eval_wr = eval_wr
                best_path = os.path.join(checkpoint_dir, "best.pt")
                torch.save(policy_net.state_dict(), best_path)
                msg += "  <- NEW BEST"
            print(msg)

    writer.close()
    print(f"\nTraining complete. Best clean_eval vs Greedy: {best_eval_wr:.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN training")
    parser.add_argument("--num-episodes", type=int, default=80000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-final", type=float, default=3e-5)
    parser.add_argument("--lr-drop-episode", type=int, default=40000)
    parser.add_argument("--eps-decay-steps", type=int, default=50_000)
    parser.add_argument("--target-sync-freq", type=int, default=2000)
    parser.add_argument("--no-shape-rewards", action="store_true",
                        help="Disable reward shaping (shaping is on by default)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/dqn")
    parser.add_argument("--log-dir", type=str, default="runs/dqn")
    args = parser.parse_args()

    train(
        num_episodes=args.num_episodes,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_final=args.lr_final,
        lr_drop_episode=args.lr_drop_episode,
        eps_decay_steps=args.eps_decay_steps,
        target_sync_freq=args.target_sync_freq,
        shape_rewards=not args.no_shape_rewards,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
    )
