"""PPO training with policy pool for opponent diversity."""

import os
import copy
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


def collect_rollout(network, opponent_fn, device, num_games=50):
    """Collect transitions from the learning agent's perspective.

    The agent always plays as P1. Opponent plays as P2.
    Returns (transitions, outcomes) where outcomes is a list of "win"/"loss"/"draw".
    """
    transitions = []
    outcomes = []

    for _ in range(num_games):
        board = Board()
        current_player = P1
        game_transitions = []  # agent's transitions for this game

        while True:
            if current_player == P1:
                # Agent's turn
                state = board_to_tensor(board, P1)
                state_batch = state.unsqueeze(0).to(device)
                with torch.no_grad():
                    logits, value = network(state_batch)
                logits = logits.squeeze(0).cpu()
                value = value.item()

                masked_logits = mask_invalid(logits, board)
                dist = Categorical(logits=masked_logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                game_transitions.append({
                    "state": state,
                    "action": action.item(),
                    "log_prob": log_prob.item(),
                    "value": value,
                    "reward": 0.0,
                    "done": False,
                })

                board.drop_piece(action.item(), P1)
            else:
                # Opponent's turn
                action = opponent_fn(board, P2)
                board.drop_piece(action, P2)

            # Check terminal after each move
            winner = board.check_winner()
            if winner is not None:
                if game_transitions:
                    if winner == P1:
                        game_transitions[-1]["reward"] = 1.0
                        outcomes.append("win")
                    else:
                        game_transitions[-1]["reward"] = -1.0
                        outcomes.append("loss")
                    game_transitions[-1]["done"] = True
                break

            if board.is_draw():
                if game_transitions:
                    game_transitions[-1]["reward"] = 0.0
                    game_transitions[-1]["done"] = True
                    outcomes.append("draw")
                break

            current_player = P2 if current_player == P1 else P1

        transitions.extend(game_transitions)

    return transitions, outcomes


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


def random_opponent(board, player_id):
    return random.choice(board.get_valid_moves())


def make_network_opponent(network, device):
    """Create an opponent function from a network snapshot."""
    net = copy.deepcopy(network)
    net.eval()

    def opponent_fn(board, player_id):
        state = board_to_tensor(board, player_id).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, _ = net(state)
        logits = logits.squeeze(0).cpu()
        masked = mask_invalid(logits, board)
        return masked.argmax().item()

    return opponent_fn


def train(
    num_iterations: int = 1000,
    games_per_iter: int = 50,
    clip_eps: float = 0.2,
    ppo_epochs: int = 4,
    batch_size: int = 64,
    lr: float = 3e-4,
    gamma: float = 1.0,
    pool_update_freq: int = 50,
    checkpoint_dir: str = "checkpoints/ppo",
):
    device = get_device()
    network = PPONet().to(device)
    optimizer = optim.Adam(network.parameters(), lr=lr)

    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join("runs", "ppo"))

    # Policy pool starts with random
    opponent_pool = [random_opponent]

    for iteration in range(1, num_iterations + 1):
        # Pick random opponent from pool
        opponent_fn = random.choice(opponent_pool)

        # Collect rollouts
        network.eval()
        transitions, outcomes = collect_rollout(network, opponent_fn, device, games_per_iter)

        if not transitions:
            continue

        # Log game outcomes
        n_games = len(outcomes)
        if n_games > 0:
            wins = sum(1 for o in outcomes if o == "win")
            losses = sum(1 for o in outcomes if o == "loss")
            draws = sum(1 for o in outcomes if o == "draw")
            writer.add_scalar("outcome/win_rate", wins / n_games, iteration)
            writer.add_scalar("outcome/loss_rate", losses / n_games, iteration)
            writer.add_scalar("outcome/draw_rate", draws / n_games, iteration)

        advantages, returns = compute_gae(transitions, gamma)

        # Prepare tensors
        states = torch.stack([t["state"] for t in transitions]).to(device)
        actions = torch.tensor([t["action"] for t in transitions], dtype=torch.long).to(device)
        old_log_probs = torch.tensor([t["log_prob"] for t in transitions], dtype=torch.float32).to(device)
        advantages_t = torch.tensor(advantages, dtype=torch.float32).to(device)
        returns_t = torch.tensor(returns, dtype=torch.float32).to(device)

        # Normalize advantages
        if len(advantages_t) > 1:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # PPO update
        network.train()
        dataset_size = len(transitions)
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        epoch_entropy = 0.0
        num_updates = 0

        for _ in range(ppo_epochs):
            indices = list(range(dataset_size))
            random.shuffle(indices)

            for start in range(0, dataset_size, batch_size):
                end = min(start + batch_size, dataset_size)
                idx = indices[start:end]

                b_states = states[idx]
                b_actions = actions[idx]
                b_old_lp = old_log_probs[idx]
                b_adv = advantages_t[idx]
                b_ret = returns_t[idx]

                logits, values = network(b_states)
                values = values.squeeze(-1)

                # Mask invalid moves for each state in batch
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(b_actions)
                entropy = dist.entropy()

                ratio = torch.exp(new_log_probs - b_old_lp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values, b_ret)
                entropy_bonus = -0.01 * entropy.mean()

                loss = policy_loss + 0.5 * value_loss + entropy_bonus

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 0.5)
                optimizer.step()

                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                epoch_entropy += entropy.mean().item()
                num_updates += 1

        if num_updates > 0:
            writer.add_scalar("training/policy_loss", epoch_policy_loss / num_updates, iteration)
            writer.add_scalar("training/value_loss", epoch_value_loss / num_updates, iteration)
            writer.add_scalar("training/entropy", epoch_entropy / num_updates, iteration)
        writer.add_scalar("training/transitions", dataset_size, iteration)
        writer.add_scalar("training/pool_size", len(opponent_pool), iteration)

        # Add snapshot to pool periodically
        if iteration % pool_update_freq == 0:
            opponent_pool.append(make_network_opponent(network, device))
            if len(opponent_pool) > 10:
                opponent_pool.pop(1)  # keep random + last 9

        if iteration % 50 == 0:
            win_rate = sum(1 for o in outcomes if o == "win") / max(len(outcomes), 1)
            print(f"Iteration {iteration}/{num_iterations} | transitions: {dataset_size} | win_rate: {win_rate:.2%}")

        if iteration % 100 == 0 or iteration == num_iterations:
            path = os.path.join(checkpoint_dir, f"iter_{iteration}.pt")
            torch.save(network.state_dict(), path)
            best_path = os.path.join(checkpoint_dir, "best.pt")
            torch.save(network.state_dict(), best_path)
            print(f"  Saved checkpoint: {path}")

    writer.close()
    print("\nTraining complete.")


if __name__ == "__main__":
    train()
