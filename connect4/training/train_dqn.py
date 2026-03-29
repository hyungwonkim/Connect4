"""DQN training with experience replay and target network."""

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


class ReplayBuffer:
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(list(self.buffer), batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


def play_episode(policy_net, device, replay_buffer, epsilon):
    """Play one game: agent (P1) vs random opponent (P2).

    Agent's 'next state' is the board after the opponent also moves.
    If the opponent wins, reward = -1.
    """
    board = Board()
    current_player = P1

    agent_last_state = None
    agent_last_action = None

    while True:
        if current_player == P1:
            # Agent's turn
            state = board_to_tensor(board, P1)

            # Store previous transition (agent sees board after opponent moved)
            if agent_last_state is not None:
                replay_buffer.push(agent_last_state, agent_last_action, 0.0, state, False)

            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.choice(board.get_valid_moves())
            else:
                with torch.no_grad():
                    q_values = policy_net(state.unsqueeze(0).to(device)).squeeze(0).cpu()
                    masked = mask_invalid(q_values, board)
                    action = masked.argmax().item()

            agent_last_state = state
            agent_last_action = action
            board.drop_piece(action, P1)

            winner = board.check_winner()
            if winner == P1:
                final_state = board_to_tensor(board, P1)
                replay_buffer.push(agent_last_state, agent_last_action, 1.0, final_state, True)
                return "win"
            if board.is_draw():
                final_state = board_to_tensor(board, P1)
                replay_buffer.push(agent_last_state, agent_last_action, 0.0, final_state, True)
                return "draw"

        else:
            # Random opponent
            action = random.choice(board.get_valid_moves())
            board.drop_piece(action, P2)

            winner = board.check_winner()
            if winner == P2:
                # Opponent won — agent gets -1
                if agent_last_state is not None:
                    final_state = board_to_tensor(board, P1)
                    replay_buffer.push(agent_last_state, agent_last_action, -1.0, final_state, True)
                return "loss"
            if board.is_draw():
                if agent_last_state is not None:
                    final_state = board_to_tensor(board, P1)
                    replay_buffer.push(agent_last_state, agent_last_action, 0.0, final_state, True)
                return "draw"

        current_player = P2 if current_player == P1 else P1


def train(
    num_episodes: int = 50000,
    batch_size: int = 256,
    lr: float = 1e-4,
    gamma: float = 1.0,
    buffer_size: int = 100000,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay_steps: int = 30000,
    target_sync_freq: int = 1000,
    checkpoint_dir: str = "checkpoints/dqn",
):
    device = get_device()
    policy_net = DQNNet().to(device)
    target_net = DQNNet().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(buffer_size)

    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join("runs", "dqn"))

    total_steps = 0
    recent_outcomes = deque(maxlen=1000)

    for episode in range(1, num_episodes + 1):
        # Linear epsilon decay
        epsilon = max(eps_end, eps_start - (eps_start - eps_end) * total_steps / eps_decay_steps)

        outcome = play_episode(policy_net, device, replay_buffer, epsilon)
        recent_outcomes.append(outcome)

        # Train
        loss = None
        if len(replay_buffer) >= batch_size:
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
            states = states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            next_states = next_states.to(device)
            dones = dones.to(device)

            # Current Q values
            q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

            # Target Q values (Double DQN: use policy net to select action, target net to evaluate)
            with torch.no_grad():
                next_q_policy = policy_net(next_states)
                next_actions = next_q_policy.argmax(dim=1)
                next_q_target = target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                target = rewards + gamma * next_q_target * (1 - dones)

            loss = F.smooth_l1_loss(q_values, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
            optimizer.step()

            total_steps += 1

        # Sync target network
        if total_steps % target_sync_freq == 0 and total_steps > 0:
            target_net.load_state_dict(policy_net.state_dict())

        # TensorBoard logging every 100 episodes
        if episode % 100 == 0:
            n = len(recent_outcomes)
            wins = sum(1 for o in recent_outcomes if o == "win")
            losses = sum(1 for o in recent_outcomes if o == "loss")
            draws = sum(1 for o in recent_outcomes if o == "draw")
            writer.add_scalar("outcome/win_rate", wins / n, episode)
            writer.add_scalar("outcome/loss_rate", losses / n, episode)
            writer.add_scalar("outcome/draw_rate", draws / n, episode)
            writer.add_scalar("training/epsilon", epsilon, episode)
            writer.add_scalar("training/buffer_size", len(replay_buffer), episode)
            if loss is not None:
                writer.add_scalar("training/loss", loss.item(), episode)
            writer.add_scalar("training/mean_q", q_values.mean().item() if loss is not None else 0, episode)

        if episode % 1000 == 0:
            win_rate = sum(1 for o in recent_outcomes if o == "win") / len(recent_outcomes)
            print(f"Episode {episode}/{num_episodes} | eps: {epsilon:.3f} | "
                  f"buffer: {len(replay_buffer)} | win_rate: {win_rate:.2%}")

        if episode % 5000 == 0 or episode == num_episodes:
            path = os.path.join(checkpoint_dir, f"episode_{episode}.pt")
            torch.save(policy_net.state_dict(), path)
            best_path = os.path.join(checkpoint_dir, "best.pt")
            torch.save(policy_net.state_dict(), best_path)
            print(f"  Saved checkpoint: {path}")

    writer.close()
    print("\nTraining complete.")


if __name__ == "__main__":
    train()
