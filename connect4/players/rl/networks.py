"""Neural network architectures for RL players."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Connect4CNN(nn.Module):
    """Shared CNN backbone: 3 conv layers → flatten to 2688."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x.view(x.size(0), -1)  # (batch, 2688)


class AlphaZeroNet(nn.Module):
    """Policy + value network for AlphaZero."""

    def __init__(self):
        super().__init__()
        self.backbone = Connect4CNN()
        # Policy head
        self.policy_fc1 = nn.Linear(2688, 256)
        self.policy_fc2 = nn.Linear(256, 7)
        # Value head
        self.value_fc1 = nn.Linear(2688, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        # Policy
        p = F.relu(self.policy_fc1(features))
        log_policy = F.log_softmax(self.policy_fc2(p), dim=-1)
        # Value
        v = F.relu(self.value_fc1(features))
        value = torch.tanh(self.value_fc2(v))
        return log_policy, value


class PPONet(nn.Module):
    """Actor-critic network for PPO."""

    def __init__(self):
        super().__init__()
        self.backbone = Connect4CNN()
        # Actor head
        self.actor_fc1 = nn.Linear(2688, 256)
        self.actor_fc2 = nn.Linear(256, 7)
        # Critic head
        self.critic_fc1 = nn.Linear(2688, 256)
        self.critic_fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        # Actor (raw logits)
        a = F.relu(self.actor_fc1(features))
        logits = self.actor_fc2(a)
        # Critic (raw value)
        c = F.relu(self.critic_fc1(features))
        value = self.critic_fc2(c)
        return logits, value


class DQNNet(nn.Module):
    """Q-network for DQN."""

    def __init__(self):
        super().__init__()
        self.backbone = Connect4CNN()
        self.q_fc1 = nn.Linear(2688, 256)
        self.q_fc2 = nn.Linear(256, 7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        q = F.relu(self.q_fc1(features))
        return self.q_fc2(q)
