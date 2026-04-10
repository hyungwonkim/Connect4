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


class ResidualBlock(nn.Module):
    """Residual block: conv → BN → ReLU → conv → BN → skip → ReLU."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class AlphaZeroNetV2(nn.Module):
    """Larger policy + value network: 256 channels, 6 residual blocks."""

    def __init__(self, channels: int = 256, num_blocks: int = 6):
        super().__init__()
        # Input convolution
        self.conv_in = nn.Conv2d(3, channels, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(channels)
        # Residual tower
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_blocks)]
        )
        # Policy head
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 6 * 7, 7)  # 84 → 7
        # Value head
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * 6 * 7, 128)  # 42 → 128
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Backbone
        x = F.relu(self.bn_in(self.conv_in(x)))
        x = self.res_blocks(x)
        # Policy
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        log_policy = F.log_softmax(self.policy_fc(p), dim=-1)
        # Value
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
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
    """Dueling DQN: separate V(s) and A(s,a) streams.

    Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))

    Helps Connect 4 where many moves are equally "meh" but one move is vital:
    the value stream learns board-state value independent of action choice.
    """

    def __init__(self):
        super().__init__()
        self.backbone = Connect4CNN()
        # Value stream
        self.value_fc1 = nn.Linear(2688, 256)
        self.value_fc2 = nn.Linear(256, 1)
        # Advantage stream
        self.adv_fc1 = nn.Linear(2688, 256)
        self.adv_fc2 = nn.Linear(256, 7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        v = F.relu(self.value_fc1(features))
        v = self.value_fc2(v)  # (batch, 1)
        a = F.relu(self.adv_fc1(features))
        a = self.adv_fc2(a)  # (batch, 7)
        # Recombine: Q = V + (A - mean(A))
        return v + (a - a.mean(dim=1, keepdim=True))
