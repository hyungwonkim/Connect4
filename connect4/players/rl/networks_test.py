"""Tests for neural network architectures."""

import torch

from connect4.players.rl.networks import AlphaZeroNetV2, PPONet, DQNNet


def test_alphazero_net_output_shapes():
    net = AlphaZeroNetV2()
    x = torch.randn(2, 3, 6, 7)  # batch of 2
    log_policy, value = net(x)
    assert log_policy.shape == (2, 7)
    assert value.shape == (2, 1)


def test_alphazero_net_value_range():
    net = AlphaZeroNetV2()
    x = torch.randn(4, 3, 6, 7)
    _, value = net(x)
    # tanh output should be in [-1, 1]
    assert (value >= -1.0).all()
    assert (value <= 1.0).all()


def test_alphazero_net_log_policy():
    net = AlphaZeroNetV2()
    x = torch.randn(1, 3, 6, 7)
    log_policy, _ = net(x)
    # exp(log_softmax) should sum to ~1
    probs = torch.exp(log_policy)
    assert abs(probs.sum().item() - 1.0) < 1e-5


def test_ppo_net_output_shapes():
    net = PPONet()
    x = torch.randn(2, 3, 6, 7)
    logits, value = net(x)
    assert logits.shape == (2, 7)
    assert value.shape == (2, 1)


def test_dqn_net_output_shapes():
    """DQNNet is a Dueling architecture: V(s) + A(s,a) streams."""
    net = DQNNet()
    x = torch.randn(2, 3, 6, 7)
    q_values = net(x)
    assert q_values.shape == (2, 7)
