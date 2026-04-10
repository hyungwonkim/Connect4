"""Prioritized Experience Replay buffer.

Proportional variant (Schaul et al. 2016). Uses a simple numpy priority array;
O(N) sampling is fine for buffers up to ~1M on Connect 4 workloads.
"""

from __future__ import annotations

import numpy as np
import torch


class PrioritizedReplayBuffer:
    def __init__(
        self,
        capacity: int = 100_000,
        alpha: float = 0.6,
        eps: float = 1e-6,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.eps = eps

        self._storage: list[tuple] = []
        self._priorities = np.zeros(capacity, dtype=np.float64)
        self._pos = 0
        self._max_priority = 1.0  # Initial priority for new samples

    def __len__(self) -> int:
        return len(self._storage)

    def push(self, state, action, reward, next_state, done) -> None:
        transition = (state, action, reward, next_state, done)
        if len(self._storage) < self.capacity:
            self._storage.append(transition)
        else:
            self._storage[self._pos] = transition
        # New transitions get max priority so they are sampled at least once
        self._priorities[self._pos] = self._max_priority
        self._pos = (self._pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4):
        """Sample a batch. Returns (states, actions, rewards, next_states, dones, is_weights, indices)."""
        n = len(self._storage)
        prios = self._priorities[:n] ** self.alpha
        probs = prios / prios.sum()

        indices = np.random.choice(n, batch_size, p=probs)
        batch = [self._storage[i] for i in indices]

        # Importance-sampling weights
        weights = (n * probs[indices]) ** (-beta)
        weights = weights / weights.max()  # normalise to [0, 1]
        is_weights = torch.tensor(weights, dtype=torch.float32)

        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.float32),
            is_weights,
            indices,
        )

    def update_priorities(self, indices, td_errors) -> None:
        """Update priorities for sampled transitions based on |TD error|."""
        if isinstance(td_errors, torch.Tensor):
            td_errors = td_errors.detach().cpu().numpy()
        new_prios = np.abs(td_errors) + self.eps
        for idx, p in zip(indices, new_prios):
            self._priorities[idx] = p
            if p > self._max_priority:
                self._max_priority = float(p)
