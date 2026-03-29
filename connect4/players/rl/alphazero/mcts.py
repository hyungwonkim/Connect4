"""Monte Carlo Tree Search for AlphaZero."""

import math

import torch

from connect4.board import Board, P1, P2
from connect4.players.rl.common import board_to_tensor


class MCTSNode:
    """A node in the MCTS tree."""

    def __init__(self, board: Board, player: int, prior: float = 0.0):
        self.board = board
        self.player = player  # player whose turn it is
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: dict[int, MCTSNode] = {}

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_expanded(self) -> bool:
        return len(self.children) > 0


class MCTS:
    """MCTS with neural network guidance for AlphaZero."""

    def __init__(self, network, num_simulations: int = 200, c_puct: float = 1.41):
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        # Use the device the network is already on
        self.device = next(network.parameters()).device

    @torch.no_grad()
    def _evaluate(self, board: Board, player: int) -> tuple[list[float], float]:
        """Get policy and value from network."""
        state = board_to_tensor(board, player).unsqueeze(0).to(self.device)
        log_policy, value = self.network(state)
        policy = torch.exp(log_policy).squeeze(0).cpu().tolist()
        return policy, value.item()

    def search(self, board: Board, current_player: int) -> list[float]:
        """Run MCTS and return visit count distribution over 7 columns."""
        root = MCTSNode(board.copy(), current_player)
        # Expand root
        self._expand(root)

        for _ in range(self.num_simulations):
            node = root
            search_path = [node]

            # Selection — descend to a leaf
            while node.is_expanded():
                action, node = self._select_child(node)
                search_path.append(node)

            # Check terminal
            winner = node.board.check_winner()
            if winner is not None:
                # Value from the perspective of the node's parent (who just moved)
                parent_player = P2 if node.player == P1 else P1
                value = 1.0 if winner == parent_player else -1.0
            elif node.board.is_draw():
                value = 0.0
            else:
                # Expansion + evaluation
                self._expand(node)
                _, value = self._evaluate(node.board, node.player)

            # Backpropagation — negate at each level (two-player)
            self._backpropagate(search_path, value)

        # Return visit distribution
        visits = [0.0] * 7
        for action, child in root.children.items():
            visits[action] = child.visit_count
        total = sum(visits)
        if total > 0:
            visits = [v / total for v in visits]
        return visits

    def _expand(self, node: MCTSNode):
        """Expand node by adding children for all valid moves."""
        valid_moves = node.board.get_valid_moves()
        if not valid_moves:
            return

        policy, _ = self._evaluate(node.board, node.player)
        next_player = P2 if node.player == P1 else P1

        for col in valid_moves:
            child_board = node.board.copy()
            child_board.drop_piece(col, node.player)
            node.children[col] = MCTSNode(child_board, next_player, prior=policy[col])

    def _select_child(self, node: MCTSNode) -> tuple[int, MCTSNode]:
        """Select child with highest UCB score."""
        total_visits = sum(c.visit_count for c in node.children.values())
        sqrt_total = math.sqrt(total_visits + 1)

        best_score = float("-inf")
        best_action = -1
        best_child = None

        for action, child in node.children.items():
            ucb = child.q_value + self.c_puct * child.prior * sqrt_total / (1 + child.visit_count)
            if ucb > best_score:
                best_score = ucb
                best_action = action
                best_child = child

        return best_action, best_child

    def _backpropagate(self, search_path: list[MCTSNode], value: float):
        """Backpropagate value up the tree, negating at each level."""
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            value = -value
