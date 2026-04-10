"""Monte Carlo Tree Search for AlphaZero."""

import math

import numpy as np
import torch

from connect4.board import Board, P1, P2
from connect4.players.rl.common import board_to_tensor

VIRTUAL_LOSS = 1.0


class MCTSNode:
    """A node in the MCTS tree."""

    def __init__(self, board: Board, player: int, prior: float = 0.0):
        self.board = board
        self.player = player  # player whose turn it is
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: dict[int, MCTSNode] = {}
        # Terminal info (set at creation if the move that produced this node ended the game).
        # terminal_value is from the perspective of the player who just moved
        # (i.e., the player whose turn it is NOT, i.e. parent's player).
        self.is_terminal = False
        self.terminal_value = 0.0  # +1 win / -1 loss / 0 draw from mover's perspective

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_expanded(self) -> bool:
        return len(self.children) > 0 or self.is_terminal


class MCTS:
    """MCTS with neural network guidance for AlphaZero."""

    def __init__(self, network, num_simulations: int = 200, c_puct: float = 1.41,
                 batch_size: int = 8, dirichlet_alpha: float = 0.3,
                 dirichlet_eps: float = 0.25):
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.batch_size = batch_size
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.device = next(network.parameters()).device

    @torch.no_grad()
    def _evaluate(self, board: Board, player: int) -> tuple[list[float], float]:
        """Get policy and value from network (single state)."""
        state = board_to_tensor(board, player).unsqueeze(0).to(self.device)
        log_policy, value = self.network(state)
        policy = torch.exp(log_policy).squeeze(0).cpu().tolist()
        return policy, value.item()

    @torch.no_grad()
    def _evaluate_batch(self, boards: list, players: list) -> list:
        """Evaluate multiple board states in a single forward pass."""
        states = torch.stack([board_to_tensor(b, p) for b, p in zip(boards, players)])
        states = states.to(self.device)
        log_policies, values = self.network(states)
        policies = torch.exp(log_policies).cpu().tolist()
        values = values.squeeze(-1).cpu().tolist()
        return list(zip(policies, values))

    def search(self, board: Board, current_player: int,
               add_noise: bool = False) -> list[float]:
        """Run MCTS and return visit count distribution over 7 columns."""
        root = MCTSNode(board.copy(), current_player)
        # Expand root
        self._expand(root)

        # Inject Dirichlet noise at root over valid-move children.
        if add_noise and root.children:
            actions = list(root.children.keys())
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))
            for a, n in zip(actions, noise):
                child = root.children[a]
                child.prior = (1 - self.dirichlet_eps) * child.prior + self.dirichlet_eps * float(n)

        sims_done = 0
        while sims_done < self.num_simulations:
            batch = min(self.batch_size, self.num_simulations - sims_done)

            # Collect leaves from parallel traversals
            paths = []
            leaf_nodes = []
            leaf_terminals = []  # (search_path, value) for terminal nodes

            for i in range(batch):
                node = root
                search_path = [node]

                # Selection with virtual loss
                while node.is_expanded() and not node.is_terminal:
                    _, node = self._select_child(node)
                    search_path.append(node)
                    # Apply virtual loss to discourage parallel paths from converging
                    node.visit_count += 1
                    node.value_sum -= VIRTUAL_LOSS

                # Terminal node: terminal_value is from mover's (= parent's) POV.
                # This matches the convention of the original terminal-handling code
                # (see git history): _backpropagate is called with parent's POV.
                if node.is_terminal:
                    leaf_terminals.append((search_path, node.terminal_value))
                else:
                    paths.append(search_path)
                    leaf_nodes.append(node)

            # Batched expansion + evaluation for non-terminal leaves
            if leaf_nodes:
                boards = [n.board for n in leaf_nodes]
                players = [n.player for n in leaf_nodes]
                results = self._evaluate_batch(boards, players)

                for node, search_path, (policy, value) in zip(leaf_nodes, paths, results):
                    self._expand_with_policy(node, policy)
                    # Remove virtual loss and backpropagate real value.
                    # Network returns value from leaf-player's POV, but _backpropagate
                    # expects value from leaf-parent's POV (= opposite). Negate.
                    self._remove_virtual_loss(search_path)
                    self._backpropagate(search_path, -value)

            # Handle terminal nodes
            for search_path, value in leaf_terminals:
                self._remove_virtual_loss(search_path)
                self._backpropagate(search_path, value)

            sims_done += batch

        # Return visit distribution
        visits = [0.0] * 7
        for action, child in root.children.items():
            visits[action] = child.visit_count
        total = sum(visits)
        if total > 0:
            visits = [v / total for v in visits]
        return visits

    def _expand(self, node: MCTSNode):
        """Expand root node: evaluate with NN, create children (with terminal detection)."""
        valid_moves = node.board.get_valid_moves()
        if not valid_moves:
            return
        policy, _ = self._evaluate(node.board, node.player)
        self._expand_with_policy(node, policy)

    def _expand_with_policy(self, node: MCTSNode, policy: list[float]):
        """Create children for node using provided policy priors, detecting terminals."""
        valid_moves = node.board.get_valid_moves()
        if not valid_moves:
            return
        next_player = P2 if node.player == P1 else P1
        for col in valid_moves:
            child_board = node.board.copy()
            child_board.drop_piece(col, node.player)
            child = MCTSNode(child_board, next_player, prior=policy[col])
            # Check terminal from mover's (node.player's) perspective
            winner = child_board.check_winner()
            if winner is not None:
                child.is_terminal = True
                child.terminal_value = 1.0  # mover won
            elif child_board.is_draw():
                child.is_terminal = True
                child.terminal_value = 0.0
            node.children[col] = child

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

    def _remove_virtual_loss(self, search_path: list[MCTSNode]):
        """Remove virtual loss from all nodes in path (skip root)."""
        for node in search_path[1:]:
            node.visit_count -= 1
            node.value_sum += VIRTUAL_LOSS

    def _backpropagate(self, search_path: list[MCTSNode], value: float):
        """Backpropagate value up the tree, negating at each level."""
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            value = -value
