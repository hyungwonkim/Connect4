import { Board, COLS, type Player } from "../../engine/board";
import { boardToTensor } from "../canonicalize";
import {
  runInferenceBatch,
  type NNSession,
  type NNTensorConstructor,
} from "../nn";

const VIRTUAL_LOSS = 1.0;
const INPUT_NAME = "board";

export interface MCTSOptions {
  numSimulations?: number;
  cPuct?: number;
  batchSize?: number;
}

/**
 * A node in the MCTS tree. `player` is the player whose turn it is at this
 * board. `terminalValue` (for terminal nodes) is from the POV of the player
 * who just moved — i.e. the parent's player — matching the Python version.
 */
export class MCTSNode {
  visitCount = 0;
  valueSum = 0;
  isTerminal = false;
  terminalValue = 0;
  children: Map<number, MCTSNode> = new Map();

  constructor(
    readonly board: Board,
    readonly player: Player,
    public prior: number = 0,
  ) {}

  get qValue(): number {
    return this.visitCount === 0 ? 0 : this.valueSum / this.visitCount;
  }

  isExpanded(): boolean {
    return this.children.size > 0 || this.isTerminal;
  }
}

function otherPlayer(p: Player): Player {
  return p === 1 ? 2 : 1;
}

export class MCTS {
  readonly numSimulations: number;
  readonly cPuct: number;
  readonly batchSize: number;

  constructor(
    private readonly session: NNSession,
    private readonly TensorCtor: NNTensorConstructor,
    options: MCTSOptions = {},
  ) {
    this.numSimulations = options.numSimulations ?? 200;
    this.cPuct = options.cPuct ?? 1.41;
    this.batchSize = options.batchSize ?? 8;
  }

  /**
   * Run MCTS from the given root and return the normalized visit distribution
   * over COLS columns (invalid/unvisited columns are 0).
   */
  async search(board: Board, currentPlayer: Player): Promise<number[]> {
    const root = new MCTSNode(board.copy(), currentPlayer);

    // Expand root with a single NN call.
    const rootEval = await this.evaluateBatch([root.board], [root.player]);
    this.expandWithPolicy(root, rootEval[0].policy);

    let simsDone = 0;
    while (simsDone < this.numSimulations) {
      const batch = Math.min(this.batchSize, this.numSimulations - simsDone);

      const nonTerminalPaths: MCTSNode[][] = [];
      const nonTerminalLeaves: MCTSNode[] = [];
      const terminalResults: { path: MCTSNode[]; value: number }[] = [];

      for (let i = 0; i < batch; i++) {
        let node: MCTSNode = root;
        const searchPath: MCTSNode[] = [node];

        // Selection: descend until we hit an unexpanded or terminal node.
        while (node.isExpanded() && !node.isTerminal) {
          const next = this.selectChild(node);
          node = next;
          searchPath.push(node);
          // Virtual loss discourages parallel traversals from converging.
          node.visitCount += 1;
          node.valueSum -= VIRTUAL_LOSS;
        }

        if (node.isTerminal) {
          terminalResults.push({ path: searchPath, value: node.terminalValue });
        } else {
          nonTerminalPaths.push(searchPath);
          nonTerminalLeaves.push(node);
        }
      }

      // Batched NN evaluation of non-terminal leaves.
      if (nonTerminalLeaves.length > 0) {
        const boards = nonTerminalLeaves.map((n) => n.board);
        const players = nonTerminalLeaves.map((n) => n.player);
        const results = await this.evaluateBatch(boards, players);

        for (let i = 0; i < nonTerminalLeaves.length; i++) {
          const leaf = nonTerminalLeaves[i];
          const path = nonTerminalPaths[i];
          const { policy, value } = results[i];
          this.expandWithPolicy(leaf, policy);
          this.removeVirtualLoss(path);
          // Network returns value from leaf-player's POV; backprop expects
          // parent's POV (= opposite). Negate before backprop.
          this.backpropagate(path, -value);
        }
      }

      // Terminal leaves don't need NN evaluation.
      for (const { path, value } of terminalResults) {
        this.removeVirtualLoss(path);
        this.backpropagate(path, value);
      }

      simsDone += batch;
    }

    // Build visit distribution (normalized).
    const visits = new Array<number>(COLS).fill(0);
    for (const [action, child] of root.children) {
      visits[action] = child.visitCount;
    }
    const total = visits.reduce((a, b) => a + b, 0);
    if (total > 0) {
      for (let i = 0; i < COLS; i++) visits[i] /= total;
    }
    return visits;
  }

  /**
   * Evaluate a list of (board, player) states in a single batched NN call.
   * Returns one `{ policy, value }` per input. `policy` is exp(log_policy),
   * matching the Python version which exponentiates after the network.
   */
  private async evaluateBatch(
    boards: Board[],
    players: Player[],
  ): Promise<{ policy: Float32Array; value: number }[]> {
    const tensors = boards.map((b, i) => boardToTensor(b, players[i]));
    const outputs = await runInferenceBatch(
      this.session,
      this.TensorCtor,
      INPUT_NAME,
      tensors,
    );
    const logPolicy = outputs["log_policy"]; // shape [batch, COLS]
    const value = outputs["value"]; // shape [batch, 1]

    const out: { policy: Float32Array; value: number }[] = [];
    for (let i = 0; i < boards.length; i++) {
      const policy = new Float32Array(COLS);
      for (let c = 0; c < COLS; c++) {
        policy[c] = Math.exp(logPolicy[i * COLS + c]);
      }
      out.push({ policy, value: value[i] });
    }
    return out;
  }

  private expandWithPolicy(node: MCTSNode, policy: Float32Array): void {
    const validMoves = node.board.getValidMoves();
    if (validMoves.length === 0) return;
    const nextPlayer = otherPlayer(node.player);
    for (const col of validMoves) {
      const childBoard = node.board.copy();
      childBoard.dropPiece(col, node.player);
      const child = new MCTSNode(childBoard, nextPlayer, policy[col]);
      const winner = childBoard.checkWinner();
      if (winner !== null) {
        child.isTerminal = true;
        child.terminalValue = 1.0; // mover won
      } else if (childBoard.isDraw()) {
        child.isTerminal = true;
        child.terminalValue = 0.0;
      }
      node.children.set(col, child);
    }
  }

  private selectChild(node: MCTSNode): MCTSNode {
    let totalVisits = 0;
    for (const c of node.children.values()) totalVisits += c.visitCount;
    const sqrtTotal = Math.sqrt(totalVisits + 1);

    let bestScore = -Infinity;
    let bestChild: MCTSNode | null = null;
    // Map iteration order is insertion order, which matches Python dict
    // iteration order — so tie-breaking between equal-UCB children stays
    // consistent between the two implementations.
    for (const child of node.children.values()) {
      const ucb =
        child.qValue + this.cPuct * child.prior * sqrtTotal / (1 + child.visitCount);
      if (ucb > bestScore) {
        bestScore = ucb;
        bestChild = child;
      }
    }
    if (!bestChild) throw new Error("selectChild: no children");
    return bestChild;
  }

  private removeVirtualLoss(path: MCTSNode[]): void {
    // Skip root (index 0), matching Python convention.
    for (let i = 1; i < path.length; i++) {
      path[i].visitCount -= 1;
      path[i].valueSum += VIRTUAL_LOSS;
    }
  }

  private backpropagate(path: MCTSNode[], initialValue: number): void {
    let value = initialValue;
    for (let i = path.length - 1; i >= 0; i--) {
      path[i].valueSum += value;
      path[i].visitCount += 1;
      value = -value;
    }
  }
}
