import { Board, COLS, ROWS, type Player } from "../engine/board";
import { otherPlayer, type BasePlayer } from "./base";

const DIRECTIONS: ReadonlyArray<[number, number]> = [
  [0, 1], // horizontal
  [1, 0], // vertical
  [1, 1], // diagonal down-right
  [1, -1], // diagonal down-left
];

function longestSequence(
  board: Board,
  row: number,
  col: number,
  player: Player,
): number {
  let best = 1;
  for (const [dr, dc] of DIRECTIONS) {
    let count = 1;
    for (const sign of [1, -1]) {
      let r = row + dr * sign;
      let c = col + dc * sign;
      while (r >= 0 && r < ROWS && c >= 0 && c < COLS && board.get(r, c) === player) {
        count++;
        r += dr * sign;
        c += dc * sign;
      }
    }
    if (count > best) best = count;
  }
  return best;
}

export class GreedyPlayer implements BasePlayer {
  readonly name: string = "Greedy";
  protected readonly playerId: Player;
  protected readonly opponentId: Player;

  constructor(playerId: Player) {
    this.playerId = playerId;
    this.opponentId = otherPlayer(playerId);
  }

  chooseMove(board: Board): number {
    const valid = board.getValidMoves();

    // Win immediately if possible.
    for (const col of valid) {
      const sim = board.copy();
      sim.dropPiece(col, this.playerId);
      if (sim.checkWinner() === this.playerId) return col;
    }

    // Block the opponent's immediate win.
    for (const col of valid) {
      const sim = board.copy();
      sim.dropPiece(col, this.opponentId);
      if (sim.checkWinner() === this.opponentId) return col;
    }

    // Extend the longest existing sequence.
    let bestLen = 0;
    let bestCols: number[] = [];
    for (const col of valid) {
      const sim = board.copy();
      const row = sim.dropPiece(col, this.playerId);
      const seq = longestSequence(sim, row, col, this.playerId);
      if (seq > bestLen) {
        bestLen = seq;
        bestCols = [col];
      } else if (seq === bestLen) {
        bestCols.push(col);
      }
    }
    return bestCols[Math.floor(Math.random() * bestCols.length)];
  }
}

export class EpsilonGreedyPlayer extends GreedyPlayer {
  readonly name: string;
  private readonly epsilon: number;

  constructor(playerId: Player, epsilon = 0.1) {
    super(playerId);
    this.epsilon = epsilon;
    this.name = `EpsilonGreedy(ε=${epsilon})`;
  }

  override chooseMove(board: Board): number {
    if (Math.random() < this.epsilon) {
      const valid = board.getValidMoves();
      return valid[Math.floor(Math.random() * valid.length)];
    }
    return super.chooseMove(board);
  }
}
