import type { Board } from "../engine/board";
import type { BasePlayer } from "./base";

export class RandomPlayer implements BasePlayer {
  readonly name = "Random";

  chooseMove(board: Board): number {
    const valid = board.getValidMoves();
    return valid[Math.floor(Math.random() * valid.length)];
  }
}
