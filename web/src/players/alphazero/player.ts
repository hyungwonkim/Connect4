import type { Board, Player } from "../../engine/board";
import type { BasePlayer } from "../base";
import type { NNSession, NNTensorConstructor } from "../nn";
import { MCTS, type MCTSOptions } from "./mcts";

export class AlphaZeroPlayer implements BasePlayer {
  readonly name: string;
  private readonly playerId: Player;
  private readonly mcts: MCTS;

  constructor(
    playerId: Player,
    session: NNSession,
    TensorCtor: NNTensorConstructor,
    options: MCTSOptions = {},
  ) {
    this.playerId = playerId;
    this.mcts = new MCTS(session, TensorCtor, options);
    this.name = `AlphaZero(${this.mcts.numSimulations} sims)`;
  }

  async chooseMove(board: Board): Promise<number> {
    const visits = await this.mcts.search(board, this.playerId);
    let best = -Infinity;
    let bestCol = 0;
    for (let c = 0; c < visits.length; c++) {
      if (visits[c] > best) {
        best = visits[c];
        bestCol = c;
      }
    }
    return bestCol;
  }
}
