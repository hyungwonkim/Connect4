import type { Board, Player } from "../engine/board";
import type { BasePlayer } from "./base";
import { boardToTensor, maskInvalid } from "./canonicalize";
import {
  argmax,
  runInference,
  type NNSession,
  type NNTensorConstructor,
} from "./nn";

export class DQNPlayer implements BasePlayer {
  readonly name = "DQN";
  private readonly playerId: Player;
  private readonly session: NNSession;
  private readonly TensorCtor: NNTensorConstructor;

  constructor(playerId: Player, session: NNSession, TensorCtor: NNTensorConstructor) {
    this.playerId = playerId;
    this.session = session;
    this.TensorCtor = TensorCtor;
  }

  async chooseMove(board: Board): Promise<number> {
    const input = boardToTensor(board, this.playerId);
    const outputs = await runInference(this.session, this.TensorCtor, "board", input);
    const q = outputs["q_values"];
    maskInvalid(q, board);
    return argmax(q);
  }
}
