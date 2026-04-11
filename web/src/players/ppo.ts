import type { Board, Player } from "../engine/board";
import type { BasePlayer } from "./base";
import { boardToTensor, maskInvalid } from "./canonicalize";
import {
  argmax,
  runInference,
  type NNSession,
  type NNTensorConstructor,
} from "./nn";

export class PPOPlayer implements BasePlayer {
  readonly name = "PPO";
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
    const logits = outputs["policy_logits"];
    maskInvalid(logits, board);
    return argmax(logits);
  }
}
