import type { Board, Player } from "../engine/board";

export interface BasePlayer {
  readonly name: string;
  chooseMove(board: Board): Promise<number> | number;
  reset?(): void;
}

export function otherPlayer(p: Player): Player {
  return p === 1 ? 2 : 1;
}
