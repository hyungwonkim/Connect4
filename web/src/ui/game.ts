import { Board, P1, P2, type Player } from "../engine/board";
import type { BasePlayer } from "../players/base";
import { BoardView } from "./view";

/**
 * Drives a single game between two players and routes updates to the view.
 * `null` for a side means "this side is controlled by the human via clicks".
 */
export class GameController {
  private board: Board;
  private current: Player = P1;
  private finished = false;
  private pendingHumanResolve: ((col: number) => void) | null = null;

  constructor(
    private readonly view: BoardView,
    private readonly players: { 1: BasePlayer | null; 2: BasePlayer | null },
  ) {
    this.board = new Board();
    this.view.setOnColumnClick((col) => this.handleHumanClick(col));
  }

  async run(): Promise<void> {
    this.view.render(this.board);
    while (!this.finished) {
      const player = this.players[this.current];
      if (player === null) {
        const color = this.view.colorFor(this.current);
        const label = color === "yellow" ? "Yellow" : "Red";
        this.view.setStatus(
          `${label}'s turn — click a column`,
          "info",
        );
        this.view.setClicksEnabled(true);
        const col = await this.waitForHumanMove();
        if (this.finished) return;
        await this.applyMove(col);
      } else {
        this.view.setClicksEnabled(false);
        this.view.setStatus(`${player.name} is thinking…`, "thinking");
        // Yield a frame so the status update paints before potential heavy work.
        await new Promise((r) => setTimeout(r, 16));
        const col = await player.chooseMove(this.board);
        await this.applyMove(col);
      }
    }
  }

  private waitForHumanMove(): Promise<number> {
    return new Promise((resolve) => {
      this.pendingHumanResolve = resolve;
    });
  }

  private handleHumanClick(col: number): void {
    if (!this.pendingHumanResolve) return;
    if (!this.board.isValidMove(col)) return;
    const resolve = this.pendingHumanResolve;
    this.pendingHumanResolve = null;
    this.view.setClicksEnabled(false);
    resolve(col);
  }

  private async applyMove(col: number): Promise<void> {
    const row = this.board.dropPiece(col, this.current);
    await this.view.animateDrop(this.board, col, row, this.current);

    const winner = this.board.checkWinner();
    if (winner !== null) {
      this.finished = true;
      const label = this.view.colorFor(winner) === "yellow" ? "Yellow" : "Red";
      this.view.setStatus(`${label} wins!`, "win");
      this.view.showRestart(true);
      return;
    }
    if (this.board.isDraw()) {
      this.finished = true;
      this.view.setStatus("Draw.", "draw");
      this.view.showRestart(true);
      return;
    }
    this.current = this.current === P1 ? P2 : P1;
  }

  cancel(): void {
    this.finished = true;
    if (this.pendingHumanResolve) {
      this.pendingHumanResolve(-1);
      this.pendingHumanResolve = null;
    }
  }
}
