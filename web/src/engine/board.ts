export const ROWS = 6;
export const COLS = 7;
export const EMPTY = 0;
export const P1 = 1;
export const P2 = 2;

export type Cell = 0 | 1 | 2;
export type Player = 1 | 2;
export type Winner = Player | null;

export class Board {
  readonly grid: Int8Array;

  constructor(grid?: Int8Array) {
    this.grid = grid ?? new Int8Array(ROWS * COLS);
  }

  private at(row: number, col: number): number {
    return this.grid[row * COLS + col];
  }

  private set(row: number, col: number, value: number): void {
    this.grid[row * COLS + col] = value;
  }

  isValidMove(col: number): boolean {
    return col >= 0 && col < COLS && this.at(0, col) === EMPTY;
  }

  getValidMoves(): number[] {
    const moves: number[] = [];
    for (let c = 0; c < COLS; c++) {
      if (this.isValidMove(c)) moves.push(c);
    }
    return moves;
  }

  dropPiece(col: number, player: Player): number {
    if (!this.isValidMove(col)) {
      throw new Error(`Invalid move: column ${col}`);
    }
    for (let row = ROWS - 1; row >= 0; row--) {
      if (this.at(row, col) === EMPTY) {
        this.set(row, col, player);
        return row;
      }
    }
    throw new Error(`Column ${col} is full`);
  }

  checkWinner(): Winner {
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        const p = this.at(r, c);
        if (p === EMPTY) continue;
        // horizontal
        if (
          c + 3 < COLS &&
          this.at(r, c + 1) === p &&
          this.at(r, c + 2) === p &&
          this.at(r, c + 3) === p
        ) {
          return p as Player;
        }
        // vertical
        if (
          r + 3 < ROWS &&
          this.at(r + 1, c) === p &&
          this.at(r + 2, c) === p &&
          this.at(r + 3, c) === p
        ) {
          return p as Player;
        }
        // diagonal down-right
        if (
          r + 3 < ROWS &&
          c + 3 < COLS &&
          this.at(r + 1, c + 1) === p &&
          this.at(r + 2, c + 2) === p &&
          this.at(r + 3, c + 3) === p
        ) {
          return p as Player;
        }
        // diagonal down-left
        if (
          r + 3 < ROWS &&
          c - 3 >= 0 &&
          this.at(r + 1, c - 1) === p &&
          this.at(r + 2, c - 2) === p &&
          this.at(r + 3, c - 3) === p
        ) {
          return p as Player;
        }
      }
    }
    return null;
  }

  isDraw(): boolean {
    return this.checkWinner() === null && this.getValidMoves().length === 0;
  }

  copy(): Board {
    return new Board(new Int8Array(this.grid));
  }

  get(row: number, col: number): Cell {
    return this.at(row, col) as Cell;
  }

  toString(): string {
    const symbols = [".", "X", "O"];
    const lines: string[] = [];
    for (let r = 0; r < ROWS; r++) {
      const cells: string[] = [];
      for (let c = 0; c < COLS; c++) {
        cells.push(symbols[this.at(r, c)]);
      }
      lines.push(cells.join(" "));
    }
    const header = Array.from({ length: COLS }, (_, i) => String(i)).join(" ");
    return lines.join("\n") + "\n" + header;
  }
}
