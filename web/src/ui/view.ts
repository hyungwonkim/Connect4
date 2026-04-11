import { Board, COLS, ROWS, type Player } from "../engine/board";

export type ChipColor = "yellow" | "red";
export type ColorMap = Record<Player, ChipColor>;

export const DEFAULT_COLORS: ColorMap = { 1: "yellow", 2: "red" };

export type OnColumnClick = (col: number) => void;

export class BoardView {
  private cellEls: HTMLDivElement[] = [];
  private columnEls: HTMLDivElement[] = [];
  private statusEl: HTMLDivElement;
  private restartEl: HTMLButtonElement;
  private onClick: OnColumnClick = () => {};
  private clicksEnabled = true;
  private colors: ColorMap;

  constructor(private root: HTMLElement, colors: ColorMap = DEFAULT_COLORS) {
    this.colors = colors;
    root.innerHTML = "";
    const wrapper = document.createElement("div");
    wrapper.className = "game-wrapper";

    this.statusEl = document.createElement("div");
    this.statusEl.className = "status";
    wrapper.appendChild(this.statusEl);

    const boardEl = document.createElement("div");
    boardEl.className = "board";
    boardEl.style.gridTemplateColumns = `repeat(${COLS}, 1fr)`;
    for (let c = 0; c < COLS; c++) {
      const col = document.createElement("div");
      col.className = "column";
      col.dataset.col = String(c);
      col.addEventListener("click", () => {
        if (this.clicksEnabled) this.onClick(c);
      });
      for (let r = 0; r < ROWS; r++) {
        const cell = document.createElement("div");
        cell.className = "cell";
        this.cellEls[r * COLS + c] = cell;
        col.appendChild(cell);
      }
      this.columnEls[c] = col;
      boardEl.appendChild(col);
    }
    wrapper.appendChild(boardEl);

    this.restartEl = document.createElement("button");
    this.restartEl.className = "restart";
    this.restartEl.textContent = "New game";
    this.restartEl.style.display = "none";
    wrapper.appendChild(this.restartEl);

    root.appendChild(wrapper);
  }

  setOnColumnClick(fn: OnColumnClick): void {
    this.onClick = fn;
  }

  setOnRestart(fn: () => void): void {
    this.restartEl.onclick = fn;
  }

  setClicksEnabled(enabled: boolean): void {
    this.clicksEnabled = enabled;
    this.root.classList.toggle("clickable", enabled);
  }

  showRestart(show: boolean): void {
    this.restartEl.style.display = show ? "inline-block" : "none";
  }

  setStatus(message: string, kind: "info" | "win" | "draw" | "thinking" = "info"): void {
    this.statusEl.textContent = message;
    this.statusEl.className = `status status-${kind}`;
  }

  colorFor(player: Player): ChipColor {
    return this.colors[player];
  }

  render(board: Board): void {
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        const cell = this.cellEls[r * COLS + c];
        const value = board.get(r, c);
        cell.classList.remove("yellow", "red", "drop");
        if (value === 1 || value === 2) {
          cell.classList.add(this.colors[value]);
        }
      }
    }
  }

  async animateDrop(
    board: Board,
    col: number,
    targetRow: number,
    player: Player,
  ): Promise<void> {
    // Update the board render first, then apply a CSS class on the landed
    // cell for a quick drop animation.
    this.render(board);
    const cell = this.cellEls[targetRow * COLS + col];
    cell.classList.add("drop");
    await new Promise<void>((resolve) => setTimeout(resolve, 260));
    void player;
  }
}
