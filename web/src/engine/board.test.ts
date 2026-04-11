import { describe, it, expect } from "vitest";
import { Board, P1, P2, COLS, ROWS } from "./board";

describe("Board.dropPiece", () => {
  it("stacks pieces from the bottom up", () => {
    const b = new Board();
    expect(b.dropPiece(3, P1)).toBe(ROWS - 1);
    expect(b.dropPiece(3, P2)).toBe(ROWS - 2);
    expect(b.dropPiece(3, P1)).toBe(ROWS - 3);
  });

  it("throws on invalid column", () => {
    const b = new Board();
    expect(() => b.dropPiece(-1, P1)).toThrow();
    expect(() => b.dropPiece(COLS, P1)).toThrow();
  });

  it("throws when column is full", () => {
    const b = new Board();
    for (let i = 0; i < ROWS; i++) b.dropPiece(0, P1);
    expect(b.isValidMove(0)).toBe(false);
    expect(() => b.dropPiece(0, P1)).toThrow();
  });
});

describe("Board.checkWinner", () => {
  it("detects horizontal win P1", () => {
    const b = new Board();
    for (let c = 0; c < 4; c++) b.dropPiece(c, P1);
    expect(b.checkWinner()).toBe(P1);
  });

  it("detects horizontal win P2 on the right edge", () => {
    const b = new Board();
    for (let c = 3; c < 7; c++) b.dropPiece(c, P2);
    expect(b.checkWinner()).toBe(P2);
  });

  it("detects vertical win", () => {
    const b = new Board();
    for (let i = 0; i < 4; i++) b.dropPiece(0, P1);
    expect(b.checkWinner()).toBe(P1);
  });

  it("detects diagonal down-right win (staircase left→right)", () => {
    const b = new Board();
    b.dropPiece(0, P1);
    b.dropPiece(1, P2);
    b.dropPiece(1, P1);
    b.dropPiece(2, P2);
    b.dropPiece(2, P2);
    b.dropPiece(2, P1);
    b.dropPiece(3, P2);
    b.dropPiece(3, P2);
    b.dropPiece(3, P2);
    b.dropPiece(3, P1);
    expect(b.checkWinner()).toBe(P1);
  });

  it("detects diagonal down-left win (staircase right→left)", () => {
    const b = new Board();
    b.dropPiece(6, P1);
    b.dropPiece(5, P2);
    b.dropPiece(5, P1);
    b.dropPiece(4, P2);
    b.dropPiece(4, P2);
    b.dropPiece(4, P1);
    b.dropPiece(3, P2);
    b.dropPiece(3, P2);
    b.dropPiece(3, P2);
    b.dropPiece(3, P1);
    expect(b.checkWinner()).toBe(P1);
  });

  it("returns null for empty board", () => {
    expect(new Board().checkWinner()).toBeNull();
  });

  it("returns null for three in a row", () => {
    const b = new Board();
    for (let c = 0; c < 3; c++) b.dropPiece(c, P1);
    expect(b.checkWinner()).toBeNull();
  });

  it("returns null when a run is broken by the opponent", () => {
    const b = new Board();
    b.dropPiece(0, P1);
    b.dropPiece(1, P1);
    b.dropPiece(2, P2);
    b.dropPiece(3, P1);
    b.dropPiece(4, P1);
    expect(b.checkWinner()).toBeNull();
  });

  it("does not attribute a P1 win to P2", () => {
    const b = new Board();
    for (let c = 0; c < 4; c++) b.dropPiece(c, P1);
    expect(b.checkWinner()).not.toBe(P2);
  });
});

describe("Board.isDraw", () => {
  it("is false on a new board", () => {
    expect(new Board().isDraw()).toBe(false);
  });

  it("is true on a full board with no winner", () => {
    const b = new Board();
    // Fill columns with a pattern that avoids any 4-in-a-row.
    // Column layout per row (from top row 0 to bottom row 5):
    // Using a known draw pattern: stagger players so no 4 line up.
    const pattern: number[][] = [
      [P1, P1, P1, P2, P2, P2, P1],
      [P2, P2, P2, P1, P1, P1, P2],
      [P1, P1, P1, P2, P2, P2, P1],
      [P2, P2, P2, P1, P1, P1, P2],
      [P1, P1, P1, P2, P2, P2, P1],
      [P2, P2, P2, P1, P1, P1, P2],
    ];
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        (b as unknown as { grid: Int8Array }).grid[r * COLS + c] = pattern[r][c];
      }
    }
    expect(b.checkWinner()).toBeNull();
    expect(b.getValidMoves().length).toBe(0);
    expect(b.isDraw()).toBe(true);
  });
});

describe("Board.copy", () => {
  it("produces an independent copy", () => {
    const b = new Board();
    b.dropPiece(3, P1);
    const c = b.copy();
    c.dropPiece(3, P2);
    expect(b.get(ROWS - 1, 3)).toBe(P1);
    expect(b.get(ROWS - 2, 3)).toBe(0);
    expect(c.get(ROWS - 2, 3)).toBe(P2);
  });
});
