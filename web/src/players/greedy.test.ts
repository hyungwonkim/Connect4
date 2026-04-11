import { describe, it, expect } from "vitest";
import { Board, P1, P2 } from "../engine/board";
import { GreedyPlayer } from "./greedy";
import { RandomPlayer } from "./random";

describe("RandomPlayer", () => {
  it("always picks a valid column", () => {
    const p = new RandomPlayer();
    const b = new Board();
    for (let i = 0; i < 20; i++) {
      const col = p.chooseMove(b) as number;
      expect(b.getValidMoves()).toContain(col);
    }
  });
});

describe("GreedyPlayer", () => {
  it("takes an immediate winning move", () => {
    const b = new Board();
    // P1 has three in a row on the bottom, col 3 is the winning drop.
    b.dropPiece(0, P1);
    b.dropPiece(1, P1);
    b.dropPiece(2, P1);
    // Put some P2 stones somewhere neutral.
    b.dropPiece(5, P2);
    b.dropPiece(5, P2);
    const p = new GreedyPlayer(P1);
    expect(p.chooseMove(b)).toBe(3);
  });

  it("blocks the opponent's immediate winning move", () => {
    const b = new Board();
    // P2 has three in a row, GreedyPlayer (P1) must block at col 3.
    b.dropPiece(0, P2);
    b.dropPiece(1, P2);
    b.dropPiece(2, P2);
    // Neutral P1 stones.
    b.dropPiece(5, P1);
    const p = new GreedyPlayer(P1);
    expect(p.chooseMove(b)).toBe(3);
  });

  it("prefers the winning move over blocking", () => {
    const b = new Board();
    // Both P1 and P2 have three in a row. P1 should win, not block.
    b.dropPiece(0, P1);
    b.dropPiece(0, P2);
    b.dropPiece(1, P1);
    b.dropPiece(1, P2);
    b.dropPiece(2, P1);
    b.dropPiece(2, P2);
    // Now bottom row: col 3 wins for P1; row above: col 3 wins for P2.
    const p = new GreedyPlayer(P1);
    expect(p.chooseMove(b)).toBe(3);
  });
});
