import { describe, it, expect } from "vitest";
import { Board, ROWS, COLS, type Cell, type Player } from "../engine/board";
import { boardToTensor, maskInvalid, TENSOR_SIZE } from "./canonicalize";
import fixtures from "./__fixtures__/canonical.json";

interface Fixture {
  name: string;
  current_player: number;
  grid: number[]; // length 42, row-major, values in {0,1,2}
  tensor: number[]; // length 126, CHW flat
}

function boardFromGrid(grid: number[]): Board {
  const b = new Board();
  const internal = (b as unknown as { grid: Int8Array }).grid;
  for (let i = 0; i < ROWS * COLS; i++) {
    internal[i] = grid[i];
  }
  return b;
}

describe("boardToTensor — parity with Python board_to_tensor", () => {
  for (const raw of fixtures as Fixture[]) {
    it(`matches Python fixture: ${raw.name}`, () => {
      const board = boardFromGrid(raw.grid);
      const tensor = boardToTensor(board, raw.current_player as Player);
      expect(tensor.length).toBe(TENSOR_SIZE);
      expect(tensor.length).toBe(raw.tensor.length);
      for (let i = 0; i < TENSOR_SIZE; i++) {
        if (tensor[i] !== raw.tensor[i]) {
          throw new Error(
            `mismatch at index ${i}: ts=${tensor[i]} py=${raw.tensor[i]}`,
          );
        }
      }
    });
  }
});

describe("maskInvalid", () => {
  it("sets logits for full columns to -Infinity, preserves valid ones", () => {
    const b = new Board();
    for (let i = 0; i < ROWS; i++) b.dropPiece(0, 1); // fill column 0
    const logits = new Float32Array(COLS);
    for (let c = 0; c < COLS; c++) logits[c] = c + 1;
    maskInvalid(logits, b);
    expect(logits[0]).toBe(Number.NEGATIVE_INFINITY);
    for (let c = 1; c < COLS; c++) {
      expect(logits[c]).toBe(c + 1);
    }
  });
});

describe("canonicalize spot checks", () => {
  it("empty board: planes 0 and 1 are zero, plane 2 is all ones in top row", () => {
    const t = boardToTensor(new Board(), 1);
    for (let i = 0; i < 42; i++) expect(t[i]).toBe(0);
    for (let i = 42; i < 84; i++) expect(t[i]).toBe(0);
    // Plane 2 starts at offset 84. Top row of valid columns = indices 84..84+6.
    for (let c = 0; c < COLS; c++) expect(t[84 + c]).toBe(1);
    // Rest of plane 2 should be zero.
    for (let i = 84 + COLS; i < TENSOR_SIZE; i++) expect(t[i]).toBe(0);
  });

  it("swaps planes when perspective changes", () => {
    const b = new Board();
    b.dropPiece(3, 1 as Cell as Player);
    const fromP1 = boardToTensor(b, 1);
    const fromP2 = boardToTensor(b, 2);
    // From P1's view the piece is in plane 0; from P2's view it's in plane 1.
    const idx = (ROWS - 1) * COLS + 3;
    expect(fromP1[idx]).toBe(1);
    expect(fromP1[42 + idx]).toBe(0);
    expect(fromP2[idx]).toBe(0);
    expect(fromP2[42 + idx]).toBe(1);
  });
});
