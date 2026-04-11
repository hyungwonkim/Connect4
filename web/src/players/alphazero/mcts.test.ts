import { describe, it, expect, beforeAll } from "vitest";
import * as path from "node:path";
import * as fs from "node:fs";
import * as ort from "onnxruntime-node";

import { Board, ROWS, COLS, P1, P2, type Player } from "../../engine/board";
import type { NNSession, NNTensorConstructor } from "../nn";
import { MCTS } from "./mcts";
import { AlphaZeroPlayer } from "./player";
import fixtures from "../__fixtures__/mcts_positions.json";

interface MCTSFixture {
  name: string;
  current_player: number;
  grid: number[];
  num_simulations: number;
  visits: number[];
  chosen_move: number;
}

const REPO_ROOT = path.resolve(__dirname, "../../../..");
const ALPHAZERO_PATH = path.join(REPO_ROOT, "web", "public", "models", "alphazero.onnx");

function boardFromGrid(grid: number[]): Board {
  const b = new Board();
  const internal = (b as unknown as { grid: Int8Array }).grid;
  for (let i = 0; i < ROWS * COLS; i++) internal[i] = grid[i];
  return b;
}

describe("AlphaZero MCTS — behavioral + parity", () => {
  let session: NNSession;
  let TensorCtor: NNTensorConstructor;

  beforeAll(async () => {
    if (!fs.existsSync(ALPHAZERO_PATH)) {
      throw new Error("Missing alphazero.onnx — run scripts/export_onnx.py");
    }
    const s = await ort.InferenceSession.create(ALPHAZERO_PATH);
    session = s as unknown as NNSession;
    TensorCtor = ort.Tensor as unknown as NNTensorConstructor;
  });

  it("returns a distribution summing to ~1", async () => {
    const mcts = new MCTS(session, TensorCtor, { numSimulations: 50 });
    const visits = await mcts.search(new Board(), P1);
    expect(visits.length).toBe(COLS);
    const total = visits.reduce((a, b) => a + b, 0);
    expect(total).toBeCloseTo(1, 5);
  });

  it("avoids full columns", async () => {
    const mcts = new MCTS(session, TensorCtor, { numSimulations: 50 });
    const b = new Board();
    // Fill column 3.
    for (let i = 0; i < 3; i++) {
      b.dropPiece(3, P1);
      b.dropPiece(3, P2);
    }
    const visits = await mcts.search(b, P1);
    expect(visits[3]).toBe(0);
  });

  it("finds an obvious winning move", async () => {
    const player = new AlphaZeroPlayer(P1, session, TensorCtor, { numSimulations: 100 });
    const b = new Board();
    b.dropPiece(0, P1); b.dropPiece(0, P2);
    b.dropPiece(1, P1); b.dropPiece(1, P2);
    b.dropPiece(2, P1); b.dropPiece(2, P2);
    expect(await player.chooseMove(b)).toBe(3);
  });

  it("blocks an obvious opponent winning move", async () => {
    const player = new AlphaZeroPlayer(P1, session, TensorCtor, { numSimulations: 100 });
    const b = new Board();
    b.dropPiece(0, P2); b.dropPiece(0, P1);
    b.dropPiece(1, P2); b.dropPiece(1, P1);
    b.dropPiece(2, P2); b.dropPiece(6, P1);
    expect(await player.chooseMove(b)).toBe(3);
  });

  it("top-move parity with Python on ≥90% of fixtures", async () => {
    let matched = 0;
    const mismatches: string[] = [];
    for (const raw of fixtures as MCTSFixture[]) {
      const board = boardFromGrid(raw.grid);
      const player = new AlphaZeroPlayer(
        raw.current_player as Player,
        session,
        TensorCtor,
        { numSimulations: raw.num_simulations },
      );
      const chosen = await player.chooseMove(board);
      if (chosen === raw.chosen_move) {
        matched++;
      } else {
        mismatches.push(
          `${raw.name}: ts=${chosen} py=${raw.chosen_move}`,
        );
      }
    }
    const rate = matched / fixtures.length;
    if (mismatches.length > 0) {
      console.log(`MCTS parity mismatches (${mismatches.length}/${fixtures.length}):`);
      for (const m of mismatches) console.log(`  ${m}`);
    }
    expect(rate).toBeGreaterThanOrEqual(0.9);
  }, 60_000);
});
