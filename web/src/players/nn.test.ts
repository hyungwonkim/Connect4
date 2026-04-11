import { describe, it, expect, beforeAll } from "vitest";
import * as path from "node:path";
import * as fs from "node:fs";
import * as ort from "onnxruntime-node";

import { Board, ROWS, COLS, type Player } from "../engine/board";
import { DQNPlayer } from "./dqn";
import { PPOPlayer } from "./ppo";
import { boardToTensor, maskInvalid } from "./canonicalize";
import { argmax } from "./nn";
import fixtures from "./__fixtures__/nn_outputs.json";

interface Fixture {
  name: string;
  current_player: number;
  grid: number[];
  dqn_q_values: number[];
  dqn_chosen_move: number;
  ppo_logits: number[];
  ppo_chosen_move: number;
}

// Point at the exported ONNX files produced by scripts/export_onnx.py.
const REPO_ROOT = path.resolve(__dirname, "../../..");
const DQN_PATH = path.join(REPO_ROOT, "web", "public", "models", "dqn.onnx");
const PPO_PATH = path.join(REPO_ROOT, "web", "public", "models", "ppo.onnx");

function boardFromGrid(grid: number[]): Board {
  const b = new Board();
  const internal = (b as unknown as { grid: Int8Array }).grid;
  for (let i = 0; i < ROWS * COLS; i++) internal[i] = grid[i];
  return b;
}

describe("DQN + PPO TS inference ↔ Python parity", () => {
  let dqnSession: ort.InferenceSession;
  let ppoSession: ort.InferenceSession;

  beforeAll(async () => {
    if (!fs.existsSync(DQN_PATH) || !fs.existsSync(PPO_PATH)) {
      throw new Error(
        `Missing ONNX models. Run: python scripts/export_onnx.py`,
      );
    }
    dqnSession = await ort.InferenceSession.create(DQN_PATH);
    ppoSession = await ort.InferenceSession.create(PPO_PATH);
  });

  for (const raw of fixtures as Fixture[]) {
    it(`DQN matches Python on ${raw.name}`, async () => {
      const board = boardFromGrid(raw.grid);
      const player = new DQNPlayer(
        raw.current_player as Player,
        dqnSession as unknown as import("./nn").NNSession,
        ort.Tensor as unknown as import("./nn").NNTensorConstructor,
      );
      const chosen = await player.chooseMove(board);
      expect(chosen).toBe(raw.dqn_chosen_move);

      // Also sanity-check raw Q-values: max abs diff to Python should be tiny.
      const input = boardToTensor(board, raw.current_player as Player);
      const feeds = {
        board: new ort.Tensor("float32", input, [1, 3, ROWS, COLS]),
      };
      const out = await dqnSession.run(feeds);
      const q = out["q_values"].data as Float32Array;
      let maxDiff = 0;
      for (let i = 0; i < COLS; i++) {
        maxDiff = Math.max(maxDiff, Math.abs(q[i] - raw.dqn_q_values[i]));
      }
      expect(maxDiff).toBeLessThan(1e-4);
    });

    it(`PPO matches Python on ${raw.name}`, async () => {
      const board = boardFromGrid(raw.grid);
      const player = new PPOPlayer(
        raw.current_player as Player,
        ppoSession as unknown as import("./nn").NNSession,
        ort.Tensor as unknown as import("./nn").NNTensorConstructor,
      );
      const chosen = await player.chooseMove(board);
      expect(chosen).toBe(raw.ppo_chosen_move);

      const input = boardToTensor(board, raw.current_player as Player);
      const feeds = {
        board: new ort.Tensor("float32", input, [1, 3, ROWS, COLS]),
      };
      const out = await ppoSession.run(feeds);
      const logits = out["policy_logits"].data as Float32Array;
      let maxDiff = 0;
      for (let i = 0; i < COLS; i++) {
        maxDiff = Math.max(maxDiff, Math.abs(logits[i] - raw.ppo_logits[i]));
      }
      expect(maxDiff).toBeLessThan(1e-4);
    });
  }
});

describe("argmax tie-breaking matches PyTorch first-index convention", () => {
  it("returns lowest index on ties", () => {
    const v = new Float32Array([1, 1, 1, 0, 0, 0, 0]);
    const b = new Board();
    maskInvalid(v, b);
    // all valid, all tied at 1 for first three; lowest index = 0
    expect(argmax(v)).toBe(0);
  });
});
