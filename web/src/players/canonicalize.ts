import { Board, COLS, ROWS, type Player } from "../engine/board";
import { otherPlayer } from "./base";

export const TENSOR_SIZE = 3 * ROWS * COLS; // 126

/**
 * Canonicalize a board to a (3, 6, 7) Float32Array in CHW order:
 *   plane 0: current player's pieces
 *   plane 1: opponent's pieces
 *   plane 2: valid-move indicator (1 in row 0 for each playable column)
 *
 * Must match connect4/players/rl/common.py::board_to_tensor byte-for-byte.
 */
export function boardToTensor(board: Board, currentPlayer: Player): Float32Array {
  const out = new Float32Array(TENSOR_SIZE);
  const plane = ROWS * COLS; // 42
  const opponent = otherPlayer(currentPlayer);

  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const cell = board.get(r, c);
      const idx = r * COLS + c;
      if (cell === currentPlayer) out[idx] = 1;
      else if (cell === opponent) out[plane + idx] = 1;
    }
  }

  // Plane 2: top row of each valid column = 1.0.
  for (const col of board.getValidMoves()) {
    out[2 * plane + col] = 1;
  }

  return out;
}

/**
 * In-place logit mask: set full-column logits to -Infinity before softmax/argmax.
 */
export function maskInvalid(logits: Float32Array, board: Board): Float32Array {
  const valid = new Set(board.getValidMoves());
  for (let c = 0; c < COLS; c++) {
    if (!valid.has(c)) logits[c] = Number.NEGATIVE_INFINITY;
  }
  return logits;
}
