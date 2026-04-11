import { COLS, ROWS } from "../engine/board";

/**
 * Minimal runtime-agnostic interface that both `onnxruntime-web` and
 * `onnxruntime-node` satisfy. The players never import either package
 * directly — the caller is responsible for creating the session and
 * passing it in. This keeps test (Node) and prod (browser) isolated.
 */
export interface NNSession {
  run(feeds: Record<string, NNTensor>): Promise<Record<string, NNTensor>>;
}

export interface NNTensor {
  readonly data: Float32Array | Float64Array | Int32Array | BigInt64Array;
  readonly dims: readonly number[];
}

export interface NNTensorConstructor {
  new (type: "float32", data: Float32Array, dims: readonly number[]): NNTensor;
}

/**
 * Run a single-sample forward pass. `boardTensor` must be a Float32Array of
 * length 3*ROWS*COLS (output of `boardToTensor`). Returns a map from the
 * model's declared output names to Float32Array payloads.
 *
 * The `TensorCtor` is injected so that we don't import any specific
 * onnxruntime package here; the caller (browser or Node test) provides the
 * right constructor from the package it already loaded.
 */
export async function runInference(
  session: NNSession,
  TensorCtor: NNTensorConstructor,
  inputName: string,
  boardTensor: Float32Array,
): Promise<Record<string, Float32Array>> {
  if (boardTensor.length !== 3 * ROWS * COLS) {
    throw new Error(
      `runInference: expected ${3 * ROWS * COLS} floats, got ${boardTensor.length}`,
    );
  }
  const input = new TensorCtor("float32", boardTensor, [1, 3, ROWS, COLS]);
  const outputs = await session.run({ [inputName]: input });
  const result: Record<string, Float32Array> = {};
  for (const [name, tensor] of Object.entries(outputs)) {
    result[name] = tensor.data as Float32Array;
  }
  return result;
}

/**
 * Run a batched forward pass. `boardTensors` is a list of per-sample
 * canonicalized tensors (each length 3*ROWS*COLS). Returns a map from
 * output name to Float32Array of length `batch * outputSize`.
 */
export async function runInferenceBatch(
  session: NNSession,
  TensorCtor: NNTensorConstructor,
  inputName: string,
  boardTensors: Float32Array[],
): Promise<Record<string, Float32Array>> {
  const batch = boardTensors.length;
  if (batch === 0) return {};
  const stride = 3 * ROWS * COLS;
  const flat = new Float32Array(batch * stride);
  for (let i = 0; i < batch; i++) {
    if (boardTensors[i].length !== stride) {
      throw new Error(`runInferenceBatch: sample ${i} has wrong length`);
    }
    flat.set(boardTensors[i], i * stride);
  }
  const input = new TensorCtor("float32", flat, [batch, 3, ROWS, COLS]);
  const outputs = await session.run({ [inputName]: input });
  const result: Record<string, Float32Array> = {};
  for (const [name, tensor] of Object.entries(outputs)) {
    result[name] = tensor.data as Float32Array;
  }
  return result;
}

/**
 * argmax with NEGATIVE_INFINITY skipping. Ties are broken by first-index
 * (matching PyTorch's argmax default).
 */
export function argmax(values: Float32Array): number {
  let best = -Infinity;
  let idx = 0;
  for (let i = 0; i < values.length; i++) {
    if (values[i] > best) {
      best = values[i];
      idx = i;
    }
  }
  return idx;
}
