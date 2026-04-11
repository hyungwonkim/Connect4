import type { NNSession, NNTensorConstructor } from "../players/nn";
import type { SessionLoader } from "./opponents";

/**
 * onnxruntime-web is loaded via a <script> tag in index.html (from the
 * jsDelivr CDN) to keep the GitHub Pages bundle small (otherwise Vite
 * copies a ~25MB WASM file into the build output). This module reaches
 * the library off of `window.ort`.
 */
declare global {
  interface Window {
    ort: {
      InferenceSession: {
        create(
          uri: string,
          options?: { executionProviders?: string[] },
        ): Promise<{
          run(feeds: Record<string, unknown>): Promise<Record<string, { data: unknown; dims: readonly number[] }>>;
        }>;
      };
      Tensor: new (
        type: string,
        data: Float32Array | Float64Array | Int32Array,
        dims: readonly number[],
      ) => { data: unknown; dims: readonly number[] };
      env: { wasm: { wasmPaths: string } };
    };
  }
}

type ModelName = "dqn" | "ppo" | "alphazero";

/**
 * Lazy, cached ONNX session loader. First call per model fetches and
 * initializes; subsequent calls return the cached session.
 */
export class OrtLoader implements SessionLoader {
  private cache = new Map<ModelName, Promise<unknown>>();

  constructor(private readonly baseUrl: string) {
    // Tell onnxruntime-web to fetch its WASM assets from the same CDN
    // version as the ort.min.js loaded in index.html.
    if (window.ort) {
      window.ort.env.wasm.wasmPaths =
        "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/";
    }
  }

  async load(modelName: ModelName): Promise<{
    session: NNSession;
    TensorCtor: NNTensorConstructor;
  }> {
    if (!window.ort) {
      throw new Error(
        "onnxruntime-web failed to load from CDN — check your network.",
      );
    }
    if (!this.cache.has(modelName)) {
      const url = `${this.baseUrl}models/${modelName}.onnx`;
      this.cache.set(modelName, window.ort.InferenceSession.create(url));
    }
    const session = await this.cache.get(modelName)!;
    return {
      session: session as NNSession,
      TensorCtor: window.ort.Tensor as unknown as NNTensorConstructor,
    };
  }
}
