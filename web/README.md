# Connect 4 — Web

Static Vite + TypeScript app that plays Connect 4 against the RL agents in-browser.
The PyTorch checkpoints are exported to ONNX and run with `onnxruntime-web` (WASM backend, loaded
from CDN). AlphaZero's MCTS is ported to TypeScript and runs entirely client-side.

## Dev

```bash
cd web
npm install
npm run dev       # http://localhost:5173
npm test          # vitest (engine + canonicalization + MCTS parity)
npm run build     # writes to ../docs/play/
```

## ONNX export

Before building, the models in `public/models/` must exist. Regenerate from the latest Python
checkpoints with:

```bash
# from repo root
python scripts/export_onnx.py
```

This writes `dqn.onnx`, `ppo.onnx`, and `alphazero.onnx` into `web/public/models/`, and verifies
each exported model against the PyTorch version on random board positions (max abs diff < 1e-4).

## Deploy

`npm run build` outputs to `../docs/play/`, which GitHub Pages serves directly. Commit the built
artifacts so no CI is needed.
