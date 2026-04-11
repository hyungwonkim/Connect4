"""Export trained DQN, PPO, and AlphaZero checkpoints to ONNX.

Outputs are written to `web/models/` for the TypeScript frontend to consume
via onnxruntime-web.

Run:
    python scripts/export_onnx.py

Verifies each exported model by comparing PyTorch vs onnxruntime outputs on a
handful of random board positions; fails loudly if max abs diff exceeds 1e-4.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from connect4.board import Board
from connect4.players.rl.common import board_to_tensor
from connect4.players.rl.networks import AlphaZeroNetV2, DQNNet, PPONet

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS = REPO_ROOT / "checkpoints"
OUT_DIR = REPO_ROOT / "web" / "public" / "models"

# Max absolute difference tolerated between PyTorch and ONNX outputs.
TOL = 1e-4


def _random_board_tensors(n: int, seed: int = 0) -> torch.Tensor:
    """Play n random partial games to sample realistic board positions."""
    rng = np.random.default_rng(seed)
    tensors = []
    for _ in range(n):
        board = Board()
        num_moves = int(rng.integers(0, 20))
        current = 1
        for _ in range(num_moves):
            valid = board.get_valid_moves()
            if not valid or board.check_winner() is not None:
                break
            col = int(rng.choice(valid))
            board.drop_piece(col, current)
            current = 2 if current == 1 else 1
        tensors.append(board_to_tensor(board, current))
    return torch.stack(tensors)  # (n, 3, 6, 7)


def _export(
    name: str,
    model: torch.nn.Module,
    checkpoint_path: Path,
    output_names: list[str],
) -> Path:
    """Load checkpoint into model, export to ONNX, and return output path."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    out_path = OUT_DIR / f"{name}.onnx"
    dummy = torch.zeros(1, 3, 6, 7, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy,
        out_path.as_posix(),
        input_names=["board"],
        output_names=output_names,
        dynamic_axes={"board": {0: "batch"}, **{n: {0: "batch"} for n in output_names}},
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"  wrote {out_path.relative_to(REPO_ROOT)}  ({out_path.stat().st_size / 1024:.1f} KB)")
    return out_path


def _verify(name: str, model: torch.nn.Module, onnx_path: Path) -> None:
    """Compare PyTorch vs ONNX outputs on 10 sampled board positions."""
    model.eval()
    boards = _random_board_tensors(10)

    with torch.no_grad():
        torch_out = model(boards)
    if not isinstance(torch_out, tuple):
        torch_out = (torch_out,)

    sess = ort.InferenceSession(onnx_path.as_posix(), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"board": boards.numpy()})

    assert len(torch_out) == len(onnx_out), (
        f"{name}: expected {len(torch_out)} outputs, got {len(onnx_out)}"
    )
    for i, (t, o) in enumerate(zip(torch_out, onnx_out)):
        diff = np.max(np.abs(t.numpy() - o))
        status = "OK" if diff < TOL else "FAIL"
        print(f"  [{status}] output[{i}] max abs diff = {diff:.2e}")
        if diff >= TOL:
            raise AssertionError(
                f"{name}: output[{i}] diff {diff:.2e} exceeds tolerance {TOL:.0e}"
            )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    targets = [
        ("dqn",        DQNNet(),          CHECKPOINTS / "dqn" / "best.pt",        ["q_values"]),
        ("ppo",        PPONet(),          CHECKPOINTS / "ppo" / "best.pt",        ["policy_logits", "value"]),
        ("alphazero",  AlphaZeroNetV2(channels=96, num_blocks=5),  CHECKPOINTS / "alphazero" / "best.pt",  ["log_policy", "value"]),
    ]

    for name, model, ckpt, outputs in targets:
        print(f"\n[{name}] exporting from {ckpt.relative_to(REPO_ROOT)}")
        onnx_path = _export(name, model, ckpt, outputs)
        print(f"[{name}] verifying PyTorch ↔ ONNX parity")
        _verify(name, model, onnx_path)

    print(f"\nAll models exported to {OUT_DIR.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
