"""Load model weights from .safetensors (HF distribution) or training .pt."""

from __future__ import annotations

from pathlib import Path
import hashlib

import torch


def initialize_training_weights(model, path: Path) -> dict:
    """Strictly initialize a fresh experiment from inference-only tensors.

    No optimizer, epoch or best score is restored. The input file is read only.
    """
    from safetensors import safe_open
    from safetensors.torch import load_file

    path = Path(path).resolve()
    if path.suffix.lower() != ".safetensors":
        raise ValueError("Initial weights must be a .safetensors file; use --resume for training .pt")
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    state = load_file(str(path), device="cpu")
    expected = model.state_dict()
    missing, extra = sorted(expected.keys() - state.keys()), sorted(state.keys() - expected.keys())
    if missing or extra:
        raise ValueError(f"Initial weight keys differ: missing={missing[:8]}, extra={extra[:8]}")
    for name, value in state.items():
        target = expected[name]
        if value.shape != target.shape or value.dtype != target.dtype:
            raise ValueError(f"Initial tensor does not match model: {name} "
                             f"{tuple(value.shape)}/{value.dtype} vs {tuple(target.shape)}/{target.dtype}")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"Initial tensor contains non-finite values: {name}")
    model.load_state_dict(state, strict=True)
    with safe_open(str(path), framework="pt", device="cpu") as weights:
        metadata = weights.metadata() or {}
    return {"source": str(path), "sha256": digest, "tensor_count": len(state),
            "source_metadata": metadata, "optimizer_restored": False,
            "epoch_restored": False, "best_score_restored": False,
            "parameter_count": sum(p.numel() for p in model.parameters())}


def load_inference_checkpoint(
    path: Path, device: torch.device
) -> tuple[dict[str, torch.Tensor], int | None]:
    """Return (state_dict, epoch). Optimizer state in .pt files is ignored."""
    if path.suffix == ".safetensors":
        from safetensors import safe_open
        from safetensors.torch import load_file

        # safetensors load_file accepts "cpu" / "cuda"; for mps we load on cpu
        # and let load_state_dict move tensors to the model's device.
        target = "cuda" if device.type == "cuda" else "cpu"
        state = load_file(str(path), device=target)
        with safe_open(str(path), framework="pt") as f:
            meta = f.metadata() or {}
        epoch = int(meta["epoch"]) if meta.get("epoch") else None
        return state, epoch

    ckpt = torch.load(path, map_location=device, weights_only=False)
    return ckpt["model"], ckpt.get("epoch")
