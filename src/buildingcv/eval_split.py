"""Evaluate a trained checkpoint on a held-out split, reusing train.evaluate so numbers match training-time val."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import ConcatDataset, DataLoader, Subset

from .checkpoint import load_inference_checkpoint
from .data import CubiCasaDataset
from .labels import CLASS_NAMES, IGNORE_ID, NUM_CLASSES
from .model import build_model
from .train import StrictSvgDataset, evaluate, format_iou, resolve_device


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def evaluation_dataset(cfg, split, source="all", limit=None):
    """Use configured sources, but always fail on an unreadable SVG."""
    if split not in ("train", "val", "test") or source not in ("all", "svg", "raster"):
        raise ValueError("Unknown evaluation split/source")
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("Evaluation limit must be a positive integer")
    data = cfg["data"]
    kwargs = dict(size=tuple(data["image_size"]), normalize=data["normalize"],
                  letterbox=data.get("letterbox", False))
    parts, receipts = [], []
    if source in ("all", "svg") and data.get("include_cubicasa", True):
        svg = CubiCasaDataset(data["data_dir"], split, split_dir=data.get("split_dir"), **kwargs)
        names = svg.samples[:limit] if limit is not None else svg.samples
        sample_hashes = [{"sample": name, "svg_sha256": sha256(svg.data_dir / name / "model.svg")}
                         for name in names]
        strict = StrictSvgDataset(svg)
        parts.append(Subset(strict, range(len(names))) if limit is not None else strict)
        split_file = Path(data.get("split_dir") or data["data_dir"]) / f"{split}.txt"
        receipts.append({"source": "svg", "count": len(names), "samples": sample_hashes,
                         "split_sha256": sha256(split_file), "strict": True})
    if source in ("all", "raster") and data.get("raster_manifest"):
        from .raster_data import RasterMaskDataset
        raster = RasterMaskDataset(data["raster_manifest"], split,
                                   root=data.get("raster_root", "."),
                                   allow_partial=data.get("allow_partial_annotations", False), **kwargs)
        rows = raster.samples[:limit] if limit is not None else raster.samples
        parts.append(Subset(raster, range(len(rows))) if limit is not None else raster)
        receipts.append({"source": "raster", "count": len(rows), "samples": rows,
                         "manifest_sha256": raster.manifest_sha256})
    if not parts or not sum(len(part) for part in parts):
        raise ValueError("No samples for the requested evaluation source/split")
    return (parts[0] if len(parts) == 1 else ConcatDataset(parts)), receipts


def evaluate_checkpoint(cfg_path, ckpt_path, output, *, split="val", source="all",
                        device="cpu", limit=None):
    """Measure unchanged weights and save enough provenance to repeat the run."""
    cfg_path, ckpt_path, output = map(Path, (cfg_path, ckpt_path, output))
    if output.exists():
        raise FileExistsError(f"Evaluation output already exists: {output}")
    with cfg_path.open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    dataset, sources = evaluation_dataset(cfg, split, source, limit)
    device = resolve_device(device)
    print(f"device: {device}; {split}/{source}: {len(dataset)} samples", flush=True)
    model = build_model(encoder_name=cfg["model"]["encoder_name"],
                        encoder_weights=None, num_classes=NUM_CLASSES).to(device)
    digest = sha256(ckpt_path)
    state, epoch = load_inference_checkpoint(ckpt_path, device)
    model.load_state_dict(state, strict=True)
    loader = DataLoader(dataset, batch_size=cfg["optim"]["batch_size"], shuffle=False,
                        num_workers=cfg["data"]["num_workers"], pin_memory=device.type == "cuda")
    metrics = evaluate(model, loader, nn.CrossEntropyLoss(ignore_index=IGNORE_ID), device)
    if sha256(ckpt_path) != digest:
        raise ValueError("Checkpoint changed during evaluation")
    from . import train, data, raster_data, svg_to_mask, svg_render, model as model_module
    modules = (train, data, raster_data, svg_to_mask, svg_render, model_module)
    payload = {"version": 1, "split": split, "source": source, "ckpt": str(ckpt_path),
               "checkpoint_sha256": digest, "epoch": epoch, "device": str(device),
               "config": cfg, "config_sha256": sha256(cfg_path),
               "sample_count": len(dataset), "limit_per_source": limit, "sources": sources,
               "model_training_started": False,
               "code_sha256": {m.__name__: sha256(m.__file__) for m in modules},
               "evaluator_sha256": sha256(__file__),
               **metrics, "iou": dict(zip(CLASS_NAMES, metrics["iou"]))}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    print(f"{split}/{source}: mIoU {metrics['miou']:.4f}; {format_iou(metrics['iou'])}", flush=True)
    print(f"wrote {output}", flush=True)
    return payload


def main() -> None:
    p = argparse.ArgumentParser()
    config = p.add_mutually_exclusive_group(required=True)
    config.add_argument("--run-dir", type=Path)
    config.add_argument("--config", type=Path)
    p.add_argument("--ckpt", default="best.pt")
    p.add_argument("--weights", type=Path, help="explicit weights; required with --config")
    p.add_argument("--split", default="test", choices=("train", "val", "test"))
    p.add_argument("--device", default="auto")
    p.add_argument("--source", default="all", choices=("all", "svg", "raster"))
    p.add_argument("--limit", type=int, help="limit per source; omit for the complete split")
    p.add_argument("--output", type=Path, help="new metrics JSON; required with --config")
    args = p.parse_args()
    if args.config is not None and (args.weights is None or args.output is None):
        p.error("--config requires --weights and --output")
    cfg_path = args.config if args.config is not None else args.run_dir / "config.yaml"
    ckpt_path = args.weights if args.weights is not None else args.run_dir / args.ckpt
    suffix = "" if args.source == "all" else f"_{args.source}"
    output = args.output if args.output is not None else args.run_dir / f"{args.split}{suffix}_metrics.json"
    evaluate_checkpoint(cfg_path, ckpt_path, output, split=args.split, source=args.source,
                        device=args.device, limit=args.limit)


if __name__ == "__main__":
    main()
