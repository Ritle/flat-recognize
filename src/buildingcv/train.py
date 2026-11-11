"""Training entrypoint. Resumes from any saved checkpoint; metrics.csv appends."""

from __future__ import annotations

import argparse
from bisect import bisect_right
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from .data import CubiCasaDataset
from .labels import CLASS_NAMES, IGNORE_ID, NUM_CLASSES


class Tee:
    """Print to stdout and append timestamped lines to a log file.

    Stdout stays clean (just the message). The file gets `[HH:MM:SS +Ns]`
    prefixes so we can see how long each phase took after the fact.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._f = path.open("a", buffering=1)  # line-buffered
        self._start = datetime.now()

    def __call__(self, msg: str = "") -> None:
        print(msg)
        ts = datetime.now().strftime("%H:%M:%S")
        elapsed = (datetime.now() - self._start).total_seconds()
        self._f.write(f"[{ts} +{elapsed:7.1f}s] {msg}\n")

    def close(self) -> None:
        self._f.close()


def resolve_device(name: str) -> torch.device:
    """Pick a device. "auto" prefers cuda, then mps, then cpu."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_run_dir(base: Path) -> Path:
    """Timestamped subdirectory under `base`, created."""
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    run_dir = base / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def maybe_subset(ds, limit: int | None):
    """Optionally cap a dataset to its first `limit` samples (for sanity runs)."""
    if limit is None or limit >= len(ds):
        return ds
    return Subset(ds, range(limit))


class StrictSvgDataset(Dataset):
    """Opt-in training view that never substitutes a failed SVG sample."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset.load_strict(index)


def build_datasets(cfg, limit_train=None, limit_val=None, *, limits_per_source=False):
    """Keep the existing SVG path; optionally concatenate reviewed raster data."""
    data = cfg["data"]
    kwargs = dict(size=tuple(data["image_size"]), normalize=data["normalize"],
                  letterbox=data.get("letterbox", False))
    raster_manifest = data.get("raster_manifest")
    if not data.get("include_cubicasa", True) and not raster_manifest:
        raise ValueError("Enable CubiCasa or supply a reviewed raster_manifest")
    datasets = []
    for split, limit in (("train", limit_train), ("val", limit_val)):
        if limit is not None and (type(limit) is not int or limit <= 0):
            raise ValueError("Dataset debug limits must be positive integers")
        parts = []
        if data.get("include_cubicasa", True):
            svg = CubiCasaDataset(data["data_dir"], split, split_dir=data.get("split_dir"), **kwargs)
            parts.append(StrictSvgDataset(svg) if data.get("strict_svg", False) else svg)
        if raster_manifest:
            from .raster_data import RasterMaskDataset
            parts.append(RasterMaskDataset(raster_manifest, split,
                                           root=data.get("raster_root", "."),
                                           allow_partial=data.get("allow_partial_annotations", False), **kwargs))
        if limits_per_source:
            parts = [maybe_subset(part, limit) for part in parts]
        dataset = parts[0] if len(parts) == 1 else ConcatDataset(parts)
        if not len(dataset):
            raise ValueError(f"Empty training/evaluation dataset: {split}")
        datasets.append(dataset if limits_per_source else maybe_subset(dataset, limit))
    return tuple(datasets)


def checked_item(dataset, index):
    """Preflight must fail on a bad SVG instead of silently replacing it."""
    if isinstance(dataset, Subset):
        return checked_item(dataset.dataset, dataset.indices[index])
    if isinstance(dataset, ConcatDataset):
        part = bisect_right(dataset.cumulative_sizes, index)
        offset = dataset.cumulative_sizes[part - 1] if part else 0
        return checked_item(dataset.datasets[part], index - offset)
    strict = getattr(dataset, "load_strict", None)
    return strict(index) if strict is not None else dataset[index]


def check_data(cfg, args):
    """Exercise selected samples without allocating a model or creating a run."""
    datasets = build_datasets(cfg, args.limit_train, args.limit_val,
                              limits_per_source=getattr(args, "check_data_per_source", False))
    for split, dataset in zip(("train", "val"), datasets):
        counts = torch.zeros(NUM_CLASSES, dtype=torch.long)
        ignored = 0
        for index in range(len(dataset)):
            _, mask = checked_item(dataset, index)
            ignored += int((mask == IGNORE_ID).sum())
            counts += torch.bincount(mask[mask != IGNORE_ID], minlength=NUM_CLASSES)
        print(f"{split}: {len(dataset)} samples; resized class pixels: "
              + ", ".join(f"{name}={count}" for name, count in zip(CLASS_NAMES, counts.tolist()))
              + f"; ignored={ignored}")


@torch.no_grad()
def evaluate(model, loader, loss_fn, device) -> dict[str, Any]:
    """Run validation and return loss, pixel accuracy, and per-class IoU.

    IoU per class is computed from accumulated TP / FP / FN counts across the
    full loader — not as a per-batch average — so each pixel weighs the same
    regardless of which batch it landed in. Mean IoU (mIoU) is the unweighted
    average across classes; it does NOT match pixel accuracy because it gives
    each class equal weight regardless of how many pixels that class covers.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_pixels = 0
    ignored_pixels = 0
    tp = torch.zeros(NUM_CLASSES, dtype=torch.long, device=device)
    fp = torch.zeros(NUM_CLASSES, dtype=torch.long, device=device)
    fn = torch.zeros(NUM_CLASSES, dtype=torch.long, device=device)

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        valid = masks != IGNORE_ID
        pixels = int(valid.sum())
        if not pixels:
            raise ValueError("Evaluation batch has no reviewed pixels")
        logits = model(images)
        loss = loss_fn(logits, masks)
        total_loss += loss.item() * pixels
        preds = logits.argmax(dim=1)
        total_correct += ((preds == masks) & valid).sum().item()
        total_pixels += pixels
        ignored_pixels += masks.numel() - pixels
        for c in range(NUM_CLASSES):
            pred_c = (preds == c) & valid
            true_c = masks == c
            tp[c] += (pred_c & true_c).sum()
            fp[c] += (pred_c & ~true_c).sum()
            fn[c] += (~pred_c & true_c).sum()

    denom = (tp + fp + fn).clamp(min=1).float()
    iou = (tp.float() / denom).cpu().tolist()
    return {
        "loss": total_loss / total_pixels,
        "evaluated_pixels": total_pixels,
        "ignored_pixels": ignored_pixels,
        "pixel_acc": total_correct / total_pixels,
        "iou": iou,
        "miou": sum(iou) / len(iou),
    }


def format_iou(iou: list[float]) -> str:
    """Render per-class IoU as `name 0.91 | name 0.15 | ...` for logging."""
    return " | ".join(f"{name} {v:.3f}" for name, v in zip(CLASS_NAMES, iou))


def append_metrics_row(csv_path: Path, epoch: int, train_loss: float, m: dict) -> None:
    """One row per epoch into metrics.csv. Header is written once on creation."""
    new = not csv_path.exists()
    with csv_path.open("a", buffering=1) as f:
        if new:
            cols = ["epoch", "train_loss", "val_loss", "pixel_acc", "miou"] + [
                f"iou_{n}" for n in CLASS_NAMES
            ]
            f.write(",".join(cols) + "\n")
        row = [
            str(epoch),
            f"{train_loss:.6f}",
            f"{m['loss']:.6f}",
            f"{m['pixel_acc']:.6f}",
            f"{m['miou']:.6f}",
        ] + [f"{v:.6f}" for v in m["iou"]]
        f.write(",".join(row) + "\n")


def save_checkpoint(path: Path, *, epoch: int, model, optimizer, cfg, best_miou: float) -> None:
    """Full state for resume: model + optimizer + epoch + best-so-far + config."""
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
            "best_miou": best_miou,
        },
        path,
    )


def train(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    init_weights = getattr(args, "init_weights", None)
    if init_weights is not None and args.resume is not None:
        raise ValueError("--init-weights and --resume cannot be combined")
    device = resolve_device(cfg["train"]["device"])
    train_ds, val_ds = build_datasets(cfg, args.limit_train, args.limit_val)
    raster_manifest = cfg["data"].get("raster_manifest")
    if args.resume is not None:
        snapshot = Path(args.resume).parent / "raster_masks.json"
        if ((raster_manifest and (not snapshot.is_file() or
             snapshot.read_bytes() != Path(raster_manifest).read_bytes())) or
                (not raster_manifest and snapshot.exists())):
            raise ValueError("Raster dataset changed or lacks a snapshot; start a new run")

    from .model import build_model
    model = build_model(
        encoder_name=cfg["model"]["encoder_name"],
        encoder_weights=None if init_weights is not None or args.resume is not None
        else cfg["model"]["encoder_weights"],
    ).to(device)
    initialization = None
    if init_weights is not None:
        from .checkpoint import initialize_training_weights
        initialization = initialize_training_weights(model, init_weights)

    # Resume-vs-fresh-run setup. On resume we reuse the original run dir so
    # logs and metrics.csv accumulate; otherwise we make a new timestamped one.
    if args.resume is not None:
        resume_path = Path(args.resume)
        run_dir = resume_path.parent
        # config.yaml + train.log already exist — append, don't clobber.
    else:
        run_dir = make_run_dir(Path(cfg["output"]["run_dir"]))
        shutil.copy(args.config, run_dir / "config.yaml")

    log = Tee(run_dir / "train.log")
    log(f"device: {device}   run dir: {run_dir}")
    if initialization is not None:
        (run_dir / "initialization.json").write_text(
            json.dumps(initialization, indent=2) + "\n", encoding="utf-8")
        log(f"initial weights SHA-256: {initialization['sha256']}; new optimizer and epoch counter")

    if raster_manifest:
        snapshot = run_dir / "raster_masks.json"
        if args.resume is None:
            shutil.copy(raster_manifest, snapshot)
        log(f"raster manifest SHA-256: {hashlib.sha256(snapshot.read_bytes()).hexdigest()}")
    log(f"train samples: {len(train_ds)}   val samples: {len(val_ds)}")

    loader_kwargs = dict(
        batch_size=cfg["optim"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        pin_memory=(device.type == "cuda"),
        persistent_workers=cfg["data"]["num_workers"] > 0,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    log(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_ID)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["optim"]["lr"],
        weight_decay=cfg["optim"]["weight_decay"],
    )

    # Restore from checkpoint if resuming. weights_only=False because the
    # checkpoint embeds the cfg dict (not a pure tensor file).
    start_epoch = 1
    best_miou = 0.0
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_miou = ckpt.get("best_miou", 0.0)
        log(f"resumed from {args.resume}: epoch {ckpt['epoch']} done, best mIoU {best_miou:.4f}")

    epochs = args.epochs if args.epochs is not None else cfg["train"]["epochs"]
    log_every = cfg["train"]["log_every"]
    ckpt_every = cfg["train"].get("checkpoint_every", 0)
    metrics_csv = run_dir / "metrics.csv"

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running = 0.0          # loss sum since last log_every print
        epoch_loss_sum = 0.0   # loss sum across the whole epoch (for CSV)
        for step, (images, masks) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            if not torch.any(masks != IGNORE_ID):
                raise ValueError("Training batch has no reviewed pixels")

            logits = model(images)
            loss = loss_fn(logits, masks)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running += loss.item()
            epoch_loss_sum += loss.item()
            if step % log_every == 0:
                log(f"  epoch {epoch} step {step}/{len(train_loader)}  train loss {running/log_every:.4f}")
                running = 0.0

        train_loss_avg = epoch_loss_sum / len(train_loader)

        metrics = evaluate(model, val_loader, loss_fn, device)
        log(
            f"epoch {epoch} done — train loss {train_loss_avg:.4f}   "
            f"val loss {metrics['loss']:.4f}   "
            f"pixel acc {metrics['pixel_acc']:.4f}   mIoU {metrics['miou']:.4f}"
        )
        log(f"  per-class IoU: {format_iou(metrics['iou'])}")
        log(f"  validation pixels: reviewed {metrics['evaluated_pixels']} | ignored {metrics['ignored_pixels']}")
        append_metrics_row(metrics_csv, epoch, train_loss_avg, metrics)

        is_best = metrics["miou"] > best_miou
        if is_best:
            best_miou = metrics["miou"]

        # last.pt — overwritten every epoch, used for resume.
        save_checkpoint(
            run_dir / "last.pt",
            epoch=epoch, model=model, optimizer=optimizer, cfg=cfg, best_miou=best_miou,
        )
        # best.pt — only when val mIoU improves; the model to use for inference.
        if is_best:
            save_checkpoint(
                run_dir / "best.pt",
                epoch=epoch, model=model, optimizer=optimizer, cfg=cfg, best_miou=best_miou,
            )
            log(f"  new best mIoU {best_miou:.4f} -> best.pt")
        # epoch_NN.pt — every Nth epoch, so we can roll back to a specific point.
        if ckpt_every > 0 and epoch % ckpt_every == 0:
            save_checkpoint(
                run_dir / f"epoch_{epoch:02d}.pt",
                epoch=epoch, model=model, optimizer=optimizer, cfg=cfg, best_miou=best_miou,
            )

    log(f"done. last: {run_dir / 'last.pt'}   best: {run_dir / 'best.pt'} (mIoU {best_miou:.4f})")
    log.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--epochs", type=int, default=None, help="override cfg.train.epochs")
    p.add_argument("--limit-train", type=int, default=None, help="cap train samples (debug)")
    p.add_argument("--limit-val", type=int, default=None, help="cap val samples (debug)")
    p.add_argument("--check-data", action="store_true",
                   help="validate train/val pairs without creating a model or training run")
    p.add_argument("--check-data-per-source", action="store_true",
                   help="with --check-data, cap each SVG/raster source separately so both are checked")
    weights = p.add_mutually_exclusive_group()
    weights.add_argument("--init-weights", type=Path,
                         help="start a new experiment from .safetensors, with a fresh optimizer")
    p.add_argument("--check-init-weights", action="store_true",
                   help="validate --init-weights against the configured model without creating a run")
    weights.add_argument(
        "--resume", type=Path, default=None,
        help="path to a checkpoint (last.pt / best.pt / epoch_NN.pt). "
             "Loads model+optimizer+epoch and continues into the same run dir.",
    )
    args = p.parse_args()
    if args.check_data_per_source and not args.check_data:
        p.error("--check-data-per-source requires --check-data")
    if args.check_init_weights and (args.init_weights is None or args.check_data):
        p.error("--check-init-weights requires --init-weights and cannot be combined with --check-data")
    if args.check_data and (args.init_weights is not None or args.resume is not None):
        p.error("--check-data does not load weights; use --check-init-weights separately")

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.check_data:
        check_data(cfg, args)
    elif args.check_init_weights:
        from .checkpoint import initialize_training_weights
        from .model import build_model
        model = build_model(encoder_name=cfg["model"]["encoder_name"], encoder_weights=None)
        print(json.dumps(initialize_training_weights(model, args.init_weights), indent=2))
    else:
        train(cfg, args)


if __name__ == "__main__":
    main()
