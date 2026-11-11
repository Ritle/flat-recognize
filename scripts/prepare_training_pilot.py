"""Prepare a small mixed training experiment without shrinking held-out lists."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

import yaml

from buildingcv.train import build_datasets


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(config, output, *, count=32, seed=42):
    config, output = Path(config), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an integer in [0, 2**32)")
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    data = cfg["data"]
    if not (data.get("include_cubicasa", True) and data.get("strict_svg") and data.get("raster_manifest")):
        raise ValueError("Pilot requires strict SVG and reviewed raster sources")
    source = Path(data.get("split_dir") or data["data_dir"])
    lists = {split: (source / f"{split}.txt").read_bytes() for split in ("train", "val", "test")}
    names = {split: [line.strip().strip("/") for line in raw.decode("utf-8").splitlines() if line.strip()]
             for split, raw in lists.items()}
    if any(len(rows) != len(set(rows)) for rows in names.values()):
        raise ValueError("Repeated split entries")
    if any(set(names[a]) & set(names[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Sample crosses training/evaluation splits")
    if type(count) is not int or not 0 < count <= len(names["train"]):
        raise ValueError("Base training count must be positive and fit the source split")
    buckets = {}
    dataset = Path(data["data_dir"]).resolve()
    for name in names["train"]:
        path = (dataset / name / "model.svg").resolve()
        if not path.is_relative_to(dataset) or not path.is_file():
            raise ValueError(f"Missing or unsafe training SVG: {name}")
        buckets.setdefault(name.split("/")[0], []).append(name)
    rng = random.Random(seed)
    for bucket in sorted(buckets):
        buckets[bucket].sort()
        rng.shuffle(buckets[bucket])
    selected = []
    while len(selected) < count:
        for bucket in sorted(buckets):
            if buckets[bucket] and len(selected) < count:
                selected.append(buckets[bucket].pop())
    full_train, full_val = build_datasets(cfg)
    raster_train = len(full_train.datasets[1])
    raster_val = len(full_val.datasets[1])
    if not raster_train or not raster_val:
        raise ValueError("Pilot requires reviewed raster train and validation samples")
    selected_hashes = [{"sample": name, "svg_sha256": digest(dataset / name / "model.svg")}
                       for name in selected]
    split_dir = output / "splits"
    cfg["data"]["split_dir"] = str(split_dir)
    cfg["model"]["encoder_weights"] = None
    cfg["train"].update(seed=seed, epochs=1, log_every=1, checkpoint_every=0)
    cfg["output"]["run_dir"] = str(output / "runs")
    receipt = {"version": 1, "source_config_sha256": digest(config), "seed": seed,
               "source_split_sha256": {key: hashlib.sha256(raw).hexdigest() for key, raw in lists.items()},
               "selected_base_train": selected_hashes,
               "selected_buckets": dict(Counter(name.split("/")[0] for name in selected)),
               "base_train": count, "raster_train": raster_train, "mixed_train": count + raster_train,
               "base_val": len(names["val"]), "raster_val": raster_val, "mixed_val": len(full_val),
               "evaluation_lists_unchanged": True, "training_started": False}
    split_dir.mkdir(parents=True, exist_ok=False)
    (split_dir / "train.txt").write_text("".join(f"/{name}/\n" for name in selected), encoding="utf-8")
    for split in ("val", "test"):
        (split_dir / f"{split}.txt").write_bytes(lists[split])
    (output / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    (output / "selection.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-train-count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = prepare(args.config, args.output, count=args.base_train_count, seed=args.seed)
    print(f"Prepared {result['mixed_train']} train / {result['mixed_val']} val; seed {result['seed']}; no training started.")


if __name__ == "__main__":
    main()
