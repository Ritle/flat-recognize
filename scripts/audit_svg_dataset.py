"""Exercise every declared CubiCasa SVG with the strict training loader on CPU."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import torch

from buildingcv.data import CubiCasaDataset
from buildingcv.labels import NUM_CLASSES

ROOT = Path(__file__).resolve().parent.parent
_DATASETS = {}


def initialize_worker(dataset, size):
    torch.set_num_threads(1)
    for split in ("train", "val", "test"):
        _DATASETS[split] = CubiCasaDataset(dataset, split, size=tuple(size),
                                         normalize=True, letterbox=True)


def inspect_sample(job):
    split, index, sample, digest = job
    record = {"split": split, "sample": sample, "svg_sha256": digest}
    try:
        image, mask = _DATASETS[split].load_strict(index)
        height, width = _DATASETS[split].size
        if tuple(image.shape) != (3, height, width) or tuple(mask.shape) != (height, width):
            raise ValueError("Unexpected tensor dimensions")
        if image.dtype != torch.float32 or mask.dtype != torch.int64:
            raise ValueError("Unexpected tensor dtype")
        if not torch.isfinite(image).all():
            raise ValueError("Non-finite input pixels")
        labels = mask.unique().tolist()
        if any(value < 0 or value >= NUM_CLASSES for value in labels):
            raise ValueError("Invalid class labels")
        if not torch.any(mask == 1):
            raise ValueError("No wall pixels after rendering")
        record.update(status="ok", labels=labels)
    except Exception as exc:
        record.update(status="error", error_type=type(exc).__name__, error=str(exc)[:2000])
    return record


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def audit(dataset, output, *, size=(512, 512), workers=2, cache=None):
    from buildingcv import data, svg_render, svg_to_mask, labels
    dataset, output = Path(dataset).resolve(), Path(output)
    if output.exists():
        raise FileExistsError(output)
    if workers < 1 or len(size) != 2 or any(type(v) is not int or v <= 0 for v in size):
        raise ValueError("Positive worker count and image dimensions required")
    # Check the native runtime once; do not turn one missing dependency into 5000 failures.
    import cairosvg
    import cairocffi
    code_hash = hashlib.sha256()
    for module in (data, svg_render, svg_to_mask, labels):
        code_hash.update(Path(module.__file__).read_bytes())
    code_hash.update(Path(__file__).read_bytes())
    settings = {"dataset": str(dataset), "size": list(size), "loader_sha256": code_hash.hexdigest(),
                "cairo": cairocffi.cairo_version_string(), "cairosvg": cairosvg.__version__}
    saved = json.loads(Path(cache).read_text(encoding="utf-8")) if cache and Path(cache).exists() else {}
    records = saved.get("records", {}) if saved.get("settings") == settings else {}
    jobs, finished = [], {}
    split_hashes = {}
    for split in ("train", "val", "test"):
        split_hashes[split] = hashlib.sha256((dataset / f"{split}.txt").read_bytes()).hexdigest()
        ds = CubiCasaDataset(dataset, split, size=tuple(size), normalize=True, letterbox=True)
        for index, sample in enumerate(ds.samples):
            path = (dataset / sample / "model.svg").resolve()
            if not path.is_relative_to(dataset):
                raise ValueError("SVG path escapes dataset")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            key = f"{split}:{sample}"
            if key in finished or any(key == f"{j[0]}:{j[2]}" for j in jobs):
                raise ValueError("Duplicate SVG path in split")
            old = records.get(key)
            if old and old.get("svg_sha256") == digest:
                finished[key] = old
            else:
                jobs.append((split, index, sample, digest))
    total = len(jobs) + len(finished)
    print(f"Strict audit: {total} samples; {len(finished)} cached; {len(jobs)} to render at {size}.", flush=True)
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize_worker,
                             initargs=(str(dataset), tuple(size))) as pool:
        for record in pool.map(inspect_sample, jobs, chunksize=4):
            finished[f"{record['split']}:{record['sample']}"] = record
            if record["status"] != "ok":
                print(f"ERROR {record['split']}/{record['sample']}: {record['error']}", flush=True)
            if len(finished) % 100 == 0 or len(finished) == total:
                if cache:
                    write_json(cache, {"settings": settings, "records": finished})
                print(f"Checked {len(finished)}/{total} SVGs", flush=True)
    rows = sorted(finished.values(), key=lambda r: (r["split"], r["sample"]))
    failures = [r for r in rows if r["status"] != "ok"]
    report = {"version": 1, "settings": settings, "source_split_sha256": split_hashes,
              "total": total, "ok": len(rows) - len(failures), "failures": failures,
              "samples": rows, "substituted_samples": 0, "model_training_started": False}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "report.json", report)
    print(f"Completed: {report['ok']}/{total} valid; {len(failures)} failures; no substitutions.", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/cubicasa5k")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()
    report = audit(args.dataset, args.output, size=(args.size, args.size),
                   workers=args.workers, cache=args.cache)
    return int(bool(report["failures"]))


if __name__ == "__main__":
    raise SystemExit(main())
