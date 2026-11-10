"""Screen local plans against every original raster in the CubiCasa splits.

This ranks candidates for manual inspection, never assigns family approval.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from audit_training_families import distance, signatures

ROOT = Path(__file__).resolve().parent.parent
FEATURE_VERSION = "downsample512_rot4_dhash_phash_v1"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def feature(path):
    path = Path(path)
    data = path.read_bytes()
    from io import BytesIO
    with Image.open(BytesIO(data)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    size = image.size
    image.thumbnail((512, 512), Image.Resampling.LANCZOS)
    return {"sha256": hashlib.sha256(data).hexdigest(), "width": size[0], "height": size[1],
            "signatures": signatures(image)}


def base_samples(dataset):
    dataset = Path(dataset).resolve()
    rows, seen = [], set()
    split_hashes = {}
    for split in ("train", "val", "test"):
        file = dataset / f"{split}.txt"
        split_hashes[split] = hashlib.sha256(file.read_bytes()).hexdigest()
        for line in file.read_text(encoding="utf-8").splitlines():
            sample = line.strip().strip("/")
            if not sample:
                continue
            path = (dataset / sample / "F1_original.png").resolve()
            if not path.is_relative_to(dataset) or not path.is_file():
                raise ValueError(f"Missing or unsafe CubiCasa source: {sample}")
            if path in seen:
                raise ValueError(f"Repeated CubiCasa path across/within splits: {sample}")
            seen.add(path)
            rows.append({"sample": sample, "split": split, "path": str(path)})
    return rows, split_hashes


def prepare_deduplicated_splits(report, output):
    """Remove only proven training-side copies; preserve official eval lists."""
    dataset = Path(report["base_dataset"])
    for split, expected in report["split_sha256"].items():
        if hashlib.sha256((dataset / f"{split}.txt").read_bytes()).hexdigest() != expected:
            raise ValueError("Official split changed after the audit")
    exclude = set()
    for pair in report["base_cross_split_exact_duplicates"]:
        parts = pair["splits"]
        if "train" not in parts:
            raise ValueError("Evaluation splits share an exact image; manual review required")
        files = [(dataset / sample / "F1_original.png").resolve() for sample in pair["samples"]]
        if any(not path.is_relative_to(dataset.resolve()) for path in files):
            raise ValueError("Unsafe base sample path")
        hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in files]
        if hashes[0] != hashes[1]:
            raise ValueError("Duplicate image changed after the audit")
        exclude.add(pair["samples"][parts.index("train")])
    texts = {split: (dataset / f"{split}.txt").read_text(encoding="utf-8")
             for split in ("train", "val", "test")}
    training = [line for line in texts["train"].splitlines()
                if line.strip() and line.strip().strip("/") not in exclude]
    known = {line.strip().strip("/") for line in texts["train"].splitlines() if line.strip()}
    if not exclude <= known:
        raise ValueError("Audit exclusions are absent from training split")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "train.txt").write_text("\n".join(training) + "\n", encoding="utf-8")
    for split in ("val", "test"):
        # Byte-for-byte preservation, including original line endings.
        (output / f"{split}.txt").write_bytes((dataset / f"{split}.txt").read_bytes())
    summary = {"version": 1, "base_dataset": str(dataset), "excluded_train_samples": sorted(exclude),
               "source_split_sha256": report["split_sha256"],
               "counts": {"train": len(training), **{
                   split: len([line for line in texts[split].splitlines() if line.strip()])
                   for split in ("val", "test")}}}
    write_json(output / "provenance.json", summary)
    return summary


def audit(manifest, dataset, output, *, root=ROOT, workers=4, nearest=5, cache=None):
    if workers < 1 or nearest < 1:
        raise ValueError("workers and nearest must be positive")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    rows, split_hashes = base_samples(dataset)
    if not rows:
        raise ValueError("CubiCasa splits are empty")
    manifest_data = load_json(manifest)
    local = []
    local_ids = set()
    for sample in manifest_data["samples"]:
        sid = sample.get("id")
        if (not isinstance(sid, str) or not sid or sid in local_ids or
                any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in sid)):
            raise ValueError("Invalid or duplicate local sample ID")
        local_ids.add(sid)
        record = feature(Path(root) / sample["image"])
        if record["sha256"] != sample["sha256"]:
            raise ValueError(f"Local image changed: {sample['id']}")
        local.append({**sample, **record})
    if not local:
        raise ValueError("No local samples to compare")
    cached = load_json(cache) if cache and Path(cache).exists() else {}
    if (cached.get("feature_version") != FEATURE_VERSION or
            cached.get("dataset") != str(Path(dataset).resolve())):
        cached = {}
    records = cached.get("records", {})

    def describe(row):
        path = Path(row["path"])
        info = path.stat()
        stamp = [info.st_size, info.st_mtime_ns]
        old = records.get(row["sample"])
        if old and old.get("stamp") == stamp:
            return row, old
        return row, {**feature(path), "stamp": stamp}

    bases = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, (row, info) in enumerate(pool.map(describe, rows), 1):
            records[row["sample"]] = info
            bases.append({**row, **info})
            if index % 100 == 0 or index == len(rows):
                if cache:
                    write_json(cache, {"feature_version": FEATURE_VERSION,
                                       "dataset": str(Path(dataset).resolve()), "records": records})
                print(f"Indexed {index}/{len(rows)} CubiCasa rasters", flush=True)
    local_results = []
    all_exact = []
    for sample in local:
        ranked = []
        for base in bases:
            dh, ph, angle = distance(sample["signatures"], base["signatures"])
            pair = {"base_sample": base["sample"], "base_split": base["split"],
                    "base_sha256": base["sha256"], "exact": sample["sha256"] == base["sha256"],
                    "dhash_distance": dh, "phash_distance": ph, "rotation_deg": angle}
            pair["suggested"] = pair["exact"] or (dh <= 12 and ph <= 12)
            ranked.append(pair)
            if pair["exact"]:
                all_exact.append({"local_id": sample["id"], **pair})
        ranked.sort(key=lambda r: (not r["exact"], r["dhash_distance"] + r["phash_distance"],
                                   r["phash_distance"], r["base_sample"]))
        local_results.append({"id": sample["id"], "image": sample["image"],
                              "sha256": sample["sha256"], "split": sample["split"],
                              "nearest": ranked[:nearest],
                              "suggested_pairs": [r for r in ranked if r["suggested"]]})
    duplicates, by_sha = [], {}
    for base in bases:
        previous = by_sha.setdefault(base["sha256"], [])
        for old in previous:
            if old["split"] != base["split"]:
                duplicates.append({"samples": [old["sample"], base["sample"]],
                                   "splits": [old["split"], base["split"]]})
        previous.append(base)
    report = {"version": 1, "feature_version": FEATURE_VERSION,
              "base_dataset": str(Path(dataset).resolve()), "base_count": len(bases),
              "split_sha256": split_hashes,
              "manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
              "automatic_group_approval": False, "exact_local_matches": all_exact,
              "base_cross_split_exact_duplicates": duplicates, "local_plans": local_results,
              "scope": "F1_original.png in all listed CubiCasa splits; nearest views require manual review"}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "report.json", report)
    for sample in local_results:
        cells = [(sample["id"], Path(root) / sample["image"], 0)]
        cells += [(r["base_sample"] + f" d={r['dhash_distance']} p={r['phash_distance']}",
                   Path(dataset) / r["base_sample"] / "F1_original.png", r["rotation_deg"])
                  for r in sample["nearest"]]
        sheet = Image.new("RGB", (320 * len(cells), 300), "white")
        draw = ImageDraw.Draw(sheet)
        for index, (label, path, angle) in enumerate(cells):
            draw.text((index * 320 + 5, 5), label[:50], fill="black")
            with Image.open(path) as image:
                thumbnail = ImageOps.exif_transpose(image).convert("RGB").rotate(angle, expand=True)
            thumbnail.thumbnail((310, 270))
            sheet.paste(thumbnail, (index * 320 + (320 - thumbnail.width) // 2, 25))
        sheet.save(output / f"{sample['id']}.jpg")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs/training_samples.json")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/cubicasa5k")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prepare-splits-from", type=Path,
                        help="create derived splits from an existing audit, removing proven training copies only")
    args = parser.parse_args()
    if args.prepare_splits_from:
        summary = prepare_deduplicated_splits(load_json(args.prepare_splits_from), args.output)
        print(json.dumps(summary, indent=2))
        return
    report = audit(args.manifest, args.dataset, args.output, root=args.root,
                   workers=args.workers, cache=args.cache)
    print(f"Checked {report['base_count']} base images; {len(report['exact_local_matches'])} exact local matches; "
          f"{len(report['base_cross_split_exact_duplicates'])} exact duplicate base pairs crossing splits.")


if __name__ == "__main__":
    main()
